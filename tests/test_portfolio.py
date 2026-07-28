import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import research


def _synthetic_panel(n_times=30, n_symbols=20, seed=0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_times):
        dt = datetime(2022, 1, 1) + timedelta(hours=4 * t)  # noqa: DTZ001 - naive, matches production panel datetimes
        for s in range(n_symbols):
            rows.append(
                {
                    "datetime": dt,
                    "symbol": f"S{s}",
                    "pred": float(rng.normal()),
                    "vol": float(abs(rng.normal()) + 0.01),
                    "fwd_return_1": float(rng.normal(scale=0.01)),
                }
            )
    return pl.DataFrame(rows)


def test_dollar_neutral_weights_are_net_zero_and_within_gross_cap():
    panel = _synthetic_panel()
    weights = research.dollar_neutral_weights(
        panel, "pred", top_frac=0.2, gross_exposure=1.0, max_position_per_symbol=0.5
    )
    per_bar = weights.group_by("datetime").agg(
        pl.col("weight").sum().alias("net"),
        pl.col("weight").abs().sum().alias("gross"),
    )
    np.testing.assert_allclose(per_bar["net"].to_numpy(), 0.0, atol=1e-9)
    assert (per_bar["gross"].to_numpy() <= 1.0 + 1e-9).all()


def test_dollar_neutral_weights_only_long_top_and_short_bottom():
    panel = _synthetic_panel(n_times=1, n_symbols=10, seed=1)
    weights = research.dollar_neutral_weights(panel, "pred", top_frac=0.2)
    joined = weights.join(panel.select("symbol", "pred"), on="symbol")
    long_leg = joined.filter(pl.col("weight") > 0)
    short_leg = joined.filter(pl.col("weight") < 0)
    flat = joined.filter(pl.col("weight") == 0)
    assert len(long_leg) == 2  # top_frac=0.2 of 10 symbols
    assert len(short_leg) == 2
    assert len(flat) == 6
    assert long_leg["pred"].min() > flat["pred"].max()
    assert short_leg["pred"].max() < flat["pred"].min()


def test_max_position_per_symbol_caps_weight():
    panel = _synthetic_panel(n_times=1, n_symbols=10, seed=2)
    weights = research.dollar_neutral_weights(
        panel, "pred", top_frac=0.5, max_position_per_symbol=0.05
    )
    assert weights["weight"].abs().max() <= 0.05 + 1e-12


def test_size_col_weights_leg_proportionally():
    panel = pl.DataFrame(
        {
            "datetime": [datetime(2022, 1, 1)] * 4,  # noqa: DTZ001
            "symbol": ["A", "B", "C", "D"],
            "pred": [1.0, 2.0, -2.0, -1.0],
            "size": [1.0, 3.0, 1.0, 3.0],
        }
    )
    weights = research.dollar_neutral_weights(
        panel,
        "pred",
        top_frac=0.5,
        size_col="size",
        gross_exposure=1.0,
        max_position_per_symbol=1.0,
    )
    w = dict(zip(weights["symbol"].to_list(), weights["weight"].to_list(), strict=True))
    # long leg = A, B (top 2 preds); B has 3x A's size -> B should get 3x A's weight
    assert abs(w["B"] / w["A"] - 3.0) < 1e-9
    # short leg = C, D; D has 3x C's size
    assert abs(w["D"] / w["C"] - 3.0) < 1e-9


def test_portfolio_turnover_charges_entry_and_exit():
    weights = pl.DataFrame(
        {
            "datetime": [
                datetime(2022, 1, 1) + timedelta(hours=4 * i) for i in range(3)
            ],  # noqa: DTZ001
            "symbol": ["A", "A", "A"],
            "weight": [0.5, 0.5, 0.0],
        }
    )
    turnover = research.portfolio_turnover(weights)
    assert turnover["turnover"].to_list() == [0.5, 0.0, 0.5]


def test_portfolio_trade_frame_matches_manual_weighted_sum():
    dt = datetime(2022, 1, 1)  # noqa: DTZ001
    weights = pl.DataFrame(
        {
            "datetime": [dt, dt],
            "symbol": ["A", "B"],
            "weight": [0.5, -0.5],
        }
    )
    returns = pl.DataFrame(
        {
            "datetime": [dt, dt],
            "symbol": ["A", "B"],
            "fwd_return_1": [0.02, -0.01],
        }
    )
    tf = research.portfolio_trade_frame(weights, returns)
    expected = 0.5 * 0.02 + (-0.5) * (-0.01)
    assert abs(tf["trade_log_return"][0] - expected) < 1e-12


def test_portfolio_metrics_reports_net_when_fee_given():
    panel = _synthetic_panel()
    weights = research.dollar_neutral_weights(panel, "pred", top_frac=0.2)
    tf = research.portfolio_trade_frame(weights, panel)
    ar = research.sharpe_to_annualized_rate("4h")
    metrics = research.portfolio_metrics(tf, ar, taker_fee=0.0004)
    assert "sharpe_net" in metrics
    assert metrics["total_log_return_net"] <= metrics["total_log_return"] + 1e-9

    gross_only = research.portfolio_metrics(tf, ar)
    assert "sharpe_net" not in gross_only


def test_vol_targeted_size_clips_and_scales():
    df = pl.DataFrame({"pred": [2.0, -2.0, 0.3], "vol": [0.02, 0.01, 0.04]})
    out = df.with_columns(research.vol_targeted_size("pred", "vol", vol_target=0.02))
    sizes = out["vol_targeted_size"].to_list()
    # pred clipped to [-1, 1] before scaling
    assert abs(sizes[0] - 1.0 * (0.02 / 0.02)) < 1e-12
    assert abs(sizes[1] - (-1.0) * (0.02 / 0.01)) < 1e-12
    assert abs(sizes[2] - 0.3 * (0.02 / 0.04)) < 1e-12


def test_panel_walk_forward_splits_keeps_symbols_together_per_timestamp():
    panel = _synthetic_panel(n_times=30, n_symbols=5)
    splits = research.panel_walk_forward_splits(panel, train_bars=10, test_bars=5)
    assert len(splits) > 0
    datetimes = panel["datetime"].to_numpy()
    for train_idx, test_idx in splits:
        # every timestamp appearing in train/test must have ALL its rows
        # (all symbols) on the same side - no partial-bar leakage
        train_times = set(datetimes[train_idx].tolist())
        test_times = set(datetimes[test_idx].tolist())
        assert train_times.isdisjoint(test_times)
        for t in train_times:
            assert (datetimes == t).sum() == (datetimes[train_idx] == t).sum()
        for t in test_times:
            assert (datetimes == t).sum() == (datetimes[test_idx] == t).sum()
