import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import research

TAKER_FEE = 0.0004
SLIPPAGE = 0.0001
COST_FRAC = TAKER_FEE + SLIPPAGE


def _trades(positions: list[float]) -> pl.DataFrame:
    # trade_log_return is irrelevant to the cost math itself, set to 0 so
    # cost tests aren't entangled with pnl.
    return pl.DataFrame(
        {
            "position": positions,
            "trade_log_return": [0.0] * len(positions),
        }
    )


def test_zero_turnover_zero_cost():
    trades = _trades([0.0, 0.0, 0.0, 0.0])
    costed = research.add_trading_costs(trades, TAKER_FEE, SLIPPAGE)
    assert costed["turnover"].to_list() == [0.0, 0.0, 0.0, 0.0]
    assert costed["cost_log_return"].abs().sum() == 0.0
    assert (
        costed["trade_log_return_net"].to_list() == costed["trade_log_return"].to_list()
    )


def test_single_flip_charges_exactly_one_round_trip():
    # flat -> long -> flip to short -> hold short -> hold short
    trades = _trades([0.0, 1.0, -1.0, -1.0, -1.0])
    costed = research.add_trading_costs(trades, TAKER_FEE, SLIPPAGE)
    turnover = costed["turnover"].to_list()
    # entry (1 unit), the flip (2 units: close long + open short), then flat
    assert turnover == [0.0, 1.0, 2.0, 0.0, 0.0]

    expected_cost_at_flip = np.log(1 - COST_FRAC * 2.0)
    assert costed["cost_log_return"][2] == expected_cost_at_flip
    assert costed["cost_log_return"][3] == 0.0
    assert costed["cost_log_return"][4] == 0.0


def test_holding_position_across_n_bars_charged_once():
    trades = _trades([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    costed = research.add_trading_costs(trades, TAKER_FEE, SLIPPAGE)
    turnover = costed["turnover"].to_list()
    assert turnover == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    # exactly one non-zero cost bar, at entry
    non_zero_cost_bars = (costed["cost_log_return"].abs() > 0).sum()
    assert non_zero_cost_bars == 1


def test_cost_summary_matches_manual_annualization():
    trades = _trades([0.0, 1.0, 1.0, 1.0, 1.0])
    costed = research.add_trading_costs(trades, TAKER_FEE, SLIPPAGE)
    annualized_rate = research.sharpe_to_annualized_rate("1d")
    periods_per_year = annualized_rate**2  # 365

    summary = research.cost_summary(costed, annualized_rate)
    mean_turnover = 1.0 / 5.0
    assert summary["mean_turnover_per_bar"] == mean_turnover
    assert summary["turnover_per_year"] == mean_turnover * periods_per_year

    mean_cost = float(np.log(1 - COST_FRAC * 1.0)) / 5.0
    expected_drag_log = -mean_cost * periods_per_year
    assert abs(summary["annual_fee_drag_log"] - expected_drag_log) < 1e-9
    assert abs(summary["annual_fee_drag_pct"] - (np.expm1(expected_drag_log))) < 1e-9


def test_walk_forward_run_reports_gross_and_net_when_fee_given():
    rng = np.random.default_rng(0)
    n = 400
    feature = rng.normal(size=n).astype(np.float32)
    target = (0.5 * feature + rng.normal(scale=0.1, size=n)).astype(np.float32)
    df = pl.DataFrame({"feat": feature, "target": target})

    splits = research.walk_forward_splits(n, train_bars=200, test_bars=50)
    annualized_rate = research.sharpe_to_annualized_rate("1h")

    def model_factory(n_features: int):
        from torch import nn

        return nn.Linear(n_features, 1)

    research.set_seed(0)
    result = research.walk_forward_run(
        df,
        ["feat"],
        "target",
        model_factory,
        splits,
        annualized_rate,
        no_epochs=50,
        taker_fee=TAKER_FEE,
        slippage=SLIPPAGE,
    )

    folds = result["folds"]
    assert "sharpe_net" in folds.columns
    assert "turnover_per_year" in folds.columns

    stitched = result["stitched_trades"]
    assert "trade_log_return_net" in stitched.columns
    assert "equity_curve_net" in stitched.columns

    metrics = research.stitched_metrics(stitched, annualized_rate)
    assert "sharpe_net" in metrics
    assert "annual_fee_drag_pct" in metrics
    # Net can never beat gross when costs are strictly non-negative.
    assert metrics["total_log_return_net"] <= metrics["total_log_return"] + 1e-9


def test_load_universe_panel_rejects_reads_into_holdout():
    import pytest

    with pytest.raises(ValueError, match="holdout"):
        research.load_universe_panel(
            ["BTCUSDT"],
            "1d",
            research.HOLDOUT_START - research.timedelta(days=1),
            research.HOLDOUT_START + research.timedelta(days=1),
        )


def test_stitched_metrics_gross_only_without_fee():
    trades = research.model_trade_results(
        y_true=np.array([0.01, -0.02, 0.03, -0.01]),
        y_pred=np.array([0.02, -0.01, 0.01, -0.02]),
    )
    metrics = research.stitched_metrics(
        trades, research.sharpe_to_annualized_rate("1d")
    )
    assert "sharpe_net" not in metrics
