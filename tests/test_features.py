import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import features


def _synthetic_symbol_df(
    n: int = 300, seed: int = 0, symbol: str = "BTCUSDT"
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2022, 1, 1, tzinfo=UTC)
    datetimes = [start + timedelta(hours=4 * i) for i in range(n)]
    log_returns = rng.normal(scale=0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    open_ = np.roll(close, 1)
    open_[0] = 100.0
    high = np.maximum(open_, close) * (1 + rng.random(n) * 0.005)
    low = np.minimum(open_, close) * (1 - rng.random(n) * 0.005)
    volume = rng.uniform(100, 1000, size=n)
    taker_buy_volume = volume * rng.uniform(0.2, 0.8, size=n)
    count = rng.integers(50, 500, size=n).astype(np.int64)

    return pl.DataFrame(
        {
            "datetime": datetimes,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "count": count,
            "taker_buy_volume": taker_buy_volume,
            "symbol": [symbol] * n,
        }
    )


def _synthetic_panel(symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"), n=300) -> pl.DataFrame:
    parts = [
        _synthetic_symbol_df(n=n, seed=i, symbol=sym) for i, sym in enumerate(symbols)
    ]
    return pl.concat(parts).sort(["datetime", "symbol"])


CAUSAL_BUILDERS = [
    features.add_order_flow_features,
    features.add_seasonality_features,
    features.add_vol_features,
    features.add_momentum_features,
]


def test_raw_features_are_causal_under_truncation():
    """Truncating history at any point must not change any earlier feature
    value - if it did, that feature would be using data from beyond the
    truncation point, i.e. the future relative to that point.
    """
    df = _synthetic_symbol_df(n=200)
    full = features.add_all_raw_features(df)

    for cut in (50, 120, 199):
        truncated = features.add_all_raw_features(df.head(cut))
        feature_cols = [c for c in full.columns if c not in df.columns]
        for col in feature_cols:
            full_vals = full[col].head(cut).to_numpy()
            trunc_vals = truncated[col].to_numpy()
            np.testing.assert_allclose(
                full_vals,
                trunc_vals,
                equal_nan=True,
                err_msg=f"{col} is not causal: changed when future rows were removed",
            )


def test_forward_return_is_shifted_forward_not_causal_by_design():
    df = _synthetic_symbol_df(n=50)
    out = df.with_columns(features.forward_return())
    expected = np.log(df["close"][1:].to_numpy() / df["close"][:-1].to_numpy())
    got = out["fwd_return_1"][:-1].to_numpy()
    np.testing.assert_allclose(got, expected)
    assert out["fwd_return_1"][-1] is None


def test_order_flow_imbalance_bounds_and_formula():
    df = _synthetic_symbol_df(n=20)
    out = df.with_columns(features.order_flow_imbalance())
    imbalance = out["order_flow_imbalance"].to_numpy()
    assert (imbalance >= -1).all() and (imbalance <= 1).all()
    expected = 2 * (df["taker_buy_volume"] / df["volume"]).to_numpy() - 1
    np.testing.assert_allclose(imbalance, expected)


def test_momentum_matches_manual_log_return():
    df = _synthetic_symbol_df(n=30)
    out = df.with_columns(features.momentum(4))
    close = df["close"].to_numpy()
    expected = np.log(close[4:] / close[:-4])
    got = out["momentum_4"].to_numpy()[4:]
    np.testing.assert_allclose(got, expected)
    # first 4 rows have no full lookback -> null
    assert out["momentum_4"][:4].is_null().all()


def test_mean_reversion_is_negative_momentum():
    df = _synthetic_symbol_df(n=30)
    out = df.with_columns(features.momentum(4), features.mean_reversion(4))
    np.testing.assert_allclose(
        out["mean_reversion_4"].to_numpy(), -out["momentum_4"].to_numpy()
    )


def test_no_lookahead_magnitude_tripwire():
    """A feature this strongly correlated with its own forward return at bar
    level would be the lookahead-bug class of result the run's guardrails
    call out (Sharpe 8.89 from a shift() mistake), not real alpha - assert
    our raw features don't trip that magnitude on synthetic random-walk data
    with no implanted signal.
    """
    df = _synthetic_symbol_df(n=2000)
    featured = features.add_all_raw_features(df).with_columns(features.forward_return())
    feature_cols = [
        c
        for c in featured.columns
        if c not in df.columns and not c.startswith("fwd_return_")
    ]
    fwd = featured["fwd_return_1"].to_numpy()
    valid = ~np.isnan(fwd)
    for col in feature_cols:
        vals = featured[col].to_numpy()
        mask = valid & ~np.isnan(vals)
        if mask.sum() < 30:
            continue
        corr = np.corrcoef(vals[mask], fwd[mask])[0, 1]
        assert abs(corr) < 0.10, (
            f"{col} correlates {corr:.3f} with fwd_return_1 - check for a shift() bug"
        )


def test_cross_sectional_zscore_is_zero_mean_unit_std_per_bar():
    panel = _synthetic_panel()
    featured = panel.with_columns(features.momentum(1)).pipe(
        lambda df: features.add_cross_sectional_features(df, ["momentum_1"])
    )
    per_bar = (
        featured.drop_nulls(["momentum_1_cs_z"])
        .group_by("datetime")
        .agg(
            pl.col("momentum_1_cs_z").mean().alias("mean"),
            pl.col("momentum_1_cs_z").std().alias("std"),
        )
    )
    # only bars with all 3 symbols present are meaningful (std needs >1 obs)
    per_bar = per_bar.filter(pl.col("std").is_not_null())
    np.testing.assert_allclose(per_bar["mean"].to_numpy(), 0.0, atol=1e-8)
    np.testing.assert_allclose(per_bar["std"].to_numpy(), 1.0, atol=1e-8)


def test_funding_rate_join_is_backward_causal():
    bars = _synthetic_symbol_df(n=10)
    funding = pl.DataFrame(
        {
            "datetime": [
                datetime(2022, 1, 1, tzinfo=UTC) - timedelta(hours=1),
                datetime(2022, 1, 1, 8, tzinfo=UTC),
                datetime(2022, 1, 1, 20, tzinfo=UTC),
            ],
            "funding_rate": [0.0001, 0.0002, 0.0003],
        }
    ).with_columns(pl.col("datetime").dt.replace_time_zone(None))
    bars = bars.with_columns(pl.col("datetime").dt.replace_time_zone(None))

    out = features.add_funding_rate_feature(bars, funding)
    # bar at hour 0 should see the funding rate published just before it (0.0001)
    assert out["funding_rate"][0] == 0.0001
    # bar at hour 8 (index 2, since bars are every 4h) sees the 08:00 rate
    assert out["funding_rate"][2] == 0.0002
    # no bar should ever see the 20:00 rate before it actually happens
    hour_20_bar = out["datetime"][5]
    before_20h = out.filter(pl.col("datetime") < hour_20_bar)
    assert (before_20h["funding_rate"] != 0.0003).all()


def test_build_feature_panel_end_to_end_smoke():
    panel = _synthetic_panel(n=150)
    out = features.build_feature_panel(panel)
    assert "fwd_return_1" in out.columns
    assert "order_flow_imbalance_cs_z" in out.columns
    assert len(out) == len(panel)
