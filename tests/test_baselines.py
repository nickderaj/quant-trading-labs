import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import research


def _synthetic_panel(n_times=40, n_symbols=15, seed=0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_times):
        dt = datetime(2022, 1, 1) + timedelta(hours=4 * t)  # noqa: DTZ001 - naive, matches production panel datetimes
        for s in range(n_symbols):
            rows.append(
                {
                    "datetime": dt,
                    "symbol": f"S{s}",
                    "fwd_return_1": float(rng.normal(scale=0.01)),
                }
            )
    return pl.DataFrame(rows)


def test_equal_weight_basket_matches_manual_mean():
    panel = _synthetic_panel(n_times=5, n_symbols=4, seed=1)
    basket = research.equal_weight_basket_returns(panel)
    manual = (
        panel.group_by("datetime")
        .agg(pl.col("fwd_return_1").mean())
        .sort("datetime")["fwd_return_1"]
        .to_numpy()
    )
    np.testing.assert_allclose(basket["trade_log_return"].to_numpy(), manual)


def test_random_dollar_neutral_metrics_shape_and_distribution():
    panel = _synthetic_panel()
    ar = research.sharpe_to_annualized_rate("4h")
    result = research.random_dollar_neutral_metrics(panel, ar, no_seeds=15, seed=0)
    assert len(result) == 15
    assert set(result.columns) >= {
        "seed",
        "sharpe",
        "total_log_return",
        "compound_return",
    }
    # a null (random-ranking) baseline on i.i.d. noise shouldn't be
    # systematically wildly one-sided - mean sharpe should be roughly near 0
    assert abs(result["sharpe"].mean()) < 3.0


def test_random_dollar_neutral_metrics_charges_cost_when_fee_given():
    panel = _synthetic_panel()
    ar = research.sharpe_to_annualized_rate("4h")
    gross = research.random_dollar_neutral_metrics(panel, ar, no_seeds=10, seed=1)
    net = research.random_dollar_neutral_metrics(
        panel, ar, no_seeds=10, seed=1, taker_fee=0.01, slippage=0.001
    )
    # heavy fees on the same seeds/random rankings should drag returns down
    assert net["total_log_return"].mean() < gross["total_log_return"].mean()


def test_bootstrap_ci_contains_true_mean_most_of_the_time():
    rng = np.random.default_rng(0)
    contained = 0
    trials = 30
    for i in range(trials):
        sample = rng.normal(loc=0.5, scale=1.0, size=100)
        lo, hi = research.bootstrap_ci(sample, n_boot=500, ci=0.95, seed=i)
        if lo <= 0.5 <= hi:
            contained += 1
    assert contained / trials > 0.7  # loose check, not exact 95% coverage


def test_bootstrap_ci_excludes_far_off_null_for_strong_signal():
    values = np.full(200, 1.0)  # constant, no variance
    lo, hi = research.bootstrap_ci(values, n_boot=200)
    assert lo == hi == 1.0


def test_bootstrap_ci_empty_input_returns_nan():
    lo, hi = research.bootstrap_ci(np.array([]))
    assert np.isnan(lo) and np.isnan(hi)
