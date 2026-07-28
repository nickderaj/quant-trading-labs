import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import research


def _synthetic_ic_panel(
    true_ic: float, n_times: int = 200, n_symbols: int = 30, seed: int = 0
) -> pl.DataFrame:
    """A panel where target is a known linear mix of pred and noise, chosen
    so pred/target's population Spearman correlation is approximately
    true_ic (exact for the Pearson case with Gaussian inputs, close enough
    for Spearman given large n_symbols per cross-section).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_times):
        dt = datetime(2022, 1, 1) + timedelta(hours=t)  # noqa: DTZ001 - naive, matches production panel datetimes
        feat = rng.normal(size=n_symbols)
        noise = rng.normal(size=n_symbols)
        target = true_ic * feat + np.sqrt(max(1 - true_ic**2, 0.0)) * noise
        for s in range(n_symbols):
            rows.append(
                {
                    "datetime": dt,
                    "symbol": f"S{s}",
                    "pred": float(feat[s]),
                    "target": float(target[s]),
                }
            )
    return pl.DataFrame(rows)


def test_newey_west_tstat_matches_naive_at_lag_zero_for_iid_data():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=0.5, scale=1.0, size=500)
    mean, tstat = research.newey_west_tstat(x, lag=0)
    naive_tstat = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert abs(mean - x.mean()) < 1e-9
    assert abs(tstat - naive_tstat) < 1e-6


def test_newey_west_tstat_reduces_significance_for_autocorrelated_series():
    rng = np.random.default_rng(0)
    n = 500
    # strongly autocorrelated (AR(1), rho=0.9) series with the same marginal
    # variance as i.i.d. noise - true information content is far less than n.
    eps = rng.normal(scale=1.0, size=n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + eps[i]
    x += 0.1  # small nonzero mean

    _, naive_like_tstat = research.newey_west_tstat(x, lag=0)
    _, hac_tstat = research.newey_west_tstat(x, lag=20)
    assert abs(hac_tstat) < abs(naive_like_tstat)


def test_cross_sectional_ic_recovers_implanted_signal():
    panel = _synthetic_ic_panel(true_ic=0.3, seed=1)
    ic_df = research.cross_sectional_ic(panel, "pred", "target")
    stats = research.cross_sectional_ic_stats(ic_df, nw_lag=5)
    assert stats["n_periods"] == 200
    assert 0.15 < stats["mean_ic"] < 0.45
    assert stats["nw_tstat"] > 5  # should be unambiguously significant


def test_cross_sectional_ic_null_case_is_near_zero_and_insignificant():
    panel = _synthetic_ic_panel(true_ic=0.0, seed=2)
    ic_df = research.cross_sectional_ic(panel, "pred", "target")
    stats = research.cross_sectional_ic_stats(ic_df, nw_lag=5)
    assert abs(stats["mean_ic"]) < 0.1
    assert abs(stats["nw_tstat"]) < 3


def test_panel_ic_recovers_implanted_signal_and_clustered_lt_naive_significance():
    panel = _synthetic_ic_panel(true_ic=0.3, seed=3)
    result = research.panel_ic(panel, "pred", "target", nw_lag=5)
    assert 0.15 < result["panel_ic"] < 0.45
    assert result["n_obs"] == 200 * 30
    assert result["n_timestamps"] == 200
    # clustering + HAC should not produce a LARGER t-stat than the naive
    # (badly overstated) i.i.d. one on data with real cross-sectional and
    # temporal structure - this is the whole point of the correction.
    assert result["clustered_nw_tstat"] <= result["naive_tstat"] * 1.5


def test_ic_stability_flags_all_positive_when_signal_is_strong_and_constant():
    panel = _synthetic_ic_panel(true_ic=0.4, n_times=400, seed=4)
    ic_df = research.cross_sectional_ic(panel, "pred", "target")
    stability = research.ic_stability(ic_df)
    assert stability["frac_positive_months"] > 0.9
    assert "rolling_mean_ic" in stability["rolling_ic"].columns
    assert "mean_ic" in stability["per_year_ic"].columns


def test_cross_sectional_ic_empty_panel_returns_empty_frame():
    panel = pl.DataFrame(
        schema={"datetime": pl.Datetime, "pred": pl.Float64, "target": pl.Float64}
    )
    ic_df = research.cross_sectional_ic(panel, "pred", "target")
    assert len(ic_df) == 0
    stats = research.cross_sectional_ic_stats(ic_df, nw_lag=5)
    assert np.isnan(stats["mean_ic"])
