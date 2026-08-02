"""Unit tests for notebook-9-local machinery (src/research/tmp/research_lib9.py):
the AR(1)-in-differences mean-reversion test and rolling z-score IC used by
Phase 4's spread-mean-reversion probe. Mirrors tests/test_commod_lib8.py's
conventions.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src" / "research" / "tmp"))

import research_lib9 as R9

SEED = 0


class TestOlsAr1Diff:
    def test_strongly_mean_reverting_series_detected(self):
        """A synthetic OU-like AR(1) process with strong mean reversion should
        recover a negative beta with a large |t-stat| and be flagged
        mean_reverting=True."""
        rng = np.random.default_rng(SEED)
        n = 2000
        v = np.zeros(n)
        true_beta = -0.3  # strongly mean-reverting
        for t in range(1, n):
            v[t] = v[t - 1] * (1 + true_beta) + rng.normal(0, 1)
        result = R9.ols_ar1_diff(v)
        assert result["beta"] < 0
        assert abs(result["beta"] - true_beta) < 0.05
        assert abs(result["t_stat_beta"]) > 10  # very strong signal at n=2000
        assert result["mean_reverting"] is True
        assert result["half_life_days"] is not None
        assert result["half_life_days"] > 0

    def test_random_walk_not_flagged_mean_reverting(self):
        """A pure random walk (beta=0 in expectation) should not be flagged
        as mean-reverting, and its beta should be close to zero."""
        rng = np.random.default_rng(SEED)
        n = 2000
        v = np.cumsum(rng.normal(0, 1, n))
        result = R9.ols_ar1_diff(v)
        assert abs(result["beta"]) < 0.02
        assert result["mean_reverting"] is False

    def test_half_life_none_for_explosive_or_random_walk_beta(self):
        """half_life_days is only defined for -1 < beta < 0 (stationary case);
        a non-negative or explosive beta should report None instead of a
        nonsensical or complex half-life."""
        rng = np.random.default_rng(SEED)
        v = 1.0 + rng.normal(0, 1e-6, 50)  # near-constant, tiny noise, beta ~ 0
        result = R9.ols_ar1_diff(v)
        if result["beta"] >= 0:
            assert result["half_life_days"] is None

    def test_half_life_matches_closed_form(self):
        """For a known beta, half_life should match the closed-form
        -ln(2)/ln(1+beta) exactly (a correctness check on the formula, not a
        statistical property)."""
        rng = np.random.default_rng(SEED)
        n = 5000
        true_beta = -0.05
        v = np.zeros(n)
        for t in range(1, n):
            v[t] = v[t - 1] * (1 + true_beta) + rng.normal(0, 0.1)
        result = R9.ols_ar1_diff(v)
        expected_half_life = -np.log(2) / np.log(1 + result["beta"])
        assert result["half_life_days"] == pytest.approx(expected_half_life)

    def test_raises_on_too_few_observations(self):
        with pytest.raises(ValueError):
            R9.ols_ar1_diff(np.array([1.0, 2.0]))


class TestZscoreIc:
    def test_mean_reverting_series_gives_negative_ic(self):
        """A strongly mean-reverting synthetic series should show a clearly
        negative Spearman IC between the rolling z-score and the forward
        change (high z predicts a subsequent decline)."""
        rng = np.random.default_rng(SEED)
        n = 3000
        v = np.zeros(n)
        for t in range(1, n):
            v[t] = v[t - 1] * 0.9 + rng.normal(0, 1)
        result = R9.zscore_ic(v, window=60, horizon=5)
        assert result["ic"] is not None
        assert result["ic"] < -0.1
        assert result["p_value"] < 0.01

    def test_too_short_series_returns_none(self):
        v = np.arange(20, dtype=float)
        result = R9.zscore_ic(v, window=60, horizon=5)
        assert result["ic"] is None
        assert result["p_value"] is None

    def test_n_matches_finite_overlap_count(self):
        rng = np.random.default_rng(SEED)
        v = rng.normal(0, 1, 500).cumsum()
        result = R9.zscore_ic(v, window=60, horizon=5)
        assert result["n"] > 0
        assert result["n"] <= len(v)
