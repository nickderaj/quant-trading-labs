"""Unit tests for notebook-6-local machinery (src/research/tmp/dist_lib6.py):
the Phase 4 violation-process PMFs (count/duration models) - the new
*modelling* code this notebook introduces, per the same convention as
tests/test_dist_lib5.py (driver scripts themselves aren't unit-tested by
design; new modelling machinery is).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as st

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src" / "research" / "tmp"))
sys.path.insert(0, str(_ROOT / "src"))

import dist_lib6 as L6

SEED = 0


class TestViolationBlocksAndDurations:
    def test_counts_sum_to_total_hits(self):
        rng = np.random.default_rng(SEED)
        hit = rng.random(2000) < 0.02
        counts, _durations = L6.violation_blocks_and_durations(hit, block_size=100)
        assert counts.sum() == hit.sum()

    def test_durations_are_gaps_minus_one(self):
        hit = np.zeros(50, dtype=bool)
        hit[[5, 10, 20]] = True
        _counts, durations = L6.violation_blocks_and_durations(hit, block_size=10)
        # gaps: 10-5=5, 20-10=10 -> durations = gap-1
        assert list(durations) == [4, 9]

    def test_too_few_blocks_returns_empty(self):
        hit = np.zeros(5, dtype=bool)
        counts, durations = L6.violation_blocks_and_durations(hit, block_size=100)
        assert len(counts) == 0
        assert len(durations) == 0


class TestCountModels:
    def test_poisson_truth_not_significantly_overdispersed(self):
        rng = np.random.default_rng(SEED)
        counts = rng.poisson(3.0, 600)
        pois = L6.fit_poisson_counts(counts)
        nb = L6.fit_nb_counts(counts)
        assert pois is not None and nb is not None
        _lr, p = L6.boundary_lr_test(nb["loglik"], pois["loglik"])
        assert p > 0.05

    def test_negative_binomial_truth_is_significantly_overdispersed(self):
        rng = np.random.default_rng(SEED)
        counts = rng.negative_binomial(2.0, 0.3, 600)
        pois = L6.fit_poisson_counts(counts)
        nb = L6.fit_nb_counts(counts)
        assert pois is not None and nb is not None
        _lr, p = L6.boundary_lr_test(nb["loglik"], pois["loglik"])
        assert p < 0.01
        assert nb["mu"] == pytest.approx(np.mean(counts), rel=0.05)

    def test_fit_poisson_returns_none_on_all_zero(self):
        assert L6.fit_poisson_counts(np.zeros(20)) is None

    def test_fit_nb_on_equidispersed_data_finds_near_zero_alpha(self):
        # constant counts (zero variance, equidispersed relative to itself):
        # unlike distributions._fit_nbinom's method-of-moments estimator
        # (which requires var>mean and nulls otherwise), this MLE-based fit
        # can and should still converge - to alpha near the Poisson boundary
        # (alpha=0), which is exactly what the boundary LR test is built to
        # recognize as "no real evidence of overdispersion" rather than a
        # fit failure.
        counts = np.full(50, 5)
        fit = L6.fit_nb_counts(counts)
        assert fit is not None
        assert fit["alpha"] < 1e-4

    def test_fit_nb_returns_none_on_all_zero(self):
        assert L6.fit_nb_counts(np.zeros(20)) is None

    def test_boundary_lr_test_is_half_the_naive_chi2_pvalue(self):
        lr = 4.0
        _lr_out, p = L6.boundary_lr_test(2.0, 0.0)  # ll_full - ll_null = 2.0 -> LR=4.0
        naive_p = float(st.chi2.sf(lr, df=1))
        assert p == pytest.approx(0.5 * naive_p)


class TestDurationModels:
    def test_geometric_truth_beta_near_one(self):
        rng = np.random.default_rng(SEED)
        durations = rng.geometric(0.3, 3000) - 1
        geo = L6.fit_geometric_durations(durations)
        dw = L6.fit_discrete_weibull_durations(durations)
        assert geo is not None and dw is not None
        assert dw["beta"] == pytest.approx(1.0, abs=0.15)

    def test_clustered_durations_have_beta_below_one_and_are_significant(self):
        rng = np.random.default_rng(SEED)
        # mixture of many short gaps and a few long ones -> falling hazard
        durations = np.concatenate([rng.geometric(0.8, 1500) - 1, rng.geometric(0.05, 500) - 1])
        rng.shuffle(durations)
        geo = L6.fit_geometric_durations(durations)
        dw = L6.fit_discrete_weibull_durations(durations)
        assert geo is not None and dw is not None
        assert dw["beta"] < 1.0
        lr = max(0.0, 2.0 * (dw["loglik"] - geo["loglik"]))
        p = float(st.chi2.sf(lr, df=1))
        assert p < 0.01

    def test_discrete_weibull_nests_geometric_at_beta_one(self):
        # geometric's own log-likelihood should equal discrete-Weibull's
        # negloglik evaluated at beta=1 for the same q
        rng = np.random.default_rng(SEED)
        durations = rng.geometric(0.4, 500) - 1
        geo = L6.fit_geometric_durations(durations)
        assert geo is not None
        nll_at_geo_q_beta1 = L6._discrete_weibull_negloglik(
            np.array([np.log(geo["q"] / (1 - geo["q"]))]), durations, fix_beta1=True,
        )
        assert -nll_at_geo_q_beta1 == pytest.approx(geo["loglik"], rel=1e-8)
