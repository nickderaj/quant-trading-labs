"""Unit tests for the Johnson SU innovation density
(src/research/tmp/densities/johnsonsu.py), part of notebook 6's Phase 3
wider-distribution-zoo work. Mirrors tests/test_dist_lib5.py's conventions:
class-per-topic, pytest.approx for numeric checks, explicit reasoning in
comments for why each assertion matters.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate
from scipy import stats as st

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src" / "research" / "tmp" / "densities"))

import johnsonsu as J  # type: ignore[import-not-found]

SEED = 0

SHAPES = [
    (0.0, 2.0),  # symmetric
    (1.5, 3.0),  # asymmetric (gamma != 0)
    (-2.0, 1.5),  # asymmetric, opposite skew, lower delta (heavier tail)
]


# --------------------------------------------------------------------------
# Standardization: mean 0, variance 1
# --------------------------------------------------------------------------


class TestStandardization:
    @pytest.mark.parametrize("shape", SHAPES)
    def test_loc_scale_gives_mean_zero_var_one_via_scipy_stats(self, shape):
        gamma, delta = shape
        loc, scale = J._loc_scale(gamma, delta)
        mu, var = st.johnsonsu(gamma, delta, loc=loc, scale=scale).stats(moments="mv")
        assert mu == pytest.approx(0.0, abs=1e-8)
        assert var == pytest.approx(1.0, rel=1e-8)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_logpdf_integrates_to_mean_zero_var_one_numerically(self, shape):
        # independent check of the same claim, integrating our own logpdf
        # directly (not scipy's .stats()) over the real line.
        def pdf(z):
            return np.exp(J.logpdf(np.array([z]), shape))[0]

        mean, _ = integrate.quad(lambda z: z * pdf(z), -50, 50, limit=200)
        second_moment, _ = integrate.quad(lambda z: z * z * pdf(z), -50, 50, limit=200)
        var = second_moment - mean**2
        assert mean == pytest.approx(0.0, abs=1e-3)
        assert var == pytest.approx(1.0, abs=1e-2)


# --------------------------------------------------------------------------
# ppf / cdf round trip
# --------------------------------------------------------------------------


class TestPpfCdfRoundTrip:
    @pytest.mark.parametrize("shape", SHAPES)
    def test_ppf_cdf_round_trip(self, shape):
        gamma, delta = shape
        loc, scale = J._loc_scale(gamma, delta)
        for q in [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]:
            z_q = J.ppf(q, shape)
            back = st.johnsonsu.cdf(z_q, gamma, delta, loc=loc, scale=scale)
            assert back == pytest.approx(q, abs=1e-6)

    def test_ppf_vectorized_over_q(self):
        shape = (1.0, 3.0)
        qs = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
        out = J.ppf(qs, shape)
        assert out.shape == qs.shape
        assert np.all(np.diff(out) > 0)  # ppf must be monotonic increasing


# --------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------


class TestFit:
    def test_fit_returns_none_on_constant_array(self):
        z = np.full(500, 3.0)
        assert J.fit(z) is None

    def test_fit_returns_none_on_too_few_points(self):
        rng = np.random.default_rng(SEED)
        z = rng.normal(0, 1, 10)  # below the 30-point floor
        assert J.fit(z) is None

    @pytest.mark.parametrize("shape", SHAPES)
    def test_fit_recovers_shape_on_large_synthetic_sample(self, shape):
        gamma, delta = shape
        loc, scale = J._loc_scale(gamma, delta)
        rng = np.random.default_rng(SEED)
        z = st.johnsonsu.rvs(
            gamma, delta, loc=loc, scale=scale, size=20_000, random_state=rng
        )
        fitted = J.fit(z)
        assert fitted is not None
        g_hat, d_hat = fitted
        assert g_hat == pytest.approx(gamma, abs=0.3)
        assert d_hat == pytest.approx(delta, abs=0.6)


# --------------------------------------------------------------------------
# es (expected shortfall)
# --------------------------------------------------------------------------


class TestExpectedShortfall:
    @pytest.mark.parametrize("shape", SHAPES)
    def test_es_finite_negative_and_more_extreme_than_ppf(self, shape):
        q = 0.01
        z_q = J.ppf(q, shape)
        es_q = J.es(q, shape)
        assert np.isfinite(es_q)
        assert es_q < 0
        # ES averages over the whole tail beyond the quantile, so it must be
        # at least as extreme (more negative) than the quantile itself.
        assert es_q < z_q

    def test_es_consistent_with_direct_quad_integration_reference(self):
        # independent re-derivation of ES via quad on ppf, to catch a
        # transcription bug in J.es itself (not just re-testing scipy).
        shape = (0.5, 2.5)
        q = 0.025
        value, _ = integrate.quad(lambda u: J.ppf(u, shape), 0.0, q, limit=200)
        expected = value / q
        assert J.es(q, shape) == pytest.approx(expected, rel=1e-6)


# --------------------------------------------------------------------------
# near-normal sanity check
# --------------------------------------------------------------------------


class TestNearNormalLimit:
    def test_logpdf_at_zero_close_to_normal_for_large_delta_symmetric(self):
        shape = (0.0, 20.0)
        got = J.logpdf(np.array([0.0]), shape)[0]
        ref = st.norm.logpdf(0.0)
        # not exact - Johnson SU only approaches normal as delta -> infinity
        # with gamma = 0 - but should be in the right ballpark.
        assert got == pytest.approx(ref, abs=0.2)
