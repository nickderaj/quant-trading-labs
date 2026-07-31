"""Unit tests for the Normal-Inverse Gaussian (NIG) density
(src/research/tmp/densities/nig.py), part of notebook 6's Phase 3
distribution zoo. Mirrors tests/test_dist_lib6_ged.py and
tests/test_dist_lib6_hansen_skewt.py's conventions.

Unlike GED and Hansen skew-t, NIG has no closed-form ppf/cdf, so ppf is a
numerical CDF inversion (quad + brentq) here - the round-trip test below is
the most load-bearing one, since a sign error or bad bracket would silently
give a plausible-looking but wrong quantile. The zero-mean/unit-variance
(delta, mu) solved from (alpha, beta) is the other place a sign error could
hide (per dist_lib5.py's acerbi_szekely_z precedent for this kind of bug),
so moments are checked directly via numerical integration for several shape
pairs, including beta=0 (symmetric) and beta != 0 (skewed) cases.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as st
from scipy.integrate import quad

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src" / "research" / "tmp" / "densities"))

import nig as N

SEED = 0

# (alpha, beta) pairs used across several tests: symmetric and skewed, small
# and large alpha.
PARAM_GRID = [(2.0, 0.0), (3.0, 1.0), (5.0, -2.0), (1.5, 0.5)]


def _pdf(z, shape):
    return np.exp(N.logpdf(np.atleast_1d(z), shape))


# --------------------------------------------------------------------------
# Density integrates to 1, mean~=0, var~=1
# --------------------------------------------------------------------------


class TestDensityMoments:
    @pytest.mark.parametrize("alpha,beta", PARAM_GRID)
    def test_density_integrates_to_one(self, alpha, beta):
        total, _ = quad(lambda z: _pdf(z, (alpha, beta))[0], -80, 80, limit=500)
        assert total == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.parametrize("alpha,beta", PARAM_GRID)
    def test_mean_is_zero(self, alpha, beta):
        mean, _ = quad(lambda z: z * _pdf(z, (alpha, beta))[0], -80, 80, limit=500)
        assert mean == pytest.approx(0.0, abs=1e-3)

    @pytest.mark.parametrize("alpha,beta", PARAM_GRID)
    def test_variance_is_one(self, alpha, beta):
        mean, _ = quad(lambda z: z * _pdf(z, (alpha, beta))[0], -80, 80, limit=500)
        m2, _ = quad(lambda z: z**2 * _pdf(z, (alpha, beta))[0], -80, 80, limit=500)
        var = m2 - mean**2
        assert var == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------
# ppf / cdf round-trip
# --------------------------------------------------------------------------


class TestPpfCdfRoundTrip:
    @pytest.mark.parametrize("alpha,beta", [(2.0, 0.0), (3.0, 1.0), (5.0, -2.0)])
    @pytest.mark.parametrize("q", [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    def test_ppf_matches_cdf(self, alpha, beta, q):
        zq = N.ppf(q, (alpha, beta))
        cdf_at_zq, _ = quad(lambda z: _pdf(z, (alpha, beta))[0], -80, zq, limit=500)
        assert cdf_at_zq == pytest.approx(q, abs=1e-5)

    def test_ppf_vectorized_matches_scalar(self):
        qs = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
        shape = (3.0, 1.0)
        vec = N.ppf(qs, shape)
        scalar = np.array([N.ppf(float(q), shape) for q in qs])
        assert vec == pytest.approx(scalar)

    def test_ppf_is_monotonic_increasing_in_q(self):
        qs = np.linspace(0.01, 0.99, 25)
        shape = (2.5, -1.0)
        vals = N.ppf(qs, shape)
        assert np.all(np.diff(vals) > 0)


# --------------------------------------------------------------------------
# fit()
# --------------------------------------------------------------------------


class TestFit:
    def test_fit_returns_none_on_insufficient_data(self):
        rng = np.random.default_rng(SEED)
        z = rng.standard_normal(10)  # < the 30-point floor
        assert N.fit(z) is None

    def test_fit_returns_none_on_degenerate_constant_input(self):
        z = np.ones(200)
        assert N.fit(z) is None

    def test_fit_recovers_known_params_on_synthetic_nig_data(self):
        # simulate directly via inverse-cdf (ppf of uniforms) with a known
        # (alpha, beta), then check fit recovers something in the right
        # ballpark. NIG's (alpha, beta) likelihood has a well-known ridge
        # (alpha and beta are highly correlated - both control tail
        # heaviness/skew jointly), so exact recovery at a few thousand
        # points is noisier than GED's or Hansen skew-t's better-conditioned
        # shape params; an ad hoc check before writing this test confirmed
        # the negative log-likelihood at the fitted params was in fact lower
        # than at the true params (as an MLE optimum must be), so the loose
        # tolerances below are about estimator variance, not a fit() bug.
        rng = np.random.default_rng(SEED)
        true_alpha, true_beta = 3.0, 1.0
        u = rng.uniform(1e-3, 1 - 1e-3, 3_000)
        z = N.ppf(u, (true_alpha, true_beta))
        fitted = N.fit(z)
        assert fitted is not None
        alpha_hat, beta_hat = fitted
        assert alpha_hat > 0
        assert alpha_hat == pytest.approx(true_alpha, rel=1.0)
        # correct sign and right order of magnitude, rather than a tight
        # value match (see ridge note above)
        assert beta_hat > 0
        assert beta_hat == pytest.approx(true_beta, rel=2.0)


# --------------------------------------------------------------------------
# es()
# --------------------------------------------------------------------------


class TestExpectedShortfall:
    @pytest.mark.parametrize("alpha,beta", PARAM_GRID)
    def test_es_more_extreme_than_ppf_in_lower_tail(self, alpha, beta):
        q = 0.01
        z_q = N.ppf(q, (alpha, beta))
        es_q = N.es(q, (alpha, beta))
        assert np.isfinite(es_q)
        assert es_q < z_q

    def test_es_matches_direct_probability_space_integral(self):
        # es() uses a fixed 24-point Gauss-Legendre rule (bounding ppf calls
        # to a fixed count - see nig.py's es() docstring for why an adaptive
        # quad on top of the root-finding ppf was ~1000x too slow), so this
        # compares against an adaptive-quad reference at a loose relative
        # tolerance rather than requiring near-exact agreement.
        shape = (3.0, 1.0)
        q = 0.025
        es_q = N.es(q, shape)
        expected, _ = quad(lambda u: N.ppf(u, shape), 0.0, q, limit=50)
        expected /= q
        assert es_q == pytest.approx(expected, rel=1e-2)

    def test_es_is_reasonably_fast(self):
        shape = (3.0, 1.0)
        t0 = time.time()
        N.es(0.01, shape)
        elapsed = time.time() - t0
        assert elapsed < 5.0  # fixed-node GL keeps this to ~24 ppf calls


# --------------------------------------------------------------------------
# large-alpha, beta=0 limit approaches the standard normal
# --------------------------------------------------------------------------


class TestNormalLimit:
    def test_logpdf_at_zero_approaches_standard_normal_for_large_alpha(self):
        got = N.logpdf(np.array([0.0]), (50.0, 0.0))[0]
        want = st.norm.logpdf(0.0)
        assert got == pytest.approx(want, abs=1e-2)

    def test_logpdf_curve_approaches_standard_normal_for_large_alpha(self):
        z = np.linspace(-2, 2, 21)
        got = N.logpdf(z, (50.0, 0.0))
        want = st.norm.logpdf(z)
        assert got == pytest.approx(want, abs=5e-2)
