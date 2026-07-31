"""Unit tests for the Hansen (1994) skewed Student-t density
(src/research/tmp/densities/hansen_skewt.py), part of notebook 6's Phase 3
distribution zoo. Mirrors tests/test_dist_lib5.py's conventions.

Hansen's skew-t is constructed to already have zero mean and unit variance
by design (no separate standardization step, unlike GED's gennorm rescale)
- the most load-bearing test here is confirming that at lam=0 the density
collapses EXACTLY to a standardized Student-t (same standardization
dist_lib5.py's GJR-t branch already uses), since a wrong sign in the (a, b)
shift-and-scale constants would silently break that nesting while still
looking plausible elsewhere. See dist_lib5.py's acerbi_szekely_z docstring
for this repo's precedent of a prior sign-error catch on exactly this kind
of formula.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as st
from scipy.integrate import quad

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src" / "research" / "tmp" / "densities"))

import hansen_skewt as H

SEED = 0

# (nu, lam) pairs used across several tests, including clearly asymmetric ones
PARAM_GRID = [(5.0, 0.3), (8.0, -0.3), (4.5, 0.5), (15.0, 0.0)]


def _pdf(z, shape):
    return np.exp(H.logpdf(np.atleast_1d(z), shape))


# --------------------------------------------------------------------------
# Density integrates to 1, mean~=0, var~=1
# --------------------------------------------------------------------------


class TestDensityMoments:
    @pytest.mark.parametrize("nu,lam", PARAM_GRID)
    def test_density_integrates_to_one(self, nu, lam):
        total, _ = quad(lambda z: _pdf(z, (nu, lam))[0], -60, 60, limit=500)
        assert total == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.parametrize("nu,lam", PARAM_GRID)
    def test_mean_is_zero(self, nu, lam):
        mean, _ = quad(lambda z: z * _pdf(z, (nu, lam))[0], -60, 60, limit=500)
        assert mean == pytest.approx(0.0, abs=1e-3)

    @pytest.mark.parametrize("nu,lam", PARAM_GRID)
    def test_variance_is_one(self, nu, lam):
        mean, _ = quad(lambda z: z * _pdf(z, (nu, lam))[0], -60, 60, limit=500)
        m2, _ = quad(lambda z: z**2 * _pdf(z, (nu, lam))[0], -60, 60, limit=500)
        var = m2 - mean**2
        assert var == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------
# ppf / cdf round-trip
# --------------------------------------------------------------------------


class TestPpfCdfRoundTrip:
    @pytest.mark.parametrize("nu,lam", [(5.0, 0.3), (8.0, -0.3), (4.5, 0.5), (6.0, -0.6)])
    @pytest.mark.parametrize("q", [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    def test_ppf_matches_cdf(self, nu, lam, q):
        # q values deliberately span both below and above the density's own
        # threshold q0 = (1-lam)/2, for both signs of lam.
        zq = H.ppf(q, (nu, lam))
        cdf_at_zq, _ = quad(lambda z: _pdf(z, (nu, lam))[0], -60, zq, limit=500)
        assert cdf_at_zq == pytest.approx(q, abs=1e-6)

    def test_ppf_vectorized_matches_scalar(self):
        qs = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
        shape = (6.0, 0.3)
        vec = H.ppf(qs, shape)
        scalar = np.array([H.ppf(float(q), shape) for q in qs])
        assert vec == pytest.approx(scalar)


# --------------------------------------------------------------------------
# lam=0 nesting: must reduce exactly to a standardized Student-t
# --------------------------------------------------------------------------


class TestSymmetricNesting:
    @pytest.mark.parametrize("nu", [3.0, 4.0, 5.0, 8.0, 15.0, 30.0])
    def test_logpdf_matches_standardized_student_t_at_lam_zero(self, nu):
        z = np.linspace(-6, 6, 101)
        got = H.logpdf(z, (nu, 0.0))
        c = np.sqrt(nu / (nu - 2.0))
        expected = st.t.logpdf(z * c, df=nu) + np.log(c)
        assert got == pytest.approx(expected, abs=1e-6)

    def test_ppf_matches_standardized_student_t_at_lam_zero(self):
        nu = 7.0
        c = np.sqrt(nu / (nu - 2.0))
        qs = np.array([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        got = H.ppf(qs, (nu, 0.0))
        expected = st.t.ppf(qs, df=nu) / c
        assert got == pytest.approx(expected, abs=1e-6)


# --------------------------------------------------------------------------
# fit()
# --------------------------------------------------------------------------


class TestFit:
    def test_fit_returns_none_on_insufficient_data(self):
        rng = np.random.default_rng(SEED)
        z = rng.standard_normal(10)  # < the 30-point floor
        assert H.fit(z) is None

    def test_fit_returns_none_on_degenerate_constant_input(self):
        z = np.ones(200)
        assert H.fit(z) is None

    def test_fit_recovers_known_params_on_synthetic_skewt_data(self):
        # simulate directly via inverse-cdf (ppf of uniforms) with a known
        # (nu, lam), then check fit recovers something in the right ballpark.
        rng = np.random.default_rng(SEED)
        true_nu, true_lam = 6.0, 0.35
        u = rng.uniform(1e-4, 1 - 1e-4, 20_000)
        z = H.ppf(u, (true_nu, true_lam))
        fitted = H.fit(z)
        assert fitted is not None
        nu_hat, lam_hat = fitted
        assert nu_hat == pytest.approx(true_nu, rel=0.35)
        assert lam_hat == pytest.approx(true_lam, abs=0.1)


# --------------------------------------------------------------------------
# es()
# --------------------------------------------------------------------------


class TestExpectedShortfall:
    @pytest.mark.parametrize("nu,lam", PARAM_GRID)
    def test_es_more_extreme_than_ppf_in_lower_tail(self, nu, lam):
        q = 0.01
        z_q = H.ppf(q, (nu, lam))
        es_q = H.es(q, (nu, lam))
        assert es_q < z_q

    def test_es_matches_direct_probability_space_integral(self):
        shape = (6.0, 0.2)
        q = 0.025
        es_q = H.es(q, shape)
        expected, _ = quad(lambda u: H.ppf(u, shape), 0.0, q, limit=200)
        expected /= q
        assert es_q == pytest.approx(expected, rel=1e-6)
