"""Unit tests for the GED (generalized error / generalized normal) innovation
family (src/research/tmp/densities/ged.py), notebook 6's Phase 3 distribution
zoo. Mirrors tests/test_dist_lib5.py's conventions.

NEXT_RUN_PROMPT.md's Phase 3 interface requires, for each family: unit
variance of the standardized density (numerically integrated), a ppf/cdf
round-trip, that logpdf matches a scipy reference where one exists (here:
the normal, at kappa=2), that fit returns None on pathological input, and
(for GED specifically) that the nesting case kappa=2 reduces to the known
standard normal.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as st
from scipy.integrate import quad

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src" / "research" / "tmp" / "densities"))

import ged as G  # type: ignore[import-not-found]

# --------------------------------------------------------------------------
# Unit variance of the standardized density
# --------------------------------------------------------------------------


class TestUnitVariance:
    @pytest.mark.parametrize("kappa", [0.8, 2.0, 4.0])
    def test_numerically_integrated_variance_is_one(self, kappa):
        s = G._unit_scale(kappa)
        var, _ = quad(
            lambda x: x**2 * st.gennorm.pdf(x, beta=kappa, scale=s),
            -50,
            50,
            limit=200,
        )
        assert var == pytest.approx(1.0, abs=1e-6)

    def test_logpdf_integrates_to_a_unit_variance_density(self):
        # cross-check via ged.logpdf itself, not just the internal helper.
        kappa = 1.5
        var, _ = quad(
            lambda x: x**2 * np.exp(G.logpdf(np.array([x]), (kappa,)))[0],
            -50,
            50,
            limit=200,
        )
        assert var == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# ppf / cdf round-trip
# --------------------------------------------------------------------------


class TestPpfCdfRoundTrip:
    @pytest.mark.parametrize("kappa", [0.8, 1.0, 2.0, 4.0])
    @pytest.mark.parametrize("q", [0.01, 0.1, 0.5, 0.9, 0.99])
    def test_gennorm_cdf_of_ppf_recovers_q(self, kappa, q):
        s = G._unit_scale(kappa)
        x = G.ppf(q, (kappa,))
        recovered_q = st.gennorm.cdf(x, beta=kappa, scale=s)
        assert recovered_q == pytest.approx(q, rel=1e-6, abs=1e-8)

    @pytest.mark.parametrize("kappa", [0.8, 2.0, 4.0])
    def test_quad_of_pdf_matches_ppf_based_cdf(self, kappa):
        # numerically differentiate the CDF via quad of the pdf (exp(logpdf))
        # and compare to the direct ppf inversion above, per the task's
        # "OR" alternative check.
        q = 0.2
        x = G.ppf(q, (kappa,))
        cdf_via_quad, _ = quad(
            lambda t: np.exp(G.logpdf(np.array([t]), (kappa,)))[0],
            -50,
            x,
            limit=200,
        )
        assert cdf_via_quad == pytest.approx(q, abs=1e-5)


# --------------------------------------------------------------------------
# kappa=2 nests the standard normal exactly
# --------------------------------------------------------------------------


class TestNormalNesting:
    def test_logpdf_matches_standard_normal_at_kappa_2(self):
        z = np.linspace(-4, 4, 201)
        got = G.logpdf(z, (2.0,))
        want = st.norm.logpdf(z)
        assert got == pytest.approx(want, abs=1e-9)

    def test_ppf_matches_standard_normal_at_kappa_2(self):
        q = np.array([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        got = G.ppf(q, (2.0,))
        want = st.norm.ppf(q)
        assert got == pytest.approx(want, abs=1e-6)


# --------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------


class TestFit:
    def test_fit_returns_none_on_zero_variance_input(self):
        assert G.fit(np.array([1.0, 1.0, 1.0])) is None

    def test_fit_returns_none_on_too_few_points(self):
        assert G.fit(np.array([1.0, 2.0, 3.0, 4.0, 5.0])) is None

    def test_fit_recovers_known_kappa_on_synthetic_data(self):
        rng = np.random.default_rng(0)
        true_kappa = 1.3
        s = G._unit_scale(true_kappa)
        z = st.gennorm.rvs(beta=true_kappa, scale=s, size=20_000, random_state=rng)
        fit = G.fit(z)
        assert fit is not None
        assert fit[0] == pytest.approx(true_kappa, abs=0.15)

    def test_fit_recovers_normal_kappa_near_2_on_gaussian_data(self):
        rng = np.random.default_rng(1)
        z = rng.standard_normal(20_000)
        fit = G.fit(z)
        assert fit is not None
        assert fit[0] == pytest.approx(2.0, abs=0.2)


# --------------------------------------------------------------------------
# es
# --------------------------------------------------------------------------


class TestExpectedShortfall:
    def test_es_at_kappa_2_matches_closed_form_normal_es(self):
        q = 0.01
        got = G.es(q, (2.0,))
        want = -st.norm.pdf(st.norm.ppf(q)) / q
        assert got == pytest.approx(want, rel=1e-4)

    def test_es_exceeds_var_in_magnitude(self):
        q = 0.01
        for kappa in [0.8, 1.0, 2.0, 4.0]:
            var_q = G.ppf(q, (kappa,))
            es_q = G.es(q, (kappa,))
            assert es_q < var_q < 0

    def test_es_is_more_negative_for_heavier_tail_kappa(self):
        # smaller kappa -> heavier shoulders/tails than normal -> deeper ES
        # at the same probability level.
        q = 0.01
        es_heavy = G.es(q, (0.8,))
        es_light = G.es(q, (2.0,))
        assert es_heavy < es_light
