"""Unit tests for notebook-5-local machinery (src/research/tmp/dist_lib5.py):
Hill estimator, GJR-GARCH, and GPD/EVT (peaks-over-threshold). Mirrors
tests/test_distributions.py's conventions; unlike dist_lib.py (no test suite,
by design - forecasting-contest driver machinery), the runbook for notebook 5
asks explicitly for tests on the new GJR/GPD machinery, since it introduces
new modelling code (not just new driver scripts) to this research programme.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as st

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src" / "research" / "tmp"))
sys.path.insert(0, str(_ROOT / "src"))

import dist_lib as L
import dist_lib5 as L5

SEED = 0


# --------------------------------------------------------------------------
# Hill estimator
# --------------------------------------------------------------------------


class TestHillEstimator:
    def test_hill_alpha_path_matches_reference_at_spot_checked_k(self):
        rng = np.random.default_rng(SEED)
        x = rng.standard_t(3, 5000) * 0.01
        path = L5.hill_alpha_path(x, tail="upper", k_min=20, k_max=500)
        for k_check in [20, 50, 100, 250, 499]:
            ref = L5.hill_estimator(x, k_check, tail="upper")
            idx = np.where(path["k"] == k_check)[0][0]
            assert path["alpha"][idx] == pytest.approx(ref, rel=1e-9)

    def test_hill_estimator_recovers_known_pareto_tail_index(self):
        # a pure Pareto(alpha) tail is the textbook case the Hill estimator
        # is exact for asymptotically: X = U^{-1/alpha}, U ~ Uniform(0,1).
        rng = np.random.default_rng(SEED)
        true_alpha = 3.0
        u = rng.uniform(1e-6, 1.0, 200_000)
        x = u ** (-1.0 / true_alpha)
        est = L5.hill_estimator(x, k=2000, tail="upper")
        assert est == pytest.approx(true_alpha, rel=0.1)

    def test_hill_estimator_lower_alpha_for_heavier_tail(self):
        rng = np.random.default_rng(SEED)
        heavy = rng.standard_t(2.5, 20_000)
        light = rng.standard_t(10.0, 20_000)
        alpha_heavy = L5.hill_estimator(heavy, k=500, tail="upper")
        alpha_light = L5.hill_estimator(light, k=500, tail="upper")
        assert alpha_heavy < alpha_light

    def test_find_hill_plateau_detects_stable_region(self):
        rng = np.random.default_rng(SEED)
        x = rng.standard_t(3, 50_000)
        path = L5.hill_alpha_path(x, tail="upper", k_min=20, k_max=5000)
        plateau = L5.find_hill_plateau(
            path["alpha"], path["k"], window=50, rel_tol=0.15
        )
        assert plateau["found"]
        assert 2.0 < plateau["alpha_median"] < 5.0

    def test_find_hill_plateau_reports_not_found_honestly(self):
        # too few finite estimates for even one window -> must say so, not
        # silently return a point value.
        alpha = np.array([2.0, 2.1, np.nan, 2.3])
        ks = np.array([20, 21, 22, 23])
        result = L5.find_hill_plateau(alpha, ks, window=50)
        assert result["found"] is False


# --------------------------------------------------------------------------
# GJR-GARCH
# --------------------------------------------------------------------------


class TestGJRGarch:
    def test_gjr_negloglik_reduces_to_garch_at_gamma_zero(self):
        rng = np.random.default_rng(SEED)
        r = rng.normal(0, 0.01, 2000)
        omega, alpha, beta = 1e-6, 0.05, 0.9
        gjr_params = np.array([omega, alpha, 0.0, beta])
        garch_params = np.array([omega, alpha, beta])
        gjr_nll = L5._gjr_negloglik(gjr_params, r, "normal")
        garch_nll = L._garch_negloglik(garch_params, r, "normal")
        assert gjr_nll == pytest.approx(garch_nll, rel=1e-9)

    def test_gjr_variance_path_matches_garch_at_gamma_zero(self):
        rng = np.random.default_rng(SEED)
        r = rng.normal(0, 0.01, 500)
        omega, alpha, beta = 1e-6, 0.05, 0.9
        uncond = omega / (1 - alpha - beta)
        gjr_path = L5._gjr_variance_path(omega, alpha, 0.0, beta, r, uncond)
        garch_path = L._garch_variance_path(omega, alpha, beta, r, uncond)
        assert gjr_path == pytest.approx(garch_path)

    def test_gjr_recovers_leverage_in_asymmetric_synthetic_data(self):
        # simulate a GJR process with a genuine, strong leverage effect and
        # check the fit recovers gamma > 0 and rejects gamma=0 via the LR test.
        rng = np.random.default_rng(SEED)
        n = 3000
        omega, alpha, gamma, beta = 2e-6, 0.03, 0.15, 0.80
        sig2 = np.empty(n)
        sig2[0] = omega / (1 - alpha - gamma / 2 - beta)
        r = np.empty(n)
        r[0] = rng.normal(0, np.sqrt(sig2[0]))
        for t in range(1, n):
            shock = r[t - 1] ** 2
            lev = gamma * shock if r[t - 1] < 0 else 0.0
            sig2[t] = omega + alpha * shock + lev + beta * sig2[t - 1]
            r[t] = rng.normal(0, np.sqrt(sig2[t]))
        fit = L5.fit_gjr11(r, innovation="normal")
        assert fit is not None
        assert fit["gamma"] > 0
        assert fit["lr_gamma0_pvalue"] is not None
        assert fit["lr_gamma0_pvalue"] < 0.05

    def test_rolling_gjr_forecast_is_causal_and_forward_filled(self):
        rng = np.random.default_rng(SEED)
        r = rng.standard_t(5, 4000) * 0.01
        fc, fits = L5.rolling_gjr_forecast(
            r, refit_every=500, min_train=1000, innovation="normal", max_train=1000
        )
        assert len(fits) >= 2
        # forecast must be NaN before the first refit (no lookahead warm-up leak)
        first_refit_t = fits[0]["t"]
        assert np.all(np.isnan(fc[:first_refit_t]))
        assert np.all(np.isfinite(fc[first_refit_t:]))
        # the forecast must actually vary between refits (not silently constant
        # the way a broken forward-fill would look), per NEXT_RUN_PROMPT.md's
        # own tripwire.
        assert np.std(fc[first_refit_t:]) > 0


# --------------------------------------------------------------------------
# GPD / peaks-over-threshold
# --------------------------------------------------------------------------


class TestGPD:
    def test_fit_gpd_tail_recovers_known_shape_upper(self):
        rng = np.random.default_rng(SEED)
        true_xi, true_beta = 0.3, 1.0
        exceedances = st.genpareto.rvs(
            true_xi, scale=true_beta, size=5000, random_state=rng
        )
        # embed the exceedances above a threshold in a wider body so
        # fit_gpd_tail's own thresholding logic is exercised, not bypassed.
        body = rng.normal(0, 1, 20000)
        z = np.concatenate([body, exceedances + 3.0])
        fit = L5.fit_gpd_tail(z, tail_frac=0.10, tail="upper")
        assert fit is not None
        assert fit["xi"] == pytest.approx(true_xi, abs=0.15)

    def test_fit_gpd_tail_lower_is_upper_of_negated_series(self):
        rng = np.random.default_rng(SEED)
        z = rng.standard_t(4, 5000)
        fit_lower = L5.fit_gpd_tail(z, tail_frac=0.10, tail="lower")
        fit_upper_negated = L5.fit_gpd_tail(-z, tail_frac=0.10, tail="upper")
        assert fit_lower["xi"] == pytest.approx(fit_upper_negated["xi"])
        assert fit_lower["beta"] == pytest.approx(fit_upper_negated["beta"])

    def test_fit_gpd_tail_returns_none_below_min_exceedances(self):
        rng = np.random.default_rng(SEED)
        z = rng.normal(0, 1, 100)  # 10% of 100 = 10 exceedances < the 30 floor
        fit = L5.fit_gpd_tail(z, tail_frac=0.10, tail="upper")
        assert fit is None

    def test_gpd_var_es_es_exceeds_var_in_magnitude(self):
        fit = {"xi": 0.3, "beta": 1.0, "u": 1.5, "n_exceed": 100, "n": 1000}
        z_q, es_q = L5.gpd_var_es(fit, q=0.01)
        assert es_q > z_q > 0

    def test_gpd_var_es_undefined_when_xi_ge_one(self):
        fit = {"xi": 1.2, "beta": 1.0, "u": 1.5, "n_exceed": 100, "n": 1000}
        z_q, es_q = L5.gpd_var_es(fit, q=0.01)
        assert np.isfinite(z_q)
        assert np.isnan(es_q)

    def test_rolling_gpd_paths_forward_filled_and_causal(self):
        rng = np.random.default_rng(SEED)
        r = rng.standard_t(4, 4000) * 0.01
        _fc, fits = L.rolling_garch_forecast(
            r,
            refit_every=500,
            min_train=1000,
            innovation="normal",
            max_train=1000,
        )
        paths, _gpd_fits = L5.rolling_gpd_paths(
            r, fits, model="garch", max_train=1000, tail_frac=0.10
        )
        first_t = fits[0]["t"]
        # before any GPD fit exists, paths must be NaN (no lookahead)
        assert np.all(np.isnan(paths["upper"]["xi"][:first_t]))
        # xi must be plausible (not wildly negative - a bounded tail is
        # implausible for returns, per NEXT_RUN_PROMPT.md's own tripwire)
        valid_xi = paths["upper"]["xi"][np.isfinite(paths["upper"]["xi"])]
        assert len(valid_xi) > 0
        assert np.all(valid_xi > -0.5)
