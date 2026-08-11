"""Unit tests for `src/risk/model.py`: `RiskModel`, `ewma_vol`,
`fit_risk_model`, and the numerical PIT/quantile helpers.

Split from `tests/test_commod_lib8.py` (NEXT_PROMPT.md sec 3.6) when the
underlying functions were promoted to `src/risk/model.py`. New tests added
per sec 3.6's list: scale-conditioning edge cases (`sigma_t == self.std`,
`sigma_t == 2*self.std`, `self.std == 0`), `ewma_vol` causality, and
`fit_risk_model`'s three guard conditions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from risk.model import (
    RiskModel,
    ewma_vol,
    fit_risk_model,
    numerical_cdf_grid,
    numerical_pit,
    numerical_ppf,
)

SEED = 0


# --------------------------------------------------------------------------
# Numerical PIT/CDF (shape-only density families have no closed-form cdf)
# --------------------------------------------------------------------------


class TestNumericalPit:
    def test_recovers_standard_normal_cdf(self):
        from scipy import stats as sp_stats

        logpdf_fn = sp_stats.norm.logpdf
        pit = numerical_pit(logpdf_fn, np.array([-1.0, 0.0, 1.0, 2.0]))
        expected = sp_stats.norm.cdf(np.array([-1.0, 0.0, 1.0, 2.0]))
        np.testing.assert_allclose(pit, expected, atol=1e-3)

    def test_pit_is_uniform_for_true_model(self):
        from scipy import stats as sp_stats

        rng = np.random.default_rng(SEED)
        z = rng.standard_normal(2000)
        pit = numerical_pit(sp_stats.norm.logpdf, z)
        _stat, p = sp_stats.kstest(pit, "uniform")
        assert p > 0.01

    def test_cdf_grid_is_monotonic_and_bounded(self):
        from scipy import stats as sp_stats

        g = numerical_cdf_grid(sp_stats.norm.logpdf)
        assert np.all(np.diff(g["cdf"]) >= -1e-12)
        assert g["cdf"][0] >= 0
        assert g["cdf"][-1] <= 1.0 + 1e-9

    def test_numerical_ppf_recovers_standard_normal_quantiles(self):
        from scipy import stats as sp_stats

        u = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
        z = numerical_ppf(sp_stats.norm.logpdf, u)
        expected = sp_stats.norm.ppf(u)
        np.testing.assert_allclose(z, expected, atol=0.02)

    def test_numerical_ppf_is_fast_for_many_points(self):
        import time

        from scipy import stats as sp_stats

        rng = np.random.default_rng(SEED)
        u = rng.uniform(1e-4, 1 - 1e-4, 20000)
        t0 = time.time()
        numerical_ppf(sp_stats.norm.logpdf, u)
        assert time.time() - t0 < 2.0


# --------------------------------------------------------------------------
# RiskModel
# --------------------------------------------------------------------------


class TestRiskModel:
    def test_fit_normal_recovers_var_close_to_analytic(self):
        rng = np.random.default_rng(SEED)
        r = rng.standard_normal(5000) * 0.02
        model = fit_risk_model(r, "TEST", "normal")
        assert model is not None
        from scipy import stats as sp_stats

        expected_var = -0.02 * sp_stats.norm.ppf(0.01)
        assert model.var(0.01) == pytest.approx(expected_var, rel=0.1)

    def test_es_exceeds_var_in_magnitude(self):
        rng = np.random.default_rng(SEED)
        r = rng.standard_t(5, 3000) * 0.02
        model = fit_risk_model(r, "TEST", "t")
        assert model is not None
        assert model.es(0.01) > model.var(0.01)

    def test_var_scales_with_sqrt_horizon(self):
        rng = np.random.default_rng(SEED)
        r = rng.standard_normal(3000) * 0.02
        model = fit_risk_model(r, "TEST", "normal")
        assert model.var(0.01, horizon=4) == pytest.approx(
            model.var(0.01, horizon=1) * 2, rel=1e-6
        )

    def test_simulate_matches_fitted_moments(self):
        rng = np.random.default_rng(SEED)
        r = rng.standard_normal(4000) * 0.03 + 0.001
        model = fit_risk_model(r, "TEST", "normal")
        sim = model.simulate(20000, seed=1)
        assert sim.mean() == pytest.approx(0.001, abs=0.01)
        assert sim.std() == pytest.approx(0.03, rel=0.1)

    def test_shape_only_family_fits_and_scores(self):
        rng = np.random.default_rng(SEED)
        r = rng.standard_t(6, 4000) * 0.02
        model = fit_risk_model(r, "TEST", "ged")
        assert model is not None
        assert model.kind == "standardized"
        assert np.isfinite(model.var(0.01))

    def test_stress_replays_scenario(self):
        rng = np.random.default_rng(SEED)
        r = rng.standard_normal(3000) * 0.02
        model = fit_risk_model(r, "TEST", "normal")
        scenario = np.array([-0.05, -0.03, 0.01])
        result = model.stress(scenario)
        assert result["n_days"] == 3
        assert result["worst_day"] == pytest.approx(-0.05)

    def test_conditional_var_scales_with_supplied_sigma(self):
        rng = np.random.default_rng(SEED)
        r = rng.standard_normal(4000) * 0.02
        model = fit_risk_model(r, "TEST", "normal")
        low_vol_var = model.var_conditional(0.01, sigma_t=0.01)
        high_vol_var = model.var_conditional(0.01, sigma_t=0.04)
        assert high_vol_var > low_vol_var
        assert high_vol_var == pytest.approx(4 * low_vol_var, rel=0.05)

    def test_conditional_var_at_own_std_matches_unconditional(self):
        rng = np.random.default_rng(SEED)
        r = rng.standard_normal(4000) * 0.02
        model = fit_risk_model(r, "TEST", "normal")
        assert model is not None
        assert model.var_conditional(0.01, sigma_t=model.std) == pytest.approx(
            model.var(0.01), rel=1e-6
        )

    def test_scale_conditioning_at_own_std_is_exact(self):
        # NEXT_PROMPT.md sec 3.6: sigma_t == self.std must return exactly
        # _lower_q (not merely "close").
        model = RiskModel(
            "TEST", "normal", "loc_scale", mean=0.0, std=0.02, params=(0.0, 0.02)
        )
        assert model._lower_q_at_scale(0.01, model.std) == pytest.approx(
            model._lower_q(0.01), rel=1e-12
        )

    def test_scale_conditioning_doubles_distance_from_mean_at_double_std(self):
        model = RiskModel(
            "TEST", "normal", "loc_scale", mean=0.001, std=0.02, params=(0.001, 0.02)
        )
        base = model._lower_q(0.01)
        scaled = model._lower_q_at_scale(0.01, 2 * model.std)
        assert (scaled - model.mean) == pytest.approx(
            2 * (base - model.mean), rel=1e-12
        )

    def test_scale_conditioning_falls_back_when_std_is_zero(self):
        # self.std == 0 must fall back to the unconditional quantile without
        # dividing by zero -- an untested branch in the commod_lib8.py source.
        model = RiskModel(
            "TEST", "normal", "loc_scale", mean=0.0, std=0.0, params=(0.0, 1e-9)
        )
        base = model._lower_q(0.01)
        assert model._lower_q_at_scale(0.01, sigma_t=0.05) == pytest.approx(
            base, rel=1e-9
        )
        base_es = model._lower_es(0.01)
        assert model._lower_es_at_scale(0.01, sigma_t=0.05) == pytest.approx(
            base_es, rel=1e-9
        )


class TestFitRiskModelGuards:
    def test_fewer_than_100_observations_returns_none(self):
        rng = np.random.default_rng(SEED)
        r = rng.standard_normal(50) * 0.02
        assert fit_risk_model(r, "TEST", "normal") is None

    def test_spliced_evt_returns_none(self):
        rng = np.random.default_rng(SEED)
        r = rng.standard_normal(500) * 0.02
        assert fit_risk_model(r, "TEST", "spliced_evt") is None

    def test_unknown_family_raises(self):
        rng = np.random.default_rng(SEED)
        r = rng.standard_normal(500) * 0.02
        with pytest.raises(KeyError):
            fit_risk_model(r, "TEST", "not_a_real_family")


# --------------------------------------------------------------------------
# ewma_vol
# --------------------------------------------------------------------------


class TestEwmaVol:
    def test_seeds_from_initial_window_variance(self):
        rng = np.random.default_rng(SEED)
        r = rng.standard_normal(500) * 0.02
        vol = ewma_vol(r, lam=0.94, seed_window=20)
        assert vol[20] == pytest.approx(np.std(r[:20]), rel=1e-6)
        assert np.all(np.isnan(vol[:20]))

    def test_reacts_to_a_volatility_regime_shift(self):
        rng = np.random.default_rng(SEED)
        calm = rng.standard_normal(300) * 0.005
        stormy = rng.standard_normal(300) * 0.05
        r = np.concatenate([calm, stormy])
        vol = ewma_vol(r, lam=0.94, seed_window=20)
        # vol should be materially higher deep into the stormy regime than
        # deep into the calm regime
        assert np.nanmean(vol[550:600]) > np.nanmean(vol[100:150]) * 2

    def test_never_uses_current_bar(self):
        # a single huge outlier at t should NOT move vol[t] itself, only vol[t+1:]
        rng = np.random.default_rng(SEED)
        r = rng.standard_normal(200) * 0.01
        r[100] = 5.0
        vol = ewma_vol(r, lam=0.94, seed_window=20)
        assert vol[100] < 1.0  # unaffected by its own bar's shock
        assert vol[101] > vol[99]  # reacts the bar after

    def test_causality_mutating_a_future_bar_does_not_change_past_sigma(self):
        # NEXT_PROMPT.md sec 3.6: mutate r[t] and assert sigma[t] is
        # unchanged while sigma[t+1] changes -- the property the engine's
        # no-lookahead claim rests on.
        rng = np.random.default_rng(SEED)
        r = rng.standard_normal(300) * 0.02
        sigma_before = ewma_vol(r, lam=0.94, seed_window=20)
        r2 = r.copy()
        r2[150] = 3.0
        sigma_after = ewma_vol(r2, lam=0.94, seed_window=20)
        np.testing.assert_allclose(
            sigma_before[:151], sigma_after[:151], equal_nan=True
        )
        assert sigma_before[151] != pytest.approx(sigma_after[151])
