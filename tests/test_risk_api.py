"""Tests for `src/risk/__init__.py`'s public API (NEXT_PROMPT.md sec 8.1).

Each entry point is a thin wrapper; these tests exercise the wrapping
(argument plumbing, the contract check at `fit()`, `size()`'s guard) rather
than re-testing the underlying machinery covered elsewhere
(`tests/test_risk_model.py`, `tests/test_risk_portfolio.py`, etc).
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import risk
from risk import hygiene
from risk.families import UnseenProductError

SEED = 0


def _clean_returns_frame(n: int = 500, seed: int = SEED) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    ret = rng.standard_t(6, n) * 0.02
    curve = pl.DataFrame({"date": dates, "log_return": ret})
    setattr(curve, hygiene.PROVENANCE_ATTR, hygiene.PROVENANCE_VALUE)
    return curve


class TestFit:
    def test_fit_returns_a_model_for_a_known_product(self):
        model = risk.fit("CL", _clean_returns_frame())
        assert isinstance(model, risk.RiskModel)
        assert model.product == "CL"

    def test_fit_rejects_a_frame_without_provenance(self):
        curve = _clean_returns_frame()
        curve = curve.clone()  # drops the stamped attribute
        with pytest.raises(hygiene.RiskInputError):
            risk.fit("CL", curve)

    def test_fit_refuses_an_unseen_product(self):
        with pytest.raises(UnseenProductError):
            risk.fit("NOT_A_REAL_PRODUCT", _clean_returns_frame())

    def test_fit_refuses_a_frame_that_leaks_into_the_spent_holdout(self):
        # NEXT_PROMPT.md sec 2 ground rule 1 / sec 12: the futures holdout
        # (>= 2025-01-01) was already spent once by 008 Phase 8 and the
        # fitting path must refuse to touch it again, even though the same
        # frame shape is fine for assert_risk_inputs alone (sec 7.4 permits
        # ingest/serve to see current dates).
        n = 500
        dates = [date(2025, 1, 1) - timedelta(days=n - 1 - i) for i in range(n)]
        rng = np.random.default_rng(SEED)
        ret = rng.standard_t(6, n) * 0.02
        curve = pl.DataFrame({"date": dates, "log_return": ret})
        setattr(curve, hygiene.PROVENANCE_ATTR, hygiene.PROVENANCE_VALUE)
        with pytest.raises(hygiene.HoldoutLeakError, match="holdout"):
            risk.fit("CL", curve)


class TestVarEsSize:
    def _model(self):
        rng = np.random.default_rng(SEED)
        r = rng.standard_normal(3000) * 0.02
        from risk.model import fit_risk_model

        return fit_risk_model(r, "TEST", "normal")

    def test_var_and_es_take_an_explicit_sigma_t(self):
        model = self._model()
        v = risk.var(model, 0.01, sigma_t=0.02)
        e = risk.es(model, 0.01, sigma_t=0.02)
        assert e >= v > 0

    def test_size_is_risk_budget_over_var(self):
        model = self._model()
        v = risk.var(model, 0.01, sigma_t=0.02)
        notional = risk.size(model, 0.01, sigma_t=0.02, risk_budget=1000.0)
        assert notional == pytest.approx(1000.0 / v)

    def test_size_raises_on_nonpositive_var(self):
        # A hand-constructed degenerate model with a positive mean and
        # zero std: at sigma_t=0 the conditional quantile collapses to the
        # unconditional mean (RiskModel._lower_q_at_scale's own std==0
        # fallback), so VaR = -mean < 0 -- the guard size() exists for.
        model = risk.RiskModel(
            "TEST", "normal", "loc_scale", mean=0.01, std=0.0, params=(0.01, 1e-9)
        )
        with pytest.raises(ValueError, match="positive"):
            risk.size(model, 0.01, sigma_t=0.0, risk_budget=1000.0)


class TestStress:
    def test_stress_replays_a_return_path(self):
        rng = np.random.default_rng(SEED)
        r = rng.standard_normal(3000) * 0.02
        from risk.model import fit_risk_model

        model = fit_risk_model(r, "TEST", "normal")
        result = risk.stress(model, np.array([-0.05, -0.03, 0.01]))
        assert result["n_days"] == 3
        assert result["worst_day"] == pytest.approx(-0.05)


class TestPortfolio:
    def test_portfolio_defaults_to_empirical_never_gaussian(self):
        import inspect

        sig = inspect.signature(risk.portfolio)
        assert sig.parameters["dependence"].default == "empirical"

    def test_portfolio_runs_end_to_end(self):
        rng = np.random.default_rng(SEED)
        n = 2000
        r1 = rng.standard_normal(n) * 0.02
        r2 = rng.standard_normal(n) * 0.02
        from risk.model import fit_risk_model

        models = {
            "A": fit_risk_model(r1, "A", "normal"),
            "B": fit_risk_model(r2, "B", "normal"),
        }
        result = risk.portfolio(
            models,
            {"A": 0.5, "B": 0.5},
            historical_returns={"A": r1, "B": r2},
            n_sims=5000,
        )
        assert result["var_01"] > 0


class TestMonitor:
    def test_monitor_returns_a_calibration_status(self):
        rng = np.random.default_rng(SEED)
        r = rng.standard_normal(3000) * 0.02
        from risk.model import ewma_vol, fit_risk_model

        model = fit_risk_model(r, "TEST", "normal")
        sigma = ewma_vol(r)
        status = risk.monitor("TEST", model, r, sigma, compute_acerbi=False)
        assert status.status in ("ok", "warn", "breach")


class TestPublicApiSurface:
    def test_all_sec_8_1_entry_points_are_exported(self):
        # NEXT_PROMPT.md sec 8.1's table, minus refresh/snapshot which are
        # exercised in test_risk_ingest.py/test_risk_serve.py against real
        # or synthetic data rather than here (they touch the filesystem).
        for name in ("fit", "var", "es", "portfolio", "stress", "monitor", "size"):
            assert hasattr(risk, name), name
        assert hasattr(risk, "refresh")
        assert hasattr(risk, "snapshot")
