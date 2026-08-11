"""Unit tests for `src/risk/calibration.py`: `kupiec_by_state` (split from
`tests/test_commod_lib8.py`, NEXT_PROMPT.md sec 3.6) and `CalibrationMonitor`
(NEXT_PROMPT.md sec 6.3-6.4).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from risk.calibration import (
    CalibrationMonitor,
    _benjamini_hochberg_significant,
    _standardized_ppf,
    _upper_tail_es_z,
    kupiec_by_state,
)
from risk.model import RiskModel, ewma_vol, fit_risk_model

SEED = 0


class TestKupiecByState:
    def test_flags_a_state_with_bad_coverage(self):
        rng = np.random.default_rng(SEED)
        n = 2000
        # state A: correctly calibrated 1% hit rate; state B: hits at 5% (miscalibrated)
        states = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
        hits = np.concatenate([rng.random(n // 2) < 0.01, rng.random(n // 2) < 0.05])
        result = kupiec_by_state(hits, states, expected_rate=0.01)
        assert result["A"]["kupiec_p"] > 0.01
        assert result["B"]["kupiec_p"] < 0.01


class TestBenjaminiHochberg:
    def test_all_null_nothing_significant(self):
        rng = np.random.default_rng(SEED)
        pvalues = {f"p{i}": float(rng.uniform(0.1, 1.0)) for i in range(16)}
        sig = _benjamini_hochberg_significant(pvalues, alpha=0.05)
        assert not any(sig.values())

    def test_one_strong_signal_among_many_nulls_survives(self):
        rng = np.random.default_rng(SEED)
        pvalues = {f"null{i}": float(rng.uniform(0.2, 1.0)) for i in range(15)}
        pvalues["signal"] = 1e-8
        sig = _benjamini_hochberg_significant(pvalues, alpha=0.05)
        assert sig["signal"] is True
        assert sum(sig.values()) == 1

    def test_non_finite_pvalues_never_significant(self):
        sig = _benjamini_hochberg_significant({"a": float("nan"), "b": 0.5}, alpha=0.05)
        assert sig["a"] is False


class TestUpperTailEsZ:
    def test_matches_analytic_standard_normal_value(self):
        # E[Z | Z > z_{1-q}] for a standard normal has the closed form
        # phi(z_{1-q}) / q.
        from scipy import stats as st

        model = RiskModel(
            "TEST", "normal", "loc_scale", mean=0.0, std=1.0, params=(0.0, 1.0)
        )
        q = 0.01
        z_q = st.norm.ppf(1 - q)
        analytic = st.norm.pdf(z_q) / q
        # numerical integration over 200 grid points (matching
        # zoo_es_forecast_upper's coarse-grid convention) -- close but not
        # exact.
        assert _upper_tail_es_z(model, q) == pytest.approx(analytic, rel=5e-3)

    def test_standardized_ppf_recovers_median_at_zero(self):
        model = RiskModel(
            "TEST", "normal", "loc_scale", mean=0.5, std=2.0, params=(0.5, 2.0)
        )
        z = _standardized_ppf(model, np.array([0.5]))
        assert z[0] == pytest.approx(0.0, abs=1e-6)


class TestCalibrationMonitorVerdict:
    """Synthetic hit series engineered to isolate each failure mode
    (NEXT_PROMPT.md sec 6.3's table), using `evaluate_from_hits` so no model
    fit is needed."""

    def _monitor(self) -> CalibrationMonitor:
        return CalibrationMonitor(
            thresholds={
                "p_value_threshold": 0.05,
                "violation_rate_thresholds": {
                    "warn_ratio_observed_over_expected": 1.5,
                    "breach_ratio_observed_over_expected": 2.0,
                },
                "max_cluster_length_thresholds": {"warn": 4, "breach": 6},
            }
        )

    def test_well_calibrated_iid_hits_are_ok(self):
        rng = np.random.default_rng(SEED)
        hits = rng.random(2000) < 0.01
        status = self._monitor().evaluate_from_hits("TEST", {0.01: hits})
        assert status.status == "ok"
        assert status.failure_mode is None

    def test_clustered_hits_at_correct_rate_flag_clustering(self):
        # Kupiec-correct overall rate (~1%), but every violation crammed
        # into a handful of tight clusters -- PA's exact signature
        # (kupiec passes, independence fails).
        n = 2000
        hits = np.zeros(n, dtype=bool)
        rng = np.random.default_rng(SEED)
        cluster_starts = rng.choice(np.arange(0, n - 5, 5), size=4, replace=False)
        for start in cluster_starts:
            hits[start : start + 5] = True
        status = self._monitor().evaluate_from_hits("TEST", {0.01: hits})
        assert status.failure_mode == "clustering"
        assert status.status == "breach"

    def test_excess_iid_violations_flag_coverage(self):
        # seed=1: this particular iid draw at 5x the expected rate happens
        # to pass independence cleanly (p=0.89), isolating a pure coverage
        # failure -- iid draws at this rate occasionally also fail
        # independence by chance (a "both" verdict), which is a separate,
        # already-covered case, not a bug in this seed's choice.
        rng = np.random.default_rng(1)
        hits = rng.random(2000) < 0.05  # 5x the expected 1% rate, iid
        status = self._monitor().evaluate_from_hits("TEST", {0.01: hits})
        assert status.failure_mode == "coverage"
        assert status.status == "breach"

    def test_max_cluster_length_is_computed_correctly(self):
        hits = np.array([False, True, True, True, False, True, False])
        status = self._monitor().evaluate_from_hits("TEST", {0.01: hits})
        assert status.levels[0.01].max_cluster_length == 3


class TestCalibrationMonitorEvaluate:
    def test_evaluate_end_to_end_with_a_fitted_model(self):
        rng = np.random.default_rng(SEED)
        r = rng.standard_normal(3000) * 0.02
        model = fit_risk_model(r, "TEST", "normal")
        assert model is not None
        sigma = ewma_vol(r)
        status = CalibrationMonitor(
            thresholds={
                "p_value_threshold": 0.05,
                "violation_rate_thresholds": {
                    "warn_ratio_observed_over_expected": 1.5,
                    "breach_ratio_observed_over_expected": 2.0,
                },
                "max_cluster_length_thresholds": {"warn": 4, "breach": 6},
            }
        ).evaluate("TEST", model, r, sigma)
        assert 0.01 in status.levels
        assert status.acerbi is not None
        assert np.isfinite(status.acerbi.z_lower)
        assert np.isfinite(status.acerbi.z_upper)


class TestGateMBRegression:
    """Gate MB (NEXT_PROMPT.md sec 6.4, sec 10): the monitor must rediscover
    PA's development clustering failure and RB/SI's holdout failures. This
    is a cheap regression check against the frozen gate output
    (`src/research/tmp/run_risk_04_monitor_results.json`) -- the full
    walk-forward recomputation lives in `run_risk_04_monitor.py` and is not
    re-run on every test invocation."""

    def test_gate_mb_fires(self):
        path = _ROOT / "src" / "research" / "tmp" / "run_risk_04_monitor_results.json"
        if not path.exists():
            pytest.skip("run_risk_04_monitor.py has not been run yet")
        with path.open() as f:
            results = json.load(f)
        assert results["gate_MB"]["fires"] is True
        assert results["gate_MB"]["pa_flagged_clustering_in_development"] is True
        assert results["gate_MB"]["rb_si_flagged_in_holdout"] is True
        assert results["gate_MB"]["no_false_positives_in_development"] is True
