"""Notebook 019 Phase 3 (NEXT_PROMPT.md sec 4 row 3, sec 3.4/DS-7): the
pre-registered prediction, written BEFORE Phase 4's confirmation Monte
Carlo runs (sec 4's ordering rule -- Phase 4 must not start before this
file is on disk).

Mixture-predicts every cell in 017's own 756-cell certificate that Phase 2
profiled: predicted_rate = p*rate_v1 + (1-p)*rate_v0, where p is Phase 2's
measured P(mean_pairwise_corr_estimate >= tau) at that (N, T, rho, moments,
mode) design point and rate_v0/rate_v1 are 017's own stored, already-
measured rates for that exact cell -- no new Monte Carlo. Runs 017's own
gate code (run_phase_4_17_adoption.evaluate_variant, imported, not
reimplemented -- sec 3.1) on the predicted cells to get Objective-A's
predicted verdict, plus the in-sample restricted-regime scan.

Only cells inside 017's original 756-cell grid can be predicted this way
(C2 and C3 in Phase 4's confirmation grid); C1's axis values (T=5000,
rho=0.95) do not exist in 017's certificate and carry no prediction --
they are scored by DS-8 directly, not DS-7.

Usage: uv run python src/research/tmp/run_phase_3_19_prediction.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

CALIBRATION_PATH = "src/research/tmp/phase_3_17_calibration.json"
PROFILE_PATH = "src/research/tmp/phase_2_19_switch_profile.json"
OUT_PATH = "src/research/tmp/phase_3_19_prediction.json"

TAUS = (0.15, 0.30)

# dsr_lib17.VARIANT_SPECS hardcodes the literal strings "v3_tau0.15" and
# "v3_tau0.30" as dict keys in mc_cell's rate/mc_se output. f"v3_tau{tau}"
# for tau=0.30 (a Python float) formats as "v3_tau0.3" -- Python drops the
# trailing zero -- which would silently mismatch every real Phase 4 cell's
# stored key. This map is the single source of truth for that key string.
TAU_VARIANT_KEY = {0.15: "v3_tau0.15", 0.30: "v3_tau0.30"}

_spec = importlib.util.spec_from_file_location(
    "run_phase_4_17_adoption", "src/research/tmp/run_phase_4_17_adoption.py"
)
assert _spec and _spec.loader
p4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p4)  # type: ignore[union-attr]


def _profile_key(row: dict) -> tuple:
    return (
        row["n_trials"],
        row["n_obs"],
        row["rho"],
        row["moments_label"],
        row["mode"],
    )


def build_predicted_cells(
    cert_cells: list[dict], profile_cells: list[dict], tau: float
) -> list[dict]:
    profile_by_key = {_profile_key(r): r for r in profile_cells}
    tau_key = str(tau)
    predicted = []
    for c in cert_cells:
        key = _profile_key(c)
        prow = profile_by_key.get(key)
        if prow is None:
            continue  # not profiled by Phase 2 -- should not happen, all 756 are
        p = prow["p_select_v1_by_tau"][tau_key]
        se_p = prow["mc_se_by_tau"][tau_key]
        rate_v0 = c["rate"]["v0"]
        rate_v1 = c["rate"]["v1"]
        se_v0 = c["mc_se"]["v0"]
        se_v1 = c["mc_se"]["v1"]
        predicted_rate = p * rate_v1 + (1 - p) * rate_v0

        # Error propagation on predicted = p*rate_v1 + (1-p)*rate_v0,
        # treating p, rate_v0, rate_v1 as independent estimates (they are
        # measured on independent Monte Carlo draws -- Phase 2's profile
        # uses its own seed namespace, distinct from 017's cell seeds).
        var_predicted = (
            p**2 * se_v1**2
            + (1 - p) ** 2 * se_v0**2
            + (rate_v1 - rate_v0) ** 2 * se_p**2
        )
        se_predicted = float(var_predicted**0.5)

        c2 = dict(c)
        c2["rate"] = dict(c["rate"])
        c2["rate"][TAU_VARIANT_KEY[tau]] = predicted_rate
        c2["mc_se"] = dict(c["mc_se"])
        c2["mc_se"][TAU_VARIANT_KEY[tau]] = se_predicted
        c2["_predicted_p_select_v1"] = p
        c2["_predicted_se_p"] = se_p
        predicted.append(c2)
    return predicted


def restricted_regime_scan(cells: list[dict], variant: str) -> list[dict]:
    ns = [4, 8, 12, 18, 36, 95, 122]
    ts = [300, 1000, 3840]
    out = []
    for min_n in ns:
        for min_t in ts:
            sub = [c for c in cells if c["n_trials"] >= min_n and c["n_obs"] >= min_t]
            if not sub:
                continue
            ev = p4.evaluate_variant(sub, variant)
            n_null = sum(1 for c in sub if c["mode"] == "null")
            n_edge = sum(1 for c in sub if c["mode"] == "edge")
            out.append(
                {
                    "min_n_trials": min_n,
                    "min_n_obs": min_t,
                    "n_null": n_null,
                    "n_edge": n_edge,
                    "DS2a_violations": ev["DS2a"]["n_violations"],
                    "DS2b_violations": ev["DS2b"]["n_violations"],
                    "DS3_high_rho_violations": len(ev["DS3"]["high_rho_violations"]),
                    "DS3_rho0_violations": len(ev["DS3"]["rho0_violations"]),
                    "passes": ev["passes_both_DS2_and_DS3"],
                }
            )
    return out


def main() -> None:
    with open(CALIBRATION_PATH) as f:
        cert = json.load(f)
    cert_cells = cert["cells"]

    with open(PROFILE_PATH) as f:
        profile = json.load(f)
    profile_cells = profile["cells"]

    by_tau = {}
    for tau in TAUS:
        variant_name = TAU_VARIANT_KEY[tau]
        predicted_cells = build_predicted_cells(cert_cells, profile_cells, tau)
        ev = p4.evaluate_variant(predicted_cells, variant_name)
        # sec 0.5/sec 3.1: record the UNCAPPED violation counts alongside
        # evaluate_variant's own capped (violations[:20]) lists, without
        # editing 017's script.
        ev_uncapped = {
            "DS2a_n_violations_uncapped": ev["DS2a"]["n_violations"],
            "DS2b_n_violations_uncapped": ev["DS2b"]["n_violations"],
            "DS3_high_rho_n_violations_uncapped": len(ev["DS3"]["high_rho_violations"]),
            "DS3_rho0_n_violations_uncapped": len(ev["DS3"]["rho0_violations"]),
        }
        by_tau[str(tau)] = {
            "variant": variant_name,
            "n_predicted_cells": len(predicted_cells),
            "objective_A_gate_evaluation": ev,
            "uncapped_violation_counts": ev_uncapped,
            "restricted_regime_scan": restricted_regime_scan(
                predicted_cells, variant_name
            ),
            "predicted_cells": predicted_cells,
        }

    objective_a_verdict = {}
    for tau in TAUS:
        ev = by_tau[str(tau)]["objective_A_gate_evaluation"]
        objective_a_verdict[str(tau)] = (
            "PASSES (adopted)" if ev["passes_both_DS2_and_DS3"] else "FAILS"
        )

    doc = {
        "notebook": "019_dsr_correlation_switch",
        "phase": 3,
        "written_before_phase_4_runs": True,
        "prediction_mechanism": "predicted_rate = p*rate_v1 + (1-p)*rate_v0, "
        "p = Phase 2's measured P(mean_pairwise_corr_estimate >= tau) at the "
        "same (N, T, rho, moments, mode) design point, rate_v0/rate_v1 = "
        "017's own stored, already-measured rates for that exact cell. No "
        "new Monte Carlo.",
        "combined_se_formula": "Var(predicted) = p^2*se(v1)^2 + "
        "(1-p)^2*se(v0)^2 + (v1-v0)^2*se(p)^2 -- error propagation "
        "treating p, rate_v0, rate_v1 as independent MC estimates (Phase "
        "2's profile uses a seed namespace disjoint from 017's cell seeds).",
        "by_tau": by_tau,
        "objective_A_predicted_verdict": objective_a_verdict,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(doc, f)
    print(f"written {OUT_PATH}")
    print("Objective A predicted verdict:", objective_a_verdict)
    for tau in TAUS:
        print(
            f"tau={tau} uncapped violations:",
            by_tau[str(tau)]["uncapped_violation_counts"],
        )
        widest = [
            (r["min_n_trials"], r["min_n_obs"])
            for r in by_tau[str(tau)]["restricted_regime_scan"]
            if r["passes"]
        ]
        print(f"tau={tau} predicted widest passing boxes:", widest)


if __name__ == "__main__":
    main()
