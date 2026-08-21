"""Notebook 019 Phase 5 (NEXT_PROMPT.md sec 4 row 5, sec 3.6): adoption.

Resolves DS-5 (by executing its three unit tests directly), DS-6 (Phase 2's
first clause plus Phase 4's rho=0 shortfall clause), DS-7 (predicted vs
measured agreement on C2 union C3), and DS-8 (zero gate-clause violations
on C1, all of which sits inside N>=12 and T>=3840 by construction), then
applies sec 3.6's adoption rules for Objective A and Objective B.

Objective A's failure is additionally PROVEN, not just measured (sec 0.2):
for any tau<0.9, dsr_variant(variant="v3") is bit-for-bit dsr_variant(
variant="v1") whenever mean_pairwise_corr > tau, which Phase 2 measures
happens with probability ~1 at rho>=0.9 -- so DS-3's high-rho clause fails
for V3 exactly because it fails for V1, independent of whether DS-7's
prediction mechanism is trustworthy. This is stated plainly rather than
resting Objective A's verdict solely on the Phase 3 mixture prediction.

If Objective B is adopted, patches research.py per sec 8.1 (a NEW sibling
function, deflated_sharpe_prob_switched -- deflated_sharpe_prob itself is
untouched) and re-stamps the hash certificate (sec 8.1's own instruction);
this script does NOT perform the patch itself (source-code edits are done
by hand, mirroring 017's Phase 4, sec 8's compatibility contract) -- it
only writes the adoption decision that governs whether the patch happens.

Usage: uv run python src/research/tmp/run_phase_5_19_adoption.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))

PROFILE_PATH = "src/research/tmp/phase_2_19_switch_profile.json"
PREDICTION_PATH = "src/research/tmp/phase_3_19_prediction.json"
CONFIRMATION_PATH = "src/research/tmp/phase_4_19_confirmation.json"
OUT_PATH = "src/research/tmp/phase_5_19_adoption.json"

TAUS_IN_ORDER = (0.15, 0.30)

# dsr_lib17.VARIANT_SPECS hardcodes the literal strings "v3_tau0.15" and
# "v3_tau0.30" as dict keys in mc_cell's rate/mc_se output. f"v3_tau{tau}"
# for tau=0.30 (a Python float) formats as "v3_tau0.3" -- Python drops the
# trailing zero -- which would silently mismatch every real Phase 4 cell's
# stored key. This map is the single source of truth for that key string
# (matches run_phase_3_19_prediction.py's own TAU_VARIANT_KEY).
TAU_VARIANT_KEY = {0.15: "v3_tau0.15", 0.30: "v3_tau0.30"}

_spec = importlib.util.spec_from_file_location(
    "run_phase_4_17_adoption", "src/research/tmp/run_phase_4_17_adoption.py"
)
assert _spec and _spec.loader
p4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p4)  # type: ignore[union-attr]


def run_ds5() -> dict:
    import test_dsr_lib17 as T

    tests = [
        "test_v3_reduces_to_v0_below_threshold",
        "test_v3_reduces_to_v1_above_threshold",
        "test_v3_boundary_resolves_to_v0",
    ]
    results: dict[str, dict[str, bool | str]] = {}
    for name in tests:
        try:
            getattr(T, name)()
            results[name] = {"passed": True}
        except Exception as e:  # noqa: BLE001
            results[name] = {"passed": False, "error": f"{e}\n{traceback.format_exc()}"}
    fires = all(r["passed"] for r in results.values())
    return {"fires": fires, "tests": results}


def compute_ds6(profile: dict, confirmation_cells: list[dict], tau: float) -> dict:
    first_clause = profile["ds6_first_clause"]["fires"]
    tau_key = TAU_VARIANT_KEY[tau]

    rho0_edge_cells = [
        c for c in confirmation_cells if c["rho"] == 0.0 and c["mode"] == "edge"
    ]
    shortfalls = []
    for c in rho0_edge_cells:
        diff = c["rate"][tau_key] - c["rate"]["v0"]
        if diff < 0:
            shortfalls.append(
                {
                    "n_trials": c["n_trials"],
                    "n_obs": c["n_obs"],
                    "moments_label": c["moments_label"],
                    "subset": c["subset"],
                    "diff": diff,
                }
            )
    second_clause = len(shortfalls) == 0
    return {
        "fires": bool(first_clause) and second_clause,
        "first_clause_fires": first_clause,
        "second_clause_fires": second_clause,
        "second_clause_n_rho0_edge_cells_checked": len(rho0_edge_cells),
        "second_clause_shortfalls": shortfalls,
    }


def compute_ds7(
    predicted_cells: list[dict], confirmation_cells: list[dict], tau: float
) -> dict:
    tau_key = TAU_VARIANT_KEY[tau]

    def key(c: dict) -> tuple:
        return (c["n_trials"], c["n_obs"], c["rho"], c["moments_label"], c["mode"])

    predicted_by_key = {key(c): c for c in predicted_cells}

    comparisons = []
    for c in confirmation_cells:
        if c["subset"] not in ("C2", "C3"):
            continue
        pred = predicted_by_key.get(key(c))
        if pred is None:
            continue  # not predictable (should not happen for C2/C3)
        predicted_rate = pred["rate"][tau_key]
        predicted_se = pred["mc_se"][tau_key]
        measured_rate = c["rate"][tau_key]
        measured_se = c["mc_se"][tau_key]
        combined_se = (predicted_se**2 + measured_se**2) ** 0.5
        diff = abs(predicted_rate - measured_rate)
        se_multiple = (
            diff / combined_se
            if combined_se > 0
            else (0.0 if diff == 0 else float("inf"))
        )
        comparisons.append(
            {
                **{
                    k: c[k]
                    for k in (
                        "n_trials",
                        "n_obs",
                        "rho",
                        "moments_label",
                        "mode",
                        "subset",
                    )
                },
                "predicted_rate": predicted_rate,
                "measured_rate": measured_rate,
                "combined_se": combined_se,
                "diff": diff,
                "se_multiple": se_multiple,
            }
        )

    n = len(comparisons)
    within_3se = sum(1 for c in comparisons if c["se_multiple"] <= 3.0)
    frac_within_3se = within_3se / n if n else 0.0
    max_se_multiple = max((c["se_multiple"] for c in comparisons), default=0.0)
    fires = frac_within_3se >= 0.95 and max_se_multiple <= 5.0

    return {
        "fires": fires,
        "n_comparisons": n,
        "frac_within_3_combined_se": frac_within_3se,
        "max_se_multiple": max_se_multiple,
        "worst_comparisons": sorted(comparisons, key=lambda c: -c["se_multiple"])[:10],
    }


def compute_ds8(confirmation_cells: list[dict], tau: float) -> dict:
    tau_key = TAU_VARIANT_KEY[tau]
    c1_cells = [c for c in confirmation_cells if c["subset"] == "C1"]
    in_regime = [c for c in c1_cells if c["n_trials"] >= 12 and c["n_obs"] >= 3840]
    assert len(in_regime) == len(c1_cells), (
        "C1 was supposed to sit entirely inside N>=12 and T>=3840 by "
        "construction -- build_phase4_grid_19.py's C1 axes are wrong if not"
    )
    ev = p4.evaluate_variant(in_regime, tau_key)
    fires = ev["passes_both_DS2_and_DS3"]
    return {
        "fires": fires,
        "n_c1_cells_in_regime": len(in_regime),
        "gate_evaluation": ev,
    }


def objective_a_verdict(prediction: dict, ds5: dict, ds6_by_tau: dict) -> dict:
    out = {}
    for tau in TAUS_IN_ORDER:
        predicted_ev = prediction["by_tau"][str(tau)]["objective_A_gate_evaluation"]
        passes_ds2_ds3 = predicted_ev["passes_both_DS2_and_DS3"]
        fires = ds5["fires"] and ds6_by_tau[str(tau)]["fires"] and passes_ds2_ds3
        out[str(tau)] = {
            "fires": fires,
            "DS2_DS3_predicted_pass": passes_ds2_ds3,
            "DS5_fires": ds5["fires"],
            "DS6_fires": ds6_by_tau[str(tau)]["fires"],
            "note": "DS-3's high-rho clause failure is analytically proven "
            "(sec 0.2), not merely predicted: V3 is bit-for-bit V1 at "
            "rho>=0.9 for any tau<0.9, and V1 already fails that clause. "
            "This holds regardless of DS-7.",
        }
    return out


def choose_tau_for_objective_b(ds6_by_tau: dict) -> float:
    if ds6_by_tau[str(TAUS_IN_ORDER[0])]["fires"]:
        return TAUS_IN_ORDER[0]
    return TAUS_IN_ORDER[1]


def main() -> None:
    with open(PROFILE_PATH) as f:
        profile = json.load(f)
    with open(PREDICTION_PATH) as f:
        prediction = json.load(f)
    with open(CONFIRMATION_PATH) as f:
        confirmation = json.load(f)
    confirmation_cells = confirmation["cells"]

    ds5 = run_ds5()

    ds6_by_tau = {
        str(tau): compute_ds6(profile, confirmation_cells, tau) for tau in TAUS_IN_ORDER
    }
    ds7_by_tau = {
        str(tau): compute_ds7(
            prediction["by_tau"][str(tau)]["predicted_cells"], confirmation_cells, tau
        )
        for tau in TAUS_IN_ORDER
    }
    ds8_by_tau = {
        str(tau): compute_ds8(confirmation_cells, tau) for tau in TAUS_IN_ORDER
    }

    obj_a = objective_a_verdict(prediction, ds5, ds6_by_tau)
    obj_a_adopted = next((t for t in TAUS_IN_ORDER if obj_a[str(t)]["fires"]), None)

    tau_b = choose_tau_for_objective_b(ds6_by_tau)
    obj_b_fires = (
        ds5["fires"]
        and ds6_by_tau[str(tau_b)]["fires"]
        and ds7_by_tau[str(tau_b)]["fires"]
        and ds8_by_tau[str(tau_b)]["fires"]
    )

    if obj_a_adopted is not None:
        final = f"Objective A adopted at tau={obj_a_adopted}: V3 adopted at full scope."
    elif obj_b_fires:
        final = f"Objective B adopted at tau={tau_b}: V3 adopted restricted to N>=12 and T>=3840."
    else:
        final = "Neither objective adopted. research.py stays exactly as 017 left it."

    doc = {
        "notebook": "019_dsr_correlation_switch",
        "phase": 5,
        "DS5": ds5,
        "DS6_by_tau": ds6_by_tau,
        "DS7_by_tau": ds7_by_tau,
        "DS8_by_tau": ds8_by_tau,
        "objective_A": {
            "by_tau": obj_a,
            "adopted_at_tau": obj_a_adopted,
        },
        "objective_B": {
            "tau_used": tau_b,
            "tau_selection_rule": "tau=0.15 unless it fails DS-6, then "
            "tau=0.30 (sec 1.2/sec 3.6)",
            "fires": obj_b_fires,
        },
        "final_verdict": final,
        "research_py_patch_required": obj_b_fires and obj_a_adopted is None,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"written {OUT_PATH}")
    print(final)
    print(
        "DS-5:", ds5["fires"], "DS-6:", {k: v["fires"] for k, v in ds6_by_tau.items()}
    )
    print("DS-7:", {k: v["fires"] for k, v in ds7_by_tau.items()})
    print("DS-8:", {k: v["fires"] for k, v in ds8_by_tau.items()})


if __name__ == "__main__":
    main()
