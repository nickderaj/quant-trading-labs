"""Notebook 017 Phase 4 (NEXT_PROMPT.md sec 4, sec 5.2, sec 5.3, sec 2.3):
evaluates DS-2 and DS-3 on Phase 3's collated calibration certificate and
applies the sec 2.3 adoption rule.

This script decides WHICH variant (if any) is adopted and writes that
decision, with the full gate evidence, to phase_4_17_adoption.json. It does
NOT edit research.py itself -- sec 8's backward-compatibility contract
(exact default-path preservation, docstring updates, keyword-only args) is
applied by hand against this script's verdict, the same way every other
function in research.py was added by hand rather than by a codegen script.
Once research.py is patched, re-run this script's sibling collation step
(run_phase_3_17_calibration.py --collate) to re-stamp the sec 10 hash gate
-- the grid itself is never re-run (Phase 4 moves code, not arithmetic;
Test 1 and Test 2 are what prove that).

Usage: uv run python src/research/tmp/run_phase_4_17_adoption.py
"""

from __future__ import annotations

import json

CALIBRATION_PATH = "src/research/tmp/phase_3_17_calibration.json"
OUT_PATH = "src/research/tmp/phase_4_17_adoption.json"

DS2A_FPR_CEILING = 0.075
DS2B_FPR_FLOOR = 0.010
DS2B_RHO_FLOOR = 0.5
DS3_POWER_MARGIN_HIGH_RHO = 0.10
DS3_POWER_MARGIN_RHO0 = 0.02
DS3_HIGH_RHO_FLOOR = 0.9

# sec 2.3's total order: V1 first (no tuning constant, matches the source
# paper), then V1b (c=0.25 preferred over c=0.5 -- closer to unshrunk V1),
# then V2 last (needs the full correlation structure, least re-scorable).
ADOPTION_ORDER = ["v1", "v1b_c0.25", "v1b_c0.5", "v2"]


def evaluate_variant(cells: list[dict], variant: str) -> dict:
    null_cells = [c for c in cells if c["mode"] == "null"]
    edge_cells = [c for c in cells if c["mode"] == "edge"]

    # --- DS-2a: FPR <= 0.075 in EVERY null cell ---
    ds2a_violations = [
        {
            "n_trials": c["n_trials"],
            "n_obs": c["n_obs"],
            "rho": c["rho"],
            "moments_label": c["moments_label"],
            "fpr": c["rate"][variant],
        }
        for c in null_cells
        if c["rate"][variant] > DS2A_FPR_CEILING
    ]
    ds2a_fires = len(ds2a_violations) == 0

    # --- DS-2b: FPR >= 0.010 in every null cell with rho >= 0.5 ---
    ds2b_violations = [
        {
            "n_trials": c["n_trials"],
            "n_obs": c["n_obs"],
            "rho": c["rho"],
            "moments_label": c["moments_label"],
            "fpr": c["rate"][variant],
        }
        for c in null_cells
        if c["rho"] >= DS2B_RHO_FLOOR and c["rate"][variant] < DS2B_FPR_FLOOR
    ]
    ds2b_fires = len(ds2b_violations) == 0

    # --- DS-3: power. Each edge cell already carries both V0's and the
    # candidate variant's detection rate (rate[...]), computed on the SAME
    # replications -- no cross-cell matching needed. ---
    ds3_high_rho_violations = []
    ds3_rho0_violations = []
    for c in edge_cells:
        key = (c["n_trials"], c["n_obs"], c["rho"], c["moments_label"])
        v0_edge_rate = c["rate"]["v0"]
        variant_rate = c["rate"][variant]
        diff = variant_rate - v0_edge_rate
        if c["rho"] >= DS3_HIGH_RHO_FLOOR and diff < DS3_POWER_MARGIN_HIGH_RHO:
            ds3_high_rho_violations.append(
                {
                    **dict(zip(["n_trials", "n_obs", "rho", "moments_label"], key)),
                    "diff": diff,
                }
            )
        if c["rho"] == 0.0 and diff < -DS3_POWER_MARGIN_RHO0:
            ds3_rho0_violations.append(
                {
                    **dict(zip(["n_trials", "n_obs", "rho", "moments_label"], key)),
                    "diff": diff,
                }
            )
    ds3_fires = len(ds3_high_rho_violations) == 0 and len(ds3_rho0_violations) == 0

    return {
        "variant": variant,
        "DS2a": {
            "fires": ds2a_fires,
            "n_violations": len(ds2a_violations),
            "violations": ds2a_violations[:20],
        },
        "DS2b": {
            "fires": ds2b_fires,
            "n_violations": len(ds2b_violations),
            "violations": ds2b_violations[:20],
        },
        "DS2": ds2a_fires and ds2b_fires,
        "DS3": {
            "fires": ds3_fires,
            "high_rho_violations": ds3_high_rho_violations[:20],
            "rho0_violations": ds3_rho0_violations[:20],
        },
        "passes_both_DS2_and_DS3": ds2a_fires and ds2b_fires and ds3_fires,
    }


def main() -> None:
    with open(CALIBRATION_PATH) as f:
        cert = json.load(f)
    cells = cert["cells"]

    evaluations = {v: evaluate_variant(cells, v) for v in ADOPTION_ORDER}

    adopted = None
    for v in ADOPTION_ORDER:
        if evaluations[v]["passes_both_DS2_and_DS3"]:
            adopted = v
            break

    doc = {
        "notebook": "017_deflated_sharpe_correction",
        "phase": 4,
        "n_cells_in_certificate": len(cells),
        "adoption_order": ADOPTION_ORDER,
        "evaluations": evaluations,
        "adopted_variant": adopted,
        "verdict": (
            f"Adopted: {adopted}."
            if adopted
            else "None of V1, V1b(0.25), V1b(0.5), V2 passes both DS-2 and DS-3. "
            "No variant adopted; research.py is left unmodified; the estimator "
            "is not repairable at this scope."
        ),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"written {OUT_PATH}")
    print(doc["verdict"])


if __name__ == "__main__":
    main()
