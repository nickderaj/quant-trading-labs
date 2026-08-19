"""Notebook 017 Phase 2 (NEXT_PROMPT.md sec 4, sec 5.1): the DS-1 kill switch.

Reproduces the sec 3 disclosed pilot's rho axis at honest M (>=20000, vs.
the pilot's M=400), across the FULL rho axis frozen in Phase 0 (not just the
pilot's four points), at the pilot's own (N, T, moments) = (18, 3840,
Gaussian) -- 018's own trial count and bar count, Gaussian since this is
the pilot's own regime, not yet the moment-robustness question Phase 3
covers. Also scores V2, which the pilot did not test.

If DS-1 does not fire: STOP. Do not patch research.py. Do not run Phase 3,
4, or 5 (sec 5.1). This script only diagnoses; it makes no changes to
research.py and produces no calibration certificate.

Usage: uv run python src/research/tmp/run_phase_2_17_diagnosis.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsr_lib17 as L

OUT_PATH = "src/research/tmp/phase_2_17_results.json"
PREREG_PATH = "src/research/tmp/phase_0_17_preregistration.json"

N_TRIALS = 18
N_OBS = 3840
TRUE_SHARPE = 0.0  # null: DS-1 is about the FPR under the null only


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    with open(PREREG_PATH) as f:
        prereg = json.load(f)
    rho_axis = prereg["grid"]["rho"]
    pilot = prereg["disclosed_pilot"]

    n_reps = 500 if args.smoke else 20000
    predicted_seconds = sum(2.6e-8 * N_TRIALS * N_OBS * n_reps for _ in rho_axis)
    print(
        f"Phase 2: {len(rho_axis)} cells, M={n_reps}, predicted ~{predicted_seconds:.1f}s"
    )

    t0 = time.time()
    cells = []
    for rho in rho_axis:
        key = (N_TRIALS, N_OBS, rho, L.GAUSSIAN_MOMENTS, n_reps, TRUE_SHARPE)
        seed = L.seed_for_cell(key)
        result = L.mc_cell(
            N_TRIALS, N_OBS, rho, L.GAUSSIAN_MOMENTS, n_reps, seed, TRUE_SHARPE
        )
        cells.append(result)
        print(
            f"  rho={rho:.2f}  V0 FPR={result['rate']['v0']:.4f}+-{result['mc_se']['v0']:.4f}"
            f"  V1={result['rate']['v1']:.4f}  V2={result['rate']['v2']:.4f}"
        )
    elapsed = time.time() - t0
    print(f"Phase 2 grid done in {elapsed:.1f}s")

    # --- DS-1 (sec 5.1) ---
    v0_by_rho = {c["rho"]: c["rate"]["v0"] for c in cells}
    v0_se_by_rho = {c["rho"]: c["mc_se"]["v0"] for c in cells}

    high_rho_cells = [rho for rho in rho_axis if rho >= 0.9]
    clause_1 = all(v0_by_rho[rho] <= 0.005 for rho in high_rho_cells)

    baseline = v0_by_rho[0.0]
    baseline_se = v0_se_by_rho[0.0]
    clause_2 = True
    clause_2_detail = []
    for rho in rho_axis:
        if rho == 0.0:
            continue
        margin = 2 * (baseline_se + v0_se_by_rho[rho])
        exceeds = v0_by_rho[rho] > baseline + margin
        clause_2_detail.append(
            {
                "rho": rho,
                "fpr": v0_by_rho[rho],
                "baseline_fpr": baseline,
                "2_mc_se_margin": margin,
                "exceeds_baseline_beyond_margin": exceeds,
            }
        )
        if exceeds:
            clause_2 = False

    ds1_fires = clause_1 and clause_2

    doc = {
        "notebook": "017_deflated_sharpe_correction",
        "phase": 2,
        "params": {
            "n_trials": N_TRIALS,
            "n_obs": N_OBS,
            "n_reps": n_reps,
            "smoke": args.smoke,
        },
        "rho_axis": rho_axis,
        "pilot_reference": pilot["results_fpr_gt_0.95"],
        "cells": cells,
        "gate_DS1": {
            "clause_1_high_rho_fpr_le_0.005": {
                "cells_checked": high_rho_cells,
                "fires": clause_1,
                "values": {rho: v0_by_rho[rho] for rho in high_rho_cells},
            },
            "clause_2_fpr_non_increasing_in_rho": {
                "baseline_rho0_fpr": baseline,
                "detail": clause_2_detail,
                "fires": clause_2,
            },
            "fires": ds1_fires,
        },
        "elapsed_seconds": elapsed,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"\nwritten {OUT_PATH}")
    print(f"DS-1 fires: {ds1_fires}")
    if not ds1_fires:
        print(
            "DS-1 DID NOT FIRE. Per sec 5.1: STOP. Do not patch research.py. "
            "Do not run Phase 3, 4, or 5."
        )


if __name__ == "__main__":
    main()
