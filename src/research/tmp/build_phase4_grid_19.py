"""Notebook 019 (NEXT_PROMPT.md sec 5.1): assembles Phase 4's confirmation
grid -- the three pre-declared subsets C1/C2/C3 -- into one manifest, run
AFTER Phase 2 (C2 needs its measured branch probabilities) and AFTER Phase
3 is on disk (sec 4's ordering rule: Phase 4 must not start before Phase
3's prediction exists).

C1 -- new points 017 never ran (sec 5.1): N x T x rho x moments x mode,
enumerated directly.

C2 -- ambiguous points (sec 5.1): every 017-grid DESIGN POINT (N, T, rho,
moments) where Phase 2 measures p(tau=0.15) in (0.02, 0.98) in EITHER mode
-- both modes are then included for that design point. Capped at 60 design
points (up to 120 cells), ranked by |p-0.5|, truncation disclosed if hit.

C3 -- a deterministic control sample (sec 5.1): 24 cells drawn from 017's
own 756 with numpy.random.default_rng(19), excluding N>=95 and T=3840.

Every cell gets a FRESH seed, namespaced apart from 017's own
seed_for_cell(key) (dsr_lib17.py) even where the (N, T, rho, moments, mode)
tuple coincides with a cell 017 already ran (true for all of C2 and C3) --
sec 5.1: "genuinely independent replications, not a re-run of 017's exact
random draws."

Usage: uv run python src/research/tmp/build_phase4_grid_19.py
Writes: scratch/019/phase4_manifest.json, scratch/019/phase4_cells.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsr_lib17 as L

CALIBRATION_PATH = "src/research/tmp/phase_3_17_calibration.json"
PROFILE_PATH = "src/research/tmp/phase_2_19_switch_profile.json"
MANIFEST_OUT = "scratch/019/phase4_manifest.json"
CELLS_OUT = "scratch/019/phase4_cells.txt"

C2_TAU_FOR_SELECTION = 0.15
C2_LOW, C2_HIGH = 0.02, 0.98
C2_CAP = 60
C3_N = 24
C3_SEED = 19

MOMENTS_LABEL_TO_VALUES = {
    "gaussian": L.GAUSSIAN_MOMENTS,
    "moderate_nongaussian": L.MODERATE_MOMENTS,
    "018_measured": L.EXTREME_MOMENTS,
}


def build_c1() -> list[dict]:
    ns = [12, 14, 24, 50]
    ts = [5000]
    rhos = [0.0, 0.5, 0.9, 0.95]
    cells = []
    for n in ns:
        for t in ts:
            for rho in rhos:
                for label, (skew, kurt) in MOMENTS_LABEL_TO_VALUES.items():
                    for mode in ("null", "edge"):
                        cells.append(
                            {
                                "subset": "C1",
                                "n_trials": n,
                                "n_obs": t,
                                "rho": rho,
                                "skew": skew,
                                "kurt": kurt,
                                "moments_label": label,
                                "mode": mode,
                            }
                        )
    return cells


def build_c2(profile_cells: list[dict]) -> tuple[list[dict], dict]:
    by_design_point: dict[tuple, dict[str, dict]] = {}
    for row in profile_cells:
        key = (row["n_trials"], row["n_obs"], row["rho"], row["moments_label"])
        by_design_point.setdefault(key, {})[row["mode"]] = row

    ambiguous = []
    for key, by_mode in by_design_point.items():
        ps = [
            r["p_select_v1_by_tau"][str(C2_TAU_FOR_SELECTION)] for r in by_mode.values()
        ]
        if any(C2_LOW < p < C2_HIGH for p in ps):
            dist = min(abs(p - 0.5) for p in ps)
            ambiguous.append((dist, key, by_mode))

    ambiguous.sort(key=lambda t: t[0])
    truncated = len(ambiguous) > C2_CAP
    selected = ambiguous[:C2_CAP]

    cells = []
    for _, (n_trials, n_obs, rho, moments_label), by_mode in selected:
        skew, kurt = MOMENTS_LABEL_TO_VALUES[moments_label]
        for mode in ("null", "edge"):
            cells.append(
                {
                    "subset": "C2",
                    "n_trials": n_trials,
                    "n_obs": n_obs,
                    "rho": rho,
                    "skew": skew,
                    "kurt": kurt,
                    "moments_label": moments_label,
                    "mode": mode,
                }
            )
    disclosure = {
        "n_ambiguous_design_points_found": len(ambiguous),
        "cap": C2_CAP,
        "truncated": truncated,
        "selection_rule": f"p(tau={C2_TAU_FOR_SELECTION}) in ({C2_LOW}, {C2_HIGH}) in "
        "either mode; ranked by min |p-0.5| across modes if truncation needed",
    }
    return cells, disclosure


def build_c3(cert_cells: list[dict]) -> list[dict]:
    eligible = [
        c for c in cert_cells if not (c["n_trials"] >= 95 and c["n_obs"] == 3840)
    ]
    rng = np.random.default_rng(C3_SEED)
    idx = rng.choice(len(eligible), size=C3_N, replace=False)
    selected = [eligible[i] for i in idx]
    cells = []
    for c in selected:
        cells.append(
            {
                "subset": "C3",
                "n_trials": c["n_trials"],
                "n_obs": c["n_obs"],
                "rho": c["rho"],
                "skew": c["moments"][0],
                "kurt": c["moments"][1],
                "moments_label": c["moments_label"],
                "mode": c["mode"],
            }
        )
    return cells


def predicted_cost_seconds(cells: list[dict], m: int) -> float:
    a, b = 2.5395633012820515e-08, 1.6332131410256405e-07
    return sum((a * c["n_trials"] + b) * c["n_obs"] * m for c in cells)


def main() -> None:
    with open(CALIBRATION_PATH) as f:
        cert_cells = json.load(f)["cells"]
    with open(PROFILE_PATH) as f:
        profile_cells = json.load(f)["cells"]

    c1 = build_c1()
    c2, c2_disclosure = build_c2(profile_cells)
    c3 = build_c3(cert_cells)

    all_cells = c1 + c2 + c3
    # de-dup exact (n_trials, n_obs, rho, skew, kurt, mode) tuples across
    # subsets (possible in principle between C2 and C3, both drawn from
    # 017's grid) -- keep the first occurrence's subset tag, disclosed.
    seen: dict[tuple, dict] = {}
    dup_count = 0
    for c in all_cells:
        key = (c["n_trials"], c["n_obs"], c["rho"], c["skew"], c["kurt"], c["mode"])
        if key in seen:
            dup_count += 1
            continue
        seen[key] = c
    deduped = list(seen.values())

    m = 20000
    manifest = {
        "notebook": "019_dsr_correlation_switch",
        "phase": "4_grid_build",
        "n_cells_c1": len(c1),
        "n_cells_c2": len(c2),
        "n_cells_c3": len(c3),
        "n_cells_total_before_dedup": len(all_cells),
        "n_duplicate_cells_collapsed": dup_count,
        "n_cells_final": len(deduped),
        "c2_disclosure": c2_disclosure,
        "m": m,
        "predicted_cost_seconds_by_subset": {
            "C1": predicted_cost_seconds(c1, m),
            "C2": predicted_cost_seconds(c2, m),
            "C3": predicted_cost_seconds(c3, m),
        },
        "cells": deduped,
    }
    with open(MANIFEST_OUT, "w") as f:
        json.dump(manifest, f, indent=2)

    with open(CELLS_OUT, "w") as f:
        for c in deduped:
            f.write(
                f"{c['n_trials']} {c['n_obs']} {c['rho']} {c['skew']} {c['kurt']} "
                f"{c['mode']} {c['subset']}\n"
            )

    total_cost = sum(predicted_cost_seconds(s, m) for s in (c1, c2, c3))
    print(f"written {MANIFEST_OUT}, {CELLS_OUT}")
    print(
        f"C1={len(c1)} C2={len(c2)} C3={len(c3)} dedup_collapsed={dup_count} "
        f"final={len(deduped)}"
    )
    print("C2 disclosure:", c2_disclosure)
    print(
        f"predicted total cost (single-core): {total_cost:.0f}s ({total_cost / 3600:.2f}h)"
    )


if __name__ == "__main__":
    main()
