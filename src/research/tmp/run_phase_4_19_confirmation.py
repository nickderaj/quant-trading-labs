"""Notebook 019 Phase 4 (NEXT_PROMPT.md sec 4 row 4, sec 5.1): the
confirmation grid. Real V3 Monte Carlo on the three pre-declared subsets
(C1/C2/C3, built by build_phase4_grid_19.py). Scores DS-7 and DS-8. The
only phase that runs new Monte Carlo -- must not start before Phase 3's
prediction (phase_3_19_prediction.json) is on disk (sec 4's ordering rule).

Reuses dsr_lib17.mc_cell unmodified (sec 4.1's reuse rule): one call per
design point computes ALL variants (v0, v1, v2, v1b_c0.25, v1b_c0.5,
v3_tau0.15, v3_tau0.30) from the SAME draws, so DS-7's predicted-vs-measured
comparison and DS-8's V3-vs-V0 comparison are always on matched
replications. One cell per invocation (resumable, one output file per
cell), plus a --collate mode.

Usage (one cell):
  uv run python src/research/tmp/run_phase_4_19_confirmation.py \
      --n-trials N --n-obs T --rho R --skew S --kurt K --mode {null,edge} \
      --subset {C1,C2,C3} --reps M --out path/to/cell.json

Usage (collate):
  uv run python src/research/tmp/run_phase_4_19_confirmation.py \
      --collate --cells scratch/019/cells/phase4 \
      --out src/research/tmp/phase_4_19_confirmation.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import dsr_lib17 as L

import research

DS3_TRUE_SHARPE_PER_PERIOD = 1.0 / research.sharpe_to_annualized_rate("8h")

MOMENTS_BY_LABEL = {
    "gaussian": L.GAUSSIAN_MOMENTS,
    "moderate_nongaussian": L.MODERATE_MOMENTS,
    "018_measured": L.EXTREME_MOMENTS,
}


def moments_label_for(skew: float, kurt: float) -> str:
    for label, (s, k) in MOMENTS_BY_LABEL.items():
        if abs(s - skew) < 1e-9 and abs(k - kurt) < 1e-9:
            return label
    raise ValueError(f"unregistered moments: skew={skew}, kurt={kurt}")


def seed_for_phase4_cell(
    n_trials: int,
    n_obs: int,
    rho: float,
    moments: tuple[float, float],
    mode: str,
    n_reps: int,
) -> int:
    """A 019-Phase-4-specific seed, namespaced apart from 017's own
    seed_for_cell(key) even where the (N, T, rho, moments, mode) tuple
    coincides with a cell 017 already ran (true for every C2/C3 cell) --
    sec 5.1: genuinely independent replications, not a replay."""
    key = ("019_phase4", n_trials, n_obs, rho, moments, mode, n_reps)
    digest = hashlib.sha256(repr(key).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def run_one_cell(args: argparse.Namespace) -> None:
    moments = (args.skew, args.kurt)
    label = moments_label_for(*moments)
    true_sharpe = DS3_TRUE_SHARPE_PER_PERIOD if args.mode == "edge" else 0.0
    seed = seed_for_phase4_cell(
        args.n_trials, args.n_obs, args.rho, moments, args.mode, args.reps
    )

    result = L.mc_cell(
        args.n_trials,
        args.n_obs,
        args.rho,
        moments,
        args.reps,
        seed,
        true_sharpe,
        chunk=args.chunk,
    )
    result["mode"] = args.mode
    result["moments_label"] = label
    result["subset"] = args.subset

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f)


def collate(cells_dir: str, out_path: str) -> None:
    cell_files = sorted(glob.glob(f"{cells_dir}/*.json"))
    cells = []
    for fp in cell_files:
        with open(fp) as f:
            cells.append(json.load(f))

    by_subset: dict[str, int] = {}
    for c in cells:
        by_subset[c["subset"]] = by_subset.get(c["subset"], 0) + 1

    doc = {
        "notebook": "019_dsr_correlation_switch",
        "phase": 4,
        "n_cells": len(cells),
        "n_cells_by_subset": by_subset,
        "cells": cells,
        "estimator_source_sha256": {
            "dsr_lib17.dsr_variant": hashlib.sha256(
                inspect.getsource(L.dsr_variant).encode()
            ).hexdigest(),
        },
    }
    with open(out_path, "w") as f:
        json.dump(doc, f)
    print(f"collated {len(cells)} cells -> {out_path} ({by_subset})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collate", action="store_true")
    parser.add_argument("--n-trials", type=int)
    parser.add_argument("--n-obs", type=int)
    parser.add_argument("--rho", type=float)
    parser.add_argument("--skew", type=float)
    parser.add_argument("--kurt", type=float)
    parser.add_argument("--mode", choices=["null", "edge"])
    parser.add_argument("--subset", choices=["C1", "C2", "C3"])
    parser.add_argument("--reps", type=int)
    parser.add_argument("--chunk", type=int, default=200)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cells")
    args = parser.parse_args()

    if args.collate:
        collate(args.cells, args.out)
    else:
        run_one_cell(args)


if __name__ == "__main__":
    main()
