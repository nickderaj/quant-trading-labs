"""Notebook 017 Phase 3 (NEXT_PROMPT.md sec 4.1, sec 9.2): one grid cell per
invocation (resumable, one output file per cell), plus a --collate mode
that assembles all cell files into the calibration certificate with the
sec 10 hash gate.

Usage (one cell):
  uv run python src/research/tmp/run_phase_3_17_calibration.py \
      --n-trials N --n-obs T --rho R --skew S --kurt K --mode {null,edge} \
      --reps M --out path/to/cell.json

Usage (collate):
  uv run python src/research/tmp/run_phase_3_17_calibration.py --collate \
      --cells scratch/017/cells --out src/research/tmp/phase_3_17_calibration.json
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

# sec 5.3: "inject a true per-period Sharpe corresponding to annualized
# 1.0". The grid's T axis mirrors 018's 8h-bar-count scale (sec 4.1), so
# the same annualization convention is used throughout.
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


def run_one_cell(args: argparse.Namespace) -> None:
    moments = (args.skew, args.kurt)
    label = moments_label_for(*moments)
    true_sharpe = DS3_TRUE_SHARPE_PER_PERIOD if args.mode == "edge" else 0.0

    key = (
        args.n_trials,
        args.n_obs,
        args.rho,
        moments,
        args.reps,
        true_sharpe,
        args.mode,
    )
    seed = L.seed_for_cell(key)

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

    dsr_source = inspect.getsource(research.deflated_sharpe_prob)
    dsr_variant_source = inspect.getsource(L.dsr_variant)

    doc = {
        "notebook": "017_deflated_sharpe_correction",
        "phase": 3,
        "n_cells": len(cells),
        "cells": cells,
        "estimator_source_sha256": {
            "research.deflated_sharpe_prob": hashlib.sha256(
                dsr_source.encode()
            ).hexdigest(),
            "dsr_lib17.dsr_variant": hashlib.sha256(
                dsr_variant_source.encode()
            ).hexdigest(),
        },
    }
    with open(out_path, "w") as f:
        json.dump(doc, f)
    print(f"collated {len(cells)} cells -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collate", action="store_true")
    parser.add_argument("--n-trials", type=int)
    parser.add_argument("--n-obs", type=int)
    parser.add_argument("--rho", type=float)
    parser.add_argument("--skew", type=float)
    parser.add_argument("--kurt", type=float)
    parser.add_argument("--mode", choices=["null", "edge"])
    parser.add_argument("--reps", type=int)
    parser.add_argument(
        "--chunk",
        type=int,
        default=200,
        help="replications per mc_cell chunk; lower trades speed for peak memory "
        "(default 200 -> ~0.75GB per (chunk,T,N) array at N=122,T=3840; reduced to "
        "50 for the tail of this grid's run after repeated out-of-memory crashes "
        "at N=95/122 under -P 4)",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--cells")
    args = parser.parse_args()

    if args.collate:
        collate(args.cells, args.out)
    else:
        run_one_cell(args)


if __name__ == "__main__":
    main()
