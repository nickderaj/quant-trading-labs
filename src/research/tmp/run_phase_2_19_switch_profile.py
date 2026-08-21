"""Notebook 019 Phase 2 (NEXT_PROMPT.md sec 4 row 2, sec 3.3/DS-6): the
switch-activation profile. P(mean_pairwise_corr_estimate >= tau) at all 378
(N, T, rho, moments) design points x both modes (null, edge), M=2000.
Completes the M=200 preflight probe (partial outside the Gaussian regime;
scratch/019/preflight_switch_probe.py) at the grid's real M.

One cell per invocation (resumable, one output file per cell, mirroring
017's Phase 3 shape -- sec 4.1's reuse rule), plus a --collate mode.

Usage (one cell):
  uv run python src/research/tmp/run_phase_2_19_switch_profile.py \
      --n-trials N --n-obs T --rho R --skew S --kurt K --mode {null,edge} \
      --reps M --out path/to/cell.json

Usage (collate):
  uv run python src/research/tmp/run_phase_2_19_switch_profile.py \
      --collate --cells scratch/019/cells/phase2 \
      --out src/research/tmp/phase_2_19_switch_profile.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import dsr_lib17 as L

import research

TAUS = (0.15, 0.30)

# Same annualization convention as 017's Phase 3 edge injection (sec 5.3):
# the grid's T axis mirrors 018's 8h-bar-count scale.
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


def seed_for_phase2_cell(
    n_trials: int,
    n_obs: int,
    rho: float,
    moments: tuple[float, float],
    mode: str,
    n_reps: int,
) -> int:
    """A 019-Phase-2-specific seed, deliberately namespaced apart from
    017's seed_for_cell(key) and from the preflight probe's own key shape
    -- this is a fresh measurement at the grid's real M, not a replay."""
    key = ("019_phase2", n_trials, n_obs, rho, moments, mode, n_reps)
    digest = hashlib.sha256(repr(key).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def run_one_cell(args: argparse.Namespace) -> None:
    moments = (args.skew, args.kurt)
    label = moments_label_for(*moments)
    seed = seed_for_phase2_cell(
        args.n_trials, args.n_obs, args.rho, moments, args.mode, args.reps
    )
    rng = np.random.default_rng(seed)
    true_sharpe = DS3_TRUE_SHARPE_PER_PERIOD if args.mode == "edge" else 0.0

    chunk = args.chunk
    if args.n_trials >= 95 and args.n_obs >= 3840:
        chunk = min(chunk, 50)

    ests = []
    done = 0
    while done < args.reps:
        m = min(chunk, args.reps - done)
        x = L.draw_trial_returns(rng, m, args.n_obs, args.n_trials, args.rho, moments)
        if true_sharpe != 0.0:
            x = x.copy()
            x[:, :, 0] = x[:, :, 0] + true_sharpe
        ests.append(L.mean_pairwise_corr_estimate(x))
        done += m
    est = np.concatenate(ests)

    p_by_tau = {str(t): float((est >= t).mean()) for t in TAUS}
    mc_se_by_tau = {
        str(t): float(np.sqrt(p_by_tau[str(t)] * (1 - p_by_tau[str(t)]) / args.reps))
        for t in TAUS
    }

    result = {
        "n_trials": args.n_trials,
        "n_obs": args.n_obs,
        "rho": args.rho,
        "moments": list(moments),
        "moments_label": label,
        "mode": args.mode,
        "n_reps": args.reps,
        "seed": seed,
        "mean_est": float(est.mean()),
        "sd_est": float(est.std()),
        "p_select_v1_by_tau": p_by_tau,
        "mc_se_by_tau": mc_se_by_tau,
    }

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

    at_zero = [c for c in cells if c["rho"] == 0.0]
    rho0_false_trigger = {
        f"max_p_select_v1_tau{t}": max(
            (c["p_select_v1_by_tau"][str(t)] for c in at_zero), default=float("nan")
        )
        for t in TAUS
    }
    ds6_first_clause_fires = all(
        rho0_false_trigger[f"max_p_select_v1_tau{t}"] <= 0.005 for t in TAUS
    )

    doc = {
        "notebook": "019_dsr_correlation_switch",
        "phase": 2,
        "n_cells": len(cells),
        "cells": cells,
        "rho0_false_trigger": rho0_false_trigger,
        "max_sd_of_estimate_at_true_rho0": max(
            (c["sd_est"] for c in at_zero), default=float("nan")
        ),
        "ds6_first_clause": {
            "fires_if": "P(estimate >= tau | true rho=0) <= 0.005 at every "
            "design point, both tau",
            "fires": ds6_first_clause_fires,
        },
    }
    with open(out_path, "w") as f:
        json.dump(doc, f)
    print(f"collated {len(cells)} cells -> {out_path}")
    print(
        "rho=0 false-trigger:",
        rho0_false_trigger,
        "DS-6 first clause fires:",
        ds6_first_clause_fires,
    )


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
