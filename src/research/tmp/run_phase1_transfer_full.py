"""Phase 1 driver: does GARCH-t's Gate A density win generalize across
symbols at 1h/4h/12h?

Re-runs notebook 5's entire Phase 3 density contest (same 8 competitors, same
all-pairs DM, same BH adjustment, same bootstrap p-values) via dist_lib6's
shared machinery (build_gate_a_forecasts / score_gate_a_models /
all_pairs_dm_bh / beats_all_significantly - factored out of
run_phase3_density.py in checkpoint 0 specifically so this driver reuses it
byte-for-byte rather than re-implementing a second, drifting copy).

Run per-symbol: `python run_phase1_transfer_full.py SYMBOL`. Each invocation
writes its own `phase1_transfer_{symbol}.json` so subagents fanned out by
symbol (NEXT_RUN_PROMPT.md section 1) never write-conflict. BTC is handled
specially: rather than recomputing (already done in phase3_density_results.json),
this script's BTC invocation is a VALIDATION run whose 12h numbers must match
phase3_density_results.json before any transfer symbol is trusted.

Interval order is 12h, then 4h, then 1h (cheapest and most decisive first per
the runbook's own compute-budget instruction) - if 1h cannot complete, drop it
and say so rather than deliver a ragged partial table.
"""

import json
import sys
import time

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import numpy as np  # noqa: E402

import dist_lib6 as L6  # noqa: E402
import research  # noqa: E402

DM_BOOTSTRAP_N = 500
INTERVAL_ORDER = ["12h", "4h", "1h"]


def run_one(symbol: str, interval: str) -> dict:
    import dist_lib as L

    t0 = time.time()
    df = L.build_asset_frame(symbol, interval, end=research.HOLDOUT_START)
    n = len(df)
    ret = df["log_return"].fill_null(0.0).to_numpy()

    variance_fc, nu_paths, _fits = L6.build_gate_a_forecasts(df, interval, ret)
    log_score_full, scores = L6.score_gate_a_models(ret, variance_fc, nu_paths)

    model_names = L6.GATE_A_MODEL_IDS
    all_pairs_dm = L6.all_pairs_dm_bh(model_names, log_score_full, dm_bootstrap_n=DM_BOOTSTRAP_N, seed=0)

    mean_log_score = {name: scores[name]["log_score_mean"] for name in model_names}
    best_name = max(mean_log_score, key=mean_log_score.get)
    gate_fires_boot = L6.beats_all_significantly(best_name, model_names, all_pairs_dm, "bh_bootstrap")
    gate_fires_normal = L6.beats_all_significantly(best_name, model_names, all_pairs_dm, "bh_normal")

    return {
        "n_obs": n, "scores": scores, "all_pairs_dm": all_pairs_dm,
        "best_by_log_score": best_name,
        "beats_every_other_significantly_bootstrap_bh": gate_fires_boot,
        "beats_every_other_significantly_normal_bh": gate_fires_normal,
        "elapsed_sec": time.time() - t0,
    }


def main():
    if len(sys.argv) < 2:
        print("usage: run_phase1_transfer_full.py SYMBOL [interval ...]")
        sys.exit(1)
    symbol = sys.argv[1]
    intervals = sys.argv[2:] if len(sys.argv) > 2 else INTERVAL_ORDER

    out = {"symbol": symbol, "intervals": {}}
    for interval in intervals:
        res = run_one(symbol, interval)
        out["intervals"][interval] = res
        print(f"{symbol} {interval}: n={res['n_obs']} elapsed={res['elapsed_sec']:.1f}s "
              f"best={res['best_by_log_score']} "
              f"gate(boot)={res['beats_every_other_significantly_bootstrap_bh']} "
              f"gate(normal)={res['beats_every_other_significantly_normal_bh']}")
        out_path = f"src/research/tmp/phase1_transfer_{symbol}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=1, default=float)
        print(f"  (incremental write to {out_path})")


if __name__ == "__main__":
    main()
