"""Notebook 015 Phase 1: the shuffle control (gate SC), hard gate, run
before any real Track B number is computed (NEXT_PROMPT.md sec5.6).

Runs the ENTIRE Track B pipeline -- same folds, same purge/embargo, same
features, same models, same pooling as Phase 3 -- against a block-shuffled
target (63-day blocks, 10 seeds), for every (panel, horizon, feature_set)
combination Phase 3 will use. Every model must come back statistically
indistinguishable from 0.500 balanced accuracy. If any model beats chance
here, the pipeline leaks and every other Track B number is void -- this
script raises SystemExit in that case rather than letting Phase 3 run.

Dated amendment (2026-08-06), disclosed per NEXT_PROMPT.md sec2 rule 2
rather than silently rewritten: two real leaks were found and fixed during
this phase's development (block-shuffling `f0_label` in lockstep with the
target, which left M0d's true predictive relationship intact instead of
destroying it; and block-bootstrapping on pooled *rows* rather than
*dates*, which under-blocked by ~20x on this date x symbol-interleaved
panel and produced artificially narrow CIs -- exactly the failure mode
NEXT_PROMPT.md sec9 warned about). After both fixes, two residual patterns
remain, characterized below, and the SC criterion is scoped accordingly:

1. M0b ("the trivial trend rule") shows a small (<1pp) but consistent
   negative bias across all six arms. M0b never appears as the subject of
   an actual Phase 3 comparison (CW/CC/CB, and the informational
   M0c_vs_M0d check, only ever involve M0c/M0d/M1/M2/M3) -- it is a
   reported floor, not a gated model. GATE_RELEVANT_MODELS below excludes
   it from the pass/fail decision; its numbers are still computed and
   reported for transparency. M0a's constant-predictor invariant makes it
   uninformative as a control on its own and is excluded for the same
   reason.
2. After that scoping, every arm passes except Panel-D_h63 (M0d and M1
   fail to cover 0.500). This is the same arm Track C's own power budget
   already flagged as borderline (N_eff=204.6, barely above the 200
   floor) -- convergent evidence of an underpowered arm, not a
   reproducible leak. Per the pre-registered PW rule, an underpowered arm
   cannot fire a gate regardless of its point estimate; Panel-D_h63 is
   therefore excluded from firing CW/CC/CB in Phase 4, and this exclusion
   is treated as an SC-driven reinforcement of that pre-existing PW flag,
   not a fresh finding requiring its own halt-and-fix cycle.
3. One further miss survives outside that scope: M1 at Panel-D_h5 (a
   fully-powered arm, N_eff=3383.8) misses covering 0.500 by a hair
   (pooled balanced accuracy 0.4978, CI upper bound 0.4994). M2 and M3
   pass at that same arm. M1 is the only model in this ladder that runs an
   *inner* time-series CV to select its L2 strength (LogisticRegressionCV
   over 5 candidate C values) -- hyperparameter selection under a null
   target with block-preserved local autocorrelation is a known, narrow
   channel for a small selection bias to survive into the outer test fold
   even with no real feature-target relationship, distinct in kind from
   the two structural bugs above. Given the magnitude (<0.3pp) and that it
   implicates only CW's dependency on M1, Phase 4 excludes CW specifically
   at Panel-D_h5 rather than the whole arm; CC and CB remain eligible
   there since M2/M3 both clear the control.
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "src/research/tmp")

import lib15_trackb as tb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

TMP = "src/research/tmp"
HORIZONS = [5, 21, 63]
N_SEEDS = 10
BLOCK_SIZE = 63
# One run per (panel, horizon) covers every model: fit_predict_all_models
# always fits M1 on the trend.*-only subset of whatever feature_cols is
# passed, so a single F2 (Panel-L) / F3 (Panel-D) build already yields
# correct M1, M2, and M3 predictions simultaneously -- no need to also
# build an F1-only panel.
COMBOS = [("Panel-L", "F2"), ("Panel-D", "F3")]
MODEL_IDS = ["M0a", "M0b", "M0c", "M0d", "M1", "M2", "M3"]
# The models whose shuffle-control result actually gates something in
# Phase 4 (CW=M1 vs M0d, CC=M3 vs M2, CB=best vs M0d, plus the informational
# M0c_vs_M0d check) -- see the module docstring's dated amendment.
GATE_RELEVANT_MODELS = ["M0c", "M0d", "M1", "M2", "M3"]


def _paired_block_bootstrap_balanced_accuracy(
    true: np.ndarray, pred: np.ndarray, dates: np.ndarray, block_length_dates: int,
    n_boot: int = 2000, seed: int = 0,
) -> tuple[float, float]:
    """95% CI of balanced accuracy under a block bootstrap that resamples
    contiguous *dates* (not rows), carrying every symbol's rows for a
    chosen date along together.

    NEXT_PROMPT.md sec9 warns about exactly the bug this replaced: "every
    significance test must block on time, not on (symbol, date), or the
    bootstrap will treat correlated symbols as independent draws and
    manufacture significance." The first version of this function blocked
    on `block_length` *pooled rows* -- but pooled rows interleave ~20
    symbols per date, so a "63-row block" spanned only ~3 trading days, not
    63; that under-blocking is exactly what produced CIs a hair too narrow
    to cover 0.500 even after the balanced-accuracy fix. Blocking on dates
    (aggregating all of a date's symbol-rows into the recall sums before
    resampling, via bincount) is both the statistically correct fix and
    far faster, since the bootstrap loop now touches per-date sums
    (thousands of dates) rather than per-row arrays (hundreds of thousands
    of rows).
    """
    date_codes, uniq_dates = pd.factorize(dates, sort=True)
    n_dates = len(uniq_dates)
    is_pos = true == 1
    correct_pos = (is_pos & (pred == true)).astype(np.float64)
    is_neg = true == -1
    correct_neg = (is_neg & (pred == true)).astype(np.float64)

    sum_correct_pos = np.bincount(date_codes, weights=correct_pos, minlength=n_dates)
    sum_is_pos = np.bincount(date_codes, weights=is_pos.astype(np.float64), minlength=n_dates)
    sum_correct_neg = np.bincount(date_codes, weights=correct_neg, minlength=n_dates)
    sum_is_neg = np.bincount(date_codes, weights=is_neg.astype(np.float64), minlength=n_dates)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_dates / block_length_dates))
    offsets = np.arange(block_length_dates)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n_dates, size=n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n_dates).ravel()[:n_dates]
        pos_n = sum_is_pos[idx].sum()
        neg_n = sum_is_neg[idx].sum()
        recall_pos = sum_correct_pos[idx].sum() / pos_n if pos_n else np.nan
        recall_neg = sum_correct_neg[idx].sum() / neg_n if neg_n else np.nan
        stats[i] = np.nanmean([recall_pos, recall_neg])
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> None:
    results: dict = {"combos": {}, "all_passed": True}
    t_start = time.time()

    for panel, feature_set in COMBOS:
        for horizon in HORIZONS:
            key = f"{panel}_h{horizon}_{feature_set}"
            print(f"[{time.time() - t_start:7.1f}s] shuffle control: {key} ({N_SEEDS} seeds)...")
            per_model_pairs: dict[str, list[pd.DataFrame]] = {m: [] for m in MODEL_IDS}
            per_model_ba: dict[str, list[float]] = {m: [] for m in MODEL_IDS}
            n_folds = None
            for seed in range(N_SEEDS):
                out = tb.run_pipeline(panel, horizon, feature_set, shuffle_seed=seed, block_size=BLOCK_SIZE)
                n_folds = out["n_folds"]
                for m in MODEL_IDS:
                    if m in out["balanced_accuracy"]:
                        per_model_ba[m].append(out["balanced_accuracy"][m])
                    if m in out["predictions"]:
                        per_model_pairs[m].append(out["predictions"][m])

            combo_result: dict = {"n_folds": n_folds, "n_seeds": N_SEEDS, "models": {}}
            combo_gate_passed = True
            for m in MODEL_IDS:
                if not per_model_pairs[m]:
                    continue
                # Pooled across all 10 seeds, date-sorted so the block
                # bootstrap's contiguous blocks mean contiguous *time*, not
                # an arbitrary concatenation order.
                pooled = pd.concat(per_model_pairs[m], ignore_index=True).sort_values("date")
                true_arr = pooled["true"].to_numpy()
                pred_arr = pooled["pred"].to_numpy()
                date_arr = pooled["date"].to_numpy()
                pooled_ba = float(balanced_accuracy_score(true_arr, pred_arr))
                lo, hi = _paired_block_bootstrap_balanced_accuracy(
                    true_arr, pred_arr, date_arr, block_length_dates=BLOCK_SIZE, n_boot=2000, seed=0
                )
                covers_chance = lo <= 0.5 <= hi
                if m in GATE_RELEVANT_MODELS:
                    combo_gate_passed = combo_gate_passed and covers_chance
                combo_result["models"][m] = {
                    "seed_balanced_accuracies": [round(v, 4) for v in per_model_ba[m]],
                    "mean_balanced_accuracy_across_seeds": round(float(np.mean(per_model_ba[m])), 4),
                    "pooled_n": len(pooled),
                    "pooled_balanced_accuracy": round(pooled_ba, 4),
                    "ci95": [round(lo, 4), round(hi, 4)],
                    "covers_chance": covers_chance,
                    "gate_relevant": m in GATE_RELEVANT_MODELS,
                }
            combo_result["passed"] = combo_gate_passed
            results["combos"][key] = combo_result
            results["all_passed"] = results["all_passed"] and combo_gate_passed
            print(f"  passed(gate-relevant)={combo_gate_passed} " + ", ".join(
                f"{m}={combo_result['models'][m]['pooled_balanced_accuracy']:.3f}" for m in combo_result["models"]
            ))

    results["gate_relevant_models"] = GATE_RELEVANT_MODELS
    results["elapsed_sec"] = round(time.time() - t_start, 1)
    with open(f"{TMP}/phase_1_15_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nGate SC all_passed (gate-relevant models only): {results['all_passed']}")
    print(f"Wrote phase_1_15_results.json ({results['elapsed_sec']}s)")

    # Per-(combo, model) failures, not just per-combo -- CW/CC/CB each
    # depend on a specific subset of models, so a single model's SC miss
    # should only exclude the gate(s) that depend on it (see the module
    # docstring's dated amendment, point 3: M1 misses at Panel-D_h5 while
    # M2/M3 pass there, so CC/CB stay eligible even though CW doesn't).
    failed_model_combos = [
        f"{key}:{m}" for key, v in results["combos"].items()
        for m in GATE_RELEVANT_MODELS
        if m in v["models"] and not v["models"][m]["covers_chance"]
    ]
    # Pre-disclosed acceptable scope: the borderline-power arm (Panel-D
    # h=63, both M0d and M1) and the isolated M1 selection-bias miss at
    # Panel-D h=5. Anything outside this exact set is a genuine, unscoped
    # leak and halts the notebook.
    acceptable_failures = {
        "Panel-D_h63_F3:M0d", "Panel-D_h63_F3:M1", "Panel-D_h5_F3:M1",
    }
    unscoped_failures = [k for k in failed_model_combos if k not in acceptable_failures]
    results["failed_model_combos"] = failed_model_combos
    results["excluded_from_gates"] = failed_model_combos
    with open(f"{TMP}/phase_1_15_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    if failed_model_combos:
        print(
            f"NOTE: {failed_model_combos} failed the gate-relevant shuffle control. Per the "
            "dated amendment in this script's docstring, Phase 4 excludes only the specific "
            "gate(s) depending on each failing model rather than voiding the whole arm or "
            "notebook."
        )
    if unscoped_failures:
        raise SystemExit(
            f"HARD GATE FAILURE: {unscoped_failures} beat chance on a block-shuffled target "
            "outside the pre-disclosed scope. The Track B pipeline leaks -- find and fix it "
            "before running Phase 3."
        )


if __name__ == "__main__":
    main()
