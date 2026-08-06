"""Notebook 015 Phase 3: the real Track B ladder (NEXT_PROMPT.md sec5).
Runs only after Phase 1's shuffle control (gate SC) has passed -- if it
hasn't, this script refuses to run.

One pipeline call per (panel, horizon) covers every model: F2 (Panel-L) /
F3 (Panel-D) is a strict superset of F1's columns, and M1 always fits on
just the trend.* subset internally (lib15_trackb.fit_predict_all_models),
so a single build yields correct M0a-M0d, M1, M2, and M3 predictions
together -- no need to also build an F1-only panel per sec5.3's ladder.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "src/research/tmp")

import lib15_trackb as tb
import numpy as np

import research

TMP = "src/research/tmp"
HORIZONS = [5, 21, 63]
PANEL_FEATURE_SET = {"Panel-L": "F2", "Panel-D": "F3"}
N_BOOT = 2000


def _paired_diff_significance(hits_a, hits_b) -> dict:
    """Block-bootstrap significance of the paired daily hit-rate difference
    (model A minus model B), aligned on dates both have a valid prediction
    for -- the same discipline as Track A and 014 (research.block_bootstrap_*,
    default auto block length, n_boot=2000, seed=0)."""
    common = hits_a.index.intersection(hits_b.index)
    if len(common) < 5:
        return {"n_obs": len(common), "insufficient_data": True}
    diff = (hits_a.loc[common] - hits_b.loc[common]).to_numpy()
    lo, hi = research.block_bootstrap_ci(diff, n_boot=N_BOOT, seed=0)
    pvalue = research.block_bootstrap_pvalue(diff, null_value=0.0, n_boot=N_BOOT, seed=0)
    return {"n_obs": len(common), "mean_diff": float(diff.mean()), "ci95": [lo, hi], "pvalue": pvalue}


def main() -> None:
    with open(f"{TMP}/phase_1_15_results.json") as f:
        phase1 = json.load(f)
    # Gate SC's dated amendment (run_phase_1_15_shuffle_control.py's
    # docstring) scopes three (combo, model) misses as pre-disclosed and
    # acceptable rather than blanket-passing every model; Phase 3 must
    # therefore check for *unscoped* failures specifically, not
    # `all_passed` (which is False whenever any scoped miss exists, by
    # design -- that flag means "nothing to exclude", not "safe to run").
    excluded = set(phase1.get("excluded_from_gates", []))
    acceptable = {"Panel-D_h63_F3:M0d", "Panel-D_h63_F3:M1", "Panel-D_h5_F3:M1"}
    unscoped = excluded - acceptable
    if unscoped:
        raise SystemExit(
            f"Phase 1 shuffle control (gate SC) has unscoped failures: {sorted(unscoped)}. "
            "Refusing to run Phase 3 -- fix the leak and re-run Phase 1 first."
        )
    with open(f"{TMP}/phase_0_15_preregistration.json") as f:
        prereg = json.load(f)
    power_budget = prereg["track_c"]["power_budget"]

    results: dict = {"combos": {}}

    for panel, feature_set in PANEL_FEATURE_SET.items():
        for horizon in HORIZONS:
            key = f"{panel}_h{horizon}"
            sc_key = f"{panel}_h{horizon}_{feature_set}"  # phase 1's combo naming
            print(f"Running Track B: {key} (feature_set={feature_set})...")
            out = tb.run_pipeline(panel, horizon, feature_set, shuffle_seed=None)
            hits = {}
            for m, series in out["daily_hit_rate"].items():
                hits[m] = series

            power_key = f"{panel}_h{horizon}"
            underpowered = power_budget.get(power_key, {}).get("underpowered", True)

            def _sc_excluded(*models: str, _sc_key: str = sc_key) -> bool:
                return any(f"{_sc_key}:{m}" in excluded for m in models)

            comparisons = {}
            if "M0c" in hits and "M0d" in hits:
                c = _paired_diff_significance(hits["M0c"], hits["M0d"])
                c["sc_excluded"] = _sc_excluded("M0c", "M0d")
                comparisons["M0c_vs_M0d"] = c
            if "M1" in hits and "M0d" in hits:
                c = _paired_diff_significance(hits["M1"], hits["M0d"])
                c["sc_excluded"] = _sc_excluded("M1", "M0d")
                comparisons["CW_M1_vs_M0d"] = c
            if "M3" in hits and "M2" in hits:
                c = _paired_diff_significance(hits["M3"], hits["M2"])
                c["sc_excluded"] = _sc_excluded("M3", "M2")
                comparisons["CC_M3_vs_M2"] = c

            best_model = None
            best_ba = -np.inf
            for m in ("M1", "M2", "M3"):
                if m in out["balanced_accuracy"] and out["balanced_accuracy"][m] > best_ba:
                    best_ba, best_model = out["balanced_accuracy"][m], m
            if best_model is not None and "M0d" in hits:
                cb = _paired_diff_significance(hits[best_model], hits["M0d"])
                cb["best_model"] = best_model
                cb["point_estimate_balanced_accuracy_gain"] = (
                    out["balanced_accuracy"][best_model] - out["balanced_accuracy"].get("M0d", float("nan"))
                )
                cb["sc_excluded"] = _sc_excluded(best_model, "M0d")
                comparisons["CB_best_vs_M0d"] = cb

            results["combos"][key] = {
                "panel": panel, "horizon": horizon, "feature_set": feature_set,
                "n_folds": out["n_folds"], "n_rows": out["n_rows"],
                "balanced_accuracy": out["balanced_accuracy"], "n_obs": out["n_obs"],
                "abstention_rate": out["abstention_rate"],
                "underpowered": underpowered,
                "comparisons": comparisons,
            }
            print(f"  balanced_accuracy: {out['balanced_accuracy']}")
            print(f"  underpowered: {underpowered}")

    with open(f"{TMP}/phase_3_15_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Wrote phase_3_15_results.json")


if __name__ == "__main__":
    main()
