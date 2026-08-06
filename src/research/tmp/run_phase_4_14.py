"""Notebook 014 Phase 4: pre-registered gates (NEXT_PROMPT.md sec6 Phase 4).

Scores the six gates frozen in phase_0_14_preregistration.json against
Phase 0-3's actual results, applying Bonferroni correction across the
honestly-counted n_trials from Phase 3 (39: 14 episode-table + 25
mechanical-label sector/dimension pairs) for the two significance gates
(RA, RM).

Unlike every alpha-gate notebook before this one, a gate FIRING (passing)
is the desired outcome here -- the engine already ships in production, so a
null means production has been shipping noise. Report whichever way it
lands without hedging.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")

import numpy as np

TMP = "src/research/tmp"
RL_THRESHOLD_DAYS = 21
BONFERRONI_ALPHA = 0.05


def main() -> None:
    with open(f"{TMP}/phase_0_14_preregistration.json") as f:
        prereg = json.load(f)
    with open(f"{TMP}/phase_0_14_results.json") as f:
        phase0 = json.load(f)
    with open(f"{TMP}/phase_2_14_results.json") as f:
        phase2 = json.load(f)
    with open(f"{TMP}/phase_3_14_results.json") as f:
        phase3 = json.load(f)

    n_trials = phase3["n_trials"]["total"]
    bonferroni_threshold = BONFERRONI_ALPHA / n_trials

    # NL -- hard gate, already run in Phase 0.
    nl_pass = (
        phase0["no_lookahead_gate"]["all_passed"]
        and phase0["no_lookahead_gate"]["oil_products_cot_opt_in"]["passed"]
    )

    # RC -- port fidelity.
    fidelity = phase0["port_fidelity"]
    rc_pass = all(fidelity.get("config_hash_equal", {}).values())
    if fidelity.get("end_to_end_equal") is not None:
        rc_pass = rc_pass and all(
            v["scores_equal"] and v["labels_equal"]
            for v in fidelity["end_to_end_equal"].values()
        )

    # RS -- descriptive sanity: no scored (non-disqualified) dimension trips
    # the >90% occupancy / >1-flip-per-10-bars thresholds. Phase 2 already
    # disqualifies and excludes any dimension that does; RS asks whether
    # anything SCORED in Phase 3 still trips them (it shouldn't, since
    # disqualified dims are excluded from Phase 3 by construction) plus
    # reports the raw disqualification count for transparency.
    disqualified_pairs = {(d["sector"], d["dimension"]) for d in phase2["disqualified"]}
    scored_pairs = set()
    for key, v in phase3["episode_table"]["per_sector_dimension"].items():
        if v.get("excluded"):
            continue
        sector, dimension = key.split("|", 1)
        scored_pairs.add((sector, dimension))
    for key, v in phase3["mechanical_labels"].items():
        if v.get("insufficient_data") or v.get("excluded"):
            continue
        sector, dimension = key.split("|")[0], key.split("|")[1]
        scored_pairs.add((sector, dimension))
    rs_violations = scored_pairs & disqualified_pairs
    rs_pass = len(rs_violations) == 0

    # RL -- median label lag at episode onset <= 21 trading days.
    lags = [lag for lag in phase3["episode_table"]["lead_lag_days"] if lag is not None]
    median_lag = float(np.median(lags)) if lags else None
    rl_pass = median_lag is not None and abs(median_lag) <= RL_THRESHOLD_DAYS
    rl_n_censored = sum(
        1
        for v in phase3["episode_table"]["per_sector_dimension"].values()
        if not v.get("excluded")
        for ep in v.get("episodes", [])
        if ep["lead_lag_days"] is None
    )

    # RA -- episode-table balanced accuracy significantly beats class-prior,
    # Bonferroni-corrected across all n_trials.
    def _significant_wins(source: dict, baseline_name_filter=None) -> list[dict]:
        wins = []
        for key, v in source.items():
            vb = v.get("vs_best_baseline")
            if not vb:
                continue
            if baseline_name_filter and vb["baseline"] not in baseline_name_filter:
                continue
            if vb["pvalue"] < bonferroni_threshold and vb["mean_hit_rate_diff"] > 0:
                wins.append({"key": key, **vb})
        return wins

    ra_wins = _significant_wins(phase3["episode_table"]["per_sector_dimension"])
    ra_pass = len(ra_wins) > 0

    rm_wins = _significant_wins(phase3["mechanical_labels"])
    rm_pass = len(rm_wins) > 0

    gates = {
        "NL": {
            "claim": prereg["gates"]["NL"]["claim"],
            "fires": bool(nl_pass),
            "hard_gate": True,
            "detail": phase0["no_lookahead_gate"]["all_passed"],
        },
        "RA": {
            "claim": prereg["gates"]["RA"]["claim"],
            "fires": bool(ra_pass),
            "n_trials": n_trials,
            "bonferroni_threshold": bonferroni_threshold,
            "significant_wins": ra_wins,
        },
        "RM": {
            "claim": prereg["gates"]["RM"]["claim"],
            "fires": bool(rm_pass),
            "n_trials": n_trials,
            "bonferroni_threshold": bonferroni_threshold,
            "significant_wins": rm_wins,
        },
        "RL": {
            "claim": prereg["gates"]["RL"]["claim"],
            "fires": bool(rl_pass),
            "median_lag_days": median_lag,
            "n_episodes_scored": len(lags),
            "n_censored_never_matched": rl_n_censored,
        },
        "RS": {
            "claim": prereg["gates"]["RS"]["claim"],
            "fires": bool(rs_pass),
            "n_disqualified_in_phase2": len(disqualified_pairs),
            "disqualified": sorted(f"{s}|{d}" for s, d in disqualified_pairs),
            "violations_among_scored": sorted(f"{s}|{d}" for s, d in rs_violations),
        },
        "RC": {
            "claim": prereg["gates"]["RC"]["claim"],
            "fires": bool(rc_pass),
            "config_hash_equal": fidelity.get("config_hash_equal"),
            "end_to_end_equal": fidelity.get("end_to_end_equal"),
        },
    }

    with open(f"{TMP}/phase_4_14_results.json", "w") as f:
        json.dump(gates, f, indent=2, default=str)

    for gate_id, g in gates.items():
        print(f"{gate_id}: {'FIRES' if g['fires'] else 'null'} -- {g['claim']}")
    print(f"\nn_trials={n_trials}, Bonferroni alpha={bonferroni_threshold:.6f}")
    print("Wrote phase_4_14_results.json")


if __name__ == "__main__":
    main()
