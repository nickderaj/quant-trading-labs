"""Notebook 015 Phase 4: gate evaluation (NEXT_PROMPT.md sec7). Reads
phase_0-3 results, applies each pre-registered gate's threshold exactly as
frozen in phase_0_15_preregistration.json, and writes phase_4_15_results.json
-- the notebook's single source of truth for "which gates fired."
"""

from __future__ import annotations

import json

TMP = "src/research/tmp"


def load(name: str) -> dict:
    with open(f"{TMP}/{name}") as f:
        return json.load(f)


def _significant_positive(
    entry: dict | None, alpha: float, min_effect: float = 0.0
) -> bool:
    """True if `entry` (a _score_against_target / _paired_diff_significance
    style dict) clears Bonferroni-corrected significance, is directionally
    positive (engine/challenger beats baseline), isn't flagged structurally
    uninformative (014's degeneracy check), and clears any effect-size
    floor (CB's +0.05 requirement)."""
    if not entry or entry.get("insufficient_data") or entry.get("sc_excluded"):
        return False
    pvalue = entry.get("pvalue")
    if pvalue is None or pvalue >= alpha:
        return False
    if entry.get("structurally_uninformative"):
        return False
    effect = entry.get("mean_hit_rate_diff", entry.get("mean_diff", entry.get("mean")))
    if effect is None or effect <= 0:
        return False
    return bool(effect >= min_effect)


def evaluate_id_gate(phase0: dict) -> dict:
    pairs = phase0["track_a"]["disjointness_table"]["pairs"]
    scored_pairs = [p for p in pairs if not p["disqualified"]]
    all_disjoint = all(
        not (set(p["dimension_inputs"]) & set(p["target_inputs"])) for p in scored_pairs
    )
    return {
        "fires": all_disjoint,
        "n_pairs_checked": len(pairs),
        "n_disqualified": len(pairs) - len(scored_pairs),
        "n_scored_disjoint": len(scored_pairs) if all_disjoint else None,
    }


def evaluate_sc_gate(phase1: dict) -> dict:
    excluded = set(phase1.get("excluded_from_gates", []))
    acceptable = {"Panel-D_h63_F3:M0d", "Panel-D_h63_F3:M1", "Panel-D_h5_F3:M1"}
    unscoped = excluded - acceptable
    return {
        "fires": len(unscoped) == 0,
        "excluded_from_gates": sorted(excluded),
        "unscoped_failures": sorted(unscoped),
        "note": "excluded_from_gates are pre-disclosed, dated-amendment misses (see "
        "run_phase_1_15_shuffle_control.py docstring); they exclude specific Phase 3 "
        "comparisons, not whole arms, and do not themselves cause SC to fail.",
    }


def evaluate_ia_gate(phase2: dict, alpha: float) -> dict:
    yc = phase2["yield_curve"]
    trials = {
        "A1_dff_fwd126": yc.get("A1_dff_fwd126", {}).get("vs_best_baseline"),
        "A2_es_drawdown_fwd126": yc.get("A2_es_drawdown_fwd126", {}).get(
            "vs_best_baseline"
        ),
        # A3 excluded per pre-registration (underpowered).
    }
    fired_on = [
        name for name, entry in trials.items() if _significant_positive(entry, alpha)
    ]
    return {
        "fires": len(fired_on) > 0,
        "fired_on": fired_on,
        "trials_considered": list(trials),
    }


def evaluate_dimension_gate(phase2: dict, dimension_key: str, alpha: float) -> dict:
    """IT/IC: fires if ANY per-symbol A4 trial, the A5 spread, or (for
    term_structure) A6 clears significance."""
    ts = phase2["term_structure_and_carry"]
    fired_on = []
    for horizon in (21, 63):
        block = ts.get(f"A4_{dimension_key}_price_only_h{horizon}", {})
        for symbol, entry in block.items():
            if _significant_positive(entry.get("vs_best_baseline"), alpha):
                fired_on.append(f"A4_{dimension_key}_h{horizon}_{symbol}")
        spread = ts.get(f"A5_{dimension_key}_cross_sectional_h{horizon}", {}).get(
            "spread_top2_minus_bottom2"
        )
        if _significant_positive(spread, alpha):
            fired_on.append(f"A5_{dimension_key}_spread_h{horizon}")
    if dimension_key == "term_structure":
        a6 = ts.get("A6_cot_positioning_fwd21", {}).get("vs_best_baseline")
        if _significant_positive(a6, alpha):
            fired_on.append("A6_cot_positioning")
    return {"fires": len(fired_on) > 0, "fired_on": fired_on}


def evaluate_ic_roll_yield_only(phase2: dict, alpha: float) -> dict:
    """Reported separately, per sec4.2/sec7 -- never substitutes for IC."""
    ts = phase2["term_structure_and_carry"]
    fired_on = []
    for horizon in (21, 63):
        block = ts.get(f"A4_carry_roll_yield_only_price_only_h{horizon}", {})
        for symbol, entry in block.items():
            if _significant_positive(entry.get("vs_best_baseline"), alpha):
                fired_on.append(f"A4_carry_roll_yield_only_h{horizon}_{symbol}")
    return {"fires": len(fired_on) > 0, "fired_on": fired_on}


def evaluate_track_b_gate(
    phase3: dict, comparison_key: str, alpha: float, min_effect: float = 0.0
) -> dict:
    fired_on = []
    for combo_key, combo in phase3["combos"].items():
        if combo.get("underpowered"):
            continue
        entry = combo.get("comparisons", {}).get(comparison_key)
        if entry is None:
            continue
        if _significant_positive(entry, alpha, min_effect=min_effect):
            fired_on.append(combo_key)
    return {"fires": len(fired_on) > 0, "fired_on": fired_on}


def evaluate_pw_gate(phase0: dict) -> dict:
    budget = phase0["track_c"]["power_budget"]
    adequate = [k for k, v in budget.items() if not v["underpowered"]]
    return {
        "fires": len(adequate) > 0,
        "adequate_arms": adequate,
        "all_arms": list(budget),
    }


def main() -> None:
    phase0 = load("phase_0_15_preregistration.json")
    phase1 = load("phase_1_15_results.json")
    phase2 = load("phase_2_15_results.json")
    phase3 = load("phase_3_15_results.json")
    alpha = phase0["significance_procedure"]["alpha_bonferroni"]

    gates = {
        "SC": evaluate_sc_gate(phase1),
        "ID": evaluate_id_gate(phase0),
        "IA": evaluate_ia_gate(phase2, alpha),
        "IT": evaluate_dimension_gate(phase2, "term_structure", alpha),
        "IC": evaluate_dimension_gate(phase2, "carry", alpha),
        "IC_roll_yield_only_informational": evaluate_ic_roll_yield_only(phase2, alpha),
        "CW": evaluate_track_b_gate(phase3, "CW_M1_vs_M0d", alpha),
        "CC": evaluate_track_b_gate(phase3, "CC_M3_vs_M2", alpha),
        "CB": evaluate_track_b_gate(phase3, "CB_best_vs_M0d", alpha, min_effect=0.05),
        "M0c_vs_M0d_informational": evaluate_track_b_gate(phase3, "M0c_vs_M0d", alpha),
        "PW": evaluate_pw_gate(phase0),
    }

    # SC and ID are hard gates: if either fails, nothing else is
    # interpretable (NEXT_PROMPT.md sec7's "same sense NL was in 014").
    hard_gates_pass = gates["SC"]["fires"] and gates["ID"]["fires"]

    # The headline outcome table (sec10): CW/CC/CB null while SC passes is
    # the single most informative outcome available.
    outcome = {
        "hard_gates_pass": hard_gates_pass,
        "cw_fires": gates["CW"]["fires"],
        "cc_fires": gates["CC"]["fires"],
        "cb_fires": gates["CB"]["fires"],
        "pw_satisfied": gates["PW"]["fires"],
        "ia_fires": gates["IA"]["fires"],
        "it_fires": gates["IT"]["fires"],
        "ic_fires": gates["IC"]["fires"],
    }
    if not hard_gates_pass:
        outcome["authorization"] = (
            "SC or ID failed unscoped: nothing else is valid. Fix and re-run."
        )
    elif (
        not gates["CW"]["fires"]
        and not gates["CC"]["fires"]
        and not gates["CB"]["fires"]
        and gates["PW"]["fires"]
    ):
        outcome["authorization"] = (
            "Close the directional-trend line of enquiry. Record the MDE bound. Future notebooks "
            "may use trend labels as descriptive context but may not condition a directional "
            "strategy on them."
        )
    elif gates["CW"]["fires"] and not gates["CC"]["fires"] and not gates["CB"]["fires"]:
        outcome["authorization"] = (
            "The shipped config's weights are the constraint, not the concept. Consider porting "
            "sweep.py as its own notebook."
        )
    elif gates["CC"]["fires"] and gates["CB"]["fires"]:
        outcome["authorization"] = (
            "A genuine non-linear lead. Notebook 016 builds a cost-aware, execution-aware "
            "backtest under the usual gate discipline, still on dev data; the holdout stays shut "
            "until a dev gate fires."
        )
    else:
        outcome["authorization"] = (
            "Mixed result -- see the gate table and results write-up for detail."
        )

    if gates["IA"]["fires"] or gates["IT"]["fires"] or gates["IC"]["fires"]:
        outcome["track_a_verdict"] = (
            "At least one of yield_curve/term_structure/carry is validated for conditioning; "
            "014's Phase 5 verdict table updates from 'No, without further work' to conditionally yes."
        )
    else:
        outcome["track_a_verdict"] = (
            "014's high-accuracy dimensions were arithmetic. Update the verdict table to 'No' "
            "outright and stop citing 0.981 and 0.87 anywhere."
        )

    results = {"alpha_bonferroni": alpha, "gates": gates, "outcome": outcome}
    with open(f"{TMP}/phase_4_15_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    for name, g in gates.items():
        print(f"{name}: fires={g['fires']}")
    print("\nOutcome:", outcome["authorization"])
    print("Track A verdict:", outcome["track_a_verdict"])
    print("Wrote phase_4_15_results.json")


if __name__ == "__main__":
    main()
