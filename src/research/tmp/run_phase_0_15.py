"""Notebook 015 Phase 0: data introspection, Track A disjointness table
(gate ID), Track C power budget (N_eff / MDE), and pre-registration
(NEXT_PROMPT.md sec4.1, sec6.3, sec7). Writes phase_0_15_results.json and
phase_0_15_preregistration.json, committed before any model is fit or any
Track A target is scored, and never edited afterward.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "src/research/tmp")

import lib15 as lib

TMP = "src/research/tmp"

N_TRIALS_TRACK_A = (
    16  # sec7: yield_curve(3) + term_structure(5) + carry(4+4 roll-yield-only variant)
)
N_TRIALS_TRACK_B = (
    24  # sec9: 3 horizons x 2 panels x 4 trials/combo (M0c_vs_M0d, CW, CC, CB)
)
N_TRIALS_TOTAL = N_TRIALS_TRACK_A + N_TRIALS_TRACK_B  # 40
ALPHA_FAMILY = 0.05
ALPHA_BONFERRONI = ALPHA_FAMILY / N_TRIALS_TOTAL

TRACK_A_TRIAL_LEDGER = [
    {
        "dimension": "yield_curve",
        "target": "A1_dff_fwd126",
        "counts_toward_gate_IA": True,
    },
    {
        "dimension": "yield_curve",
        "target": "A2_es_drawdown_fwd126",
        "counts_toward_gate_IA": True,
    },
    {
        "dimension": "yield_curve",
        "target": "A3_hy_oas_fwd63",
        "counts_toward_gate_IA": False,
        "reason": "underpowered: BAMLH0A0HYM2 starts 2023-07-17, ~18mo after truncation to 2024-12-31",
    },
    {
        "dimension": "term_structure",
        "target": "A4_front_month_return",
        "horizon": 21,
        "counts_toward_gate_IT": True,
    },
    {
        "dimension": "term_structure",
        "target": "A4_front_month_return",
        "horizon": 63,
        "counts_toward_gate_IT": True,
    },
    {
        "dimension": "term_structure",
        "target": "A5_cross_sectional_spread",
        "horizon": 21,
        "counts_toward_gate_IT": True,
    },
    {
        "dimension": "term_structure",
        "target": "A5_cross_sectional_spread",
        "horizon": 63,
        "counts_toward_gate_IT": True,
    },
    {
        "dimension": "term_structure",
        "target": "A6_cot_positioning",
        "horizon": 21,
        "counts_toward_gate_IT": True,
    },
    {
        "dimension": "carry",
        "target": "A4_front_month_return",
        "horizon": 21,
        "counts_toward_gate_IC": True,
    },
    {
        "dimension": "carry",
        "target": "A4_front_month_return",
        "horizon": 63,
        "counts_toward_gate_IC": True,
    },
    {
        "dimension": "carry",
        "target": "A5_cross_sectional_spread",
        "horizon": 21,
        "counts_toward_gate_IC": True,
    },
    {
        "dimension": "carry",
        "target": "A5_cross_sectional_spread",
        "horizon": 63,
        "counts_toward_gate_IC": True,
    },
    {
        "dimension": "carry_roll_yield_only",
        "target": "A4_front_month_return",
        "horizon": 21,
        "counts_toward_gate_IC": False,
        "reason": "measurement variant, reported alongside shipped config, not instead of",
    },
    {
        "dimension": "carry_roll_yield_only",
        "target": "A4_front_month_return",
        "horizon": 63,
        "counts_toward_gate_IC": False,
        "reason": "measurement variant",
    },
    {
        "dimension": "carry_roll_yield_only",
        "target": "A5_cross_sectional_spread",
        "horizon": 21,
        "counts_toward_gate_IC": False,
        "reason": "measurement variant",
    },
    {
        "dimension": "carry_roll_yield_only",
        "target": "A5_cross_sectional_spread",
        "horizon": 63,
        "counts_toward_gate_IC": False,
        "reason": "measurement variant",
    },
]
assert len(TRACK_A_TRIAL_LEDGER) == N_TRIALS_TRACK_A

TRACK_B_MODELS = [
    {
        "id": "M0a",
        "model": "majority class (per fold, from train)",
        "features": None,
        "purpose": "floor",
    },
    {
        "id": "M0b",
        "model": "persistence of yesterday's realised sign",
        "features": None,
        "purpose": "trivial trend rule",
    },
    {
        "id": "M0c",
        "model": "sign of trailing 60-day return",
        "features": None,
        "purpose": "dumbest defensible trend rule",
    },
    {
        "id": "M0d",
        "model": "engine's trend label (F0)",
        "features": "F0",
        "purpose": "incumbent -- the number to beat",
    },
    {
        "id": "M1",
        "model": "L2-regularized logistic regression",
        "features": "F1",
        "purpose": "are the shipped weights the bottleneck?",
    },
    {
        "id": "M2",
        "model": "L2-regularized logistic regression",
        "features": "F2 (F3 on Panel-D)",
        "purpose": "does more information help a linear model?",
    },
    {
        "id": "M3",
        "model": "HistGradientBoostingClassifier",
        "features": "F2 (F3 on Panel-D)",
        "purpose": "does capacity help?",
    },
]
M3_FIXED_HYPERPARAMS = {
    "max_iter": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_samples_leaf": 200,
    "l2_regularization": 1.0,
    "early_stopping": True,
}
FEATURE_SETS = {
    "F0": "the engine's own trend label for that sector/date, read from regime_panel.parquet",
    "F1": [
        "trend.price_vs_ma (window 200, zscore 252)",
        "trend.ma_slope (100/20, zscore 252)",
        "trend.nday_log_return (n 60, zscore 252)",
        "trend.adx (14, linear hr 50)",
        "trend.efficiency_ratio (20, raw)",
    ],
    "F2": "F1 + volatility(3) + mean_reversion(3) + [curve-available: term_structure(4), carry(2)] "
    "+ cross-sectional rank (within basket, within panel) on each F1 feature + calendar sin/cos",
    "F3": "F2 + Panel-D per-date front-three settlement spreads and days-to-expiry",
}
HORIZONS = [5, 21, 63]
FOLD_GEOMETRY = {
    "min_train_bars": 1260,
    "test_bars": 252,
    "step_bars": 252,
    "expanding": True,
    "purge_bars": "= horizon (drops final `horizon` rows of every training window)",
    "embargo_bars": "= horizon (drops first `horizon` rows of every test window)",
    "panel_l_expected_folds": 19,
    "panel_d_expected_folds": 9,
}
SHUFFLE_CONTROL = {
    "block_size_days": 63,
    "n_seeds": 10,
    "tolerance": "95% bootstrap CI over balanced accuracy covers 0.500 for every model x horizon x panel",
    "hard_gate": True,
}
SIGNIFICANCE_PROCEDURE = {
    "test": "paired block bootstrap on the daily correctness/hit-rate difference (engine or "
    "challenger minus baseline), quarterly blocks for Track A, n_boot=2000, seed=0, via "
    "research.block_bootstrap_ci and research.block_bootstrap_pvalue",
    "family": "Track A and Track B trials pool into ONE Bonferroni family",
    "n_trials_track_a": N_TRIALS_TRACK_A,
    "n_trials_track_b": N_TRIALS_TRACK_B,
    "n_trials_total": N_TRIALS_TOTAL,
    "alpha_family": ALPHA_FAMILY,
    "alpha_bonferroni": ALPHA_BONFERRONI,
    "baseline_degeneracy_check": "for every Track A comparison, report the fraction of days the "
    "baseline and engine label differ; below 0.05, report as structurally uninformative "
    "rather than as a null (014's discovery, pre-registered here)",
}

GATES = {
    "SC": {
        "track": "B",
        "claim": "Shuffle control: no model beats chance on block-shuffled targets",
        "threshold": "every model x horizon x panel has a 95% CI covering 0.500",
        "hard_gate": True,
    },
    "ID": {
        "track": "A",
        "claim": "Independence proof: every scored (dimension, target) pair has provably disjoint raw inputs",
        "threshold": "INPUTS(dim) intersect INPUTS(target) == empty, table emitted",
        "hard_gate": True,
    },
    "IA": {
        "track": "A",
        "claim": "yield_curve beats its best baseline on >=1 independent target",
        "threshold": f"Bonferroni-corrected p < {ALPHA_BONFERRONI:.5f}, excluding underpowered target A3",
    },
    "IT": {
        "track": "A",
        "claim": "term_structure beats its best baseline on >=1 independent target",
        "threshold": f"Bonferroni-corrected p < {ALPHA_BONFERRONI:.5f}, arm not flagged underpowered",
    },
    "IC": {
        "track": "A",
        "claim": "carry beats its best baseline on >=1 independent target",
        "threshold": "as IT; report shipped-config and roll-yield-only variants separately",
    },
    "CW": {
        "track": "B",
        "claim": "Weights were the bottleneck: M1 (same inputs, learned weights) beats M0d",
        "threshold": f"Bonferroni-corrected p < {ALPHA_BONFERRONI:.5f} at >=1 non-underpowered horizon",
    },
    "CC": {
        "track": "B",
        "claim": "Capacity helps: M3 beats M2",
        "threshold": f"Bonferroni-corrected p < {ALPHA_BONFERRONI:.5f} at >=1 non-underpowered horizon",
    },
    "CB": {
        "track": "B",
        "claim": "Ceiling beaten: best model beats M0d by >= +0.05 balanced accuracy",
        "threshold": f"Bonferroni-corrected p < {ALPHA_BONFERRONI:.5f}, and point estimate clears +0.05",
    },
    "PW": {
        "track": "C",
        "claim": "Power is adequate to conclude: >=1 arm has N_eff >= 200",
        "threshold": "reported per arm; determines which gates are eligible to fire",
    },
}

SCORING = {
    "metric": "balanced_accuracy (regime.forecast_eval.balanced_accuracy), plus Cohen's kappa, plus "
    "confusion matrix / per_class_stats for every reported pair",
    "baselines": ["persistence_forecast", "markov_forecast", "prior_forecast"],
}

SIDEWAYS_MAPPING = {
    "rule": "bear -> -1, bull -> +1, sideways -> abstain (scored as the class prior on those days, "
    "not a coin flip); abstention rate reported; sideways days are never dropped from the panel",
    "why": "Commodities/trend sits in sideways 90.24% of the time (014 Phase 2); dropping sideways "
    "days would score the incumbent on a cherry-picked 10% subsample and flatter it enormously",
}


def build_preregistration(disjointness: dict, power_budget: dict) -> dict:
    return {
        "notebook": "015_trend_ceiling_and_independent_validation",
        "committed_before_first_model_fit": True,
        "supersedes": "none -- extends 014's Phase 3/5 findings without reopening them",
        "scope": {
            "authorizes_trading": False,
            "spends_holdout": False,
            "holdouts_untouched": {
                "crypto": "2025-07-01",
                "futures": "2025-01-01 to 2026-07-28",
            },
            "truncation": str(lib.TRUNCATION.date()),
            "note": "This notebook measures label accuracy and a ceiling on directional "
            "predictability. It builds no strategy, computes no Sharpe, models no costs. "
            "A firing gate does not authorize a backtest here -- that is notebook 016's job.",
        },
        "track_a": {
            "dimensions_tested": [
                "yield_curve",
                "term_structure",
                "carry",
                "carry_roll_yield_only",
            ],
            "disjointness_table": disjointness,
            "trial_ledger": TRACK_A_TRIAL_LEDGER,
        },
        "track_b": {
            "panels": {
                "Panel-L": {
                    "source": "yfinance daily",
                    "symbols": lib.PANEL_L_SYMBOLS,
                    "span": "2000 to 2024-12-31",
                },
                "Panel-D": {
                    "source": "databento per-contract",
                    "symbols": lib.PANEL_D_SYMBOLS,
                    "span": "2010-06 to 2024-12-31",
                    "note": "ES dropped for the trend target (equity index, not a commodity)",
                },
            },
            "horizons": HORIZONS,
            "feature_sets": FEATURE_SETS,
            "models": TRACK_B_MODELS,
            "m3_fixed_hyperparameters": M3_FIXED_HYPERPARAMS,
            "fold_geometry": FOLD_GEOMETRY,
            "shuffle_control": SHUFFLE_CONTROL,
            "sideways_mapping": SIDEWAYS_MAPPING,
            "zero_return_rule": "drop rows where |forward return| < 1e-12 (frozen-bar rule)",
            "neural_model_policy": "no M4/MLP unless CC fires with a Bonferroni-significant margin; "
            "torch is present but not used by default (013's LSTM/GatedLSTM already lost to linear)",
            "no_grid_search_m3": True,
        },
        "track_c": {
            "power_budget": power_budget,
            "underpowered_rule": "any (panel, horizon) arm with N_eff < 200 is reported as "
            "underpowered and cannot fire a gate, regardless of the accuracy it produces",
        },
        "significance_procedure": SIGNIFICANCE_PROCEDURE,
        "gates": GATES,
        "scoring": SCORING,
        "outcome_authorization_table": "see NEXT_PROMPT.md sec10 -- reproduced verbatim in the "
        "results file, filled in with actual outcomes",
    }


def main() -> None:
    print("Building Track A disjointness table (gate ID)...")
    disjointness = lib.build_disjointness_table()
    n_disqualified = sum(1 for p in disjointness["pairs"] if p["disqualified"])
    print(
        f"  {len(disjointness['pairs'])} (dimension, target) pairs checked, "
        f"{n_disqualified} disqualified by non-empty intersection"
    )

    print("Computing Track C power budget (N_eff, MDE) for Panel-L and Panel-D...")
    power_budget = lib.track_c_power_budget(
        horizons=tuple(HORIZONS), alpha=ALPHA_BONFERRONI
    )
    for key, row in power_budget.items():
        print(
            f"  {key}: N_eff={row['n_eff']} underpowered={row['underpowered']} "
            f"MDE={row['minimum_detectable_effect_balanced_accuracy']}"
        )

    results = {
        "disjointness_table": disjointness,
        "power_budget": power_budget,
        "n_trials": {
            "track_a": N_TRIALS_TRACK_A,
            "track_b": N_TRIALS_TRACK_B,
            "total": N_TRIALS_TOTAL,
            "alpha_bonferroni": ALPHA_BONFERRONI,
        },
    }
    with open(f"{TMP}/phase_0_15_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    prereg = build_preregistration(disjointness, power_budget)
    with open(f"{TMP}/phase_0_15_preregistration.json", "w") as f:
        json.dump(prereg, f, indent=2, default=str)

    print(
        f"n_trials: track_a={N_TRIALS_TRACK_A} track_b={N_TRIALS_TRACK_B} total={N_TRIALS_TOTAL}"
    )
    print(f"alpha_bonferroni = 0.05/{N_TRIALS_TOTAL} = {ALPHA_BONFERRONI:.6f}")
    print("Wrote phase_0_15_results.json and phase_0_15_preregistration.json")

    # ID hard gate: every scored pair (i.e. everything NOT flagged
    # disqualified) must have an empty intersection.
    for pair in disjointness["pairs"]:
        if not pair["disqualified"]:
            assert not (set(pair["dimension_inputs"]) & set(pair["target_inputs"]))
    print("Gate ID: all non-disqualified pairs are provably disjoint. PASS")


if __name__ == "__main__":
    main()
