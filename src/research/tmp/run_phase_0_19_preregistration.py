"""Notebook 019 Phase 0 (NEXT_PROMPT.md sec 4, row "0").

Freezes sec 1-3 verbatim before any new Monte Carlo runs (Phase 2 onward):
the V3 candidate definition, the two tau candidates, what this notebook does
and does not claim, the two objectives, and all six gates (DS-2/DS-3 reused
from 017, DS-5/DS-6/DS-7/DS-8 new). Re-verifies (does not recompute) 018's
0.8316743360550332 rho->1 upper bound from 017's own stored addendum, and
folds in both preflight scripts' output (re-run fresh, not the possibly
stale files already on disk -- sec 0's disclosure must reflect what Phase 0
actually saw).

Committed before Phase 2 runs. Not editable afterward.

Usage: uv run python src/research/tmp/run_phase_0_19_preregistration.py
"""

from __future__ import annotations

import json

PREFLIGHT_CERT_PATH = "scratch/019/preflight.json"
PREFLIGHT_SWITCH_PATH = "scratch/019/preflight_switch_probe_rerun.json"
ADDENDUM_18_PATH = "src/research/tmp/phase_7_18_dsr_addendum.json"
OUT_PATH = "src/research/tmp/phase_0_19_preregistration.json"

EXPECTED_018_UPPER_BOUND = 0.8316743360550332

CANDIDATE = {
    "name": "V3 -- correlation-thresholded switch",
    "definition": (
        "V3(sharpe, n_trials, n_obs, skew, kurtosis, trial_sharpes, "
        "mean_pairwise_corr; tau): if mean_pairwise_corr <= tau, return "
        "dsr_variant(..., variant='v0'); else return dsr_variant(..., "
        "variant='v1', trial_sharpes=trial_sharpes). The boundary "
        "(mean_pairwise_corr == tau exactly) resolves to V0 -- sec 3.2/DS-5."
    ),
    "implementation": "dsr_lib17.dsr_variant(variant='v3', tau=..., mean_pairwise_corr=...)",
    "extra_input_needed": (
        "mean_pairwise_corr, via dsr_lib17.mean_pairwise_corr_estimate -- the "
        "same O(N*T) estimator V2 already uses (017 sec 2.3: V2 ranked last "
        "partly because most stored artifacts keep summary Sharpes, not trial "
        "return series). V3 inherits that limitation in full."
    ),
}

TAU_CANDIDATES = {
    "primary": 0.15,
    "fallback": 0.30,
    "rule": (
        "tau=0.15 primary, tau=0.30 fallback. No third value under any "
        "outcome. Both are carried through the whole notebook (sec 0.3: "
        "DS-2/DS-3 classify every measured cell identically at both taus, so "
        "the adoption decision cannot distinguish them -- the interesting "
        "comparison is the rho=0.25 column and the false-trigger rate, not "
        "the verdict)."
    ),
}

WHAT_THIS_NOTEBOOK_DOES_NOT_CLAIM = [
    (
        "Not that V3 fixes the high-rho power shortfall (sec 0.2 proves it "
        "cannot: for any tau<0.9, V3 IS V1 bit for bit in every cell DS-3's "
        "high-rho clause reads)."
    ),
    "Not a re-opening of 017's V1/V1b/V2 adoption decision, which is closed.",
    (
        "Not applicable to 018 under any outcome -- 018's ceiling is "
        f"{EXPECTED_018_UPPER_BOUND}, re-verified below, below the 0.95 bar, and "
        "017 Test 7 proves psr_upper_bound dominates every dispersion-based "
        "variant including V3's V1 branch."
    ),
    "Not a fix for sequential/nested search dependence (017 sec 12.5).",
]

OBJECTIVES = {
    "A_full_scope_adoption": {
        "definition": "adopt V3 at full scope at the first tau in (0.15, "
        "0.30) passing DS-2, DS-3, DS-5, DS-6.",
        "pre_registered_prediction": "fails at both tau -- DS-3's high-rho "
        "clause is unsatisfiable by any tau<0.9 (sec 0.2), and V3 inherits it "
        "verbatim from V1.",
        "falsifiable": True,
    },
    "B_validated_regime_adoption": {
        "definition": "adopt V3 restricted to N>=12 and T>=3840 iff DS-5, "
        "DS-6, DS-7, and DS-8 all fire, at tau=0.15 (tau=0.30 only if 0.15 "
        "fails DS-6, per sec 1.2).",
        "pre_registered_prediction": "exploratory-until-confirmed. sec 0.4's "
        "in-sample scan of 017's own certificate shows V3(tau=0.15) passing "
        "all four gate clauses down to N>=12 and T>=3840 where V1 needed "
        "N>=95 -- an 8x wider range -- but this was found by looking at "
        "already-visible results and is not adoptable on that evidence "
        "alone. Phase 4's C1 confirmation grid (points 017 never ran) is "
        "the out-of-sample test.",
        "falsifiable": True,
    },
}

GATES = {
    "DS-2": {
        "name": "calibrated (reused verbatim from 017)",
        "sub_gates": {
            "DS-2a": "FPR <= 0.075 in every null cell",
            "DS-2b": "FPR >= 0.010 in every null cell with rho >= 0.5",
        },
        "reuse": "run_phase_4_17_adoption.evaluate_variant, imported as a "
        "function, not reimplemented (sec 3.1). Thresholds not retuned "
        "(017 sec 12.3).",
        "mechanical_fix": "evaluate_variant truncates violation lists at "
        "20; 019's wrapper additionally records the uncapped counts (sec "
        "0.5/sec 3.1) without editing 017's script.",
    },
    "DS-3": {
        "name": "power (reused verbatim from 017)",
        "fires_if": [
            ("detection rate exceeds V0's by >= 10pp in every cell with rho >= 0.9"),
            (
                "detection rate is not below V0's by more than 2pp in any cell "
                "with rho = 0"
            ),
        ],
        "known_result": "no tau < 0.9 can ever pass the high-rho clause "
        "(sec 0.2, provable not empirical): V3 IS V1 bit for bit in every "
        "cell that clause reads, and V1 fails it. V3 DOES fix the rho=0 "
        "clause completely (sec 0.1: 20 violations -> 0, deterministically, "
        "measured on 017's own certificate).",
    },
    "DS-5": {
        "name": "the switch reduces correctly at its boundary",
        "kind": "unit-test gate, not statistical",
        "fires_if": "dsr_variant(variant='v3') returns output identical to "
        "the V0 branch whenever mean_pairwise_corr < tau, identical to the "
        "V1 branch whenever it is >= tau, and resolves the exact-equality "
        "boundary to the V0 branch",
        "evaluated_on": "tests/test_dsr_lib17.py: "
        "test_v3_reduces_to_v0_below_threshold, "
        "test_v3_reduces_to_v1_above_threshold, "
        "test_v3_boundary_resolves_to_v0. Fires iff all three pass.",
        "note": "sec 3.2: this is a property of the function, not of "
        "whether rho=0 cells actually take the V0 branch -- that is DS-6.",
    },
    "DS-6": {
        "name": "the switch actually engages where it should",
        "fires_if": [
            (
                "P(estimated correlation >= tau | true rho = 0) <= 0.005 at "
                "every Phase 2 design point"
            ),
            (
                "V3's rho=0 detection-rate shortfall against V0 is exactly "
                "zero in every confirmed (Phase 4) cell"
            ),
        ],
        "evaluated_on": "Phase 2 (first clause), Phase 4 (second clause)",
        "preflight_signal": "the M=200 preflight probe below found 0/200 "
        "everywhere at true rho=0 across all 63 (N,T,moments) points -- "
        "Phase 2 at M=2000 should confirm <=0.005, but a nonzero rate must "
        "be disclosed and quantified, never buried.",
    },
    "DS-7": {
        "name": "the prediction is accurate (the load-bearing new gate)",
        "mechanism": "Phase 3 predicts every predictable cell's V3 rate as "
        "p*rate_v1 + (1-p)*rate_v0 from Phase 2's measured branch "
        "probability p and 017's stored V0/V1 rates. Only cells inside "
        "017's original 756-cell grid are predictable this way (C2 and "
        "C3); C1's axis values (T=5000, rho=0.95) do not exist in 017's "
        "certificate, so C1 carries no prediction to check DS-7 against "
        "and is scored by DS-8 instead.",
        "fires_if": "predicted and measured agree within 3 combined MC "
        "standard errors in >=95% of confirmation cells (C2 union C3), and "
        "no disagreement exceeds 5 SE",
        "if_it_does_not_fire": "the prediction mechanism is wrong, sec "
        "0.1's verdict is not established, and the notebook must say so "
        "plainly rather than fall back on the prediction anyway.",
    },
    "DS-8": {
        "name": "the restricted regime survives out of sample",
        "fires_if": "across Phase 4's C1 cells (grid points 017 never ran) "
        "that lie inside N>=12 and T>=3840, V3 has zero violations of "
        "DS-2a, DS-2b, DS-3-high-rho, and DS-3-rho0",
        "note": "the only evidence on which Objective B may be adopted",
    },
    "MC_SE_note": "every reported rate carries its MC standard error; a "
    "gate comparison inside the stated SE multiple is not a difference.",
}

ADOPTION_RULES = {
    "objective_A": "adopt V3 at full scope at the first tau in (0.15, "
    "0.30) passing DS-2, DS-3, DS-5, DS-6. Predicted to fail at both.",
    "objective_B": "adopt V3 restricted to N>=12 and T>=3840 iff DS-5, "
    "DS-6, DS-7, and DS-8 all fire, at tau=0.15 (tau=0.30 only if 0.15 "
    "fails DS-6).",
    "neither": "research.py stays exactly as 017 left it and the write-up "
    "says so -- a second no-adoption outcome is a legitimate, publishable "
    "result.",
    "frozen_boundary": "the regime boundary N>=12 and T>=3840 is frozen "
    "now. It may not be widened, narrowed, or shifted after Phase 4's "
    "numbers are visible. If N>=12 fails out of sample, the answer is "
    "'the in-sample boundary did not replicate,' not 'try N>=18.'",
}

CONFIRMATION_GRIDS = {
    "C1_out_of_sample": {
        "purpose": "the out-of-sample evidence; scores DS-8",
        "axes": {
            "N": [12, 14, 24, 50],
            "T": [5000],
            "rho": [0.0, 0.5, 0.9, 0.95],
            "moments": ["gaussian", "moderate_nongaussian", "018_measured"],
            "mode": ["null", "edge"],
        },
        "n_cells": 96,
        "M": 20000,
        "note": "N=12/T=5000 sit on/just beyond the claimed regime "
        "boundary; rho=0.95 and T=5000 are values 017's axes never "
        "contained, so no cell here is a re-read of stored numbers.",
        "predicted_cost_hours_single_core": 2.1,
    },
    "C2_ambiguous": {
        "purpose": "the ambiguous points where the prediction is "
        "non-trivial; scores DS-7 hardest",
        "definition": "every 017-grid design point where Phase 2 measures "
        "p in (0.02, 0.98), both modes, M=20000",
        "truncation_rule": "if Phase 2 finds more than 60 such points, "
        "take the 60 with p closest to 0.5 and disclose the truncation",
        "M": 20000,
    },
    "C3_deterministic_control": {
        "purpose": "a control sample where DS-7 should be easy",
        "definition": "24 cells drawn from 017's own 756 with "
        "numpy.random.default_rng(19), excluding N>=95 and T=3840 for "
        "cost, both tau",
        "seed_discipline": "seed derivation must differ from 017's own "
        "seed_for_cell(key) for the identical (N,T,rho,moments,mode) "
        "tuple, so these are genuinely independent replications, not a "
        "re-run of 017's exact random draws (implemented by tagging the "
        "seed key with a 019-specific string, disclosed in "
        "build_phase4_grid_19.py).",
        "M": 20000,
        "predicted_cost_minutes": 30,
    },
    "not_run": "the full 756-cell V3 grid. sec 0.1-0.2 establish its result "
    "analytically; DS-7 is what licenses using the prediction instead of "
    "the measurement. Do not add it back (sec 5.2/sec 9).",
    "total_budget_hours_single_core": 3,
}

SCOPE_DISCIPLINE = [
    (
        "017's five variants, gates, thresholds, and adoption verdict are "
        "frozen. The only permitted touch is the sec 8.3 erratum."
    ),
    (
        "018's construction, gate verdicts, and holdout are frozen. This "
        "notebook cannot grant holdout access under any outcome."
    ),
    (
        "No third tau. No moving the N>=12 and T>=3840 boundary after results "
        "are visible. No widening DS-2/DS-3."
    ),
    "Do not add the full 756-cell V3 grid back.",
    "One notebook. No 020 work, no dashboard changes, no productionisation.",
]


def _verify_018_ceiling() -> dict:
    with open(ADDENDUM_18_PATH) as f:
        addendum = json.load(f)
    upper_bound = addendum["018_own_row_in_phase_5_rescore"]["upper_bound"]
    assert abs(upper_bound - EXPECTED_018_UPPER_BOUND) < 1e-12, (
        f"018's stored rho->1 upper bound has changed: {upper_bound} != "
        f"{EXPECTED_018_UPPER_BOUND}. This notebook must not proceed if "
        "018's frozen ceiling has moved."
    )
    return {
        "source": ADDENDUM_18_PATH,
        "json_pointer": "/018_own_row_in_phase_5_rescore/upper_bound",
        "value": upper_bound,
        "below_0.95_bar": upper_bound < 0.95,
        "conclusion": (
            "Re-verified, not recomputed: 018's ceiling stands at "
            f"{upper_bound}, below the 0.95 bar. 018's case is closed "
            "regardless of anything this notebook finds (sec 1.3)."
        ),
    }


def main() -> None:
    with open(PREFLIGHT_CERT_PATH) as f:
        preflight_cert = json.load(f)
    with open(PREFLIGHT_SWITCH_PATH) as f:
        preflight_switch = json.load(f)

    verification_018 = _verify_018_ceiling()

    doc = {
        "notebook": "019_dsr_correlation_switch",
        "supersedes": "the prior draft of NEXT_PROMPT.md in full -- that "
        "draft pre-registered a full 756-cell V3 grid whose result sec "
        "0.1-0.2 show is analytically foregone; this rewrite fixes that "
        "design flaw (sec 0's own disclosure).",
        "committed_before_phase_2_runs": True,
        "editable_after_commit": False,
        "candidate": CANDIDATE,
        "tau_candidates": TAU_CANDIDATES,
        "what_this_notebook_does_not_claim": WHAT_THIS_NOTEBOOK_DOES_NOT_CLAIM,
        "objectives": OBJECTIVES,
        "gates": GATES,
        "adoption_rules": ADOPTION_RULES,
        "confirmation_grids": CONFIRMATION_GRIDS,
        "scope_discipline": SCOPE_DISCIPLINE,
        "notebook_018_ceiling_reverification": verification_018,
        "preflight_disclosure": {
            "note": "sec 0's full disclosure of what was already visible "
            "before any Phase 2+ Monte Carlo ran, per the pre-registration "
            "discipline this repo uses (017's own Phase 0 pilot precedent). "
            "Both preflight scripts were re-run fresh for this file, not "
            "read stale off disk.",
            "from_017_certificate": preflight_cert,
            "switch_activation_probe": {
                "what": preflight_switch["what"],
                "n_reps_per_design_point": preflight_switch["n_reps_per_design_point"],
                "n_design_points": preflight_switch["n_design_points"],
                "rho0_false_trigger": preflight_switch["rho0_false_trigger"],
                "max_sd_of_estimate_at_true_rho0": preflight_switch[
                    "max_sd_of_estimate_at_true_rho0"
                ],
            },
        },
    }

    with open(OUT_PATH, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"written {OUT_PATH}")
    print("018 ceiling:", verification_018["conclusion"])
    print(
        "preflight rho=0 false-trigger (M=200):",
        preflight_switch["rho0_false_trigger"],
    )


if __name__ == "__main__":
    main()
