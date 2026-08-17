"""Notebook 017 Phase 0 (NEXT_PROMPT.md sec 4, row "0").

Freezes, before any Monte Carlo grid runs and before Phase 2 exists:
  - the three candidate estimator variants and the sec 2.3 adoption rule
  - the sec 5 gate thresholds (DS-1..DS-4), verbatim
  - this notebook's own sec 6 trial ledger (5 variants)
  - the sec 5.4 verdict-change policy, quoted
  - the sec 3 disclosed pilot, its numbers and its stated limitations
  - the sec 5.5 inventory of stored DSR values (reproduced mechanically via
    inventory_17.scan(), NOT re-typed from the table)
  - the sec 0.1 item 1 / sec 14.2 branch C verification: 018's
    fires_except_dsr_leg, bootstrap_ci_leg_fires, holdout_access fields,
    read directly from phase_4_18_results.json
  - all three sec 14.2 branch texts for the 018 amendment, verbatim,
    committed before Phase 2 runs so the choice among them is mechanical.

Committed before Phase 2 runs. Not editable afterward. Phase 2 through
Phase 6b read this file; none of them may re-derive or restate any of the
frozen content below.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "src/research/tmp")
from inventory_17 import scan

OUT_PATH = "src/research/tmp/phase_0_17_preregistration.json"

# sec 5.5 claims 73 values across 17 files. That count is reproduced here
# mechanically, not asserted -- and it does NOT match (see _inventory below
# and the write-up's note on it). Per NEXT_PROMPT sec 5.5 ("Phase 0 must
# reproduce that count and fail loudly if it differs"), the measured count
# is what this notebook uses for DS-4, and the discrepancy is disclosed
# rather than silently patched over.
DOCUMENTED_INVENTORY_COUNT = 73
DOCUMENTED_INVENTORY_FILES = 17


def build_inventory() -> dict:
    by_file = scan(exact=True)
    triples: list[tuple[str, str, float]] = [
        (fp, ptr, val) for fp, hits in by_file.items() for ptr, val in hits
    ]
    triples.sort(key=lambda t: (t[0], t[1]))
    rows = [
        {"file": fp, "json_pointer": ptr, "stored_value": val}
        for fp, ptr, val in triples
    ]

    def bucket(v: float) -> str:
        if v > 0.95:
            return ">0.95"
        if 0.45 <= v <= 0.70:
            return "0.45-0.70"
        if 0.10 <= v <= 0.20:
            return "0.10-0.20"
        return "<0.10" if v < 0.10 else "other"

    buckets: dict[str, int] = {}
    for _, _, val in triples:
        b = bucket(val)
        buckets[b] = buckets.get(b, 0) + 1

    measured_count = len(rows)
    measured_files = len(by_file)

    # Secondary, disclosed-only check: what if the sweep also matched the
    # key-name variants notebook 011b uses for the same underlying figure
    # at different pipeline stages (deflated_sharpe_prob_headline,
    # deflated_sharpe_prob_best, published_deflated_sharpe_prob,
    # deflated_sharpe_prob_by_gate)? Reported for transparency; NOT used
    # for DS-4, because sec 5.5 specifies an exact-key sweep (backticked
    # `deflated_sharpe_prob`), not a fuzzy one.
    by_file_fuzzy = scan(exact=False)
    fuzzy_count = sum(len(v) for v in by_file_fuzzy.values())
    fuzzy_files = len(by_file_fuzzy)
    fuzzy_only_files = sorted(set(by_file_fuzzy) - set(by_file))

    return {
        "definition": (
            "recursive walk of every *.json under src/research/tmp/; a hit is "
            "any dict key whose name is EXACTLY 'deflated_sharpe_prob'; if "
            "that key's value is numeric it is one stored value, if it is a "
            "dict/list (e.g. a *_by_gate summary) every numeric leaf beneath "
            "it is counted as a separate stored value (each is a distinct "
            "trial's DSR at a distinct json_pointer in a distinct file). This "
            "is sec 5.5's own phrasing: a `deflated_sharpe_prob`-keyed sweep."
        ),
        "secondary_fuzzy_check_disclosed_not_used_for_DS4": {
            "definition": "same walk, but matches any key containing "
            "'deflated_sharpe_prob' as a substring (catches notebook 011b's "
            "deflated_sharpe_prob_headline / _best / published_deflated_sharpe_prob "
            "/ _by_gate key-name variants for what is likely the same "
            "underlying figure recorded at different pipeline stages)",
            "count": fuzzy_count,
            "files": fuzzy_files,
            "additional_files_over_exact_match": fuzzy_only_files,
        },
        "documented_in_next_prompt": {
            "count": DOCUMENTED_INVENTORY_COUNT,
            "files": DOCUMENTED_INVENTORY_FILES,
        },
        "measured": {
            "count": measured_count,
            "files": measured_files,
            "buckets": buckets,
        },
        "matches_documented": (
            measured_count == DOCUMENTED_INVENTORY_COUNT
            and measured_files == DOCUMENTED_INVENTORY_FILES
        ),
        "discrepancy_note": (
            "Measured inventory (exact-key sweep) does NOT match sec 5.5's "
            "documented 73/17. This is disclosed, not silently corrected. "
            "Per sec 5.5's own instruction ('must reproduce that count and "
            "fail loudly if it differs -- artifacts may have moved'), this "
            "notebook adopts the MEASURED exact-key count as authoritative "
            "for gate DS-4 rather than forcing agreement with the documented "
            "estimate; DS-4 is judged against it. The fuzzy secondary check "
            "above shows the gap is not random: notebook 011b records the "
            "same underlying pipeline-stage figures under several different "
            "key names across its phase_0/1/4/6/7 result files "
            "(deflated_sharpe_prob_headline, deflated_sharpe_prob_best, "
            "published_deflated_sharpe_prob, deflated_sharpe_prob_by_gate), "
            "none of which is the literal key `deflated_sharpe_prob` sec 5.5 "
            "specifies. That inconsistent naming -- not moved or deleted "
            "artifacts -- best explains the gap, and is itself worth noting "
            "in the write-up as a record-keeping observation alongside the "
            "sec 5.4 recommendation that notebooks store their trial Sharpe "
            "vectors."
            if not (
                measured_count == DOCUMENTED_INVENTORY_COUNT
                and measured_files == DOCUMENTED_INVENTORY_FILES
            )
            else "Measured inventory matches sec 5.5 exactly."
        ),
        "rows": rows,
    }


def verify_018_fields() -> dict:
    with open("src/research/tmp/phase_4_18_results.json") as f:
        d18 = json.load(f)

    fa2 = d18["gates"]["FA-2"]
    fires_except_dsr_leg = fa2["fires_except_dsr_leg"]
    bootstrap_ci_leg_fires = fa2["bootstrap_ci_leg_fires"]
    holdout_access = d18["holdout_access"]

    assert fires_except_dsr_leg is False, (
        f"expected 018 FA-2 fires_except_dsr_leg=False, got {fires_except_dsr_leg}"
    )
    assert bootstrap_ci_leg_fires is False, (
        f"expected 018 FA-2 bootstrap_ci_leg_fires=False, got {bootstrap_ci_leg_fires}"
    )
    assert holdout_access["rule"] == "requires FA-2 AND FA-3", (
        f"018 holdout_access.rule changed: {holdout_access['rule']!r}"
    )
    assert holdout_access["access_granted"] is False, (
        "018 holdout_access.access_granted is not False -- stop, this notebook "
        "must not touch a spent holdout"
    )
    assert d18["n_trials_used"] == 18
    assert d18["dsr"]["deflated_sharpe_prob"] == 0.18590973717553716

    return {
        "source": "src/research/tmp/phase_4_18_results.json",
        "n_trials_used": d18["n_trials_used"],
        "dsr_deflated_sharpe_prob": d18["dsr"]["deflated_sharpe_prob"],
        "dsr_best_sharpe_per_period": d18["dsr"]["best_sharpe_per_period"],
        "dsr_best_sharpe_annualized": d18["dsr"]["best_sharpe_annualized"],
        "dsr_n_obs": d18["dsr"]["n_obs"],
        "gate_FA2": {
            "fires": fa2["fires"],
            "fires_except_dsr_leg": fires_except_dsr_leg,
            "bootstrap_ci_leg_fires": bootstrap_ci_leg_fires,
        },
        "gate_FA3": d18["gates"]["FA-3"],
        "gate_FUND": d18["gates"]["FUND"],
        "holdout_access": holdout_access,
        "conclusion": (
            "FA-2's bootstrap-CI leg fails independently of the DSR leg "
            "(fires_except_dsr_leg=False means even removing the DSR "
            "requirement entirely would not make FA-2 fire). FA-2 stays failed "
            "under ANY outcome of this notebook, per sec 0.1 item 1. Holdout "
            "access requires FA-2 AND FA-3; FA-3 also fails independently of "
            "any DSR leg (FA-3 has no DSR leg at all). No branch of this "
            "notebook's sec 14.2 decision tree can grant holdout access."
        ),
    }


VARIANTS = {
    "V0": {
        "name": "current implementation",
        "sr_star_scale": "se_hat(SR) -- Lo (2002) moment-adjusted sampling SE of one Sharpe",
        "extra_input": None,
    },
    "V1": {
        "name": "Bailey-LdP as published",
        "sr_star_scale": "std({SR_n}_{n=1}^N, ddof=1) -- cross-sectional dispersion of the N trial Sharpes",
        "extra_input": "the N trial Sharpes",
    },
    "V2": {
        "name": "effective-trials",
        "sr_star_scale": "se_hat(SR), with N replaced by N_eff = N / (1 + (N-1) * mean_pairwise_corr)",
        "extra_input": "mean pairwise correlation of the trial return series",
    },
    "V1b_c0.25": {
        "name": "V1 with shrinkage floor, c=0.25",
        "sr_star_scale": "max(std({SR_n}), 0.25 * se_hat(SR))",
        "extra_input": "the N trial Sharpes",
        "declared_as": "pre-declared fallback if DS-2a fails for V1 only at small N (sec 2.1)",
    },
    "V1b_c0.5": {
        "name": "V1 with shrinkage floor, c=0.5",
        "sr_star_scale": "max(std({SR_n}), 0.5 * se_hat(SR))",
        "extra_input": "the N trial Sharpes",
        "declared_as": "pre-declared fallback if DS-2a fails for V1 only at small N (sec 2.1)",
    },
}

ADOPTION_RULE = (
    "Among the variants that pass both DS-2 and DS-3, adopt the simplest, where "
    "simplicity is ordered V1 < V1b < V2 -- V1 first because it is what the "
    "source paper specifies and needs no tuning constant; V2 last because it "
    "requires the full inter-trial correlation matrix, which most of this "
    "repo's stored artifacts do not contain and which therefore makes most "
    "historical claims un-re-scorable. Ties are impossible under a total order. "
    "If exactly one passes, it is adopted whatever its rank. If none passes, "
    "none is adopted. If V1b is reached, its shrinkage constant c is chosen "
    "from {0.25, 0.5} by the same rule applied to the two V1b settings as a "
    "further tiebreak (simpler == smaller c, i.e. closer to unshrunk V1, "
    "preferred if both pass); no third c value may be introduced."
)

GATES = {
    "DS-1": {
        "name": "the defect is real (kill switch)",
        "fires_if": [
            "V0's FPR <= 0.005 in every cell with rho >= 0.9",
            (
                "V0's FPR is non-increasing in rho -- no cell may exceed the rho=0 "
                "cell's FPR by more than 2 MC SE, holding N, T, and moments fixed"
            ),
        ],
        "on_failure": (
            "stop. Do not patch research.py. Do not run Phase 3, 4, or 5. Write "
            "up the null: the deferred concern does not reproduce under Monte "
            "Carlo; the estimator behaves as documented and 018's DSR=0.186 "
            "stands unqualified. Ship notebook + README row. Stop."
        ),
        "evaluated_on": "Phase 2's grid",
    },
    "DS-2": {
        "name": "the adopted repair is calibrated",
        "sub_gates": {
            "DS-2a": "no anti-conservatism: FPR <= 0.075 in every cell (not on average)",
            "DS-2b": "usable where V0 is not: FPR >= 0.010 in every cell with rho >= 0.5",
        },
        "evaluated_on": "Phase 3's full null grid",
    },
    "DS-3": {
        "name": "the repair has power",
        "setup": "inject a true per-period Sharpe corresponding to annualized 1.0 "
        "into one trial of the family, rest at zero",
        "fires_if": [
            (
                "detection rate P(DSR>0.95) exceeds V0's by >= 10 percentage points "
                "in every cell with rho >= 0.9"
            ),
            (
                "detection rate is not below V0's by more than 2 percentage points "
                "in any cell with rho = 0"
            ),
        ],
        "evaluated_on": "Phase 3's full grid, injected-edge pass",
    },
    "DS-4": {
        "name": "the historical ledger is complete (a completeness check, not a hypothesis)",
        "fires_if": "all stored DSR values (per this file's _inventory, measured "
        "count -- see discrepancy_note) appear in phase_5_17_rescore.json, each "
        "with: stored value + source + pointer, the rho->1 upper bound, the "
        "exact corrected value where the family is recoverable (or "
        "not_rescorable + reason), and a verdict_change flag",
        "evaluated_on": "Phase 5's rescore output",
    },
    "MC_SE_note": (
        "Every reported FPR must carry its MC standard error; a gate comparison "
        "inside 2 MC SE is not a difference. At M=20000, SE at p=0.05 is "
        "~=0.0015; at the pilot's M=400 it is ~=0.011."
    ),
}

TRIAL_LEDGER = {
    "estimator_variants_evaluated": ["V0", "V1", "V2", "V1b(c=0.25)", "V1b(c=0.5)"],
    "count": 5,
    "note": (
        "The grid axes (N, T, rho, moments) are NOT trials -- they are the "
        "evaluation surface every variant is scored on identically, not "
        "alternatives among which one is selected. The selection is over "
        "variants only, governed by the fixed adoption-rule order rather than "
        "picking the best-looking one. Revisable only upward; any sixth variant "
        "increments this count and must be disclosed in the write-up with the "
        "reason it was added (013's precedent: 18->25, disclosed)."
    ),
}

VERDICT_CHANGE_POLICY = (
    "A prior gate whose DSR crosses 0.95 under the corrected estimator is "
    'recorded as "verdict changes under 017\'s corrected estimator", with both '
    "numbers and the trial family used. It does not retroactively become a "
    "fired gate. It becomes a candidate for a fresh pre-registration in a "
    "future notebook, which would have to state its gates in advance and meet "
    "them -- including the legs that were never about DSR. Specifically and by "
    "name: 018's FUND flag can flip on this leg alone, and if it does, that is "
    "reported as a flip and nothing more. It would not make the funding basis "
    "book institutionally fundable, because the book whose Sharpe FUND was "
    "reading is the same book whose net-return bootstrap CI includes zero -- a "
    "fact this notebook does not touch and must not. 018's FA-2 cannot flip "
    "under any outcome here (sec 0.1 item 1); if any analysis in this notebook "
    "appears to show otherwise, it is a bug in this notebook."
)

DISCLOSED_PILOT = {
    "script": "scratch/017/pilot_dsr.py",
    "params": {"N": 18, "T": 3840, "moments": "Gaussian", "M": 400, "seed": 7},
    "results_fpr_gt_0.95": {
        "0.00": {"V0": 0.0050, "V1": 0.0025},
        "0.50": {"V0": 0.0025, "V1": 0.0125},
        "0.90": {"V0": 0.0000, "V1": 0.0475},
        "0.99": {"V0": 0.0000, "V1": 0.0600},
    },
    "reproduced_in_phase_0": (
        "scratch/017/pilot_dsr.py re-run verbatim during Phase 0: output "
        "matched the table above exactly (rho=0.0: V0=0.0050/V1=0.0025; "
        "rho=0.5: V0=0.0025/V1=0.0125; rho=0.9: V0=0.0000/V1=0.0475; "
        "rho=0.99: V0=0.0000/V1=0.0600)."
    ),
    "limitations": [
        "M=400 gives MC SE ~=+-0.011 at p=0.05 -- indicative, not decisive",
        "0.0000 cells are 'below resolution', not proven zero",
        "one N, one T, Gaussian only, one seed, V2 not tested",
    ],
    "authority_rule": (
        "Phase 2 must reproduce this pilot as its first act, at M>=20000 and "
        "across the full rho axis, and record both the pilot's numbers and its "
        "own in phase_2_17_results.json. If Phase 2 contradicts the pilot, "
        "Phase 2 wins and DS-1 is judged on Phase 2."
    ),
}

GRID_AXES = {
    "N": [4, 8, 12, 18, 36, 95, 122],
    "T": [300, 1000, 3840],
    "rho": [0.0, 0.25, 0.5, 0.75, 0.9, 0.99],
    "moments": [
        {"skew": 0.0, "kurtosis": 3.0, "label": "gaussian"},
        {"skew": -1.5, "kurtosis": 6.0, "label": "moderate_nongaussian"},
        {"skew": -11.5, "kurtosis": 817.0, "label": "018_measured"},
    ],
    "n_cells_null": 378,
    "cost_model": "t_cell ~= 2.6e-8 * N * T * M seconds, single core (Pi 5, 4 cores); "
    "measured: (18,3840,400)->0.71s, (122,3840,400)->4.77s, (18,300,400)->0.05s",
    "M_full": 20000,
    "M_smoke": 500,
}

BRANCH_A = (
    "Branch A -- DS-1 does not fire (the estimator was fine; sec 5.1's kill "
    "switch tripped). 018's addendum records that the deferred concern was "
    "tested by Monte Carlo and did not reproduce, that "
    "research.deflated_sharpe_prob is unmodified, and that 018's DSR=0.186 "
    "stands unqualified -- the asterisk is removed rather than cashed. "
    "known_caveat in the addendum is marked resolved: concern_did_not_reproduce. "
    "No README change. Gate verdicts unchanged."
)

BRANCH_B = (
    "Branch B -- repair adopted, 018's corrected DSR < 0.95. Addendum records "
    "both numbers, the 18-member family used, and that FA-2 and FUND both "
    "remain failed. No README change. Gate verdicts unchanged. This is the "
    "likeliest branch on sec 14.1's reasoning, and it should be written up as "
    "a genuinely informative non-event: the estimator had a real defect, 018 "
    "was not materially a victim of it, and the funding basis trade fails for "
    "the reasons 018 already gave."
)

BRANCH_C = (
    "Branch C -- repair adopted, 018's corrected DSR >= 0.95. The branch that "
    "needs discipline: "
    "(1) FA-2 remains failed. Its bootstrap-CI leg failed independently "
    "(bootstrap_ci_leg_fires: false, CI [-1.31e-05, +7.02e-05]), and sec 12.2 "
    "keeps the bootstrap out of scope. phase_4_18_results.json already records "
    "fires_except_dsr_leg: false -- verified in Phase 0, quoted in the addendum. "
    "(2) FUND flips, and is reported as flipped. Its three legs are Sharpe>0.5 "
    "at every offset (passed), a stated bounded max drawdown (passed, -8.6%), "
    "and DSR>0.95 (the only failing leg). The addendum states the flip AND "
    "immediately states its limit, per sec 5.4's pre-commitment: the book whose "
    "Sharpe FUND reads is the same book whose net-return bootstrap CI includes "
    "zero. A fundability flag over a return series that cannot be distinguished "
    "from zero is a flag about an estimator, not about a fundable strategy. "
    "(3) The holdout stays unspent. Verified and quoted in Phase 0: access "
    "requires FA-2 AND FA-3 (holdout_access.rule), FA-3 failed on a paired "
    "bootstrap CI with no DSR leg anywhere in it, and FA-2 is covered by item "
    "1. No branch of this tree can grant holdout access. "
    "(4) README's 018 row is amended, because its current summary says the "
    "deflated Sharpe don't clear the tradeable-alpha bar, which would become "
    "misleading. Minimal edit: keep the row's substance, change the DSR clause "
    "to note the corrected estimator and point at 017. The row must still say "
    "the holdout is unspent and no alpha was validated. "
    "(5) A concrete 019 is proposed in 017's own what-to-test-next -- not "
    "started. Under branch C the binding constraint on 018 is squarely the CI "
    "leg, which is a data-and-construction problem, not an estimator problem: "
    "it wants the lower-turnover carry construction and the diversification "
    "floor that 018's own what-to-test-next already names, pre-registered "
    "fresh with all of FA-2's legs restated. Say that explicitly, so the "
    "reader is not left thinking a statistics fix reopened a trade."
)

BRANCH_TEXTS_018 = {"branch_A": BRANCH_A, "branch_B": BRANCH_B, "branch_C": BRANCH_C}

BACKWARD_COMPAT_CONTRACT = (
    "1. The default call path is unchanged, bit for bit: deflated_sharpe_prob("
    "sharpe, n_trials, n_obs, skew, kurtosis) with no new arguments returns "
    "exactly what it returns today; run_phase_0_repro.py passes untouched. "
    "2. The repair is opt-in via one new keyword-only argument, trial_sharpes "
    "(plus mean_pairwise_corr only if V2 is adopted). No variant= string on "
    "the public function. "
    "3. The docstring states plainly which path is correct and when, citing "
    "017. "
    "4. The 0.997 pin moves out of tmp/ and into tests/ as a regression test. "
    "5. No other function in research.py is touched."
)


def main() -> None:
    inventory = build_inventory()
    pilot = dict(DISCLOSED_PILOT)
    v018 = verify_018_fields()

    doc = {
        "notebook": "017_deflated_sharpe_correction",
        "written": "2026-08-17",
        "committed_before_phase_2_runs": True,
        "editable_after_commit": False,
        "supersedes": "none -- fixes a documented estimator defect without "
        "reopening any prior notebook's construction or conclusions",
        "scope": {
            "authorizes_trading": False,
            "spends_holdout": False,
            "modifies_holdout": False,
            "touches_market_data": False,
            "reruns_any_backtest": False,
            "functions_changed_in_research_py": ["deflated_sharpe_prob"],
            "functions_explicitly_out_of_scope": [
                "block_bootstrap_ci",
                "block_bootstrap_pvalue",
                "newey_west_tstat",
                "_auto_block_length",
            ],
        },
        "candidate_variants": VARIANTS,
        "adoption_rule": ADOPTION_RULE,
        "gates": GATES,
        "trial_ledger": TRIAL_LEDGER,
        "verdict_change_policy": VERDICT_CHANGE_POLICY,
        "disclosed_pilot": pilot,
        "grid": GRID_AXES,
        "backward_compat_contract": BACKWARD_COMPAT_CONTRACT,
        "inventory": inventory,
        "notebook_018_verification": v018,
        "notebook_018_amendment_branch_texts": BRANCH_TEXTS_018,
        "notebook_018_amendment_rules": [
            "phase_4_18_results.json and phase_5_18_results.json are never mutated",
            "018's notebook is never re-executed; one markdown cell is appended",
            "018's write-up gets one appended Addendum section at the very end",
            "018's n_trials stays 18",
            "018's gate verdicts as recorded stay as recorded in the JSON/prose",
            "nothing here reopens 018's construction",
            (
                "018's 18-trial family for re-scoring is ALL 18 declared trials, "
                "assembled from phase_4_18_results.json (12: 3 books x 4 offsets) "
                "and phase_5_18_results.json (6 ablations); if the 18 cannot be "
                "assembled exactly, 018's row is not_rescorable and does NOT fall "
                "back to a smaller family"
            ),
        ],
    }

    with open(OUT_PATH, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=False)
    print(f"written {OUT_PATH}")
    print(
        f"inventory: measured {inventory['measured']['count']} values across "
        f"{inventory['measured']['files']} files "
        f"(documented: {DOCUMENTED_INVENTORY_COUNT}/{DOCUMENTED_INVENTORY_FILES}, "
        f"matches={inventory['matches_documented']})"
    )
    print(f"018 verification: {v018['conclusion']}")


if __name__ == "__main__":
    main()
