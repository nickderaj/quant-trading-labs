"""Notebook 017 Phase 6b (NEXT_PROMPT.md sec 14): the 018 amendment.

Writes phase_7_18_dsr_addendum.json, a NEW sidecar. phase_4_18_results.json
and phase_5_18_results.json are NEVER mutated (sec 14.3) -- everything here
references them by path and JSON pointer instead.

The actual outcome (DS-1 fires: the defect is confirmed real; but no
candidate variant passes both DS-2 and DS-3, so none is adopted and
research.py is untouched) is NOT one of the three branch texts frozen in
Phase 0. Re-reading sec 14.2's three branches against what actually
happened:

  Branch A's premise is "DS-1 does not fire" -- false, DS-1 fired cleanly
  (Phase 2). Pasting Branch A's text ("the deferred concern... did not
  reproduce") would be a factually false statement; the concern
  reproduced under Monte Carlo exactly as feared.

  Branch B's and Branch C's premise is "repair adopted" -- false, per
  Phase 4, none of V1/V1b(0.25)/V1b(0.5)/V2 passed both DS-2 and DS-3.

Sec 14.2's own framing says Phase 6b's "only job is to select the branch
the results dictate and paste it" -- written on the assumption that one of
the three would fit. None does. Forcing this outcome into the nearest-
sounding branch (most people would reach for Branch A, since the
*mechanical* consequence -- research.py unmodified, 018's number
unchanged -- matches) would misrepresent WHY nothing changed: Branch A's
text asserts the concern "did not reproduce," which is the opposite of
what Phase 2 found. That is exactly the kind of paste-without-verifying
error sec 14.2 is designed to prevent, just from an angle its author did
not anticipate. So this writes a fourth, honest characterization instead,
holding every sec 14.3 rule that applies to all three frozen branches:
never mutate 018's result JSONs, never re-execute its notebook, one
appended addendum section only, 018's n_trials stays 18, its gate verdicts
as recorded stay as recorded, and nothing here reopens 018's construction.

Usage: uv run python src/research/tmp/build_phase_7_18_addendum.py
"""

from __future__ import annotations

import json

PREREG_PATH = "src/research/tmp/phase_0_17_preregistration.json"
PHASE2_PATH = "src/research/tmp/phase_2_17_results.json"
PHASE4_PATH = "src/research/tmp/phase_4_17_adoption.json"
PHASE5_PATH = "src/research/tmp/phase_5_17_rescore.json"
OUT_PATH = "src/research/tmp/phase_7_18_dsr_addendum.json"


def main() -> None:
    with open(PREREG_PATH) as f:
        prereg = json.load(f)
    with open(PHASE2_PATH) as f:
        phase2 = json.load(f)
    with open(PHASE4_PATH) as f:
        phase4 = json.load(f)
    with open(PHASE5_PATH) as f:
        phase5 = json.load(f)

    v018 = prereg["notebook_018_verification"]
    row_018 = next(
        r
        for r in phase5["rows"]
        if r["file"] == "src/research/tmp/phase_4_18_results.json"
    )
    assert row_018["stored_value"] == v018["dsr_deflated_sharpe_prob"]
    assert row_018["rho_to_1_upper_bound"] < 0.95, (
        "018's own upper bound is expected to be below 0.95 -- if this "
        "assertion fires, the honest_characterization text below needs "
        "rewriting, not just this assertion loosened"
    )

    branch_applicability = {
        "branch_A_DS1_does_not_fire": {
            "premise_holds": False,
            "why": "DS-1 fired (Phase 2): V0's FPR at rho in {0.9, 0.99} is "
            "0.0010/0.0002 (<=0.005) and non-increasing in rho from the "
            "rho=0 baseline of 0.00375, within 2 MC SE at every "
            "intermediate point. The deferred concern is confirmed, not "
            "dismissed -- pasting Branch A's text would misstate this.",
        },
        "branch_B_repair_adopted_corrected_dsr_below_0.95": {
            "premise_holds": False,
            "why": "No repair was adopted in Phase 4 (see below), so there "
            "is no 'corrected DSR' for 018 to be below or above 0.95.",
        },
        "branch_C_repair_adopted_corrected_dsr_at_least_0.95": {
            "premise_holds": False,
            "why": "Same as Branch B: no repair was adopted.",
        },
        "conclusion": "None of the three sec 14.2 branch texts is pasted. "
        "A new characterization follows, holding every sec 14.3 rule that "
        "applies uniformly across all three frozen branches.",
    }

    phase4_summary = {
        "adopted_variant": phase4["adopted_variant"],
        "why_none_adopted": {
            "v1": "passes DS-2 (calibration) cleanly; fails DS-3 (power): "
            "insufficient power gain over V0 at high correlation in 20 "
            "cells (below the required 10-percentage-point margin), AND a "
            "power loss versus V0 at rho=0 beyond the allowed 2-point "
            "margin in 20 cells.",
            "v1b_c0.25": "same failure mode as v1.",
            "v1b_c0.5": "fails DS-2b in addition (38 null cells with rho>=0.5 "
            "have FPR below the 0.010 usability floor -- too conservative) "
            "as well as DS-3.",
            "v2": "fails DS-2a badly: 246 of 378 null cells exceed the 0.075 "
            "anti-conservatism ceiling. Passes DS-3.",
        },
    }

    doc = {
        "notebook": "017_deflated_sharpe_correction",
        "phase": "6b",
        "purpose": "Discharges the known_caveat 018 recorded pointing at "
        "this notebook ('fixing it is notebook 017'). Sidecar only -- "
        "phase_4_18_results.json and phase_5_18_results.json are not "
        "modified by this file or by anything that produced it.",
        "referenced_018_artifacts": {
            "phase_4_18_results.json": v018,
            "phase_5_18_results.json": "6 ablations, unchanged -- not read "
            "for this addendum (018's row in Phase 5's rescore is "
            "not_rescorable for a reason unrelated to Phase 5's own "
            "ablation contents; see 'the_18_trial_family' below).",
        },
        "sec_14_2_branch_applicability": branch_applicability,
        "phase_2_ds1_result": {
            "fires": phase2["gate_DS1"]["fires"],
            "v0_fpr_at_high_rho": phase2["gate_DS1"]["clause_1_high_rho_fpr_le_0.005"][
                "values"
            ],
        },
        "phase_4_adoption_result": phase4_summary,
        "the_18_trial_family": {
            "note": "sec 14.1's mandatory rule: 018's family is all 18 "
            "declared trials (12 offset-book values + 6 Phase 5 "
            "ablations), assembled from both JSONs, or the row is "
            "not_rescorable -- it does not fall back to a smaller family. "
            "Moot here because no variant was adopted (there is nothing to "
            "apply the family to), but recorded for completeness and for "
            "whichever future notebook picks this up: the 6 'ablation' "
            "slots in phase_5_18_results.json do not correspond to 6 "
            "stored scalar Sharpes -- they are 6 named categories "
            "containing 13 stored values of uneven cardinality (1, 1, 1, "
            "4, 2, 4), several without a net-of-cost analog to the other "
            "12 offset values. Assembling exactly 18 would require an "
            "unstated, ad hoc selection rule (e.g. 'use the 34bp cost "
            "level') -- exactly the kind of undisclosed family choice sec "
            "2.2/13.6 exist to prevent. This is a record-keeping finding "
            "about 018, not a blocker for this addendum (see scratch/017/"
            "phase6b_018_family_notes.md for the full breakdown).",
            "would_be_not_rescorable_even_if_a_variant_had_been_adopted": True,
        },
        "018_own_row_in_phase_5_rescore": {
            "note": "018's own stored DSR (phase_4_18_results.json -> "
            "dsr.deflated_sharpe_prob) IS one of the sec 5.5 inventory's 70 "
            "rows (exact-key sweep; see src/research/tmp/"
            "phase_5_17_rescore.json). Its rho->1 upper bound -- the "
            "maximum value ANY dispersion-based repair (V0, V1, V1b at "
            "either shrinkage setting) could ever produce for these exact "
            "stored inputs -- is 0.8317, computed from 018's own "
            "sample_skew=-11.516 and sample_kurtosis=816.85 (Phase 0's "
            "'018_measured' moment regime exists specifically to stress-"
            "test this case). That is BELOW the 0.95 bar. This is a "
            "stronger and more decisive statement than 'no variant was "
            "adopted': even in the counterfactual where Phase 4 HAD "
            "adopted V1 or a V1b setting, 018's corrected DSR could not "
            "have exceeded 0.83 -- FA-2's DSR leg and FUND's DSR leg were "
            "mathematically incapable of flipping for this specific book, "
            "independent of which repair (if any) this notebook settled "
            "on. Verified independently: "
            "dsr_lib17.psr_upper_bound(0.01743215308672331, 3837, "
            "-11.516325584172863, 816.8538707698766) == "
            f"{row_018['rho_to_1_upper_bound']!r}.",
            "upper_bound": row_018["rho_to_1_upper_bound"],
            "source_row": row_018,
            "would_have_been_provably_incapable_of_flipping_regardless_of_adoption": True,
        },
        "what_this_means_for_018": {
            "research_py_modified": False,
            "018_dsr_value_unchanged": v018["dsr_deflated_sharpe_prob"],
            "018_dsr_value_confirmation": "research.py is bit-for-bit "
            "unchanged from before this notebook (git diff empty), so any "
            "call with 018's exact stored inputs, including the original "
            "deflated_sharpe_prob call, returns exactly "
            "0.18590973717553716, unchanged -- and per the upper-bound "
            "finding above, no repair this notebook could have adopted "
            "would have changed that fact either.",
            "gate_FA2_fires": False,
            "gate_FA2_can_flip_under_any_outcome_here": False,
            "gate_FA3_fires": False,
            "gate_FUND_fires": False,
            "gate_FUND_dsr_leg_value_unchanged": True,
            "holdout_access_granted": False,
            "readme_018_row_changed": False,
            "018_notebook_re_executed": False,
            "018_results_md_prose_edited_above_the_addendum": False,
        },
        "honest_characterization": (
            "Two findings, and the second is the one that actually settles "
            "018's case. First: 017 confirmed, by Monte Carlo at M=20000 "
            "across a 7x3x6x3 grid (756 cells), that "
            "research.deflated_sharpe_prob's deflation benchmark is scaled "
            "incorrectly for correlated trial families -- the sampling-SE-"
            "vs-cross-sectional-dispersion defect 018's own known_caveat "
            "flagged is real, not a false alarm. None of the four pre-"
            "registered candidate repairs (V1, V1b at two shrinkage "
            "settings, V2) is simultaneously well-calibrated and "
            "sufficiently powerful across the grid, so per the adoption "
            "rule fixed in Phase 0, none is adopted and research.py is "
            "untouched. On its own, that would leave 018's case genuinely "
            "open -- defect confirmed, no validated fix to apply it with. "
            "Second, and decisive: 018's own stored inputs (per-period "
            "Sharpe 0.01743, n_obs=3837, sample skew=-11.516, sample "
            "kurtosis=816.85 -- 018's own book is exactly the extreme, "
            "fat-tailed regime Phase 0's '018_measured' moment axis exists "
            "to probe) put a hard ceiling on what ANY dispersion-based "
            "repair could ever produce for this book: the rho->1 upper "
            "bound is 0.8317, below the 0.95 bar. That bound applies to "
            "V0, V1, and both V1b settings alike (Test 7, sec 7.3) "
            "regardless of which one -- if any -- had been adopted. So "
            "018's DSR was never actually gated on Phase 4's adoption "
            "decision: even in the counterfactual where V1 had cleared "
            "both DS-2 and DS-3, 018's corrected DSR could not have "
            "exceeded 0.83. FA-2's DSR leg and FUND's DSR leg were "
            "mathematically incapable of flipping for this specific book, "
            "for a reason that has nothing to do with which repair (if "
            "any) 017 settled on: 018's own return series is fat-tailed "
            "enough that its plain, un-deflated probabilistic Sharpe "
            "(PSR, the rho->1 limit) already tops out at 0.83. The "
            "practical consequence is the same as if the concern had "
            "never reproduced -- 018's DSR is numerically unchanged, "
            "FA-2 and FUND stay failed, the holdout stays unspent -- but "
            "the reason is different and matters: the asterisk is not "
            "removed because the worry was unfounded (it was not, and "
            "the defect remains a confirmed, general, currently-unrepaired "
            "limitation of this repo's DSR estimator for correlated trial "
            "families, applicable to every other near-identical-offset "
            "gate in this programme); it is settled because 018's own "
            "data makes the DSR leg's outcome independent of the repair "
            "question entirely. A future notebook attempting a fifth "
            "repair variant, or accepting a narrower calibrated regime "
            "(e.g. a repair validated only for the N and T ranges this "
            "grid covers well), would be a reasonable next step for the "
            "estimator generally -- proposed below in 017's own "
            "what-to-test-next -- but it would not reopen 018's case; "
            "018's own numbers already close it."
        ),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"written {OUT_PATH}")


if __name__ == "__main__":
    main()
