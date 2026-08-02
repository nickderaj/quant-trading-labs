"""11a Phase 6: pre-registration for 11b/11c/11d (NEXT_PROMPT.md sec 3
Phase 6). Fixes the gate table, DSR trial counts (sec 9, verbatim, never to
be shrunk), and the sec 4.3 include/exclude decision, BEFORE any 11b/11c/11d
backtest exists. This is the binding contract those notebooks execute
against -- nothing here is a result, everything here is a commitment.

Writes phase_6_11a_results.json.
"""

import json

OUT_PATH = "src/research/tmp/phase_6_11a_results.json"

# ---------------------------------------------------------------------------
# Gate table, NEXT_PROMPT.md sec 4.2 + sec 6 (LC) + sec 7 (MB/MB-E),
# transcribed verbatim.
# ---------------------------------------------------------------------------

GATES = {
    "TS": {
        "notebook": "11b",
        "claim": "trade structure, not the market, explains Gate SP's weak Sharpe",
        "fires_if": (
            "net Sharpe > 0 at every origin offset AND paired block-bootstrap 95% CI on "
            "(structured - 10b's continuous Gate SP), same spreads/costs/offsets, excludes "
            "zero AND DSR > 0.95 on the cumulative count"
        ),
    },
    "TS-S": {
        "notebook": "11b",
        "claim": "the stop is the active ingredient, not the discrete-trade packaging",
        "fires_if": "Gate TS fires AND an otherwise-identical stop-disabled variant fails the same paired CI",
    },
    "BF": {
        "notebook": "11b",
        "claim": "mild backwardation should be traded in reverse, not filtered out",
        "fires_if": (
            "sign-flipped book's net Sharpe exceeds the unconditional book's at every offset AND "
            "paired-difference CI excludes zero AND DSR > 0.95 AND trade count within 10% of "
            "unconditional (sec 0.3 anti-throughput-collapse check, mandatory)"
        ),
    },
    "BF-X": {
        "notebook": "11b",
        "claim": "BF's improvement is not a brent_calendar artifact (they measured it on two spreads only)",
        "fires_if": "BF's improvement holds independently for >= 3 eligible spreads with a computable carry ratio",
    },
    "SCR": {
        "notebook": "11b",
        "claim": "our own 10a sec 4.3 ADF exclusion earns its place",
        "fires_if": (
            "KEEP only if the ADF-passing universe's net Sharpe beats the full eligible universe "
            "(incl. kc_chicago_wheat, gc_cal_m2m3, es_calendar) with a paired CI excluding zero. "
            "A screen that cannot beat its own absence does not survive."
        ),
    },
    "VA": {
        "notebook": "11b",
        "claim": "the vol-adaptive stop, re-opened by their data correction",
        "fires_if": (
            "their 0.75x-1.25x scaling of stop_atr_mult by realized-vol percentile: net Sharpe > 0 "
            "every offset AND paired CI excludes zero AND drawdown no worse than control"
        ),
    },
    "RE": {
        "notebook": "11b",
        "claim": "the re-entry validity gates, re-opened by their data correction",
        "fires_if": (
            "sweep half_life_max in {30,45,60} and adf_pmax in {0.05,0.10,0.20} -- their own "
            "flagged follow-up. Fires on the same three-way criterion."
        ),
    },
    "LC": {
        "notebook": "11c",
        "claim": "entry-time features predict which trades will exit via stop",
        "fires_if": (
            "a classifier trained walk-forward achieves out-of-sample AUC > 0.60 on the stop-exit "
            "label AND a book that suppresses its top-decile predicted-loss entries beats the "
            "unsuppressed book on the three-way risk gate"
        ),
    },
    "MB": {
        "notebook": "11d",
        "claim": "the breakout setup transfers to our 30 crypto perpetuals",
        "fires_if": (
            "net Sharpe > 0 at every offset AND paired CI excludes zero AND DSR > 0.95, full cost "
            "model, 1x/2x/3x cost stress, walk-forward, three-way risk gate, includes delisted pairs"
        ),
    },
    "MB-E": {
        "notebook": "11d",
        "claim": "the breakout setup on the 69-ticker commodity-equity universe (survivorship-unknown, labelled)",
        "fires_if": (
            "same construction as MB; NOT eligible for the institutionally-fundable flag "
            "(survivorship-unknown universe cannot support an absolute-performance claim)"
        ),
    },
}

# ---------------------------------------------------------------------------
# DSR trial counts, NEXT_PROMPT.md sec 9, transcribed verbatim. "Never
# shrunk" -- if a gate's search space turns out unreachable at this count,
# that is itself the reported finding (sec 9's own instruction), not a
# reason to lower the count.
# ---------------------------------------------------------------------------

DSR_COUNTS: dict[str, dict] = {
    "TS": {
        "n_trials": 4,
        "breakdown": "4 offsets x 1 parameter set (external priors, not swept)",
    },
    "TS-S": {
        "n_trials": 1,
        "breakdown": "stop-disabled variant, run once as a diagnostic on TS's own book",
    },
    "BF": {"n_trials": 12, "breakdown": "3 storage constants x 4 offsets"},
    "BF-X": {
        "n_trials": 0,
        "breakdown": "per-spread breakdown of BF's own book (pooling convention)",
    },
    "SCR": {"n_trials": 8, "breakdown": "2 universes x 4 offsets"},
    "VA": {
        "n_trials": 4,
        "breakdown": "4 offsets x 1 pre-declared scaling (0.75x-1.25x, taken from them, not swept)",
    },
    "RE": {
        "n_trials": 36,
        "breakdown": "3 half_life_max x 3 adf_pmax x 4 offsets -- an explicit grid, counted in full",
    },
    "LC": {
        "n_trials": 4,
        "breakdown": "4 offsets on one pre-declared feature set and one model class",
    },
    "MB": {"n_trials": 12, "breakdown": "3 cost multipliers x 4 offsets"},
    "MB-E": {"n_trials": 4, "breakdown": "4 offsets, equity universe"},
}
DSR_ACCOUNTING_NOTE = (
    "Counting convention unchanged from 10a Phase 5 (phase_5_10a_results.json): count every "
    "freely chosen axis searched to produce a gate's reported headline; do not multiply a "
    "declared, non-cherry-picked pooled universe by its member count; do not count mechanical, "
    "non-performance-based eligibility filters. Gate RE's count of 36 is deliberately painful "
    "and must not be reduced; if it becomes unreachable at n_trials=36, that is the finding."
)

# ---------------------------------------------------------------------------
# Sec 4.3 include/exclude decision for 11b's SCR gate universes, fixed here
# from 11a's own Phase 3/Phase 4 output before any 11b backtest runs.
# ---------------------------------------------------------------------------

SEC_4_3_DECISION = {
    "adf_passing_universe": (
        "The 10a Phase 2 sec 4.3 ADF-at-5%-level screen (phase_2_10a_results.json's own "
        "include_in_10b flags), unchanged from 10a/10b's own convention -- this is the "
        "SMALLER, stricter universe Gate SCR is asked to beat."
    ),
    "full_eligible_universe": (
        "Every spread with a computable trading rule under sec 4.1's parameterization, "
        "explicitly INCLUDING the three spreads sec 2.2 names as live cross-repo conflicts "
        "regardless of their 10a ADF verdict: kc_chicago_wheat (one of the five external live "
        "spreads; passes 10a ADF, no conflict on inclusion itself), gc_cal_m2m3 (10a: ADF t=-1.59, "
        "REJECT; theirs: standalone-promising, contradicts its own atlas per sec 2.2), es_calendar "
        "(10a: ADF t=+0.08, the worst in our panel, REJECT; theirs: 'the strongest breadth "
        "candidate across v1-v4'). This is the LARGER universe Gate SCR is asked to beat FROM."
    ),
    "resolution_mechanism": (
        "Gate SCR (sec 4.2), not assumption: KEEP the ADF screen only if the ADF-passing "
        "universe's net Sharpe beats the full eligible universe's, paired CI excluding zero. "
        "The two named conflicts (kc_chicago_wheat is not actually in conflict; gc_cal_m2m3 and "
        "es_calendar are) are resolved by whichever universe wins Gate SCR, not decided here."
    ),
    "cot_data_gap": (
        "COT positioning extremes (sec 2.2) are BLOCKED: this repo's data/market/cot/ holds "
        "only CL (067651); their proposal needs corn/wheat/soybeans COT, which we do not have. "
        "Reported as a data gap in 11a, not proxied, and out of scope for every 11b/c/d gate."
    ),
}

# ---------------------------------------------------------------------------
# Holdout contamination disclosure, NEXT_PROMPT.md sec 8, carried forward
# verbatim as the disclosure every future write-up touching the holdout
# must attach.
# ---------------------------------------------------------------------------

HOLDOUT_DISCLOSURE = {
    "dev_window": "2010-06-06 to 2024-12-31",
    "holdout_window": "2025-01-01 to 2026-07-28",
    "holdout_spent": False,
    "holdout_independence_reduced_for": "commodity spread strategies (11a/11b/11c), NOT 11d's crypto momentum work",
    "required_disclosure_text": (
        "holdout independence reduced -- external-repo 2024-2026 results were read during "
        "notebook 11's design"
    ),
    "reason": (
        "the external programme's held-out window (2024-01-01 to 2026-07-21) overlaps this "
        "repo's own holdout, and their held-out numbers (gc_cal_m2m3 mean ATR +0.62, "
        "ke_cal_m1m2's +$203,561 -> +$85 collapse, book held-out Sharpe 0.995, max DD -8.52%, "
        "calendar-basket held-out reversal) were read during this design, per NEXT_PROMPT.md sec 8"
    ),
}


def main() -> None:
    out = {
        "gates": GATES,
        "dsr_counts": DSR_COUNTS,
        "dsr_accounting_note": DSR_ACCOUNTING_NOTE,
        "sec_4_3_decision": SEC_4_3_DECISION,
        "holdout_disclosure": HOLDOUT_DISCLOSURE,
        "reporting_standard": {
            "tradeable_alpha_gate": "net Sharpe > 0 at every offset AND the relevant bootstrap CI excludes zero AND DSR > 0.95 on the true cumulative count",
            "institutionally_fundable_flag": "net Sharpe > 0.5 at every offset AND DSR > 0.95 AND max drawdown <= 25% of peak equity",
            "three_way_risk_gate": "report Sharpe, max drawdown and return/drawdown together; never accept a mechanism that raises return while worsening drawdown and Sharpe",
        },
        "ordering_constraint": (
            "11a must fully complete and this pre-registration must be committed before any "
            "11b/11c/11d backtest runs (NEXT_PROMPT.md sec 1). 11b, 11c, 11d are mutually "
            "independent and may run in any order once this file is committed."
        ),
        "no_gate_verdicts_in_11a": True,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(
        f"Phase 6: pre-registered {len(GATES)} gates across 11b/11c/11d, "
        f"DSR counts total={sum(g['n_trials'] for g in DSR_COUNTS.values())}, "
        f"sec_4.3 decision fixed, holdout disclosure fixed."
    )


if __name__ == "__main__":
    main()
