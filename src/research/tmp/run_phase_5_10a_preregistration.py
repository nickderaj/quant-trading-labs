"""10a Phase 5: pre-registration for 10b (NEXT_PROMPT.md sec 5 Phase 5). This
IS 10b's gate table, written and committed before 10b's first backtest runs
(sec 1 rule 2). Nothing in this file may be edited after a 10b backtest
exists (sec 1 rule 2) -- if 10a's own descriptive Phase 1-4 results suggest a
change, that change belongs HERE, now, before 10b, with the reasoning
explicit, never as a post-hoc edit once a number is in hand.

This script contains no new statistics -- it is the deliberately hand-written
statistical design (gate table, regime-definition primary/secondary
declaration, entry/exit rule, and the DSR configuration-count justification)
that the orchestrating researcher is responsible for getting right (sec 1
and sec 4 of NEXT_PROMPT.md are explicit that a wrong call here invalidates
everything 10b produces). It writes phase_5_10a_results.json.
"""

import json

OUT_PATH = "src/research/tmp/phase_5_10a_results.json"

# ---------------------------------------------------------------------------
# Regime-definition primary/secondary declaration -- a correction made HERE,
# at pre-registration time, before any 10b backtest, based on a structural
# property of the raw-sign definition discovered while designing this phase
# (not on which definition's descriptive numbers "looked better" in 10a
# Phase 3 -- Phase 3 already ran and reported all three variants' descriptive
# stats regardless of which becomes primary, so no data-snooping occurred).
#
# `commod_lib8.term_structure_state`'s raw sign is defined for essentially
# every trading day (only an exactly-zero annualised roll slope, vanishingly
# rare in continuous price data, returns null) -- so gating a book by raw
# sign alone is trivially close to an unconditional book: it excludes almost
# nothing. That makes raw sign structurally unable to test the operator's
# actual claim (sec 0: "should only be traded when the underlying term
# structure is in a DEFINITE state ... trading it regime-blind is what
# destroys the edge") -- "regime-blind" and "raw-sign-gated" are nearly the
# same book. The DEADBAND variant (ii) is the only one of the three that
# creates a genuine no-trade zone (a "flat" state, sec 4.1), and is therefore
# the only variant that can actually distinguish the operator's hypothesis
# from the unconditional baseline. It is declared PRIMARY for Gate SPR's
# fire condition here. Raw sign (i) and the persistence requirement (iii)
# are declared SECONDARY / robustness variants, run and reported, but not
# the headline.
REGIME_DEFINITIONS = {
    "primary": "deadband",
    "secondary": ["raw_sign", "persistent"],
    "deadband_threshold_annualized": 0.02,
    "persistence_days": 5,
    "rationale": (
        "raw_sign is defined on ~100% of trading days (only an exactly-zero slope is "
        "null), so gating on it alone barely differs from the unconditional book and "
        "cannot test the operator's 'definite state, not regime-blind' claim. deadband "
        "is the only variant of the three that creates a real no-trade zone and is "
        "promoted to primary for that structural reason, decided before any 10b "
        "backtest number exists."
    ),
}

PRIMARY_REGIME_LEG_RULE = (
    "leg1 (the first leg in the spread's own leg_roles ordering), applied identically "
    "to every inter-commodity spread -- a fixed, mechanical, pre-declared rule, not "
    "chosen per spread. For brent_wti this is BZ. A 'both legs agree' variant is run "
    "ONLY for brent_wti (sec 4.1's own named operator case) as a labelled secondary "
    "robustness check for Gate SPR-BW, not as an additional configuration for every spread."
)

# ---------------------------------------------------------------------------
# Entry/exit rule and position sizing for Gate SP/SPR (sec 6 Phase 1/2).
# ---------------------------------------------------------------------------

TRADING_RULE = {
    "signal": (
        "60-day rolling z-score of the spread's own value series (the same window "
        "research_lib9.zscore_ic already used in notebook 9's Phase 4 probe and in "
        "10a Phase 2's descriptive extension to all 30 spreads -- reused for pipeline "
        "coherence, not re-tuned to whatever looks best in a backtest)."
    ),
    "position": "position_t = -clip(z_t, -2, 2) / 2, i.e. long the spread when it is cheap (very negative z), short when rich (very positive z), scaled to +-1 gross.",
    "regime_gate_(SPR_only)": "position_t is zeroed on any day where the primary (deadband) regime label is 'flat' -- SP's own unconditional position is otherwise identical.",
    "rebalance": "daily",
    "cost_model": (
        "TWO round-turn costs per unit of executed weight change -- one per leg, each "
        "leg's own commod_lib8.round_turn_cost_per_contract at that leg's own product "
        "and price (NEXT_PROMPT.md's own explicit warning: a spread trade is two round "
        "turns, not one -- getting this wrong halves the true cost)."
    ),
    "roll_window_exclusion": (
        "applied to the backtest itself, not only the descriptive screen (sec 6 Phase "
        "1): rows flagged roll_window_flag=True are dropped from both signal "
        "computation and P&L accrual before any metric is computed."
    ),
    "origin_offsets": [0, 7, 14, 21],
    "pooling": (
        "SP/SPR fire conditions are evaluated on an equal-weighted book of every "
        "eligible spread in the relevant taxonomy group (sec 4.2) -- the same "
        "cross-sectional-book convention notebook 8 used for carry/momentum. "
        "Per-spread breakdown (feeding Gate SPR-BW) is a diagnostic view of the SAME "
        "book's own constituent legs, not a separate search over which spread to report "
        "-- consistent with how notebook 8 never multiplied gate_AC/gate_AM's DSR "
        "n_trials by its own 16-product panel size."
    ),
}

# ---------------------------------------------------------------------------
# Sec 4.3 include/exclude decision -- restated here from 10a Phase 2 (already
# made there, before this phase, on a pre-declared mechanical criterion; not
# revisited after seeing any backtest number).
# ---------------------------------------------------------------------------

INCLUDE_EXCLUDE_DECISION = (
    "Spreads failing the ADF cointegration/stationarity test at the 5% level "
    "(spread_lib10.adf_test, stationary_5pct == False) are EXCLUDED from Gate SP/SPR's "
    "10b backtest universe. Decided in 10a Phase 2 (src/research/tmp/"
    "run_phase_2_10a_spread_taxonomy.py's own _sec_4_3_decision field), before this "
    "pre-registration and before any 10b backtest. gold_silver and platinum_palladium "
    "-- notebook 9's own flagged AR1-vs-IC disagreement -- both fail ADF (t=-1.76 and "
    "-1.41 respectively, vs. the 5% critical value of -2.86), resolving that "
    "disagreement: neither pair is actually cointegrated, which is consistent with both "
    "their weak AR1 significance and their insignificant IC. Excluded spreads' full "
    "descriptive record remains reported in 10a Phase 2's output, never silently dropped."
)


# ---------------------------------------------------------------------------
# Gate table -- copied verbatim from NEXT_PROMPT.md sec 4 (fire conditions
# not weakened or edited, per sec 1 rule 2). FA-data is a data check, not a
# strategy gate, and carries no DSR count.
# ---------------------------------------------------------------------------

GATE_TABLE = {
    "SP": {
        "notebook": "10b",
        "claim": "unconditional structural mean-reversion in commodity spreads survives cost",
        "fires_if": "net Sharpe > 0 at every origin offset AND block-bootstrap 95% CI on net return (self-financing, baseline zero) excludes zero AND DSR > 0.95 on the cumulative config count",
        "scope": "run on both taxonomy groups (inter-commodity, calendar), reported separately by group, on eligible (cointegrated) spreads only",
    },
    "SPR": {
        "notebook": "10b",
        "claim": "regime-gating improves it: trading a spread only when its underlying term structure is in a definite contango or backwardation state beats trading it regime-blind",
        "fires_if": "Gate SP's full criterion met by the regime-gated variant AND the regime-gated net Sharpe exceeds the unconditional net Sharpe at every origin offset AND a block-bootstrap 95% CI on the difference (gated minus unconditional) excludes zero",
        "scope": "inter-commodity spreads only (sec 4.2), primary regime definition = deadband (see REGIME_DEFINITIONS above)",
    },
    "SPR-BW": {
        "notebook": "10b",
        "claim": "the operator's specific prior: the effect is not a brent_wti artifact",
        "fires_if": "Gate SPR's improvement holds for brent_wti AND for at least 3 other inter-commodity spreads independently",
        "scope": "per-spread breakdown of Gate SPR's own book, plus brent_wti's both-legs-agree secondary variant",
    },
    "VS": {
        "notebook": "10b",
        "claim": "volatility-scaled commodity carry closes Gate AC's excess-vs-basket gap",
        "fires_if": "the SAME criterion Gate AC used (net Sharpe > 0 every offset AND excess-vs-equal-weight-basket bootstrap CI excludes zero AND DSR > 0.95), with the DSR count carrying forward notebook 8's already-logged carry configurations -- not reset to 1",
        "scope": "reuses notebook 8's carry panel/cost model, only the position-sizing rule (vol-scaled, not constant-weight) changes",
    },
    "BM": {
        "notebook": "10b",
        "claim": "an equal-weighted blend of notebook 8's four momentum lookbacks is sign-consistent and survives cost",
        "fires_if": "same criterion, counting the blend as a new configuration on top of the already-logged single-lookback configs",
        "scope": "reuses notebook 8's momentum panel/cost model",
    },
    "FA-data": {
        "notebook": "10b",
        "claim": "(data check, not a strategy gate) this repo's cache holds a crypto spot series distinct from the perpetuals",
        "fires_if": "resolved TRUE or FALSE before any Gate FA work; if FALSE, Gate FA is deferred with a data-acquisition note -- no proxy",
        "scope": "cheap check only",
    },
}

# ---------------------------------------------------------------------------
# DSR configuration count -- the cumulative, per-gate n_trials fed to
# research.deflated_sharpe_prob, and the reasoning behind each number (sec 1
# rule 3: cumulative, never shrunk to make a gate reachable).
#
# Counting convention, applied uniformly and declared here in advance:
# n_trials for a gate counts every distinct, freely-chosen axis that was
# swept/searched in producing that gate's own reported headline statistic --
# every value on that axis is a candidate that COULD have been reported
# instead of the one that was. Origin-offset robustness runs are counted as
# part of this (continuing notebook 8's own established convention for
# gate_AC/gate_AM -- not revised here, since silently tightening a
# counting convention mid-programme is itself a form of after-the-fact
# goalpost movement). A pooled, equal-weighted book across many assets (e.g.
# "all eligible inter-commodity spreads") is NOT multiplied by the number of
# assets in the book -- every asset in a declared, non-cherry-picked universe
# is part of the reported population, not a competing configuration (exactly
# how notebook 8 never multiplied gate_AC's n_trials=4 by its own 16-product
# panel). A mechanical, pre-declared, non-performance-based inclusion filter
# (the sec 4.3 ADF screen; notebook 8's own liquidity/hygiene screens) is
# likewise not counted -- it does not involve looking at any performance
# number to decide inclusion.
# ---------------------------------------------------------------------------

DSR_CONFIG_COUNTS = {
    "SP": {
        "n_trials": 8,
        "breakdown": "2 taxonomy groups (inter_commodity, calendar) x 4 origin offsets",
    },
    "SPR": {
        "n_trials": 12,
        "breakdown": "3 regime definitions (raw_sign, deadband, persistent) x 4 origin offsets, inter-commodity group only",
    },
    "SPR-BW": {
        "n_trials": 1,
        "breakdown": (
            "the brent_wti both-legs-agree variant, run once (not at every offset -- it "
            "is a diagnostic robustness check on SPR's own already-counted per-spread "
            "results, not itself a fire-condition search). SPR-BW's per-spread breakdown "
            "of Gate SPR's own book contributes 0 additional trials, per the pooling note "
            "above."
        ),
    },
    "VS": {
        "n_trials": 8,
        "breakdown": (
            "4 already-logged notebook-8 carry configurations (phase_5_results.json's "
            "gate_AC n_trials, itself 4 origin offsets on the single 21-day carry "
            "horizon) forwarded, not reset to 1, PLUS 4 new vol-scaled-variant "
            "configurations (the same 4 origin offsets, re-run under vol-scaled sizing)."
        ),
    },
    "BM": {
        "n_trials": 20,
        "breakdown": (
            "16 already-logged notebook-8 momentum configurations forwarded (the literal "
            "n_trials phase_5_results.json's gate_AM computation used: 4 lookbacks x 4 "
            "origin offsets -- NOT the 4-lookback-only figure NEXT_PROMPT.md's own "
            "summary prose uses; the larger, technically precise historical number is "
            "used here per sec 1 rule 3's binding 'do not shrink the count' instruction, "
            "flagging this explicitly as a resolved ambiguity in NEXT_PROMPT.md's own "
            "text) PLUS 4 new blend configurations (the blend re-run at the same 4 origin "
            "offsets)."
        ),
    },
    "_transparency_log": {
        "total_spreads_descriptively_screened_10a": 30,
        "total_regime_conditional_descriptive_runs_10a": 11
        * 3,  # 11 inter-commodity spreads x 3 regime defs, Phase 3
        "note": (
            "These 10a screening/descriptive numbers are logged for full transparency "
            "(sec 8: 'log every configuration tried') but are NOT part of any gate's "
            "n_trials above -- they are either a mechanical, non-performance-based "
            "eligibility filter (30-spread ADF screen) or a full, non-cherry-picked "
            "population report (11x3 regime-conditional descriptive stats, all of which "
            "are reported in 10a Phase 3 regardless of outcome, not searched over to "
            "pick a favourite). deflated_sharpe_prob's n_trials parameter is specifically "
            "the search space behind ONE gate's OWN reported statistic, not an "
            "undifferentiated global count of every computation in the notebook."
        ),
    },
}

PREREGISTRATION_TIMESTAMP_NOTE = (
    "This file is 10b's pre-registration. Committed before 10b's first backtest runs "
    "(sec 1 rule 2). Nothing in GATE_TABLE's fire_if conditions, REGIME_DEFINITIONS' "
    "primary choice, TRADING_RULE, INCLUDE_EXCLUDE_DECISION, or DSR_CONFIG_COUNTS may be "
    "edited after a 10b backtest exists."
)


def main():
    results = {
        "regime_definitions": REGIME_DEFINITIONS,
        "primary_regime_leg_rule": PRIMARY_REGIME_LEG_RULE,
        "trading_rule": TRADING_RULE,
        "include_exclude_decision_sec_4_3": INCLUDE_EXCLUDE_DECISION,
        "gate_table": GATE_TABLE,
        "dsr_config_counts": DSR_CONFIG_COUNTS,
        "_preregistration_note": PREREGISTRATION_TIMESTAMP_NOTE,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"written {OUT_PATH}")


if __name__ == "__main__":
    main()
