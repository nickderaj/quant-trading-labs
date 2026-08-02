"""Phase 2: the diagnosis (NEXT_PROMPT.md sec 4, Phase 2) - this notebook's
headline deliverable.

Ranks the five competing hypotheses (NEXT_PROMPT.md sec 2) against the Phase 1
survey (`phase_1_survey_results.json`) and the Phase 0 reproduction
(`phase_0_repro9_results.json`). This script does not run any new statistics - the
"computation" here is applying a written-down scoring rule to the tiered
source record, exactly as any other Phase runner applies a written-down rule
to computed numbers. The verdict text is this notebook's own synthesis and is
labelled as such throughout (per sec 6: "a claim without a source is this
notebook's own inference").

Verdict scale (declared before scoring, sec 4 Phase 2 requirement):
  well-supported      - multiple Tier 1/2 sources point the same direction,
                         no material Tier 1/2 conflict
  partially supported - real Tier 1/2 evidence exists but is mixed, narrow in
                         scope, or contradicted by other Tier 1/2 evidence
  contradicted         - the weight of Tier 1/2 evidence points against the
                         hypothesis as stated
  insufficient evidence - fewer than 2 Tier 1/2 sources bear on it, or the
                         evidence found doesn't actually speak to the claim

Writes phase_2_diagnosis_results.json.
"""

import json

SOURCES_PATH = "src/research/tmp/phase_1_survey_results.json"
REPRO_PATH = "src/research/tmp/phase_0_repro9_results.json"
OUT_PATH = "src/research/tmp/phase_2_diagnosis_results.json"


def load(path):
    with open(path) as f:
        return json.load(f)


HYPOTHESES = {
    "a": {
        "name": "Our strategies are too naive",
        "verdict": "partially supported",
        "supporting_sources": [
            "barroso_santa_clara_2015", "aqr_century_trend_following",
            "aqr_demystifying_managed_futures", "jensen_kelly_pedersen_2023",
        ],
        "contrary_or_qualifying_sources": ["man_group_vol_targeting"],
        "reasoning": (
            "Real, quantified engineering gaps exist in the Tier 1/2 literature: "
            "vol-scaling alone nearly doubled equity momentum's Sharpe (0.53->0.97, "
            "Barroso-Santa-Clara, Tier 1), and AQR's cross-asset, multi-lookback trend "
            "blend (Tier 2, 67 markets, 137 years) is a materially more engineered "
            "construction than notebook 8's single-lookback-at-a-time, single-asset-class "
            "momentum test. Jensen-Kelly-Pedersen (Tier 1) shows careful, consistent "
            "construction recovers >80% of factor significance that naive replication "
            "loses. BUT the one Tier 2 source that speaks directly to whether the "
            "best-evidenced fix (vol-scaling) transfers to THIS repo's own asset classes "
            "(Man Group, Tier 2) reports the benefit is concentrated in equities/credit and "
            "'negligible' for commodities/currencies specifically - the asset classes this "
            "repo actually trades. Verdict is 'partially supported' rather than "
            "'well-supported' because the strongest evidence for (a) comes from equities, "
            "and the one source testing transferability to commodities specifically is "
            "unfavourable. Blended multi-lookback construction (not vol-scaling) is the "
            "least-contradicted, most-actionable version of (a) for this repo."
        ),
    },
    "b": {
        "name": "Our cost model is too pessimistic",
        "verdict": "contradicted",
        "supporting_sources": [],
        "contrary_or_qualifying_sources": [
            "novy_marx_velikov_2016", "chen_velikov_2023", "quantpedia_oos_analysis",
        ],
        "reasoning": (
            "No Tier 1/2 source found in this survey argues that realistic trading costs "
            "are LOWER than naive backtest assumptions. The opposite pattern dominates the "
            "Tier 1 cost literature: Novy-Marx & Velikov (2016) find momentum's all-in "
            "implementation costs (7.2-7.6%/yr) are considerably larger than bid-ask-spread-"
            "only estimates and eliminate most of momentum's published profit; Chen & "
            "Velikov (2023) extend the same conclusion across a broad anomaly set. "
            "Quantpedia's meta-analysis (Tier 2) reports well-built strategies typically "
            "losing 10-20% of performance backtest-to-live even before considering whether "
            "the backtest's own cost assumptions were adequate. This repo's crypto cost "
            "assumption (4bps taker + 1bp slippage) sits within the range Tier 3 practitioner "
            "sources (not counted toward this verdict) suggest for retail/small-institutional "
            "execution, and Binance's own VIP fee tiers (found only via a Tier 3 secondary "
            "source, not independently verified against Binance's primary schedule) would let "
            "a large, high-volume desk trade cheaper - but this repo's backtests were never "
            "sized to a volume tier that would qualify, so this is not evidence the current "
            "assumption is wrong for THIS repo's implied trade sizes. Verdict: contradicted, "
            "on the specific claim that the cost model is too harsh; the weight of Tier 1/2 "
            "evidence instead cautions that unmodelled costs are more often larger than "
            "assumed, not smaller."
        ),
    },
    "c": {
        "name": "Our statistical bar is too strict",
        "verdict": "partially supported",
        "supporting_sources": [
            "man_ahl_track_record", "asness_ritholtz_transcript", "quantpedia_oos_analysis",
        ],
        "contrary_or_qualifying_sources": [
            "harvey_liu_zhu_2016", "bailey_lopez_de_prado_dsr",
        ],
        "reasoning": (
            "This hypothesis needed the most careful separation of two DIFFERENT bars this "
            "repo actually applies, which the survey evidence resolves differently. "
            "(1) The deflated-Sharpe-probability threshold (>0.95): Harvey, Liu & Zhu "
            "(Tier 1) argue the finance literature's historical bar (t~2.0) is too LOOSE "
            "given the true scale of factor search, and that a defensible bar is closer to "
            "t~3.0 - if anything this argues this repo's own multiple-testing discipline is "
            "not excessive. Not contradicted by any Tier 1/2 source found. "
            "(2) The absolute-performance bar implicit in what counts as 'real': Man AHL's "
            "own disclosed historical Sharpe (~0.86, Tier 2) and Asness's explicit "
            "characterization (Tier 2) of a 0.5 Sharpe/IR edge as 'already ambitious' both "
            "suggest notebook 8's carry near-miss (net Sharpe 0.90-0.95, DSR 0.997) sits "
            "comfortably inside, not below, what a real institutional systematic strategy "
            "looks like on an ABSOLUTE-Sharpe basis. "
            "(3) The specific criterion that actually failed carry - a block-bootstrap CI on "
            "excess return VERSUS A PASSIVE BASKET excluding zero - is not the same test as "
            "'has an institutionally acceptable absolute Sharpe', and no Tier 1/2 source "
            "surveyed states that beating a passive basket with a CI-excluding-zero is the "
            "industry-standard funding bar; allocators screen on absolute Sharpe/drawdown/"
            "track-record length (Tier 3 allocator commentary, not counted toward this "
            "verdict, but directionally consistent). Quantpedia's finding (Tier 2) that "
            "real strategies normally lose 1/3-1/2 of backtest Sharpe out-of-sample is a "
            "genuine caution the other direction: carry's true live Sharpe, if it degrades "
            "similarly, could fall toward 0.45-0.6 - near, not comfortably above, even the "
            "loosest institutional screens cited. Verdict: partially supported. The DSR "
            "threshold itself is not shown to be too strict (arguably it is exactly right or "
            "even lenient per Harvey-Liu-Zhu). The excess-vs-basket CI requirement is a "
            "second, additional, non-industry-standard bar stacked on top of an already-"
            "defensible DSR threshold - see the actionable recommendation below."
        ),
        "actionable_recommendation": {
            "summary": (
                "Do NOT lower the deflated-Sharpe-probability threshold (0.95) - the "
                "strongest Tier 1 evidence found (Harvey-Liu-Zhu) argues the finance "
                "literature's traditional bar is too loose, not that this repo's is too "
                "strict, and no Tier 1/2 source argues DSR>0.95 specifically is excessive."
            ),
            "recommendation": (
                "Introduce a SECOND, separately-labelled reporting flag for notebook 10 "
                "onward - 'institutionally fundable absolute performance' - defined "
                "PROSPECTIVELY as: net Sharpe > 0.5 at every tested origin offset, "
                "deflated Sharpe probability > 0.95, and max drawdown within a stated bound "
                "- sourced directly from the Man AHL (Sharpe ~0.86 institutional) and "
                "Asness (~0.5 IR/Sharpe edge is 'ambitious') Tier 2 evidence above. This "
                "flag is ADDITIONAL to, not a replacement for, the existing 'tradeable "
                "alpha' gate (net Sharpe>0 at every offset AND excess-return-vs-passive-"
                "basket CI excludes zero AND DSR>0.95). A strategy can clear the new flag "
                "without clearing the existing gate, and must be reported that way: "
                "'fundable-looking on absolute performance, not shown to beat passive "
                "exposure to the same asset class' - a materially weaker and more honest "
                "claim than 'found alpha'."
            ),
            "consequences_for_existing_nulls_labelled_hypothetical": (
                "HYPOTHETICAL ONLY, not a re-score (docs/08-research-methodology.md's "
                "pre-declared-gates discipline: this flag did not exist when notebooks 3/7/8 "
                "ran, and their existing verdicts under the gate that WAS pre-declared at the "
                "time stand unchanged). Had the new flag existed at the time: notebook 8's "
                "carry (Sharpe 0.90-0.95 at every offset, DSR 0.997) would have cleared it; "
                "notebook 8's momentum (best-lookback Sharpe 0.10-0.12, DSR 0.098) would NOT "
                "have cleared it; notebook 3's cfg2_12h (net Sharpe +0.42 at offset 0, but "
                "-2.45 at offset 7 - fails 'every offset') would NOT have cleared it, since "
                "instability across offsets is exactly the failure mode the 'every offset' "
                "requirement exists to catch; notebook 7's Gate TC (turnover-cut version of "
                "cfg2_12h) would need its own per-offset Sharpe re-checked before any claim "
                "either way - not done here, since this is a labelled hypothetical, not a "
                "re-audit. This is stated explicitly, per docs/08-research-methodology.md's "
                "own warning against moving goalposts after seeing results: the ORIGINAL gate "
                "verdicts (carry: does not fire; momentum: does not fire; cfg2_12h: not "
                "stable) are UNCHANGED and remain this repo's record. The new flag changes "
                "nothing about what has already been reported - it only changes how notebook "
                "10 onward can additionally characterize a FUTURE result that clears DSR and "
                "absolute-Sharpe but not the passive-basket-CI test, giving it an honest, "
                "distinctly-named category instead of forcing a binary fire/no-fire call that "
                "conflates two different questions ('is this a real, repeatable absolute "
                "return' vs. 'does this beat doing nothing more sophisticated')."
            ),
        },
    },
    "d": {
        "name": "The markets are efficient at the horizons/instruments we can reach",
        "verdict": "well-supported",
        "supporting_sources": [
            "mclean_pontiff_2016", "bitcoin_efficiency_scientific_reports",
            "crypto_efficiency_political_uncertainty_2025", "nber_gorton_rouwenhorst",
            "zhu_2024_pairs_trading", "gatev_goetzmann_rouwenhorst_2006",
        ],
        "contrary_or_qualifying_sources": ["jensen_kelly_pedersen_2023"],
        "reasoning": (
            "Multiple independent Tier 1 lines of evidence converge specifically on the "
            "instruments/horizons this repo actually tests. Bitcoin's daily returns are "
            "reported consistent with weak-form efficiency (variance ratios ~1, random walk "
            "not rejected - Nature Scientific Reports, Tier 1); crypto market efficiency is "
            "reported as time-varying around major events but not persistently exploitable "
            "at daily frequency otherwise (Tier 1). Commodity futures' pre-financialization "
            "risk premium (Gorton-Rouwenhorst, Tier 1) is well documented to have decayed "
            "post-2004 - this repo's own commodity sample (2010+) sits entirely inside that "
            "decayed regime, exactly matching notebook 8's own carry/momentum nulls. "
            "Well-known strategies decaying once documented (pairs trading's original "
            "mechanical form, Zhu 2024, Tier 1) is a second, independent replication of the "
            "same story in a completely different strategy family. The one genuine Tier 1 "
            "tension: Jensen-Kelly-Pedersen finds >80% of a large, carefully-constructed "
            "equity factor set still holds internationally, in real conflict with the more "
            "pessimistic McLean-Pontiff reading - reported here as a genuine unresolved "
            "disagreement in the literature, NOT averaged into a false consensus, per sec 6's "
            "explicit instruction. That disagreement is itself about a DIFFERENT universe "
            "(cross-country equities) than this repo's own (liquid crypto majors, front-month "
            "commodity futures) - the instruments this repo actually tests are among the most "
            "heavily arbitraged in existence, which is exactly where the efficiency story is "
            "strongest even within the more optimistic Jensen-Kelly-Pedersen reading."
        ),
    },
    "e": {
        "name": "We have been structurally looking in the wrong place",
        "verdict": "well-supported",
        "supporting_sources": [
            "fed_hedge_fund_treasury_exposures", "ofr_treasury_basis_2021",
            "dallas_fed_basis_funding", "cftc_mrac_basis_trade",
            "gatev_goetzmann_rouwenhorst_2006", "zhu_2024_pairs_trading",
            "avellaneda_stoikov_2008", "hedge_fund_journal_vrp",
        ],
        "contrary_or_qualifying_sources": [],
        "reasoning": (
            "The evidence base for structural/mechanical return sources this repo has never "
            "tested is unusually strong and unusually consistent - a rare case where "
            "multiple Tier 1 regulatory sources (Fed, OFR, Dallas Fed, CFTC) all "
            "independently corroborate that the Treasury cash-futures basis trade alone "
            "represents ~$4tn of real, funded, institutional positioning, explicitly built "
            "on a mechanical mispricing rather than directional prediction - precisely "
            "sec 1's 'never tested at all' category. Structural mean-reversion in "
            "co-integrated pairs survived costs historically (Gatev-Goetzmann-Rouwenhorst) "
            "and adaptive versions reportedly still do (Zhu 2024) - both Tier 1. Market-"
            "making's basic mechanism (Avellaneda-Stoikov, Tier 1) and the volatility risk "
            "premium (Hedge Fund Journal, Tier 2, with explicit, undodged discussion of "
            "catastrophic tail risk - Volmageddon 2018, COVID 2020) round out a genuinely "
            "broad set of real, non-directional return sources. The critical, load-bearing "
            "caveat (not a contradiction of the hypothesis, but essential to reading it "
            "honestly): almost none of this category's BEST-evidenced examples are testable "
            "in this repo as-is. The Treasury basis trade needs repo-financing data and "
            "leveraged margin access this repo has none of; market-making needs L2 order-"
            "book data and low-latency execution this repo has never had; the volatility "
            "risk premium needs options data this repo has never touched at all. The ONE "
            "clear exception - structural mean-reversion in already-existing commodity "
            "spread series - IS directly testable with data already in this repo (see the "
            "Phase 3 shortlist and Phase 4 probe). Verdict: well-supported as an explanation "
            "for why eight directional, daily-horizon, liquid-instrument nulls say very "
            "little about whether alpha exists elsewhere - but 'elsewhere' is mostly walled "
            "off from this repo by infrastructure, not by lack of evidence, and that "
            "distinction is the hypothesis's own most important finding."
        ),
    },
}


def gate_DX(hypotheses):
    well_supported = [k for k, v in hypotheses.items() if v["verdict"] == "well-supported"]
    contradicted = [k for k, v in hypotheses.items() if v["verdict"] == "contradicted"]
    # both must be backed by >=1 tier1/2 source (already true by construction: every
    # hypothesis entry above lists only tier1/2 sources in supporting_sources)
    fires = len(well_supported) >= 1 and len(contradicted) >= 1
    return {
        "well_supported_hypotheses": well_supported,
        "contradicted_hypotheses": contradicted,
        "fires": fires,
        "reasoning": (
            f"Well-supported: {well_supported}. Contradicted: {contradicted}. "
            "The survey discriminates rather than concluding everything is possible: "
            "hypotheses (d) and (e) are well-supported on Tier 1/2 evidence while "
            "hypothesis (b) is contradicted on Tier 1/2 evidence - not every hypothesis "
            "converged to 'partially supported' or 'insufficient evidence'."
            if fires else "Gate does not fire."
        ),
    }


def gate_BAR(hypotheses):
    rec = hypotheses["c"].get("actionable_recommendation")
    has_recommendation = rec is not None and "recommendation" in rec
    has_consequences = rec is not None and "consequences_for_existing_nulls_labelled_hypothetical" in rec
    fires = has_recommendation and has_consequences
    return {
        "has_sourced_recommendation": has_recommendation,
        "states_consequences_for_existing_nulls": has_consequences,
        "fires": fires,
        "summary": (
            "Recommendation: do not lower the DSR>0.95 threshold (Harvey-Liu-Zhu, Tier 1, "
            "argues the literature's traditional bar is too loose, not too strict); instead "
            "add a second, additional 'institutionally fundable absolute performance' flag "
            "(net Sharpe>0.5 every offset, DSR>0.95, bounded drawdown) prospectively from "
            "notebook 10 onward, reported alongside (never instead of) the existing "
            "tradeable-alpha gate. Consequences for existing nulls stated as a labelled "
            "hypothetical only: carry would have cleared the new flag, momentum and "
            "cfg2_12h would not - original verdicts unchanged."
        ),
    }


def main():
    sources_data = load(SOURCES_PATH)
    load(REPRO_PATH)  # loaded to confirm Phase 0 ran first; not otherwise used here
    source_ids = {s["id"] for s in sources_data["sources"]}

    # sanity: every source id referenced in a hypothesis verdict actually exists
    # in the Phase 1 record, and is tier 1 or 2 (this notebook's own discipline
    # check against citing a tier 3/4 source as if it were supporting evidence)
    source_by_id = {s["id"]: s for s in sources_data["sources"]}
    for hkey, h in HYPOTHESES.items():
        for sid in h["supporting_sources"] + h["contrary_or_qualifying_sources"]:
            assert sid in source_ids, f"hypothesis {hkey} cites unknown source {sid}"
            tier = source_by_id[sid]["tier"]
            assert tier in (1, 2), (
                f"hypothesis {hkey} cites tier-{tier} source {sid} as if it were "
                "tier 1/2 evidence - this would be exactly the 'laundering' failure "
                "mode sec 3.1/sec 6 warn against"
            )

    dx = gate_DX(HYPOTHESES)
    bar = gate_BAR(HYPOTHESES)
    out = {"hypotheses": HYPOTHESES, "gate_DX": dx, "gate_BAR": bar}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    for hkey, h in HYPOTHESES.items():
        print(f"({hkey}) {h['name']}: {h['verdict'].upper()}")
    print()
    print("Gate DX:", json.dumps({k: v for k, v in dx.items() if k != "reasoning"}, indent=2))
    print(dx["reasoning"])
    print()
    print("Gate BAR:", json.dumps(bar, indent=2))
    print(f"\nwritten {OUT_PATH}")


if __name__ == "__main__":
    main()
