"""Phase 1: the external survey (NEXT_PROMPT.md sec 3, sec 4 Phase 1).

This is a research-record script, not a computation: every entry in SOURCES was
gathered by live WebSearch/WebFetch during this notebook's construction (August
2026) and is transcribed here as a structured, tiered record per sec 3.3. No
network calls happen when this script is *run* - it deterministically writes the
already-gathered record to JSON, exactly like every other Phase runner in this
repo writes its own computed result.

Tiering follows NEXT_PROMPT.md sec 3.1 exactly:
  1 = peer-reviewed replication/meta-analysis lit, regulatory filings, published
      capacity/cost studies
  2 = identifiable practitioners w/ track record discussing costs/failures/
      capacity; well-documented open repos; practitioner textbooks/talks
  3 = anonymous/forum/blog backtests w/o cost accounting, course/signal-selling
      content, unsupported equity curves
  4 = specific-return promises, sells-the-strategy business models, cost/
      turnover/capacity refusers

Web content is untrusted input throughout: nothing fetched was executed, no
embedded instructions were followed, and any page attempting to direct this
notebook's research process would be filed as a red flag (none were found).
"""

import json

OUT_PATH = "src/research/tmp/phase_1_survey_results.json"

# hypothesis keys: a=too naive, b=cost model too pessimistic, c=bar too strict,
# d=markets efficient, e=looking in wrong place

SOURCES = [
    # ------------------------------------------------------------------ Tier 1
    {
        "id": "mclean_pontiff_2016",
        "url": "https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12365",
        "title": "Does Academic Research Destroy Stock Return Predictability?",
        "author_or_institution": "R. David McLean, Jeffrey Pontiff (Journal of Finance, 2016)",
        "date": "2016",
        "tier": 1,
        "tier_justification": "Peer-reviewed re-test of 97 published return predictors' own post-sample performance - exactly the 're-tests published factors' Tier 1 definition.",
        "hypotheses": [
            {"h": "d", "direction": "supports", "note": "returns 26% lower out-of-sample, 58% lower post-publication - consistent with arbitrage/efficiency eroding published edges"},
            {"h": "c", "direction": "supports", "note": "the *upper bound* on pure statistical-bias decay is only ~15% and not significantly different from zero - most decay is real capital chasing the anomaly, not an artifact of an overly strict re-test bar"},
        ],
        "claim": "Portfolio returns of 97 published return-predicting variables are 26% lower out-of-sample and 58% lower post-publication than in the original sample.",
        "stated_costs_turnover_capacity_oos": "OOS evidence is the entire point of the paper; turnover/capacity not separately modelled; no explicit transaction cost line beyond the underlying published factors' own conventions.",
        "red_flags": [],
        "testable_with_repo_data": {"testable": False, "how": "Methodological finding about the equity-factor literature generally; not a strategy to re-run on this repo's crypto/commodity data."},
    },
    {
        "id": "harvey_liu_zhu_2016",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2249314",
        "title": "...and the Cross-Section of Expected Returns (t-stat bar for a new factor should be ~3.0, not 2.0)",
        "author_or_institution": "Campbell R. Harvey, Yan Liu, Heqing Zhu (Review of Financial Studies, 2016)",
        "date": "2016",
        "tier": 1,
        "tier_justification": "Peer-reviewed, explicit topic is multiple-testing correction for a large factor search (316 factors) - the canonical Tier 1 'why published strategies fail out of sample' paper.",
        "hypotheses": [
            {"h": "c", "direction": "contradicts", "note": "argues the finance literature's own historical bar (t>2.0) is too LOOSE, not too strict - the opposite direction from 'our bar is too strict'; if anything this is evidence for tightening, not loosening"},
            {"h": "d", "direction": "supports", "note": "most of a large factor zoo is plausibly false discovery once corrected for the true number of trials"},
        ],
        "claim": "After correcting for the true number of factors tested in the finance literature (316 at the time), a credible new factor needs a t-statistic near 3.0, not the conventional 2.0.",
        "stated_costs_turnover_capacity_oos": "Not a trading-cost paper; purely a statistical multiple-testing correction over published t-stats.",
        "red_flags": [],
        "testable_with_repo_data": {"testable": False, "how": "A statistical-methodology reference, not a strategy."},
    },
    {
        "id": "jensen_kelly_pedersen_2023",
        "url": "https://onlinelibrary.wiley.com/doi/10.1111/jofi.13249",
        "title": "Is There a Replication Crisis in Finance?",
        "author_or_institution": "Theis Ingerslev Jensen, Bryan Kelly, Lasse Heje Pedersen (Journal of Finance, 2023)",
        "date": "2023",
        "tier": 1,
        "tier_justification": "Peer-reviewed replication study across 153 factors in 93 countries with a standardized construction methodology - exactly the Tier 1 replication-literature definition.",
        "hypotheses": [
            {"h": "a", "direction": "supports", "note": "over 80% of factors remain significant once construction is made consistent/implementable - i.e. HOW a factor is engineered (not just which factor) materially changes whether it survives"},
            {"h": "d", "direction": "contradicts", "note": "finds much less decay than the 'replication crisis' narrative implies once construction choices are standardized - a genuine tension with McLean-Pontiff's more pessimistic reading, reported as a conflict rather than averaged away"},
        ],
        "claim": "With careful, consistent, implementable factor construction, >80% of previously-published equity factors remain statistically significant out-of-sample, across 93 countries.",
        "stated_costs_turnover_capacity_oos": "Discusses implementability explicitly (trading-cost-aware construction) but full transaction-cost/capacity modelling is a secondary concern of the paper, not its headline number.",
        "red_flags": [],
        "testable_with_repo_data": {"testable": False, "how": "Equity cross-country factor library; no equity data of any kind exists in this repo."},
    },
    {
        "id": "bailey_lopez_de_prado_dsr",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551",
        "title": "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality",
        "author_or_institution": "David H. Bailey, Marcos Lopez de Prado",
        "date": "2014 (working paper; published Journal of Portfolio Management)",
        "tier": 1,
        "tier_justification": "The peer-reviewed methodology paper this repo's own `deflated_sharpe_prob` implements - primary source, not a secondary retelling.",
        "hypotheses": [
            {"h": "c", "direction": "neutral", "note": "defines the exact correction this repo uses; does not itself say what probability threshold ('>0.95') should be treated as a pass - that threshold choice is this repo's own convention, not something the source paper prescribes"},
        ],
        "claim": "A Sharpe ratio should be deflated to account for the number of trials searched and the true (non-normal) skew/kurtosis of the return series before being trusted as evidence of real skill.",
        "stated_costs_turnover_capacity_oos": "Purely a statistical-inference correction; no cost/turnover/capacity content.",
        "red_flags": [],
        "testable_with_repo_data": {"testable": False, "how": "Already implemented and used throughout this repo (notebooks 3, 7, 8)."},
    },
    {
        "id": "portfoliooptimizer_psr_mtrl",
        "url": "https://portfoliooptimizer.io/blog/the-probabilistic-sharpe-ratio-bias-adjustment-confidence-intervals-hypothesis-testing-and-minimum-track-record-length/",
        "title": "The Probabilistic Sharpe Ratio: Bias-Adjustment, Confidence Intervals, Hypothesis Testing and Minimum Track Record Length",
        "author_or_institution": "Portfolio Optimizer (quant tooling vendor, summarizing Bailey/Lopez de Prado's PSR/MinTRL literature)",
        "date": "not stated on page",
        "tier": 2,
        "tier_justification": "Practitioner/vendor exposition of a peer-reviewed method, not the primary paper itself, and the vendor has a commercial interest in the tooling described - useful with care, not primary evidence.",
        "hypotheses": [
            {"h": "c", "direction": "neutral", "note": "reinforces that DSR/PSR-style corrections are standard practitioner tooling for judging a Sharpe ratio's credibility, not an eccentric academic-only bar"},
        ],
        "claim": "Practitioner-grade quant tooling implements PSR/minimum-track-record-length alongside DSR as standard due-diligence statistics.",
        "stated_costs_turnover_capacity_oos": "Not addressed; purely a statistical-methodology page.",
        "red_flags": [],
        "testable_with_repo_data": {"testable": False, "how": "Confirms an existing methodology choice; nothing new to test."},
    },
    {
        "id": "fed_hedge_fund_treasury_exposures",
        "url": "https://www.federalreserve.gov/econres/notes/feds-notes/decomposing-hedge-funds-u-s-treasury-exposures-20260622.html",
        "title": "Decomposing Hedge Funds' U.S. Treasury Exposures",
        "author_or_institution": "Federal Reserve Board (FEDS Notes)",
        "date": "2026-06-22",
        "tier": 1,
        "tier_justification": "Primary regulatory filing/analysis of aggregate positioning data - the Tier 1 'regulatory filings and fund disclosures' category by definition.",
        "hypotheses": [
            {"h": "e", "direction": "supports", "note": "hedge funds' gross US Treasury exposure (driven substantially by the cash-futures basis trade) doubled to $4.0tn (2023-Sep 2025) - real, large-scale money is made in structural basis trades this repo has never touched"},
        ],
        "claim": "Large hedge funds' gross US Treasury exposure reached ~$4.0tn by Sept 2025, split ~$2.4tn long / $1.6tn short, with the cash-futures basis trade a key driver.",
        "stated_costs_turnover_capacity_oos": "Discusses leverage and repo-funding sensitivity explicitly; does not state a Sharpe ratio for the trade (this is a stability/exposure report, not a performance report).",
        "red_flags": [],
        "testable_with_repo_data": {"testable": False, "how": "Requires repo-market financing data and Treasury cash-bond data this repo has none of; also enormously capital/leverage-dependent."},
    },
    {
        "id": "ofr_treasury_basis_2021",
        "url": "https://www.financialresearch.gov/working-papers/files/OFRwp-21-01-hedge-funds-and-the-treasury-cash-futures-disconnect.pdf",
        "title": "Hedge Funds and the Treasury Cash-Futures Disconnect",
        "author_or_institution": "Office of Financial Research (US Treasury)",
        "date": "2021-04-01",
        "tier": 1,
        "tier_justification": "US Treasury regulatory working paper analyzing the mechanics and scale of the basis trade directly from position data.",
        "hypotheses": [
            {"h": "e", "direction": "supports", "note": "documents the basis trade as a real, scaled, structural (mechanical) source of return - exactly the 'P&L from a structural source rather than predicting direction' category sec 1 says this repo has never tested"},
        ],
        "claim": "The Treasury cash-futures basis trade is a large, leveraged, repo-financed arbitrage exploiting a persistent futures-cash mispricing, not a directional bet.",
        "stated_costs_turnover_capacity_oos": "Extensively discusses repo funding cost sensitivity and leverage (the trade's real 'cost model' is financing rate risk, not bps/trade); capacity is implicitly enormous ($tn) but requires prime-brokerage-grade leverage access.",
        "red_flags": [],
        "testable_with_repo_data": {"testable": False, "how": "Needs repo/funding-rate data, Treasury cash-bond prices, and leveraged margin access this repo has none of - explicitly infrastructure this repo lacks."},
    },
    {
        "id": "dallas_fed_basis_funding",
        "url": "https://www.dallasfed.org/research/economics/2025/0715",
        "title": "How sensitive is the Treasury cash-futures basis trade to funding condition shifts?",
        "author_or_institution": "Federal Reserve Bank of Dallas",
        "date": "2025-07-15",
        "tier": 1,
        "tier_justification": "Regional Fed research note, regulatory/official analysis.",
        "hypotheses": [
            {"h": "e", "direction": "supports", "note": "the trade's stability is more sensitive to intermediation (dealer balance sheet) capacity than to funding-rate level per se - a capacity/infrastructure-first story, reinforcing that this category's edge is inseparable from access most retail-scale researchers don't have"},
        ],
        "claim": "The basis trade's fragility comes primarily from dealer intermediation-capacity constraints, not funding-rate levels alone.",
        "stated_costs_turnover_capacity_oos": "Capacity is the explicit subject of the note.",
        "red_flags": [],
        "testable_with_repo_data": {"testable": False, "how": "Same infrastructure gap as the other basis-trade sources above."},
    },
    {
        "id": "cftc_mrac_basis_trade",
        "url": "https://www.cftc.gov/media/11671/mrac121024_TreasuryCashFuturesBasisTrade/download",
        "title": "The Treasury Cash-Futures Basis Trade and Effective Risk Management",
        "author_or_institution": "CFTC Market Risk Advisory Committee",
        "date": "2024-12-10",
        "tier": 1,
        "tier_justification": "Regulatory committee document.",
        "hypotheses": [
            {"h": "e", "direction": "supports", "note": "a regulator-convened committee treating this as a live systemic-risk topic underlines both the trade's real scale and its inseparability from leverage/margin infrastructure"},
        ],
        "claim": "The basis trade's scale and leverage are large enough to be a standing financial-stability agenda item.",
        "stated_costs_turnover_capacity_oos": "Risk-management framing throughout; not a per-trade cost model.",
        "red_flags": [],
        "testable_with_repo_data": {"testable": False, "how": "Same as above."},
    },
    {
        "id": "gatev_goetzmann_rouwenhorst_2006",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=141615",
        "title": "Pairs Trading: Performance of a Relative Value Arbitrage Rule",
        "author_or_institution": "Evan Gatev, William N. Goetzmann, K. Geert Rouwenhorst (Review of Financial Studies, 2006)",
        "date": "2006 (sample 1962-2002)",
        "tier": 1,
        "tier_justification": "Peer-reviewed, canonical stat-arb replication study, reports the strategy net of conservative transaction cost estimates.",
        "hypotheses": [
            {"h": "e", "direction": "supports", "note": "a structural/mean-reversion mechanism (co-integrated pairs) genuinely survived transaction costs historically - a category this repo has never tested at all"},
            {"h": "d", "direction": "supports", "note": "the same category's modern-era decay (see Zhu 2024 below) is consistent with the efficiency/arbitraged-away story once a strategy is well known"},
        ],
        "claim": "Top pairs-trading portfolios earned up to ~11%/yr excess return net of conservative transaction cost estimates, 1962-2002.",
        "stated_costs_turnover_capacity_oos": "Explicit conservative transaction-cost estimates stated; turnover is inherent to the daily-rebalance construction; no capacity figure stated.",
        "red_flags": [],
        "testable_with_repo_data": {"testable": True, "how": "This repo has no equities, but the same co-integration mechanism is directly testable on the 16-commodity panel's related products (e.g. CL-BZ, GC-SI) and the 30 pre-built spread series never yet backtested - flagged in the Phase 3 shortlist below."},
    },
    {
        "id": "zhu_2024_pairs_trading",
        "url": "https://economics.yale.edu/sites/default/files/2024-05/Zhu_Pairs_Trading.pdf",
        "title": "Examining Pairs Trading Profitability",
        "author_or_institution": "Xuanchi Zhu (Yale Economics)",
        "date": "2024-04-03",
        "tier": 1,
        "tier_justification": "Academic working paper directly re-testing the Gatev-Goetzmann-Rouwenhorst rule out to the present.",
        "hypotheses": [
            {"h": "d", "direction": "supports", "note": "the original mechanical GGR method no longer delivers robust profits - a well-known strategy decaying is exactly the efficiency/arbitraged-away story"},
            {"h": "e", "direction": "partially supports", "note": "more adaptive, model-driven implementations reportedly still work - the mechanism isn't dead, the naive textbook version is, echoing hypothesis (a) as much as (e)"},
        ],
        "claim": "The original 1990s-2000s mechanical pairs-trading rule no longer delivers robust risk-adjusted returns; more adaptive implementations that account for changing vol/correlation/liquidity reportedly still work.",
        "stated_costs_turnover_capacity_oos": "Frames profitability as 'much more sensitive to transaction costs and execution' post-decay, without giving a single all-in cost figure.",
        "red_flags": [],
        "testable_with_repo_data": {"testable": True, "how": "Same as the GGR entry above - directly testable on this repo's commodity spread data."},
    },
    {
        "id": "square_root_impact_tse_survey",
        "url": "https://arxiv.org/pdf/2411.13965",
        "title": "Strict universality of the square-root law in price impact across stocks: a complete survey of the Tokyo stock exchange",
        "author_or_institution": "Academic authors (Tokyo Stock Exchange complete-market survey)",
        "date": "2024",
        "tier": 1,
        "tier_justification": "Full-market empirical survey of a well-established, peer-reviewed market-microstructure literature (the Bouchaud/Toth square-root impact law tradition), not a single blog's claim.",
        "hypotheses": [
            {"h": "b", "direction": "neutral", "note": "cost/impact scales with sqrt(volume/ADV), roughly independent of instrument/venue - a genuine, quantitative alternative cost model this repo's flat-bps assumption doesn't use, but it needs an ADV/participation-rate input this repo's cost model has never estimated"},
        ],
        "claim": "Price impact of a metaorder of size Q scales as ~sigma*sqrt(Q/ADV), confirmed across a complete exchange's stock universe and, per the wider literature this survey sits in, across futures and options too.",
        "stated_costs_turnover_capacity_oos": "The entire paper is a cost/impact model; capacity is implicit (impact grows, doesn't cap, as size grows - a 'soft' capacity signal).",
        "red_flags": [],
        "testable_with_repo_data": {"testable": False, "how": "This repo's crypto/commodity data has no participation-rate or ADV-relative order-size information to calibrate sqrt-impact against; the existing flat bps+tick model is a coarser but not obviously wrong substitute at the position sizes implied by this repo's backtests."},
    },
    {
        "id": "capacity_of_trading_strategies_aea",
        "url": "https://www.aeaweb.org/conference/2016/retrieve.php?pdfid=21020&tk=BGQnasd4",
        "title": "The Capacity of Trading Strategies",
        "author_or_institution": "Academic conference paper (AEA)",
        "date": "2016",
        "tier": 1,
        "tier_justification": "Peer-reviewed-track academic conference paper explicitly about capacity limits - the Tier 1 'published capacity/cost studies' category.",
        "hypotheses": [
            {"h": "e", "direction": "supports", "note": "formalizes that a strategy's realistic capacity (not just its unconstrained Sharpe) is a first-order determinant of whether it is fundable/tradeable at all - directly relevant to filtering the Phase 3 shortlist"},
        ],
        "claim": "Strategy capacity - how much capital can be deployed before returns compress to the cost of trading - is a formal, modelable constraint distinct from raw signal quality.",
        "stated_costs_turnover_capacity_oos": "Capacity is the paper's subject.",
        "red_flags": [],
        "testable_with_repo_data": {"testable": False, "how": "A framework, not a specific strategy; informs how the shortlist should be assessed rather than being itself testable."},
    },
    {
        "id": "avellaneda_stoikov_2008",
        "url": "https://web.stanford.edu/class/msande448/2018/Final/Reports/gr5.pdf",
        "title": "High-Frequency Trading in a Limit Order Book (Avellaneda-Stoikov model)",
        "author_or_institution": "Marco Avellaneda, Sasha Stoikov (2008); linked page is a Stanford course report applying the model",
        "date": "2008 (original); course report undated",
        "tier": 1,
        "tier_justification": "The original Avellaneda-Stoikov paper is the peer-reviewed theoretical foundation of essentially all modern inventory-risk market-making; cited here via the widely-used derivation, tiered on the strength of the original paper's status, not the course report's rigor (flagged explicitly).",
        "hypotheses": [
            {"h": "e", "direction": "supports", "note": "gives the formal mechanism by which market-making earns the spread while managing inventory risk - a structural P&L source this repo has never tested, exactly sec 1's 'never tested at all' list"},
        ],
        "claim": "A market maker earns the bid-ask spread by continuously quoting both sides and skewing quotes away from symmetric as inventory accumulates, to manage the risk of a directional move against an unwanted position.",
        "stated_costs_turnover_capacity_oos": "The model assumes but doesn't itself measure exchange fees/rebates or capacity; capacity in market-making is a function of quoted size vs. queue position, not addressed quantitatively here.",
        "red_flags": ["Secondary application note (Stanford course project) tiered down mentally from the primary 2008 paper - the specific numeric results in the course report itself should not be treated as Tier 1."],
        "testable_with_repo_data": {"testable": False, "how": "Requires L2/LOB (order book) data and colocated/low-latency execution this repo has never had - explicitly not testable here."},
    },
    {
        "id": "bitcoin_efficiency_scientific_reports",
        "url": "https://www.nature.com/articles/s41598-023-31618-4",
        "title": "Market efficiency of cryptocurrency: evidence from the Bitcoin market",
        "author_or_institution": "Academic authors (Scientific Reports / Nature portfolio, peer-reviewed)",
        "date": "2023",
        "tier": 1,
        "tier_justification": "Peer-reviewed journal article directly testing weak-form market efficiency on Bitcoin.",
        "hypotheses": [
            {"h": "d", "direction": "supports", "note": "reports Bitcoin daily returns consistent with weak-form efficiency (variance ratios near 1, random walk not rejected)"},
        ],
        "claim": "Daily Bitcoin returns are broadly consistent with weak-form market efficiency; efficiency is time-varying and regulation-sensitive rather than a fixed property.",
        "stated_costs_turnover_capacity_oos": "Not a trading-strategy paper; no cost/turnover figures.",
        "red_flags": [],
        "testable_with_repo_data": {"testable": False, "how": "A market-efficiency finding, not itself a strategy."},
    },
    {
        "id": "crypto_efficiency_political_uncertainty_2025",
        "url": "https://www.tandfonline.com/doi/full/10.1080/15140326.2025.2551627",
        "title": "Political uncertainty and cryptocurrency futures and spot market efficiency: evidence from the 2024 U.S. presidential election",
        "author_or_institution": "Academic authors, peer-reviewed (Taylor & Francis journal)",
        "date": "2025",
        "tier": 1,
        "tier_justification": "Peer-reviewed journal article.",
        "hypotheses": [
            {"h": "d", "direction": "partially supports", "note": "efficiency is time-varying around major events, i.e. mostly efficient but with event-driven windows of inefficiency this repo's daily-bar, non-event-conditioned strategies would not be positioned to exploit anyway"},
        ],
        "claim": "Crypto futures/spot market efficiency shifts measurably around a major political-uncertainty event (the 2024 US election).",
        "stated_costs_turnover_capacity_oos": "Not addressed; an efficiency-testing paper.",
        "red_flags": [],
        "testable_with_repo_data": {"testable": False, "how": "Event-study design this repo's daily-bar crypto data could technically support, but not attempted here - filed as a hypothesis-generation note rather than a shortlist candidate given its narrow, single-event scope."},
    },
    {
        "id": "nber_gorton_rouwenhorst",
        "url": "https://www.nber.org/system/files/working_papers/w11222/w11222.pdf",
        "title": "The Tactical and Strategic Value of Commodity Futures",
        "author_or_institution": "Gary Gorton, K. Geert Rouwenhorst (NBER Working Paper)",
        "date": "2005",
        "tier": 1,
        "tier_justification": "NBER working paper, the direct predecessor literature to the Bhardwaj-Gorton-Rouwenhorst decade-later update NEXT_PROMPT.md sec 2 already cites.",
        "hypotheses": [
            {"h": "d", "direction": "supports", "note": "establishes the pre-financialization commodity risk-premium baseline against which this repo's own post-2010 carry/momentum nulls (notebook 8) should be read as consistent with a decayed, financialized regime rather than a surprising failure"},
        ],
        "claim": "Commodity futures historically carried a real risk premium comparable to equities before the mid-2000s financialization wave.",
        "stated_costs_turnover_capacity_oos": "Academic index-return level analysis; no retail/institutional cost model.",
        "red_flags": [],
        "testable_with_repo_data": {"testable": False, "how": "Historical/context paper; this repo's own sample (2010+) is already inside the post-financialization regime the paper's own follow-on literature describes."},
    },
    {
        "id": "barroso_santa_clara_2015",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2041429",
        "title": "Momentum Has Its Moments",
        "author_or_institution": "Pedro Barroso, Pedro Santa-Clara (Journal of Financial Economics, 2015)",
        "date": "2015",
        "tier": 1,
        "tier_justification": "Peer-reviewed paper quantifying a specific engineering gap (vol-scaling) between a textbook factor and its risk-managed implementation - directly on-point for hypothesis (a).",
        "hypotheses": [
            {"h": "a", "direction": "supports", "note": "scaling equity momentum by inverse trailing realized vol nearly doubles its Sharpe ratio (0.53 -> 0.97) and slashes crash risk (excess kurtosis 18.24 -> 2.68) - a quantified, specific version of 'the textbook factor and the implemented factor are not the same thing'"},
        ],
        "claim": "Volatility-scaling a momentum strategy by its own trailing 6-month realized vol improves Sharpe from 0.53 to 0.97 and materially reduces crash risk, versus the unscaled version.",
        "stated_costs_turnover_capacity_oos": "Academic equity-factor paper; transaction costs of the rebalancing implied by vol-scaling are not the paper's central focus (vol-scaling itself adds turnover, a real caveat this repo would need to price in before adopting the technique).",
        "red_flags": [],
        "testable_with_repo_data": {"testable": True, "how": "This repo's own commodity carry/momentum (notebook 8) used constant, not vol-scaled, position sizing - a vol-scaled re-run of the existing carry/momentum signals is directly buildable from data already in this repo, flagged as a shortlist candidate below."},
    },
    {
        "id": "man_group_vol_targeting",
        "url": "https://www.man.com/insights/the-impact-of-volatility-targeting",
        "title": "The Impact of Volatility Targeting",
        "author_or_institution": "Man Group (Man Institute research)",
        "date": "not stated on retrieved excerpt",
        "tier": 2,
        "tier_justification": "Identifiable, disclosed-track-record institutional manager's own research, tested across 60+ assets and explicitly reporting where the technique does NOT help (bonds/commodities/FX) rather than only where it does - the Tier 2 'discusses failures as well as successes' standard.",
        "hypotheses": [
            {"h": "a", "direction": "partially supports", "note": "vol targeting improves Sharpe for equities/credit (0.40 -> 0.48-0.51) but has NEGLIGIBLE impact for commodities, currencies, and bonds specifically - directly cautions against assuming this repo's commodity notebook 8 would gain much from the same technique that helped equity momentum in the Barroso-Santa-Clara source above"},
        ],
        "claim": "Volatility targeting improves risk-adjusted returns mainly for equity/credit-like 'risk assets' exhibiting a leverage effect; the benefit is negligible for bonds, currencies, and commodities.",
        "stated_costs_turnover_capacity_oos": "Discusses drawdown and tail-risk reduction as the main benefit channel; does not isolate the added turnover cost of a vol-targeting overlay from the reported net Sharpe figures.",
        "red_flags": [],
        "testable_with_repo_data": {"testable": True, "how": "Directly contradicts the naive read of the Barroso-Santa-Clara source for THIS repo's specific asset classes (commodities/crypto, not equities) - an important caution recorded before any vol-scaling probe is proposed."},
    },
    {
        "id": "novy_marx_velikov_2016",
        "url": "https://academic.oup.com/rfs/article-abstract/29/1/104/1844518",
        "title": "A Taxonomy of Anomalies and Their Trading Costs",
        "author_or_institution": "Robert Novy-Marx, Mihail Velikov (Review of Financial Studies, 2016)",
        "date": "2016",
        "tier": 1,
        "tier_justification": "Peer-reviewed, explicit topic is realistic all-in trading costs for 23 published equity anomalies - the Tier 1 'published capacity/cost studies' category.",
        "hypotheses": [
            {"h": "b", "direction": "contradicts", "note": "full-sample all-in implementation costs for momentum are estimated at 7.2-7.6%/yr, considerably LARGER than bid-ask-spread-only estimates, and large enough to eliminate most 1970-2016 momentum profit - evidence that realistic costs are often materially HIGHER than naive assumptions, the opposite direction from 'our cost model is too pessimistic'"},
        ],
        "claim": "Anomalies with under 50% monthly turnover mostly survive realistic implementation costs; high-turnover strategies like momentum mostly do not once full-cost (not just bid-ask) estimates are used.",
        "stated_costs_turnover_capacity_oos": "The paper's entire content is a per-strategy cost/turnover taxonomy - the most directly on-topic cost source found in this survey.",
        "red_flags": [],
        "testable_with_repo_data": {"testable": False, "how": "Equity-anomaly cost estimates specifically; not directly transferable to crypto/commodity cost assumptions, but the general lesson (naive cost estimates often UNDERSTATE true cost) argues against loosening this repo's own cost model."},
    },
    {
        "id": "chen_velikov_2023",
        "url": "https://www.sciencedirect.com/science/article/abs/pii/S0304405X20300453",
        "title": "What you see is not what you get: The costs of trading market anomalies",
        "author_or_institution": "Andrew Y. Chen, Mihail Velikov (Journal of Financial Economics, 2023)",
        "date": "2023",
        "tier": 1,
        "tier_justification": "Peer-reviewed follow-up cost study, same Tier 1 'published capacity/cost studies' category as Novy-Marx-Velikov above, extending the analysis to a much larger anomaly set.",
        "hypotheses": [
            {"h": "b", "direction": "contradicts", "note": "reinforces that published (in-sample, often cost-light) anomaly returns systematically overstate what a real trader could capture net of realistic costs - again arguing this repo's cost assumptions are unlikely to be the too-harsh direction of error"},
        ],
        "claim": "Net-of-cost returns to a broad set of published anomalies are systematically and substantially lower than the gross, in-sample-published figures.",
        "stated_costs_turnover_capacity_oos": "Cost modelling is the paper's explicit subject.",
        "red_flags": [],
        "testable_with_repo_data": {"testable": False, "how": "Equity-anomaly specific; directional lesson only for this repo's own cost-model calibration."},
    },
    # ------------------------------------------------------------------ Tier 2
    {
        "id": "asness_ritholtz_transcript",
        "url": "https://ritholtz.com/2023/03/transcript-cliff-asness/",
        "title": "Transcript: Cliff Asness - The Big Picture",
        "author_or_institution": "Cliff Asness (AQR co-founder/CIO), interviewed by Barry Ritholtz",
        "date": "2023-03",
        "tier": 2,
        "tier_justification": "Identifiable practitioner with a multi-decade, disclosed track record, discussing realistic expectations and skepticism of extreme claims - exactly the Tier 2 definition, though AQR sells factor products so is not disinterested.",
        "hypotheses": [
            {"h": "c", "direction": "supports", "note": "explicit skepticism of very high Sharpe claims ('why do you think you're better than Medallion') and framing a 0.5 information-ratio edge over a good existing strategy as already ambitious - directly informs what 'a real result' looks like at the top of the industry"},
        ],
        "claim": "Realistic, credible edges in quant investing are modest (IR/Sharpe differentials around 0.5 relative to strong existing strategies are already ambitious); very high claimed Sharpe ratios should be treated with default skepticism.",
        "stated_costs_turnover_capacity_oos": "Discusses long-run mean reversion of factor premia and the danger of overfitting to a short backtest window; no specific cost figures given (interview format).",
        "red_flags": [],
        "testable_with_repo_data": {"testable": False, "how": "An expert opinion informing Phase 2's bar recommendation, not a testable strategy."},
    },
    {
        "id": "aqr_century_trend_following",
        "url": "https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing",
        "title": "A Century of Evidence on Trend-Following Investing",
        "author_or_institution": "Brian Hurst, Yao Hua Ooi, Lasse Heje Pedersen (AQR; published Journal of Portfolio Management)",
        "date": "2017",
        "tier": 2,
        "tier_justification": "Practitioner-authored (AQR sells trend-following products - a real conflict of interest) but published in a peer-reviewed practitioner journal with an unusually long, transparent 137-year sample; downgraded from Tier 1 to Tier 2 specifically for the commercial interest, per sec 3.1's discipline against laundering an interesting claim upward.",
        "hypotheses": [
            {"h": "a", "direction": "supports", "note": "time-series momentum, applied consistently across 67 markets/4 asset classes over 137 years, is the kind of broad, engineered, multi-market implementation this repo's notebook 8 tested only as a single-lookback, single-asset-class (commodities-only), 15-year-sample version of"},
        ],
        "claim": "A diversified, cross-asset-class, multi-lookback time-series momentum portfolio (67 markets, 1880-2016) delivered positive average returns in every decade, improving a 60/40 portfolio's Sharpe from 0.39 to 0.55 with a 20% allocation.",
        "stated_costs_turnover_capacity_oos": "Reports gross returns with 'transaction cost adjustments' referenced generally but no single all-in bps figure is prominent in available excerpts; capacity not quantified.",
        "red_flags": ["Published by a firm that sells the exact strategy described - a conflict of interest, not a disqualifier, but reason for the Tier 2 (not 1) placement."],
        "testable_with_repo_data": {"testable": True, "how": "This repo's notebook 8 Gate AM already tested a narrower version (commodities-only, 4 single lookbacks, no cross-asset blending) and found it sign-inconsistent - the multi-lookback BLEND (not any single lookback) is the specific, not-yet-tested variant this source suggests."},
    },
    {
        "id": "aqr_demystifying_managed_futures",
        "url": "https://www.aqr.com/-/media/AQR/Documents/Insights/Journal-Article/Demystifying-Managed-Futures.pdf",
        "title": "Demystifying Managed Futures",
        "author_or_institution": "AQR Capital Management",
        "date": "not stated on retrieved excerpt",
        "tier": 2,
        "tier_justification": "Practitioner white paper from an identifiable, disclosed-track-record firm; commercial interest noted, same Tier 2 downgrade logic as the century-of-evidence entry.",
        "hypotheses": [
            {"h": "a", "direction": "supports", "note": "practitioner CTA construction blends signals and asset classes in ways textbook single-factor backtests (like this repo's own) do not"},
        ],
        "claim": "Managed-futures/CTA returns are best explained as a diversified, systematically-implemented trend exposure, not an idiosyncratic manager skill story.",
        "stated_costs_turnover_capacity_oos": "General discussion of fee drag on managed-futures vehicles; no specific bps figure retrieved.",
        "red_flags": ["Same commercial-interest caveat as the century-of-evidence AQR paper."],
        "testable_with_repo_data": {"testable": False, "how": "Descriptive/explanatory paper, not a specific testable construction beyond what's already covered by the century-of-evidence entry."},
    },
    {
        "id": "man_ahl_track_record",
        "url": "https://www.man.com/ahl",
        "title": "Man AHL (firm track record and strategy overview)",
        "author_or_institution": "Man Group / Man AHL",
        "date": "ongoing, page reflects 2026 state",
        "tier": 2,
        "tier_justification": "A real, disclosed, multi-decade institutional track record from an identifiable manager - the clearest Tier 2 'here is what a funded institutional strategy's Sharpe actually looks like' data point found in this survey.",
        "hypotheses": [
            {"h": "c", "direction": "supports", "note": "AHL Diversified's own historical Sharpe ratio is reported at 0.86 (1996-2009) - a real, large, long-running, institutionally-funded systematic strategy running at a Sharpe below this repo's carry near-miss (0.90-0.95) is direct, if secondary-sourced, evidence that Sharpe 0.9 is not obviously below the industry's own funding bar"},
        ],
        "claim": "Man AHL Diversified (trend, ~400 liquid futures/FX markets) has a historical Sharpe ratio of ~0.86 and max drawdown -17.9% over 1996-2009, and remains an actively marketed, multi-billion-AUM strategy today.",
        "stated_costs_turnover_capacity_oos": "AUM implies real institutional capacity (multi-billion); explicit per-trade cost/turnover figures not found in the retrieved excerpt (would require a fund factsheet, not attempted here).",
        "red_flags": ["Sourced via a secondary substack/PDF summary rather than Man Group's own primary factsheet for the specific 0.86 figure - flagged as a secondary-sourcing caveat even though Man AHL itself is unambiguously a real, identifiable institution."],
        "testable_with_repo_data": {"testable": False, "how": "A benchmark data point for Phase 2's bar discussion, not a strategy to re-run."},
    },
    {
        "id": "alpha_architect_replication",
        "url": "https://alphaarchitect.com/is-there-a-replication-crisis-in-finance/",
        "title": "Is There a Replication Crisis in Finance?",
        "author_or_institution": "Alpha Architect (asset manager; identifiable authors, publicly disclosed research process)",
        "date": "not stated on retrieved excerpt",
        "tier": 2,
        "tier_justification": "Practitioner commentary from a firm that publishes its own research methodology and discusses replication failures candidly, summarizing Jensen-Kelly-Pedersen (Tier 1) rather than substituting for it.",
        "hypotheses": [
            {"h": "d", "direction": "neutral", "note": "summarizes the Tier 1 finding without adding independent evidence; used here only as a secondary confirmation, not counted toward hyp (d)'s Tier 1/2 support independently of the primary paper"},
        ],
        "claim": "Practitioner summary of the Jensen-Kelly-Pedersen replication finding for a non-academic audience.",
        "stated_costs_turnover_capacity_oos": "Inherits whatever the underlying paper states; no independent figures added.",
        "red_flags": [],
        "testable_with_repo_data": {"testable": False, "how": "Secondary commentary."},
    },
    {
        "id": "quantpedia_oos_analysis",
        "url": "https://quantpedia.com/in-sample-vs-out-of-sample-analysis-of-trading-strategies/",
        "title": "In-Sample vs. Out-Of-Sample Analysis of Trading Strategies",
        "author_or_institution": "Quantpedia (a large, publicly inspectable strategy-replication database run by an identifiable team)",
        "date": "not stated on retrieved excerpt",
        "tier": 2,
        "tier_justification": "A well-documented, semi-open quant research repository whose whole business is candidly reporting in-sample vs out-of-sample decay across hundreds of published strategies - the Tier 2 'well-documented open-source-adjacent quant repository' category, though it also sells subscriptions (a Tier-3/4-adjacent business model caveat noted explicitly).",
        "hypotheses": [
            {"h": "b", "direction": "partially supports", "note": "reports well-built strategies degrading only 10-20% backtest-to-live, suggesting cost/execution modelling gaps are a real but secondary factor next to genuine signal decay"},
            {"h": "c", "direction": "supports", "note": "Sharpe degradation of 1/3 to 1/2 out-of-sample is presented as NORMAL and EXPECTED for a genuinely real strategy, not a sign the strategy was fake - relevant context for reading notebook 8's carry near-miss"},
            {"h": "d", "direction": "supports", "note": "R^2 < 0.25 for predicting a strategy's OOS performance from its own backtest stats underscores how weakly a backtest alone should be trusted"},
        ],
        "claim": "Published strategies typically retain only ~4/5 of in-sample performance out-of-sample (from the point the paper's own data ends onward), with a normal expected Sharpe degradation of 1/3-1/2, and backtest statistics alone poorly predict (R^2<0.25) live/OOS performance.",
        "stated_costs_turnover_capacity_oos": "Discusses OOS decay broadly; specific per-strategy cost/turnover/capacity figures vary by the underlying database entry, not summarized at this page's level.",
        "red_flags": ["Quantpedia's core business is selling a strategy database/subscription - a Tier 3/4-adjacent commercial model - but this specific meta-analysis content is about honestly reporting decay (the opposite of the marketing failure mode sec 3.1 warns about), which is why it is tiered 2 rather than 3/4."],
        "testable_with_repo_data": {"testable": False, "how": "A meta-analysis of the general decay phenomenon, directly informing how Phase 2 should read this repo's own OOS results, not a specific strategy."},
    },
    {
        "id": "hedge_fund_journal_vrp",
        "url": "https://thehedgefundjournal.com/harvesting-the-s-p500-volatility-risk-premium/",
        "title": "Harvesting the S&P500 Volatility Risk Premium",
        "author_or_institution": "The Hedge Fund Journal (established practitioner trade publication, interviews real fund managers)",
        "date": "not stated on retrieved excerpt",
        "tier": 2,
        "tier_justification": "Trade-press coverage of identifiable, named funds running real disclosed strategies, including their drawdown history - discusses failures/tail risk, not just wins.",
        "hypotheses": [
            {"h": "e", "direction": "partially supports", "note": "the volatility risk premium (selling options) is a real, structurally-motivated (someone pays for insurance) source of return this repo has never touched, but with well-documented catastrophic tail risk (Volmageddon Feb 2018, -90%+ on a related inverse-vol product; COVID March 2020)"},
        ],
        "claim": "Selling volatility (options) captures a persistent risk premium (implied > realized vol on average) but with severe, well-documented left-tail crash risk that has wiped out unhedged implementations multiple times.",
        "stated_costs_turnover_capacity_oos": "Discusses drawdowns explicitly (a red-flag-avoidance signal per sec 3.1: 'refusal to discuss drawdowns' is the red flag, and this source does the opposite); no bps-level cost figure for options execution specifically.",
        "red_flags": [],
        "testable_with_repo_data": {"testable": False, "how": "This repo has no options data of any kind, and options/vol-surface strategies are explicitly listed in sec 1's 'never tested at all' category - an infrastructure gap, not a research gap."},
    },
    # ------------------------------------------------------------------ Tier 3
    {
        "id": "hackernoon_backtest_costs",
        "url": "https://hackernoon.com/your-backtest-left-out-the-two-costs-that-kill-it",
        "title": "Your Backtest Left Out the Two Costs That Kill It",
        "author_or_institution": "HackerNoon contributor (unverified track record)",
        "date": "not stated",
        "tier": 3,
        "tier_justification": "Blog content with no disclosed author track record and no formal cost study behind it - hypothesis-generation only, per sec 3.1.",
        "hypotheses": [
            {"h": "b", "direction": "partially supports", "note": "cites 0.05-0.30% typical major-pair crypto slippage and recommends taker fee + 0.1-0.2% slippage as a conservative haircut - directionally close to, slightly above, this repo's own 4bps+1bp assumption, but unverifiable as a primary figure"},
        ],
        "claim": "Realistic crypto backtests should charge taker fee + 0.1-0.2% slippage per side; retail active traders give up 10-20% of gross returns to fees generally.",
        "stated_costs_turnover_capacity_oos": "States a specific cost recommendation but no methodology, data source, or sample behind the number.",
        "red_flags": ["No stated methodology or data source for the 0.05-0.30% slippage figure - an unsupported number, exactly the Tier 3 pattern."],
        "testable_with_repo_data": {"testable": False, "how": "Directional color only; not a source to size a cost-model change on."},
    },
    {
        "id": "binance_fee_guide_bitdegree",
        "url": "https://www.bitdegree.org/crypto/tutorials/binance-fees",
        "title": "Binance Fees Breakdown: A Detailed Guide for 2026",
        "author_or_institution": "BitDegree (crypto education/content site, not Binance itself)",
        "date": "2026",
        "tier": 3,
        "tier_justification": "A third-party SEO content guide, not Binance's own primary fee-schedule documentation - the specific numbers were not cross-checked against Binance's own docs during this survey, so this is filed as unverified secondary content per sec 3.1's discipline, not laundered up to Tier 2 just because the underlying facts (fee tiers) are plausibly accurate.",
        "hypotheses": [
            {"h": "b", "direction": "partially supports", "note": "reports VIP-tier maker fees falling to 0% and taker fees to ~0.017% at the top futures tier, with a further 10% BNB discount - if directionally accurate, a large real-desk taker could pay well under this repo's assumed 4bps taker fee at high volume, though this repo's own cost assumption already approximates a non-VIP retail/small-institutional taker rate reasonably"},
        ],
        "claim": "Binance USD-M futures VIP fee tiers range from 5bps taker/2bps maker (base) down to 1.7bps taker/0bps maker (VIP 9), before any BNB discount.",
        "stated_costs_turnover_capacity_oos": "States fee numbers but not the 30-day volume required to reach each tier in a way independently verified here, nor slippage/impact at size.",
        "red_flags": ["Third-party content-farm-style guide, not the primary exchange source - the specific bps figures should be verified against Binance's own current fee schedule before being used to justify any cost-model change."],
        "testable_with_repo_data": {"testable": True, "how": "This repo's own crypto cost model (4bps taker + 1bp slippage) could be re-run at a lower, VIP-tier-consistent fee assumption as a sensitivity check - but only after verifying the fee schedule against a primary source, not this one."},
    },
    {
        "id": "geometry_of_alpha_blog",
        "url": "https://waylandz.com/blog/geometry-of-alpha/",
        "title": "The Geometry of Alpha: Why the Quant Moat Is a Factory, Not a Recipe",
        "author_or_institution": "Personal blog (Wayland Zhang), no disclosed track record",
        "date": "not stated",
        "tier": 3,
        "tier_justification": "Personal blog with no disclosed trading track record, costs, or capacity - hypothesis-generation only.",
        "hypotheses": [
            {"h": "a", "direction": "supports", "note": "articulates the 'combine many weak, low-correlation signals' engineering-gap story qualitatively"},
        ],
        "claim": "Sustainable quant edge comes from a production 'factory' of many engineered signals combined, not any single clever recipe.",
        "stated_costs_turnover_capacity_oos": "None stated - no backtest, no cost figures, no track record.",
        "red_flags": ["No costs, no track record, no out-of-sample evidence - a pure narrative/opinion piece."],
        "testable_with_repo_data": {"testable": False, "how": "Not a specific, testable claim."},
    },
    {
        "id": "worldquant_alpha_count_substack",
        "url": "https://youngandcalculated.substack.com/p/how-quant-hedge-funds-actually-build",
        "title": "How Quant Hedge Funds Actually Build and Vet Trading Signals (incl. the 'WorldQuant runs ~4 million alphas' claim)",
        "author_or_institution": "Substack newsletter, author track record not disclosed",
        "date": "not stated",
        "tier": 3,
        "tier_justification": "Unverified specific numeric claim (4 million live alphas) repeated without primary sourcing to WorldQuant itself - the exact pattern sec 3.1 warns against laundering into a firmer tier because the claim is interesting.",
        "hypotheses": [
            {"h": "a", "direction": "supports", "note": "if directionally true, illustrates an extreme version of the signal-combination engineering gap - but the specific number is unverifiable and should not be treated as established fact"},
        ],
        "claim": "WorldQuant is claimed to run approximately 4 million individual trading signals live.",
        "stated_costs_turnover_capacity_oos": "No cost, turnover, or capacity figures; the '4 million' number itself has no stated methodology or primary source.",
        "red_flags": ["A suspiciously large, round-sounding number with no primary source or methodology - flagged explicitly rather than repeated as fact."],
        "testable_with_repo_data": {"testable": False, "how": "Not independently verifiable or testable."},
    },
    {
        "id": "renaissance_medallion_quantifiedstrategies",
        "url": "https://www.quantifiedstrategies.com/jim-simons/",
        "title": "How Jim Simons' Trading Strategies Achieved 66% Annual Returns (Medallion Fund Algorithm)",
        "author_or_institution": "QuantifiedStrategies.com (content site), secondary retelling of widely-reported but never officially disclosed Medallion figures",
        "date": "not stated",
        "tier": 3,
        "tier_justification": "Repeats a widely-circulated but never primary-sourced return figure (Renaissance has never published audited performance) with no cost model, no capacity discussion, and an implied Sharpe far above the sec 3.1 red-flag threshold ('Sharpe > 3 on a liquid instrument at daily frequency') - filed explicitly as the notebook's worked example of a claim that must NOT be treated as evidence, however many blogs repeat it (sec 6's 'repetition is not corroboration' standard).",
        "hypotheses": [
            {"h": "a", "direction": "supports", "note": "even taking the popular account at face value, it describes an intensely engineered, high-frequency, many-signal system - consistent with hypothesis (a) directionally, but not usable as quantitative evidence"},
            {"h": "e", "direction": "supports", "note": "Medallion is explicitly closed to outside capital and has been for decades - the clearest possible illustration of sec 3.1's prior that reliably-profitable-at-scale strategies get walled off, not left available to be tested in a repo like this one"},
        ],
        "claim": "Popular retelling of Medallion Fund's ~66%/yr gross returns via extremely high-frequency, many-signal statistical arbitrage.",
        "stated_costs_turnover_capacity_oos": "No verified cost model, no capacity figure beyond 'closed to outside investors', no primary-source performance disclosure of any kind.",
        "red_flags": [
            "Implausibly high implied Sharpe with no stated cost model - the exact sec 3.1 red flag.",
            "No out-of-sample or independent validation possible; Renaissance has never published audited figures.",
            "Repeated across dozens of near-identical blog posts - repetition, not corroboration.",
        ],
        "testable_with_repo_data": {"testable": False, "how": "Not evidence of anything testable; included specifically as this survey's documented example of a Tier 3 claim correctly NOT elevated despite its fame and appeal."},
    },
    {
        "id": "hedgeco_capacity_squeeze",
        "url": "https://www.hedgeco.net/news/04/2026/quant-funds-face-a-capacity-squeeze-when-too-much-capital-threatens-alpha.html",
        "title": "Quant Funds Face a 'Capacity Squeeze': When Too Much Capital Threatens Alpha",
        "author_or_institution": "HedgeCo (trade-press aggregator)",
        "date": "2026-04",
        "tier": 3,
        "tier_justification": "Trade-press summary/commentary without primary data or named, verifiable sources for its specific claims.",
        "hypotheses": [
            {"h": "e", "direction": "supports", "note": "directional color that crowding/capacity is a live, widely-discussed industry concern, consistent with sec 3.1's 'hard, capacity-limited' prior"},
        ],
        "claim": "Excess capital chasing similar systematic signals compresses returns and raises systemic crowding risk.",
        "stated_costs_turnover_capacity_oos": "General commentary; no specific figures.",
        "red_flags": [],
        "testable_with_repo_data": {"testable": False, "how": "Not a specific strategy."},
    },
    {
        "id": "hummingbot_as_guide",
        "url": "https://hummingbot.org/blog/technical-deep-dive-into-the-avellaneda--stoikov-strategy/",
        "title": "Technical Deep Dive into the Avellaneda & Stoikov Strategy",
        "author_or_institution": "Hummingbot (open-source market-making bot project)",
        "date": "not stated",
        "tier": 2,
        "tier_justification": "An open-source, inspectable implementation (code is public) of a real market-making strategy - meets the Tier 2 'well-documented open-source quant repository where the code is inspectable' bar, distinct from the closed-source crypto MM marketing content also found in this search.",
        "hypotheses": [
            {"h": "e", "direction": "partially supports", "note": "confirms the AS model is implementable in retail-accessible open-source tooling, lowering (but not eliminating) the infrastructure bar for a crypto market-making probe versus the colocated-HFT extreme"},
        ],
        "claim": "The Avellaneda-Stoikov inventory-skew market-making model has a public, inspectable open-source implementation usable on retail crypto exchange APIs.",
        "stated_costs_turnover_capacity_oos": "Discusses maker-fee/rebate economics generally as the mechanism (not a specific backtested P&L, cost, or capacity figure).",
        "red_flags": [],
        "testable_with_repo_data": {"testable": False, "how": "This repo has OHLCV bar data only, no order-book/L2 depth - the strategy's own core risk (inventory vs. the book) cannot be simulated from bars alone, so even with open-source code available this remains untestable with the data this repo actually has."},
    },
    {
        "id": "crypto_microstructure_mdpi_2026",
        "url": "https://www.mdpi.com/2227-7072/14/5/103",
        "title": "Temporal Dynamics of Market Microstructure in Cryptocurrency Perpetual Futures",
        "author_or_institution": "Academic authors, MDPI (International Journal of Financial Studies)",
        "date": "2026",
        "tier": 2,
        "tier_justification": "Peer-reviewed but in an MDPI journal (a publisher whose peer-review rigor and editorial standards are inconsistent across titles and has drawn documented criticism in parts of the academic community) - downgraded from Tier 1 to Tier 2 for that reason rather than accepted at face value because it is nominally 'peer-reviewed'.",
        "hypotheses": [
            {"h": "b", "direction": "neutral", "note": "documents that intraday spreads peak ~2 hours after funding settlement and mid-tier exchanges sometimes lead Binance in price discovery - a real microstructure fact this repo's daily/hourly-bar strategies are too coarse to exploit or be materially hurt by"},
        ],
        "claim": "Cryptocurrency perpetual futures spreads and cross-exchange price leadership show statistically significant intraday patterns tied to funding-settlement timing, across 26 exchanges/812 symbols (Nov 2025-Jan 2026).",
        "stated_costs_turnover_capacity_oos": "Describes spread dynamics but not a full per-trade cost model applicable to this repo's own bar frequencies (1h+).",
        "red_flags": ["MDPI journal-quality caveat noted explicitly rather than assumed away."],
        "testable_with_repo_data": {"testable": False, "how": "Requires sub-hourly, cross-exchange, order-book-adjacent data this repo does not have."},
    },
]


def summarize(sources):
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    hyp_tier12 = {"a": 0, "b": 0, "c": 0, "d": 0, "e": 0}
    hyp_total = {"a": 0, "b": 0, "c": 0, "d": 0, "e": 0}
    red_flag_count = 0
    for s in sources:
        tier_counts[s["tier"]] += 1
        if s["red_flags"]:
            red_flag_count += 1
        for h in s["hypotheses"]:
            hyp_total[h["h"]] += 1
            if s["tier"] in (1, 2):
                hyp_tier12[h["h"]] += 1
    return {
        "n_sources": len(sources),
        "tier_counts": tier_counts,
        "hypothesis_coverage_all_tiers": hyp_total,
        "hypothesis_coverage_tier1_or_2": hyp_tier12,
        "sources_with_red_flags": red_flag_count,
        "gate_S_threshold": {"min_sources": 30, "min_tier12_per_hypothesis": 5},
        "gate_S_fires": (
            len(sources) >= 30
            and all(v >= 5 for v in hyp_tier12.values())
        ),
    }


def main():
    summary = summarize(SOURCES)
    out = {"sources": SOURCES, "summary": summary}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nwritten {OUT_PATH}")


if __name__ == "__main__":
    main()
