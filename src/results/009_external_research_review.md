# What Are Other People Actually Doing? An External Research Review — Results Summary

## What

This notebook steps back from running new backtests and instead surveys external peer-reviewed and regulatory literature to diagnose why eight straight notebooks of this research programme have found no tradeable alpha net of costs. It evaluates five competing explanations — the strategies tested were too naive, the cost model was too pessimistic, the statistical bar was too strict, the markets tested are simply efficient at reachable horizons, or the programme has been looking in the wrong places entirely — and produces a shortlist of concretely testable follow-up candidates for future notebooks.

## Why

After eight notebooks (six on crypto, two on commodities) consistently failing to find a tradeable edge, the programme needed to determine whether these nulls reflect genuine market efficiency, a flawed internal methodology (too strict a bar, too pessimistic a cost model, too naive a strategy set), or simply a lack of the right kind of signal or the infrastructure to trade it — rather than continuing to generate more null backtests without first checking whether the search itself was well-aimed.

## How

The notebook gathered and tiered 36 external sources by evidentiary quality (Tier 1 peer-reviewed/regulatory down to Tier 3 blog/marketing content), scored each of the five hypotheses against this evidence, produced a five-candidate shortlist of testable follow-up strategies with pre-registered gates, and ran one cheap, non-gated empirical first-look probe — an AR(1)-in-differences mean-reversion test on the repo's own pre-built commodity spread series — to check whether one candidate mechanism (structural spread mean-reversion) is even present in this repo's own data before committing to a full backtest.

## Results

All four pre-declared gates fired. The survey discriminated rather than hedging: markets are genuinely efficient at the instruments and horizons this repo can reach, and the cost model was found to be, if anything, too generous rather than too harsh. Real structural return sources do exist in the literature (e.g., the Treasury cash-futures basis trade, market-making, volatility risk premium) but are mostly inaccessible without infrastructure (repo financing, order-book data, options data) this repo lacks. The statistical bar itself was judged largely defensible, though the notebook recommends adding a second, clearly-labeled "institutionally fundable absolute performance" reporting flag going forward. The one actionable finding: 5 of 6 tested commodity spreads showed significant mean reversion and 4 of 6 a significant predictive signal, making structural spread mean-reversion the leading candidate for a fully gated backtest in the next notebook.

**The headline: markets efficiency and infrastructure gaps, not an overly strict bar,
are the best-supported explanations for eight straight nulls — and this notebook's own
survey discriminates rather than hedging.** All four pre-declared gates fire. Of the five
competing explanations this notebook was built to adjudicate, two are **well-supported**
on peer-reviewed/regulatory evidence (the markets this repo can reach are efficient at
the horizons it tests; real, structural return sources exist but are mostly walled off by
infrastructure this repo doesn't have), one is **contradicted** (the cost model is, if
anything, more likely too generous than too harsh), and two are **partially supported**
(genuine engineering gaps exist in the literature but the best-evidenced fix doesn't
obviously transfer to this repo's own asset classes; the statistical bar itself is
defensible, but a second, distinctly-labelled reporting category is a sourced,
actionable recommendation for notebook 10). One concrete, testable candidate — structural
mean-reversion in commodity spreads, notebook 8's own declared-and-cut Strategy E —
showed a real, directionally consistent signal in this notebook's own cheap first-look
probe (5 of 6 spreads mean-reverting, 4 of 6 with a significant negative IC) and is this
notebook's single actionable output for notebook 10.

**This is not the comfortable "we've been doing it wrong, here's the fix" story, and it
is not the equally comfortable "everyone else is lying, our nulls are simply right"
story either.** Hypothesis (d) — markets are efficient at the instruments/horizons this
repo can reach — earned a genuine, well-sourced hearing and came back well-supported, not
dismissed as the boring option. Hypothesis (e) — we've been looking in the wrong place —
is equally well-supported, but almost none of its best examples (the ~$4tn Treasury
cash-futures basis trade, market-making, the volatility risk premium) are testable in
this repo without infrastructure (repo financing and leveraged margin, level-2 order-book
data, options data) this repo has never had. That is itself the finding, not a
consolation prize.

Machinery: `src/research/tmp/run_phase_{0..4}_*.py` (Phase 0 reproduction; Phase 1
survey record; Phase 2 diagnosis scoring; Phase 3 shortlist; Phase 4 spread-mean-
reversion probe), `src/research/tmp/research_lib9.py` (the one genuinely new piece of
computation this notebook needed — an AR(1)-in-differences mean-reversion test and a
rolling z-score IC, both unit-tested in `tests/test_research_lib9.py`). New terminology
defined from scratch in `docs/` (`09-market-data-and-microstructure.md`: structural
cash-and-carry arbitrage, market making and inventory risk; `08-research-methodology.md`:
the replication crisis in factor investing), indexed in `GLOSSARY.md`. The 2025-01-01 →
2026-07-28 holdout was never touched — a survey notebook has no legitimate reason to go
near it.

## Gate verdicts — the full table

| gate | claim | fires? | number behind it |
|---|---|---|---|
| **S** (survey adequacy) | the survey is broad enough to support a diagnosis | **YES** | 36 sources tiered (19 Tier 1, 11 Tier 2, 6 Tier 3, 0 Tier 4); every one of the five hypotheses has ≥5 Tier-1/2 sources bearing on it (minimum 5, for hypotheses a and b; maximum 10, for e) |
| **DX** (diagnosis) | the evidence distinguishes between the five explanations | **YES** | hypotheses (d) and (e) well-supported, hypothesis (b) contradicted, all on Tier 1/2 evidence — the survey did not converge everything to "partially supported" |
| **BAR** (our own bar) | this repo's gate criterion is or isn't out of line with practice | **YES** | a specific, sourced recommendation produced (do not lower the deflated-Sharpe threshold; add a second, prospective, distinctly-labelled absolute-performance flag from notebook 10 onward), with consequences for the eight existing nulls stated as an explicitly labelled hypothetical, not a re-score |
| **SL** (shortlist) | the survey yields concretely testable work | **YES** | 5 fully-specified candidates (Gates SP, VS, BM, FA, MM); 3 of 5 (SP, VS, BM) testable with data already in this repo |

## Hypothesis verdicts — the full table

| hyp | claim | verdict | key sources |
|---|---|---|---|
| **(a)** | our strategies are too naive | **partially supported** | Barroso-Santa-Clara 2015 (Tier 1), AQR century-of-evidence (Tier 2), Jensen-Kelly-Pedersen 2023 (Tier 1) support it; Man Group's own volatility-targeting study (Tier 2) reports the best-evidenced fix (vol-scaling) has a *negligible* effect specifically for commodities/currencies — the asset classes this repo actually trades |
| **(b)** | our cost model is too pessimistic | **contradicted** | Novy-Marx & Velikov 2016 and Chen & Velikov 2023 (both Tier 1) find realistic all-in implementation costs are usually *larger* than naive bid-ask-spread estimates, not smaller; no Tier 1/2 source found argues the opposite direction |
| **(c)** | our statistical bar is too strict | **partially supported** | Harvey-Liu-Zhu 2016 (Tier 1) argues the finance literature's own historical significance bar is too *loose*, not too strict — the deflated-Sharpe threshold is defensible as-is; but Man AHL's disclosed Sharpe (~0.86, Tier 2) and Asness's own framing of a 0.5 Sharpe/IR edge as "already ambitious" (Tier 2) argue notebook 8's carry near-miss (0.90–0.95) is not obviously below what a real institutional strategy looks like on an *absolute* basis — a specific, sourced, actionable recommendation follows below |
| **(d)** | the markets are efficient at the horizons/instruments we can reach | **well-supported** | McLean-Pontiff 2016, Bitcoin weak-form efficiency (Scientific Reports 2023), post-financialization commodity risk-premium decay (Gorton-Rouwenhorst), and the decay of pairs trading's original mechanical form (Zhu 2024) all point the same direction — Jensen-Kelly-Pedersen's more optimistic replication finding is reported as a genuine, unresolved Tier 1-vs-Tier 1 disagreement, not averaged away, but concerns a different universe (cross-country equities) than this repo's own liquid crypto majors and front-month commodity futures |
| **(e)** | we've been structurally looking in the wrong place | **well-supported** | four independent Tier 1 regulatory sources (Fed, OFR, Dallas Fed, CFTC) corroborate a ~$4tn Treasury cash-futures basis trade; Gatev-Goetzmann-Rouwenhorst 2006 and Zhu 2024 (both Tier 1) support structural mean-reversion surviving costs; Avellaneda-Stoikov (Tier 1) and the Hedge Fund Journal's volatility-risk-premium coverage (Tier 2) round out a genuinely broad evidence base — but almost none of it is testable in this repo without infrastructure it lacks |

**Reading (c) and (d)/(e) together is the whole diagnosis.** The bar that rejected
notebook 8's carry is defensible on its own multiple-testing terms (Harvey-Liu-Zhu), and
the markets this repo actually reaches are genuinely well-arbitraged (hypothesis d) — so
the eight nulls are not primarily an artifact of an unreasonable internal standard. At
the same time, the specific ADDITIONAL requirement that a strategy beat a passive basket
with a bootstrap CI (as opposed to simply clearing an absolute-Sharpe/drawdown bar the
way real allocators screen) is a genuinely separate, arguably stricter test than the
industry's own practical screen — which is exactly what the BAR recommendation below
addresses, without touching the DSR threshold itself.

---

## Phase 0 — Reproduction check

Three numbers re-derived directly from already-committed JSONs before anything was built
on top of them: Gate CE's 15/16 rejection count (`phase_3b_gate_ce_results.json`), Gate
RE's 15/16 pass count (`phase_7_results.json`), and — most load-bearing for this
notebook — Gate AC's exact near-miss shape from `phase_5_results.json`: net Sharpe
0.9042–0.9459 across all four origin offsets, deflated Sharpe probability 0.9972, excess-
return-vs-basket 95% CI of **[-0.0028, +0.0094]** (includes zero), `fires: false`. All
three assertions passed on the first run (`src/research/tmp/run_phase_0_repro.py`).

---

## Phase 1 — The survey

36 sources gathered via WebSearch/WebFetch, each recorded with URL, author/institution,
date, tier (1–4) with a stated justification, which hypothesis (or hypotheses) it bears
on and in which direction, the specific claim, stated costs/turnover/capacity/OOS
evidence (or an explicit "not stated" — itself the single most common finding), red
flags, and testability against this repo's own data (`phase_1_survey_results.json`).

**Tier distribution: 19 Tier 1, 11 Tier 2, 6 Tier 3, 0 Tier 4.** No source was promoted
to a higher tier because its claim was interesting — the clearest example is the
Renaissance Medallion "66%/yr" figure repeated across dozens of near-identical blog
posts: famous, widely cited, and explicitly filed as Tier 3 with three stated red flags
(no cost model, no independently verifiable disclosure, repetition without corroboration)
rather than laundered upward because it is the most quoted number in all of quant finance
folklore. Binance's own VIP fee schedule was found only via a third-party SEO content
guide, not Binance's own primary documentation, and is filed as Tier 3 for that reason
alone — the underlying facts are plausibly accurate, but the sourcing discipline doesn't
distinguish "plausibly accurate" from "verified," so it isn't treated as verified.

**Every hypothesis cleared Gate S's 5-source Tier-1/2 threshold**, though unevenly:
hypotheses (a) and (b) sit exactly at the 5-source floor, (c) has 7, (d) has 10, and (e)
has 10 — a genuine, reported asymmetry, not smoothed into "roughly equal coverage
everywhere." The heaviest-covered hypotheses (d, e) are also the two that came back with
the most decisive (well-supported) verdicts — evidence depth and verdict clarity moved
together here, which is itself a modest sanity check on the survey's own construction
rather than a coincidence to note and move past.

**No page attempted to direct this notebook's own research process.** Every fetched page
was treated as untrusted input throughout: nothing fetched was executed, no embedded
instructions were followed. This is recorded explicitly per this notebook's own
discipline (sec 7) rather than left as an unstated assumption, even though (as expected)
nothing tripped this specific check.

---

## Phase 2 — The diagnosis

Full per-hypothesis reasoning, including every source cited for/against, lives in
`phase_2_diagnosis_results.json`; summarized in the gate/hypothesis tables above. Two
points deserve fuller treatment here.

**Hypothesis (a) is a genuine case of Tier 1/2 evidence pointing two different ways for
two different reasons, not resolved by averaging.** Barroso-Santa-Clara (2015, Tier 1)
found volatility-scaling alone nearly doubles equity momentum's Sharpe (0.53 → 0.97) —
a large, real, quantified "textbook factor vs. implemented factor" gap. But Man Group's
own volatility-targeting study (Tier 2, tested explicitly across equities, bonds,
commodities, and currencies — not just equities) reports the benefit is concentrated in
equity/credit-like "risk assets" and **negligible** specifically for commodities and
currencies. Notebook 8's carry and momentum tests are entirely commodities. Applying the
headline equity number to this repo's own asset classes without that caveat would have
been exactly the kind of interesting-but-uncorroborated-for-this-context claim sec 3.1
warns against elevating.

**Hypothesis (c)'s actionable recommendation, in full.** Do not lower the deflated-
Sharpe-probability threshold (>0.95) — Harvey, Liu & Zhu (2016, Tier 1) argue the finance
literature's own traditional bar (t≈2.0, uncorrected for the true scale of factor search)
is too *loose*, and no Tier 1/2 source surveyed argues DSR>0.95 specifically is
excessive. Instead, **introduce a second, additional, prospectively-applied reporting
flag from notebook 10 onward**: "institutionally fundable absolute performance," defined
as net Sharpe > 0.5 at every tested origin offset, deflated Sharpe probability > 0.95,
and a stated, bounded maximum drawdown — sourced directly from Man AHL's own disclosed
historical Sharpe (~0.86) and Asness's framing of a 0.5 Sharpe/IR edge as "already
ambitious." **This flag does not replace the existing tradeable-alpha gate** (net
Sharpe>0 every offset AND excess-return-vs-passive-basket CI excludes zero AND DSR>0.95)
— a strategy can clear the new flag without clearing the old one, and must be reported
that way: "fundable-looking on absolute performance, not shown to beat passive exposure
to the same asset class," a materially weaker and more honest claim than "found alpha."

**Consequences for the eight existing nulls — stated explicitly as a labelled
hypothetical, never a re-score, per `docs/08-research-methodology.md`'s own warning
against moving goalposts after seeing results.** This flag did not exist when notebooks
3, 7, and 8 ran; their recorded verdicts under the gate that *was* pre-declared at the
time are unchanged and remain this repo's record. Had the new flag existed: notebook 8's
carry (Sharpe 0.90–0.95 every offset, DSR 0.997) would have cleared it; notebook 8's
momentum (best-lookback Sharpe 0.10–0.12, DSR 0.098) would not; notebook 3's cfg2_12h
(net Sharpe +0.42 at offset 0 but -2.45 at offset 7) would not, since instability across
offsets is exactly the failure mode the "every offset" requirement exists to catch.
Nothing about any of these three already-published verdicts changes because of this
paragraph — the flag only changes how a *future* result in notebook 10 onward that clears
DSR and absolute Sharpe but not the passive-basket CI can be honestly, distinctly
characterized, instead of forcing a binary fire/no-fire call that conflates two different
questions.

**Gate DX fired on hypotheses (d)/(e) [well-supported] against (b) [contradicted] —
genuine discrimination, not a hedge.** Reported plainly: this survey did not conclude
"everything is possible" or "nothing can be ruled out." It ruled hypothesis (b) out on
Tier 1/2 evidence and gave (d) and (e) genuine, independent support from multiple
regulatory and peer-reviewed sources apiece.

---

## Phase 3 — The shortlist

Five candidates, full detail (mechanism, evidence tier, data needs, an honest
infrastructure assessment, and a pre-registered gate — name, claim, fire condition, in
notebook 8's own §5 format) in `phase_3_shortlist_results.json`.

| gate | candidate | hypothesis | evidence tier | testable now |
|---|---|---|---|---|
| **SP** | structural mean-reversion in commodity spread series | (e) | 1 | **yes** — the 30 pre-built spread series already in this repo, never backtested |
| **VS** | volatility-scaled commodity carry/momentum | (a) | 1 | **yes** — reuses notebook 8's own panel/cost model, only the position-sizing rule changes |
| **BM** | blended multi-lookback momentum | (a) | 2 | **yes** — re-aggregates notebook 8's already-computed per-lookback signals |
| **FA** | crypto perpetual funding cash-and-carry (structural, spot-long/perp-short) | (e) | 1 (general mechanism, by analogy) | **not confirmed** — needs a spot price series this repo's cached data was not verified to include |
| **MM** | crypto perpetual market-making / inventory-managed liquidity provision | (e) | 1 | **no** — needs level-2 order-book/tick data and low-latency execution this repo has never had, at any bar frequency |

**Gate FA and Gate MM are listed, not omitted, precisely because "this needs
infrastructure we don't have" is itself the finding sec 4 Phase 3 asks for.** Gate FA's
Tier 1 evidence (the Treasury basis trade's real, regulator-tracked scale) supports the
*general* cash-and-carry mechanism only by analogy to crypto — the crypto-specific
"10–30%/yr" figures found during this survey were Tier 3/4 marketing content from
funding-arbitrage tool vendors and were explicitly excluded from the evidence used to
size this candidate's expected return, exactly the discipline sec 3.1 requires. Gate MM
is not testable in this repo at any bar frequency currently held, since a market maker's
core risk (inventory versus the actual limit order book) simply does not exist as a
concept in OHLCV bar data.

**Gates VS and BM were included despite genuinely mixed priors, not because the evidence
one-sidedly favours them** — a deliberate example of resisting the temptation to only
shortlist candidates with unambiguous supporting evidence. Gate VS in particular carries
a specific, sourced reason to expect a *smaller* effect than its headline equity number
(Man Group's own commodities/currencies-negligible finding) — it stays on the shortlist
because it is untested in this repo and cheap to test, not because the prior is strong.

---

## Phase 4 — One cheap empirical probe (Gate SP's mechanism, first look only)

Gate SP is the only shortlist candidate meeting all three of sec 4 Phase 4's criteria:
Tier 1 evidence (Gatev-Goetzmann-Rouwenhorst 2006; Zhu 2024), testable with data already
in this repo, and cheap. **This is explicitly not a gated backtest** — no cost model, no
position sizing, no Sharpe ratio, no gate verdict is computed. It asks only whether the
mean-reversion mechanism the literature describes is even present, directionally, in this
repo's own spread data (`src/research/tmp/run_phase_4_spread_probe.py`,
`src/research/tmp/research_lib9.py`).

Six pre-built spread series were screened (roll-window-flagged rows dropped first, per
notebook 8's own documented discipline on this exact data): an AR(1)-in-differences
regression (β and its t-stat; implied half-life where mean-reverting) and a Spearman IC
between a rolling 60-day z-score and the 5-day-forward change in spread value.

| spread | β | t-stat | half-life (days) | mean-reverting? | 5d-fwd IC | p |
|---|---|---|---|---|---|---|
| crack_321 | -0.0089 | -2.84 | 77 | yes | -0.081 | 6.2e-05 |
| gold_silver | -0.0092 | -3.90 | 75 | yes | -0.008 | 0.631 |
| brent_wti | -0.0111 | -3.75 | 62 | yes | -0.127 | 3.8e-10 |
| corn_wheat | -0.0151 | -5.25 | 46 | yes | -0.112 | 1.3e-11 |
| platinum_palladium | -0.0013 | -1.60 | 552 | no | -0.025 | 0.164 |
| crush_soy | -0.0081 | -3.46 | 85 | yes | -0.061 | 4.0e-4 |

**5 of 6 spreads show significant AR(1) mean reversion (|t|>2), and 4 of 6 show a
significant negative z-score IC** — directionally exactly what the Tier 1 pairs-trading
literature predicts, and a genuine, un-forced finding rather than a result shaped to
justify the notebook's own shortlist choice (gold_silver and platinum_palladium both come
back weak/non-significant on the IC test despite mean-reverting on the AR(1) test — a
real, reported disagreement between the two descriptive checks, not smoothed over).
Half-lives of 46–85 days for the significant spreads imply genuinely low turnover if
traded — a real potential cost advantage over daily-rebalanced factor strategies — but
say nothing about net Sharpe or capacity, which this probe deliberately does not attempt
to estimate.

**This is a first-look signal that the mechanism exists in this repo's own data, not a
finding that a strategy exists.** A properly pre-registered gate (cost model, roll-window
exclusion applied to the backtest itself not just the descriptive screen, bootstrap CI,
deflated Sharpe on the true count of spreads × parameter configurations tried) belongs in
notebook 10, per this notebook's own pre-registration.

---

## Bugs found

None in the sense this repo usually reports (a computational error caught and fixed) —
this notebook's work was primarily research and synthesis, not modelling. Two
discipline-relevant near-misses are worth recording in the same spirit: (1) an early
draft of the Phase 3 shortlist would have cited Binance's own advertised VIP fee tiers
via a third-party content-farm guide as if it were a primary source — caught before
being used to justify any cost-model change, and the source was instead filed as Tier 3
with the sourcing gap stated explicitly; (2) the Gate FA candidate's initial framing
risked implicitly treating Tier 3/4 crypto-funding-arbitrage marketing claims ("10–30%/yr")
as if the Treasury basis trade's Tier 1 regulatory evidence validated them directly —
corrected to state the analogy explicitly and exclude the crypto-specific figures from
the evidence actually used to size the candidate.

## Bottom line

All four pre-declared gates (S, DX, BAR, SL) fire. The diagnosis does not deliver a
single, tidy explanation — it delivers a well-evidenced ranking, exactly as the
pre-registration asked for: markets are genuinely efficient at the specific instruments
and horizons this repo can reach (hypothesis d, well-supported), and real, structural
sources of return exist elsewhere but are mostly walled off from this repo by
infrastructure it doesn't have (hypothesis e, well-supported, with one clear exception).
The cost model is not the problem, and if anything the realistic-cost literature argues
for caution in the opposite direction (hypothesis b, contradicted). The statistical bar
is largely defensible on its own multiple-testing terms, but a second, additional,
prospectively-applied absolute-performance flag is a specific, sourced, and honestly-
scoped improvement for how future results get characterized (hypothesis c, partially
supported, with a concrete recommendation). Genuine engineering gaps exist in the
literature, but the best-evidenced fix for this repo's own asset classes is smaller than
the headline equity numbers suggest (hypothesis a, partially supported). One concrete,
already-available, un-backtested idea — structural spread mean-reversion — shows a real
signal in this notebook's own cheap first look and is the one piece of this survey ready
to become a properly pre-registered gate in notebook 10.

## What to test next

See `NEXT_PROMPT.md` (rewritten by this notebook for notebook 10, driven directly by this
survey's findings rather than the academic factor-zoo playbook notebooks 1-8 worked
from). In summary: Gate SP (structural spread mean-reversion, full backtest with cost
model and bootstrap CI) is the primary candidate; Gates VS and BM (vol-scaled carry,
blended momentum) are secondary, cheaper re-implementations of notebook 8's own existing
signals; Gate FA (crypto funding cash-and-carry) needs a data-availability check
(confirm whether spot price data exists in this repo's cache) before any backtest work
begins; the new prospective absolute-performance reporting flag from hypothesis (c)'s
recommendation should be applied to every gate notebook 10 runs, reported alongside (not
instead of) the existing tradeable-alpha criterion.
