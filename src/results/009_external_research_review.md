# 009 — Why Eight Notebooks Found Nothing: A Review of Outside Research

## The question

Eight notebooks — six on crypto, two on commodities — have found no tradeable edge that survives
realistic costs and this repo's own robustness bar. Before generating a ninth null, it is worth
asking whether the search itself is well-aimed.

Five competing explanations, adjudicated against outside peer-reviewed and regulatory literature
rather than against another backtest:

- **(a)** The strategies tested were too naive.
- **(b)** The cost model was too pessimistic.
- **(c)** The statistical bar was too strict.
- **(d)** The markets tested are simply efficient at the horizons this repo can reach.
- **(e)** The programme has been structurally looking in the wrong places.

No new backtest is run here, with one deliberate exception: a single cheap descriptive probe to
check whether one candidate mechanism is even present in this repo's own data before committing a
future notebook to it.

## The headline

**Market efficiency and infrastructure gaps — not an overly strict bar — are the best-supported
explanations, and this survey discriminates rather than hedging.**

Two hypotheses come back **well-supported**: the markets this repo can reach are efficient at the
horizons it tests, and real structural return sources exist but are mostly walled off by
infrastructure this repo doesn't have. One is **contradicted**: the cost model is, if anything,
more likely too generous than too harsh. Two are **partially supported**.

This is neither the comfortable "we've been doing it wrong, here's the fix" story nor the equally
comfortable "everyone else is exaggerating, our nulls are simply right" story. Hypothesis (d)
earned a genuine hearing and came back well-supported rather than dismissed as the boring option.
Hypothesis (e) is equally well-supported — but almost none of its best examples are testable here
without infrastructure this repo has never had. **That is itself the finding, not a consolation
prize.**

One concrete, testable candidate emerged: structural mean-reversion in commodity spreads, which
showed a real, directionally consistent signal in the probe.

## What the four checks required, and how they came out

| Check | Requirement | Result |
|---|---|---|
| **Survey adequacy** | Broad enough evidence to support a diagnosis | **Met** — 36 sources tiered (19 top-tier, 11 second-tier, 6 third-tier); every hypothesis has at least 5 top-two-tier sources bearing on it |
| **Discrimination** | The evidence actually distinguishes between the five explanations | **Met** — two well-supported, one contradicted, on top-tier evidence. The survey did not converge everything to "partially supported" |
| **Our own bar** | A specific verdict on whether this repo's criterion is out of line with practice | **Met** — a sourced recommendation, with consequences for existing results stated as an explicit hypothetical rather than a re-score |
| **Actionable output** | The survey yields concretely testable work | **Met** — 5 fully specified candidates, 3 of which are testable with data already here |

## The five hypotheses in detail

### (a) Our strategies are too naive — *partially supported*

This is a genuine case of good evidence pointing two ways for two different reasons, and it is not
resolved by averaging.

One peer-reviewed study found that volatility-scaling alone nearly doubles equity momentum's
Sharpe, from 0.53 to 0.97 — a large, real, quantified gap between a textbook factor and a properly
implemented one.

But a practitioner study that tested volatility targeting explicitly across equities, bonds,
commodities *and* currencies reports the benefit is concentrated in equity- and credit-like risk
assets and is **negligible specifically for commodities and currencies**. Notebook 008's carry and
momentum tests are entirely commodities.

Applying the headline equity number to this repo's asset classes without that caveat would have
been exactly the kind of interesting-but-uncorroborated claim this survey was built to avoid
elevating.

### (b) Our cost model is too pessimistic — *contradicted*

Two independent peer-reviewed studies find that realistic all-in implementation costs are usually
**larger** than naive bid-ask-spread estimates, not smaller. No top-two-tier source found argues
the opposite direction.

### (c) Our statistical bar is too strict — *partially supported, with a concrete recommendation*

The finance literature's own traditional significance bar is argued by a major peer-reviewed
survey to be too **loose**, not too strict, once corrected for the true scale of factor searching.
No source surveyed argues this repo's deflated-Sharpe threshold specifically is excessive.

On the other hand, a large systematic manager's publicly disclosed long-run Sharpe of about 0.86,
and a well-known practitioner's framing of a 0.5 Sharpe as "already ambitious", both argue that
notebook 008's carry near-miss (0.90–0.95) is not obviously below what a real institutional
strategy looks like on an *absolute* basis.

**The recommendation, in full.** Do not lower the deflated-Sharpe threshold. Instead, introduce a
**second, additional, prospectively applied reporting flag** from notebook 010 onward:
*institutionally fundable absolute performance*, defined as net Sharpe above 0.5 at every tested
origin offset, deflated Sharpe probability above 0.95, and a stated bounded maximum drawdown.

**This flag does not replace the existing tradeable-alpha criterion** (net Sharpe positive at every
offset **and** an excess-return interval over a passive basket that excludes zero **and** a
deflated Sharpe probability above 0.95). A strategy can clear the new flag without clearing the old
one, and must be reported that way: *"fundable-looking on absolute performance, not shown to beat
passive exposure to the same asset class"* — a materially weaker and more honest claim than
"found alpha".

**Consequences for existing results, stated as a labelled hypothetical and never a re-score.** This
flag did not exist when the earlier notebooks ran. Their recorded verdicts under the criterion
pre-declared at the time are unchanged and remain the record.

Had the flag existed: notebook 008's carry (Sharpe 0.90–0.95 at every offset, deflated probability
0.997) would have cleared it. Notebook 008's momentum (best-lookback Sharpe 0.10–0.12, deflated
probability 0.098) would not. Notebook 003's cross-sectional configuration (+0.42 at one offset but
−2.45 at another) would not, since instability across offsets is exactly what the "every offset"
requirement exists to catch.

Nothing about those three published verdicts changes. The flag only changes how a *future* result
that clears the deflation and absolute-Sharpe bars but not the passive-basket comparison can be
honestly and distinctly characterised, rather than forcing a binary call that conflates two
different questions.

### (d) The markets are efficient at the horizons we can reach — *well-supported*

Four independent lines of evidence point the same way: a large study of factor decay after
publication, a peer-reviewed finding of weak-form efficiency in Bitcoin, documented
post-financialisation decay of commodity risk premia, and the documented decay of pairs trading's
original mechanical form.

One top-tier study reaches a more optimistic replication conclusion, and that is reported as a
genuine unresolved disagreement rather than averaged away — but it concerns cross-country equities,
a different universe from this repo's liquid crypto majors and front-month commodity futures.

### (e) We've been looking in the wrong place — *well-supported*

Four independent regulatory sources corroborate a roughly $4 trillion Treasury cash-futures basis
trade. Two peer-reviewed studies support structural mean-reversion surviving costs. Market-making
theory and coverage of the volatility risk premium round out a genuinely broad evidence base.

**But almost none of it is testable here.** The basis trade needs repo financing and leveraged
margin. Market-making needs level-2 order-book data and low-latency execution. The volatility risk
premium needs options data. This repo has never had any of them.

### Reading (c) alongside (d) and (e) is the whole diagnosis

The bar that rejected notebook 008's carry is defensible on its own multiple-testing terms, and the
markets this repo actually reaches are genuinely well-arbitraged. So the eight nulls are **not**
primarily an artefact of an unreasonable internal standard.

At the same time, the *additional* requirement that a strategy beat a passive basket with a
bootstrap interval — as opposed to simply clearing an absolute Sharpe and drawdown bar, the way real
allocators screen — is a genuinely separate and arguably stricter test than the industry's own
practical screen. That is exactly what the recommendation above addresses, without touching the
deflation threshold itself.

---

## How the sources were handled

36 sources, each recorded with its URL, author or institution, date, evidence tier with a stated
justification, which hypotheses it bears on and in which direction, the specific claim, any stated
costs, turnover, capacity or out-of-sample evidence (or an explicit "not stated" — itself the single
most common finding), red flags, and testability against this repo's data.

**No source was promoted to a higher tier because its claim was interesting.** The clearest example
is the famous "66% a year" figure attributed to a well-known quantitative fund, repeated across
dozens of near-identical blog posts. It is filed at the lowest tier used here with three stated red
flags — no cost model, no independently verifiable disclosure, and repetition without corroboration
— rather than laundered upward because it is the most-quoted number in quant finance folklore.

Similarly, an exchange's VIP fee schedule was found only via a third-party content guide rather
than the exchange's own documentation, and is filed at the lowest tier for that reason alone. The
underlying facts are plausibly accurate, but the sourcing doesn't distinguish "plausibly accurate"
from "verified", so it isn't treated as verified.

**Coverage is uneven, and reported as such.** Hypotheses (a) and (b) sit exactly at the five-source
floor; (c) has 7; (d) and (e) have 10 each. The two most heavily covered hypotheses are also the two
with the most decisive verdicts — evidence depth and verdict clarity moved together, which is a
modest sanity check on the survey's construction rather than a coincidence to note and move past.

Every fetched page was treated as untrusted input: nothing fetched was executed and no embedded
instructions were followed. Recorded explicitly even though nothing tripped the check.

---

## The shortlist

Five candidates, each with a mechanism, an evidence tier, its data requirements, an honest
infrastructure assessment, and a pre-registered criterion.

| Candidate | Hypothesis | Evidence | Testable now? |
|---|---|---|---|
| **Structural mean-reversion in commodity spreads** | (e) | Top tier | **Yes** — 30 pre-built spread series already here, never backtested |
| **Volatility-scaled commodity carry and momentum** | (a) | Top tier | **Yes** — reuses notebook 008's panel and cost model; only the sizing rule changes |
| **Blended multi-lookback momentum** | (a) | Second tier | **Yes** — re-aggregates already-computed signals |
| **Crypto perpetual funding cash-and-carry** | (e) | Top tier, by analogy | **Unconfirmed** — needs a spot price series not verified to exist in the cache |
| **Crypto perpetual market-making** | (e) | Top tier | **No** — needs order-book data and low-latency execution this repo has never had |

**The last two are listed rather than omitted, precisely because "this needs infrastructure we
don't have" is itself the finding.**

The cash-and-carry candidate's top-tier evidence — the Treasury basis trade's real, regulator-tracked
scale — supports the *general* mechanism only by analogy to crypto. The crypto-specific "10–30% a
year" figures found during the survey were marketing content from arbitrage-tool vendors and were
explicitly excluded from the evidence used to size this candidate's expected return.

Market-making is untestable here at any bar frequency, since a market maker's core risk — inventory
against the actual limit order book — simply does not exist as a concept in bar data.

**The two volatility-scaling candidates were included despite genuinely mixed priors**, as a
deliberate example of not only shortlisting candidates with one-sided supporting evidence. The
volatility-scaled one in particular carries a specific sourced reason to expect a *smaller* effect
than its headline equity number. It stays on the list because it is untested here and cheap to
test, not because the prior is strong.

---

## One cheap empirical probe

Spread mean-reversion is the only candidate meeting all three criteria: top-tier evidence, testable
with data already here, and cheap.

**This is explicitly not a backtest.** No cost model, no position sizing, no Sharpe ratio, no
verdict. It asks only whether the mechanism the literature describes is even present, directionally,
in this repo's own spread data.

Six pre-built spread series, with roll-window-flagged rows dropped first. Each gets an
autoregressive mean-reversion regression on differences (giving a coefficient, a t-statistic, and an
implied half-life) and a rank correlation between a rolling 60-day z-score and the 5-day-forward
change in the spread.

| Spread | Coefficient | t-stat | Half-life (days) | Mean-reverting? | 5-day correlation | p |
|---|---|---|---|---|---|---|
| 3-2-1 crack | −0.0089 | −2.84 | 77 | yes | −0.081 | 6.2e−05 |
| Gold–silver | −0.0092 | −3.90 | 75 | yes | −0.008 | 0.631 |
| Brent–WTI | −0.0111 | −3.75 | 62 | yes | −0.127 | 3.8e−10 |
| Corn–wheat | −0.0151 | −5.25 | 46 | yes | −0.112 | 1.3e−11 |
| Platinum–palladium | −0.0013 | −1.60 | 552 | no | −0.025 | 0.164 |
| Soybean crush | −0.0081 | −3.46 | 85 | yes | −0.061 | 4.0e−4 |

**Five of six spreads show significant mean reversion, and four of six show a significant negative
correlation** — directionally exactly what the pairs-trading literature predicts.

This is un-forced rather than shaped to justify the shortlist choice: gold–silver and
platinum–palladium both come back weak or non-significant on the correlation test despite
mean-reverting on the regression test. That disagreement between the two checks is reported, not
smoothed over.

Half-lives of 46–85 days for the significant spreads imply genuinely low turnover if traded — a real
potential cost advantage over daily-rebalanced factor strategies. But that says nothing about net
Sharpe or capacity, which this probe deliberately does not attempt to estimate.

**This is a first look showing the mechanism exists in this repo's own data, not a finding that a
strategy exists.** A properly pre-registered test — with a cost model, roll-window exclusion applied
to the backtest itself rather than just the descriptive screen, a bootstrap interval, and deflation
over the true count of spreads and parameter configurations tried — belongs in the next notebook.

---

## Near-misses worth recording

No computational bugs, since this notebook's work was survey and synthesis rather than modelling.
Two discipline-relevant near-misses are worth recording in the same spirit:

1. An early draft of the shortlist would have cited an exchange's advertised fee tiers via a
   third-party content guide as if it were a primary source. Caught before being used to justify any
   cost-model change, and filed at the lowest tier with the sourcing gap stated explicitly.
2. The cash-and-carry candidate's initial framing risked implicitly treating low-tier crypto
   arbitrage marketing claims as if the Treasury basis trade's regulatory evidence validated them
   directly. Corrected to state the analogy explicitly and exclude the crypto-specific figures from
   the evidence used to size it.

---

## Bottom line

All four pre-declared checks are met. The diagnosis does not deliver a single tidy explanation — it
delivers a well-evidenced ranking:

- Markets are genuinely efficient at the specific instruments and horizons this repo can reach.
- Real structural sources of return exist elsewhere but are mostly walled off by missing
  infrastructure, with one clear exception.
- The cost model is not the problem, and the realistic-cost literature argues for caution in the
  opposite direction.
- The statistical bar is largely defensible, but a second, distinctly labelled absolute-performance
  flag is a specific and honestly scoped improvement to how future results get characterised.
- Genuine implementation gaps exist in the literature, but the best-evidenced fix for this repo's
  asset classes is smaller than the headline equity numbers suggest.

One concrete, already-available, un-backtested idea — structural spread mean-reversion — shows a real
signal in the probe and is ready to become a properly pre-registered test.

## What to test next

- **Spread mean-reversion** — a full backtest with a cost model, roll-window exclusion, bootstrap
  interval and deflation. The primary candidate.
- **Volatility-scaled carry and blended momentum** — secondary, cheaper re-implementations of
  notebook 008's existing signals.
- **Crypto funding cash-and-carry** — needs a data-availability check first: does spot price data
  exist in the cache?
- **The new absolute-performance reporting flag** should be applied to every test from the next
  notebook onward, reported alongside rather than instead of the existing criterion.

*Notebook: `src/research/009_external_research_review.ipynb`. New terminology — structural
cash-and-carry arbitrage, market making and inventory risk, the replication crisis in factor
investing — is defined in `docs/`. The holdout period was never touched; a survey has no legitimate
reason to go near it.*
