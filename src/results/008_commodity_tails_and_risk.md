# 008 — Do the Crypto Findings Hold in Commodities?

## The question

Every finding in this programme so far came from crypto — a market with unusual structure: 24/7
trading, funding-rate mechanics, no physical delivery.

Commodity futures are a genuinely different test. Real storage costs, physical delivery, producer
hedging demand, and — unlike crypto — a research literature that predicts carry and momentum
*should* work here.

Sixteen commodity futures plus an equity-index control, tested on:

**The risk findings, which prior notebooks certified in crypto**

- Are commodities fat-tailed?
- Do thin-tailed models understate their own expected shortfall?
- Can a well-calibrated conditional risk engine be built for them?

**The alpha findings, where the literature disagrees with this programme**

- Does commodity carry survive realistic costs?
- Does time-series momentum?

**Two genuinely uncertain questions**

- Does commodity tail asymmetry flip sign relative to equities, as inventory theory predicts?
- Does term-structure state (backwardation versus contango) carry tail-risk information beyond an
  unconditional model?

## The headline

**The crypto risk findings are not crypto findings. They are facts about financial returns.**

Fat tails, thin-tailed models understating expected shortfall, and a well-calibrated conditional
risk engine all replicate cleanly across three different asset classes with none of crypto's
structure. The alpha side replicates too, in the opposite sense: **commodity carry and time-series
momentum, the two most-cited edges in the futures literature, both come back null** — extending
this programme's run of honest nulls into a market where a real edge was the literature's prior,
not the surprise.

Both genuinely uncertain questions came back negative, reported at the same rigour as everything
that fired.

## Results at a glance

| Claim | Verdict | The number behind it |
|---|---|---|
| **Commodities are fat-tailed** | **Yes** | Tail index below 5 in 16/16 products; the equity control's range excluded from the product's own in 13/16; normality rejected at p ≈ 0 in all 16 |
| **Tail asymmetry flips sign vs. equities** | **No** | Only 5 of 14 measurable products show the predicted right-skew sign — no better than chance |
| **No single density family wins everywhere** | **Partial** | Five distinct families win somewhere in the raw ranking, supporting the story qualitatively — but only 2 of 16 products clear significance, and both are won by the *same* family, so the strict criterion isn't met |
| **Thin-tailed models understate 1% expected shortfall** | **Yes** | 15/16 products reject at 5% (corrected), both tails; holdout: 11/16, same direction |
| **Term-structure state adds tail information** | **No** | Only 4 of 16 products show a state where coverage fails while the pooled test passes — well short of the 10-product bar |
| **Carry survives cost** | **No** | Net Sharpe 0.90–0.95 at every offset and a deflated Sharpe probability of 0.997 — but the interval on excess return over a passive basket includes zero |
| **Momentum survives cost** | **No** | Best lookback net Sharpe 0.10–0.12, sign-inconsistent across lookbacks, deflated Sharpe probability 0.098 |
| **The risk engine is correctly calibrated out of sample** | **Yes** | 15/16 products pass walk-forward coverage; holdout: 14/16 |

Three of the four claims predicted in advance as likely fired outright. The fourth fired in spirit
but not by the letter of its own significance bar, and is reported as partial rather than rounded
up. Both genuinely uncertain questions came back negative. Every alpha attempt came back null,
exactly as predicted.

**The holdout (2025-01-01 to 2026-07-28) was spent once**, since the three risk claims all fired.
Every finding replicates directionally on data no fitting procedure ever saw.

---

## Data hygiene: four bugs that would have invalidated everything

Building a reliable continuous futures price series turned out to be most of the work. All four of
these were caught before a single tail statistic was trusted.

### 1. Separating real events from junk needs persistence, not volume

The instinct is "low volume plus a big price deviation means bad data". It fails here.

Crude oil's genuine negative settlement on 2020-04-20 (−$2.67 close) traded on **8.4%** of that
day's total crude volume. Natural gas's mislabelled contract — tagged as an outright but actually
some kind of calendar-spread quote — traded on a comparable **9.9%** of that day's gas volume. Any
volume cutoff, absolute or relative, flags both or neither.

What actually separates them is how long the anomaly persists across each contract's *life*. The
mislabelled gas contract prints a near-zero or negative close on **97%** of the ~575 days it
appears. The crude contract deviates on **0.6%** of its ~343 days.

The production rule is therefore two-tier. A contract with at least 10 days of history whose price
deviates more than 30% from that date's highest-volume anchor contract on more than half its days
is flagged **in its entirety** — that's a mislabelled series, not an outright having a bad day. A
contract that passes is still flagged row by row if a single day deviates more than 30% *and*
trades under 50,000 contracts — a genuine one-off glitch.

Verified directly against the raw exchange statistics: one confirmed junk gold contract shows a
settlement price of exactly 0.0 while its own session high, low and best bid/offer print around
$1,127–1,128, in line with real gold that week. The settlement feed is broken for that contract,
not gold itself.

Contamination rates after the fix: 0.0–3.98% for 15 of 16 products, and 35.09% for natural gas —
the single worst-affected product, matching the scale of its mislabelling problem.

### 2. A liquidity screen applied too early deletes data rather than cleaning it

The roll schedule here is calendar-driven, not volume-driven. So a quiet day on the contract the
calendar already designates as front month is a real trading day, not a contract needing
replacement. Screening it out *before* building the continuous curve carved holes in the series —
**38–69% of rows null** for the thinnest products (palladium, platinum) before this was caught.

Fixed by moving the screen downstream, where it's reported as a diagnostic and never applied before
curve construction.

### 3. The roll calendar lists contract months that don't really trade

Platinum and palladium list a ticker for every calendar month but only trade real size in January,
April, July and October. The in-between months print a handful of trades over a handful of days —
total lifetime volume in the tens to low hundreds, against hundreds of thousands in the real
months.

Rolling into a listed-but-dead month left platinum's front-month series **null on 57% of days**.
Fixed by requiring a minimum lifetime volume of 5,000 contracts on top of — not instead of — the
calendar-membership check, which dropped both products' null rates below 10%.

### 4. Back-adjusted prices go negative over a long history

Additive back-adjustment accumulates an offset at every roll. Over 16 years and roughly 200 rolls,
several products' early back-adjusted prices crossed **zero**, producing single-day "returns" over
100% that were pure splice artefact.

Ratio adjustment never crosses zero as long as the raw price doesn't, and is the series used for
every tail, autocorrelation and density statistic here once the bug was caught. Crude's post-fix
excess kurtosis (36.09) matches its independently computed figure closely, confirming internal
consistency.

### The roll rule, and how sensitive results are to it

Production rule: roll five calendar days before first notice (or last trade date where notice isn't
populated), snapped backward off a weekend.

Sensitivity on crude at 3, 5 and 10 days: excess kurtosis 35.4 / 36.1 / 36.1, annualised volatility
37.8% / 37.6% / 37.6%. Stable across the range.

Crude's 2020-04-20 print is instructive. Under the production rule the front-month series had
already rolled into the June contract three trading days earlier, so a real book following this rule
never touches the negative print at all. It survives only in the raw per-contract data, correctly
preserved by the hygiene filter — which is the economically right outcome for a book that respects
delivery risk.

### External validation: attempted, and it does not clear its own bar

| Check | Pass threshold | Result |
|---|---|---|
| Against an independent curve series (4 products) | ≥99% of dates within 1 tick | **64–84%** |
| Against an independent realised-volatility series (12 products) | ≥90% within 25% relative tolerance | **7–68%** (the equity control at 100% is the one clean pass) |
| Daily-return correlation against a public data source (4 products) | > 0.98 | **0.80–0.86** |

Two real bugs were found and fixed inside this check itself — a join fan-out that inflated overlap
counts by 10–25×, and null-handling that silently dropped gold's usable comparison rows to near
zero. Both are reflected in the numbers above.

What remains is a genuine, unresolved discrepancy rather than a bug. The raw evidence pulled for
the hygiene investigation showed a settlement price printing a materially different value from the
same contract's session high and low, which is circumstantial evidence that this vendor's close
field does not always equal the official settlement price other reference series use. That would
explain a *level* discrepancy but not fully explain return correlations as low as 0.80–0.86 against
a public source whose own roll conventions are unknown and plausibly differ.

**Read every number here as internally consistent** — the hygiene, roll-sensitivity and 2020-04-20
checks all agree with each other — **but not independently externally validated to the pre-declared
tolerance.** Stated plainly rather than left to be inferred from a pass/fail table.

### After hygiene

Excess kurtosis ranges from 0.64 (wheat, genuinely the thinnest-tailed product here, consistent
with its short history) to 53.10 (palladium). Every naive pre-hygiene artefact — platinum at 4,810,
natural gas at 1,893 — is gone. No product remains in the hundreds.

A stale-bar audit found the problem minor throughout: the worst run is three consecutive identical
closes, with 23 stale days out of roughly 5,000 for crude, and most products showing only one- to
two-day runs. No exclusion applied.

---

## The tail atlas

Moments, tail index on both tails, volatility clustering, the leverage effect, the Samuelson
effect, seasonality and named-event annotation, across all 16 products plus a Bitcoin bridge
series.

**Skew is mixed, not clean.** Grains lean positive as predicted (wheat +0.25, KC wheat +0.23, soy
meal +0.28, corn +0.06) — the low-inventory-upside-shock story holds best exactly where the
literature says it should. But energy and metals lean **negative** (crude −1.52, palladium −1.61,
silver −1.87), the opposite of the naive "commodities are right-skewed" prior.

That is plausibly because a 2010–2026 sample is dominated by large *downside* shocks — the 2014–15
oil collapse, the 2020 crash — rather than upside supply shocks. A sample-composition effect as much
as a structural one. The formal test tells the same story: 5 of 14 products match the predicted
sign, no better than a coin flip.

**The predicted inverse-leverage effect does not show up either.** The correlation between a return
and next-period volatility sits close to zero with wide, mostly zero-including intervals for most
products. Where a *significant* correlation appears (palladium, −0.234, interval excluding zero) it
is **negative** — the equity-style sign, not the predicted commodity sign.

An independent measurement agrees. The asymmetry parameter of the GJR models fitted later comes
back positive — the equity sign, where down-moves raise next-period variance more — in essentially
100% of refits for every energy product and the equity control, and mostly positive for the metals
(platinum 99%, gold 87%, silver 81%, palladium 74%). Grains are the one sub-group leaning the
predicted way: wheat 31% positive, KC wheat 23%, soy meal 8% — though corn, at 100% positive, does
not.

**Two separate measurements agree that the commodity inverse-leverage effect is, at best, a
grains-specific phenomenon in this sample, not a market-wide one.**

Every product's excess kurtosis is large and normality is rejected at p ≈ 0.

---

## Density selection

### Unconditional

Seven families — normal, Student-t, Hansen skew-t, normal-inverse Gaussian, generalised error,
Johnson SU and spliced EVT — fitted on an expanding walk-forward window (5 folds per product) and
scored out of sample.

**Five distinct families win somewhere** in the raw ranking, which directionally supports the "no
universal winner" story. **But no product's win is individually significant** once corrected for
multiple comparisons — the gaps between the top two or three families are real in ranking but too
close, at this sample size, to certify per product.

### Conditional

GARCH(1,1) and GJR, each with six innovation densities, plus one spliced-EVT model — 13 models per
product, rolling out of sample with annual refits and a minimum 750-day training window, over
2010-06-06 to 2024-12-31. About 80 minutes of compute across 16 products.

**GARCH with generalised-error innovations dominates, not GARCH-t.** Nine of 16 products pick it as
their out-of-sample winner, and the equity control picks its GJR variant.

That is a genuine, informative **departure** from the crypto result, where Student-t and
normal-inverse Gaussian led. The generalised error distribution has finite tails, flexible enough to
capture moderate excess kurtosis without Student-t's very heavy polynomial tails — and it appears
to fit commodities' more moderate (though still decisively non-normal) kurtosis better than crypto's
more extreme tails required.

Grains split toward skewed families exactly as predicted: corn, wheat and soy oil win with Hansen
skew-t; KC wheat and soy meal with normal-inverse Gaussian.

Only 2 of 16 products clear significance for their winner, and both are won by the same family —
which is why the "no universal winner" claim is reported as partial despite the qualitative pattern
holding.

### The expected-shortfall result is the cleanest thing here

**15 of 16 products reject normal-innovation expected-shortfall calibration** at the 5% level,
corrected for multiple testing, on both tails, at the 1% quantile. Only wheat's lower tail fails to
reject (p = 0.147).

This was called the highest-probability headline result in advance, and it delivered.

---

## Does term-structure state carry tail information?

Term-structure state (backwardation versus contango, from the front-to-second-month roll slope),
seasonal state (heating season for gas; planting, growing and harvest for grains), and macro state
(volatility-index terciles, yield-curve sign, policy-rate terciles, all lagged) were each tested
against a single reference model's 1% coverage — deliberately one consistent, cheap baseline for
every product rather than re-deploying the full 13-model battery per state.

**The claim does not fire, and the honest number matters here.**

A first pass scored "conditioning adds information" whenever *either* the pooled test failed and a
state also failed, *or* the pooled test passed while a state failed. That gave 14 of 16 — which
would have cleared the bar.

That criterion is too permissive. A product where the *pooled* test already fails is evidence about
the reference model's calibration, not evidence that conditioning reveals anything beyond it. The
correct, strict criterion — pooled test **passes** while at least one state's test **fails** — gives
**4 of 16**, well short of the 10-product bar.

Caught and corrected before being reported.

---

## Alpha: carry and momentum

**A declared scope cut, stated up front.** Six strategy families were planned. This pass implements
**carry** and **time-series momentum** only — the two most central in the literature — and
explicitly does not run cross-sectional momentum, basis momentum, spread mean reversion, or
hedging-pressure strategies. This follows the stated priority of cutting whole strategies rather
than thinning every strategy to fit the time available.

All 16 products, cross-sectional dollar-neutral construction (top and bottom 30%, roughly five names
per leg), with a futures-specific cost model, a block-bootstrap interval on excess return over an
equal-weight basket, and deflation over the true count of 20 configurations tried.

**Carry: a near-miss, not a wipeout.** Net Sharpe of 0.90–0.95 at every origin offset — genuinely
strong — and a deflated Sharpe probability of 0.997, which says this is very unlikely to be pure
multiple-testing luck.

What fails is the comparison test. The 95% interval on carry's return *minus the equal-weight
basket's* is **[−0.0028, +0.0094]**, which includes zero. Carry's absolute Sharpe is real; it is not
shown to **beat a passive commodity basket** by a margin this bar can distinguish from noise.

That matches the prior almost exactly: commodity carry survives on turnover grounds in a way
crypto's carry did not, but a 2010-onward sample sits entirely inside the post-financialisation
regime where the literature itself expects weak-to-zero net-of-cost alpha.

**Momentum: less ambiguous.** The best lookback (one month) gives net Sharpe 0.10–0.12 across
offsets — real but small — and is **sign-inconsistent across lookbacks**: three-month (−0.13) and
six-month (−0.11) are net-negative, while one-month and twelve-month are weakly positive. Deflated
Sharpe probability 0.098, nowhere near the bar.

---

## Intraday appendix (descriptive only)

One-minute bars for four energy products over six months in 2026, explicitly outside every
conclusion elsewhere in this notebook.

**The weekly petroleum status announcement shows a modest, consistent effect.** Mean absolute return
in a ±30-minute window around the announcement, on announcement days versus other weekdays: crude
1.05×, heating oil 1.08×, gasoline 1.11×. Small but directionally consistent across all three
relevant products, over 28 announcements.

**The realised-volatility signature plot is nearly flat** across 1, 5, 15, 30 and 60-minute
sampling (crude's mean daily figure moves only between 0.00142 and 0.00152), consistent with a
liquid, well-arbitraged market where microstructure noise isn't a first-order concern at this level
of aggregation.

A real bug was caught here: the hour-of-day extraction returns an 8-bit integer, and multiplying by
60 silently overflows its ±127 range for any hour from 3 onwards. Every minute-of-day computation
was corrupted — zero rows ever matched the announcement window — until fixed with an explicit cast.

---

## The risk engine

Family selection is read from each product's own out-of-sample ranking, never hardcoded.

**Coverage passes.** The naive version of this — a Value-at-Risk computed once from a full-sample
fitted density and held static — failed out-of-sample coverage badly in development, with violations
clustering exactly where the independence test is built to catch them. That's the textbook failure
of an unconditional VaR during a volatility regime shift.

The fix: the engine accepts a caller-supplied current volatility from a causal exponentially
weighted estimate and rescales the fitted shape's quantile by it. Shape is fixed per fold, scale
updates daily. This is deliberately a lighter conditioning step than a full GARCH refit, not a claim
that it replaces one.

### Portfolio risk under three dependence assumptions

Equal-weighted across all 16 products, 20,000 Monte Carlo draws each:

| Dependence assumption | 1% VaR | 1% Expected shortfall |
|---|---|---|
| Empirical (bootstrap of the real joint history) | 1.86% | **2.37%** |
| Gaussian copula | 1.60% | **1.97%** |
| t-copula (5 df) | 1.72% | 2.32% |

**The Gaussian copula understates portfolio expected shortfall by about 17%** relative to the
empirical estimate. That was the predicted failure mode, and it is now a measured number rather than
an assumed footnote.

Mean lower-tail dependence across all 120 product pairs: 0.146 empirical, 0.128 Gaussian, 0.177
t-copula. The t-copula, not the Gaussian, is the better match to observed tail co-movement.

### Stress scenarios

Replaying the named historical events at portfolio level: the 2014–15 supply collapse costs −7.97%,
and the 2023–24 normalisation window −21.4% (the largest figure, reflecting its multi-year span
rather than a single acute shock). The 2020-04-20 crude event shows only −2.66% at the diversified
portfolio level — underlining that a single-product tail event is a very different risk to a
portfolio than to that product alone.

A performance bug was caught before the full run: the normal-inverse Gaussian quantile function has
no closed form and root-finds its CDF per point, at about 50 milliseconds each. A naive
20,000-point draw would have taken roughly **16 minutes per affected asset**, making the
three-assumption comparison impractical. Fixed by building the CDF once on a grid and interpolating,
reducing a 20,000-point draw from ~16 minutes to ~5 milliseconds.

---

## The holdout

2025-01-01 to 2026-07-28, roughly 490 trading days per product, touched for the first and only time
here. Nothing is re-tuned on this window — every model evaluated was already fitted on the
development window, and this only scores those frozen fits.

**Risk-engine coverage replicates closely: 14 of 16 pass**, against 15 of 16 in development. Two
products that passed in development fail on holdout, but the overall rate is materially unchanged.

**The expected-shortfall finding replicates directionally: 11 of 16 reject**, against 15 of 16 in
development — weaker, as expected from a roughly 8× smaller sample reducing test power. These
holdout p-values are deliberately *not* multiple-testing corrected, since a 16-product correction on
an already underpowered sample would only weaken the picture further; they're reported raw for that
reason. The direction is unchanged.

**The fat-tail finding is directionally consistent but too noisy to trust quantitatively at this
sample size.** Tail-index estimates on ~490 observations are visibly less stable than the
~4,800–5,000-observation development estimates (crude's holdout upper-tail index of ≈1.48 against
development's 2.42). Still comfortably fat-tailed, but the point estimate shouldn't be
over-interpreted from a sample this short.

**The holdout confirms the central claims without materially changing any of them.**

---

## Bugs found

Eight, each documented in place above: the volume-only hygiene rule; the liquidity screen deleting
front-month data; the roll calendar listing nominally dead months; back-adjusted prices drifting
negative; a join fan-out and null-handling bug inside the validation check itself; an integer
overflow in the intraday minute-of-day computation; a per-point root-finding quantile function
making the Monte Carlo impractical; and an over-permissive scoring rule that would have reported a
false positive (14 of 16) before being tightened to the criterion that actually tests the claim
(4 of 16).

Eight is not a small number for one notebook. Each is reported here in the same detail as a positive
result.

## Bottom line

The two risk-side headline findings from the crypto work — fat tails understated by every
thin-tailed model, and a well-calibrated conditional risk engine being buildable at all — replicate
cleanly in an entirely different market structure, holdout included. That is no longer a claim about
crypto; it is a claim about financial returns generally, to the extent 16 commodities and one
equity-index control can support one.

The alpha side replicates too, in the negative. Two of the three most literature-favoured commodity
factors fail this repo's own bar exactly as crypto's factors did, in a market where a real edge was
the prior rather than the surprise.

Both genuinely open questions — whether tail asymmetry flips sign the way inventory theory predicts,
and whether term-structure state carries information beyond an unconditional model — came back
negative, at the same rigour as everything that fired.

The risk engine was the guaranteed deliverable, and it delivered: family selection driven entirely
by the fitted results, portfolio risk under three explicit and compared dependence assumptions, and
out-of-sample coverage that holds up on data no part of the pipeline touched until the very end.

## What to test next

- **Sharper tests of the two negative results.** Both used relatively coarse operationalisations — a
  plateau-range proxy for the tail-asymmetry comparison, and one shared reference model for the
  conditioning test. A purpose-built interval on the left-minus-right tail-index difference, and a
  state-conditioned refit of each product's *own* best model, might sharpen both nulls or reverse
  one.
- **The grains-specific inverse-leverage signal**, found independently in two different
  measurements. Worth a dedicated test of whether storage and inventory dynamics specific to the
  grain complex are the mechanism, rather than treating it as noise in an otherwise-null market-wide
  result.
- **The four strategy families cut from this pass** — cross-sectional momentum, basis momentum,
  spread mean reversion (with the roll-window exclusion discipline this data supports), and
  hedging-pressure strategies.
- **A lower-turnover carry construction.** The near-miss is close enough that a slower rebalance
  frequency or a wider no-trade band, using the turnover machinery built in notebook 007 but not yet
  applied here, could plausibly close the gap.
- **Resolving the external validation discrepancy properly.** The settlement-versus-close hypothesis
  was only checked for one contract. A systematic comparison across more contracts would either
  confirm it explains the gap or point to a real construction difference not yet found.

*Notebook: `src/research/008_commodity_tails_and_risk.ipynb`. Futures and spread mechanics are
defined in `docs/09-market-data-and-microstructure.md`; copulas and tail dependence in
`docs/01-probability-and-distributions.md`.*
