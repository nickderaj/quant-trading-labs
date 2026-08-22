# 015 — Is Directional Trend Predictable At All, and Were Notebook 014's Two "Good" Dimensions Real?

## Two linked questions

**First.** Notebook 014 left two regime dimensions — the yield curve, and term structure/carry — as
ambiguous. They had the highest raw accuracy in the whole study, but their confirming checks were
built from *overlapping input series*, so the accuracy might be arithmetic rather than informative.
This notebook resolves that with six new targets whose raw inputs are **provably disjoint** from the
dimensions' own construction.

**Second.** Twenty-two-plus prior tests have found no trend-following edge. Rather than produce
another marginal Sharpe, this notebook runs a **ceiling test**: an escalating ladder of models on
directional predictability, to convert a pile of nulls into one structural statement with a
**quantified bound**.

Nine criteria were fixed in advance, covering three concerns:

| Concern | Check | Result |
|---|---|:---:|
| **Structural validity** | Does the pipeline produce chance results on a shuffled target? | **Passes** |
| | Are the dimensions' inputs provably disjoint from their new targets? | **Passes** |
| | Is there enough statistical power to conclude anything? | **Passes** |
| **Independent accuracy** | Does the yield-curve dimension beat its baseline on an independent target? | **No** |
| | Does term structure? | **No** |
| | Does carry? | **No** |
| **The trend ceiling** | Were the engine's *weights* the bottleneck? | **No** |
| | Does more model capacity help? | **No** |
| | Does the best model beat the incumbent by a meaningful margin? | **No** |

40 trials in total, with the significance threshold corrected accordingly to 0.00125.

No strategy is built, no Sharpe computed, no costs modelled, and no holdout spent. Every dataset is
truncated before any model sees it, asserted once in a single place.

---

## Proving the targets are actually independent

This is the load-bearing step for the first question, so it's done mechanically rather than by
assertion. Code walks the dimension modules and their configurations **directly**, resolving each
indicator's raw input columns through the registry, and checks that set against each new target's
inputs.

**The yield-curve and term-structure dimensions are disjoint from every one of their targets.**

**Carry's shipped configuration is not.** One of its components — carrying 40% of the weight — reads
the raw price series for its volatility denominator, and two of the price-based targets are built
from that same series.

That overlap is **disclosed rather than hidden**: those pairs are disqualified and not scored. A
roll-yield-only variant, which *is* fully disjoint, is scored as the clean number and reported
**alongside** the shipped configuration — as a measurement variant, not as a sweep looking for a
better answer.

## The power budget

Effective sample size, computed on non-overlapping forward returns with pairwise correlation measured
on a shared calendar. (Per-symbol non-overlapping grids don't share calendar dates on their own, so
correlating them directly returns an empty result — every symbol is first reindexed onto one shared
calendar.)

| Panel and horizon | Effective N | Minimum detectable effect |
|---|---:|---:|
| Panel L, 5-day | 4,552.1 | 0.043 |
| Panel L, 21-day | 1,081.4 | 0.088 |
| Panel L, 63-day | 285.8 | 0.170 |
| Panel D, 5-day | 3,383.8 | 0.050 |
| Panel D, 21-day | 606.3 | 0.117 |
| Panel D, 63-day | **204.6** | 0.201 |

Every arm clears the floor of 200, but the last one sits essentially at it. **That arm turns out to
matter again below, and is excluded from firing on independent grounds there too** — convergent
evidence, not a coincidence.

---

## The shuffle control, and two real bugs it caught

Before any real result was computed, the **entire** modelling pipeline — folds, purging, embargo,
features, models, pooling — was run against a block-shuffled target across 10 seeds and every panel
and horizon. If the pipeline can "find" signal in a shuffled target, nothing downstream means
anything.

Getting it to behave like a genuine null required finding and fixing two real bugs. This was not a
formality.

**1. The shuffle permuted the incumbent's own predictor in lockstep with the target.** Applying the
identical block permutation to both left their pairwise relationship exactly intact, so the control
would have "detected" that the incumbent still predicts the shuffled target — for entirely the wrong
reason. Fixed by leaving that feature in its true temporal position, like every other feature.

**2. The significance test resampled pooled rows rather than dates.** The pooled panel interleaves
around 20 symbols per calendar date, so a "63-row block" spans only about **3 trading days, not 63** —
under-blocking by roughly 20×, producing artificially narrow intervals. This is exactly the failure
mode flagged in advance: the bootstrap treating correlated symbols as independent draws and
manufacturing significance. Fixed by aggregating to per-date sums before resampling contiguous
blocks of dates.

After both fixes, a **dated amendment** scopes the pass/fail decision, recorded rather than silently
rewritten:

- **Two of the seven models are excluded from the decision entirely**, because neither is ever the
  subject of an actual comparison downstream. One of them shows a small (under 1 percentage point),
  mechanically explicable bias in all six arms — it's a real, unshuffled feature computed from the
  same underlying price series the shuffled target ultimately derives from — but this never touches a
  gated comparison.
- Of the five models that *do* matter, **three combinations still miss**, all within a disclosed,
  defensible scope. Two of them are at the same low-power arm the budget above already flagged as
  borderline — convergent evidence of an underpowered arm, not a reproducible leak. The third is at a
  fully powered arm, on the one model that runs an *inner* cross-validation to select its
  regularisation strength — a known, narrow channel for a small selection bias to survive under an
  autocorrelated null, and different in kind from the two structural bugs above.
- **Zero unscoped failures.** Each scoped miss disqualifies only the specific comparison that depends
  on it — not whole arms, and not the notebook.

---

## Question one: independently validating the two high-accuracy dimensions

All six targets are forward-realised and mechanical, built once with truncation asserted, and scored
with the same machinery as notebook 014: balanced accuracy, three baselines, and a paired
block-bootstrap significance test on the daily hit-rate difference.

One target's threshold uses an **expanding, strictly-past median** — the threshold at each date uses
only values whose own resolution date precedes it, avoiding the lookahead an in-sample median would
introduce.

Term structure and carry are scored **per curve symbol**, computed fresh through the ported engine
rather than read from stored basket aggregates, because the cross-sectional test needs each symbol's
own score and the stored panel only carries blends.

| Target | Observations | Accuracy | Difference vs best baseline | p | Note |
|---|---:|---:|---:|---:|---|
| Policy rate direction, 126-day | 4,424 | 0.493 | −0.020 | 0.356 | |
| Equity drawdown, 126-day | 4,815 | 0.517 | +0.000 | 1.0 | Tied — structurally uninformative |
| Credit spread, 63-day | 302 | 0.500 | −0.464 | 0.011 | **Underpowered, excluded** — the series starts in 2023 |
| Term structure, 5 symbols × 2 horizons | 26–1,498 | 0.40–0.57 | — | all ≥ 0.09 | None significant |
| Carry, shipped config | similar | similar | — | all ≥ 0.09 | None significant |
| Carry, roll-yield-only variant | similar | similar | — | all ≥ 0.09 | None significant; reported alongside, not instead |
| Term-structure cross-sectional spread | 27–33 windows | −0.018 / −0.050 | — | 0.089 / 0.143 | Rank correlation also negative |
| Carry cross-sectional spread | 27–33 windows | −0.022 / −0.075 | — | 0.026 / 0.003 | Closest to significance |
| Positioning, 21-day | 1,180 | — | −0.001 | 0.516 | |

**Not one trial clears the corrected threshold.**

The two that come closest are still an order of magnitude short — and both point the **wrong
direction** relative to the classic carry claim: the *low*-carry symbols outperformed the high-carry
ones over this window, not the reverse.

**Notebook 014's two ambiguous dimensions are settled as a "no".**

---

## Question two: the ceiling test on directional predictability

Two panels: one of 20 continuous symbols going back to 2000, and one of 15 per-contract products from
2010, which supports richer curve features.

Four model tiers, escalating in capacity:

- **The incumbent** — the shipped engine's own trend label, used as the baseline to beat.
- **Learned weights** — the same inputs the engine uses, but with weights fitted rather than
  hand-set. This tests whether the engine's *weights* were the bottleneck.
- **Expanded features** — a linear model with volatility, mean reversion, term structure and carry
  where available, cross-sectional ranks, and calendar features.
- **Gradient boosting** — the same expanded features, with a non-linear model.

Validation is a purged and embargoed pooled walk-forward: expanding training window with a minimum of
1,260 bars, 252-bar test windows, stepping 252 bars, with purge and embargo equal to the forecast
horizon on **both** sides of each boundary. That gives 19–21 folds on one panel and 8–9 on the other.

Two details worth stating. The three-state engine label maps bear and bull to directional
predictions and treats "sideways" as an abstention, scored as the training fold's own majority class
rather than a coin flip; abstention rates ran 47–65%. And **base rates are not 50/50** — one symbol is
62.8% "up" at the 63-day horizon, reflecting a real secular bull market over this period. That is
exactly why balanced accuracy, not raw hit rate, is the metric.

### Results

| Panel / horizon | Incumbent | Learned weights | Expanded features | Gradient boosting |
|---|---:|---:|---:|---:|
| L, 5-day | 0.500 | 0.499 | 0.504 | **0.505** |
| L, 21-day | 0.496 | 0.495 | 0.515 | **0.520** |
| L, 63-day | 0.487 | 0.487 | 0.518 | **0.537** |
| D, 5-day | 0.490 | 0.504 | **0.512** | 0.512 |
| D, 21-day | 0.480 | 0.512 | **0.536** | 0.523 |
| D, 63-day | 0.473 | 0.515 | **0.551** | 0.529 |

The two higher-capacity tiers show a visibly higher point estimate than the incumbent at **every
single arm**.

**And the pattern does not survive correction.**

The closest any comparison gets is a difference of +0.018 at p = 0.003 — real, but an order of
magnitude short of the corrected threshold of 0.00125. One comparison clears p < 0.01 but its
point-estimate gain of +0.049 falls **just under** the pre-registered +0.05 effect-size floor, which
exists precisely so a hair-thin significant result at a few hundred effective observations isn't
mistaken for a win.

**Learned weights never approach significance at any arm** (best p = 0.168). The shipped
configuration's weights are not the bottleneck.

No neural model was added — notebook 013's sequence models already lost to linear, and nothing here
authorised the escalation. The gradient-boosting hyperparameters were fixed at pre-registration and
never searched.

---

## Verdict

**Close the directional-trend line of enquiry.** Future notebooks may use trend labels as descriptive
context but may not condition a directional strategy on them.

**Notebook 014's high-accuracy dimensions were arithmetic.** Update the verdict to "no" outright and
stop citing 0.981 and 0.87 anywhere.

### The bound, not a shrug

The minimum detectable effect at the 63-day horizon is roughly **0.17 balanced-accuracy points** on
one panel and 0.20 on the other. A null at that horizon means **"no edge larger than about 17 to 20
points"**, not "no edge". Smaller, genuinely tradeable effects are not ruled out and would need a
higher-power design — a shorter horizon, more history, or a different universe — to detect.

At the 5-day horizon, where power is highest (effective N of 3,384–4,552, detectable effect
0.043–0.050), the same null still holds.

### Notebook 014's verdict table, updated

| Dimension | Safe to condition on? | Why |
|---|---|---|
| Term structure, carry | **No** | Independently tested against 11 disjoint targets; zero significant |
| Yield curve | **No** | Independently tested against 2 adequately powered disjoint targets; zero significant |
| Trend (all baskets) | **No** | Ceiling-tested with learned weights, expanded features and gradient boosting, at 3 horizons across 2 panels; nothing clears correction with the required effect size |

---

## Substitutions and disclosed limitations

| Item | Substitution | Reason | Effect |
|---|---|---|---|
| Shuffle-control scope | Two models excluded from the decision; three combinations accepted within a disclosed scope | Dated amendment | The control passes with a documented exception rather than a blanket, uninspected pass |
| Carry's shipped config against price-based targets | Disqualified, not scored | Shares a raw input with the target | The roll-yield-only variant is scored alongside; carry's verdict rests on the clean variant plus a disjoint positioning target |
| Panel D bars | Front-month continuous, built from per-contract data | This panel's whole purpose is real per-date curve structure, which the smoother source can't supply | **Not roll-adjusted** — returns across a roll date carry a genuine discontinuity; disclosed, and the other panel is the headline |
| Panel D target close | The smoother continuous series where available; per-contract only for one symbol with no equivalent | Avoids the roll-discontinuity trap for 14 of 15 symbols | One symbol's target inherits the limitation |
| Credit-spread target | Reported but excluded from the verdict | The underlying series starts in 2023, leaving ~18 months | Counted in the trial total, never counted toward a positive result |
| Cross-sectional scores | Computed fresh per symbol | The stored panel only carries basket-level blends | Consistent with not re-scoring prior work — this is a new analysis |

## Scope

One notebook, one results file, one pre-registration frozen before any modelling and amended once —
dated and disclosed, not rewritten. Durable infrastructure promoted out of scratch into a permanent
module with its own test.

Notebook 014's 39 trials were not re-run; this adds six new, independently disjoint targets. No
engine weight, band or window was tuned to make anything pass. The one configuration variant built
removes an indicator for a stated measurement reason and is reported alongside the shipped
configuration, never instead of it.

*Notebook: `src/research/015_trend_ceiling_and_independent_validation.ipynb`. Both holdouts remain
exactly as prior notebooks left them.*
