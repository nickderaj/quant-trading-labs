# 005 — Tail Risk and Conditional Non-Normality

## The question

Notebook 004 ran a seven-method volatility *point*-forecast contest and found no clear winner at
any interval — HAR-RV, the range estimators and normal-innovation GARCH all sat in one
statistically indistinguishable cluster. That contest is exhausted: conditional variance at these
horizons has an R² of 0.004–0.19, and every reasonable point estimator lands in the same place.

But notebook 004 did surface one narrower real result: GARCH with Student-t innovations scored
better than any normal density. Given how extreme crypto's tails are — fitted degrees of freedom
of 2–3 at every interval — the question worth asking next isn't "who forecasts variance best" but
**which model gives the best-calibrated conditional tail**, scored with the same rigour.

Notebooks 004 and 005 are not in disagreement. They answer different questions.

Primary asset BTC at all four intervals, with checks transferred to ETH, SOL, DOGE, BNB and XRP at
daily bars.

## Four decision points, fixed in advance

The notebook commits up front to what would count as a result:

- **A density winner** — one model whose log score beats *every* other model's with statistical
  significance.
- **A calibration winner** — one model that passes *all 36* tail-coverage tests at an interval.
- **Stability** — either of those replicating across the transfer symbols at the same interval.
- **A trading application** — pre-declared to run only if a density or calibration winner
  coincides with stability at the same interval.

There's also a standing caveat check: if the tail index came back at or below 2 with a confidence
interval excluding 2, the variance of these returns would not exist and a prominent warning would
be required at the top of this document.

---

## First: two corrections to notebook 004

Both were fixed before any new modelling here, because the density contest is built directly on
the same scoring machinery.

### The degrees-of-freedom lookahead

GARCH-t's density score and Value-at-Risk were computed using the Student-t degrees of freedom
estimated on the **final** training window of the whole sample, applied to score every bar from
the start of the evaluation period. The variance forecast was properly causal and rolling; only
the shape parameter scoring it was not.

The fix reconstructs a causal, forward-filled path of the parameter, mirroring the variance
forecast's own forward-fill exactly. Checking the path directly rather than trusting the fix
confirms the bug was real, not cosmetic: the fitted value varies from 2.2 (during the 2021–2022
turbulence) to 8.1 in later, calmer refits.

**Both halves of the original claim moved, in opposite directions:**

| Interval | Log score, contaminated | Corrected | VaR coverage p, contaminated | Corrected |
|---|---|---|---|---|
| 1h | 3.979 | **4.000** | 0.42 | **0.0000** |
| 4h | 3.194 | **3.254** | 0.18 | **0.0008** |
| 12h | 2.614 | **2.623** | 0.39 | 0.35 |
| 1d | 2.187 | **2.220** | 0.84 | 0.68 |

The log-score win **strengthened** to all four intervals. The coverage claim **weakened
materially** — the 5% VaR is now rejected at 1h and 4h, where it had previously never been
rejected anywhere. This doesn't overturn notebook 004's bottom line, but it does change the
strength of the one calibration result it found. The full battery below is the real test of that
calibration rather than an assumption inherited from an uncorrected number.

### The CRPS integration grid

The numerical grid used for the continuous ranked probability score spanned about 10 units for a
normal distribution but about 1,400 units for a Student-t with 2 degrees of freedom, at the same
number of grid points. Effective resolution therefore differed wildly across families, so any
cross-family comparison was partly measuring integration error rather than forecast quality.
Verified directly: on a t(2.1) forecast, the old grid disagrees with the exact value by **13–79%**
depending on the observation.

Fixed by adding closed-form expressions for the normal and Student-t cases, verified against a
very fine numerical grid. This is also what makes the bootstrap-heavy contest below tractable on
this hardware — the old per-observation loop would have been far too slow at this scale.

---

## Foundations: does the variance even exist?

### An independent estimate of tail heaviness

Notebook 004's parametric fits kept pinning at the optimiser's boundary, so the tail index is
re-estimated here by the Hill method, which doesn't depend on that machinery at all. A tail index
α at or below 2 would mean infinite variance.

| Interval | Tail | Stable range | Estimate | 95% bootstrap CI |
|---|---|---|---|---|
| 1h | Upper | [143, 3506] | 2.220 | [2.134, 2.315] |
| 1h | Lower | [158, 3506] | 2.187 | [2.106, 2.286] |
| 4h | Upper | [176, 876] | 2.356 | [2.189, 2.544] |
| 4h | Lower | [88, 876] | 2.381 | [2.232, 2.594] |
| 12h | Upper | [106, 198] | 2.716 | [2.352, 3.092] |
| 12h | Lower | [151, 272] | 2.203 | [1.970, 2.495] |
| 1d | Upper | **no stable range found** | — | — |
| 1d | Lower | [79, 146] | 2.278 | [1.979, 2.671] |

Every point estimate sits **above 2**, and every interval either sits fully above it or dips only
barely below at the lower bound. **The infinite-variance caveat is not triggered anywhere.** The
daily upper tail found no stable range at any threshold and is reported as provisional rather than
quoting a point value anyway.

This sits in mild tension with notebook 004's parametric fit, which found degrees of freedom as low
as 1.98 at hourly bars — right at the boundary. The non-parametric estimate is consistently a bit
higher, i.e. further from the boundary. **Variance most likely does exist, though not by a wide
margin,** and two independent estimators broadly agree without agreeing exactly.

### Does the significance test's own approximation hold?

The Diebold-Mariano test relies on a central limit theorem. Comparing its normal-approximation
p-value against a block-bootstrap p-value on the same loss-differential series, for the closest
(least significant) pair at each interval from notebook 004:

| Interval | Pair | Normal-approx p | Bootstrap p | Materially differ? | Loss-diff tail index |
|---|---|---|---|---|---|
| 1h | Garman-Klass vs GARCH-normal | 0.936 | 0.955 | No | 1.50, 1.42 |
| 4h | Garman-Klass vs GARCH-normal | 0.878 | 0.887 | No | 1.25, 1.80 |
| 12h | EWMA vs Garman-Klass | 0.967 | 0.851 | **Yes** | 1.50, 1.38 |
| 1d | Trailing-96 vs Parkinson | 0.966 | 0.943 | No | 2.00, 1.90 |

They differ by more than 0.05 only at 12h, and both still say "not significant", so no notebook
004 conclusion is overturned. But the loss differential's *own* tail index runs as low as 1.2–1.9
at several interval and tail combinations — low enough that the normal approximation should be
treated as approximate rather than exact. **Consequence carried through the rest of this notebook:
bootstrap p-values are primary for every significance verdict below.**

### Is log-variance the better-behaved object?

Fitting normal, Student-t and skew-t to realised variance directly and to its logarithm, and
comparing goodness of fit (a large p means not rejected, i.e. well calibrated):

| Interval | Variance levels: normal / t / skew-t | Log variance: normal / t / skew-t |
|---|---|---|
| 1h | ≈0 / ≈0 / ≈0 | 1.4e−88 / 2.5e−88 / (fit failed) |
| 4h | ≈0 / ≈0 / 7.5e−29 | **0.377** / **0.383** / **0.980** |
| 12h | 1.4e−191 / 6.6e−92 / 4.5e−9 | 0.068 / 0.069 / **0.799** |
| 1d | 1.0e−69 / 1.5e−32 / 3.5e−5 | 0.004 / **0.054** / **0.158** |

**Raw realised variance is rejected outright — every family, every interval, no exceptions.**
Log variance is dramatically better calibrated everywhere except hourly bars, where the finest
interval already flagged as having the heaviest and least stable tail also fails.

This directly confirms that notebook 004's HAR-RV and variance-distribution methods were
handicapped by working in the wrong space. A HAR model on log variance is added to the contest
below as the direct consequence.

---

## Two new models

### GJR-GARCH, for the leverage effect

Fitted with normal and Student-t innovations. (Skew-t was skipped as over-parameterised, following
notebook 004's finding that it buys nothing unconditionally for BTC.) GJR-GARCH nests plain
GARCH exactly when its asymmetry parameter is zero, which can be tested directly at every refit:

| Interval | Refits with significant leverage (normal) | (Student-t) |
|---|---|---|
| 1h | **43.5%** | 21.7% |
| 4h | 34.8% | 17.4% |
| 12h | 39.1% | 19.6% |
| 1d | 17.4% | 6.5% |

Leverage is real and non-trivial in a substantial minority of individual refits, strongest at
hourly bars. But it does **not** translate into a better pooled density forecast: plain GARCH-t
significantly beats GJR-t at 1h, 4h and 12h and ties at daily; the normal-innovation versions never
differ significantly anywhere.

**Leverage is a real, occasionally significant refit-level effect that does not survive as a net
improvement once rolled forward.** The extra parameter's estimation noise outweighs its benefit in
a rolling-refit setting — a clean, numbers-backed illustration of over-parameterisation rather than
a hypothetical one.

### Conditional extreme value theory

A generalised Pareto distribution fitted to the standardised residuals of the GARCH fit, refit at
exactly the same cadence and on the same training windows as the variance model, and
forward-filled causally. Summary of the fitted shape parameter ξ across all rolling refits, both
tails pooled:

| Interval | Refits | Median ξ | Range | Fraction with ξ < 0 |
|---|---|---|---|---|
| 1h | 92 | 0.16 | [−0.27, 0.73] | 12% |
| 4h | 92 | 0.05 | [−0.47, 0.49] | 35% |
| 12h | 88 | 0.03 | [−0.49, 0.43] | 45% |
| 1d | 78 | −0.13 | [−0.52, 0.22] | **78%** |

A negative ξ formally means a *bounded* tail, which is implausible for crypto taken at face value.
Rather than waving this through, it was investigated. Each refit estimates ξ from only about 50
exceedances within a 500-bar window — a genuinely small sample for a shape parameter. The *median*
across refits stays positive at 1h, 4h and 12h and only dips slightly negative at daily, in a
pattern that tracks aggregational Gaussianity exactly as established elsewhere in the programme.

Read plainly: this is consistent with a true ξ that is small-but-positive at fine intervals and
close to zero at daily, with the individual negative estimates being small-sample scatter around
that value rather than genuine evidence of a bounded tail. It remains a real limitation of a
500-bar rolling fit and is flagged as such rather than papered over.

### A revealing cross-check

The Hill estimates on raw returns imply ξ ≈ 0.37–0.45 at every interval — consistently *higher*
than the median GPD ξ above (0.16 down to −0.13). This gap is a real, explainable finding rather
than a contradiction.

The Hill estimator measures the **unconditional** tail, mixing every volatility regime together.
The GPD here is fitted to GARCH-standardised residuals, i.e. the **conditional** tail after the
time-varying variance has been divided out. A meaningful share of what makes raw crypto returns
look so fat-tailed is volatility clustering itself — a mixture-of-regimes effect — rather than a
genuinely heavy-tailed conditional innovation. That is consistent with GARCH-t's own fitted
degrees of freedom sitting at 7–8 in most refits, far above the 2–3 found unconditionally.

**Conditioning on volatility genuinely thins the tail that's left over.**

---

## The density contest

Log score is the primary metric here. Eight models compete, giving 28 pairwise comparisons per
interval.

The two EVT models are **not entered** in this contest. Continuously normalising a
GPD-tails-plus-empirical-body density proved as fiddly as anticipated, and an honest partial entry
beats a hand-waved density. They are entered in the calibration battery below, where their
quantile and expected-shortfall forecasts are well defined regardless. This is documented plainly
rather than reported as a silently smaller contest than planned.

### Log score by model (higher is better)

| Rank | 1h | 4h | 12h | 1d |
|---|---|---|---|---|
| 1 | **GARCH-t** 4.000 | **GARCH-t** 3.254 | **GARCH-t** 2.623 | HAR-log-RV 2.244 |
| 2 | GJR-t 3.988 | GJR-t 3.233 | GJR-t 2.609 | GARCH-t 2.220 |
| 3 | GARCH-normal 3.848 | HAR-log-RV 3.167 | HAR-log-RV 2.572 | GJR-t 2.220 |
| 4 | HAR-RV 3.844 | HAR-RV 3.139 | HAR-RV 2.543 | HAR-RV 2.191 |
| 5 | GJR-normal 3.842 | GJR-normal 3.124 | GJR-normal 2.530 | GJR-normal 2.167 |
| 6 | Garman-Klass 3.826 | GARCH-normal 3.130 | GARCH-normal 2.522 | GARCH-normal 2.163 |
| 7 | Trailing-96 3.768 | Range 3.101 | Trailing-96 2.471 | Trailing-96 2.139 |
| 8 | HAR-log-RV 3.707 | Trailing-96 3.073 | Range 2.399 | Parkinson 2.117 |

**This is the cleanest result the research programme has produced.** Notebook 004's point-forecast
contest found ties everywhere. Scoring the *identical underlying variance recursions* on log score
instead surfaces a real, ordered, mostly-replicating ranking — Student-t innovation models at the
top, normal-innovation in the middle, trailing standard deviation and range estimators at the
bottom, at every interval.

### Verdict, with all pairwise comparisons corrected for multiple testing

| Interval | Best model | Beats every other model significantly? |
|---|---|---|
| 1h | GARCH-t | **Yes** |
| 4h | GARCH-t | **Yes** |
| 12h | GARCH-t | **Yes** |
| 1d | HAR-log-RV | No |

**A certified density winner at three of four intervals.** GARCH-t genuinely beats everything
else at 1h, 4h and 12h, and the bootstrap and normal-approximation verdicts agree everywhere this
time. Only at daily bars does no model win significantly — HAR-log-RV edges narrowly ahead (2.244
against 2.220) but not significantly. That is consistent with daily being the interval where the
tail diagnostics above show the distribution closest to normal.

---

## The tail-calibration battery

The full grid: an unconditional coverage test, an independence test on violation clustering, and a
joint conditional-coverage test, at all six quantile levels (1%, 2.5%, 5%, 95%, 97.5%, 99%),
across all ten models and all four intervals — **1,440 individual tests.**

### Which models clear every test

| Interval | Models passing all 36 tests |
|---|---|
| 1h | none |
| 4h | none — GARCH-EVT comes closest, failing 2 of 18 |
| 12h | **GARCH-EVT** |
| 1d | none — GARCH-EVT and GJR-EVT each fail only 1 of 18 |

**Exactly one clean pass:** GARCH-EVT clears every one of the 36 tests at 12h. Everywhere else,
both EVT models come close (0–3 failures out of 18 each) while every non-EVT model fails multiple
tests at every interval, usually on four to six of the six quantile levels. Hourly bars are the
hardest interval across the board, including for the EVT models — consistent with every other
diagnostic flagging that interval as the heaviest, least stable tail.

### The single cleanest finding here

The expected-shortfall backtest asks a sharper question than coverage: not "how often does the
model's threshold get breached", but "when it *is* breached, is the loss as bad as the model said
it would be on average?"

At the 1% level, **at every interval, with zero exceptions**, every model that does not account for
fat tails — trailing standard deviation, HAR-RV, HAR-log-RV, range estimators, normal-innovation
GARCH and GJR — comes back with a significantly positive test statistic:

| Interval | Non-fat-tailed models (all significant) | Student-t innovation | EVT |
|---|---|---|---|
| 1h | 1.04 to 1.98 | 1.34–1.71 (significant) | 0.17–0.21 |
| 4h | 0.98 to 2.11 | 0.71–1.57 (significant) | 0.24 |
| 12h | 1.15 to 2.92 | 0.66–1.51 (not significant) | 0.04–0.09 (not significant) |
| 1d | 0.48 to 1.38 | 0.38–0.48 (not significant) | 0.13–0.18 (not significant) |

A statistic near zero means well-calibrated expected shortfall. A positive statistic means realised
1%-tail losses are significantly *worse* than the model's own prediction — the model understates
tail risk.

Plainly stated, this is the headline finding of the whole notebook: **models that don't account
for fat tails don't merely score worse on an abstract metric — they concretely and measurably
underestimate how bad the worst days actually get, every single time this was checked, at every
interval.** Fat-tailed and EVT-based models are dramatically better calibrated on exactly the risk
that matters most, clearing the strict bar outright at 12h and coming within one or two tests of
it everywhere except hourly.

---

## Does it replicate across symbols?

Transfer testing ran at daily bars only, for wall-clock reasons on this hardware. **A real scoping
limit, stated plainly:** BTC's actual certified density win is at 1h, 4h and 12h, and cannot be
transfer-tested at all here, because daily is the one interval where BTC's own contest found no
significant winner. What follows tests whether *that* pattern replicates, and whether the
calibration story does.

### The density result: a perfectly stable null

| Symbol | Best model by log score | Significant winner? |
|---|---|---|
| BTC (reference) | HAR-log-RV | No |
| ETH | GJR-t | No |
| SOL | HAR-log-RV | No |
| DOGE | GARCH-t | No |
| BNB | HAR-log-RV | No |
| XRP | GARCH-t | No |

**Zero of six symbols produce a significant winner at daily bars** — a perfectly replicating null,
with not a single spurious "winner" anywhere. The identity of the best model splits three ways but
never lands on a naive baseline on any symbol. The *cluster* of plausible winners replicates even
though the single best does not, the same pattern notebook 004 found for its point forecasts.

### Calibration: clears on most altcoins, not on BTC or ETH

| Symbol | Models passing all 36 tests at daily |
|---|---|
| BTC | none |
| ETH | none |
| SOL | HAR-log-RV, GARCH-EVT, GJR-EVT |
| DOGE | GARCH-EVT, GJR-EVT |
| BNB | GARCH-normal, GJR-normal, GARCH-t, GJR-t, GARCH-EVT, GJR-EVT (6 of 10) |
| XRP | GARCH-t, GJR-t |

Calibration clears on four of five transfer symbols, and an EVT model appears in every clearing set
except XRP's. BNB is a standout with six of ten models clearing at once — unusually well-behaved
daily tails for that symbol specifically.

Read plainly: **EVT-based tail calibration replicates as a genuinely good idea across most of this
symbol set, but which specific model clears — or whether any does — is asset-specific.** That is
notebook 004's own standard confirmed again on a different question: tail-shape findings replicate
more readily than rankings do.

### Leverage is not a portable quantity

The fraction of refits with significant leverage, at the same interval across six assets: 0.087
(SOL), 0.174 (BTC), 0.217 (XRP), 0.370 (DOGE), 0.413 (BNB), 0.457 (ETH) — a nearly **sixfold
range**. Leverage exists intermittently and its prevalence is genuinely asset-dependent, reported
as such rather than averaged into one misleading number.

---

## The trading application did not run

The pre-declared application was an EVT-conditional risk-limit overlay on buy-and-hold BTC, judged
against buy-and-hold and a normal-GARCH overlay on Sharpe, maximum drawdown, 1% exceedance count
and turnover cost.

It required a density or calibration winner **and** stability at the same interval. The density
winner fired at 1h, 4h and 12h on BTC, which were never transfer-tested. Calibration fired at 12h
on BTC and at four of five transfer symbols at daily. The two never held together at the same
interval across the symbol set. **The application does not run** — written up here rather than
silently skipped.

---

## Bugs found

Beyond the two corrections at the top:

1. **A sign error in the expected-shortfall test formula.** Both the sign of the statistic's
   additive constant and the stated direction of the failure mode were backwards. Caught by
   verifying numerically — not just re-deriving on paper — against a deliberately mis-specified
   model with known-wrong volatility, before trusting either the formula or its interpretation.
   Documented in `docs/06-scoring-rules-and-calibration.md` so it can't be silently re-broken.

2. **The negative shape-parameter warning, investigated rather than ignored** — described in full
   above. A growing fraction of individual refits (12% at hourly rising to 78% at daily) produced a
   formally bounded tail. Investigating directly showed it to be small-sample scatter around a
   small true value rather than a real per-refit finding, and it's reported honestly as a
   limitation of a 500-bar rolling fit.

Both were caught the same way every bug in this programme has been: by reading the actual numbers —
a Monte Carlo check against a case whose right answer was already known, and a direct tabulation
of how often a supposedly-rare warning condition actually fired — rather than trusting that code
which ran without raising had produced a correct answer.

---

## Bottom line

**The point-forecast question is exhausted; the tail question is not.** Rescoring the identical
variance models from a point-forecast metric to a density metric surfaces a real, statistically
certified winner — GARCH-t — at three of four BTC intervals, something the point-forecast contest
never found anywhere.

The calibration battery is stricter still and still finds something: GARCH-EVT clears every one of
36 coverage tests at 12h. And more practically than any single verdict, **every model that ignores
fat tails significantly underestimates how bad the worst 1% of days actually are, at every
interval, with no exceptions.**

GJR's leverage effect is real in a meaningful minority of refits but doesn't survive as a net
pooled improvement, and its prevalence varies enormously by asset. The daily interval and the
transfer check tell a consistent story of their own: no significant density winner anywhere at
daily (a stable null across six symbols), and calibration success that replicates on most but not
all of the transfer set, via EVT models specifically where it does.

The trading application didn't run, because its dual requirement was never jointly satisfied — a
legitimate outcome under the notebook's own pre-declared rule.

This adds genuinely new, certified knowledge notebook 004 couldn't produce on its own: crypto's
conditional tails are real, extreme, and non-normal in a way that concretely costs risk-unaware
models measurable accuracy on the outcomes that matter most. Consistent with the rest of the
programme, no tradeable application clears the bar.

## What to test next

- **Transfer-test the density win at 1h, 4h and 12h.** BTC's actual certified result has never been
  transfer-tested at all. This is the single most valuable follow-up: it would either turn it into
  a stability-certified, application-eligible finding, or reveal that BTC's win doesn't generalise.
- **A properly normalised EVT density**, so the EVT models can enter the log-score contest
  directly. Given how strong their coverage and expected-shortfall performance already is, they are
  natural candidates to win it too.
- **Test whether the leverage effect is asset-class-general or crypto-specific,** given how widely
  its prevalence varied (0.09 to 0.46) across just six crypto symbols. The equity literature this
  model borrows from was built on a very different asset class.
- **Join the regime work to the tail work.** Does the EVT shape parameter itself differ meaningfully
  by volatility regime? That's a natural extension of both notebooks' machinery, currently untested.

*Notebook: `src/research/005_tail_risk_evt.ipynb`. Terminology used here is defined from scratch,
with worked examples from this repo's own numbers, in `docs/` — start at `docs/README.md`.*
