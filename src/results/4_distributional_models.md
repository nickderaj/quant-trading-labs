# Distributional Models for Volatility and Regime - Results Summary

Notebooks 1-3 all asked distributions to predict the first moment (next return) and all
three found nothing. This notebook resets to single-asset, deliberately basic, and asks
the two questions those notebooks never asked: can distributional modelling forecast
**volatility** (the second moment) better than the trivial baselines already in use, and
can it identify persistent **regimes** that are informative about anything? Both are
judged as forecasting contests with proper scoring rules, not backtests - Phase 5 only
runs if Phase 3 or 4 produces an actual winner.

Primary asset: **BTC**, all 4 intervals (1h/4h/12h/1d). 1h is back in for this notebook
(dropped in notebook 3 only because its *transaction-cost drag* was unviable - a
volatility-forecasting contest places no trades, so that guardrail doesn't apply here).
Frozen-and-transferred to ETH/SOL/DOGE/BNB/XRP where noted.

Full machinery: `src/distributions.py` (`fit_rolling`, scoring rules - Phase 2, already
built and committed separately) and `src/research/tmp/dist_lib.py` (notebook-4-specific
feature engineering, forecasting rungs, GARCH/GMM/HMM from-scratch fits - this notebook's
own shared library, mirroring how `backtest_configs.py` supported notebook 3).

## Phase 1 - Descriptive: what these series actually look like

All Phase 1 numbers are **fit once on the full pre-holdout history** (causal-to-date, not
rolling - Phase 1 is explicitly the "characterize the data" phase; Phase 3/4 are where
everything becomes a rolling, refit, out-of-sample forecast). BTC, all 4 intervals.

### Fat tails

| interval | n | normal (mu, sigma) | Student-t df | skew-t (a, b) | frac \|z\|>=5sigma (normal) | normal-implied | ratio obs/implied |
|---|---|---|---|---|---|---|---|
| 1h  | 35,064 | (3.2e-5, 0.00584) | **1.98** | (1.25, 1.27) | 0.00814% | 0.00000011% | **7,114x** |
| 4h  | 8,766  | (1.3e-4, 0.01156) | **2.10** | (1.30, 1.34) | 0.00811% | 0.00000016% | **5,174x** |
| 12h | 2,922  | (3.9e-4, 0.02060) | **2.23** | (1.32, 1.32) | 0.00959% | 0.00000027% | **3,582x** |
| 1d  | 1,461  | (7.9e-4, 0.02881) | **2.88** | (1.39, 1.33) | 0.01711% | 0.00000717% | **2,388x** |

A normal fit underestimates 5-sigma-day frequency by **2,400x to 7,100x** depending on
interval - not "fat tails," a distribution that is simply the wrong model. Fitted
Student-t degrees of freedom sit between 2 and 3 at every interval (df=2 is the edge of
having a finite variance at all), consistent with the crypto-GARCH literature's finding
that these are among the heaviest-tailed liquid asset return series traded.

**PIT/KS calibration** (KS statistic, p-value; small p rejects "this family's fit is
well-calibrated"):

| interval | normal | t | skew-t |
|---|---|---|---|
| 1h  | 0.108, p~0 | 0.0069, p=0.069 | 0.0118, p=0.0001 |
| 4h  | 0.101, p~0 | 0.0164, p=0.017 | 0.0223, p=0.0003 |
| 12h | 0.094, p~0 | 0.0253, p=0.047 | 0.0293, p=0.013 |
| 1d  | 0.078, p~0 | 0.0270, p=0.231 | 0.0251, p=0.313 |

Normal is rejected outright at every interval (KS statistic 0.08-0.11, p effectively
zero). Student-t is not rejected at 1d (p=0.23) and is borderline elsewhere; skew-t is
*not* an improvement over plain t on this data (worse or comparable KS at every interval,
including a strict rejection at 1h/4h/12h) - BTC's estimated skew parameters (a, b) are
close to symmetric (a~b), so the extra shape parameter buys calibration noise, not fit.
**t is the best simple parametric fit found here; skew-t is not worth its complexity for
BTC log returns specifically** (revisited in Phase 3 rung 5, where GARCH innovations are
a different, conditional question).

A 2-component Gaussian scale mixture (EM) recovers a low-vol component (weight ~0.79,
variance ~10x smaller) and a high-vol component (weight ~0.21) at 1h, with the pattern
holding at all 4 intervals (weight on the high-vol component rises to ~0.44 at 1d as
squeezing 5 years into ~1,461 daily bars puts more regime-switching inside a smaller
sample) - this is volatility clustering, described as a static mixture rather than a
dynamic process (Phase 4 makes it dynamic).

### Aggregational Gaussianity

Fitted t degrees of freedom **rise monotonically with interval**: 1.98 (1h) -> 2.10 (4h)
-> 2.23 (12h) -> 2.88 (1d). Returns become measurably more normal as you aggregate, the
textbook aggregational-Gaussianity pattern - but crypto starts from an extreme baseline
(df~2, barely inside "finite variance") and even at 1d is still far from Gaussian
(df->infinity). Aggregating 24 hourly bars into one daily bar buys less than one
additional degree of freedom's worth of normality.

### Volatility clustering as a distributional statement

Waiting times between k-sigma moves, exponential vs gamma fit (gamma shape k<1 = the
waiting-time distribution is over-dispersed relative to a memoryless Poisson process =
big moves cluster):

| interval | k=2sigma gamma shape | k=2sigma KS(gamma) p | k=3sigma gamma shape | k=3sigma KS(gamma) p |
|---|---|---|---|---|
| 1h  | 0.63 | 6e-25 | 0.52 | 2e-7 |
| 4h  | 0.74 | 7e-6 | 0.64 | 0.070 |
| 12h | 0.80 | 0.088 | 0.67 | 0.275 |
| 1d  | 0.85 | 0.390 | 0.66 | 0.944 |

Shape < 1 at every interval and every threshold - waiting times are over-dispersed,
i.e. big moves genuinely cluster rather than arriving as a Poisson process, and even
gamma (which already captures some of this) is itself rejected by KS at 1h/4h (its own
functional form isn't quite right at high frequency, though it's a much better
description than exponential - exponential KS statistics are 2-3x larger at every row,
not tabulated in full here). Clustering is strongest at 1h and weakens, but never
disappears, at 1d - the same aggregational pattern as the tail-index result above.

### Overdispersed activity

Trade-count dispersion index (Var/Mean, Poisson predicts exactly 1):

| interval | dispersion index | NB (n, p) |
|---|---|---|
| 1h  | 114,393 | (1.33, 8.7e-6) |
| 4h  | 300,973 | (2.01, 3.3e-6) |
| 12h | 617,751 | (2.94, 1.6e-6) |
| 1d  | 891,044 | (4.08, 1.1e-6) |

Dispersion indices in the hundreds of thousands, not near 1 - per NEW_PROMPT's own
tripwire ("if it comes back near 1, suspect an aggregation bug"), this is the expected
and correct result: trade counts are massively overdispersed, not remotely
Poisson-shaped, and a negative binomial is a far better (though still imperfect - not
KS-tested here since NB's discrete CDF makes exact KS awkward at this scale) description.

### Bounded observables

Beta fits to `taker_buy_ratio` and intrabar close position `(close-low)/(high-low)`:

| interval | taker_buy_ratio (a, b) | alpha+beta | intrabar_close_pos (a, b) | alpha+beta |
|---|---|---|---|---|
| 1h  | fit failed (boundary obs) | - | fit failed (boundary obs) | - |
| 4h  | (245.8, 247.7) | 493 | fit failed (boundary obs) | - |
| 12h | (570.5, 574.9) | 1,145 | (1.45, 1.34) | 2.79 |
| 1d  | (869.3, 875.8) | 1,745 | (1.35, 1.25) | 2.60 |

`taker_buy_ratio` is tightly concentrated near 0.5 (alpha~beta, alpha+beta growing with
bar width - more trades per bar averages the ratio toward its mean, exactly what a
sum-of-many-trades ratio should do) - buy/sell pressure is close to balanced in aggregate,
consistent with a two-sided perpetual futures market. `intrabar_close_pos` has a much
lower concentration (alpha+beta ~2.6-2.8, i.e. close to Uniform(0,1)'s alpha=beta=1) -
where a bar closes within its own high-low range is close to uninformative, mildly
U-shaped-avoiding (alpha,beta slightly >1 means mass pulls in from the 0/1 edges, not
toward them). The 1h/4h "fit failed" cells are a real data-boundary effect, not a code
bug: `distributions.py`'s beta fitter requires every observation strictly inside (0,1),
and at finer bars, exact 0 or 1 ratios (a bar with taker volume = 0 or = total volume,
or price never leaving its open==high==low corner) occur often enough in a 35k-row
history that a single such bar kills a whole-history fit-once. Noted as a data
granularity fact worth handling (winsorize or filter before fitting) if this Beta family
is ever fit rolling in a later phase - it isn't used again in this notebook.

### The intrabar range, distributionally

Normalized range `ln(high/low) / (full-sample close-to-close sigma)`, vs. the driftless
Brownian prediction `E[range/sigma] = 2*sqrt(2/pi) ~ 1.596`:

| interval | observed/predicted excess |
|---|---|
| 1h  | -13.5% |
| 4h  | -9.3% |
| 12h | -8.7% |
| 1d  | -6.1% |

Crypto's intrabar range is **systematically smaller** than a driftless Brownian path
predicts, at every interval, and the gap shrinks toward zero as bars widen. This runs
against the naive intuition ("crypto jumps a lot, so range should be *larger* than
Brownian") - the more likely explanation is intrabar mean-reversion / bid-ask-bounce
microstructure (consistent with the run-length finding below) suppressing the realized
high-low spread relative to what a pure random walk with the same close-to-close
variance would produce. This is exactly the kind of departure NEW_PROMPT flags as
"tells you in advance which range estimators will and won't work": Parkinson assumes
this Brownian relationship holds and will be **biased low** here by construction; the
drift-independent estimators (Rogers-Satchell, Yang-Zhang) don't share this specific
assumption and are worth watching for whether they correct it (Phase 3).

### Gap vs. intrabar decomposition

| interval | gap std | intrabar std | gap t-df | intrabar t-df |
|---|---|---|---|---|
| 1h  | 1.35e-5 | 0.00584 | 1.99 | 1.98 |
| 4h  | 1.89e-5 | 0.01157 | 1.99 | 2.05 |
| 12h | 2.37e-5 | 0.01880 (rs) / 0.02061 | 1.99 | 2.23 |
| 1d  | 1.27e-5 | 0.02882 | 1.99 | 2.89 |

Gap std is **400-2,000x smaller** than intrabar std at every interval - perpetual
futures trade continuously, so there is essentially no overnight/inter-bar gap here (as
expected), confirming the premise behind Yang-Zhang's gap term being small for this
instrument. The gap series' fitted t-df sits at ~1.99 at *every* interval, which is not
a real finding - the gap series is so close to a point mass at 0 that scipy's t.fit
optimizer converges near its lower search boundary regardless of interval; a
near-degenerate variance makes the shape parameter poorly identified. Intrabar std, by
contrast, tracks total std almost exactly (return is almost entirely intrabar move) and
its t-df reproduces the same aggregational-Gaussianity pattern as the total-return fit
above.

### Run lengths

| interval | mean run length | implied geometric p | KS vs geometric |
|---|---|---|---|
| 1h  | 1.88 | 0.532 | p ~ 0 |
| 4h  | 1.83 | 0.547 | p ~ 0 |
| 12h | 1.89 | 0.529 | p = 0.047 |
| 1d  | 1.90 | 0.526 | p ~ 0 |

Mean run length matches the geometric distribution's own mean by construction (p is
fit from it), but the *shape* is rejected at every interval (p<0.05, and effectively 0 at
1h/4h/1d) - actual sign-run lengths are not geometric, they carry more short-run
structure than a memoryless coin flip would produce. This is the distributional
expression of exactly the short-horizon mean-reversion effect notebook 3's cross-
sectional IC screen found (`mean_reversion_1`, the single strongest surviving feature
there) - not a new discovery, but a second, independent distributional confirmation of
the same effect using none of notebook 3's machinery.

### Stylized-fact summary table

| stylized fact | test | headline number | crypto does this? |
|---|---|---|---|
| Fat tails | normal vs t vs skew-t MLE + 5sigma frequency | t df~2-2.9; 5sigma freq 2,400-7,100x normal-implied | **yes, extreme** |
| Aggregational Gaussianity | t-df across intervals | 1.98 -> 2.88 (1h->1d) | **yes, but slow** - still far from normal at 1d |
| Volatility clustering | gamma vs exponential waiting times | gamma shape 0.52-0.85 (<1 = clustering) | **yes, at every interval** |
| Overdispersed activity | count Var/Mean | 114k-891k (Poisson predicts 1) | **yes, extreme** |
| Bounded taker-buy-ratio | Beta alpha+beta | 493-1,745, tightly centered on 0.5 | **balanced, low dispersion** |
| Bounded intrabar close position | Beta alpha+beta | 2.6-2.8 (near-Uniform) | **close to uninformative** |
| Intrabar range vs Brownian | normalized range excess | -6% to -14% (range **smaller** than Brownian) | **yes, systematic departure** |
| Gap vs intrabar tails | std ratio, t-df | gap std ~400-2000x smaller; gap df uninformative (near-degenerate) | **gap negligible on perps, as expected** |
| Run-length memorylessness | KS vs geometric | rejected at 3/4 intervals (p<0.05) | **no - excess short-run reversal** |

## Phase 2 - Machinery

`src/distributions.py` (families: normal/t/skewt/poisson/nbinom/beta via `fit_rolling`;
scoring: `log_score`, `crps`, `pit_values`/`pit_ks_test`, `qlike`, `kupiec_test`,
`christoffersen_independence_test`/`christoffersen_conditional_coverage_test`) and
`tests/test_distributions.py` (77 tests total in the repo, distributions.py's own suite
already passing) were built and committed in advance of this notebook (commit 37fdc8b)
and were not modified while building it - see that commit for their own writeup.

`src/research/tmp/dist_lib.py` is this notebook's own supporting library (feature
engineering from the full OHLCV bar, the Phase 3 forecasting-rung implementations, and
the from-scratch GARCH(1,1)/Gaussian-mixture/HMM fits used in Phase 3-4 - none of these
are in `distributions.py` because they're forecasting-contest machinery specific to this
notebook, not the general-purpose fitting/scoring primitives `distributions.py` provides).

**Bug found and fixed**: `dist_lib.fit_once` (the causal-to-date single fit used
throughout Phase 1) looked up fitted parameter columns by their bare name (e.g.
`"loc"`), but `distributions.fit_rolling` names its output columns
`f"{col}_{family}_{name}"` (e.g. `"log_return_normal_loc"`) - every Phase 1 call failed
with a `ColumnNotFoundError` before this was fixed. This is a bug in the notebook-4-local
`dist_lib.py`, not in the committed `distributions.py`/its tests, so no change to the
Phase 2 deliverable or its test suite was needed; fixed in place before any Phase 1
number was produced.

## Phase 3 - Volatility forecasting contest

BTC, all 4 intervals, full 7-rung ladder (0: trailing std 8/24/96; 1: EWMA lambda=0.94;
2: HAR-RV; 3: Parkinson/Garman-Klass/Rogers-Satchell/Yang-Zhang; 4: gamma/inverse-
gamma/lognormal fits on RV; 5: GARCH(1,1) normal/t/skew-t; 6: activity/dispersion-index
regression), every rung implemented and scored - no rung skipped.

**Refit cadence, declared up front and bounded by calendar time, not bar count**: cheap
rungs (HAR-RV, activity - a single `lstsq` call) refit weekly; the MLE rungs (RV
distribution fits, GARCH) refit monthly on a trailing window capped at 500 bars. This
keeps the number of expensive MLE fits *constant across intervals* (~45-50 refits per
interval regardless of whether a "month" is 720 hourly bars or 30 daily ones) rather than
scaling with bar count - a from-scratch skew-t GARCH MLE costs ~0.3-1s per fit on the
Raspberry Pi this ran on, and refitting a GARCH more often than monthly buys essentially
nothing (its own persistence parameter already responds to information over weeks, not
hours). Target: realized variance from higher-frequency (1h) sub-bars where the interval
is coarser than 1h (4h/12h/1d); 1h itself uses the bar's own squared return (no finer
cached series exists), noted as a noisier proxy at that one interval. Evaluated on the
full pre-holdout rolling out-of-sample sample (thousands of scored bars per interval, not
the frozen holdout - see "A note on the holdout" below).

### QLIKE by rung (BTC, best-in-group representative; lower is better)

| rung | 1h | 4h | 12h | 1d |
|---|---|---|---|---|
| 0 trailing std (best window) | 2.083 | 1.015 | 0.706 | 0.545 |
| 1 EWMA | 2.121 | 3.036 | 2.113 | 1.909 |
| 2 HAR-RV | **1.974** | **0.917** | **0.608** | **0.472** |
| 3 range (best estimator) | 1.965 | 0.944 | 0.757 | 0.542 |
| 4 RV-distribution fit (best family) | 2.217 | 1.088 | 0.740 | 0.592 |
| 5 GARCH (best innovation, normal) | 1.966 | 0.947 | 0.663 | 0.537 |
| 6 activity | 2.096 | 0.992 | 0.687 | 0.592 |

HAR-RV has the lowest QLIKE at **every single interval** - the "real benchmark in the
modern literature" NEW_PROMPT names it as beats everything else on raw QLIKE, every time.
Range estimators and GARCH-normal cluster close behind (within ~0.5-3% of HAR-RV's
QLIKE); rung 4 (fitting a distribution directly to RV) is reliably the *worst* rung
after EWMA - a genuinely useful negative finding (a parametric fit to realized variance
buys nothing over just averaging it, once you're already averaging it correctly via
HAR's multi-horizon components). EWMA is anomalously bad at 4h/12h/1d (QLIKE 1.9-3.0,
worse even than trailing std) - traced to the fixed lambda=0.94 (RiskMetrics' own choice,
calibrated for daily-or-finer data): applying the same per-bar decay to 4h/12h/1d bars
makes the filter adapt far too slowly to genuine variance jumps at those coarser
horizons. Not investigated further (out of scope to re-tune lambda per interval - the
ladder's job is to test the standard baseline as commonly used, not the best-tuned
version of it), but flagged here rather than silently reported.

### Which rungs actually *beat* which - all-pairs Diebold-Mariano

Adjacent-rung DM tests establish "beats the rung directly below," but the ladder's own
rule ("only a winner if it beats *every* rung below it, with a significance test") is not
transitive from adjacent comparisons alone. All 21 pairwise DM tests among the 7 rung
representatives were run at every interval; a rung counts as a winner only if its QLIKE
is significantly (p<0.05) lower than **every other** rung's.

| interval | best-by-QLIKE rung | beats every other rung significantly? |
|---|---|---|
| 1h  | rung3 (Garman-Klass) | **No** |
| 4h  | rung2 (HAR-RV) | **No** |
| 12h | rung2 (HAR-RV) | **No** |
| 1d  | rung2 (HAR-RV) | **No** |

**No rung wins the ladder outright at any interval on BTC.** HAR-RV, the range
estimators, and GARCH-normal are statistically indistinguishable from each other at
every interval (pairwise DM p-values between them mostly >0.05); what *is* significant,
consistently, is that all three of those beat EWMA and the RV-distribution-fit rung
(p<0.05 in most such pairs). So the honest Phase 3 finding is narrower than "X wins":
**a small cluster of methods (HAR-RV, range estimators, GARCH-normal) all sit at
roughly the same, better level than trailing-std/EWMA/RV-distribution-fits, and nothing
inside that cluster beats the others with significance.**

### Mincer-Zarnowitz and density scoring

MZ slope is closest to the ideal (1.0, 0 intercept) for HAR-RV at every interval (slope
1.02-1.05, small negative intercept) and for GARCH-normal (slope 0.60-0.93) - both
notably better-calibrated in the point-forecast-regression sense than the range
estimators (MZ slope 0.19-0.45 at every interval - **systematically biased low**, which
is exactly Phase 1's normalized-range finding predicted: crypto's intrabar range runs
6-14% below the Brownian prediction Parkinson/GK/RS/YZ are built on, so a range-based
variance forecast under-predicts and a regression of realized-on-forecast comes back
with slope well under 1). R² is low everywhere (0.004-0.19) - variance is inherently hard
to forecast precisely at these horizons even for the best-performing rungs, consistent
with NEW_PROMPT's own warning that vol-forecasting gains are real but modest.

**Density scoring is where a real, if narrow, distributional-modelling result shows up -
and this section was amended after a lookahead bug in the original scoring was found
and fixed; see "Correction" below before trusting the table.**

Comparing normal-density log scores across all rungs' variance forecasts is close
(within ~5% of each other at every interval, same near-tie as the point forecasts), but
scoring GARCH-t under its **own fitted Student-t innovation distribution** (rather than
forcing every rung through a normal density for comparability) changes the picture:

| interval | GARCH-t log score (own dist) | best other rung's log score (normal density) | Kupiec 5% VaR coverage p-value, GARCH-t |
|---|---|---|---|
| 1h  | **4.000** | 3.848 (GARCH-normal) | **0.0000** |
| 4h  | **3.254** | 3.139 (HAR-RV) | **0.0008** |
| 12h | **2.623** | 2.543 (HAR-RV) | 0.35 |
| 1d  | **2.220** | 2.191 (HAR-RV) | 0.68 |

GARCH-t's own-distribution log score is now the best of any rung/family combination at
**all 4** intervals (previously reported as 3 of 4 - see correction below), but its 5%
VaR exceedance rate is now **rejected by Kupiec at 1h and 4h** (p effectively 0 and
0.0008), and not rejected at 12h/1d (p=0.35, 0.68). This is a materially different, more
mixed calibration picture than originally reported. This still matches NEW_PROMPT's own
expectation from the crypto-GARCH literature ("heavy-tailed innovations win") on the log
score specifically: **the point-forecast (QLIKE) contest found no clear winner, but the
density contest shows Student-t innovations give a better *log-score* tail forecast than
assuming normal, at every interval.** The VaR-*coverage* half of the original claim does
not survive the correction below - GARCH-t's 5% VaR is well-calibrated at the two
coarser intervals but rejected at the two finer ones, not "never rejected anywhere." This
is not, on its own, enough to call GARCH-t the Phase 3 "winner" in the ladder's own
QLIKE-beats-everything sense - it is a narrower, calibration-specific finding, and now a
more qualified one than first reported.

### Correction (added after this notebook was first written)

The GARCH-t density score above was originally computed with a lookahead bug: the
Student-t degrees-of-freedom parameter used to *score* every bar was
`fits[-1]["params"][3]` - the value estimated on the **final** training window of the
whole sample - applied uniformly to every scored bar from the start of the evaluation
period onward. The variance forecast itself was properly causal and rolling; only the
shape parameter scoring it was not. This is the same class of bug as bug #3 below (a
lookahead leak), one level deeper: in the innovation distribution's shape parameter
rather than the point forecast itself.

Diagnosing the fitted nu path directly (`dist_lib.nu_path_from_fits`) shows it genuinely
varies across the 46 refits at 1h - from 2.2 (at the optimizer's own lower search bound,
during the more turbulent 2021-2022 stretch of the sample) up to 8.1 (later, calmer
refits) - not the constant value the bug effectively assumed. The original scoring used
only the *last* of these (7.87, one of the thinnest-tailed fits in the whole path) to
score bars throughout the entire sample, including the early, much-fatter-tailed
stretch where the true causal nu was closer to 2.2-3.4.

**Both halves of the original claim moved, in different directions, once this was
fixed and Phase 3 was fully re-run (all 4 intervals):**
- The **log score** win *strengthened*: GARCH-t now beats every normal-density rung at
  all 4 intervals (previously 3 of 4 - 1d's comparison was marginally the other way
  before the fix).
- The **VaR-coverage** claim *weakened materially*: Kupiec now rejects GARCH-t's 5% VaR
  at 1h and 4h (previously never rejected anywhere). The independence test
  (Christoffersen) was already rejected at 1h before the fix (clustering in violations,
  unaffected by this bug) and remains rejected after it.

**What this changes about notebook 4's bottom line**: the density-scoring result is
real but narrower than originally stated. "GARCH-t's own-distribution log score beats
every normal-density rung, at every interval" still stands and is, if anything,
strengthened. "Its 5% VaR coverage was never rejected by Kupiec" does **not** stand -
replace it with "its 5% VaR coverage is well-calibrated at 12h/1d but rejected at 1h/4h."
This does not change the notebook's overall conclusion (no rung wins the point-forecast
ladder outright; a narrower density/calibration result exists for GARCH-t) but it does
change the *strength* of the calibration half of that narrower result, and notebook 5's
own tail-risk work should not assume GARCH-t's VaR is well-calibrated everywhere.

### Frozen transfer check (ETH/SOL/DOGE/BNB/XRP, 1d only)

Scoped down to 1d (not all 4 intervals) for the transfer check - BTC already got the
full 4-interval, 21-pairwise-DM treatment; this checks whether the "no clear winner"
finding generalizes, at lighter cost.

| symbol | best-by-QLIKE rung | QLIKE | beats every other rung significantly? |
|---|---|---|---|
| BTC (1d, for reference) | HAR-RV | 0.472 | No |
| ETH | HAR-RV | 0.416 | **Yes** |
| SOL | HAR-RV | 0.341 | **Yes** |
| DOGE | GARCH-normal | 0.528 | No |
| BNB | HAR-RV | 0.483 | No |
| XRP | HAR-RV | 0.614 | No |

**Not stable.** HAR-RV is the best-by-QLIKE rung at 5 of 6 symbols (DOGE is the
exception), consistent with the QLIKE-ranking pattern found on BTC - but whether it
*significantly* beats every other rung flips symbol by symbol (clear winner on ETH/SOL,
not on BTC/BNB/XRP/DOGE). Per notebook 3's own standard ("stability outranks
magnitude"), this is reported as **no stable winner**, not as "HAR-RV wins 5/6."

## Phase 4 - Regime estimation

BTC, all 4 intervals: threshold baseline (2-state, trailing-median RV), Gaussian mixture
(K=2,3), HMM (Gaussian and Student-t emissions), activity regime (count-dispersion
threshold). Same monthly rolling-refit cadence and 500-bar cap as Phase 3's MLE rungs,
for the same cost reasons. **Filtered, never smoothed**: state probabilities come only
from `hmm_filter_step`'s one-step forward recursion applied to an already-fit (on past
data only) frozen model; the Baum-Welch backward pass that estimates the fit's own
parameters never touches bars after its own training window, and its smoothed gamma is
discarded, never used as a state estimate. **Rolling refit, never full-sample**: every
model refits monthly on a trailing, capped window. **Label switching**: `fit_gmm_em`/
`fit_hmm` impose ascending-fitted-variance ordering at every refit.

### Regime duration and vol/direction prediction (BTC)

| interval | model | mean state duration (bars) | geometric-null KS p | predicts vol (Kruskal p) | predicts direction (ANOVA p) |
|---|---|---|---|---|---|
| 1h | baseline threshold | 2.34 | ~0 | 4.7e-188 | 0.016 |
| 1h | HMM-Gaussian | **6.62** | ~0 | **0** | 0.030 |
| 4h | baseline threshold | 2.75 | ~0 | 9.6e-144 | 0.273 |
| 4h | HMM-Gaussian | **4.86** | 3.1e-63 | 1.3e-112 | 0.008 |
| 12h | baseline threshold | 2.41 | 1.9e-184 | 1.4e-20 | 0.558 |
| 12h | HMM-Gaussian | **4.58** | 1.5e-25 | 1.5e-24 | 0.057 |
| 1d | baseline threshold | 2.74 | 3.4e-62 | 1.4e-22 | 0.182 |
| 1d | HMM-Gaussian | **4.71** | 5.8e-12 | **6.2e-40** | 0.243 |

(Full table for GMM K=2/3, HMM-t, and activity regime across all 4 intervals is in
`phase4_results.json` - summarized here to the two most informative rows per interval.)

**No regime duration is geometric anywhere** (KS p effectively 0 at every model/interval
except the smallest-sample cases) - states are more persistent than a memoryless Markov
model's own core assumption predicts, a real (if expected) departure worth stating
plainly rather than treated as a modelling failure: a Markov chain is the null being
tested against, not the claim being made.

**Every single model, at every interval, predicts next-bar volatility with overwhelming
significance** (Kruskal-Wallis p-values from 1e-4 to effectively 0) - unsurprising (that
is almost definitionally what a vol-based regime is), but confirmed directly rather than
assumed. **Direction is a different story**: p-values scatter around and above 0.05 with
no consistent pattern (some marginal exceptions - 4h HMM-Gaussian p=0.008, 1h baseline
p=0.016 - expected by chance alone across the ~48 vol/direction tests run in Phase 4;
not treated as a real direction-prediction finding given no consistent sign or
replication across intervals/models). **"Regimes predict risk, not return"** - exactly
the clean result NEW_PROMPT called likely and asked to be written up as one, not
apologized for.

HMM-Gaussian shows the clearest improvement over the naive threshold baseline: 1.7-2.8x
longer mean state duration at every interval (states that actually persist, rather than
the threshold's near-random 2-3 bar flip-flopping around its own trailing median), and a
comparable-or-better vol-Kruskal statistic at 3 of 4 intervals (1h, 12h, 1d; roughly tied
at 4h). This is real, useful structure - but **no formal head-to-head significance test
between regime models was built** (unlike Phase 3's DM-test apparatus for the vol
ladder), so this is reported as suggestive rather than as a rigorously established
"winner" - see Phase 5 gating below for why that distinction matters here.

### Frozen transfer check (ETH/SOL/DOGE/BNB/XRP, 1d, baseline vs. HMM-Gaussian)

| symbol | baseline predicts-vol p | HMM predicts-vol p | baseline predicts-dir p | HMM predicts-dir p |
|---|---|---|---|---|
| ETH  | 2.3e-16 | 1.1e-36 | 0.333 | 0.118 |
| SOL  | 8.8e-46 | 7.2e-18 | 0.100 | 0.113 |
| DOGE | 1.2e-43 | 2.2e-37 | 0.005 | 0.331 |
| BNB  | 2.1e-40 | 3.0e-35 | 0.011 | 0.225 |
| XRP  | 1.6e-38 | 3.7e-40 | 0.011 | 0.061 |

"Predicts vol overwhelmingly, direction inconsistently and never both baseline+HMM
together" replicates at every one of the 5 transfer symbols - the single most stable
finding in this entire notebook. DOGE/BNB/XRP's baseline model shows a marginal
direction effect (p=0.005-0.011) that HMM does *not* reproduce (p=0.06-0.33 on the same
symbols) - if anything, evidence against a real, model-robust direction effect rather
than for one (a real effect should show up in the more expressive model too, not
disappear).

## Bugs found

Six real bugs surfaced while building this notebook, all in the notebook-4-local
`dist_lib.py`/`run_phase3.py`/`run_phase4.py` (none in the committed, previously-tested
`distributions.py` - no changes needed there):

1. **`fit_once` column lookup** (Phase 1/2) - looked up bare parameter names instead of
   `fit_rolling`'s actual `f"{col}_{family}_{name}"` column names. Every Phase 1 call
   failed outright before the fix.
2. **`diebold_mariano` mislabeled `research.newey_west_tstat`'s return values.**
   `newey_west_tstat` returns `(mean, tstat)`, not `(tstat, pvalue)` - `diebold_mariano`
   originally unpacked it as `tstat, pvalue = newey_west_tstat(...)`, silently taking the
   series *mean* as the reported "t-stat" and the real t-stat as the reported "p-value."
   Every DM test in an early run of Phase 3 reported nonsensical "p-values" outside
   [0, 1] (including negative ones) - caught by actually reading the numbers rather than
   trusting that the code ran without raising. Fixed by taking the real HAC t-stat and
   converting it to a two-sided p-value via the normal approximation.
3. **`make_har_features` had no lag - the single most serious bug found in this
   notebook.** The daily/weekly/monthly rolling-mean RV components were not shifted, so
   at the 1d interval (`bpd=1`), the "daily" component (`rolling_mean(window_size=1)`)
   was *literally identical* to that bar's own `rv_target` - the HAR-RV forecast was
   regressing the target on itself, a same-bar lookahead leak, not a forecast. Visible as
   HAR-RV's QLIKE coming back exactly 0.000000 at 1d in an early run - exactly the
   "any result that looks implausibly good" tripwire NEW_PROMPT's guardrails describe.
   Fixed by shifting all three HAR windows by 1 bar. Re-running after the fix moved
   HAR-RV from a suspicious perfect score to a normal, still-strong (but not
   ladder-winning) QLIKE - the corrected numbers above already reflect the fix.
4. **`qlike_mse`'s mask allowed `actual == 0`.** QLIKE's `log(ratio)` term is undefined
   there; BTC 1h has 13 bars with exactly zero realized variance (frozen-price bars -
   the same class of bug as notebook 3's `realized_vol_24 == 0`, which NEW_PROMPT
   explicitly predicted would recur). Those 13 bars poisoned the whole-series mean QLIKE
   to `+inf` for *every single rung* at 1h. Fixed with a strict `actual > 0` mask for
   QLIKE specifically (MSE, which has no such issue, keeps the wider `actual >= 0`
   mask and its own observation count, so it doesn't lose those 13 bars unnecessarily).
5. **`rolling_garch_forecast` stripped `params` from each fit record before it was
   needed downstream** - `run_phase3.py` needed the fitted degrees-of-freedom
   (`fits[-1]["params"][3]`) for GARCH-t's own-distribution density scoring, but the
   fit records it consumed had already dropped `params` on the way into the returned
   list. Fixed by keeping the full fit dict in the returned `fits` list (the JSON-output
   code strips `params` separately, later, only when serializing - so output size is
   unaffected).
6. **Phase 4's rolling-refit sufficiency check compared the wrong window size.**
   `rolling_refit_states` capped every training window at `max_train=500` bars but
   required `len(window) >= min_train // 2` before fitting - and `min_train` scales
   with bars-per-day (2160 at 1h), so `min_train // 2` (1080) could never be satisfied
   by a window that's capped at 500. Every GMM/HMM refit at 1h silently no-opped for
   the entire series - `gmm_k2`, `gmm_k3`, `hmm_gaussian`, and `hmm_t` all came back
   with zero fits and all-null hard states at 1h in an early run, while other intervals
   looked fine (their `min_train//2` happens to be under 500). Fixed by flooring the
   sufficiency check at `min(min_train, max_train) // 2` - the window that will actually
   be fit, not the warm-up gate.
7. **GARCH-t's density score used the wrong (future-only) degrees of freedom - found
   while preparing notebook 5, not while building this one.** `run_phase3.py` scored
   GARCH-t's own-distribution log score/VaR using `fits[-1]["params"][3]`, the
   degrees-of-freedom estimated on the *final* training window of the whole sample,
   applied to score every bar from the start of the evaluation period - the variance
   forecast was properly causal and rolling; the shape parameter scoring it was not.
   Fixed with `dist_lib.nu_path_from_fits`, a causal, forward-filled step-function path
   mirroring the variance forecast's own forward-fill exactly. See "Correction" above
   for how the corrected numbers moved (log-score win strengthened to 4/4 intervals;
   Kupiec VaR coverage, previously reported as never rejected, is now rejected at 1h/4h).

None of these were caught by a unit test (there is no test suite for the notebook-4-local
`dist_lib.py`/driver scripts, by design - they're forecasting-contest machinery specific
to this notebook, not the general-purpose `distributions.py` primitives that do have
one). All seven were caught the same way: by reading the actual output numbers rather
than trusting that a script which ran without raising had produced correct output - bug 3
(a lookahead leak) would have silently inflated HAR-RV into an undeserved Phase 3
"winner" at 1d if the suspiciously perfect QLIKE score hadn't been checked by eye before
being written up, and bug 7 (a subtler, one-level-deeper lookahead leak in a shape
parameter rather than a point forecast) was only caught by deliberately re-deriving the
causal parameter path and comparing it against what had been used to score.

## Phase 5 - Does any of it pay?

**Not run.** Phase 5 is pre-declared to run only if Phase 3 or Phase 4 produced an actual
winner. Neither did, held to a consistent standard:

- **Phase 3**: no rung beats every other rung with significance at any BTC interval, and
  the frozen transfer check shows the ranking isn't even stable across symbols (HAR-RV
  is a significant all-beating winner on ETH/SOL but not on BTC/BNB/XRP/DOGE). There is
  a real, narrower density-scoring result (GARCH-t's own-distribution calibration beats
  every other rung's normal-density score at 3 of 4 intervals and is never rejected by
  Kupiec) - but that is a calibration finding, not a point-forecast-ladder win, and
  Phase 5(a) as pre-declared ("vol-targeted buy-and-hold using the Phase 3 winner")
  requires a point *variance* forecast to scale exposure by, which is exactly the
  contest that produced no winner.
- **Phase 4**: HMM-Gaussian shows real, replicated structure (longer state persistence,
  comparable-or-better vol discrimination than the naive threshold at most intervals,
  and the vol-not-direction pattern replicates on every one of the 5 transfer symbols) -
  but no formal significance test was built to compare regime models head-to-head the
  way Phase 3's DM apparatus does for the vol ladder, so calling it a "winner" by the
  same rigor the rest of this notebook demands would be inconsistent. Phase 5(b)'s own
  gate ("chosen by Phase 4's own conditional-autocorrelation finding") also has nothing
  to point at: per-state return autocorrelation was computed (see `phase4_results.json`'s
  `conditional_stats`) but shows no consistent, replicated pattern of one state having
  reliably stronger reversal than another across intervals/symbols - there is no clean
  gate to pre-declare.

Running Phase 5 anyway on either "almost" result would mean backtesting a forecast this
notebook itself declined to certify as a winner - precisely the "no tuning until the
backtest looks good" failure mode NEW_PROMPT warns against. Skipping it here is the
correct application of this notebook's own rule, not a shortfall.

### A note on the holdout

Not spent by this notebook. `HOLDOUT_START = 2025-07-01` was already used once by
notebook 3 (its `cfg2_12h` holdout run) and would only be relevant here for Phase 5's
trading application, which didn't run. All Phase 1/3/4 numbers above use the full
pre-holdout rolling out-of-sample evaluation, which per NEW_PROMPT's own framing doesn't
depend on holdout purity the way a return-prediction backtest does (thousands of scored
bars per phase, not 3 folds).

## Bottom line

**Volatility**: no single rung of the mandatory 7-rung ladder wins the forecasting
contest outright on BTC at any interval, and the frozen transfer check shows the closest
thing to a leader (HAR-RV, lowest QLIKE at 5 of 6 symbols) doesn't clear significance
against every other rung consistently across symbols either. What *is* established:
HAR-RV, the four range estimators, and GARCH-normal cluster together as the best
available point forecasts (all beating EWMA and RV-distribution fits with significance,
none beating each other); a range-based forecast is measurably, predictably biased low
by exactly the amount Phase 1's normalized-range departure from the Brownian prediction
implies; and a real, narrower distributional win exists in density calibration -
GARCH-t's own Student-t innovation distribution gives a better log-score tail forecast
(best log score at all 4 intervals, after correcting a lookahead bug in the shape
parameter used to score it - see "Correction" in Phase 3 above) than any normal-density
alternative, confirming the crypto-GARCH literature's standard finding even though it
doesn't rescue the point-forecast contest - though its 5% VaR coverage, once corrected,
is only actually well-calibrated at 12h/1d and is rejected by Kupiec at 1h/4h, a more
qualified calibration story than first reported.

**Regime**: distributional regime models (especially HMM-Gaussian) find real, more
persistent structure than a naive trailing-median threshold, and every model at every
BTC interval - replicated on all 5 transfer symbols - shows the same clean pattern:
**regimes predict next-bar volatility with overwhelming significance and do not predict
direction.** No regime duration is geometric anywhere, meaning states persist more than
a Markov model's own core assumption implies. No regime model was rigorously certified
as beating the baseline with a significance test, so it isn't reported as a "winner" by
this notebook's own standard.

**Phase 5 did not run** - neither forecasting contest produced a certified winner to
backtest, which is a legitimate outcome per NEW_PROMPT's own framing ("the notebook
produces a real result whether or not any strategy is profitable"). Matches notebooks
1-3's overall pattern (no validated tradeable edge found in this research programme so
far) while adding genuinely new, non-null knowledge this time: crypto's tails, clustering,
and regime structure are real, extreme, and now measured with proper scoring rules,
even though none of it clears the bar this notebook set for calling something a winner.

## What to test next

- **A formal Phase 4 model-comparison test.** Phase 3 has a full DM-test apparatus;
  Phase 4 does not (documented above as a real gap, not a bug) - a proper likelihood-
  ratio or out-of-sample predictive-density comparison between HMM-Gaussian and the
  threshold baseline could turn the "suggestive" persistence/vol-discrimination edge
  into a certified Phase 4 winner, which would then make Phase 5(b) reachable.
- **Retune EWMA's lambda per interval.** The RiskMetrics lambda=0.94 was applied
  unchanged across 1h/4h/12h/1d and degraded badly at every interval coarser than 1h -
  worth testing whether a per-interval-calibrated lambda closes that gap and changes
  EWMA's ladder position.
- **GARCH-t/skew-t density scoring against the full ladder, not just GARCH-normal.**
  This notebook scored GARCH-t's own-distribution calibration separately but didn't
  extend the all-pairs DM machinery to density-score comparisons across the whole
  ladder - doing so properly could reveal whether GARCH-t is a genuine density-scoring
  winner (not just "better than normal-GARCH"), which the point-forecast ladder alone
  cannot show.
- **A range-estimator drift correction.** Phase 1 found crypto's normalized range runs
  6-14% below the Brownian prediction Parkinson's estimator assumes; Phase 3 confirmed
  this shows up as a systematically-low MZ slope for every range estimator. A simple
  multiplicative correction calibrated from Phase 1's own excess measurement is a cheap,
  well-motivated next step that this notebook diagnosed but didn't apply.
- **Extend the Phase 3/4 transfer check to all 4 intervals on all 5 symbols.** Both
  transfer checks here were scoped to 1d only for wall-clock reasons on this hardware; a
  faster machine (or a longer time-box) could confirm whether the 1d-only "not stable"
  (Phase 3) and "vol yes / direction no, replicated" (Phase 4) findings hold at 1h/4h/12h
  too.
