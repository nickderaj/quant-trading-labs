# 004 — Distributional Models for Volatility and Regime

## The question

Notebooks 001–003 all asked statistical tools to predict the **first moment** — the next return —
and all three found nothing. Rather than keep trying variations on direction prediction, this
notebook asks two questions those notebooks never asked:

1. Can distributional modelling forecast **volatility** better than the trivial baselines already
   in use?
2. Can it identify persistent **market regimes** that are informative about anything?

Both are judged as forecasting contests with proper scoring rules and significance tests, not as
backtests. A win here has to be statistically real, not just a good-looking Sharpe. A trading
application was pre-declared to run only if one of the two contests produced a certified winner.

Primary asset: **BTC**, at all four bar intervals (1h, 4h, 12h, 1d), with findings frozen and
transferred to ETH, SOL, DOGE, BNB and XRP where noted. Hourly bars are back in scope here —
notebook 003 dropped them because their *transaction cost drag* was unviable, and a forecasting
contest places no trades.

---

# Part 1 — What these series actually look like

All numbers in this part are fit once on the full pre-holdout history. This is explicitly the
"characterise the data" stage; everything from Part 2 onwards is rolling, refit and
out-of-sample.

## Fat tails

| Interval | n | Normal (μ, σ) | Student-t df | Skew-t (a, b) | Observed \|z\| ≥ 5σ | Normal-implied | Ratio |
|---|---|---|---|---|---|---|---|
| 1h | 35,064 | (3.2e−5, 0.00584) | **1.98** | (1.25, 1.27) | 0.00814% | 0.00000011% | **7,114×** |
| 4h | 8,766 | (1.3e−4, 0.01156) | **2.10** | (1.30, 1.34) | 0.00811% | 0.00000016% | **5,174×** |
| 12h | 2,922 | (3.9e−4, 0.02060) | **2.23** | (1.32, 1.32) | 0.00959% | 0.00000027% | **3,582×** |
| 1d | 1,461 | (7.9e−4, 0.02881) | **2.88** | (1.39, 1.33) | 0.01711% | 0.00000717% | **2,388×** |

A normal distribution underestimates the frequency of five-sigma days by **2,400× to 7,100×**.
That isn't "fat tails" — it is simply the wrong model. Fitted Student-t degrees of freedom sit
between 2 and 3 at every interval, and df = 2 is the boundary at which variance stops being finite
at all. These are among the heaviest-tailed liquid return series traded anywhere.

**Goodness of fit** (Kolmogorov-Smirnov statistic and p-value; a small p rejects the claim that
the fit is well-calibrated):

| Interval | Normal | Student-t | Skew-t |
|---|---|---|---|
| 1h | 0.108, p ≈ 0 | 0.0069, p = 0.069 | 0.0118, p = 0.0001 |
| 4h | 0.101, p ≈ 0 | 0.0164, p = 0.017 | 0.0223, p = 0.0003 |
| 12h | 0.094, p ≈ 0 | 0.0253, p = 0.047 | 0.0293, p = 0.013 |
| 1d | 0.078, p ≈ 0 | 0.0270, p = 0.231 | 0.0251, p = 0.313 |

The normal is rejected outright everywhere. Student-t is not rejected at daily bars and is
borderline elsewhere. **Skew-t is not an improvement on plain t** — it is worse or comparable at
every interval, and strictly rejected at 1h, 4h and 12h. BTC's estimated skew parameters are close
to symmetric, so the extra shape parameter buys calibration noise rather than fit. Plain
Student-t is the best simple parametric description found here.

A two-component Gaussian scale mixture separates a low-volatility component (weight ≈ 0.79,
variance ~10× smaller) from a high-volatility one (weight ≈ 0.21) at hourly bars, with the pattern
holding at every interval. The high-volatility weight rises to ≈ 0.44 at daily bars, because
compressing five years into ~1,461 daily observations puts more regime switching inside a smaller
sample. This is volatility clustering described as a static mixture; Part 3 makes it dynamic.

## Returns become more normal as you aggregate — slowly

Fitted degrees of freedom rise monotonically with interval: **1.98 → 2.10 → 2.23 → 2.88** from
hourly to daily. That is the textbook aggregational-Gaussianity pattern. But crypto starts from
an extreme baseline and is still nowhere near Gaussian at daily bars. Aggregating 24 hourly bars
into one daily bar buys less than a single additional degree of freedom's worth of normality.

## Volatility clustering, stated distributionally

Waiting times between k-sigma moves, fit with a gamma distribution. A shape parameter below 1
means waiting times are over-dispersed relative to a memoryless Poisson process — i.e. big moves
cluster:

| Interval | Shape (k = 2σ) | KS p (k = 2σ) | Shape (k = 3σ) | KS p (k = 3σ) |
|---|---|---|---|---|
| 1h | 0.63 | 6e−25 | 0.52 | 2e−7 |
| 4h | 0.74 | 7e−6 | 0.64 | 0.070 |
| 12h | 0.80 | 0.088 | 0.67 | 0.275 |
| 1d | 0.85 | 0.390 | 0.66 | 0.944 |

Shape is below 1 at every interval and every threshold. Big moves genuinely cluster. Gamma itself
is rejected at 1h and 4h — its functional form isn't quite right at high frequency — though it is
a far better description than the exponential, whose KS statistics run 2–3× larger throughout.
Clustering is strongest at hourly bars and weakens without disappearing at daily, the same
aggregational pattern as the tail index.

## Trade activity is wildly over-dispersed

Dispersion index of trade counts (variance ÷ mean; a Poisson process predicts exactly 1):

| Interval | Dispersion index | Negative binomial (n, p) |
|---|---|---|
| 1h | 114,393 | (1.33, 8.7e−6) |
| 4h | 300,973 | (2.01, 3.3e−6) |
| 12h | 617,751 | (2.94, 1.6e−6) |
| 1d | 891,044 | (4.08, 1.1e−6) |

Indices in the hundreds of thousands rather than near 1. Trade counts are not remotely
Poisson-shaped, and a negative binomial is a much better description. (A value near 1 here would
have indicated an aggregation bug, not a real finding.)

## Bounded quantities

Beta fits to the taker-buy ratio and to where a bar closes within its own high–low range:

| Interval | Taker-buy ratio (a, b) | a+b | Intrabar close position (a, b) | a+b |
|---|---|---|---|---|
| 1h | fit failed | — | fit failed | — |
| 4h | (245.8, 247.7) | 493 | fit failed | — |
| 12h | (570.5, 574.9) | 1,145 | (1.45, 1.34) | 2.79 |
| 1d | (869.3, 875.8) | 1,745 | (1.35, 1.25) | 2.60 |

The taker-buy ratio is tightly concentrated near 0.5, with concentration growing as bars widen —
more trades per bar averages the ratio toward its mean, exactly as a sum-of-many-trades ratio
should. Buying and selling pressure is close to balanced in aggregate, consistent with a two-sided
perpetual futures market.

Where a bar closes within its range is much less concentrated (a+b ≈ 2.6–2.8, against a uniform
distribution's a = b = 1) — close to uninformative, with a slight pull away from the extremes.

The failed fits at fine intervals are a data-boundary effect, not a code fault: the fitter requires
every observation strictly inside (0, 1), and at hourly bars exact 0 or 1 values occur often
enough in a 35,000-row history that a single such bar kills a whole-history fit. Worth
winsorising if this family is ever fit rolling; it isn't used again here.

## The intrabar range is smaller than Brownian motion predicts

Normalised range — log(high/low) divided by the full-sample close-to-close volatility — compared
against the driftless Brownian prediction of 2√(2/π) ≈ 1.596:

| Interval | Observed vs predicted |
|---|---|
| 1h | −13.5% |
| 4h | −9.3% |
| 12h | −8.7% |
| 1d | −6.1% |

Crypto's intrabar range is **systematically smaller** than a driftless Brownian path predicts, at
every interval, with the gap shrinking as bars widen. This runs against the naive intuition that
crypto jumps a lot so its range should be larger. The likely explanation is intrabar mean
reversion and bid-ask bounce suppressing the realised high–low spread relative to a pure random
walk with the same close-to-close variance.

This has a direct, testable consequence: the Parkinson range estimator assumes exactly this
Brownian relationship, so it will be **biased low here by construction**. The drift-independent
estimators (Rogers-Satchell, Yang-Zhang) don't share that specific assumption. Part 2 checks
whether the difference shows up.

## Gaps versus intrabar movement

| Interval | Gap std | Intrabar std | Gap t-df | Intrabar t-df |
|---|---|---|---|---|
| 1h | 1.35e−5 | 0.00584 | 1.99 | 1.98 |
| 4h | 1.89e−5 | 0.01157 | 1.99 | 2.05 |
| 12h | 2.37e−5 | 0.02061 | 1.99 | 2.23 |
| 1d | 1.27e−5 | 0.02882 | 1.99 | 2.89 |

Gap volatility is **400–2,000× smaller** than intrabar volatility. Perpetual futures trade
continuously, so there is essentially no inter-bar gap — which confirms the premise behind
Yang-Zhang's gap term being negligible for this instrument.

The gap series' fitted degrees of freedom sitting at ≈ 1.99 at *every* interval is not a finding:
the series is so close to a point mass at zero that the optimiser converges near its lower search
boundary regardless. A near-degenerate variance leaves the shape parameter unidentified.

## Sign runs are not memoryless

| Interval | Mean run length | Implied geometric p | KS vs geometric |
|---|---|---|---|
| 1h | 1.88 | 0.532 | p ≈ 0 |
| 4h | 1.83 | 0.547 | p ≈ 0 |
| 12h | 1.89 | 0.529 | p = 0.047 |
| 1d | 1.90 | 0.526 | p ≈ 0 |

The mean matches the geometric distribution by construction, since p is fitted from it — but the
*shape* is rejected at every interval. Actual sign-run lengths carry more short-run structure than
a memoryless coin flip produces.

This is the distributional expression of exactly the short-horizon mean-reversion effect notebook
003's cross-sectional screen identified as its single strongest surviving signal. Not a new
discovery, but an independent confirmation using none of that notebook's machinery.

## Summary

| Stylised fact | Test | Headline | Does crypto do this? |
|---|---|---|---|
| Fat tails | Normal / t / skew-t fits + 5σ frequency | t df ≈ 2–2.9; 5σ moves 2,400–7,100× the normal-implied rate | **Yes, extreme** |
| Aggregational Gaussianity | t-df across intervals | 1.98 → 2.88 | **Yes, but slow** — still far from normal at daily |
| Volatility clustering | Gamma vs exponential waiting times | Gamma shape 0.52–0.85 | **Yes, at every interval** |
| Over-dispersed activity | Count variance ÷ mean | 114k–891k (Poisson predicts 1) | **Yes, extreme** |
| Bounded taker-buy ratio | Beta concentration | 493–1,745, centred on 0.5 | **Balanced, low dispersion** |
| Bounded close position | Beta concentration | 2.6–2.8 (near-uniform) | **Close to uninformative** |
| Intrabar range vs Brownian | Normalised range excess | −6% to −14% (range *smaller*) | **Yes, systematic departure** |
| Gap vs intrabar tails | Std ratio | Gap 400–2,000× smaller | **Gap negligible, as expected** |
| Run-length memorylessness | KS vs geometric | Rejected at 3 of 4 intervals | **No — excess short-run reversal** |

---

# Part 2 — The volatility forecasting contest

Seven methods, evaluated against each other at all four intervals. None was skipped:

0. **Trailing standard deviation** (8, 24 and 96-bar windows)
1. **Exponentially weighted moving average** (λ = 0.94)
2. **Heterogeneous autoregressive realised volatility** (HAR-RV) — a regression on daily, weekly
   and monthly averages of past realised variance
3. **Range estimators** — Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang
4. **Parametric fits to realised variance** — gamma, inverse gamma, lognormal
5. **GARCH(1,1)** with normal, Student-t and skew-t innovations
6. **A trade-activity regression** on the count dispersion index

**Refit cadence** was declared in advance and bounded by calendar time rather than bar count. The
cheap methods refit weekly; the maximum-likelihood methods (variance distribution fits, GARCH)
refit monthly on a trailing window capped at 500 bars. This keeps the number of expensive fits
roughly constant across intervals (~45–50 per interval) rather than scaling with bar count. The
hardware here is a Raspberry Pi, where a from-scratch skew-t GARCH fit costs 0.3–1 second, and
refitting a GARCH more often than monthly buys essentially nothing — its persistence parameter
already responds to information over weeks, not hours.

**The target** is realised variance built from hourly sub-bars wherever the interval is coarser
than an hour. At hourly bars themselves the target is the bar's own squared return, which is a
noisier proxy — noted at that one interval.

## Point-forecast accuracy (QLIKE — lower is better)

| Method | 1h | 4h | 12h | 1d |
|---|---|---|---|---|
| Trailing std (best window) | 2.083 | 1.015 | 0.706 | 0.545 |
| EWMA | 2.121 | 3.036 | 2.113 | 1.909 |
| HAR-RV | **1.974** | **0.917** | **0.608** | **0.472** |
| Range (best estimator) | 1.965 | 0.944 | 0.757 | 0.542 |
| Variance-distribution fit | 2.217 | 1.088 | 0.740 | 0.592 |
| GARCH (normal innovations) | 1.966 | 0.947 | 0.663 | 0.537 |
| Trade activity | 2.096 | 0.992 | 0.687 | 0.592 |

HAR-RV has the lowest QLIKE at **every interval** — the standard modern benchmark beats everything
else on raw score every time. Range estimators and normal-innovation GARCH cluster close behind.

Two negative findings are worth stating. Fitting a distribution directly to realised variance is
reliably the worst method after EWMA: a parametric fit to realised variance buys nothing over
simply averaging it, once you are already averaging it correctly across multiple horizons the way
HAR does. And **EWMA is anomalously bad at every interval coarser than hourly** (QLIKE 1.9–3.0,
worse even than a trailing standard deviation). This traces to the fixed λ = 0.94, which is
calibrated for daily-or-finer data; applying the same per-bar decay to 4h, 12h and daily bars
makes the filter adapt far too slowly to genuine variance jumps. Re-tuning λ per interval was out
of scope — the contest tests the standard baseline as commonly used — but the effect is flagged
rather than silently reported.

## Which methods actually *beat* which

A method only counts as a winner if its QLIKE is significantly lower (p < 0.05) than **every
other** method's. That is not transitive from adjacent comparisons alone, so all 21 pairwise
Diebold-Mariano tests were run at every interval.

| Interval | Lowest QLIKE | Beats every other method significantly? |
|---|---|---|
| 1h | Range (Garman-Klass) | **No** |
| 4h | HAR-RV | **No** |
| 12h | HAR-RV | **No** |
| 1d | HAR-RV | **No** |

**No method wins the contest outright at any interval on BTC.** HAR-RV, the range estimators and
normal-innovation GARCH are statistically indistinguishable from one another everywhere. What *is*
consistently significant is that all three of those beat EWMA and the variance-distribution fits.

The honest finding is therefore narrower than "X wins": **a small cluster of methods sits at
roughly the same, better level than the rest, and nothing inside that cluster beats the others
with significance.**

## Forecast bias

Regressing realised on forecast variance (the ideal is slope 1, intercept 0) puts HAR-RV closest
at every interval (slope 1.02–1.05 with a small negative intercept) and normal-GARCH next (slope
0.60–0.93). The range estimators come back with **slope 0.19–0.45 everywhere — systematically
biased low.**

That is exactly what Part 1 predicted. Crypto's intrabar range runs 6–14% below the Brownian
prediction the range estimators are built on, so a range-based variance forecast under-predicts,
and a regression of realised on forecast returns a slope well under 1. A descriptive finding from
Part 1 correctly anticipated a forecasting result in Part 2.

R² is low everywhere (0.004–0.19). Variance is inherently hard to forecast precisely at these
horizons even for the best methods.

## Density scoring — where a real result appears

Comparing all methods through a common normal density puts them within ~5% of each other, the same
near-tie as the point forecasts. But scoring GARCH-t under **its own fitted Student-t innovation
distribution**, rather than forcing it through a normal density for comparability, changes the
picture:

| Interval | GARCH-t log score (own distribution) | Best alternative (normal density) | 5% VaR coverage test, p-value |
|---|---|---|---|
| 1h | **4.000** | 3.848 (GARCH-normal) | **0.0000** |
| 4h | **3.254** | 3.139 (HAR-RV) | **0.0008** |
| 12h | **2.623** | 2.543 (HAR-RV) | 0.35 |
| 1d | **2.220** | 2.191 (HAR-RV) | 0.68 |

GARCH-t's own-distribution log score is the best of any method-and-family combination at **all four
intervals**. Its 5% Value-at-Risk exceedance rate, however, is **rejected at 1h and 4h** and not
rejected at 12h and 1d.

So: the point-forecast contest found no clear winner, but the density contest shows Student-t
innovations give a better log-score tail forecast than assuming normality, everywhere. The
calibration half of that is mixed — well calibrated at the two coarser intervals, rejected at the
two finer ones. This is not enough to call GARCH-t the contest winner in the
beats-everything-on-QLIKE sense; it is a narrower, calibration-specific finding.

### A lookahead bug in the density scoring, found and fixed

The density scores above were originally computed with a lookahead leak. The Student-t degrees-of-
freedom parameter used to *score* every bar was the value estimated on the **final** training
window of the whole sample, applied uniformly to every scored bar from the start of the evaluation
period. The variance forecast itself was properly causal and rolling; only the shape parameter
scoring it was not.

Reconstructing the fitted degrees-of-freedom path directly shows it varies substantially across
the 46 refits at hourly bars — from 2.2 (at the optimiser's lower bound, during the turbulent
2021–2022 stretch) up to 8.1 in later, calmer refits. The original scoring used only the last of
these (7.87, one of the thinnest-tailed fits in the entire path) to score bars throughout the
sample, including the early, much fatter-tailed stretch where the true causal value was closer to
2.2–3.4.

**Both halves of the original claim moved once this was fixed, in opposite directions:**

- The **log-score win strengthened** — GARCH-t now beats every normal-density alternative at all
  four intervals, where before the correction daily was marginally the other way.
- The **coverage claim weakened materially** — the 5% VaR is now rejected at 1h and 4h, where
  before the correction it had never been rejected anywhere. (The independence test on violation
  clustering was already rejected at hourly bars before the fix, and remains so.)

The corrected numbers are what appears in the table above. Anything downstream — notebook 005's
tail-risk work in particular — must not assume GARCH-t's Value-at-Risk is well calibrated
everywhere.

## Does the "no clear winner" finding generalise?

Checked at daily bars only on the five transfer symbols, since BTC already received the full
four-interval, 21-comparison treatment:

| Symbol | Lowest QLIKE | QLIKE | Beats every other method significantly? |
|---|---|---|---|
| BTC (reference) | HAR-RV | 0.472 | No |
| ETH | HAR-RV | 0.416 | **Yes** |
| SOL | HAR-RV | 0.341 | **Yes** |
| DOGE | GARCH-normal | 0.528 | No |
| BNB | HAR-RV | 0.483 | No |
| XRP | HAR-RV | 0.614 | No |

**Not stable.** HAR-RV has the lowest QLIKE at 5 of 6 symbols, consistent with the BTC ranking —
but whether it *significantly* beats everything else flips symbol by symbol. By the standard
notebook 003 set (stability outranks magnitude) this is reported as **no stable winner**, not as
"HAR-RV wins 5 of 6".

---

# Part 3 — Regime models

Five model families at all four intervals: a threshold baseline (two states split on trailing
median realised variance), Gaussian mixtures with two and three components, hidden Markov models
with Gaussian and Student-t emissions, and an activity regime based on the count dispersion index.
Same monthly refit cadence and 500-bar cap as the maximum-likelihood methods above.

Three methodological commitments, all of which matter for whether these results are honest:

- **Filtered, never smoothed.** State probabilities come only from a one-step forward recursion
  applied to a model already fitted on past data. The backward pass that estimates the fit's own
  parameters never touches bars after its training window, and its smoothed state estimates are
  discarded rather than used.
- **Rolling refit, never full-sample.** Every model refits monthly on a trailing, capped window.
- **Label switching handled.** Fitted states are ordered by ascending variance at every refit, so
  "state 0" means the same thing across refits.

## Persistence, and what regimes predict

| Interval | Model | Mean duration (bars) | KS vs geometric | Predicts volatility (p) | Predicts direction (p) |
|---|---|---|---|---|---|
| 1h | Threshold baseline | 2.34 | ≈ 0 | 4.7e−188 | 0.016 |
| 1h | HMM-Gaussian | **6.62** | ≈ 0 | **≈ 0** | 0.030 |
| 4h | Threshold baseline | 2.75 | ≈ 0 | 9.6e−144 | 0.273 |
| 4h | HMM-Gaussian | **4.86** | 3.1e−63 | 1.3e−112 | 0.008 |
| 12h | Threshold baseline | 2.41 | 1.9e−184 | 1.4e−20 | 0.558 |
| 12h | HMM-Gaussian | **4.58** | 1.5e−25 | 1.5e−24 | 0.057 |
| 1d | Threshold baseline | 2.74 | 3.4e−62 | 1.4e−22 | 0.182 |
| 1d | HMM-Gaussian | **4.71** | 5.8e−12 | **6.2e−40** | 0.243 |

(The two most informative rows per interval; the mixtures, Student-t HMM and activity regime were
all run too.)

**No regime duration is geometric anywhere.** States persist more than a Markov model's own core
assumption predicts. That is a real departure worth stating plainly rather than treating as a
modelling failure — the Markov chain is the null being tested against, not the claim being made.

**Every model, at every interval, predicts next-bar volatility with overwhelming significance.**
Unsurprising, since that is almost definitionally what a volatility-based regime is, but confirmed
directly rather than assumed.

**Direction is a different story.** The p-values scatter around and above 0.05 with no consistent
pattern. A couple are marginal (0.008 at 4h, 0.016 at 1h) — about what chance produces across the
~48 tests run here, and not treated as a real finding given no consistent sign and no replication
across intervals or models.

**Regimes predict risk, not return.** That is the clean result.

HMM-Gaussian shows the clearest improvement over the naive threshold: **1.7–2.8× longer mean state
duration** at every interval — states that actually persist, rather than flip-flopping every two
or three bars around a trailing median — and comparable or better volatility discrimination at
three of four intervals.

That is real, useful structure. But **no formal head-to-head significance test between regime
models was built**, unlike the pairwise testing apparatus used for the volatility contest. So it
is reported as suggestive rather than established, which matters for the gating decision below.

## Does it replicate?

Daily bars, threshold baseline versus HMM-Gaussian, on the five transfer symbols:

| Symbol | Baseline, volatility p | HMM, volatility p | Baseline, direction p | HMM, direction p |
|---|---|---|---|---|
| ETH | 2.3e−16 | 1.1e−36 | 0.333 | 0.118 |
| SOL | 8.8e−46 | 7.2e−18 | 0.100 | 0.113 |
| DOGE | 1.2e−43 | 2.2e−37 | 0.005 | 0.331 |
| BNB | 2.1e−40 | 3.0e−35 | 0.011 | 0.225 |
| XRP | 1.6e−38 | 3.7e−40 | 0.011 | 0.061 |

"Predicts volatility overwhelmingly, direction inconsistently, and never both models together"
replicates at all five symbols — the single most stable finding in this notebook.

Note what happens on DOGE, BNB and XRP: the *baseline* shows a marginal direction effect
(p = 0.005–0.011) that the HMM does not reproduce (p = 0.06–0.33 on the same symbols). If
anything that is evidence *against* a real direction effect. A genuine one should appear in the
more expressive model too, not vanish.

---

# Bugs found

Seven real bugs surfaced while building this notebook. None were caught by a unit test — all seven
were caught by reading the actual output numbers rather than trusting that a script which ran
without raising had produced correct output.

1. **Parameter lookup by the wrong column name** in the descriptive-phase fitting helper. Every
   call failed outright before the fix.

2. **The Diebold-Mariano test mislabelled its own inputs.** The underlying HAC routine returns
   (mean, t-statistic), but the test unpacked it as (t-statistic, p-value) — silently reporting the
   series *mean* as the t-statistic and the real t-statistic as the p-value. Every early
   comparison reported nonsensical p-values outside [0, 1], including negative ones. Caught by
   reading the numbers, not by the code raising.

3. **The HAR features had no lag — the most serious bug found here.** The daily, weekly and monthly
   rolling averages of realised variance were never shifted, so at daily bars the "daily"
   component was *literally identical* to that bar's own target. HAR-RV was regressing the target
   on itself. This showed up as a QLIKE of exactly 0.000000 at daily bars — a result implausibly
   good enough to trip a tripwire. Fixed by shifting all three windows by one bar; re-running moved
   HAR-RV from a suspicious perfect score to a normal, still-strong but not contest-winning score.
   **All numbers reported above reflect the fix.**

4. **The QLIKE mask allowed a realised variance of exactly zero,** where its log-ratio term is
   undefined. BTC hourly bars include 13 with exactly zero realised variance (frozen-price bars —
   the same class of problem as notebook 003's zero-volatility divisor). Those 13 bars poisoned the
   mean QLIKE to infinity for *every method* at hourly bars. Fixed with a strictly-positive mask
   for QLIKE specifically.

5. **Fitted GARCH parameters were stripped before they were needed downstream,** so the
   degrees-of-freedom values required for density scoring weren't available. Fixed by retaining
   the full fit record and stripping only at serialisation time.

6. **The regime refit sufficiency check compared the wrong window size.** Training windows were
   capped at 500 bars, but the check required a window of at least half the *minimum* training
   size — and that minimum scales with bars per day (2,160 at hourly), so the requirement (1,080)
   could never be satisfied by a window capped at 500. Every mixture and HMM refit at hourly bars
   silently did nothing for the entire series, coming back with zero fits and no states, while
   other intervals looked fine. Fixed by flooring the check at the window that will actually be
   fit rather than the warm-up gate.

7. **The GARCH-t density score used a future-only degrees-of-freedom value** — described in full in
   the correction above. Found while preparing notebook 005, not while building this one, and only
   caught by deliberately re-deriving the causal parameter path and comparing it against what had
   been used.

Bug 3 in particular would have silently promoted HAR-RV to an undeserved contest win at daily
bars, had the suspiciously perfect score not been checked by eye before being written up.

---

# The trading application did not run

It was pre-declared to run only if the volatility contest or the regime contest produced a
certified winner. Neither did, held to a consistent standard:

- **Volatility:** no method beats every other method with significance at any BTC interval, and
  the transfer check shows the ranking isn't even stable across symbols. There is a real but
  narrower density-scoring result for GARCH-t — but that is a calibration finding, and the
  pre-declared application (volatility-targeted exposure using the contest winner) requires a
  point *variance* forecast to scale by, which is exactly the contest that produced no winner.
  The correction above also weakens the calibration half of that result at the two finer
  intervals.
- **Regime:** HMM-Gaussian shows real, replicated structure, but no formal head-to-head test
  between regime models was built, so calling it a winner by the same rigour the rest of this
  notebook demands would be inconsistent. The alternative gate — a conditional-autocorrelation
  finding, one state having reliably stronger reversal than another — was computed and shows no
  consistent, replicated pattern across intervals or symbols. There is nothing clean to gate on.

Running it anyway on either "almost" result would mean backtesting a forecast this notebook itself
declined to certify. Skipping it is the correct application of the notebook's own rule, not a
shortfall.

**The holdout was not spent.** It would only have been relevant to the trading application, which
didn't run. Everything above uses the full pre-holdout rolling out-of-sample evaluation —
thousands of scored bars per contest, not a handful of folds.

---

# Bottom line

**Volatility.** No method wins the forecasting contest outright on BTC at any interval, and the
transfer check shows the closest thing to a leader (HAR-RV, lowest QLIKE at 5 of 6 symbols)
doesn't clear significance against everything else consistently across symbols either. What *is*
established: HAR-RV, the four range estimators and normal-innovation GARCH cluster together as the
best available point forecasts, all beating EWMA and variance-distribution fits with significance
and none beating each other; a range-based forecast is measurably and predictably biased low, by
exactly the amount the descriptive range departure from Brownian implies; and a real, narrower
distributional win exists in density calibration, where GARCH-t's own Student-t innovation
distribution gives the best log-score tail forecast at all four intervals — though its 5% VaR
coverage, once the lookahead bug in its shape parameter was corrected, is only well calibrated at
12h and 1d.

**Regime.** Distributional regime models — HMM-Gaussian especially — find real structure that is
substantially more persistent than a naive trailing-median threshold. Every model at every BTC
interval, replicated on all five transfer symbols, shows the same clean pattern: **regimes predict
next-bar volatility with overwhelming significance and do not predict direction.** No regime
duration is geometric anywhere, meaning states persist more than a Markov model's own core
assumption implies. No regime model was certified as beating the baseline with a significance
test, so none is reported as a winner.

This matches notebooks 001–003 in finding no validated tradeable edge, while adding genuinely new,
non-null knowledge: crypto's tails, clustering and regime structure are real, extreme, and now
measured with proper scoring rules — even though none of it clears the bar set here for calling
something a winner.

# What to test next

- **A formal comparison test between regime models.** The volatility contest has a complete
  pairwise-testing apparatus; the regime work does not. A proper likelihood-ratio or
  out-of-sample predictive-density comparison could turn the suggestive persistence advantage into
  a certified result, which would in turn make the trading application reachable.
- **Re-tune EWMA's decay per interval.** The standard λ = 0.94 was applied unchanged everywhere
  and degraded badly at every interval coarser than hourly.
- **Extend density scoring across the whole ladder.** GARCH-t's own-distribution calibration was
  scored separately, but the pairwise-testing machinery was never extended to density comparisons
  across all methods. Doing so would reveal whether GARCH-t is a genuine density winner rather
  than just better than normal-innovation GARCH.
- **Apply a range-estimator drift correction.** The descriptive work found crypto's normalised
  range runs 6–14% below the Brownian prediction, and the forecasting work confirmed it shows up
  as a systematically low regression slope. A multiplicative correction calibrated from the
  measured excess is a cheap, well-motivated step this notebook diagnosed but didn't apply.
- **Extend both transfer checks to all four intervals.** Both were scoped to daily bars for
  wall-clock reasons on this hardware.

*Notebook: `src/research/004_distributional_models.ipynb`.*
