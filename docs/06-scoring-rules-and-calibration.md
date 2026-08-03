# 06 — Scoring rules and calibration

This file assumes [01](01-probability-and-distributions.md)-[04](04-volatility-models.md).
It covers how a forecast is graded — and specifically, how you grade a forecast of a
*whole distribution*, not just a single number, which is the shift notebook 5 makes from
notebook 4's point-forecast-only ladder.

---

### Forecast evaluation

**In one sentence.** The general practice of comparing what a model predicted against
what actually happened, using a well-defined numerical grading rule — as opposed to
eyeballing whether a forecast "looks reasonable."

**The maths.** No single formula; the family of specific rules below (log score, CRPS,
QLIKE) are all instances of it, each appropriate to a different kind of forecast (point
vs. density) and a different notion of "actual."

**Why it is here.** This entire research programme is structured as a sequence of
forecast-evaluation contests (notebook 3's cross-sectional IC, notebook 4's QLIKE ladder,
notebook 5's log-score contest) rather than backtests, specifically because a proper
forecast-evaluation contest can be run and trusted on far more scored observations than a
trading backtest can (thousands of scored bars vs. a handful of holdout periods) — see
[why forecast evaluation before backtesting](08-research-methodology.md#walk-forward-analysis).

**Worked example.** Notebook 4's Phase 3 scores every rung's variance forecast against
realized variance at every single bar in the pre-holdout sample (tens of thousands of
observations at 1h) — a far more statistically powerful comparison than any trading
strategy's handful of holdout trades could support.

**Pitfalls.** Which specific evaluation rule you pick matters enormously and can change
which model "wins" — this is exactly notebook 5's central methodological pivot (log
score/CRPS instead of QLIKE), not a cosmetic change.

---

### Point vs. density forecast

**In one sentence.** A point forecast is a single number (a best guess); a density
forecast is a *whole predictive distribution* (every possible outcome, with its own
probability) — the second is strictly more informative, but needs different scoring
machinery to grade properly.

**The maths.** Point forecast: $\hat{y}_{t+1}$, a single number. Density forecast:
$\hat{F}_{t+1}(\cdot)$, an entire distribution (CDF or density) for what $y_{t+1}$ might
be.

**Why it is here.** This distinction is the entire reason notebook 5 exists as a separate
notebook from notebook 4: Phase 3 of notebook 4 evaluated point forecasts (a single
variance number) via QLIKE; notebook 5 evaluates *density* forecasts (a whole predictive
distribution over returns) via log score and CRPS, a genuinely different and richer
question.

**Worked example.** Two models can produce the identical point forecast (same predicted
variance) while implying very different density forecasts (one via a normal, one via a
Student-t with the same variance but much fatter tails) — QLIKE, evaluated only on the
variance number, cannot distinguish them; log score, evaluated on the whole density, can
and does (this is exactly how GARCH-t beat GARCH-normal in notebook 4's density-scoring
side-note, despite an identical variance recursion).

**Pitfalls.** A model can be a poor point forecaster while still being a good density
forecaster in a narrower sense (correctly capturing shape/tail behavior even if its
central tendency isn't the sharpest) — conflating "wins the point-forecast contest" with
"wins the density contest" is exactly the confusion notebook 5's reframing exists to
prevent.

---

### Loss function

**In one sentence.** A rule that assigns a number to "how bad was this forecast," given
what actually happened — lower is better by convention throughout this repo, and a
forecast is judged by its *average* loss over many observations, not any single one.

**The maths.** $L(\hat{y}, y)$ — a function of the forecast $\hat{y}$ and the realized
outcome $y$, with $L \ge 0$ and (usually) $L=0$ only when $\hat{y}=y$ exactly.

**Why it is here.** QLIKE and squared error (MSE) are both loss functions for point
forecasts; negative log score and CRPS are loss functions for density forecasts (see
[proper scoring rule](#proper-scoring-rule-and-why-properness-matters) for the specific property that separates a
*good* loss function from an arbitrary one).

**Worked example.** `distributions.qlike` computes exactly `ratio - log(ratio) - 1`
where `ratio = actual/predicted` — a specific loss function, always non-negative,
minimized at exactly `ratio=1` (perfect forecast).

**Pitfalls.** Not every loss function that "seems reasonable" actually rewards honest,
accurate forecasting when averaged over many observations — see
[proper scoring rule](#proper-scoring-rule-and-why-properness-matters) for the formal property that guarantees this,
and why it's checked deliberately rather than assumed.

---

### Proper scoring rule (and why properness matters)

**In one sentence.** A scoring rule is "proper" if a forecaster who genuinely believes a
particular distribution is the truth can't do any better, on average, by reporting a
*different* distribution instead — properness is what guarantees honest forecasting is
also the *optimal* strategy under that rule.

**The maths.** A scoring rule $S$ is proper if, for the true distribution $F$,
$\mathbb{E}_{Y \sim F}[S(F, Y)] \ge \mathbb{E}_{Y \sim F}[S(G, Y)]$ for every other
candidate distribution $G$ — i.e. reporting the truth $F$ itself achieves the best
possible expected score, better than reporting anything else, even if you don't know
exactly what $Y$ will turn out to be.

**Why it is here.** [Log score](#log-score) and [CRPS](#crps) are both proper scoring
rules — this is *why* they're the right tools for grading a density forecast: a model
that genuinely reports its best honest belief about the distribution can't improve its
expected score by strategically misreporting a different distribution instead.

**Worked example.** QLIKE is a proper scoring rule specifically for a noisy *variance
proxy* target (Patton 2011) — this specific property (not just "seems like a reasonable
loss function") is the actual, technical reason `NEXT_RUN_PROMPT.md` and notebook 4 both
prefer it over plain MSE for scoring variance forecasts.

**Pitfalls.** Not every loss function that seems intuitively reasonable is proper — using
an improper scoring rule can reward a forecaster for strategically hedging or distorting
their honest forecast, which defeats the entire purpose of a forecasting contest. Always
verify (or cite a known result establishing) properness before adopting a new scoring
rule, rather than assuming any "sensible-looking" loss function has this property.

---

### AUC / ROC-AUC

**In one sentence.** For a binary label and a model's predicted score, AUC is the
probability that a randomly chosen positive-labelled case receives a higher predicted
score than a randomly chosen negative-labelled case — a rank-based measure of separation
that only cares about relative ordering, not calibrated probability values.

**The maths.** Equivalent to the Mann-Whitney U statistic: with $n_+$ positive and $n_-$
negative cases and average ranks $R_i$ assigned to the pooled predicted scores (ties
split evenly), $\text{AUC} = \dfrac{\sum_{i \in \text{positive}} R_i - n_+(n_++1)/2}{n_+ n_-}$.
AUC = 0.5 is random ranking; 1.0 is perfect separation; below 0.5 means the score ranks
backwards.

**Why it is here.** Notebook 11c's entry-time loss classifier (Gate LC) needed a
classification analogue of this file's regression/density scoring rules — none of
[log score](#log-score), [CRPS](#crps) or QLIKE apply to a binary stop-exit-vs-zscore-exit
label, and this repo's own `research.py` walk-forward harness (`walk_forward_run`,
`batch_train_reg`) is regression-only, so AUC was added as new, minimal machinery
(`spread_lib11.roc_auc_score`) rather than pulled in from a library this repo's
environment doesn't have (no `sklearn`).

**Worked example.** Gate LC's stitched out-of-sample AUC came back in the 0.67-0.76 range
at all four pre-registered origin offsets on a 55-trade sample — clearing the
pre-registered ">0.60" bar by its literal point-estimate text, but see the pitfall below
before trusting that number on its own.

**Pitfalls.** AUC computed on a handful of out-of-sample cases is a rank statistic on a
small permutation space (with 5 test cases split 2/3 or 3/2 by label, only a few discrete
AUC values are even achievable — 0, 0.25, 0.5, 0.75, 1.0), so a point estimate alone can
look decisive while being consistent with pure noise. Bootstrap the *already-collected*
out-of-sample predictions (resampling cases, not refitting the model) to get a CI on the
AUC estimate itself — exactly as this repo already bootstraps a Sharpe ratio or a
bootstrap CI on a return delta — before treating a small-sample AUC as a real result. A
CI that straddles 0.5 means the point estimate is not distinguishable from chance, even
if it clears a pre-registered threshold on its face value.

---

### Log score

**In one sentence.** The log-density (or log-probability-mass) that a fitted
distribution assigns to what actually happened — higher is better, since a good forecast
should have assigned high probability to the outcome that actually occurred.

**The maths.** $\mathrm{LogScore} = \log f(y_{\mathrm{actual}} \mid \hat{\theta})$, using
the fitted distribution's own [density or mass function](01-probability-and-distributions.md#probability-density-vs-mass-function)
evaluated at the realized value.

**Why it is here.** `distributions.log_score` computes exactly this, and it is the
*primary* metric for notebook 5's Phase 3 density contest, replacing QLIKE as the main
criterion (QLIKE remains as a secondary column only, for continuity with notebook 4).

**Worked example.** Notebook 4's own density-scoring side-note found GARCH-t's
own-distribution log score beating every normal-density rung at 3 of 4 intervals — the
finding notebook 5's entire premise is built on.

**Pitfalls.** Log score is extremely sensitive to the tail — a single observation the
model assigned near-zero probability to produces an enormous negative log score
(approaching $-\infty$ as the assigned density approaches 0), which is a *feature*, not a
bug, for tail-risk evaluation specifically (it heavily penalizes models that dismiss
events that then actually happen) but means a handful of extreme observations can
dominate a whole sample's average log score.

---

### CRPS

**In one sentence.** "Continuous Ranked Probability Score" — a proper scoring rule for a
density forecast that, unlike log score, penalizes based on the *distance* between the
forecast's predicted distribution and the actual outcome, not just the density at that
one point.

**The maths.** $\mathrm{CRPS}(F, y) = \int_{-\infty}^{\infty} \left(F(x) -
\mathbb{1}\{x \ge y\}\right)^2 dx$ — the squared difference between the forecast's CDF
and a "step function" that jumps from 0 to 1 exactly at the realized value $y$,
integrated over all possible $x$. Lower is better (0 for a perfect, certain forecast).

**Why it is here.** `distributions.crps` computes this via direct numerical integration
— and is the subject of notebook 5's §1b bug fix: the numerical integration grid was
built as `linspace(ppf(1e-6), ppf(1-1e-6), n_points)`, which spans wildly different
ranges for different families (a normal spans ~10 units; a Student-t with $\nu\approx 2$
spans ~1,400 units), making cross-family CRPS comparisons partly an artifact of
integration resolution rather than a pure measure of forecast quality.

**Worked example.** For a normal or Student-t forecast, CRPS has a known *closed-form*
formula (no numerical integration needed at all) — notebook 5's fix adds these directly
(`crps_normal_closed_form`, `crps_t_closed_form`), which is both faster and free of the
grid-resolution artifact entirely.

**Pitfalls.** Unlike log score, CRPS is finite and well-behaved even for a forecast that
assigns exactly zero density to the realized outcome — a genuinely different sensitivity
profile from log score, and part of why both are reported side by side rather than
relying on either alone.

---

### QLIKE and why it beats MSE for a noisy variance proxy

**In one sentence.** The standard loss function for grading variance *point* forecasts —
chosen over plain squared error specifically because it stays a fair, unbiased grading
rule even when the "actual" you're comparing against (realized variance) is itself only
a noisy estimate of the true, unobservable variance.

**The maths.** $\mathrm{QLIKE} = \frac{\text{actual}}{\text{predicted}} -
\log\!\left(\frac{\text{actual}}{\text{predicted}}\right) - 1$ — always $\ge 0$, equal to
0 exactly when predicted = actual.

**Why it is here.** This is the *primary* metric of notebook 4's entire Phase 3 ladder,
and remains a reported secondary column in notebook 5. Patton (2011) proved QLIKE is a
[proper scoring rule](#proper-scoring-rule-and-why-properness-matters) even when the "actual" fed into it is a noisy
*proxy* for the true conditional variance (as realized variance always is, being built
from a finite number of sub-observations) — plain MSE does *not* have this robustness
property, and can systematically favor the wrong forecast when scored against a noisy
proxy.

**Worked example.** `dist_lib.qlike_mse` masks `actual > 0` specifically because QLIKE's
`log(ratio)` term is undefined at `actual == 0` — notebook 4's bug #4 found exactly this:
13 frozen-price bars at 1h with `rv_target == 0` poisoned the whole-series mean QLIKE to
`+inf` for every single rung until the mask was corrected.

**Pitfalls.** QLIKE is asymmetric in a specific, deliberate way: under-predicting
variance is penalized differently than over-predicting it by the same ratio — this
asymmetry is *part of* why it's the preferred metric for variance forecasts specifically
(under-predicting risk is arguably the more costly error in practice), not an accident to
correct for.

---

### Mincer-Zarnowitz regression

**In one sentence.** A simple regression-based check of point-forecast calibration:
regress the actual realized value on the forecast, and see whether the resulting slope
is close to 1 and intercept close to 0 — the signature of an unbiased forecast.

**The maths.** $y_t = \beta_0 + \beta_1 \hat{y}_t + \varepsilon_t$, fit by OLS. A
perfectly calibrated forecast (in this linear-regression sense) has $\beta_0=0,
\beta_1=1$; a slope below 1 means the forecast systematically over-predicts at high
values and/or under-predicts at low values (or, more commonly interpreted here, the
forecast is scaled too aggressively relative to what actually happens).

**Why it is here.** `dist_lib.mincer_zarnowitz` runs exactly this regression for every
rung in notebook 4's Phase 3, alongside QLIKE — it's a *different* diagnostic (regression
calibration, not average loss), catching systematic bias that a pure loss-average number
might not surface as clearly.

**Worked example.** Notebook 4 found the range estimators' MZ slope running 0.19-0.45 at
every interval — systematically, substantially biased low — exactly matching Phase 1's
finding that BTC's intrabar range runs 6-14% below the driftless-Brownian prediction
Parkinson/GK/RS/YZ are all built around.

**Pitfalls.** A good MZ slope/intercept doesn't guarantee a good average loss (QLIKE),
and vice versa — they measure genuinely different things (linear-regression bias vs.
average scoring-rule loss); notebook 4 reports both rather than treating either as
sufficient on its own.

---

### $R^2$

**In one sentence.** The fraction of a target variable's variation that a regression
model actually explains — 1.0 means a perfect fit, 0 means the model explains nothing
beyond just guessing the average every time.

**The maths.** $R^2 = 1 - \frac{\sum(y_t - \hat{y}_t)^2}{\sum(y_t - \bar{y})^2}$ — one
minus the ratio of the model's own squared-error total to the squared-error total of the
trivial "always guess the average" forecast.

**Why it is here.** Reported alongside every [Mincer-Zarnowitz](#mincer-zarnowitz-regression)
regression in notebook 4's Phase 3 — and its low values (0.004-0.19 at every interval,
for even the best-performing rungs) are the direct, quantitative basis for
`NEXT_RUN_PROMPT.md`'s framing that "conditional variance at these horizons has an R² of
0.004-0.19" and that vol-forecasting gains, while real, are inherently modest.

**Worked example.** An $R^2$ of 0.19 (the *upper* end of what notebook 4 found) means
even the best-calibrated variance forecast at that interval explains only 19% of the
actual variation in realized variance — the remaining 81% is, from this model's
perspective, unpredictable noise.

**Pitfalls.** A low $R^2$ doesn't mean a forecast is worthless — it can still
significantly beat simpler alternatives (per DM tests) while explaining only a small
share of total variation; "significantly better than the alternatives" and "explains most
of the variation" are different, both worth reporting, claims.

---

### Calibration vs. sharpness

**In one sentence.** Two separate, both-necessary properties of a good density forecast:
**calibration** means the forecast's stated probabilities are honest (a 90% interval
really does contain the truth about 90% of the time); **sharpness** means the forecast is
also usefully *precise* (a narrow interval, not a hopelessly wide one) — a forecast needs
both to be genuinely useful.

**The maths.** No single formula for either; calibration is checked via
[PIT-uniformity](#pit-uniformity) or coverage tests
([Kupiec](#kupiec-unconditional-coverage), [Christoffersen](#christoffersen-independence)),
sharpness is checked by comparing interval widths (or predictive variance) across
competing, similarly-calibrated models.

**Why it is here.** Notebook 5's Gate B is explicitly a calibration test (coverage across
six quantile levels, three tests each) — deliberately separate from Gate A's log-score
contest, which rewards a combination of calibration *and* sharpness together (a sharp
forecast that's also well-calibrated gets the best log score; a well-calibrated but
needlessly wide forecast gets penalized by log score for its lack of sharpness).

**Worked example.** A forecast that always predicts "returns are somewhere between
$-\infty$ and $+\infty$ with 100% probability" is perfectly calibrated (technically never
wrong) but has zero sharpness (useless) — an extreme illustration of why calibration
alone is not sufficient, and why both properties are checked.

**Pitfalls.** It's possible to improve one property while worsening the other (widening
intervals to fix a calibration problem, at the cost of sharpness) — a forecast should be
judged on both together (which is exactly what a proper scoring rule like log score does
automatically), not on either in isolation.

---

### PIT

**In one sentence.** "Probability integral transform" — take each observed value and run
it through its own forecast's CDF; if the forecasts were genuinely correct, the resulting
numbers should look like a plain Uniform(0,1) sample, with no pattern left at all.

**The maths.** $u_t = \hat{F}_t(y_t)$ — the forecast distribution's own CDF, evaluated at
the value that actually occurred. If $\hat{F}_t$ equals the *true* generating
distribution at every $t$, then $u_1, u_2, \dots$ are i.i.d. Uniform(0,1) — a classical
result about CDFs.

**Why it is here.** `distributions.pit_values` computes exactly this, feeding
[`pit_ks_test`](#pit-uniformity) — and PIT is exactly the diagnostic notebook 5's write-up
plans to visualize as "the single most persuasive visual in the notebook" (a QQ plot of
PIT values against the uniform reference).

**Worked example.** If a model consistently under-predicts variance, its PIT values will
cluster too often near 0 and 1 (actual returns landing in the model's predicted tails
more often than they should) rather than spreading uniformly — a visually obvious,
intuitive diagnostic of exactly which direction a calibration failure runs.

**Pitfalls.** PIT-uniformity is a *necessary* condition for good calibration, not a
sufficient one for a good forecast overall — a forecast can have perfectly uniform PIT
values while still being much less sharp than a competing, equally well-calibrated
forecast; see [calibration vs. sharpness](#calibration-vs-sharpness) above.

---

### PIT-uniformity

**In one sentence.** The specific, testable claim behind PIT: if a density forecast is
correctly calibrated, its PIT values across many observations should look statistically
indistinguishable from a Uniform(0,1) sample — checking this directly is a standard,
general-purpose calibration test that works for *any* fitted family.

**The maths.** Formally tested via the
[Kolmogorov-Smirnov test](03-statistical-inference.md#kolmogorov-smirnov-test) of the PIT
sample against the Uniform(0,1) distribution — `distributions.pit_ks_test` does exactly
this.

**Why it is here.** This is the general mechanism behind every "family X's fit is/isn't
well calibrated" verdict in notebook 4's Phase 1 table — comparing normal, Student-t, and
skew-t fits' KS statistics against Uniform(0,1) via their respective PIT samples.

**Worked example.** Notebook 4 found the normal's KS statistic (against its own PIT
values) around 0.08-0.11 at every interval — strongly rejected — while Student-t's
dropped to 0.007-0.027, not rejected at several intervals — directly, this is a
PIT-uniformity test, even though the write-up describes it in terms of the fitted
family's KS statistic.

**Pitfalls.** A KS test on PIT values, using parameters estimated from the *same* data
the PIT values are computed from, is a known, accepted mild approximation (see
[Kolmogorov-Smirnov test](03-statistical-inference.md#kolmogorov-smirnov-test)'s own
pitfall note) — not a fully "clean" textbook KS test, but standard practice for this kind
of applied calibration check.

---

### Value at Risk

**In one sentence.** A specific quantile of a return (or loss) distribution, framed as a
risk threshold: "the loss level such that only $q$% of outcomes are expected to be
worse" — the most widely used single-number risk measure in practical finance, despite
well-known theoretical shortcomings.

**The maths.** $\mathrm{VaR}_q$ is exactly the
[quantile](01-probability-and-distributions.md#quantile-percentile-inverse-cdf)
$Q(q)$ of the return distribution, for a chosen small $q$ (commonly 1%, 5%). A "5% VaR"
of $-3\%$ means: under the model, there's only a 5% chance of losing more than 3% on
this bar.

**Why it is here.** This is exactly what `dist_lib.density_scores`'s `q05` computes, and
what every coverage test (Kupiec, Christoffersen) in this repo is testing the accuracy of
— "was the actual VaR exceedance rate close to the declared 5%?"

**Worked example.** Notebook 4 found GARCH-t's 5% VaR exceedance rate never rejected by
Kupiec at any interval (p=0.18-0.84), while every normal-density alternative's VaR was
rejected at 1h and often elsewhere (p<0.05) — the density-calibration finding that
motivates notebook 5's entire tail-focused pivot.

**Pitfalls.** VaR only tells you the *threshold*, not how bad things get *beyond* it — two
models can have identical, well-calibrated 5% VaR while implying very different average
losses in the worst 5% of cases. This is exactly the shortcoming
[expected shortfall](#expected-shortfall) is built to address, and exactly why VaR alone
is not treated as a
[coherent risk measure](#coherent-risk-measure) in the technical sense.

---

### Expected shortfall

**In one sentence.** The average loss, *given* that the loss already exceeds the VaR
threshold — answers the question VaR can't: "given things are already bad, how bad,
typically?"

**The maths.** $\mathrm{ES}_q = \mathbb{E}[\text{loss} \mid \text{loss} >
\mathrm{VaR}_q]$ — the conditional expectation of the loss, conditional on being in the
tail region VaR defines. Always at least as large (in magnitude) as VaR itself, since it
averages over the tail rather than just marking its edge.

**Why it is here.** This is notebook 5's [GPD](01-probability-and-distributions.md#generalized-pareto)-based
Phase 2 payoff: `gpd_var_es` computes both VaR and ES in one call, and ES specifically is
called out as "the reason to do EVT at all — it is the coherent risk measure VaR is not,
and no other rung in this research programme can produce one" — no earlier notebook 4
rung produces a genuine ES estimate.

**Worked example.** A model might have a perfectly calibrated 1% VaR (correctly saying
"there's a 1% chance of losing more than X") while badly understating expected shortfall
(if it turns out that, on the rare days losses do exceed X, they tend to be catastrophically
larger than X, not just marginally so) — this is precisely why the
[Acerbi-Székely](#acerbi-székely) ES backtest is run as a separate test from VaR coverage.

**Pitfalls.** ES is only finite when the underlying tail's shape parameter is below a
certain threshold ($\xi < 1$ for a [GPD](01-probability-and-distributions.md#generalized-pareto)
tail) — `gpd_var_es` explicitly returns NaN for ES when $\xi \ge 1$ rather than a
misleadingly finite number, a real possibility this repo's own tripwires flag rather than
paper over.

---

### Coherent risk measure

**In one sentence.** A risk measure that satisfies a specific short list of mathematically
sensible properties a "reasonable" measure of risk should have — most notably,
**subadditivity** (diversifying across positions should never make your measured risk
worse than holding them separately) — VaR famously can fail this property; expected
shortfall does not.

**The maths.** A risk measure $\rho$ is coherent if it satisfies: monotonicity (worse
outcomes $\Rightarrow$ higher risk), translation invariance, positive homogeneity, and
**subadditivity**: $\rho(X+Y) \le \rho(X) + \rho(Y)$.

**Why it is here.** Directly cited in `gpd_var_es`'s own docstring as the reason ES is
the preferred tail-risk summary over VaR: "ES is the reason to do EVT at all — it is the
coherent risk measure VaR is not."

**Worked example.** VaR's classic subadditivity failure: it's mathematically possible to
construct two positions where each individually has a small VaR, but their *combined*
position has a *larger* VaR than the sum of the two individual VaRs — the opposite of
what diversification should do to risk, and the specific mathematical pathology that
motivated the search for better risk measures in the first place.

**Pitfalls.** Coherence is a property of the risk measure's *mathematical* definition,
not a guarantee that any particular *estimate* of it (fit from finite, noisy data) will
behave nicely in practice — a well-estimated VaR can still be a perfectly reasonable
practical tool even with this theoretical caveat; ES is preferred here mainly because
notebook 5's whole point is characterizing tail behavior as fully as possible, not
because VaR is unusable.

---

### Exceedance / violation

**In one sentence.** A single instance where the actual outcome breached a forecasted
quantile threshold — the basic observed event that every VaR coverage test is built from
counting.

**The maths.** For a lower-tail VaR forecast $\hat{q}_t$ at level $q$: an exceedance
(violation) at time $t$ is the event $r_t < \hat{q}_t$. Under a correctly calibrated
model, exceedances should occur with (roughly) frequency $q$ and independently over time.

**Why it is here.** `distributions.exceedances` computes exactly this boolean indicator
series, fed directly into every coverage test
([Kupiec](#kupiec-unconditional-coverage), [Christoffersen](#christoffersen-independence)).

**Worked example.** At the 1% level with ~1,460 pre-holdout 1d bars, about 14
exceedances are expected under correct calibration — `NEXT_RUN_PROMPT.md` §9's own
tripwire treats a level with under ~10 observed violations as underpowered, worth
reporting as such rather than quoting its test p-value as if it were meaningful.

**Pitfalls.** A single exceedance is a Bernoulli (yes/no) event — any single one, on its
own, carries very little statistical information; only the *pattern* of exceedances
across many observations (rate, clustering) is informative, which is exactly why
coverage tests need a reasonably large sample to have any real
[power](03-statistical-inference.md#power).

---

### Kupiec unconditional coverage

**In one sentence.** A significance test for whether the *rate* at which a quantile
forecast is exceeded matches its declared probability — the most basic VaR-calibration
check: "does a '5% VaR' actually get breached about 5% of the time?"

**The maths.** A [likelihood-ratio test](02-estimation-and-fitting.md#likelihood-ratio-test)
comparing the observed exceedance rate $\hat{p} = x/n$ (out of $n$ observations with $x$
exceedances) against the declared rate $p$: $\mathrm{LR} = -2\left[x\log p + (n-x)\log(1-p)
- x\log \hat{p} - (n-x)\log(1-\hat{p})\right] \sim \chi^2(1)$ under $H_0: \text{true
rate} = p$.

**Why it is here.** `distributions.kupiec_test` implements exactly this, and it is the
first (and, in notebook 4, only) coverage test applied — notebook 5's Phase 4 extends it
to all six quantile levels for every model, per Gate B.

**Worked example.** Notebook 4 found GARCH-t's 5% VaR never rejected by Kupiec at any
interval (p=0.18-0.84); every normal-density rung's 5% VaR was rejected at 1h and often
elsewhere — Kupiec alone was enough to surface this specific finding.

**Pitfalls.** Kupiec only checks the *overall rate* — it cannot detect a model whose
exceedances happen at exactly the right frequency but arrive all clustered together
(e.g. every violation during one crash week) rather than spread evenly — that failure
mode is invisible to Kupiec and only shows up in
[Christoffersen's independence test](#christoffersen-independence), which is exactly why
the two are always paired.

---

### Christoffersen independence

**In one sentence.** A test for whether VaR exceedances cluster together in time, rather
than occurring independently — complements Kupiec by catching a specific failure mode
Kupiec is blind to: right overall rate, but breaches arriving in clumps.

**The maths.** Models exceedances as a 2-state (hit/no-hit) first-order Markov chain and
tests whether the transition probability $P(\text{hit}_t \mid \text{hit}_{t-1})$ genuinely
differs from $P(\text{hit}_t \mid \text{no-hit}_{t-1})$ — under independence, these two
should be equal (whether you just had a violation shouldn't change your odds of having
another one next bar). A likelihood-ratio statistic, $\chi^2(1)$ under the null of
independence.

**Why it is here.** `distributions.christoffersen_independence_test` implements this, and
`NEXT_RUN_PROMPT.md`'s own coverage battery singles it out as "precisely the test
notebook 4 ran for exactly one model at exactly one level" — Phase 4 of notebook 5
extends it to the full 6-quantile x 10-model grid, motivated directly by Phase 1's own
finding that waiting times between extreme events are gamma-shaped (over-dispersed,
i.e. genuinely clustered), not exponential (memoryless) — a model whose exceedances still
cluster despite a correct overall rate would be exactly the failure this test is built to
catch.

**Worked example.** A GARCH-normal model, whose innovation distribution assumes
independent, identically-scaled normal shocks, is a natural candidate to fail this test
specifically at times when real clustering (which GARCH's variance recursion only
partially captures) still shows through in the standardized residuals' extreme tail.

**Pitfalls.** Needs a reasonably large number of *both* hit and no-hit transitions to be
well-powered — at a 1% VaR level with only ~14 expected violations at 1d,
`NEXT_RUN_PROMPT.md` §9 explicitly flags this test as "thin" (underpowered) even though
Kupiec alone remains reasonably usable at that same sample size.

---

### Duration-based coverage test

**In one sentence.** Instead of asking "is the *rate* of VaR violations right" (Kupiec)
or "does one violation raise the odds of the very next bar also being one" (Christoffersen
independence), this asks the question directly in the space that actually matters for
risk management: are the *gaps between violations* memoryless, the way an i.i.d. process's
would be?

**The maths.** Kupiec sees only a violation count; Christoffersen independence sees only
adjacent (lag-1) pairs — a 2-state Markov chain literally cannot represent clustering that
shows up three or five bars apart. The duration-based test instead fits the sequence of
gaps between consecutive violations (coded as bars-since-last-violation minus one, so the
support starts at 0) to a [geometric distribution](01-probability-and-distributions.md#geometric-distribution)
(the discrete, memoryless i.i.d. null — a plain restatement of "violations are i.i.d.
Bernoulli," reframed in duration space) against a **discrete Weibull**
(Nakagawa & Osaki 1975: survival function $P(X>k) = q^{k^\beta}$, nesting the geometric
exactly at $\beta=1$). $\beta<1$ means a *falling* hazard — having just had a violation
makes the next one more likely soon than memorylessness implies, i.e. genuine clustering,
visible at any lag, not just lag 1. Compared by a plain (non-boundary — $\beta=1$ is an
*interior* point, see [boundary likelihood-ratio test](03-statistical-inference.md#boundary-likelihood-ratio-test))
$\chi^2_1$ likelihood-ratio test.

**Why it is here.** Notebook 4 already measured gamma waiting-time shapes of 0.52-0.85
(direct evidence violations clump on scales a 2-state chain cannot see); Phase 4 of
notebook 6 (`dist_lib6.fit_geometric_durations` / `fit_discrete_weibull_durations`,
`src/results/006_distribution_zoo.md`) is the first place this repo actually tests that
observation as a formal calibration failure mode, alongside a complementary count-based
test (weekly violation counts: Poisson null vs. negative binomial, a
[boundary likelihood-ratio test](03-statistical-inference.md#boundary-likelihood-ratio-test)
since Poisson sits at the negative binomial's dispersion boundary).

**Worked example.** On BTC at 12h, GARCH-EVT's 1% violations show a fitted discrete-Weibull
$\hat\beta$ close to but not significantly below 1, and its count-based dispersion is
statistically indistinguishable from Poisson — while several thin-tailed models on the same
data show both significantly overdispersed counts and $\hat\beta$ well below 1, i.e.
genuinely clustered violations, not just a right-on-average rate.

**Pitfalls.** A model can pass Kupiec and Christoffersen independence (both individually
weak against clustering beyond lag 1) while still failing this test — read literally, that
means the practical risk statement "this model's 1% VaR is trustworthy" can be weaker than
notebook 5's Gate B implied, since Gate B never ran this test. Needs at least a handful of
violations to fit at all (`dist_lib6.fit_geometric_durations` requires 10+ gaps); at very
low violation counts (thin symbols/intervals) this test is itself underpowered, same
caveat as Christoffersen independence.

---

### Conditional coverage

**In one sentence.** A single combined test folding Kupiec's "is the rate right" question
and Christoffersen's "are violations independent" question into one joint test — passing
it means a model is calibrated in *both* senses at once.

**The maths.** $\mathrm{LR}_{\mathrm{cc}} = \mathrm{LR}_{\mathrm{Kupiec}} +
\mathrm{LR}_{\mathrm{independence}}$, following a $\chi^2(2)$ distribution under the
joint null (correct rate *and* independence) — simply the sum of the two component
statistics, since they test complementary, roughly separable aspects of calibration.

**Why it is here.** `distributions.christoffersen_conditional_coverage_test` implements
exactly this combined test, and it's the third of the three tests in notebook 5's Phase 4
coverage battery (Gate B requires passing all three, at all six quantile levels, for
every model) — `NEXT_RUN_PROMPT.md` flags this function as "currently unused anywhere in
the repo," meaning its integration path (as opposed to its unit tests) had never actually
been exercised before notebook 5, a specific, named risk ("treat first use as a place
bugs live").

**Worked example.** A model could individually pass Kupiec (right rate) and fail
independence (clustered violations), or the reverse — the conditional coverage test
folds both signals into one number and one verdict, useful as a single pass/fail summary
even though the individual component tests carry more diagnostic detail about *which*
kind of miscalibration is present.

**Pitfalls.** Passing the combined test doesn't tell you *which* of the two component
failures (if any) is close to the boundary — `NEXT_RUN_PROMPT.md` explicitly asks for
"the full grid, not just pass/fail" in Phase 4's reporting, specifically so a model
narrowly failing one component isn't indistinguishable from one failing badly on both.

---

### Acerbi-Székely

**In one sentence.** A backtest specifically for expected shortfall calibration — checks
not just whether VaR is breached the right number of times, but whether the *severity* of
losses beyond VaR matches what the model's own expected-shortfall prediction implied.

**The maths.** Test statistic (their "Test 2"):
$$Z = \frac{1}{nq}\sum_t \frac{r_t \cdot \mathbb{1}\{r_t < \mathrm{VaR}_t\}}{\mathrm{ES}_t} - 1$$
Under a correctly calibrated model, $Z \approx 0$ in expectation. $Z > 0$ means realized
tail losses, on the days they occur, are *worse* on average than the model's own ES
prediction said they'd be — the specific failure mode that matters most for risk
management ($r_t$ and $\mathrm{ES}_t$ are both negative for a lower-tail loss, so a
realized loss more extreme than predicted makes the ratio, and hence $Z$, positive — a
model that instead overstates risk produces $Z < 0$). No simple closed-form reference
distribution exists, so the p-value is computed via a bootstrap simulated under the
model's own predictive distribution.

**A note on getting this sign right.** Both the $-1$ (not $+1$) and the "$Z>0$ means
worse-than-predicted" (not "$Z<0$") direction were verified numerically against a
20-million-draw Monte Carlo before being trusted, not just re-derived on paper — an
initial pseudocode pass (matching a common way this formula gets mis-transcribed in
practice) used "$+1$" and stated the opposite direction, which a direct simulation check
(a model with the wrong, too-small volatility, so realized losses are known to be worse
than predicted) immediately showed was backwards. Worth remembering as a general lesson:
a formula "matching the literature" from memory or a written spec is not the same as a
formula checked against a case whose right answer you already know.

**Why it is here.** This is notebook 5's Phase 4 addition specifically because "VaR
coverage tests say nothing about how bad the losses are once the threshold is breached,
and ES is the one risk quantity the EVT models can produce that nothing else in this
codebase can" — the natural, matching backtest for the [expected shortfall](#expected-shortfall)
numbers the GPD-based models produce.

**Worked example.** A model could pass every VaR coverage test (right breach rate, no
clustering) while still badly understating how catastrophic the breaches actually are —
Acerbi-Székely is specifically built to catch exactly that combination, which none of
the three coverage-battery tests above can.

**Pitfalls.** Requires a bootstrap under the model's own predictive distribution (no
closed form), which is itself only as trustworthy as the model being tested — a
mis-specified model's own bootstrap could, in principle, understate the true
sampling variability of $Z$; treated in this repo as the best available option given ES
has no simpler closed-form test, not as a perfect, assumption-free check.
