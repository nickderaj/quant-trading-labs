# 04 — Volatility models

This file assumes [01](01-probability-and-distributions.md)-[03](03-statistical-inference.md).
It covers models of *how much* a return series moves — its second moment — as opposed to
which direction it moves. This is the subject of notebook 4's Phase 3 and the majority of
notebook 5.

---

### Volatility

**In one sentence.** How much a price (or return) tends to move around, regardless of
direction — high volatility means big swings are common, low volatility means the price
sits relatively still.

**The maths.** Formally, volatility is [standard deviation](01-probability-and-distributions.md#standard-deviation)
of returns, usually annualized by convention in traditional finance
(multiplying by $\sqrt{252}$ for daily data) — this repo mostly works in the bar's own
native units instead (per-bar variance/std), since the forecasting contests compare
models against each other at a fixed interval rather than needing a cross-interval,
annualized number.

**Why it is here.** It is the single quantity every model in this file exists to
forecast — every rung of notebook 4's Phase 3 ladder outputs a variance forecast for the
next bar, scored against a realized proxy.

**Worked example.** BTC's fitted 1h standard deviation of $\approx 0.00584$ (0.58%) means
a "typical" hourly move is around half a percent — small in isolation, but compounding
over a year of hourly bars into the large price swings crypto is known for.

**Pitfalls.** "Volatility" colloquially sometimes implies "risk of loss," but as defined
here it's symmetric — it says nothing about direction, only magnitude. A volatility
forecast alone never tells you whether a move is more likely up or down; see
[regime models](05-regime-models.md)'s "predicts vol, not direction" finding for the
direct empirical confirmation that these are separate questions for this data.

---

### Variance vs. standard deviation

**In one sentence.** Two ways of reporting the same underlying spread — variance is in
squared units (awkward to interpret directly but algebraically convenient), standard
deviation is its square root, back in the original units (easier to read against actual
price moves).

**The maths.** See [variance](01-probability-and-distributions.md#variance) and
[standard deviation](01-probability-and-distributions.md#standard-deviation) for the full
definitions; $\sigma = \sqrt{\mathrm{Var}}$.

**Why it is here.** This repo's models mostly forecast *variance* directly (GARCH's
$\sigma_t^2$, HAR-RV's `rv_target`), because variance is what adds up nicely across time
and what QLIKE scores directly — but VaR/quantile calculations need the *standard
deviation* (return-unit scale), so both units appear throughout the codebase and must be
tracked carefully.

**Worked example.** `dist_lib.density_scores` explicitly takes a variance forecast and
converts it: `st.norm(loc=0, scale=np.sqrt(vi))` — the `np.sqrt` there is exactly this
variance-to-standard-deviation conversion, done at the point where a return-unit quantile
is needed.

**Pitfalls.** Forgetting to convert between the two (using a variance number where a
standard deviation is needed, or vice versa) is a real, easy-to-make bug in exactly this
kind of code — always check whether a function's docstring or variable name says `var`,
`sig2`/`sigma2` (variance) vs. `sigma`/`std` (standard deviation).

---

### Realized variance / volatility

**In one sentence.** An *after-the-fact*, measured estimate of how much a price actually
moved over some period — as opposed to a *forecast* of how much it will move — built from
the actual observed sub-period returns.

**The maths.** For a bar composed of $m$ finer sub-returns $r_1,\dots,r_m$: realized
variance $\mathrm{RV} = \sum_{i=1}^m r_i^2$. When no finer sub-bar data exists, the
coarsest usable proxy is just the bar's own single squared return, $r^2$ — noisier
(built from one observation instead of many) but the only option available.

**Why it is here.** This is the `rv_target` column throughout `dist_lib.py` — the
"actual" against which every Phase 3 forecast is scored. `realized_variance_from_subbars`
builds it from 1h sub-bars for 4h/12h/1d bars specifically because summing several
finer-grained squared returns is a materially less noisy proxy than one coarse bar's own
single squared return.

**Worked example.** At 1h itself, no finer cached data exists, so `rv_target` falls back
to `bar_squared_return` — explicitly noted in this repo's own code and write-ups as a
noisier proxy at that one interval, a real limitation acknowledged rather than hidden.

**Pitfalls.** Realized variance from a *small* number of sub-observations is itself a
noisy estimate of the "true" underlying variance (whatever that means for a genuinely
time-varying process) — this is precisely why QLIKE (rather than plain squared error) is
used to score forecasts against it: QLIKE is specifically robust to the *proxy itself*
being noisy, per [QLIKE and why it beats MSE](06-scoring-rules-and-calibration.md#qlike-and-why-it-beats-mse-for-a-noisy-variance-proxy).

---

### Volatility clustering

**In one sentence.** The empirical fact that big moves tend to be followed by more big
moves (and calm periods by more calm), rather than volatility jumping around
unpredictably bar to bar — the single most important stylized fact motivating every
model in this file.

**The maths.** Formally: squared (or absolute) returns are positively
[autocorrelated](03-statistical-inference.md#autocorrelation) — $\mathrm{Corr}(r_t^2,
r_{t-k}^2) > 0$ for many lags $k$ — even when raw returns themselves show little to no
autocorrelation.

**Why it is here.** This is the entire justification for [GARCH](#garch11) existing at
all: if variance were constant over time, there would be nothing to model beyond a single
number. Notebook 4's own waiting-time analysis (gamma shape $<1$ for gaps between
$k$-sigma events, at every interval) is a distributional confirmation of exactly this
fact, independent of any GARCH fit.

**Worked example.** Notebook 4 measured waiting-time gamma shapes between 0.52 and 0.85
across intervals and thresholds — all below 1, meaning big moves arrive in a more
"clumped" pattern than a memoryless process would produce, the quantitative signature of
clustering.

**Pitfalls.** Volatility clustering is a statement about the *magnitude* of returns
clustering, not their *sign* — a highly volatile period can still have roughly balanced
up and down moves within it; conflating "volatile" with "directionally biased" is exactly
the error the regime models in [05](05-regime-models.md) explicitly test for and reject.

---

### Conditional vs. unconditional variance

**In one sentence.** Unconditional variance is "the overall, long-run average variance of
this series"; conditional variance is "the variance right now, given what's happened
recently" — the second can be much higher or lower than the first at any given moment,
even though they average out to the same long-run number.

**The maths.** Unconditional: $\mathrm{Var}(r_t)$, a single fixed number describing the
whole series. Conditional: $\mathrm{Var}(r_t \mid r_{t-1}, r_{t-2}, \dots) = \sigma_t^2$,
which genuinely changes bar to bar as new information (recent returns) arrives.

**Why it is here.** GARCH's whole point is modeling $\sigma_t^2$ (conditional), not just
reporting a single $\mathrm{Var}(r)$ (unconditional) for the whole series — `fit_garch11`'s
`uncond = omega / max(1 - alpha - beta, 1e-6)` computes the *unconditional* variance
(used only to seed the recursion at $t=0$), while the actual output forecast, `sig2`, is
the time-varying *conditional* variance path.

**Worked example.** Notebook 4's write-up explicitly separates these two questions for
GARCH innovations: "skew-t isn't worth it for BTC" was found from an *unconditional* fit
in Phase 1, but the report flags this as "a different, conditional question" for the
GARCH innovation choice specifically — a distinction notebook 5's Phase 2 GJR-GARCH work
follows up on directly.

**Pitfalls.** A model that only ever reports unconditional variance (a single number for
the whole sample) says nothing useful about *when* risk is elevated — this is why every
forecasting rung in this repo, even the simplest ones (trailing std, EWMA), outputs a
time-varying, conditional series, not one static number.

---

### One-step-ahead forecast

**In one sentence.** A prediction of the very next bar's value (return, variance,
whatever), made using only information available up to and including the current bar —
the standard, and simplest, forecasting horizon used throughout this programme.

**The maths.** Forecast for bar $t+1$, denoted $\hat{y}_{t+1|t}$, built entirely from
data available through bar $t$: $\hat{y}_{t+1|t} = g(y_t, y_{t-1}, \dots)$ for some
function $g$.

**Why it is here.** Every forecasting rung in this repo is scored as a one-step-ahead
forecast, evaluated against the corresponding realized value — the simplest, most
directly interpretable forecasting task, and the one every causality rule in
[08](08-research-methodology.md) is built to protect.

**Worked example.** `rolling_garch_forecast`'s inner loop computes exactly this: the
variance "in force" to forecast bar $t$'s own realization, using only returns strictly
before $t$ — the comment in that function notes explicitly that `forecast[t]` "holds the
variance used to forecast bar t's own realization," the precise one-step-ahead
definition.

**Pitfalls.** It is easy to accidentally build a forecast that uses information *from*
bar $t$ to forecast bar $t$ itself — exactly the [lookahead bias](08-research-methodology.md#lookahead-bias-leakage)
class of bug that produced notebook 4's HAR-RV same-bar-target-leak and notebook 5's own
§1a GARCH-t degrees-of-freedom leak.

---

### Trailing standard deviation

**In one sentence.** The simplest possible volatility forecast: just compute the actual
standard deviation of the last $w$ returns, and use that as your guess for the next
bar's volatility.

**The maths.** $\hat{\sigma}_{t+1|t} = \sqrt{\frac{1}{w}\sum_{i=t-w+1}^{t} r_i^2}$ (a
simple [rolling window](02-estimation-and-fitting.md#rolling-trailing-window), zero-mean
assumed for short horizons).

**Why it is here.** This is rung 0 of notebook 4's Phase 3 ladder (`rung0_trailing_std`,
three window sizes tried: 8, 24, 96 bars) — the baseline every more sophisticated model
in this file must beat to be worth its added complexity.

**Worked example.** Notebook 4 found trailing std's best window landed roughly in the
middle of the QLIKE ranking at every interval — not the worst rung, but consistently
beaten (though not always *significantly* beaten) by HAR-RV, the range estimators, and
GARCH-normal.

**Pitfalls.** The window size $w$ is a genuine trade-off: too short and the estimate is
noisy (few observations); too long and it reacts too slowly to real changes in volatility
(dragging in stale, no-longer-relevant history) — exactly why this repo tries multiple
window sizes rather than committing to one arbitrarily.

---

### EWMA / RiskMetrics and $\lambda$

**In one sentence.** A volatility forecast that's a weighted average of all past squared
returns, with the weights decaying exponentially into the past — recent returns matter
much more than old ones, controlled by a single decay parameter $\lambda$.

**The maths.** $\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1-\lambda) r_{t-1}^2$ — this
recursive form is mathematically equivalent to an infinite weighted sum
$\sigma_t^2 = (1-\lambda)\sum_{k=1}^{\infty} \lambda^{k-1} r_{t-k}^2$, where each
successively older squared return gets weight $\lambda$ times smaller than the one before
it. $\lambda$ close to 1 means very slow decay (long memory); $\lambda$ close to 0 means
almost no memory (reacts fully to the last observation). RiskMetrics (the industry
convention this is named after) popularized $\lambda = 0.94$ for daily data.

**Why it is here.** `dist_lib.rung1_ewma` implements this exactly, with the RiskMetrics
default $\lambda=0.94$ — deliberately used *unchanged* across all four intervals in
notebook 4 (not re-tuned per interval), to test the standard baseline as commonly used
rather than a best-tuned version of it.

**Worked example.** Notebook 4 found EWMA anomalously bad at 4h/12h/1d (QLIKE noticeably
worse than trailing std) — traced directly to the fixed $\lambda=0.94$: a decay rate
calibrated for daily-or-finer data adapts far too slowly when applied per-*bar* at
coarser intervals, since each bar already spans much more calendar time.

**Pitfalls.** $\lambda=0.94$ is not a universal constant — it's calibrated for a specific
data frequency (daily), and applying the same *per-bar* decay rate at a different bar
frequency changes its effective *calendar-time* memory length substantially. Retuning
$\lambda$ per interval is explicitly listed as a reasonable, deferred follow-up in
`NEXT_RUN_PROMPT.md`'s out-of-scope section, not something this repo has done yet.

---

### ARCH

**In one sentence.** The original conditional-heteroskedasticity model (Engle, 1982),
predecessor to GARCH: next-period variance is a function of *only* recent squared
returns, with no memory of its own past variance level.

**The maths.** ARCH($q$): $\sigma_t^2 = \omega + \sum_{i=1}^{q} \alpha_i r_{t-i}^2$ — a
weighted sum of the last $q$ squared returns, no $\sigma_{t-1}^2$ term at all (that
addition is exactly what makes GARCH "generalized").

**Why it is here.** Mentioned as the direct ancestor of [GARCH(1,1)](#garch11) below,
which is what this repo actually implements and uses — ARCH itself is not separately
fit anywhere in this codebase, since GARCH(1,1) nests it as a special case (with $q=1$
and $\beta=0$) and is a strictly more flexible, standard choice.

**Worked example.** ARCH($q$) typically needs a large $q$ (many lagged squared returns)
to capture the same persistence GARCH(1,1) achieves with just two parameters
($\alpha,\beta$) — the practical reason GARCH largely superseded plain ARCH in applied
work.

**Pitfalls.** None specific to this repo since ARCH itself isn't fit here — its role is
purely to make [GARCH(1,1)](#garch11)'s own recursion easier to understand as "ARCH, plus
a memory-of-its-own-past-value term."

---

### GARCH(1,1)

**In one sentence.** The workhorse volatility model: tomorrow's variance is a weighted
combination of a long-run baseline level, yesterday's squared return (a shock term), and
yesterday's own variance (a memory term) — the three terms letting it both react to
recent shocks and smoothly persist.

**The maths.**
$$\sigma_t^2 = \omega + \alpha\, r_{t-1}^2 + \beta\, \sigma_{t-1}^2$$
Three parameters, each with a distinct role, explained separately as `NEXT_RUN_PROMPT.md`
requires:
- $\omega$ (Greek letter "omega") — a small positive constant setting the model's
  long-run, unconditional variance baseline (specifically,
  $\omega/(1-\alpha-\beta)$ is that baseline — see [stationarity constraint](#stationarity-constraint)).
- $\alpha$ — how strongly *yesterday's shock* (squared return) feeds into today's
  variance: a large recent move raises the variance forecast by $\alpha$ times its
  square.
- $\beta$ — how strongly *yesterday's variance itself* carries forward: controls how
  long an elevated-variance episode persists once a shock has passed.

**Why it is here.** This is rung 5 of notebook 4's Phase 3 ladder (with normal, t, and
skew-t innovations) and the base model both notebook 5's GJR-GARCH and conditional-EVT
work build on top of. `dist_lib._garch_variance_path` is its literal recursive
implementation, one bar at a time (necessarily sequential — each $\sigma_t^2$ depends on
the previous one, so it can't be vectorized away).

**Worked example.** Notebook 4's GARCH-normal was one of the three-way-tied best-cluster
rungs at every interval (statistically indistinguishable from HAR-RV and the range
estimators) — the point-forecast contest found no advantage to GARCH's more elaborate
structure over HAR-RV's simpler multi-horizon averaging, on this particular data.

**Pitfalls.** GARCH's variance recursion, once fit, must still be *forward-filled*
between refits (re-rolled forward on realized returns using the last-fitted
$\omega,\alpha,\beta$) — a subtle piece of causal bookkeeping `NEXT_RUN_PROMPT.md`
explicitly warns is "correct and subtle, do not break it," and the exact mechanism that
notebook 5's §1a fix had to extend (forward-filling $\nu$ the same careful way) rather
than reinvent.

---

### Persistence ($\alpha + \beta$)

**In one sentence.** How long a volatility shock's effect lingers before fading back to
the long-run average — the closer $\alpha+\beta$ is to 1, the longer elevated variance
takes to decay back down.

**The maths.** In GARCH(1,1), $\alpha+\beta$ is exactly the decay rate of the variance
process back toward its unconditional mean after a shock — a shock's effect on variance
$k$ bars later has shrunk by a factor of roughly $(\alpha+\beta)^k$.

**Why it is here.** Governs the [stationarity constraint](#stationarity-constraint)
below ($\alpha+\beta < 1$ required) and is the single number most often used to summarize
"how sticky is volatility" for a fitted GARCH model.

**Worked example.** `_garch_negloglik`'s explicit rejection of any candidate with
`alpha + beta >= 0.999` treats persistence approaching 1 as a boundary to guard against
(a process whose variance shocks essentially never decay, edging toward
non-stationarity) rather than a legitimate high-persistence fit.

**Pitfalls.** A very high fitted persistence can be a genuine finding (real markets do
show long volatility memory) *or* a sign of numerical instability/near-non-stationarity
in the fit — always check where $\alpha+\beta$ sits relative to the `0.999` guard rail
this repo enforces, not just whether the optimizer "converged."

---

### Stationarity constraint

**In one sentence.** The requirement that a GARCH model's implied long-run variance
actually be a positive, finite number — without it, the model's variance process could
grow without bound or become mathematically nonsensical.

**The maths.** GARCH(1,1)'s unconditional variance is $\omega/(1-\alpha-\beta)$, which
requires $\alpha + \beta < 1$ (otherwise the denominator is zero or negative) alongside
$\omega > 0$, $\alpha,\beta \ge 0$.

**Why it is here.** `_garch_negloglik` enforces this directly: `if omega <= 1e-12 or
alpha < 0 or beta < 0 or alpha + beta >= 0.999: return 1e10` — any candidate parameter
set violating stationarity is given an enormous (effectively infinite) negative
log-likelihood, so the optimizer is steered away from it entirely. The GJR-GARCH
extension in notebook 5 has its own analogous constraint,
$\alpha + \gamma/2 + \beta < 0.999$ (the $\gamma/2$ reflecting that the leverage
indicator fires roughly half the time).

**Worked example.** A fit with $\alpha=0.5,\beta=0.6$ ($\alpha+\beta=1.1 > 1$) would
imply an unconditional variance of $\omega/(1-1.1) = \omega/(-0.1)$ — negative, which is
nonsensical for a variance — exactly why the optimizer is never even allowed to evaluate
such a candidate as a real answer.

**Pitfalls.** A model can be numerically "close" to violating this constraint
($\alpha+\beta$ just under 0.999) without literally violating it — this is the boundary
case discussed under [persistence](#persistence-alpha-beta) above, worth flagging
explicitly whenever it occurs rather than treating it as an ordinary interior solution.

---

### Innovations

**In one sentence.** The part of a return that's genuinely new/unpredictable *after*
accounting for the model's variance forecast — the "shock," standardized so it has
(roughly) constant scale, whose *distribution* (normal, Student-t, ...) is a separate
modeling choice from the variance recursion itself.

**The maths.** $z_t = r_t / \sigma_t$ — the return standardized by its own model's
one-step-ahead volatility forecast. GARCH assumes $z_t$ is i.i.d. from some chosen family
(normal, Student-t, skew-t); the variance recursion and the innovation distribution are
two independent modeling choices that combine multiplicatively: $r_t = \sigma_t z_t$.

**Why it is here.** This is exactly what "GARCH-normal" vs. "GARCH-t" vs. "GARCH-skewt"
mean — the *same* variance recursion, three different assumed shapes for $z_t$. This
decoupling is precisely why notebook 4's write-up distinguishes "skew-t isn't worth it
unconditionally" (a Phase 1 finding about raw returns) from "innovation choice is a
different, conditional question" (about $z_t$ specifically, after GARCH has already
absorbed the time-varying scale).

**Worked example.** `_garch_negloglik`'s `t` branch standardizes exactly this way:
`z = r / np.sqrt(sig2)`, then evaluates a standardized Student-t density on `z` (scaled
by $\sqrt{\nu/(\nu-2)}$ to keep unit variance) — the innovation distribution is fit
*given* the variance path, not instead of it.

**Pitfalls.** A model can have a perfectly good variance recursion but a badly-chosen
innovation distribution (or vice versa) — notebook 4's density-scoring result (GARCH-t
beating GARCH-normal on log score) is specifically an innovation-distribution finding,
not a claim that GARCH-t's variance forecast itself is any different from GARCH-normal's
(the recursion is identical; only the assumed shape of $z_t$ differs).

---

### GJR-GARCH and the leverage effect

**In one sentence.** A GARCH variant that lets *down*-moves raise future variance more
than *up*-moves of the same size — capturing the well-documented pattern (in equities
especially) that bad news tends to increase volatility more than equally-sized good news.

**The maths.**
$$\sigma_t^2 = \omega + (\alpha + \gamma \cdot \mathbb{1}\{r_{t-1}<0\})\, r_{t-1}^2 + \beta\,\sigma_{t-1}^2$$
where $\mathbb{1}\{\cdot\}$ is an indicator (1 if the condition holds, 0 otherwise) — so
a negative $r_{t-1}$ gets an *extra* $\gamma r_{t-1}^2$ boost to next period's variance
that a positive return of the same size doesn't. $\gamma = 0$ collapses this exactly to
[plain GARCH(1,1)](#garch11) — a nested model, directly testable via a
[likelihood-ratio test](02-estimation-and-fitting.md#likelihood-ratio-test) on
$\gamma = 0$.

**Why it is here.** This is notebook 5's Phase 2a: the one asymmetry model added to the
codebase, motivated by the observation that notebook 4's "skew-t isn't worth it" finding
was about the *unconditional* return distribution, not about whether variance itself
reacts asymmetrically to positive vs. negative shocks — a genuinely different question.

**Worked example.** "Leverage effect" gets its name from equities: a stock price drop
mechanically raises a company's debt-to-equity ratio (more leverage), which theoretically
raises the riskiness (and hence volatility) of its equity going forward — whether the
same *pattern* (not necessarily the same mechanism) shows up in a leverage-free
instrument like crypto perpetual futures is exactly what notebook 5's $\gamma$
significance test is checking.

**Pitfalls.** GJR nests GARCH, but *not* skew-t innovations in the same clean way — this
is why notebook 5 fits GJR only with normal and Student-t innovations, explicitly
skipping a skew-t GJR variant as
[over-parameterized](02-estimation-and-fitting.md#overparameterization) (5 shape
parameters plus leverage on a 500-bar window).

---

### EGARCH (mention only)

**In one sentence.** Another common asymmetric-variance GARCH variant, modeling the
*logarithm* of variance rather than variance itself — mentioned here only for
completeness; it is explicitly not implemented or used anywhere in this repo.

**The maths.** Roughly: $\log(\sigma_t^2) = \omega + \alpha\left(|z_{t-1}| -
\mathbb{E}|z_{t-1}|\right) + \gamma z_{t-1} + \beta \log(\sigma_{t-1}^2)$ — modeling
log-variance sidesteps needing to constrain $\sigma_t^2 > 0$ explicitly (a logarithm can
be any sign; exponentiating back always gives a positive variance).

**Why it is here.** `NEXT_RUN_PROMPT.md` explicitly places EGARCH in its out-of-scope
list: "GJR is the one asymmetry model in scope. Expanding the model zoo re-creates
exactly the 'everything ties' problem notebook 4 already ran into" — named here only so a
reader encountering the term elsewhere knows what it is and why this repo doesn't use it.

**Worked example.** Not applicable — not fit anywhere in this codebase.

**Pitfalls.** Not applicable for the same reason; listed for completeness of the
documentation's coverage of "asymmetric GARCH variants a reader might have heard of,"
not because it appears in any notebook here.

---

### HAR-RV and the daily/weekly/monthly cascade

**In one sentence.** A simple, surprisingly hard-to-beat volatility forecast: tomorrow's
realized variance is predicted by a weighted combination of the average realized
variance over the last day, the last week, and the last month — three different
timescales feeding one linear regression.

**The maths.** $\mathrm{RV}_{t+1} = \beta_0 + \beta_d \mathrm{RV}_t^{(d)} + \beta_w
\mathrm{RV}_t^{(w)} + \beta_m \mathrm{RV}_t^{(m)}$, where $\mathrm{RV}_t^{(d)}$ is the
trailing daily average, $\mathrm{RV}_t^{(w)}$ the trailing weekly average, $\mathrm{RV}_t^{(m)}$
the trailing monthly average — all as of (and using data only through) bar $t$, fit by
ordinary least squares.

**Why it is here.** This is rung 2 of notebook 4's Phase 3 ladder, and the single
best-by-QLIKE rung at *every* interval in that ladder — `NEXT_RUN_PROMPT.md` explicitly
calls it "the real benchmark in the modern literature."

**Worked example.** `make_har_features`'s docstring documents a real, serious bug found
and fixed in this repo: the three rolling-mean windows were originally *not shifted* by
one bar, so at the 1d interval (where the "daily" window is just 1 bar), the daily
feature was literally identical to that bar's own target — a same-bar lookahead leak
visible as an implausibly perfect (near-zero) QLIKE score before the fix. All three
windows are now explicitly `shift(1)`ed.

**Pitfalls.** Because HAR-RV is fit by plain OLS on rolling windows, it inherits every
lookahead risk of [rolling OLS refit](02-estimation-and-fitting.md#rolling-trailing-window)
generally — the bug above is the canonical, already-realized example of exactly this
failure mode in this repo's own history, precisely the kind of "implausibly good result"
tripwire `NEXT_RUN_PROMPT.md` §9 warns to check for on every new model.

---

### Range estimators (Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang)

**In one sentence.** A family of volatility estimators that use the whole OHLC (open,
high, low, close) bar — not just the close-to-close return — on the theory that the
high-low range carries extra information about how much a price moved *within* the bar,
not just where it ended up.

**The maths, each estimator's own assumption stated:**
- **Parkinson**: $\widehat{\sigma^2} = \frac{\ln(H/L)^2}{4\ln 2}$, assuming a pure,
  driftless Brownian (random-walk) path within the bar — see
  [the $2\sqrt{2/\pi}$ constant](#the-2sqrt2pi-constant) for where the $4\ln 2$
  normalization comes from.
- **Garman-Klass**: combines the high-low range with the open-to-close move, correcting
  Parkinson for a non-zero drift within the bar.
- **Rogers-Satchell**: explicitly allows for drift (a trending, not driftless, intrabar
  path) — the one estimator in this family that doesn't assume the driftless-Brownian
  relationship at all.
- **Yang-Zhang**: adds an overnight (gap) variance term to Rogers-Satchell's intrabar
  term, meant for markets with a genuine close-to-next-open gap.

**Why it is here.** These are rung 3 of notebook 4's Phase 3 ladder. Notebook 4's own
Phase 1 measured BTC's normalized intrabar range running **6-14% below** the driftless-Brownian
prediction $2\sqrt{2/\pi}$ at every interval — directly implying Parkinson (which assumes
exactly that relationship) will be systematically biased, a prediction Phase 3 confirmed
via a systematically-low Mincer-Zarnowitz slope for every range estimator (0.19-0.45,
well under the ideal 1.0).

**Worked example.** Yang-Zhang's gap term is close to irrelevant for BTC specifically
because perpetual futures trade continuously (see
[why crypto perps have no overnight gap](09-market-data-and-microstructure.md#why-crypto-perps-have-no-overnight-gap))
— notebook 4 measured gap std at 400-2,000x smaller than intrabar std, meaning
Yang-Zhang's extra machinery over Rogers-Satchell buys almost nothing on this specific
instrument, even though it would matter more for a traditional market with real overnight
gaps.

**Pitfalls.** `NEXT_RUN_PROMPT.md`'s own out-of-scope list explicitly rejects a proposed
"range-estimator drift correction" as originally specified, because the proposed
correction was itself a lookahead leak: calibrating a multiplicative bias adjustment from
a full-pre-holdout, fit-once excess measurement and applying it to rolling forecasts
scales bar-$t$'s forecast using data from after $t$ — the correct fix, if ever done,
must estimate the correction on a trailing window only.

---

### The $2\sqrt{2/\pi}$ constant

**In one sentence.** The theoretical ratio between a driftless random walk's typical
high-low range and its close-to-close standard deviation — the specific number Parkinson's
estimator is built around, and the number BTC's actual data falls systematically short
of.

**The maths.** For a driftless Brownian motion over a fixed interval with standard
deviation $\sigma$, the expected high-low range is
$\mathbb{E}[\text{range}] = 2\sqrt{2/\pi}\,\sigma \approx 1.596\,\sigma$. This specific
constant comes from the known distribution of a Brownian bridge's maximum and minimum —
a genuine, derived mathematical fact about random walks, not an empirical rule of thumb.

**Why it is here.** Notebook 4's Phase 1 measured the *observed* ratio of normalized
range to this theoretical prediction and found it **6-14% below 1** at every interval —
i.e. BTC's intrabar range is systematically *smaller* than a pure random walk with the
same close-to-close variance would produce.

**Worked example.** This is read in notebook 4's write-up as evidence of intrabar
mean-reversion or bid-ask-bounce microstructure suppressing the realized high-low spread
relative to what a pure random walk would produce — consistent with the separate
run-length finding (sign runs shorter than a memoryless coin flip predicts).

**Pitfalls.** A range estimator built on this constant (Parkinson specifically) inherits
this bias directly and by construction — any forecast built from it will be
systematically biased low for BTC specifically, which is exactly the Mincer-Zarnowitz
slope-under-1 finding in notebook 4's Phase 3.

---

### Vol targeting

**In one sentence.** A trading approach that scales position size inversely with
forecasted volatility — bigger positions when the model expects calm markets, smaller
positions when it expects turbulence — aiming to keep the *realized risk* of the strategy
roughly constant over time regardless of how volatile the underlying asset currently is.

**The maths.** Position size $\propto 1/\hat{\sigma}_t$ (or $1/\hat{\sigma}_t^2$ for a
variance-based version), where $\hat{\sigma}_t$ is a volatility *forecast*, not a
realized value (using a realized value would be non-causal).

**Why it is here.** Notebook 4's Phase 5(a) was pre-declared as exactly this kind of
strategy — using the Phase 3 ladder's winning variance forecast to scale a buy-and-hold
BTC position — but never ran, because Phase 3 produced no certified point-forecast
winner to scale by. Named here as the concrete reason a real point-forecast winner
matters beyond academic interest: it's a prerequisite for this specific class of
application.

**Worked example.** Notebook 5's own Phase 6 application (gated, likely not run) is
explicitly framed as the *risk-management* analogue of vol targeting rather than an
alpha claim — scaling exposure down when a model's own tail-risk forecast is elevated,
using a *tail-calibration* winner (Gate B) rather than a point-variance winner, since
that is what notebook 5's own contest actually tests for.

**Pitfalls.** Vol targeting only works as well as the underlying volatility *forecast* —
if no forecast beats the trivial baselines with significance (as notebook 4 found for
BTC), a vol-targeting overlay built on any one of the tied candidates is not obviously
better than one built on any other, undermining the whole premise of choosing one
specific "winning" forecast to scale by.
