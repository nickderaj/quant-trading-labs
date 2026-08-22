# 02 — Estimation and fitting

This file assumes [01](01-probability-and-distributions.md). It covers how you go from
raw observed data to a fitted distribution's parameters, and the specific ways that
process can go wrong — several of which are documented bugs in this repo's own history.

---

### Parameter vs. estimate vs. estimator

**In one sentence.** Three related but distinct ideas: a **parameter** is the true,
unknown number that governs a distribution; an **estimator** is a procedure/formula for
guessing it from data; an **estimate** is the specific number that procedure spits out
for one particular dataset.

**The maths.** If $X \sim \mathrm{Normal}(\mu, \sigma^2)$, $\mu$ is a parameter (a fixed,
unknowable-in-full-precision truth). "Take the sample mean" is an estimator — a rule you
could apply to any dataset. $\bar{x} = \frac{1}{n}\sum x_i$ computed on one specific
sample of data is an estimate, usually written $\hat{\mu}$ (see
[README's notation conventions](README.md#notation-conventions-used-throughout-docs)).

Read back in words: the parameter is the thing you want to know; the estimator is your
method; the estimate is your answer for this particular batch of data. Run the same
estimator on a different sample and you'll generally get a different estimate — that's
[sampling variability](03-statistical-inference.md#standard-error), not a contradiction.

**Why it is here.** Every `_fit_*` function in `distributions.py` is an estimator; every
call to it on a specific rolling window produces one estimate; the true GARCH $\omega$,
$\alpha$, $\beta$ (if such a thing genuinely exists and is even stable over time) is the
parameter none of this repo's code ever actually sees.

**Worked example.** `_fit_normal`'s estimator for $\sigma$ is `np.std(x, ddof=0)` — apply
it to BTC's 1h returns and you get the estimate $\hat{\sigma} \approx 0.00584$ (notebook
4, Phase 1). Apply the same estimator to a different year's worth of BTC returns and
you'd get a different $\hat{\sigma}$ — both are legitimate estimates of possibly-different
underlying volatility, not "one of them is wrong."

**Pitfalls.** Casually saying "the parameter is X" when you mean "the estimate came out
to X" erases the distinction that matters most in this whole research programme: an
estimate always carries uncertainty (see
[confidence interval](03-statistical-inference.md#confidence-interval)), and treating a
single estimate as if it were the exact, known truth is how overconfident claims get
made from noisy fits.

---

### Likelihood

**In one sentence.** A score for "how plausible are these observed data, if this
particular set of parameter values were the truth" — computed by multiplying together
each observation's density under those parameters.

**The maths.** For i.i.d. observations $x_1,\dots,x_n$ and a candidate parameter value
$\theta$: $L(\theta) = \prod_{i=1}^n f(x_i \mid \theta)$ (see [$\prod$](README.md#notation-conventions-used-throughout-docs)).
Read back in words: for each candidate $\theta$, multiply how likely every single
observed data point was under that $\theta$'s implied distribution — a $\theta$ under
which the actual observed data would have been common has a high likelihood; a $\theta$
under which the observed data would have been bizarre has a low one.

**Why it is here.** Likelihood is the object every MLE fit in this repo maximizes (see
[maximum likelihood estimation](#maximum-likelihood-estimation)) — `_garch_negloglik`
computes its *negative log*-likelihood specifically because optimizers in this codebase
minimize by convention.

**Worked example.** Compare two candidate normal fits to the same data: one with
$\hat{\sigma}$ matching the data's actual spread, another with $\hat{\sigma}$ ten times
too small. The second assigns almost-zero density to most of the actual observed values
(they'd be many standard deviations out under that tiny $\hat{\sigma}$) — its likelihood
is astronomically smaller, correctly flagging it as an implausible fit.

**Pitfalls.** Likelihood is *not* a probability that $\theta$ is correct — it's a
function of $\theta$ for **fixed, already-observed** data, going the opposite direction
from "probability of the data given a fixed $\theta$." Conflating the two is one of the
most common conceptual errors in applied statistics.

---

### Log-likelihood

**In one sentence.** The logarithm of [likelihood](#likelihood) — used everywhere in
practice instead of raw likelihood because multiplying many small probabilities together
underflows to numerical zero on a computer, while adding their logs stays well-behaved.

**The maths.** $\ell(\theta) = \log L(\theta) = \sum_{i=1}^n \log f(x_i \mid \theta)$.
Because $\log$ is a strictly increasing function, whatever $\theta$ maximizes $L(\theta)$
also maximizes $\ell(\theta)$ — nothing about the *answer* changes, only the numerical
stability of computing it.

**Why it is here.** `_garch_negloglik` sums log-densities directly (`ll = ...; return
-float(np.sum(ll))`) rather than multiplying raw densities and taking one final log —
exactly this convention, applied consistently everywhere in this repo's from-scratch MLE
code.

**Worked example.** Multiplying 35,000 (BTC 1h's sample size) probability densities each
around 0.01-1 together underflows to exactly 0.0 in floating point long before you reach
the last observation; summing their logs instead stays in a perfectly ordinary numerical
range (a few tens of thousands, negative).

**Pitfalls.** Because optimizers in `scipy.optimize` are built to *minimize*, every MLE
routine in this repo actually minimizes the *negative* log-likelihood — a sign flip that,
if dropped accidentally, silently turns "find the best fit" into "find the worst
possible fit" while still reporting a normal-looking convergence result.

---

### Maximum likelihood estimation

**In one sentence.** The most common way to fit a distribution: search over every
possible parameter value and pick the one that makes the actually-observed data look as
likely as possible.

**The maths.** $\hat{\theta}_{\mathrm{MLE}} = \arg\max_\theta \ell(\theta)$ — "the
$\theta$ that maximizes [log-likelihood](#log-likelihood)." For some families (normal:
sample mean/std) this has a clean closed-form answer; for others (Student-t's degrees of
freedom, GARCH's $\omega,\alpha,\beta$) it requires
[numerical optimization](#numerical-optimization-and-convergence) because no algebra
shortcut exists.

**Why it is here.** Every non-closed-form `_fit_*` in `distributions.py` (`t`, `skewt`)
and every GARCH fit in `dist_lib.py` is MLE. It is the default estimation method
throughout this repo, with [method of moments](#method-of-moments) used only where MLE
is specifically too slow or too unstable on small rolling windows (negative binomial,
Beta).

**Worked example.** `fit_garch11` maximizes exactly this: `scipy.optimize.minimize` on
`_garch_negloglik`, searching over $(\omega, \alpha, \beta)$ (or more parameters, for `t`
/`skewt` innovations) for the combination under which the actually-realized return
sequence has the highest possible joint log-density.

**Pitfalls.** MLE is only as good as the optimizer's ability to actually *find* the
maximum — see [numerical optimization and convergence](#numerical-optimization-and-convergence)
and [boundary solutions](#boundary-solutions) below for the two specific ways this repo
has already caught it failing silently.

---

### Numerical optimization and convergence

**In one sentence.** The algorithmic search process that finds the MLE in cases with no
closed-form shortcut — and the question of whether that search actually finished
successfully or just gave up partway through and returned its last guess.

**The maths.** No single formula; conceptually, an optimizer like `scipy.optimize.fmin`
or `minimize` starts from an initial guess and iteratively moves toward better
[likelihood](#likelihood) values until improvement stalls (declared "converged") or it
hits a limit on iterations/function evaluations (declared "did not converge," though the
optimizer may not always report this honestly).

**Why it is here.** `distributions.py`'s `_mle_fit_with_convergence_check` exists
*specifically* because scipy's own convergence flag can't always be trusted at face
value — it wraps `fmin` with a custom `optimizer=` callback that captures the exit
`warnflag` directly, and independently checks whether the returned parameters are
numerically identical to the starting guess (the exact symptom of an optimizer that
silently returned $x_0$ without ever actually searching).

**Worked example.** A degenerate near-zero-variance window can make a Student-t fit's
likelihood surface almost flat everywhere — `fmin`'s simplex search can stall on such a
surface and report "success" while having barely moved from its starting point. The
convergence check specifically catches this (`np.allclose(x0, xopt, atol=1e-8)`) and
nulls the fit rather than reporting a fake answer.

**Pitfalls.** Never trust an optimizer's own boolean "success" flag alone — check that
the returned parameters actually moved from the initial guess, and (where relevant)
sanity-check the answer against a second, independent estimation method. This exact
principle motivates notebook 5's Hill-estimator cross-check of every fitted tail
parameter.

---

### Boundary solutions

**In one sentence.** When an optimizer's "best" answer lands right at the edge of the
range you told it to search — often a sign that the true best answer is actually outside
that range, or that the data can't really pin the parameter down at all.

**The maths.** If a parameter is constrained to $[\theta_{\min}, \theta_{\max}]$ and the
MLE search converges with $\hat{\theta} = \theta_{\min}$ or $\theta_{\max}$ exactly, the
likelihood was still increasing (or flat) right up to the boundary — the constraint, not
the data, determined the answer.

**Why it is here.** Notebook 4 hit this directly: scipy's `t.fit` on the near-degenerate
`gap_return` series converged to $\hat{\nu} \approx 1.99$ at *every single interval*,
regardless of the true underlying behavior — the search's lower bound artifact, not a
real finding, because the series was so close to a point mass at zero that the shape
parameter was barely identifiable at all. `fit_garch11`'s own bounds (`(2.2, 60)` for
$\nu$ under `"t"` innovation) exist to keep this failure mode visible and boxed rather
than silently drifting to numerical nonsense.

**Worked example.** If you bound $\nu \in [2.2, 60]$ and the fit returns $\hat{\nu} =
2.2$ exactly, that's a boundary solution — worth reporting as "the optimizer wanted to
go lower (toward even heavier tails, or even an ill-defined variance) but was stopped by
the bound," not as "we found $\nu = 2.2$."

**Pitfalls.** A boundary solution can *look* like a perfectly normal converged fit (no
error raised, no obviously wrong number) — this is exactly why notebook 5's Hill
estimator exists as an independent, non-MLE cross-check specifically for tail-related
parameters, per the [tail index](01-probability-and-distributions.md#tail-index) entry.

---

### Method of moments

**In one sentence.** A simpler alternative to MLE: instead of maximizing a likelihood,
just set the distribution's theoretical mean/variance (etc.) equal to the sample's actual
mean/variance and solve algebraically for the parameters.

**The maths.** For a family with known formulas for $\mathbb{E}[X]$ and
$\mathrm{Var}(X)$ in terms of its parameters, plug in the sample mean $\bar{x}$ and
sample variance $s^2$ for those expectations and solve for the parameters directly — no
iterative search, no optimizer, no convergence question.

**Why it is here.** `distributions.py` uses method of moments (not MLE) for both
[negative binomial](01-probability-and-distributions.md#negative-binomial) and
[Beta](01-probability-and-distributions.md#beta-distribution) fits — explicitly because
each family's MLE requires solving an unstable nonlinear equation (a digamma equation for
negative binomial; a poorly-behaved general optimizer near the $[0,1]$ boundary for
Beta) that is both too slow and too unreliable to run at every bar of a rolling fit.

**Worked example.** `_fit_beta`'s formula
`common = mean*(1-mean)/var - 1; a = mean*common; b = (1-mean)*common` is a direct
algebraic solve of the Beta mean/variance equations for $(a,b)$ — no search, exact given
the two sample moments, and it either produces a valid answer or (if `var` is too large
for a valid Beta shape) fails cleanly rather than iterating toward a bad local optimum.

**Pitfalls.** Method of moments can produce nonsensical parameter values (negative shape
parameters, etc.) when the sample moments themselves don't correspond to any valid member
of the family — every method-of-moments `_fit_*` function in this repo checks for exactly
this and returns `None` rather than a nonsense answer.

---

### EM algorithm

**In one sentence.** A specific iterative recipe for maximum likelihood fitting when the
data has a "hidden" component you can't observe directly — like which mixture component
(or hidden state) each data point actually came from.

**The maths.** Alternates two steps until convergence: **E-step** ("expectation") —
given the current parameter guess, compute each observation's *soft* probability of
belonging to each component (its "responsibility"); **M-step** ("maximization") — given
those responsibilities, re-estimate each component's parameters as a responsibility-
weighted version of the ordinary MLE formula. Guaranteed to never decrease the overall
likelihood at each step (though it can get stuck in a local, not global, maximum).

**Why it is here.** `dist_lib.fit_gmm_em`'s core loop is exactly this: `resp` (the
E-step responsibilities) then `means`/`vars_`/`weights` re-estimated as
responsibility-weighted averages (the M-step) — the only estimation method available for
[mixture distributions](01-probability-and-distributions.md#mixture-distribution) since
you never actually observe which component generated which point.

**Worked example.** Fitting a 2-component Gaussian mixture to BTC returns: the E-step
asks "given the current low-vol/high-vol component guesses, how likely is *this specific*
return to have come from the high-vol one?" for every observation; the M-step then
re-estimates each component's mean/variance using those soft assignments as weights, and
the two steps repeat until the weights stop changing meaningfully.

**Pitfalls.** EM converges to *a* local maximum of the likelihood, not necessarily *the*
global one — the starting guess matters. `fit_gmm_em`'s initialization from evenly-spaced
sample quantiles (`np.linspace(0.15, 0.85, k)`) is a specific, deliberate choice to start
components spread across the data's actual range rather than risk two components
collapsing onto the same spot from a bad random start.

---

### Identifiability

**In one sentence.** Whether a model's parameters can, even in principle, be pinned down
uniquely by data — a model is unidentifiable if two genuinely different parameter
settings would produce exactly the same observable behavior, so no amount of data could
ever distinguish them.

**The maths.** A parameter $\theta$ is identifiable if $f(x \mid \theta_1) = f(x \mid
\theta_2)$ for all $x$ implies $\theta_1 = \theta_2$ — distinct parameter values must
imply distinct, distinguishable distributions.

**Why it is here.** The Student-t's $\nu$ becomes poorly identified (not literally
unidentifiable, but practically so) when the data's true variance is near-degenerate —
this is the mechanism behind notebook 4's `gap_return` boundary-solution artifact
([boundary solutions](#boundary-solutions) above): with almost no genuine spread to
inform the shape parameter, wildly different $\nu$ values fit the data almost equally
well, so the optimizer's answer is more about its search path/bounds than the data.

**Worked example.** A 2-component mixture where both components happen to have identical
means and variances is a textbook unidentifiable case — you cannot tell "one component
with weight 1" from "two identical components split 50/50," because they produce the
exact same distribution.

**Pitfalls.** Poor identifiability often masquerades as a *convergence* problem (the
optimizer seems to wander or land on a boundary) when the real issue is that the data
itself cannot distinguish the candidate answers — treating it as "needs a better
optimizer" rather than "needs a structurally different model or more informative data"
wastes effort on the wrong fix.

---

### Label switching

**In one sentence.** A specific identifiability problem in mixture/regime models: the
components (or hidden states) themselves are exchangeable — "component 1" and
"component 2" are arbitrary labels — so two fits of the identical model can assign the
opposite labels to the same underlying pattern.

**The maths.** For a $k$-component mixture, permuting the component labels
$(1,\dots,k) \to (\sigma(1),\dots,\sigma(k))$ for any relabeling $\sigma$ produces an
identical overall distribution — the likelihood genuinely cannot distinguish "component 1
is low-vol, component 2 is high-vol" from the opposite labeling.

**Why it is here.** Directly guarded against in this repo: `fit_gmm_em` and `fit_hmm`
both explicitly impose an ordering (`order = np.argsort(vars_)`, ascending fitted
variance) at the end of every fit, specifically so "component 0" always means "the
low-vol one" across every single rolling refit — without this, a rolling-refit series of
otherwise-correct fits could have "state 0" mean low-vol in one window and high-vol in
the next, making any time-series analysis of state membership meaningless.

**Worked example.** Without label ordering, refit #12 might call the calm regime "state
0" while refit #13 (an otherwise identical fit, just a coincidentally different EM
starting point) calls it "state 1" — any downstream code plotting "probability of state
0 over time" would show a meaningless discontinuity that has nothing to do with the
market and everything to do with an arbitrary labeling flip.

**Pitfalls.** Ascending-variance ordering is a reasonable, simple canonicalization but not
the only possible one — it assumes variance is the axis you care about distinguishing
states by (true for every regime model in this repo, since "vol regime" is exactly what's
being modeled), and would need reconsidering for a model where states differ primarily
along some other axis.

---

### Overparameterization

**In one sentence.** Giving a model more free parameters than the data can actually
support — the fit may look better on the training data by construction, but the extra
flexibility is mostly fitting noise, not signal, and tends to hurt out-of-sample
performance.

**The maths.** No single formula; the general symptom is a model whose number of free
parameters is large relative to the (effective, accounting for
[autocorrelation](03-statistical-inference.md#autocorrelation)) sample size used to fit
it, often flagged via unstable or wildly varying parameter estimates across refits,
or via a formal penalty like AIC/BIC (not used in this repo, but the same underlying
concern).

**Why it is here.** Explicitly why notebook 5's Phase 2 (GJR-GARCH) skips a skew-t
innovation variant: "five shape parameters plus leverage on a 500-bar window is
over-parameterized" — building directly on notebook 4's own finding that a 4-parameter
skew-t bought no calibration improvement over a 3-parameter plain Student-t on the whole,
much larger, full-history sample.

**Worked example.** Notebook 4's [skewed-t](01-probability-and-distributions.md#skewed-t-jones-faddy)
fit (4 parameters) had worse or tied KS calibration than plain Student-t (3 parameters)
at every interval — a textbook overparameterization signature: more flexibility, no
better (often worse) genuine fit, because the extra parameter is absorbing noise rather
than real asymmetry.

**Pitfalls.** A model with more parameters will almost always achieve a *strictly higher*
in-sample likelihood than a nested simpler model (more knobs can only help fit the exact
training data better) — this makes raw likelihood comparison alone misleading; see
[nested models](#nested-models) and [likelihood-ratio test](#likelihood-ratio-test) for
the correct way to test whether the extra flexibility earns its keep.

---

### Nested models

**In one sentence.** Two models are "nested" when the simpler one is a special case of
the more complex one — you can get from the complex model to the simple one by fixing
one or more of its extra parameters to a specific value (often zero).

**The maths.** Model $M_1$ (parameters $\theta$) is nested in model $M_2$ (parameters
$\theta, \phi$) if $M_2$ with $\phi = \phi_0$ reduces exactly to $M_1$.

**Why it is here.** GJR-GARCH (notebook 5's Phase 2) is deliberately built to nest plain
GARCH(1,1) exactly: setting the leverage parameter $\gamma = 0$ in the GJR variance
recursion recovers the ordinary GARCH(1,1) recursion term-for-term — this is what makes
"is there a leverage effect" a well-posed, directly testable question via a
[likelihood-ratio test](#likelihood-ratio-test) on $\gamma = 0$, rather than a vaguer
"which model looks better" comparison.

**Worked example.** Student-t (1 shape parameter, $\nu$) is *not* simply nested inside
skew-t (2 shape parameters, $a,b$) in quite the same one-parameter-to-zero way — skew-t
reduces to a *symmetric* t-like shape only in the limit $a=b$, not at a boundary value of
zero, which is part of why comparing them isn't as clean a nested-model test as GARCH vs.
GJR-GARCH is.

**Pitfalls.** A likelihood-ratio test (below) is only valid between genuinely nested
models — running one between two unrelated families (say, Student-t vs. skew-normal) with
different, non-nested parameterizations does not have the same chi-squared null
distribution and would give a misleading p-value.

---

### Likelihood-ratio test

**In one sentence.** A formal significance test for "does the more complex model's extra
parameter(s) actually earn a real improvement in fit, or could the observed improvement
easily have happened by chance even if the simpler model were true?"

**The maths.** For nested models with the complex model having $d$ more free parameters:
$$\mathrm{LR} = -2\left(\ell_{\text{simple}} - \ell_{\text{complex}}\right)$$
Under the null hypothesis that the simpler model is correct, $\mathrm{LR}$ follows
approximately a [chi-squared distribution](03-statistical-inference.md#test-statistic)
with $d$ degrees of freedom, letting you convert the observed LR into a p-value.

**Why it is here.** Notebook 5's Phase 2 runs exactly this at every GJR-GARCH refit,
testing $\gamma = 0$ (no leverage effect) against the fitted GJR model — "is there a
leverage effect in crypto" becomes a one-number-per-refit answer (the fraction of refits
where this LR test rejects $\gamma=0$ at conventional significance), rather than an
eyeballed comparison of fitted $\gamma$ values.

**Worked example.** If GARCH's log-likelihood on a training window is $-1000$ and GJR's
(nesting it, one extra parameter $\gamma$) is $-995$, $\mathrm{LR} = -2(-1000 - (-995)) =
10$ — compared against a chi-squared distribution with 1 degree of freedom, this would
give a small p-value (roughly 0.0016), i.e. strong evidence the leverage term earns its
keep on that particular window.

**Pitfalls.** The chi-squared approximation degrades when the extra parameter's true
value under the null sits at the *boundary* of its allowed range (e.g. testing $\gamma
\ge 0$ against $\gamma = 0$, a one-sided boundary case) — in that specific situation the
correct reference distribution is a mixture, not a plain chi-squared, and a naive
application understates the true p-value slightly. Worth flagging in any write-up that
reports this test, even where the correction is small in practice.

---

### Rolling / trailing window

**In one sentence.** Instead of fitting a model once to all available history, refit it
repeatedly, each time using only a fixed-size chunk of the *most recent* data — so the
model can adapt as the underlying process changes over time, and each fit only ever sees
data available up to that point.

**The maths.** At time $t$, the trailing window of size $w$ is
$\{x_{t-w+1}, \dots, x_t\}$ — always exactly $w$ observations, always ending at (never
past) $t$.

**Why it is here.** Every forecasting rung in notebooks 4 and 5 (HAR-RV, GARCH, GJR,
GPD) is fit this way, capped at `MLE_MAX_TRAIN = 500` bars in this repo's code — a
deliberate choice bounding both computational cost (an MLE fit on 500 points is fast; on
30,000 it would be slow and would blur together market conditions from years apart) and
staleness (a fit's parameters should reflect recent, not ancient, market behavior).

**Worked example.** `rolling_garch_forecast`'s `window = returns[start:t]` where
`start = max(0, t - max_train)` is a literal trailing window — an MLE fit at refit time
$t$ only ever sees the 500 bars immediately preceding $t$, never anything after it and
never more than 500 bars before it.

**Pitfalls.** A window that's too short can make MLE fits unstable or non-identifiable
(see [identifiability](#identifiability) above); too long re-introduces the staleness
problem the rolling approach exists to avoid, and increases compute cost roughly
linearly. `MLE_MAX_TRAIN=500` and `min_train` (declared in calendar days, see
[refit cadence](#refit-cadence) below) are this repo's specific, stated trade-off.

---

### Refit cadence

**In one sentence.** How often a rolling model is actually re-estimated — not necessarily
every single new observation, since refitting can be computationally expensive and the
underlying process may not change meaningfully bar-to-bar.

**The maths.** Declared in this repo as a number of **calendar days** (e.g. weekly,
monthly), converted to a number of bars via `bars_per_day` for whichever interval is
being run — so the same monthly cadence means refitting every ~30 bars at 1d but every
~720 bars at 1h, keeping the *count* of expensive refits per unit of calendar time
constant across intervals.

**Why it is here.** Explicit design choice, stated up front in every driver script in
this repo: `CHEAP_REFIT_DAYS = 7` (HAR-RV, activity — cheap, closed-form-ish, single
`lstsq` calls) vs. `MLE_REFIT_DAYS = 30` (GARCH, RV-distribution fits — expensive,
iterative MLE). Declaring cadence in calendar time rather than bar count is what keeps
this comparison fair across 1h/4h/12h/1d: a naive "refit every $k$ bars" with fixed $k$
would do 24x more expensive MLE fits at 1h than at 1d for the same wall-clock history.

**Worked example.** At 1h (`bpd=24`), a 30-calendar-day MLE refit cadence means
`mle_refit_every = 30*24 = 720` bars; at 1d (`bpd=1`), the same 30-day cadence means
`mle_refit_every = 30` bars — very different bar counts, same real-world refit
frequency, which is the entire point.

**Pitfalls.** A standing tripwire in this programme: any refit count that differs by
more than about 20% across intervals signals the calendar-day-to-bars conversion is wrong
somewhere — refit *count* (not bar-count cadence) is the invariant that should hold
roughly steady across intervals if this conversion is implemented correctly.

---

### Forward-filling parameters

**In one sentence.** Between two refits, a rolling model's parameters don't change —
whatever was estimated at the last refit stays "in force" and is used to forecast every
bar until the next refit happens.

**The maths.** If refits happen at times $t_1 < t_2 < \dots$, the parameter used to
forecast bar $t$ is whichever $\hat{\theta}_{t_i}$ has the largest $t_i \le t$ — a
step function in time, flat between refit points, jumping only at a refit.

**Why it is here.** This is the causal backbone of every rolling forecast in this repo:
`rolling_garch_forecast`'s between-refit loop explicitly re-rolls the *last-fitted*
model's own variance recursion forward on realized returns, rather than re-fitting or
looking ahead — and notebook 005's causal-parameter-path fix exists
specifically to apply this exact same discipline to the GARCH-t degrees-of-freedom
parameter, which an earlier version of the code had *not* forward-filled correctly (it
used the single *final* fit's $\nu$ to score the *entire* evaluation period instead — a
lookahead leak; see
[lookahead bias / leakage](08-research-methodology.md#lookahead-bias-leakage)).

**Worked example.** `nu_path_from_fits` builds exactly this step function:
`path[f["t"]:] = f["params"][3]` for every fit record `f`, in chronological order, so
later refits' assignments overwrite earlier ones for all bars from that refit point
onward — the value "in force" at bar $t$ is whichever refit most recently happened at or
before $t$.

**Pitfalls.** Confusing "the parameter that was last estimated overall" (which could be
from data *after* the bar being scored) with "the parameter that was in force *at* that
bar" is the exact bug notebook 005's correction addresses. The standing tripwire
for this class of bug is a parameter path that comes back **constant across the whole
sample** when it should visibly step between refits; checking this directly (plotting or
printing the path) is the mechanical way to catch it before it contaminates a scored
result.
