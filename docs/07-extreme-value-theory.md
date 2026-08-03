# 07 — Extreme value theory

This file assumes [01](01-probability-and-distributions.md)-[06](06-scoring-rules-and-calibration.md).
It covers the mathematics of tails specifically — what happens far from the middle of a
distribution — which is notebook 5's centerpiece and the reason it exists as a separate
notebook from notebook 4's whole-distribution modelling.

---

### Extreme value theory

**In one sentence.** A branch of statistics purpose-built for modeling *rare, extreme*
events specifically — rather than fitting one distribution to an entire dataset and
hoping its tail happens to be realistic, EVT asks "what is the mathematically correct
shape for extreme values, specifically, regardless of what the bulk of the data looks
like?"

**The maths.** Built on two foundational limit theorems (both broadly analogous to the
[central limit theorem](03-statistical-inference.md#central-limit-theorem-and-when-it-fails-under-heavy-tails)'s
role for averages, but for extremes instead): the Fisher-Tippett-Gnedenko theorem (for
block maxima) and the Pickands-Balkema-de Haan theorem (for
[peaks-over-threshold](#block-maxima-vs-peaks-over-threshold)), both saying that, under
broad conditions, extremes converge to one of a small, specific family of limiting
shapes — the [generalized Pareto](01-probability-and-distributions.md#generalized-pareto)
being the peaks-over-threshold answer.

**Why it is here.** This is notebook 5's whole methodological pivot: rather than forcing
one global parametric shape (like Student-t) onto the entire return distribution, EVT
lets the *tail specifically* be modeled by whatever shape the mathematics of extremes
actually implies — motivated directly by notebook 4's finding of fitted Student-t degrees
of freedom sitting at 2-3, right at the edge of a meaningful variance.

**Worked example.** McNeil-Frey [conditional EVT](#conditional-evt-mcneil-frey-two-stage)
is the specific application built in notebook 5: GARCH/GJR handles the *bulk*
(volatility dynamics), and EVT handles the *shape of the tail* of what's left over —
neither piece alone is enough (a GARCH-t forces one global shape onto both tail and body;
an unconditional EVT fit ignores volatility clustering entirely).

**Pitfalls.** EVT's limit theorems are asymptotic — they describe what the tail
*converges to* as you look further and further into it, not an exact statement at any
finite [threshold](#threshold-selection). Choosing how far into the tail counts as "far
enough" for the approximation to be trustworthy is a real, unavoidable judgment call, not
a solved problem — see [mean excess plot](#mean-excess-plot) for the standard diagnostic
used to make that call.

---

### Block maxima vs. peaks-over-threshold

**In one sentence.** Two different ways of defining "what counts as an extreme
observation" for EVT purposes: block maxima takes the single worst observation in each
fixed time block (e.g. the worst day of each month); peaks-over-threshold instead takes
*every* observation that exceeds some fixed high threshold, regardless of when it occurs.

**The maths.** Block maxima: partition data into blocks of size $m$, keep only
$\max$ of each block — this converges (Fisher-Tippett-Gnedenko) to a Generalized Extreme
Value distribution. Peaks-over-threshold: fix a threshold $u$, keep every exceedance $y =
x - u$ for $x > u$ — this converges (Pickands-Balkema-de Haan) to a
[Generalized Pareto](01-probability-and-distributions.md#generalized-pareto) distribution.

**Why it is here.** Notebook 5 uses peaks-over-threshold exclusively (`fit_gpd_tail`),
not block maxima — POT is generally preferred in practice because it uses *every*
genuinely extreme observation above the threshold, rather than discarding all but one per
block (a block could contain several genuinely extreme values, only the single worst of
which block-maxima would keep).

**Worked example.** With a year of daily data, block maxima with monthly blocks keeps
only 12 values total (the worst day of each month) — throwing away information about
the second- and third-worst days even if they were also genuinely extreme. POT with a
10% threshold on the same year keeps roughly 36 values (10% of ~365 days), using more of
the actual tail data available.

**Pitfalls.** POT requires the exceedances to be reasonably independent of each other
(or explicitly modeled if they cluster, as this repo's own volatility-clustering findings
suggest they will) — this is exactly why notebook 5's EVT is applied to *standardized
residuals* from a GARCH/GJR fit rather than raw returns: the volatility clustering has
already been (mostly) removed by the variance model before EVT is applied, so what's
left should be closer to i.i.d.

---

### Threshold selection

**In one sentence.** The choice of exactly where to draw the line between "ordinary
data" and "tail data" for a peaks-over-threshold fit — too low a threshold includes
non-extreme observations that violate the GPD's asymptotic justification; too high a
threshold leaves too few observations to fit a stable model at all.

**The maths.** No single formula; standard practice picks a fixed fraction of the sample
(e.g. the top 10%) or a diagnostic-guided cutoff (see
[mean excess plot](#mean-excess-plot)), trading off bias (too low) against variance (too
high, from too few remaining observations).

**Why it is here.** `NEXT_RUN_PROMPT.md` fixes `tail_frac=0.10` (10%) *in advance* for
notebook 5's `fit_gpd_tail`, explicitly not to be tuned to improve a score, with 5% and
15% reported only as a robustness/sensitivity check — this is a
[pre-declared](08-research-methodology.md#pre-declared-gates-and-pre-registration) choice,
not a free parameter to search over after seeing results.

**Worked example.** On a 500-bar rolling window at 10%, that's 50 exceedances —
`NEXT_RUN_PROMPT.md` explicitly calls this "marginal-but-workable," with `fit_gpd_tail`
requiring at least 30 exceedances before attempting a fit at all (returning `None`
otherwise, the same "null rather than propagate junk" convention as every other fitter
in this repo).

**Pitfalls.** Choosing a threshold *after* seeing which one produces the most favorable
result is a form of the same
[pre-registration](08-research-methodology.md#pre-declared-gates-and-pre-registration)
violation this whole research programme is built to avoid — "10% is the conventional
default and is fixed in advance" is stated explicitly for exactly this reason.

---

### Generalized Pareto with $\xi$ and $\beta$ explained separately

**In one sentence.** The two parameters of a GPD tail fit each have a distinct physical
meaning: $\xi$ (shape) controls how heavy the tail itself is; $\beta$ (scale) controls
the overall size/units of typical exceedances, independent of the tail's shape.

**The maths.** $$f(y \mid \xi, \beta) = \frac{1}{\beta}\left(1 +
\xi\frac{y}{\beta}\right)^{-\frac{1}{\xi}-1}$$
- $\xi$ (Greek "xi") — controls tail *heaviness*: negative means bounded, zero means
  exponential decay, positive means a heavy, power-law-like tail, and larger positive
  values mean a heavier tail still. See
  [what $\xi<0$ etc. mean physically](#what-xi0-xi0-0xi1-xige1-each-mean-physically)
  below for the full breakdown.
- $\beta$ (scale) — sets the typical *magnitude* of exceedances above the threshold,
  analogous to a standard deviation for the tail specifically, but doesn't by itself say
  anything about how the tail's likelihood decays.

**Why it is here.** This is `fit_gpd_tail`'s direct output (`{"xi":..., "beta":...}`),
and `gpd_var_es`'s VaR/ES formulas use both parameters together — $\xi$ determines
*whether* expected shortfall is even finite ($\xi < 1$ required), $\beta$ (along with the
threshold $u$) determines its actual magnitude.

**Worked example.** Two fits with the same $\xi$ but different $\beta$ describe tails of
the same *shape* (equally heavy, relatively speaking) but different absolute scale — like
comparing 1h and 1d returns, whose tails might have similar $\xi$ (both crypto, similar
underlying tail heaviness) but very different $\beta$ (1d moves are simply bigger in
absolute terms).

**Pitfalls.** Don't read $\beta$ alone as "how bad the tail is" — a large $\beta$ with a
small (or negative) $\xi$ can describe a less dangerous tail than a small $\beta$ with a
large $\xi$, because $\xi$ (not $\beta$) determines how fast the *relative* danger grows
as you look further into the tail.

---

### What $\xi<0$, $\xi=0$, $0<\xi<1$, $\xi\ge1$ each mean physically

**In one sentence.** The GPD shape parameter $\xi$ isn't just an abstract fitted number —
each range of its value corresponds to a genuinely different, physically meaningful kind
of tail behavior, from "bounded, can't get worse than some maximum" to "so heavy even the
average of the tail is infinite."

**The maths, each regime:**
- $\xi < 0$: **bounded tail** — exceedances cannot exceed $-\beta/\xi$ (a hard,
  finite ceiling exists). Implausible for financial losses (nothing structurally caps how
  bad a crash can get) — treated in this repo's own tripwires as almost certainly a
  threshold or sign error rather than a genuine finding.
- $\xi = 0$: the boundary case, reducing to an **exponential** tail (memoryless decay,
  same shape as the [exponential distribution](01-probability-and-distributions.md#exponential-distribution)).
- $0 < \xi < 1$: a genuinely **heavy, unbounded, power-law-like** tail, but with a
  finite mean exceedance — the range this repo's tripwires treat as the expected,
  plausible zone for crypto.
- $\xi \ge 1$: the tail is so heavy that **even the mean of the exceedances is
  infinite** — `NEXT_RUN_PROMPT.md` explicitly treats this as "possible in principle at
  1h" (the finest-grained, most extreme interval) rather than automatically an error, but
  asks that it be reported honestly (not silently emitted as NaN) and cross-checked
  against the independent [Hill estimator](#hill-estimator).

**Why it is here.** `gpd_var_es` directly encodes this last boundary: expected shortfall
is only computed (non-NaN) when $\xi < 1$, exactly matching the mathematical fact that
ES is undefined otherwise.

**Worked example.** If notebook 5's Phase 2 finds $\xi \approx 0.3$-$0.5$ at most
intervals but $\xi$ closer to or above 1 specifically at 1h, that pattern would be
directly consistent with notebook 4's own finding of the lowest fitted Student-t degrees
of freedom (heaviest tail) at 1h specifically — cross-checking two independently-derived
tail estimates for internal consistency.

**Pitfalls.** `NEXT_RUN_PROMPT.md` §9's own tripwire: a GPD $\xi$ that wildly disagrees
with the Phase 1 [Hill](#hill-estimator) tail-index estimate on the same data is worth
stopping to investigate — they're estimating related (though not identical) quantities
($\xi \approx 1/\alpha$ where $\alpha$ is the Hill tail index) and should be broadly
consistent if both are working correctly.

---

### Mean excess plot

**In one sentence.** The standard visual diagnostic for choosing a GPD threshold: plot
the *average size of exceedances* above each candidate threshold, against the threshold
itself — if the tail genuinely follows a GPD above some point, this plot should become
roughly a straight line beyond that point.

**The maths.** For candidate threshold $u$, the mean excess function is
$e(u) = \mathbb{E}[X - u \mid X > u]$ — estimated empirically as the sample average of
`(observations above u) - u`. For a true GPD tail with shape $\xi < 1$, $e(u)$ is
*exactly linear* in $u$ above the point where the GPD approximation genuinely holds:
$e(u) = \frac{\beta + \xi u}{1-\xi}$.

**Why it is here.** `NEXT_RUN_PROMPT.md`'s notebook structure (§7) calls this out as a
key required visual — "a mean-excess plot ... the standard EVT threshold-selection
diagnostic — it also visually justifies the fixed 10% [threshold]."

**Worked example.** If the mean excess plot is roughly flat/linear from the 10% tail
cutoff onward but curves noticeably below that point, this visually confirms 10% is a
reasonable, defensible threshold choice — the point where the GPD's own asymptotic
justification actually starts to look like a good approximation to the data.

**Pitfalls.** A mean excess plot that never straightens out at any reasonable threshold
(keeps curving all the way into the most extreme observations) is itself informative —
it would suggest the GPD approximation isn't kicking in cleanly anywhere in the available
data, a genuine limitation worth reporting rather than picking a threshold anyway and
hoping for the best.

---

### Hill estimator

**In one sentence.** A way of estimating a distribution's [tail index](01-probability-and-distributions.md#tail-index)
directly from the largest observed values, *without* assuming any particular parametric
family (like Student-t) for the whole distribution — a genuinely independent
cross-check against a parametric MLE fit's own shape parameter.

**The maths.**
$$\hat{\xi} = \frac{1}{k}\sum_{i=1}^{k} \log\!\left(\frac{X_{(n-i+1)}}{X_{(n-k)}}\right),
\qquad \hat{\alpha} = \frac{1}{\hat{\xi}}$$
using the top $k$ order statistics (the $k$ largest absolute values), $X_{(n-k)}$ being
the threshold (the $(k+1)$-th largest value). Read back in words: average the log-ratio
of each of the top $k$ values to the threshold just below them — a bigger average ratio
means a heavier tail, giving a smaller $\hat{\alpha}$.

**Why it is here.** This is notebook 5's Phase 1a — deliberately independent of the
scipy `t.fit` optimizer, which notebook 4 already caught pinning at its lower search
boundary on the near-degenerate gap series ([boundary solutions](02-estimation-and-fitting.md#boundary-solutions)).
A second, non-MLE estimate of the same underlying quantity is the direct guard against
exactly that failure mode recurring silently.

**Worked example.** If BTC's fitted Student-t $\hat{\nu} \approx 2$ at 1h is a genuine
feature of the data (not an optimizer artifact), the Hill estimator's $\hat{\alpha}$
computed on the same data should land in a broadly similar range — a large disagreement
between the two would be the specific signal that the t-fit's $\nu$ shouldn't be trusted
as a genuine tail-index estimate on its own.

**Pitfalls.** The Hill estimator's own answer depends heavily on the choice of $k$ (how
many top observations to include) — this is exactly why it's never reported as a single
number without a [Hill plot](#hill-plot-and-plateau-reading) showing how stable the
estimate is across a range of $k$ values.

---

### Hill plot and plateau-reading

**In one sentence.** A plot of the Hill estimator's tail-index estimate $\hat{\alpha}$
against the number of top observations $k$ used — a trustworthy estimate shows up as a
flat, stable "plateau" over a wide range of $k$; a plot that keeps drifting as $k$
changes means the estimate is too sensitive to an arbitrary choice to be quoted as a
single number.

**The maths.** Plot $\hat{\alpha}(k)$ for $k$ ranging roughly from 20 to $n/10$. A stable
region where $\hat{\alpha}(k)$ barely changes as $k$ varies is read as the credible
estimate; report the plateau's range, not one arbitrarily-chosen $k$'s single value.

**Why it is here.** `NEXT_RUN_PROMPT.md` is explicit that a Hill plot with **no**
plateau at any interval must be stated plainly, with "every downstream
tail-index-dependent claim [treated] as provisional" — the honesty discipline this whole
research programme insists on, applied specifically to a case where the estimator itself
might simply not give a clean answer on this data.

**Worked example.** A genuinely stable plateau around $\hat{\alpha} \approx 2.2$ for $k$
between, say, 200 and 800 would be reported as "the credible tail-index estimate is
approximately 2.2, stable across this range of $k$" — a materially more defensible claim
than picking one $k$ value and reporting its single point estimate without this context.

**Pitfalls.** Choosing $k$ based on which value gives the most convenient-looking
$\hat{\alpha}$ (rather than reading the plateau honestly) defeats the entire purpose of
the diagnostic — exactly analogous to the threshold-selection pitfall above, and subject
to the same pre-declared-methodology discipline.

---

### Conditional EVT / McNeil-Frey two-stage

**In one sentence.** The specific, standard two-step recipe for combining a volatility
model with EVT: first let GARCH (or GJR-GARCH) handle the time-varying *scale* of
returns; then fit a GPD to the *shape* of the tail of what's left over (the standardized
residuals) — neither piece alone captures both aspects.

**The maths.** Stage 1: fit a conditional variance model, get $\sigma_t$ and
[standardized residuals](#standardized-residuals) $z_t = r_t/\sigma_t$. Stage 2: fit a
GPD to the tail of the $z_t$ series (via
[fit_gpd_tail](01-probability-and-distributions.md#generalized-pareto)). A conditional
quantile forecast is then reconstructed as $\mathrm{VaR}_t(q) = -\sigma_t \cdot z_q$,
combining the time-varying scale from stage 1 with the fixed tail shape from stage 2.

**Why it is here.** This is notebook 5's Phase 2b centerpiece — explicitly motivated as
addressing what neither a GARCH-t (which "forces one global shape onto both tails and
the body") nor an unconditional EVT fit (which "ignores volatility clustering entirely")
can do alone.

**Worked example.** During a genuinely calm period, $\sigma_t$ is small, so even a
moderately-sized raw return translates to a large standardized residual $z_t$ — the GPD
tail fit (on standardized residuals, pooled across both calm and turbulent periods) then
correctly flags this as a real tail event *relative to current conditions*, something a
model using only the raw return's absolute size could miss entirely.

**Pitfalls.** **Critical causality note**, stated explicitly in `NEXT_RUN_PROMPT.md`: the
threshold, $\xi$, and $\beta$ must all be re-estimated on each refit's training window
and forward-filled between refits, exactly like the GARCH variance parameters — fitting
the GPD once on the whole sample would be
[the same lookahead bug as §1a](08-research-methodology.md#lookahead-bias-leakage), one
level deeper (in the tail-shape parameters rather than the degrees-of-freedom parameter).

---

### Standardized residuals

**In one sentence.** What's left of a return after dividing out the model's own
estimate of how volatile that particular bar was expected to be — the "surprise,"
rescaled to a common, comparable unit across bars of very different underlying
volatility.

**The maths.** $z_t = r_t / \sigma_t$, where $\sigma_t$ is a conditional volatility
model's (causal, one-step-ahead) forecast — exactly the same quantity as
[innovations](04-volatility-models.md#innovations) in the GARCH context, given a
specific name here because Phase 2's EVT work fits its GPD directly to this series
rather than to raw returns.

**Why it is here.** This is precisely what the GPD in
[conditional EVT](#conditional-evt-mcneil-frey-two-stage) is fit to, computed **on the
training window only** — never including data from after the point being scored,
matching the causality discipline of every other model in this repo.

**Worked example.** A $-3\%$ return during a historically calm period (small $\sigma_t$)
might standardize to $z_t = -6$ (an extreme standardized residual), while the identical
$-3\%$ return during a turbulent period (large $\sigma_t$) might standardize to only
$z_t = -1.2$ (an unremarkable one) — the whole point of standardizing before fitting the
tail model is to make these two genuinely different situations comparable on a common
scale.

**Pitfalls.** Standardized residuals are only as good as the volatility model producing
$\sigma_t$ — if the underlying GARCH/GJR fit is itself poorly calibrated, the resulting
$z_t$ series will carry that mis-calibration forward into the GPD tail fit; this is
exactly why [Gate C](08-research-methodology.md#frozen-transfer-check) (stability across
symbols) matters for the whole two-stage pipeline, not just for the GPD piece in
isolation.

---

### Spliced (semiparametric) EVT density

**In one sentence.** A full probability density built from three pieces glued
together — a [GPD](#generalized-pareto-with-xi-and-beta-explained-separately) tail on
each side, an empirically-smoothed "body" in the middle — constructed so the total area
under the curve is guaranteed to equal exactly 1, without ever needing to numerically
search for a rescaling constant after the fact.

**The maths.** [Conditional EVT](#conditional-evt-mcneil-frey-two-stage) alone gives a
quantile/ES forecast beyond each threshold, but not a full density — useful for VaR/ES
backtests, useless for a [log score](06-scoring-rules-and-calibration.md#log-score)
contest. The fix: let $k_{\text{lo}}, k_{\text{up}}$ be the number of training-window
observations beyond each threshold (out of $n$ total), and build the density as three
separately-normalized pieces, each scaled by its own **known** weight:

$$f(z) = \begin{cases}
\frac{k_{\text{lo}}}{n} \cdot f_{\text{GPD,lo}}(z) & z < u_{\text{lo}} \\
\left(1 - \frac{k_{\text{lo}}+k_{\text{up}}}{n}\right) \cdot f_{\text{KDE}}(z) & u_{\text{lo}} \le z \le u_{\text{up}} \\
\frac{k_{\text{up}}}{n} \cdot f_{\text{GPD,up}}(z) & z > u_{\text{up}}
\end{cases}$$

where $f_{\text{GPD,lo/up}}$ is each tail's own GPD exceedance density (already a proper
density on its own support) and $f_{\text{KDE}}$ is a Gaussian kernel density estimate on
the training-window "body" observations, itself rescaled by a single `scipy.integrate.quad`
call so it integrates to exactly 1 over $[u_{\text{lo}}, u_{\text{up}}]$. Because
$\frac{k_{\text{lo}}}{n} + \left(1-\frac{k_{\text{lo}}+k_{\text{up}}}{n}\right) +
\frac{k_{\text{up}}}{n} = 1$ and each piece already integrates to 1 over its own support,
the whole spliced density integrates to exactly 1 **by construction** — no post-hoc
rescaling of the spliced whole is ever needed.

**Why it is here.** Notebook 5's own d8/d9 (GARCH-EVT, GJR-EVT) never entered its
log-score contest because normalizing a GPD-tails-plus-empirical-body density proved too
fiddly to trust — "an honest partial entry beats a hand-waved density" was its stated
fallback. `dist_lib6.fit_spliced_evt_density` / `spliced_evt_logpdf`
(`src/results/006_distribution_zoo.md`) fix the *normalization* half of that problem
structurally rather than iteratively — but **not** the *continuity* half: nothing here
forces the density's height to match exactly where the pieces meet, only that each
piece integrates to its own known weight.

**Worked example.** On BTC at 12h, GARCH-EVT and GJR-EVT decisively beat every other
model in this repo's zoo (both non-EVT and the Phase 3 wider zoo) on log score, tied
only with each other — the single cleanest Gate A result this notebook produced. It does
**not** replicate cross-sectionally: on the other five symbols, EVT is rarely even the
single best model, let alone a significant winner — the same "spectacular on BTC alone"
pattern this whole research programme has repeatedly found and repeatedly refused to
over-trust.

**Pitfalls.** Verified numerically that total mass integrates to $\approx 1.0$
(to seven decimal places in a synthetic check) — but the density is **not** perfectly
continuous at the two splice points: a direct check found the relative jump in density
height at each threshold typically runs 20-33% (mean ~0.28 across all fitted refits in
this notebook), not zero. This is a real, quantified, and reported limitation, not a
hidden one — a perfectly smooth splice would need the GPD scale and the KDE bandwidth to
be jointly constrained to match at the boundary, which was judged not worth the
complexity given the explicit timebox on this piece of work. Every log score computed
from this density is still valid (it comes from a genuine, exactly-normalized
probability density) — it is only the *visual smoothness* at the seam that is
approximate.
