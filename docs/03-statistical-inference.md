# 03 — Statistical inference

This file assumes [01](01-probability-and-distributions.md) and
[02](02-estimation-and-fitting.md). It covers how you decide whether an observed
difference (a rung beating another rung, a fitted parameter differing from zero) is real
signal or could easily be noise — the machinery behind every "significant" or "not
significant" verdict in this research programme.

---

### Null and alternative hypothesis

**In one sentence.** Before running a test, you write down two competing claims: the
"null" (the boring, skeptical default — usually "no real difference/effect") and the
"alternative" (the claim you're actually interested in) — the test's job is to see if the
data gives you enough reason to reject the null in favor of the alternative.

**The maths.** Conventionally $H_0$ (null) and $H_1$ (alternative). E.g. for
[Diebold-Mariano](#diebold-mariano): $H_0$: "the two forecasts have equal expected loss";
$H_1$: "they don't."

**Why it is here.** Every significance verdict in this programme (DM test winners, Kupiec
coverage, the likelihood-ratio test on GJR's $\gamma=0$) is phrased this way: a specific,
pre-declared null being tested against a specific alternative, never a vague "does this
look better."

**Worked example.** Kupiec's test: $H_0$ is "the model's true VaR exceedance rate equals
the declared rate (e.g. 5%)"; a small p-value is evidence against that null, i.e. evidence
the model is miscalibrated.

**Pitfalls.** A test can only ever *reject* or *fail to reject* $H_0$ — it never
"proves" $H_0$ true. "Failed to reject" means "the data didn't give strong enough
evidence against it," not "the null is confirmed" — see
[what a p-value does not mean](#p-value-including-what-it-does-not-mean) below.

---

### Test statistic

**In one sentence.** A single number computed from data, purpose-built so that its
behavior *if the null hypothesis were true* is known (or well-approximated) — comparing
the actually-observed value of this number against that known behavior is what a
significance test does.

**The maths.** Generic notation $T$ (or a specific name: $t$-statistic, $\chi^2$
statistic, KS statistic $D$). The defining property: under $H_0$, the
[distribution](01-probability-and-distributions.md#probability-density-vs-mass-function)
of $T$ is known — e.g. a likelihood-ratio statistic is approximately
$\chi^2$-distributed under its null (see
[likelihood-ratio test](02-estimation-and-fitting.md#likelihood-ratio-test)).

**Why it is here.** `kupiec_test`, `christoffersen_independence_test`, and
`diebold_mariano` all follow this exact pattern: compute a specific number from data
(`lr`, `tstat`), then look up where that number falls in its known null distribution to
get a [p-value](#p-value-including-what-it-does-not-mean).

**Worked example.** Kupiec's LR statistic is built from observed vs. expected exceedance
counts; under the null of correct coverage, it follows a $\chi^2(1)$ distribution —
`st.chi2.sf(lr, df=1)` reads off exactly how extreme the observed LR is relative to that
known reference.

**Pitfalls.** A test statistic's known null distribution is usually an *approximation*
that holds well under specific conditions (large sample size, [i.i.d.](01-probability-and-distributions.md#iid)-ish
data, finite variance) — when those conditions are violated (heavy tails, small samples),
the approximation can be poor, which is exactly notebook 5's Phase 1b concern about
Diebold-Mariano's own CLT-based approximation.

---

### P-value (including what it does not mean)

**In one sentence.** The probability, *if the null hypothesis were exactly true*, of
seeing a test statistic at least as extreme as the one actually observed — a small
p-value means "this data would be a pretty weird coincidence if the null were true."

**The maths.** $p = P(T \ge t_{\mathrm{observed}} \mid H_0)$ (one-sided; two-sided tests
use $P(|T| \ge |t_{\mathrm{observed}}|\mid H_0)$).

**Why it is here.** Every significance verdict in this repo is gated on a pre-declared
p-value threshold, most often $p < 0.05$ — this is the number every DM test, Kupiec test,
and Christoffersen test in `distributions.py` ultimately returns.

**Worked example.** A DM test p-value of 0.03 for "rung A beats rung B" means: if rung A
and B truly had identical expected loss, you'd see a loss difference this large (or
larger) only about 3% of the time by chance alone — evidence against "no real
difference," reported at the pre-declared $\alpha=0.05$ threshold as significant.

**Pitfalls — what a p-value does *not* mean, stated explicitly because this is the single
most common misinterpretation in applied statistics:**
- It is **not** "the probability the null hypothesis is true." $P(T \ge t \mid H_0)$ and
  $P(H_0 \mid T = t)$ are different quantities (confusing them is exactly the logical
  error of reversing a conditional probability — see [$\mid$](README.md#notation-conventions-used-throughout-docs)).
- It is **not** "the probability the result happened by chance," stated that loosely.
- A large p-value does **not** mean the null is confirmed — it means the data didn't
  give strong evidence against it, which could be because the null is true, or because
  the test had low [power](#power), or the sample was too small.
- A tiny p-value on a *practically meaningless* effect size (a real but economically
  trivial difference) is still just evidence the effect is non-zero, not evidence it
  matters — this repo repeatedly separates "statistically significant" from "actually
  useful" (e.g. notebook 4's Phase 3 R² of 0.004-0.19: some rungs are significantly
  better than others while all still explain very little of the actual variance).

---

### Significance level

**In one sentence.** The p-value threshold you commit to, *before* seeing the results,
below which you'll call a result "significant" — the pre-declared bar, not something
chosen after looking at the data to make a result look good.

**The maths.** Conventionally $\alpha$ (Greek letter "alpha" — note this is an unrelated
reuse of the same letter as the [tail index](01-probability-and-distributions.md#tail-index)
$\alpha$; context always disambiguates). Reject $H_0$ if $p < \alpha$.

**Why it is here.** $\alpha = 0.05$ is used throughout this repo's gates (Gate A, Gate B
in `NEXT_RUN_PROMPT.md`) — declared in advance, in writing, before any model is run,
exactly matching [pre-declared gates and pre-registration](08-research-methodology.md#pre-declared-gates-and-pre-registration).

**Worked example.** Gate B (tail calibration) requires a model to survive **36** separate
tests per interval (6 quantile levels x 3 tests) all at $\alpha=0.05$ — a genuinely
strict bar precisely because it's checked so many times; see
[multiple-testing problem](#multiple-testing-problem) for why that repetition itself
needs separate handling.

**Pitfalls.** Choosing $\alpha$ *after* seeing which threshold makes your preferred
result "significant" defeats the entire purpose of pre-declaring it — this is a
specific, named form of the broader
[pre-registration](08-research-methodology.md#pre-declared-gates-and-pre-registration)
discipline this research programme insists on.

---

### Type I/II error

**In one sentence.** The two ways a significance test can be wrong: a Type I error is a
false alarm (rejecting a true null — concluding there's an effect when there isn't one);
a Type II error is a miss (failing to reject a false null — missing a real effect that's
actually there).

**The maths.** $P(\text{Type I error}) = P(\text{reject } H_0 \mid H_0 \text{ true})$ —
exactly what the [significance level](#significance-level) $\alpha$ controls by
construction. $P(\text{Type II error}) = P(\text{fail to reject } H_0 \mid H_0 \text{
false})$, conventionally called $\beta$; $1-\beta$ is [power](#power).

**Why it is here.** The [multiple-testing problem](#multiple-testing-problem) below is
entirely about controlling the *rate* of Type I errors when running many tests at once —
directly relevant to notebook 5's 45-pairs-x-4-intervals density contest and 36-test
coverage battery per interval.

**Worked example.** Running 180 independent tests at $\alpha=0.05$, if every null were
exactly true, you'd expect about $180 \times 0.05 = 9$ Type I errors (false "significant"
results) purely by chance — exactly the number `NEXT_RUN_PROMPT.md` cites as the reason
notebook 5 applies a
[Benjamini-Hochberg FDR](#benjamini-hochberg-fdr) correction to its density contest.

**Pitfalls.** Reducing Type I errors (a stricter $\alpha$) generally increases Type II
errors (harder to detect real effects) for a fixed sample size — this is a genuine
trade-off, not a free lunch; a stricter multiple-testing correction protects against
false positives at the cost of some statistical power to detect real ones.

---

### Power

**In one sentence.** The probability a test correctly detects a real effect, when one
genuinely exists — a high-power test rarely misses true effects (low Type II error
rate); a low-power test often does.

**The maths.** $\mathrm{Power} = 1 - \beta = P(\text{reject } H_0 \mid H_0 \text{ false})$.
Power generally increases with sample size, with the true effect's size, and decreases
as $\alpha$ is made stricter.

**Why it is here.** `NEXT_RUN_PROMPT.md` §9's own tripwire flags any coverage test with
fewer than ~10 violations as "underpowered" (report it as such rather than quoting its
p-value as meaningful) — a direct, explicit acknowledgment that a test run on too few
events can't reliably detect a real miscalibration even if one exists.

**Worked example.** At 1d, only ~1,460 pre-holdout bars exist, implying ~14 expected 1%
VaR violations — enough for Kupiec (which only needs a violation *count*) but thin for
Christoffersen independence (which needs enough *consecutive pairs* of violations to say
anything about clustering) — exactly the distinction `NEXT_RUN_PROMPT.md` draws between
the two tests at that quantile level.

**Pitfalls.** A non-significant result from a low-power test is not evidence of "no
effect" — it may simply mean the test never had a realistic chance of detecting the
effect at that sample size. Always report the number of events feeding a test (as this
repo's coverage battery does via `n_violations`) so a reader can judge power for
themselves.

---

### Confidence interval

**In one sentence.** A range of plausible values for an unknown parameter, built so that
if you repeated the whole estimation process many times on fresh samples, the interval
would contain the true value some declared fraction of the time (e.g. 95%).

**The maths.** A 95% confidence interval $[\hat{\theta}_{\mathrm{lo}},
\hat{\theta}_{\mathrm{hi}}]$ has the property that, across many hypothetical repeated
samples, $P(\theta \in [\hat{\theta}_{\mathrm{lo}}, \hat{\theta}_{\mathrm{hi}}]) = 0.95$.

**Why it is here.** Notebook 5's Hill estimator is reported with a bootstrapped
confidence interval, not a single point value — because a single tail-index estimate can
be noisy, and reporting a range communicates that noise honestly rather than implying
false precision.

**Worked example.** If a Hill-estimated tail index $\hat{\alpha}$ comes back as 2.3 with
a 95% CI of $[1.8, 3.1]$, that CI *includes 2* — meaning the data cannot confidently rule
out the boundary case where variance itself may not exist, a materially different (and
more honest) statement than reporting "$\hat{\alpha} = 2.3$" alone.

**Pitfalls.** The single most common misinterpretation: "there's a 95% probability the
true value is inside this specific interval." Once an interval is computed from one
specific dataset, the true parameter either is or isn't inside it — the 95% describes the
*procedure's* long-run reliability across repeated sampling, not a probability statement
about this one already-computed interval.

---

### Standard error

**In one sentence.** A measure of how much an estimate would bounce around if you
re-ran the same estimation procedure on a fresh sample of the same size — the building
block for confidence intervals and t-statistics.

**The maths.** $\mathrm{SE}(\hat{\theta}) = \sqrt{\mathrm{Var}(\hat{\theta})}$, i.e. the
standard deviation of the *estimator itself*, treated as a random variable across
hypothetical repeated samples — not to be confused with the standard deviation of the
underlying *data*.

**Why it is here.** Every t-statistic and DM-test in this repo divides an estimated
difference by its standard error — the entire logic of "is this difference big relative
to how much it could plausibly bounce around by chance" runs through this quantity.

**Worked example.** For a simple mean of $n$ i.i.d. observations with standard deviation
$\sigma$, $\mathrm{SE}(\bar{x}) = \sigma/\sqrt{n}$ — larger samples shrink the standard
error, which is why DM tests (thousands of scored bars) can detect much smaller real
differences than a 3-fold backtest could.

**Pitfalls.** The plain $\sigma/\sqrt{n}$ formula assumes i.i.d. data — it *understates*
the true standard error when observations are autocorrelated (each new observation adds
less genuinely new information than an i.i.d. one would), which is exactly why this repo
uses [Newey-West / HAC standard errors](#newey-west-hac-standard-errors) for DM tests
instead of the naive formula.

---

### T-statistic

**In one sentence.** An estimate divided by its own standard error — a unitless number
answering "how many standard errors away from zero (or from some other reference value)
is this estimate?"

**The maths.** $t = \frac{\hat{\theta} - \theta_0}{\mathrm{SE}(\hat{\theta})}$, most
commonly testing $\theta_0 = 0$ ("is this estimate significantly different from zero").
Under standard conditions, $t$ follows (approximately, for large samples) a standard
normal or Student-t reference distribution, from which a
[p-value](#p-value-including-what-it-does-not-mean) is read off.

**Why it is here.** `diebold_mariano` computes exactly this: a HAC-adjusted t-statistic
on the mean loss differential, converted to a two-sided p-value via
`2 * st.norm.sf(abs(tstat))`.

**Worked example.** `dist_lib.py`'s own commit history documents a real bug here:
`diebold_mariano` originally unpacked `research.newey_west_tstat`'s return value
backwards, silently reporting the loss-differential *series mean* where the actual
t-statistic belonged — caught by noticing reported "p-values" outside $[0,1]$ (a
mislabeled mean can easily be far from the $[0,1]$ range a real p-value must fall in).

**Pitfalls.** A t-statistic's associated p-value calculation assumes the statistic
actually follows its claimed reference distribution (normal, in this case) — this is
precisely the assumption notebook 5's Phase 1b tests directly against a
[bootstrap](#bootstrap) alternative, because heavy-tailed loss differentials can make the
normal approximation to the t-statistic's distribution unreliable.

---

### Autocorrelation

**In one sentence.** How correlated a time series is with a lagged (shifted-in-time)
version of itself — positive autocorrelation means a high value tends to be followed by
another high value soon after; this is the mathematical signature of "clustering" or
"persistence" over time.

**The maths.** At lag $k$: $\rho_k = \mathrm{Corr}(x_t, x_{t-k})$, ranging from $-1$
(perfectly anti-persistent) to $1$ (perfectly persistent), $0$ meaning no linear
relationship between values $k$ steps apart.

**Why it is here.** Volatility itself is strongly autocorrelated (this is literally what
"[volatility clustering](04-volatility-models.md#volatility-clustering)" means in time-series
language) — which is why plain i.i.d.-assuming standard errors are wrong for anything
built on volatility or its forecast errors, motivating
[HAC standard errors](#newey-west-hac-standard-errors) throughout this repo.

**Worked example.** DM loss-differential series in this repo are autocorrelated because
the underlying forecast errors are (a GARCH forecast that's too low today is likely still
too low tomorrow, since it hasn't refit yet) — exactly why `diebold_mariano` uses a HAC
correction rather than a naive standard error.

**Pitfalls.** High autocorrelation shrinks the *effective* sample size well below the
raw observation count — a series of 10,000 highly autocorrelated observations may carry
genuinely less independent information than 10,000 truly independent ones, which is
exactly why HAC standard errors (and the block bootstrap) exist rather than treating
autocorrelated data as if it were i.i.d.

---

### Heteroskedasticity

**In one sentence.** When a variable's variance (or spread) is *not* constant — it
changes depending on time, or on some other variable — the opposite of
"homoskedasticity" (constant variance), and exactly what GARCH exists to model.

**The maths.** For a time series, homoskedasticity means $\mathrm{Var}(r_t) = \sigma^2$
(the same for every $t$); heteroskedasticity means $\mathrm{Var}(r_t \mid \text{past}) =
\sigma_t^2$, a quantity that genuinely changes over time.

**Why it is here.** This is the entire premise of every volatility model in
[04](04-volatility-models.md) — "conditional heteroskedasticity" (changing variance,
conditional on recent history) is what GARCH is an acronym for
(**G**eneralized **A**uto**R**egressive **C**onditional **H**eteroskedasticity).

**Worked example.** Notebook 4's own regime models directly demonstrate this: a
"high-vol" and "low-vol" state with genuinely different variances is heteroskedasticity
made explicit and discrete, rather than smoothly time-varying as in GARCH.

**Pitfalls.** Applying a method that assumes homoskedasticity (a plain standard error
formula, an unweighted least-squares regression's usual inference) to genuinely
heteroskedastic financial data understates uncertainty during high-variance periods and
overstates it during calm ones — one specific reason
[Newey-West / HAC standard errors](#newey-west-hac-standard-errors) correct for
heteroskedasticity as well as autocorrelation (the "H" in HAC stands for exactly this).

---

### Newey-West / HAC standard errors

**In one sentence.** A way of computing a standard error that stays valid even when the
underlying data is autocorrelated and/or heteroskedastic — the two violations of the
plain i.i.d. assumption that are essentially guaranteed in financial time series.

**The maths.** "HAC" = **H**eteroskedasticity and **A**utocorrelation **C**onsistent.
Rather than the plain $\sigma/\sqrt{n}$ formula, a HAC standard error weights in
covariances between nearby observations (up to some maximum lag), correcting for the
fact that autocorrelated data provides less independent information per observation than
i.i.d. data of the same size would.

**Why it is here.** `research.newey_west_tstat`, called by `dist_lib.diebold_mariano`
with `lag = max(1, int(round(n ** (1/3))))` (a standard rule-of-thumb lag choice scaling
with sample size), is the HAC correction underlying every DM test's t-statistic in this
repo.

**Worked example.** Without a HAC correction, a DM test on autocorrelated loss
differentials would report an artificially small standard error (because it wrongly
treats each autocorrelated observation as fully independent new information), making
real ties look like significant differences — exactly the over-confidence failure mode
this correction exists to prevent.

**Pitfalls.** HAC standard errors still rely on an underlying asymptotic (large-sample)
normal approximation — they fix the autocorrelation/heteroskedasticity problem but not a
heavy-tailed-loss-differential problem, which is a separate concern notebook 5's Phase 1b
checks directly against a [block bootstrap](#block-stationary-bootstrap) alternative.

---

### Kolmogorov-Smirnov test

**In one sentence.** A test of whether a sample of data plausibly came from some specific
claimed distribution, based on the biggest gap between the sample's own cumulative
frequency and the claimed distribution's CDF.

**The maths.** The KS statistic is $D = \sup_x |F_n(x) - F(x)|$, where $F_n$ is the
sample's empirical CDF (the fraction of the sample $\le x$) and $F$ is the claimed
distribution's theoretical [CDF](01-probability-and-distributions.md#cdf). A large $D$
(and correspondingly small p-value) rejects "this family's fit is well-calibrated."

**Why it is here.** `distributions.py`'s `pit_ks_test` runs exactly this, on
[PIT values](06-scoring-rules-and-calibration.md#pit) against Uniform(0,1) — and it's the
test behind every "normal rejected, t not rejected" verdict in notebook 4's Phase 1
calibration table.

**Worked example.** Notebook 4 found the normal's KS statistic around 0.08-0.11 at every
interval (p effectively 0 — strongly rejected), while Student-t's KS statistic dropped
to 0.007-0.027 (not rejected at several intervals, most cleanly at 1d, p=0.231) — the
direct quantitative basis for "t is the best simple parametric fit found here."

**Pitfalls.** The classic KS test assumes the reference distribution's parameters are
known in advance, not estimated from the same sample being tested — using parameters
fit to the same data (as this repo does, for practicality) makes the test's own p-values
somewhat *anti-conservative* (a bit too likely to fail to reject), a known, accepted
approximation in this kind of applied calibration check rather than an oversight.

---

### Kruskal-Wallis

**In one sentence.** A non-parametric test for whether several groups have the same
underlying distribution of some outcome — like a one-way ANOVA, but based on *ranks*
rather than raw values, so it doesn't assume the groups are normally distributed.

**The maths.** Converts all observations (across all groups) to their overall rank, then
compares the average rank within each group; under the null (all groups drawn from the
same distribution), the resulting statistic follows approximately a $\chi^2$ distribution
with (number of groups $- 1$) degrees of freedom.

**Why it is here.** Notebook 4's Phase 4 uses Kruskal-Wallis to test "does the regime
state predict next-bar volatility" — comparing the distribution of realized volatility
across the (2 or 3) regime states, without assuming volatility itself is normally
distributed within each state (which it plainly isn't, per Phase 1's own findings).

**Worked example.** Notebook 4 reports Kruskal-Wallis p-values from $10^{-4}$ down to
effectively 0 for "does regime state predict volatility" at every interval and every
model — overwhelming evidence the states genuinely differ in typical volatility, which is
close to definitional for a volatility-based regime model, but confirmed directly rather
than assumed.

**Pitfalls.** Using rank-based tests avoids the normality assumption but can lose some
statistical power relative to a parametric test *if* the data really were normal — a
worthwhile trade given how badly the normality assumption fails for this data (per every
Phase 1 fat-tail finding).

---

### ANOVA

**In one sentence.** "Analysis of variance" — a classical test for whether several
groups have different *means*, by comparing how much values vary *between* groups
against how much they vary *within* groups.

**The maths.** F-statistic $= \frac{\text{between-group variance}}{\text{within-group
variance}}$; under the null (all groups share the same true mean), this follows an
F-distribution. Large F (between-group differences large relative to natural
within-group noise) rejects the null of equal means.

**Why it is here.** Notebook 4's Phase 4 uses ANOVA to test "does regime state predict
next-bar *direction*" (mean return), as the natural mean-comparison complement to
Kruskal-Wallis's rank-based variance comparison.

**Worked example.** Notebook 4's direction-prediction ANOVA p-values scatter around and
above 0.05 with no consistent pattern across intervals/models — the basis for the
write-up's "regimes predict risk, not return" conclusion, with a couple of marginal
exceptions explicitly noted as consistent with chance alone given the number of tests
run.

**Pitfalls.** ANOVA assumes each group's values are roughly normally distributed and have
similar variance (homoskedastic) — return *means* per regime aren't obviously
non-normal in the same extreme way raw returns are (averaging tends to reduce skew/
kurtosis somewhat, per
[aggregational Gaussianity](01-probability-and-distributions.md#aggregational-gaussianity)),
so ANOVA is a defensible choice here even though Kruskal-Wallis was preferred for the
volatility comparison specifically.

---

### Boundary likelihood-ratio test

**In one sentence.** An ordinary [likelihood-ratio test](02-estimation-and-fitting.md#likelihood-ratio-test)
compares a simpler ("null") model against a more flexible one that contains it as a
special case — but when that special case sits right at the *edge* of the flexible
model's allowed parameter range rather than somewhere in its interior, the usual
chi-squared reference distribution is wrong, and using it anyway roughly halves the
p-value, overstating significance.

**The maths.** The ordinary LR test statistic $LR = 2(\ell_{\text{full}} -
\ell_{\text{null}})$ is compared against a $\chi^2_1$ distribution when the null is one
interior value of one extra parameter. Chernoff's (1954) boundary result: when the null
value sits exactly on the *edge* of the full model's allowed range (so the parameter
literally cannot go past it in the "more extreme" direction), the statistic's null
distribution is instead a 50:50 **mixture** of a point mass at 0 and a $\chi^2_1$. The
correct p-value is $p = 0.5 \times P(\chi^2_1 \ge LR)$, not the plain $P(\chi^2_1 \ge
LR)$ — exactly half the naive number, because half of the mixture's probability mass
sits at exactly 0 and never contributes to the upper tail at all.

**Why it is here.** `dist_lib6.fit_nb_counts` fits a negative binomial's dispersion
parameter $\alpha \ge 0$ (Poisson is the $\alpha=0$ boundary — variance cannot be *less*
than the Poisson mean, so $\alpha$ cannot go negative); `dist_lib6.boundary_lr_test`
applies the 0.5-mixture correction when testing NB against the Poisson null in Phase 4's
violation-count analysis (`src/results/6_distribution_zoo.md`). The companion
discrete-Weibull-vs-geometric comparison ($\beta=1$) is **not** this case — $\beta=1$ is
an *interior* point of $\beta>0$'s range (a duration hazard can rise or fall from
$\beta=1$ in either direction), so `dist_lib6.fit_discrete_weibull_durations` uses a
plain, uncorrected $\chi^2_1$ test instead. Getting this distinction backwards in either
direction is the "classic error" this entry exists to prevent.

**Worked example.** A synthetic check (Poisson-generated counts, so the null is exactly
true): the boundary-corrected p-value for a modest LR statistic comes out close to 0.5,
matching "no real evidence of overdispersion" — using the uncorrected $\chi^2_1$ p-value
on the same statistic would have reported roughly half that value, a meaningfully
different-looking (though still non-significant, in this particular check) result.

**Pitfalls.** The 50:50 mixture applies specifically to *one* extra parameter pinned at
a boundary. Testing two or more boundary parameters simultaneously needs a different
(more complex) mixture and is out of scope here — Phase 4's NB-vs-Poisson comparison is
safely the one-parameter case.

---

### Diebold-Mariano

**In one sentence.** The standard test in forecasting research for "does forecast A
have significantly lower average loss than forecast B" — built directly on the loss
*differential* series (A's loss minus B's, bar by bar), not on the raw forecasts
themselves.

**The maths.** Given a loss-differential series $d_t = L(\text{actual}_t, \text{forecast
A}_t) - L(\text{actual}_t, \text{forecast B}_t)$, DM tests $H_0: \mathbb{E}[d_t] = 0$
using a [HAC](#newey-west-hac-standard-errors) [t-statistic](#t-statistic) on $\bar{d}$
(a negative mean loss differential with $p<0.05$ means A is significantly better).

**Why it is here.** `dist_lib.diebold_mariano` is the workhorse behind every "does rung
X beat rung Y" verdict in notebook 4's Phase 3, and behind notebook 5's log-score
all-pairs contest (Gate A).

**Worked example.** Notebook 4's all-pairs DM apparatus tests all $\binom{7}{2}=21$
pairs of ladder representatives, not just adjacent-rung comparisons, specifically because
"beats the rung directly below" is not [transitive](README.md) from adjacent comparisons
alone — rung5 beating rung4 and rung3 beating rung4 says nothing about rung5 vs. rung3
directly, so every pair must be tested.

**Pitfalls.** DM's own significance calculation rests on the loss differential's mean
being approximately normally distributed by the [central limit theorem](#central-limit-theorem-and-when-it-fails-under-heavy-tails)
— which can fail when the underlying loss differential is itself heavy-tailed, exactly
the concern notebook 5's Phase 1b investigates directly by comparing DM's normal-approximation
p-value against a [block bootstrap](#block-stationary-bootstrap) p-value on the same
series.

---

### Bootstrap

**In one sentence.** A way of estimating how much an estimate would vary across
different samples, *without* needing a formula for its standard error — done by
repeatedly re-sampling (with replacement) from the data you already have and
recomputing the estimate each time, treating the spread of those repeated estimates as a
stand-in for genuine sampling variability.

**The maths.** From an original sample of size $n$, draw a new sample of size $n$ *with
replacement* (so some original points appear multiple times, others not at all),
recompute the statistic of interest on this resample, and repeat many times (hundreds to
thousands). The distribution of the resulting statistic values approximates the true
sampling distribution of the estimator.

**Why it is here.** Used throughout this programme wherever a formula-based standard
error is unreliable or unavailable — notebook 5's Hill-estimator confidence interval and
its DM-validity check (comparing a bootstrap p-value against DM's normal-approximation
p-value) both lean on this.

**Worked example.** To get a confidence interval for a fitted tail index, resample the
returns 1,000 times, re-run the Hill estimator on each resample, and take the 2.5th and
97.5th percentiles of the resulting 1,000 estimates as a 95% confidence interval — no
formula for the Hill estimator's own sampling distribution is needed.

**Pitfalls.** Plain (i.i.d.) bootstrapping — resampling individual points independently —
destroys any autocorrelation structure in the original data, which is exactly wrong for
time series where nearby observations are genuinely dependent; see
[block/stationary bootstrap](#block-stationary-bootstrap) for the fix used throughout
this repo.

---

### Block / stationary bootstrap

**In one sentence.** A bootstrap variant built for time series: instead of resampling
individual points, resample whole contiguous *blocks* of consecutive observations, so
whatever autocorrelation exists within a block is preserved in the resample.

**The maths.** Choose a block length $\ell$ (fixed, or — in the "stationary" bootstrap
variant — random with a chosen mean length); build each resample by concatenating
randomly-chosen blocks of that (roughly) length until reaching the original sample size.

**Why it is here.** `research.block_bootstrap_ci`, used throughout notebook 5 (Hill
confidence intervals, GPD sensitivity checks) specifically because returns are
autocorrelated *in magnitude* even when not autocorrelated in sign — per
`NEXT_RUN_PROMPT.md`'s own framing of exactly why this matters here.
`research.block_bootstrap_pvalue` is the same block-resampling machinery aimed at a
hypothesis test instead of a CI: shift the data so its mean is exactly the null value,
resample, and see how extreme the *actual* observed mean is against that null
resampling distribution — this is exactly Phase 1b's DM-validity check, comparing
this bootstrap p-value against Diebold-Mariano's own normal-approximation p-value on
the same QLIKE loss-differential series.

**Worked example.** Return magnitudes cluster in time (volatility clustering) — an
ordinary bootstrap that shuffles individual return values independently would understate
how much a real fitted quantity (like a tail index or a DM t-stat) could plausibly vary
across genuinely different historical periods, because it breaks up the very clustering
that makes consecutive extreme values *not* independent pieces of evidence.

**Pitfalls.** The choice of block length is itself a trade-off: too short and the
resample still doesn't preserve enough of the real dependence structure; too long and
you have effectively too few independent blocks to build a meaningfully varied set of
resamples. Report the block length used, and treat the resulting CI as an approximation
whose quality depends on that choice, not an exact answer.

---

### Paired block bootstrap and why pairing on shared blocks matters

**In one sentence.** When comparing two trading strategies (or forecasts) built on the same historical price path, resample the *same* calendar blocks for both in each bootstrap draw — so their shared market noise cancels in the difference, leaving only the genuine configuration disagreement between them, and producing a much narrower confidence interval than an unpaired bootstrap would.

**The maths.** Extend [block bootstrap](#block-stationary-bootstrap) to a paired setup: given control book's trades with block IDs $\{b_i\}$ and treatment book's trades with block IDs $\{b_j\}$, resample the union of all unique block IDs once per bootstrap draw, evaluate *both* books on the *same* resampled blocks each draw, and compute the delta as (treatment sum — control sum). The delta's confidence interval relies on the fact that the shared blocks' contributions cancel: $\mathbb{E}[\text{delta}] = \mathbb{E}[\text{treatment on shared blocks}] - \mathbb{E}[\text{control on shared blocks}]$, not the independent sums each alone would have produced.

**Why it is here.** Notebook 11a's Phase 2 evaluation harness (`src/research/tmp/spread_lib11.py`, `paired_block_bootstrap`) tests whether a configuration change (a stop-loss enabled vs. disabled, a threshold adjusted, etc.) on the *same* spread data produces a real P&L difference, as distinct from "which is better on some other data." An unpaired bootstrap that resamples each book's P&L independently throws away the fact that both P&Ls are driven by the same market shock series in the same historical windows — the shared noise gets double-counted as if it were independent uncertainty in each book separately, inflating the comparison's CI needlessly.

**Worked example.** Synthetic example with quarterly blocks: a control book and treatment book (differing only in one parameter) trade the same underlying spread over 8 quarters. Both books' P&L is dominated by a massive market dislocation in Q3, which the control book turned into a loss (−$100,000) and the treatment book turned into a gain (+$80,000). An unpaired bootstrap resamples the control book's quarters independently and the treatment book's quarters independently — so Q3 sometimes appears in the control sample (contributing −$100k to the control draw) and sometimes doesn't, and separately Q3 sometimes appears in the treatment sample (contributing +$80k to the treatment draw) and sometimes doesn't. A draw that excludes Q3 from both gives a delta of roughly zero; a draw that includes Q3 in control only gives a delta of +$100k; a draw that includes Q3 in treatment only gives a delta of −$80k; a draw that includes Q3 in both gives a delta of +$180k. The unpaired resampling creates huge artificial swings (+$100k to −$80k, roughly ±$100k range from the Q3 noise alone, which neither book actually experienced that way). A paired bootstrap resamples the quarter IDs *once*, so Q3 either appears in both books' samples (delta +$180k, the true Q3 effect) or in neither (delta 0). Every draw either includes or excludes Q3 consistently across both books, so the Q3 noise cancels in the delta and never inflates the interval. Numerically: notebook 11a's actual demo on synthetic data shows a paired CI width of ±$1,751 vs. an unpaired CI width of ±$294,967 — a ~169x narrower interval, because pairing lets the shared market noise cancel.

**Pitfalls.** Pairing only works when the two books genuinely trade the same underlying price path over the same time periods — if one book skips some dates or uses a different instrument (even a related one), the block structure differs and the pairing assumption breaks. Also, a narrow CI on (treatment − control) tells you whether the configuration difference is statistically real; it says nothing about which configuration is *correct* — a narrow delta CI just means the data gave a precise answer to "did this parameter change matter," not an endorsement of the treatment over the control or vice versa.

---

### Multiple-testing problem

**In one sentence.** If you run many significance tests at once, some will come back
"significant" purely by chance, even if nothing real is going on anywhere — the more
tests you run, the more such false positives you should expect, unless you correct for
it.

**The maths.** If $m$ independent tests are each run at significance level $\alpha$, and
every null is actually true, the *expected number* of false "significant" results is
$m\alpha$ — not zero.

**Why it is here.** Notebook 5's Phase 3 density contest runs $\binom{10}{2}=45$ pairwise
comparisons at each of 4 intervals — 180 tests total. `NEXT_RUN_PROMPT.md` states this
explicitly: at $\alpha=0.05$, about 9 spuriously "significant" results are expected by
chance alone if nothing were really different between models, which is why Gate A's
determination is made on
[Benjamini-Hochberg-adjusted](#benjamini-hochberg-fdr) p-values, not the raw ones.

**Worked example.** Notebook 4's Phase 3 ran a smaller number of pairwise DM tests and
"got away with" not correcting for multiple testing, per `NEXT_RUN_PROMPT.md`'s own
framing, only because nothing came back significant regardless — a correction that would
have made zero difference to that particular conclusion. Notebook 5's contest is larger
(10 models, not 7) and might have something come back significant, which is exactly why
the omission can't be repeated safely this time.

**Pitfalls.** Running many tests and reporting only the significant-looking ones (without
disclosing how many were run in total) is a well-known way results get accidentally — or
deliberately — overstated; always report the full test count alongside any subset of
significant results.

---

### Family-wise error rate

**In one sentence.** The probability of making *at least one* false-positive error
anywhere across a whole family (batch) of tests — a stricter target than controlling
each individual test's error rate separately.

**The maths.** $\mathrm{FWER} = P(\text{at least one Type I error among all } m \text{
tests})$. Controlling FWER at $\alpha$ (e.g. via a Bonferroni correction, testing each
individual hypothesis at $\alpha/m$ instead of $\alpha$) is generally much more
conservative than controlling each test at $\alpha$ individually, especially as $m$
grows.

**Why it is here.** Explained here as the alternative this programme deliberately does
*not* use for its main density contest — [Benjamini-Hochberg FDR](#benjamini-hochberg-fdr)
(below) is a less conservative correction, chosen because controlling the *expected
proportion* of false discoveries is judged more appropriate than guaranteeing almost zero
chance of even one, given the 180-test scale of Phase 3's contest.

**Worked example.** A Bonferroni correction on 180 tests at target FWER 0.05 would test
each individual comparison at $0.05/180 \approx 0.00028$ — extremely strict, likely to
miss real but modest effects entirely (very low [power](#power)) at this test count.

**Pitfalls.** FWER control becomes extremely conservative (low power) as the number of
tests grows large — appropriate when even a single false positive is costly, less
appropriate for an exploratory contest across many models where some false positives are
tolerable as long as their expected *rate* is controlled, which is exactly the FDR
framing this repo uses instead.

---

### Benjamini-Hochberg FDR

**In one sentence.** A multiple-testing correction that controls the *expected
proportion* of false positives among all the tests you call "significant" — less
conservative than guaranteeing almost no false positives at all ([FWER](#family-wise-error-rate)),
more appropriate when you're running many tests and can tolerate a small, known rate of
mistakes among your "winners."

**The maths.** Sort $m$ p-values ascending: $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$.
Find the largest $k$ such that $p_{(k)} \le \frac{k}{m}\alpha$; declare all tests with
$p \le p_{(k)}$ significant. This guarantees the *expected* false discovery rate (fraction
of "significant" results that are actually false positives) is at most $\alpha$.

**Why it is here.** `NEXT_RUN_PROMPT.md` mandates exactly this for notebook 5's Phase 3:
"report a Benjamini-Hochberg FDR-adjusted verdict alongside the raw one, and make Gate
A's determination on the adjusted p-values" across the 180 tests (45 pairs x 4
intervals) in the density contest.

**Worked example.** If 20 of 180 raw p-values are below 0.05, BH doesn't simply keep all
20 — it re-ranks them and only keeps those below their own rank-dependent, generally
stricter threshold $\frac{k}{180}\times 0.05$, so the smallest p-values in the batch
survive more easily than the largest ones near the raw 0.05 cutoff.

**Pitfalls.** BH controls the *expected* false discovery rate across the whole declared
batch of tests — it assumes you're applying it to the complete, pre-declared set of
comparisons, not a hand-picked subset chosen after seeing which ones looked interesting;
running it only on your favorite subset of the 180 tests would silently break the
guarantee.

---

### Central limit theorem (and when it fails under heavy tails)

**In one sentence.** A foundational result saying that the *average* of many independent
random variables tends to look approximately normal, even if the individual variables
themselves don't — but this guarantee has a hidden requirement (finite variance) that
crypto returns may not actually satisfy.

**The maths.** For i.i.d. random variables $X_1,\dots,X_n$ with finite mean $\mu$ and
finite variance $\sigma^2$: $\frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \to \mathrm{Normal}(0,1)$
as $n \to \infty$. The "finite variance" clause is doing quiet, heavy lifting — it is
where the theorem's guarantee comes from, and it is exactly the assumption
[the Student-t table](01-probability-and-distributions.md#student-t-distribution) shows
can fail for $\nu \le 2$.

Read back in words: averaging washes out individual weirdness *only if* no single
observation can dominate the average — a guarantee that quietly assumes the tails aren't
so heavy that one extreme value can swamp everything else.

**Why it is here.** Every t-statistic-based test in this repo (DM test, HAC standard
errors) leans on this theorem to justify its normal-approximation p-value — and notebook
5's Phase 1b exists *precisely* to check whether that leaning is safe here, given that
BTC's fitted degrees of freedom sit at 2-3, right at or near the edge where the theorem's
own precondition may not hold.

**Worked example.** If QLIKE loss differentials are themselves heavy-tailed (plausible,
since QLIKE is built from a ratio involving realized variance, which inherits the
fat-tailedness of squared returns), the DM test's normal-approximation p-value could be
unreliable even with a large sample size — which is exactly why notebook 5 compares it
directly against a [block bootstrap](#block-stationary-bootstrap) p-value on the same
series, rather than trusting the CLT-based approximation on faith.

**Pitfalls.** "Large sample size" is not, by itself, a rescue for a CLT-based
approximation when the underlying variance is infinite or near-infinite — more
observations of a genuinely heavy-tailed quantity do not make its average behave more
normally in the way the theorem promises for a finite-variance variable; this is a
qualitative failure mode, not something that shrinks away with more data.

---

### Stationarity and the augmented Dickey-Fuller test

**In one sentence.** A series is (weakly) **stationary** if its mean, variance, and
autocovariance structure don't drift over time — a **unit root** is the specific failure
mode where a series behaves like a random walk instead (today's level is exactly
yesterday's level plus unpredictable noise, so it can wander arbitrarily far with no
tendency to return), and the **augmented Dickey-Fuller (ADF) test** is the standard test
for telling the two apart.

**The maths.** Fit $\Delta y_t = a + b\,y_{t-1} + \sum_{i=1}^{p} \gamma_i \Delta y_{t-i} +
\varepsilon_t$ by OLS (the $\gamma_i$ lagged-difference terms are the "augmented" part —
enough of them to soak up any remaining autocorrelation in $\varepsilon_t$, chosen here by
BIC). $H_0$: $b=0$ (unit root — $y$ is a random walk in levels). $H_1$: $b<0$ (stationary
— $y$ reverts toward a fixed mean). The test statistic is $b$'s own OLS t-stat, but it is
**not** compared against a standard normal/t table — under $H_0$ the statistic's sampling
distribution is skewed left, so a dedicated, tabulated critical value (MacKinnon 2010,
asymptotic for the constant-only case: −3.43 at 1%, −2.86 at 5%, −2.57 at 10%) is required
instead. Using an ordinary t-table here silently understates significance.

**Why it is here.** It's the precondition for treating any spread or pair as
mean-reverting at all (see
[cointegration and the Engle-Granger test](09-market-data-and-microstructure.md#cointegration-and-the-engle-granger-test)):
a series with a unit root can drift arbitrarily far from any "fair value," so a
mean-reversion trading signal built on it has no statistical floor under it — the position
can simply never come back.

**Worked example.** `spread_lib10.adf_test`, applied to all 30 of this repo's pre-built
commodity spreads (notebook 10a): 23 of 30 reject the unit-root null at 5%. gold_silver
and platinum_palladium do not (t = −1.76 and −1.41), resolving a disagreement notebook 9's
cheaper AR(1)/IC probe had flagged but not settled — both pairs are not actually
cointegrated, consistent with their weak showing on the cheaper tests too.

**Pitfalls.** A large ADF t-stat magnitude is reassuring but the test has genuinely low
power against a *near*-unit-root process (e.g. $b=-0.001$, technically stationary but with
a half-life of centuries) — always read the ADF result alongside an estimated half-life
(see [Ornstein-Uhlenbeck process and half-life of mean reversion](09-market-data-and-microstructure.md#ornstein-uhlenbeck-process-and-half-life-of-mean-reversion)),
never the t-stat in isolation. The lag-augmentation count also matters: too few lags leave
autocorrelation in the residual (inflating apparent significance), too many waste degrees
of freedom — an automatic rule (BIC here) is a reasonable default, not a guarantee of the
"right" answer for every series.

---

### Variance-ratio test (Lo-MacKinlay)

**In one sentence.** A statistical test for whether a price series is a random walk (and thus has no predictable mean reversion), based on comparing the variance of multi-period returns to what a random walk's variance *should* be — VR(q) = Var(q-period returns) / (q × Var(1-period returns)), which equals 1 under the null.

**The maths.** For a level series $x_t$ (e.g. a spread value), the q-period log returns are $r_t^{(q)} = x_t - x_{t-q}$. The variance ratio is:
$$\mathrm{VR}(q) = \frac{\mathrm{Var}(r_t^{(q)})}{q \cdot \mathrm{Var}(r_t^{(1)})}$$
Under the random-walk null, $\mathrm{VR}(q) = 1$ — the variance of q-period returns should be exactly $q$ times the variance of 1-period returns. The homoscedastic-null z-statistic (Lo & MacKinlay 1988, eq. 10) is:
$$z = \frac{\mathrm{VR}(q) - 1}{\sqrt{2(2q-1)(q-1)/(3qn)}}$$
asymptotically $\mathrm{Normal}(0,1)$. A one-sided test against mean reversion (the alternative that the series reverts toward a level) rejects the random walk when $z < -1.645$, i.e. when $\mathrm{VR}(q)$ is significantly below 1.

**Why it is here.** `spread_lib11.variance_ratio` is used in notebook 11a to test whether a commodity calendar spread shows mean-reverting behavior: VR < 1 indicates slower diffusion than a random walk (mean reversion / anti-persistence), while VR > 1 indicates faster diffusion (trending / momentum). This complements the [Ornstein-Uhlenbeck half-life](09-market-data-and-microstructure.md#ornstein-uhlenbeck-process-and-half-life-of-mean-reversion) and [ADF](#stationarity-and-the-augmented-dickey-fuller-test) tests — all three measure mean reversion, but at different horizons and via different mechanisms, and they do not always agree.

**Worked example.** Notebook 11a's `brent_calendar` spread (real numbers, `phase_1_11a_results.json`/`phase_3_11a_results.json`): VR(5) = 1.069 with z-stat +1.71 — *positive*, i.e. no evidence of mean reversion at the 5-day horizon, and in fact on the wrong side of the one-sided −1.645 rejection threshold. Yet the same spread's ADF test strongly rejects the unit root ($t = -5.22$), and its AR(1) half-life is 42.7 days. The two diagnostics disagree because they measure different things: `brent_calendar`'s daily changes carry short-horizon positive autocorrelation (5-day momentum) layered on top of genuine mean reversion at the 6-week half-life scale — not a contradiction, but a real, reportable finding (11a Phase 3) that a variance-ratio screen strict enough to require agreement on both fronts rejects even this spread, one of the external programme's own five live positions.

**Pitfalls.** VR and half-life measure different things: VR tests whether the variance grows exactly linearly with time (random walk null) vs. sub-linearly (mean reversion) or super-linearly (momentum/trending); half-life measures the *speed* of reversion conditional on a process being mean-reverting. A series can show VR > 1 at short horizons (daily momentum) while still having a measurable half-life at longer horizons — no inconsistency. Additionally, the homoscedastic-null approximation assumes constant variance; if volatility itself changes over time (heteroskedasticity), the test's p-values can be misleading.

---

### Hurst exponent (variance-of-lagged-differences estimator)

**In one sentence.** A descriptive measure (not a formal hypothesis test) of whether a time series behaves like a random walk (H = 0.5), mean-reverts (H < 0.5), or trends/persists (H > 0.5), computed by regressing the log-variance of lagged differences against log-lag and reading off half the slope.

**The maths.** For a self-affine series, the variance of k-step differences scales as $k^{2H}$ for some Hurst exponent H. Compute $\tau_k = \mathrm{Var}(x_t - x_{t-k})$ for a range of lags $k \in [k_{\min}, k_{\max}]$, then fit $\log(\tau_k) = \log(\text{const}) + 2H \log(k)$ via ordinary least squares. The slope is $2H$, so $H = \frac{\text{slope}}{2}$. H = 0.5 indicates a random walk, H < 0.5 indicates mean reversion (anti-persistence), H > 0.5 indicates trending or persistence.

**Why it is here.** `spread_lib11.hurst_exponent` is used descriptively in notebook 11a to characterize a spread's statistical behavior — not as a formal test with a p-value, but as a summary statistic alongside ADF and half-life to triangulate whether a series is genuinely mean-reverting. Unlike the [variance-ratio test](#variance-ratio-test-lo-mackinlay) (which tests a specific null), the Hurst exponent is a model-free, rough-and-ready measure whose finite-sample noise is large; it's reported alongside more formal tests rather than used alone to make a gate decision.

**Worked example.** A synthetic mean-reverting AR(1) process with half-life 20 days might have H ≈ 0.35–0.45 depending on the window of lags used, the sample size, and the exact AR(1) coefficient; the same dataset's ADF t-stat will robustly reject the unit root, and its half-life estimate will pin the reversion speed precisely. The Hurst estimate confirms "this doesn't look like a random walk" but doesn't sharpen the estimate — a three-way consistency check (ADF significant, H < 0.5, half-life in expected range) is stronger evidence than any one alone.

**Pitfalls.** The Hurst exponent is a finite-sample estimate with substantial noise, especially on shorter series or at the edges of the lag range; picking lag windows differently (e.g., $k_{\min}=2$ vs. $k_{\min}=10$, or varying $k_{\max}$) can materially shift the result, which is exactly why this repo uses it descriptively, not as a formal test. Do not report a Hurst estimate as a confident finding without triangulation against other stationarity tests; it is most useful as a sanity check that multiple different tests point the same direction.


