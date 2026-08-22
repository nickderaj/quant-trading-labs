# 01 — Probability and distributions

Start here if you have never seen the word "distribution" used mathematically. This file
defines what a distribution *is*, then walks through every specific shape (family) of
distribution this research programme fits to data, in the order they get harder to
picture.

---

### Random variable

**In one sentence.** A quantity whose value isn't fixed in advance — it comes out of some
process with an element of chance, and before you observe it you can only talk about
which values are *more or less likely*, not which value you'll get.

**The maths.** Written as a capital letter, usually $X$ or $R$ (distinguishing it from a
specific observed number, written lowercase, e.g. $x$ or $r$). $X$ is fully described once
you know every value it can take and how likely each one (or each range) is — that
"how likely" assignment is called its [distribution](#probability-density-vs-mass-function).

Read back in words: $X$ is not a number, it's a *description of the process that
produces a number*. "BTC's next 1-hour log return" is a random variable; "-0.0031" (an
actual observed return) is one realization of it.

**Why it is here.** Every column this programme forecasts — `log_return`, `rv_target`,
next-bar volatility — is treated as a random variable. The entire distributional-modelling
programme (notebooks 4-5) is the project of figuring out *which distribution* these
random variables plausibly come from.

**Worked example.** Rolling a six-sided die is a random variable with 6 equally likely
outcomes. BTC's hourly log return is a random variable too, but continuous (any real
number is possible, not just 6 outcomes) and, per notebook 4's Phase 1, extremely
fat-tailed (see [fat/heavy tails](#fatheavy-tails) below) compared to a die roll's flat,
bounded shape.

**Pitfalls.** Don't confuse the random variable ($X$, the process) with a realization of
it (a specific $x$, one observed number). A sentence like "the mean of $X$" only makes
sense for the random variable/distribution; "the mean of these 1,000 realized $x$ values"
is a different (though related) thing — see
[expectation](#expectation) vs. [sample statistics computed from data](02-estimation-and-fitting.md#parameter-vs-estimate-vs-estimator).

---

### Probability density vs. mass function

**In one sentence.** The rulebook that says how likely each value (or range of values) of
a random variable is — "mass" when the variable can only take specific separate values
(like whole numbers), "density" when it can take *any* value on a continuous scale (like
a return).

**The maths.** For a discrete random variable, the probability mass function (PMF) is
$p(x) = P(X = x)$ — a genuine probability, between 0 and 1, and $\sum_x p(x) = 1$ (every
possible outcome's probability adds up to 1 — see [$\sum$](README.md#notation-conventions-used-throughout-docs)).

For a continuous random variable, no single point has positive probability (there are
infinitely many possible values), so the probability density function (PDF) $f(x)$ is
defined differently: $P(a \le X \le b) = \int_a^b f(x)\,dx$ — the probability of landing
in a *range* is the area under the curve $f$ over that range. $f(x)$ itself is not a
probability and can exceed 1; only areas under it are probabilities, and the total area
under the whole curve is 1.

Read back in words: think of a density like a heat map of where a value is likely to
land — tall means "values land here often," not "this exact value is likely."

**Why it is here.** Every fitted family in this programme (normal, Student-t, GPD) is a
density, because returns and volatilities are continuous. `distributions.py`'s
`log_score` calls `.logpdf(y)` for continuous families and `.logpmf(y)` for discrete ones
(Poisson, negative binomial, used for trade counts) — this is exactly that PDF/PMF
distinction, made explicit in code.

**Worked example.** A normal density with mean 0, std 1 has $f(0) \approx 0.399$ — a
number bigger than the probability of any actual event, because it's a density, not a
probability. The probability of landing in $[-0.1, 0.1]$ is the area under the curve over
that narrow strip, roughly $0.399 \times 0.2 \approx 0.08$ (8%) as a rough rectangle
approximation.

**Pitfalls.** Reading a PDF value itself as "the probability of that exact number" is the
single most common beginner error. A continuous PDF can output values above 1 (a very
concentrated distribution, like a Student-t with a tiny scale, easily does) — that is not
a bug, "probability" was never claimed for the point value itself, only for areas.

---

### CDF

**In one sentence.** The running total of probability: "what's the chance this random
variable comes out *at or below* this value?" — one number, growing from 0 to 1 as you
scan from the smallest possible value to the largest.

**The maths.** $F(x) = P(X \le x)$. For continuous variables, $F(x) = \int_{-\infty}^{x}
f(u)\,du$ — the area under the [density](#probability-density-vs-mass-function) up to $x$.
$F$ is always non-decreasing, $F(-\infty) = 0$, $F(\infty) = 1$.

Read back in words: the CDF at $x$ answers "if I picked one realization of $X$, what
fraction of the time would it be $x$ or smaller?"

**Why it is here.** `christoffersen_independence_test` and every VaR test in
`src/distributions.py` work off exceedance indicators — "did the actual value fall below
the model's predicted CDF at some threshold?" — which is the CDF used directly, and PIT
(see [PIT](06-scoring-rules-and-calibration.md#pit)) is nothing but "the CDF evaluated at
the observed value."

**Worked example.** For a standard normal (mean 0, std 1), $F(0) = 0.5$ (half the mass is
below 0), $F(1.645) \approx 0.95$ (this is where the "1.645" in a 5% one-sided normal VaR
comes from — see [Value at Risk](06-scoring-rules-and-calibration.md#value-at-risk)).

**Pitfalls.** The CDF and the [quantile function](#quantile-percentile-inverse-cdf) are
inverses of each other — it's easy to flip which direction you're going ("given a value,
what's its probability rank" vs. "given a probability rank, what's the value") and get a
transposed result. Check units: CDF input is a value in the variable's own scale, CDF
output is always in $[0, 1]$.

---

### Quantile / percentile / inverse CDF

**In one sentence.** The reverse question from the CDF: "what value has (say) 5% of the
distribution's mass below it?" — you supply the probability, the quantile function gives
you back the value.

**The maths.** The quantile function is the [CDF](#cdf)'s inverse: $Q(q) = F^{-1}(q)$,
so $Q(q) = x$ means $F(x) = q$, i.e. $P(X \le x) = q$. A "percentile" is the same idea
with $q$ expressed as a percentage (the 5th percentile is $Q(0.05)$).

Read back in words: the quantile function answers "where's the cutoff for the bottom $q$
of outcomes?"

**Why it is here.** [Value at Risk](06-scoring-rules-and-calibration.md#value-at-risk) at
level $q$ is exactly a quantile: "the loss such that only $q$ of outcomes are worse."
`scipy`'s `.ppf(q)` method (percent-point function) is this function, called directly in
`dist_lib.density_scores` to get `q05` (the 5% quantile of the fitted distribution).

**Worked example.** For a standard normal, $Q(0.05) \approx -1.645$ — the 5% VaR
threshold in standardized units, used throughout Phase 4's coverage tests.

**Pitfalls.** A quantile function is only well-defined (single-valued) when the CDF is
strictly increasing; a distribution with a flat spot in its CDF (zero density over a
range) has an ambiguous or non-unique quantile there. Also: "5% quantile" and "95th
percentile" are *different* things (one is a low cutoff, the other a high one) — always
check which tail is meant.

---

### Expectation

**In one sentence.** The long-run average value a random variable would produce if you
could sample it infinitely many times — "the number you'd bet on if you had to guess
just one number, and you cared about being right on average over many repeats."

**The maths.** For discrete $X$: $\mathbb{E}[X] = \sum_x x \cdot p(x)$. For continuous
$X$: $\mathbb{E}[X] = \int x \cdot f(x)\,dx$. Both are a probability-weighted average:
multiply each possible value by how likely it is, and add them up.

Read back in words: expectation is not "the most likely single outcome" (that's the
[mode](#moments), a different concept) — it's the probability-weighted balance point of
the whole distribution.

**Why it is here.** Every [moment](#moments) (variance, skewness, kurtosis) is defined in
terms of an expectation of some function of $X$. [Log score](06-scoring-rules-and-calibration.md#log-score)
being a [proper scoring rule](06-scoring-rules-and-calibration.md#proper-scoring-rule-and-why-properness-matters)
means its *expected* value is maximized by the true distribution — a claim about
$\mathbb{E}[\cdot]$, not about any single observation.

**Worked example.** A fair six-sided die has $\mathbb{E}[X] = (1+2+3+4+5+6)/6 = 3.5$ — a
value the die can never actually show, which is normal: expectation is a balance point,
not a possible outcome.

**Pitfalls.** $\mathbb{E}[X]$ can be undefined (not just large, but literally not existing
as a finite number) for a heavy-tailed enough distribution — this is not a hypothetical
edge case in this programme: see
[what $\nu \le 2$ vs $\nu > 4$ means for a Student-t](#student-t-distribution) and
[tail index](#tail-index) below, both directly relevant to whether BTC's own variance is
a meaningful quantity at all (notebook 5's central question).

---

### Variance

**In one sentence.** A single number describing how spread out a distribution is: the
average squared distance of a value from its own mean. Bigger variance means values swing
further from the average, more often.

**The maths.** $\mathrm{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]$ — the
[expectation](#expectation) of the squared deviation from the mean. Squaring is what
makes "how far from the mean" always positive and heavily penalizes large deviations.

Read back in words: take every possible value, measure how far it is from the average,
square that distance (so far-above and far-below both count as "far," and big
deviations count much more than small ones), then average those squared distances.

**Why it is here.** "Volatility" in this entire programme *means* variance (or its square
root, [standard deviation](#standard-deviation)) of returns. GARCH, HAR-RV, and every
range estimator in `04-volatility-models.md` exist to forecast this one quantity.

**Worked example.** BTC 1h log returns (notebook 4, Phase 1) have a fitted normal scale
$\sigma \approx 0.00584$, so variance $\approx 0.0000341$ — a small number because log
returns themselves are small, which is exactly why forecasts are reported as variance
(`rv_target`) rather than the harder-to-read raw units.

**Pitfalls.** Variance requires $\mathbb{E}[X^2]$ to be finite — for a Student-t with
degrees of freedom $\nu \le 2$, it is *not* finite, so "the variance" is not a number
that exists, not merely one that's hard to estimate. Notebook 4 fit $\nu \approx 1.98$ at
1h — right at this boundary — which is the entire motivation for notebook 5's Phase 1
(Hill estimator, checked independently of the t-fit).

---

### Standard deviation

**In one sentence.** The square root of [variance](#variance) — spread measured back in
the same units as the original data (variance's units are squared, which is awkward to
interpret; standard deviation undoes the squaring).

**The maths.** $\sigma = \sqrt{\mathrm{Var}(X)}$.

**Why it is here.** "Volatility" is reported both ways in this repo depending on context:
`rv_target` (realized variance) is squared-return units; GARCH forecasts $\sigma_t^2$
(variance); but VaR/quantile calculations need $\sigma_t = \sqrt{\sigma_t^2}$ because a
quantile of the return itself is in return units, not squared-return units.

**Worked example.** BTC 1h: $\sigma \approx 0.00584$, i.e. about 0.58% per hour — a
number you can sanity-check against an actual price chart, unlike the variance
$0.0000341$.

**Pitfalls.** Forgetting to take the square root (using variance where standard deviation
is needed, or vice versa) silently produces numbers off by orders of magnitude,
especially since returns are small (squaring a small number makes it much smaller). This
is exactly the kind of unit slip the Student-t entry warns about: a fitted distribution's
*scale* parameter must be read in the right units before it can be called a volatility.

---

### Moments

**In one sentence.** A family of summary numbers, each capturing a different aspect of a
distribution's shape — the 1st moment is location (the mean), the 2nd is spread
(variance), the 3rd is lopsidedness (skewness), the 4th is tail weight (kurtosis).

**The maths.** The $k$-th moment about the mean is $\mathbb{E}[(X - \mathbb{E}[X])^k]$.
$k=2$ is [variance](#variance). Higher $k$ raises the deviation to a higher power, which
makes it more and more sensitive to values *far* from the mean (extreme values dominate
a 4th-power sum far more than a 2nd-power one).

Read back in words: each moment is "how much of this distribution's shape is explained by
deviations of a certain size, weighted by how many times you multiply that deviation by
itself."

**Why it is here.** [Skewness](#skewness) and [kurtosis](#kurtosis-and-excess-kurtosis)
(3rd and 4th moments) are exactly what distinguish a normal fit from a
[Student-t](#student-t-distribution) or [skewed-t](#skewed-t-jones-faddy) fit in notebook
4's Phase 1 — a normal has zero skew and a fixed kurtosis; real returns don't.

**Worked example.** A symmetric distribution (like the normal) has 0 for every odd
moment (3rd, 5th, ...) by construction — there's no lopsidedness to cancel out symmetric
positive and negative deviations. This is the diagnostic notebook 4 used to say BTC's
skew parameters are "close to symmetric" (its fitted skew-t $a \approx b$).

**Pitfalls.** Higher moments require *more* finite lower moments to exist reliably — you
need finite 4th moment for kurtosis to be meaningful, and per the
[variance](#variance) entry above, BTC's own 2nd moment is already right at the edge of
existing. A "kurtosis" number computed on data whose theoretical 4th moment may not
exist is a real number (sample statistics always compute to *something*) but not a
reliable estimate of anything — see
[central limit theorem and when it fails](03-statistical-inference.md#central-limit-theorem-and-when-it-fails-under-heavy-tails).

---

### Skewness

**In one sentence.** How lopsided a distribution is: positive skew means a long tail
stretching to the right (occasional big positive surprises), negative skew means a long
tail to the left (occasional big drops), zero means symmetric.

**The maths.** The standardized 3rd [moment](#moments):
$\mathrm{Skew}(X) = \mathbb{E}\!\left[\left(\frac{X - \mu}{\sigma}\right)^3\right]$, where
$\mu = \mathbb{E}[X]$ and $\sigma$ is the [standard deviation](#standard-deviation).
Dividing by $\sigma^3$ makes this unitless, so a skewness of, say, $-0.5$ means the same
thing regardless of whether $X$ is measured in dollars or log-return units.

**Why it is here.** [Skewed-t (Jones-Faddy)](#skewed-t-jones-faddy) has a shape parameter
pair $(a, b)$ that controls skewness directly; notebook 4 fit $a \approx b$ (near-equal)
at every interval, which is the quantitative version of "BTC log returns are close to
symmetric."

**Worked example.** Notebook 4's fitted skew-t at 1d: $(a, b) = (1.39, 1.33)$ — close
enough that the extra skew parameter didn't improve calibration (worse or tied KS
statistic vs. plain Student-t at every interval) — the write-up's stated conclusion,
"skew-t is not worth its complexity for BTC," is exactly this skewness finding read back
in plain language.

**Pitfalls.** Sample skewness is a noisy, high-variance statistic, especially with fat
tails — a handful of extreme observations can swing it wildly. Report it alongside a
[confidence interval](03-statistical-inference.md#confidence-interval) or, better, a
[bootstrap](03-statistical-inference.md#bootstrap), not as a bare point number.

---

### Kurtosis (and excess kurtosis)

**In one sentence.** How "fat-tailed" a distribution is relative to its overall spread —
high kurtosis means extreme values (far from the mean) happen more often than a bell
curve of the same variance would predict.

**The maths.** The standardized 4th moment:
$\mathrm{Kurt}(X) = \mathbb{E}\!\left[\left(\frac{X-\mu}{\sigma}\right)^4\right]$. A
normal distribution has $\mathrm{Kurt}(X) = 3$ exactly, regardless of its mean or
variance — this is why **excess kurtosis**, $\mathrm{Kurt}(X) - 3$, is usually reported
instead: it's 0 for a normal, positive for a fatter-than-normal tail.

**Why it is here.** Excess kurtosis is the single number most often used to say "this
distribution has fat tails" — see the [Student-t](#student-t-distribution) entry's table
of which degrees of freedom give finite vs. infinite kurtosis, directly relevant to
whether notebook 4/5's fitted $\nu \approx 2$-$3$ range even has a meaningful kurtosis to
quote.

**Worked example.** A Student-t needs $\nu > 4$ for kurtosis to be finite at all.
Notebook 4's fitted $\nu$ (1.98 to 2.88 across intervals) is *below* 4 everywhere —
BTC's log returns, under the fitted t model, have **infinite excess kurtosis**, a
stronger and more precise statement than "fat tails."

**Pitfalls.** A sample kurtosis computed from data whose true kurtosis is infinite will
still return some finite number (it's just an average of finite sample values) — but that
number is not a stable estimate of anything and will jump around wildly between samples,
dominated by whichever single most extreme observation happened to be included. Never
quote a sample kurtosis without checking whether the underlying tail index
([Hill estimator](07-extreme-value-theory.md#hill-estimator)) is consistent with it even
being finite.

---

### Fat/heavy tails

**In one sentence.** A distribution has fat (or "heavy") tails when extreme values — far
from the average — happen much more often than a normal (bell curve) distribution with
the same variance would predict.

**The maths.** No single universal formula; the qualitative signature is that the
density $f(x)$ decays toward zero, as $x$ moves away from the center, *slower* than the
normal's $e^{-x^2/2}$ (exponential-of-a-square) rate. A common quantitative marker: a
finite or infinite [tail index](#tail-index) below some threshold, or (for a Student-t)
low degrees of freedom.

Read back in words: "fat tails" isn't a vague adjective here, it's a comparison against
the normal distribution specifically, at the same variance, and the comparison is most
useful stated as a *ratio* (how many times more likely is a big move, really, vs. what
normal predicts).

**Why it is here.** This is the single most repeated finding in notebooks 4 and 5:
notebook 4 measured 5-sigma-or-larger daily moves at 2,400 to 7,100 times more frequent
than a normal fit with the same variance implies — see the
[Student-t](#student-t-distribution) entry's worked example for the exact numbers.

**Worked example.** Under a true normal distribution, a move of 5 standard deviations or
more happens about once every 3.5 million observations. Notebook 4 measured BTC 1d moves
of that size roughly once every 5,800 bars — a rate difference of thousands-fold, not a
rounding difference.

**Pitfalls.** "Fat tails" is sometimes used loosely to mean "the data has some big
outliers" — in this programme it is used precisely, always anchored to a specific
comparison (normal-implied frequency vs. observed frequency, or a fitted
[tail index](#tail-index)), never as a vibe.

---

### Tail index

**In one sentence.** A single number describing exactly how fat a distribution's tail is
— low values mean extremely fat tails (extreme events are relatively common and can even
make the mean or variance not exist), high values mean the tail thins out fast (close to
normal-like behavior).

**The maths.** For a distribution whose tail follows a power law,
$P(X > x) \sim x^{-\alpha}$ as $x \to \infty$ ($\alpha$, Greek letter "alpha," is the tail
index). Read back in words: the chance of exceeding a very large value $x$ shrinks like
$x$ raised to a *negative* power — a much slower shrink than the normal's exponential
decay, and the smaller $\alpha$ is, the slower still. $\alpha \le 2$ means infinite
variance; $\alpha \le 1$ means infinite mean.

**Why it is here.** This is notebook 5's foundational question: is $\alpha$ (estimated
independently of any parametric fit, via the
[Hill estimator](07-extreme-value-theory.md#hill-estimator)) safely above 2, or does BTC
sit right at the boundary where variance itself may not be a meaningful quantity — which
would undermine every QLIKE-based comparison in notebook 4.

**Worked example.** A Student-t with $\nu$ degrees of freedom has tail index
$\alpha = \nu$ exactly — this is why notebook 4's fitted $\nu \approx 2$-$3$ is read
directly as a tail-index estimate, and why notebook 5 re-derives $\alpha$ a second,
non-parametric way (Hill) rather than trusting the t-fit's $\nu$ alone.

**Pitfalls.** A tail index estimated from a parametric MLE fit (like scipy's `t.fit`) can
be an optimizer artifact rather than a real feature of the data — notebook 4 caught
exactly this on its gap-return series (fitted $\nu \approx 1.99$ at *every* interval,
regardless of the true underlying behavior, because the series was so close to a point
mass that the optimizer pinned at its search boundary). Always cross-check a
parametrically-fitted tail index against a distribution-free estimate.

---

### Support

**In one sentence.** The set of values a random variable is actually allowed to take —
everywhere else, its density or mass is exactly zero.

**The maths.** $\mathrm{supp}(X) = \{x : f(x) > 0\}$ (continuous case). E.g. the normal's
support is all real numbers $(-\infty, \infty)$; the [Beta](#beta-distribution) distribution's support
is $(0, 1)$ only; the [Poisson](#poisson-distribution)'s support is $\{0, 1, 2, \dots\}$.

**Why it is here.** Choosing a family requires matching its support to what the data can
actually be: `taker_buy_ratio` and intrabar close position both live in $(0,1)$, which is
exactly why notebook 4 fits [Beta](#beta-distribution) to them rather than a normal (whose support
would allow nonsensical values below 0 or above 1).

**Worked example.** Fitting a normal to trade counts (which must be non-negative whole
numbers) would assign positive density to negative counts, an impossibility — this is why
count data uses [Poisson](#poisson-distribution)/[negative binomial](#negative-binomial) instead,
whose support is exactly $\{0, 1, 2, \dots\}$.

**Pitfalls.** `distributions.py`'s `_fit_beta` explicitly rejects any observation `<= 0`
or `>= 1` before fitting — a single boundary observation (a bar with a taker-buy ratio of
exactly 0 or 1) breaks a whole-history Beta fit, which is exactly the "fit failed
(boundary obs)" cells in notebook 4's Phase 1 table, a real support violation in the raw
data, not a code bug.

---

### i.i.d.

**In one sentence.** Short for "independent and identically distributed" — a simplifying
assumption that a sequence of random variables are each drawn from the *same*
distribution and knowing one tells you nothing about the others.

**The maths.** $X_1, X_2, \dots, X_n$ are i.i.d. if every $X_i$ has the same distribution
$F$, and for any $i \ne j$, $X_i$ and $X_j$ are statistically independent
($P(X_i \le a, X_j \le b) = P(X_i \le a) \cdot P(X_j \le b)$).

**Why it is here.** Nearly every classical statistical test (a plain [Kolmogorov-Smirnov test](03-statistical-inference.md#kolmogorov-smirnov-test),
a naive standard error) assumes i.i.d. data. Financial returns violate this in at least
two well-documented ways this programme measures directly:
[volatility clustering](04-volatility-models.md#volatility-clustering) (not identically
distributed moment-to-moment — variance itself changes) and
[autocorrelation](03-statistical-inference.md#autocorrelation) (not independent). This is
exactly why this programme reaches for [HAC standard errors](03-statistical-inference.md#newey-west-hac-standard-errors)
and the [block bootstrap](03-statistical-inference.md#block-stationary-bootstrap) instead
of plain formulas that assume i.i.d.

**Worked example.** Coin flips are the textbook i.i.d. example: each flip has the same
50/50 distribution and doesn't affect the next. BTC returns are not this — a big move
today makes a big move tomorrow more likely (clustering), which is precisely why GARCH
exists (a model of the *non*-i.i.d. structure in the variance).

**Pitfalls.** Applying an i.i.d.-assuming formula (like a naive standard error) to
autocorrelated or clustered data doesn't just add noise — it can be systematically
wrong in a specific direction (usually understating uncertainty, making a
non-effect look significant). This is exactly the concern behind notebook 5's Phase 1b:
checking whether the Diebold-Mariano test's own normal-approximation CLT holds when the
underlying loss differentials are themselves heavy-tailed and possibly not i.i.d.

---

### Normal/Gaussian distribution

**In one sentence.** The familiar symmetric bell curve — most values cluster near the
middle, and how likely you are to be far from the middle drops off extremely fast (fast
enough that this is exactly the property real financial returns tend to violate).

**The maths.** $f(x \mid \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}}
e^{-\frac{(x-\mu)^2}{2\sigma^2}}$, where $\mu$ (the mean) sets the center and $\sigma$
(the standard deviation) sets the spread. Read back in words: the density falls off like
$e$ raised to a *negative squared* distance from the mean — squaring means far-away
points are punished extremely hard, and $e^{(\cdot)}$ (exponential) decay is about as
fast a decay as smooth distributions get.

**Why it is here.** The normal is the baseline every fatter-tailed family in this
programme is compared *against*. GARCH-normal, rung 0-4 of notebook 4's Phase 3 ladder,
and the whole "density scoring" apparatus of notebook 5 use the normal as the reference
point that gets beaten (or doesn't) by heavier-tailed alternatives.

**Worked example.** Under a normal fit, notebook 4 found the *predicted* frequency of a
5-sigma daily BTC move is about 0.0000072% — the actual observed frequency was 0.017%,
about 2,400 times higher. That gap is the entire empirical case, in this repo, for using
anything other than a normal.

**Pitfalls.** The normal's tails decay so fast that fitting it to genuinely fat-tailed
data (like BTC returns) doesn't just slightly mis-estimate risk — it can understate the
frequency of large moves by *thousands* of times, precisely because exponential decay
crushes the tail so much harder than a real fat-tailed process does. Never use a normal
VaR/ES number for BTC without checking it against a [Student-t](#student-t-distribution)
or [EVT](07-extreme-value-theory.md#extreme-value-theory)-based alternative first.

---

### Log-normal distribution

**In one sentence.** The distribution you get when the *logarithm* of a variable is
normal — useful for quantities that are always positive and tend to be right-skewed
(most values modest, occasional very large ones), like realized variance itself.

**The maths.** $X$ is log-normal if $\log(X) \sim \mathrm{Normal}(\mu, \sigma^2)$. Its own
mean (in the original, un-logged units) is $\mathbb{E}[X] = e^{\mu + \sigma^2/2}$ — not
simply $e^{\mu}$, because of [Jensen's inequality](02-estimation-and-fitting.md#numerical-optimization-and-convergence)-style
convexity: averaging after exponentiating gives a different (larger) answer than
exponentiating the average.

**Why it is here.** One of notebook 4's rung 4 candidates (fitting a distribution
directly to realized variance) uses log-normal as one of three families (alongside
gamma, inverse-gamma); notebook 5's Phase 1c tests whether fitting a normal to
$\log(\mathrm{rv})$ directly is better-calibrated than fitting distributions to
raw `rv` — the log-normal correction formula above is exactly what's needed to convert a
log-RV forecast back to RV units for a fair QLIKE comparison.

**Worked example.** Realized variance is strongly right-skewed (a few huge-volatility
bars, many calm ones) — exactly the shape a log-normal describes, and exactly why
notebook 4 found the raw-RV distribution fits performed poorly (worst rung after EWMA):
fitting a symmetric-in-its-own-space family to a variable whose *log* is closer to
symmetric is fighting the data's natural shape.

**Pitfalls.** Forgetting the $e^{\sigma^2/2}$ correction term when converting a fitted
$\log(\mathrm{rv})$ model's mean back to RV units systematically understates the
forecast (exponentiating a mean is not the same as taking the mean of the exponentials) —
this is a real, easy-to-make bug, not a hypothetical one, which is why notebook 5's
write-up calls it out explicitly wherever it applies.

---

### Student-t distribution

**In one sentence.** A bell-shaped distribution like the normal, but with a dial that
controls how often extreme values happen — turn the dial down and huge moves become far
more likely than the normal curve allows.

**The maths.** A Student-t with $\nu$ (Greek letter "nu") degrees of freedom has
probability density

$$f(x \mid \nu) = \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\,\Gamma\!\left(\frac{\nu}{2}\right)} \left(1 + \frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}$$

Read back in words: the chance of landing near $x$ falls off like $x^{-(\nu+1)}$ — a
*power* of $x$. Compare the normal, which falls off like $e^{-x^2/2}$ — an *exponential*
of $x^2$. Exponential decay crushes large values far harder than power decay does, which
is the entire reason the t has fatter tails. $\Gamma(\cdot)$ is the
[gamma function](#the-gamma-function), only there to make the area under the curve equal
1; it can be ignored on a first reading.

The parameter $\nu$ controls tail weight, and it also controls which
[moments](#moments) exist at all:

| $\nu$ | variance | kurtosis | interpretation |
|---|---|---|---|
| $\nu \le 2$ | **infinite** | infinite | variance is not a meaningful quantity |
| $2 < \nu \le 4$ | finite | **infinite** | variance exists, but "typical outlier size" does not |
| $\nu > 4$ | finite | finite | well-behaved, still fatter-tailed than normal |
| $\nu \to \infty$ | finite | $\to 3$ | becomes exactly the normal distribution |

**Why it is here.** Notebook 4 fit $\nu$ to BTC log returns and got 1.98 (1h) to 2.88
(1d) — see `src/results/004_distributional_models.md`. Those sit in or near the first row
of that table, which is the finding: BTC returns are so heavy-tailed that their variance
is at the edge of not existing. Used in `dist_lib.fit_garch11(innovation="t")` and
scored in `dist_lib.density_scores(family="t")`.

**Worked example.** Under a normal distribution a 5-sigma daily move is expected roughly
once every 7,000 years. Under a t with $\nu = 3$, roughly once every 2 years. Notebook 4
measured BTC's actual rate at ~1 in 5,800 bars at 1d — vastly closer to the t than the
normal, and the direct source of the "2,400x to 7,100x" figure in that report.

**Pitfalls.**
- SciPy's `t.fit` can pin $\nu$ at its search boundary and return it as if it were a real
  estimate. Notebook 4 hit exactly this on the near-degenerate gap series (df $\approx$
  1.99 at every interval, which was an optimizer artifact, not a finding). Always
  sanity-check a fitted $\nu$ against a second, non-MLE estimate — see
  [Hill estimator](07-extreme-value-theory.md#hill-estimator).
- "Degrees of freedom" here has **nothing to do** with the degrees of freedom in a
  t-*test*. Same name, unrelated meaning. This trips up nearly everyone.
- A t must be *standardized* (scaled by $\sqrt{\nu/(\nu-2)}$) before its scale parameter
  can be read as a volatility. `_garch_negloglik` does this; forgetting it silently
  misprices every quantile.

---

### Skewed-t (Jones-Faddy)

**In one sentence.** A Student-t distribution with an extra dial for lopsidedness — two
shape parameters instead of one, so the left and right tails can be different weights.

**The maths.** The Jones & Faddy (2003) `jf_skew_t(a, b)` parameterization uses two
shape parameters $a, b > 0$ that each control one tail's weight; $a = b$ recovers a
distribution close to symmetric-t, $a \ne b$ tilts weight toward one side. There is no
simple closed-form density worth reproducing here — the practical fact that matters is
that it costs *two* fitted shape parameters (plus location and scale) instead of the
plain t's one.

**Why it is here.** Notebook 4's Phase 1 fit skew-t alongside plain t specifically to
test whether BTC's tails are asymmetric; `FAMILY_PARAMS["skewt"] = ("a", "b", "loc",
"scale")` in `distributions.py` is exactly this parameterization.

**Worked example.** Notebook 4 fit $(a, b) \approx (1.25, 1.27)$ at 1h through
$(1.39, 1.33)$ at 1d — always close to equal — and found the KS calibration statistic
was *worse or tied* against plain Student-t at every interval. The extra parameter bought
fitting noise, not a real asymmetry, which is the write-up's stated conclusion.

**Pitfalls.** A four-parameter fit (skew-t) on a short rolling window is meaningfully
more prone to overfitting/non-convergence than the two-parameter plain t — this is why
notebook 5's Phase 2 (GJR-GARCH) explicitly skips a skew-t innovation variant: five shape
parameters plus leverage on a 500-bar window is judged over-parameterized, building
directly on this finding.

---

### Generalized error distribution (GED)

**In one sentence.** A bell-shaped distribution with one dial that moves smoothly
between a sharp-peaked, thin-tailed shape and the ordinary bell curve — unlike the
Student-t, every one of its moments (variance, kurtosis, everything) always exists.

**The maths.** With shape parameter $\kappa > 0$ (also called $\beta$ in some sources,
e.g. `scipy.stats.gennorm`),

$$f(x \mid \kappa) = \frac{\kappa}{2\lambda\Gamma(1/\kappa)} \exp\!\left(-\left|\frac{x}{\lambda}\right|^\kappa\right)$$

where $\lambda$ is chosen so the distribution has unit variance:
$\lambda = \sqrt{\Gamma(1/\kappa)/\Gamma(3/\kappa)}$. $\kappa = 2$ is exactly the
[normal distribution](#normalgaussian-distribution); $\kappa = 1$ is the Laplace
(double-exponential) distribution — sharper peak, heavier shoulders than normal, but
still every moment finite, unlike the [Student-t](#student-t-distribution) below $\nu=4$.

**Why it is here.** Notebook 5 found GARCH-t's fitted degrees of freedom sitting near
the finite-variance boundary ($\nu \approx 2$–3, see [Student-t](#student-t-distribution)
above) — a single "how fat is the tail" dial forced onto data that might actually want a
sharp peak with only moderately heavy shoulders, not genuinely unbounded kurtosis. GED
tests that directly: if it wins over Student-t, the tail isn't infinite-variance-adjacent,
it's just sharply peaked. `src/research/tmp/densities/ged.py`
(`dist_lib6`'s Phase 3 density zoo).

**Worked example.** Verified in `tests/test_dist_lib6_ged.py`: at $\kappa=2$, GED's
`logpdf` matches `scipy.stats.norm.logpdf` exactly (both being the same distribution),
and its 1%-level expected shortfall matches the closed-form normal ES to four decimal
places — confirming the unit-variance rescaling $\lambda(\kappa)$ is implemented
correctly before it's ever fit to real returns.

**Pitfalls.** $\kappa$ and the Student-t's $\nu$ measure genuinely different things and
are not interchangeable dials — $\kappa$ controls peakedness with *all moments always
finite*; $\nu$ controls a power-law tail where moments can vanish entirely. A model can
prefer small $\kappa$ (sharp peak) while still disagreeing with a low-$\nu$ t on how
extreme the very worst 1% of days can get — check the fitted expected shortfall, not
just which family wins on log score.

---

### Normal-inverse Gaussian (NIG) distribution

**In one sentence.** A distribution built from mixing normals of different, randomly
varying widths (the "inverse Gaussian" part controls how the width itself varies) — it
gets semi-heavy tails and, unlike GED or the plain Student-t, can be lopsided.

**The maths.** With tail-weight parameter $\alpha > 0$ and skew parameter $\beta$
(requiring $\alpha > |\beta|$), plus scale $\delta$ and location $\mu$ fixed to give
zero mean and unit variance:

$$f(x) = \frac{\alpha\delta}{\pi}\exp\!\big(\delta\gamma+\beta(x-\mu)\big)\,
\frac{K_1\!\left(\alpha\sqrt{\delta^2+(x-\mu)^2}\right)}{\sqrt{\delta^2+(x-\mu)^2}},
\qquad \gamma=\sqrt{\alpha^2-\beta^2}$$

$K_1$ is the modified Bessel function of the second kind (`scipy.special.kv(1, ...)`) —
like the [gamma function](#the-gamma-function), treat it as a normalizing black box.
$\beta = 0$ gives a symmetric distribution; $\beta \ne 0$ tilts one tail heavier than the
other. As $\alpha \to \infty$ with $\beta=0$, NIG approaches the normal distribution.

**Why it is here.** Every innovation distribution used in notebooks 4–5 (normal,
Student-t, semiparametric-EVT) is symmetric. NIG is the one family in this repo's zoo
that can fit an asymmetric conditional tail directly, in one density, rather than
needing separate upper/lower GPD tail fits as EVT does — a direct test of whether
crypto's down-day tail genuinely differs in *shape*, not just magnitude, from its up-day
tail. `src/research/tmp/densities/nig.py`.

**Worked example.** Verified in `tests/test_dist_lib6_nig.py`: for shape
$(\alpha,\beta)=(3.0, 1.0)$ — noticeably skewed — numerically integrating the density
confirms mean $\approx 0$ and variance $\approx 1$ to roughly 14 decimal places once
$\delta = \gamma^3/\alpha^2$ and $\mu = -\delta\beta/\gamma$ are solved for from that
constraint, and its `ppf`/CDF round-trip to within $10^{-9}$.

**Pitfalls.** NIG has no closed-form quantile function — `ppf` here numerically inverts
the CDF via root-finding, which is markedly slower than GED/Johnson-SU/Hansen-skew-t's
closed forms and made `es()`'s naive nested-integration implementation take ~25 seconds
per call before being replaced with a fixed-point Gauss-Legendre rule. $(\alpha,\beta)$
also sit on a genuine likelihood ridge (both jointly move tail weight and skew), so MLE
recovery on a few thousand points is noisier than GED's or Hansen skew-t's better-
conditioned shape parameters — a modest discrepancy between fitted and "eyeballed" shape
is expected, not necessarily a bug.

---

### Johnson SU distribution

**In one sentence.** A four-parameter family built by squashing a normal distribution
through a hyperbolic-sine transform — extremely flexible in both skew and kurtosis, and
(unlike NIG) has an exact, cheap-to-evaluate quantile function.

**The maths.** With shape parameters $\gamma$ (skew-ish) and $\delta > 0$
(kurtosis-ish), Johnson SU is the distribution of $X = \xi + \eta\sinh\!\left(
\frac{Z-\gamma}{\delta}\right)$ for $Z$ standard normal. `scipy.stats.johnsonsu(gamma,
delta)` implements it directly; this repo's standardized version solves for the location
$\xi$ (`loc`) and scale $\eta$ (`scale`) that give exactly zero mean and unit variance at
each $(\gamma,\delta)$, via `scipy.stats.johnsonsu(gamma, delta).stats(moments='mv')`.

**Why it is here.** Its quantile function is closed-form (scipy provides it directly,
no root-finding), which matters on this hardware — VaR/ES forecasts across a full rolling
history are cheap, unlike NIG's numerically-inverted CDF. It is also the most flexible
member of this zoo in both directions (skew and kurtosis independently), making it the
natural "if a shape this flexible still can't beat GARCH-t, the extra parameters aren't
buying anything real" test case. `src/research/tmp/densities/johnsonsu.py`.

**Worked example.** Verified in `tests/test_dist_lib6_johnsonsu.py`: at $\gamma=0$,
large $\delta$ (e.g. $\delta=20$), the density at $z=0$ approaches the standard normal's
density there (Johnson SU $\to$ normal as $\delta\to\infty$ with $\gamma=0$) — confirming
the standardization and the shape's own limiting behaviour agree before it is trusted on
real returns.

**Pitfalls.** Two free shape parameters plus the closed-form loc/scale solve is cheaper
to *fit* per refit than NIG or Hansen skew-t, but this repo's own GJR-GARCH finding
(notebook 5: extra parameters cost more in refit-to-refit estimation noise than they buy
in fit quality) applies here too — a flexible shape is not automatically a better
*forecast*, only a better *in-sample* fit.

---

### Hansen's skewed Student-t distribution

**In one sentence.** A Student-t with one extra dial for lopsidedness, built so that
setting that dial to zero recovers the ordinary (symmetric, unit-variance) Student-t
exactly — the most direct, single-parameter test of "are the two tails really the same
shape" this repo has.

**The maths.** With degrees of freedom $\nu > 2$ and skew $\lambda \in (-1,1)$, define
$c = \frac{\Gamma((\nu+1)/2)}{\sqrt{\pi(\nu-2)}\,\Gamma(\nu/2)}$,
$a = \frac{4\lambda c(\nu-2)}{\nu-1}$, $b=\sqrt{1+3\lambda^2-a^2}$; the density is
piecewise around $z=-a/b$, using $(1-\lambda)$ as the left tail's scale and
$(1+\lambda)$ as the right's. Built by Hansen (1994) specifically so it is *already*
zero-mean, unit-variance for every valid $(\nu,\lambda)$ — no separate rescaling needed,
unlike GED/Johnson SU/NIG above. At $\lambda=0$: $a=0$, $b=1$, and the density collapses
exactly to a standardized Student-t.

**Why it is here.** Every finding in this research programme so far is about the *lower*
tail specifically (the Acerbi-Székely ES-underestimation result, GJR's leverage effect).
"Are the two tails the same shape" has never actually been tested — Hansen's $\lambda$
answers it in one number, and because it nests the symmetric t exactly at $\lambda=0$, it
is directly likelihood-ratio-testable against the incumbent GARCH-t, the same way GJR was
tested against plain GARCH in notebook 5. `src/research/tmp/densities/hansen_skewt.py`.

**Worked example.** Verified in `tests/test_dist_lib6_hansen_skewt.py`: at $\lambda=0$,
`logpdf` matches the standardized Student-t formula already used elsewhere in this
codebase (`st.t.logpdf(z\sqrt{\nu/(\nu-2)}, df=\nu) + \log\sqrt{\nu/(\nu-2)}$, the same
expression `dist_lib5.vectorized_t_scores` uses) to within $10^{-15}$ — machine
precision, confirming the piecewise formula's boundary case is not just approximately
but *exactly* the family it claims to nest.

**Pitfalls.** $\lambda$ answers "is the tail shape asymmetric," which is a different
question from GJR's leverage $\gamma$ ("does a down-move raise *next-bar variance* more
than an up-move of the same size") — a nonzero $\hat\lambda$ and a significant leverage
effect are both about asymmetry but are not the same finding and should not be
conflated when both are reported for the same interval/symbol.

---

### Gamma distribution

**In one sentence.** A flexible, always-positive, typically right-skewed distribution
family — a natural choice for quantities like realized variance or waiting times that
can't be negative and often have a long right tail.

**The maths.** $f(x \mid k, \theta) = \frac{1}{\Gamma(k)\theta^k} x^{k-1} e^{-x/\theta}$
for $x > 0$, with shape $k$ and scale $\theta$. Mean $= k\theta$. When $k < 1$ the
density is highest near zero and decays; $k = 1$ recovers the
[exponential](#exponential-distribution) exactly; $k > 1$ gives a hump-shaped density.

**Why it is here.** Notebook 4 fits gamma to trailing-window waiting times between
$k$-sigma events (Phase 1) and directly to realized variance (Phase 3 rung 4). A fitted
shape $k < 1$ for waiting times is the specific signature of
[volatility clustering](04-volatility-models.md#volatility-clustering): if big moves were
memoryless (Poisson-arrival), waiting times between them would be exponential ($k=1$);
$k < 1$ means waiting times are over-dispersed — clumped, some very short gaps and some
very long ones, more than memorylessness predicts.

**Worked example.** Notebook 4 measured gamma shape $k$ between 0.52 and 0.85 for
waiting times between 2-3 sigma moves, at every interval — always below 1, the
quantitative signature of clustering ("big moves come in bunches"), read directly off
this one fitted parameter.

**Pitfalls.** A gamma fit itself can still be rejected by a
[KS test](03-statistical-inference.md#kolmogorov-smirnov-test) even when it's a much
better description than the exponential null — notebook 4 found exactly this at 1h/4h
(gamma itself rejected, but with a KS statistic 2-3x smaller than exponential's). "Better
than the obvious alternative" and "not rejected as the true model" are different claims;
report both.

---

### Inverse-gamma distribution

**In one sentence.** The distribution of "one over a gamma-distributed variable" —
comes up naturally as a Bayesian-style prior/conjugate choice for variance-like
quantities, and, like gamma, is always positive and right-skewed.

**The maths.** If $Y \sim \mathrm{Gamma}(k, \theta)$, then $X = 1/Y$ is inverse-gamma
with shape $\alpha = k$ and scale $\beta = 1/\theta$; density
$f(x \mid \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{-\alpha-1}
e^{-\beta/x}$ for $x>0$. Mean exists only when $\alpha > 1$: $\mathbb{E}[X] =
\beta/(\alpha-1)$.

**Why it is here.** One of the three rung-4 candidate families in notebook 4's Phase 3
(fitting a distribution directly to realized variance, alongside gamma and log-normal).

**Worked example.** `dist_lib.rolling_rv_dist_forecast` computes the invgamma's forecast
mean as `scale / (a - 1)` — directly implementing the formula above — and explicitly
returns NaN when the fitted $\alpha \le 1$ (mean doesn't exist), rather than silently
emitting a nonsensical or infinite forecast.

**Pitfalls.** Because the mean formula has $(\alpha - 1)$ in the denominator, a fitted
$\alpha$ close to 1 makes the forecast mean explode or become numerically unstable even
though the fit itself may look reasonable — always check the fitted shape parameter's
distance from its own singularity, not just whether the optimizer reported convergence.

---

### Beta distribution

**In one sentence.** The standard distribution for a quantity that's always between 0 and
1 — a ratio, a fraction, a proportion — with two shape parameters controlling where the
mass concentrates within that range.

**The maths.** $f(x \mid a, b) = \frac{x^{a-1}(1-x)^{b-1}}{B(a,b)}$ for $x \in (0,1)$
($B(a,b)$ is the Beta function, a normalizing constant analogous to
$\Gamma$). Mean $= a/(a+b)$. Both $a,b > 1$ gives a hump in the middle; $a=b=1$ is exactly
Uniform$(0,1)$; $a, b$ both large concentrates mass tightly around the mean; $a,b < 1$
pulls mass toward the 0/1 edges (U-shaped).

**Why it is here.** Notebook 4 fits Beta to `taker_buy_ratio` (fraction of volume that
was a taker-buy) and `intrabar_close_pos` — both genuinely bounded in $(0,1)$ by
construction, exactly the support Beta was built for.

**Worked example.** Notebook 4 found `taker_buy_ratio`'s fitted $(a, b)$ growing from
$(245.8, 247.7)$ at 4h to $(869.3, 875.8)$ at 1d, with $a \approx b$ throughout — tight
concentration near 0.5 that gets *tighter* as the bar widens (more individual trades
averaged into one ratio pulls it toward its mean, the same effect that shrinks any
sample-average's variance as sample size grows).

**Pitfalls.** `distributions.py`'s Beta fitter requires every observation strictly
inside $(0, 1)$ — a single bar with taker-buy ratio exactly 0 or 1 (all-taker-sell or
all-taker-buy, a genuine occurrence at finer bar granularities) breaks a whole-sample fit
outright. This is a real property of the data (see
[frozen-price bar](09-market-data-and-microstructure.md#frozen-price-bar)), not a bug,
and notebook 4's "fit failed (boundary obs)" cells at 1h/4h document it rather than
silently returning garbage.

---

### Exponential distribution

**In one sentence.** The distribution of "waiting time until the next event," assuming
events happen completely at random and independently of how long you've already waited
(memoryless) — the baseline every clustering test in this programme is compared against.

**The maths.** $f(x \mid \lambda) = \lambda e^{-\lambda x}$ for $x \ge 0$, mean
$1/\lambda$. Memoryless: $P(X > s+t \mid X > s) = P(X > t)$ — knowing you've already
waited $s$ tells you nothing about how much longer you'll wait.

**Why it is here.** It is the special case $k=1$ of the [gamma](#gamma-distribution)
above, and is the explicit null hypothesis for "do extreme return events cluster in
time" — notebook 4 tests waiting times between $k$-sigma moves against this exact
memoryless benchmark and rejects it everywhere in favor of gamma with shape $<1$.

**Worked example.** If big moves really arrived at random (a Poisson process), the gap
between them would be exponential. Notebook 4's finding that this is rejected (gamma
shape well below 1, i.e. over-dispersed relative to exponential) is the distributional
statement of "volatility clustering."

**Pitfalls.** "Memoryless" is a strong, specific mathematical claim, not just "the events
seem random" — it means literally no information carries forward, which is precisely
what fails for financial return series (their whole defining feature is that a big move
raises the odds of another big move soon after).

---

### Geometric distribution

**In one sentence.** The discrete cousin of the exponential — "how many independent coin
flips until the first success" — used here as the memoryless null for *run lengths*
(consecutive same-sign returns) instead of *time* between extreme events.

**The maths.** $P(X = k) = (1-p)^{k-1} p$ for $k = 1, 2, 3, \dots$, mean $1/p$. Also
memoryless, in the same sense as the exponential, but for whole-number counts rather than
continuous time.

**Why it is here.** Notebook 4's "run lengths" analysis fits a geometric distribution to
the length of consecutive same-sign-return streaks and tests it with a KS test — a
rejection means sign runs are not simple independent coin flips.

**Worked example.** Notebook 4 found mean run length $\approx 1.9$ at every interval
(implied $p \approx 0.53$, close to a fair coin) but the geometric *shape* rejected by
KS at 3 of 4 intervals — the mean matches (by construction, since $p$ is fit from it) but
the actual distribution of run lengths carries more short-run structure than a memoryless
model predicts, read in the write-up as a distributional echo of short-horizon mean
reversion.

**Pitfalls.** Fitting a geometric's $p$ from the sample mean and then testing goodness of
fit against the *same* data's shape is testing two different things (mean-matching is
automatic; shape-matching is not) — don't conflate "the average run length looks
geometric-ish" with "run lengths are actually geometrically distributed," which is
exactly the distinction notebook 4's table draws.

---

### Poisson distribution

**In one sentence.** The standard distribution for counting how many times something
happens in a fixed window, when events occur independently at a constant average rate —
the baseline "nothing special going on" model for trade counts.

**The maths.** $P(X = k) = \frac{\mu^k e^{-\mu}}{k!}$ for $k = 0, 1, 2, \dots$, where
$\mu$ is both the mean *and* the variance — a Poisson variable's defining signature is
$\mathrm{Var}(X) = \mathbb{E}[X]$ exactly.

**Why it is here.** Notebook 4 tests trade counts per bar against this exact
equal-mean-variance signature via the dispersion index $\mathrm{Var}/\mathrm{Mean}$
(Poisson predicts 1).

**Worked example.** Notebook 4 measured dispersion indices from 114,393 (1h) to 891,044
(1d) — many orders of magnitude above the Poisson-predicted value of 1, i.e. trade counts
are massively "overdispersed" and nowhere close to Poisson-shaped, motivating the switch
to [negative binomial](#negative-binomial) instead.

**Pitfalls.** A dispersion index *near* 1 would itself be a red flag in this context, by
this programme's own tripwire convention — it would suggest an aggregation bug
(e.g. counts summed across a window that erased the clustering), not a genuine finding,
since real trade activity almost never looks Poisson at this granularity.

---

### Negative binomial

**In one sentence.** A count distribution like Poisson, but with an extra parameter that
lets the variance be *larger* than the mean — the standard fix once a Poisson fit is
rejected for being overdispersed.

**The maths.** Parameterized here as $(n, p)$: $P(X=k) = \binom{k+n-1}{k} p^n (1-p)^k$,
mean $= n(1-p)/p$, variance $= n(1-p)/p^2$ — strictly greater than the mean whenever
$p < 1$, unlike Poisson's forced equality.

**Why it is here.** Directly fit to trade counts in notebook 4's Phase 1 once the Poisson
dispersion-index test rejected the simpler model; `distributions.py`'s `_fit_nbinom` uses
method-of-moments (mean/variance $\to n, p$) rather than MLE, explicitly because MLE's
digamma-equation solve is both slow and unstable on small rolling windows.

**Worked example.** Notebook 4's fitted $(n, p)$ pairs, e.g. $(1.33, 8.7\times10^{-6})$ at
1h, describe count data so overdispersed that $n$ (loosely, "how many independent
sub-processes are being summed") is small and $p$ is tiny — consistent with trade
arrivals being highly clustered/bursty rather than a steady independent stream.

**Pitfalls.** The method-of-moments fit requires $\mathrm{Var} > \mathrm{Mean}$
strictly — an equi- or under-dispersed window has no valid negative-binomial fit under
this method and `_fit_nbinom` returns `None` rather than a degenerate answer; this is a
real constraint of the estimator, not a bug to route around.

---

### Generalized Pareto

**In one sentence.** The distribution that describes exactly what's left over *beyond* a
high threshold — not a model of the whole dataset, but a model purpose-built for "how bad
does it get, given that it's already bad."

**The maths.** For exceedances $y = x - u$ above a threshold $u$:
$$f(y \mid \xi, \beta) = \frac{1}{\beta}\left(1 + \xi\frac{y}{\beta}\right)^{-\frac{1}{\xi}-1}, \quad y \ge 0 \text{ (}\xi \ge 0\text{)}$$
with shape $\xi$ (Greek letter "xi") and scale $\beta$. $\xi$ controls the tail's own
heaviness: $\xi < 0$ means a bounded tail (exceedances can't grow past some finite
maximum), $\xi = 0$ recovers exponential decay, $\xi > 0$ means a power-law-like, unbounded
tail (heavier the larger $\xi$ is), and $\xi \ge 1$ makes even the mean of the exceedances
infinite.

**Why it is here.** This is the whole point of [extreme value theory](07-extreme-value-theory.md#extreme-value-theory)
and notebook 5's Phase 2 centerpiece: rather than forcing one global shape onto the
entire return distribution (what a plain Student-t GARCH does), fit GARCH/GJR for the
bulk of the variance dynamics, then fit a GPD *specifically* to the tail of the leftover
standardized residuals — see
[conditional EVT / McNeil-Frey](07-extreme-value-theory.md#conditional-evt-mcneil-frey-two-stage).

**Worked example.** By the Pickands-Balkema-de Haan theorem, the GPD is the *only*
possible limiting shape for exceedances over a high-enough threshold, for a huge class of
underlying distributions (including the Student-t this programme already uses) — this is
why EVT can be applied "underneath" a GARCH-t model without contradicting it: it's
describing the same tail, just more directly.

**Pitfalls.** $\xi < 0$ implies a bounded tail — implausible for financial returns
(nothing structurally caps how bad a crash can be) and is treated in this repo as a
tripwire: a $\xi < 0$ result is investigated as a possible threshold or sign error before
being reported as a genuine finding. See
[generalized Pareto with $\xi$ and $\beta$ explained separately](07-extreme-value-theory.md#generalized-pareto-with-xi-and-beta-explained-separately)
for the full physical interpretation of each regime.

---

### Mixture distribution

**In one sentence.** A distribution built by blending two or more simpler distributions
together, each with its own weight — used when data plausibly comes from more than one
"regime" or sub-population rather than one single process.

**The maths.** $f(x) = \sum_{j=1}^k w_j f_j(x \mid \theta_j)$, weights $w_j \ge 0$ summing
to 1, each $f_j$ a full distribution in its own right (its own mean, variance, etc.).
Read back in words: to generate a value, first randomly pick component $j$ with
probability $w_j$, then draw from that component's own distribution.

**Why it is here.** Notebook 4's Phase 1 fits a 2-component Gaussian mixture directly to
returns (via [EM](02-estimation-and-fitting.md#em-algorithm)) and finds a low-vol,
high-weight component plus a high-vol, low-weight component — a static description of
exactly the same phenomenon [regime models](05-regime-models.md) later describe
dynamically (as a process that switches between components over time, not just a fixed
blend).

**Worked example.** Notebook 4 found weights $\approx 0.79$ (low-vol, variance
$\sim 10\times$ smaller) and $\approx 0.21$ (high-vol) at 1h, with the high-vol weight
rising to $\approx 0.44$ at 1d — a mixture is a snapshot of "how often is each regime
active," without saying anything about *when*.

**Pitfalls.** A static mixture cannot distinguish "regimes alternate over time" from
"regimes are randomly scattered with no temporal structure" — both produce the identical
mixture fit. This is exactly why [regime models](05-regime-models.md) (Gaussian mixture
model as a *time-indexed* extension, hidden Markov models) are needed to add the
"when" question a plain mixture can't answer.

---

### Aggregational Gaussianity

**In one sentence.** The empirical pattern that returns look progressively closer to a
normal (Gaussian) distribution as you sum/aggregate them over longer time windows — even
though individual short-interval returns are strongly non-normal.

**The maths.** No single formula; it follows loosely from the
[central limit theorem](03-statistical-inference.md#central-limit-theorem-and-when-it-fails-under-heavy-tails)
(summing many roughly-independent pieces tends toward normality) but is weaker and more
fragile in practice than that theorem's clean guarantee, especially when the pieces being
summed are themselves extremely heavy-tailed or dependent (both true here).

**Why it is here.** Notebook 4 measured exactly this: fitted Student-t degrees of freedom
rising monotonically from 1.98 (1h) to 2.88 (1d) as bars widen — real, but far slower and
weaker than a naive CLT argument would suggest, and even at 1d the data is still nowhere
near actually normal ($\nu \to \infty$).

**Worked example.** Aggregating 24 hourly bars into one daily bar buys less than one
additional degree of freedom's worth of normality (1.98 $\to$ 2.88, not 1.98 $\to$
30-plus) — the write-up's own way of stating how weak this effect is for crypto
specifically, compared to a textbook asset where aggregational Gaussianity is often
stronger.

**Pitfalls.** Don't over-read "returns get more normal at longer horizons" as license to
use a normal model at longer horizons without checking — notebook 4 explicitly found 1d
BTC returns still far from Gaussian ($\nu \approx 2.88$, well below the $\nu > 4$
threshold for even finite kurtosis).

---

### Standardization / z-scores

**In one sentence.** Rescaling a variable to have mean 0 and standard deviation 1, so
values from different distributions (or different times, with different volatility) can
be compared on the same footing.

**The maths.** $z = \frac{x - \mu}{\sigma}$. A $z$-score tells you "how many standard
deviations away from the mean is this observation" — a unitless number regardless of what
units $x$ was originally measured in.

**Why it is here.** GARCH's whole causal machinery depends on standardizing returns by
their *own time-varying* volatility forecast: $z_t = r_t / \sigma_t$ (see
[innovations](04-volatility-models.md#innovations)). Notebook 5's conditional EVT fits a
GPD to exactly these standardized residuals — the whole two-stage McNeil-Frey approach
is "let GARCH handle the changing scale via standardization, then model what's left."

**Worked example.** Notebook 4's Phase 1 waiting-time analysis standardizes returns by
the *full-sample* std to define "a $k$-sigma move" consistently — a 2-sigma move at 1h
and a 2-sigma move at 1d both mean "2 standard deviations from that series' own mean,"
even though the raw return sizes are very different.

**Pitfalls.** Standardizing by a *rolling* (time-varying) $\sigma_t$ versus the
*full-sample* $\sigma$ are genuinely different operations with different causal
implications — a rolling standardization must use only past data (causal), while
Phase 1's full-sample standardization is explicitly a descriptive, non-causal
"characterize the whole history" step (see
[in-sample vs. out-of-sample](08-research-methodology.md#in-sample-vs-out-of-sample)).
Confusing the two is a lookahead risk.

---

### The gamma function

**In one sentence.** A continuous, smooth generalization of the factorial ($n! = n
\times (n-1) \times \dots \times 1$) to numbers that aren't whole, including fractions —
it shows up purely as a normalizing constant in several distributions here and can be
treated as a black box.

**The maths.** $\Gamma(n) = (n-1)!$ for whole numbers $n$; more generally
$\Gamma(z) = \int_0^\infty t^{z-1}e^{-t}\,dt$. It appears in the
[Student-t](#student-t-distribution) and [gamma](#gamma-distribution)/
[inverse-gamma](#inverse-gamma-distribution) densities purely to make the total area
under each curve equal exactly 1.

**Why it is here.** It appears in nearly every density formula in this file. Its only
job, everywhere it's used in this documentation, is normalization — you never need to
compute it by hand (`scipy.special.gamma` or the log form `scipy.special.gammaln` does
it), and every formula in this repo's code that uses it goes through `scipy.stats`'s
built-in distribution objects rather than a hand-rolled $\Gamma$.

**Worked example.** $\Gamma(5) = 4! = 24$. $\Gamma(2.5) \approx 1.329$ — a fractional
input the plain factorial can't handle, which is exactly why $\Gamma$ is needed for
distributions like the Student-t whose degrees of freedom don't have to be whole numbers.

**Pitfalls.** None specific to this repo beyond forgetting it's a normalizing constant
and trying to interpret its value as meaningful on its own — on first reading, every
entry above that mentions $\Gamma(\cdot)$ can have that term ignored without losing the
concept being taught.

---

### Copulas (Gaussian and t)

**In one sentence.** A copula separates a joint distribution into two independent
pieces: each variable's own **marginal** shape (fit separately, however heavy-tailed or
skewed it is) and a **dependence structure** that describes how the variables move
together, expressed entirely on a rank/probability scale — this lets a portfolio built
from very different marginals (say, a symmetric-t gold and a right-skewed NIG natural
gas) still have a single, coherent joint model.

**The maths.** Sklar's theorem: any joint CDF can be written
$F(x_1, \ldots, x_n) = C(F_1(x_1), \ldots, F_n(x_n))$ for some copula $C$ on
$[0,1]^n$. The **Gaussian copula** takes $C$ from a multivariate normal with correlation
matrix $\Sigma$: transform each marginal to a normal score via $\Phi^{-1}(F_i(x_i))$,
apply the multivariate normal's own correlation structure. The **t-copula** does the
same with a multivariate Student-t (correlation matrix plus one shared degrees-of-freedom
parameter) instead of a multivariate normal.

**Why it is here.** Notebook 8's risk engine (`commod_lib8.portfolio_risk`) needs
portfolio-level VaR/ES across 16 products with very different marginal shapes, under
three explicit dependence assumptions: empirical (bootstrap-resample real historical
joint outcomes — whatever dependence, including tail dependence, is actually in the
data), Gaussian copula, and t-copula. Simulating from a copula means: draw correlated
normal (or t) variates, convert each to a $(0,1)$ pseudo-uniform via its own CDF, then
convert *that* to the target marginal's own scale via that marginal's `ppf` — the
marginal and the dependence structure never have to be the same family.

**Worked example.** Two assets with correlation 0.6: under a Gaussian copula, simulate
$(z_1, z_2)$ correlated standard normals via a Cholesky factor of the correlation
matrix, $u_i = \Phi(z_i)$, then $x_i = F_i^{-1}(u_i)$ where $F_i^{-1}$ is *that specific
asset's* fitted marginal quantile function (e.g. a skewed NIG for one, Student-t for the
other) — the correlation between $x_1$ and $x_2$ in the resulting sample will be close
to 0.6, but the *tail* co-movement depends entirely on which copula generated $u_1, u_2$,
not on the marginals.

**Pitfalls.** The Gaussian copula has **zero asymptotic tail dependence** — no matter
how high the correlation, the probability of a simultaneous extreme move in both
variables goes to zero relative to a single extreme move, in the limit. This is not a
subtle numerical quirk; it is the specific, famous, and empirically wrong assumption
blamed for understating joint default risk in the 2008 mortgage-CDO crisis. The t-copula
(lower degrees of freedom = more tail dependence) is the standard fix, and this notebook
reports [tail-dependence coefficients](#tail-dependence-coefficient) for both, side by
side against the empirical estimate, specifically so this understatement is a measured
number rather than an assumed footnote.

---

### Tail-dependence coefficient

**In one sentence.** The probability that one variable is in its own extreme lower tail
*given* that another variable already is — a direct measure of "do crises spread," which
an ordinary correlation coefficient cannot capture because correlation is a whole-
distribution average, not a tail-specific statistic.

**The maths.** $\lambda_L = \lim_{q \to 0^+} P(U_2 \le q \mid U_1 \le q)$, where
$U_1, U_2$ are the pseudo-uniform (rank-transformed) marginals. Estimated empirically at
a fixed small $q$ (this notebook uses $q=0.10$) rather than taking the formal limit:
$\hat\lambda_L = \dfrac{1}{n_q}\sum_i \mathbb{1}[U_{1,i} \le q]\,\mathbb{1}[U_{2,i} \le q]$
divided by the count of $U_{1,i} \le q$.

**Why it is here.** It is the number that makes the [Gaussian copula's](#copulas-gaussian-and-t)
tail-dependence failure concrete rather than assumed: this notebook computes
$\hat\lambda_L$ from real historical joint returns, from Gaussian-copula-simulated data
at the same empirical correlation, and from t-copula-simulated data, and reports all
three side by side for every product pair in the portfolio.

**Worked example.** Two independent standard normal series give
$\hat\lambda_L \approx q = 0.10$ at the $q=0.10$ threshold (no tail dependence beyond
what the threshold itself implies) — the correct sanity-check baseline. A comonotonic
pair (a series compared with itself) gives $\hat\lambda_L = 1.0$ exactly — perfect tail
dependence, the other sanity-check baseline these two extremes bracket every real
estimate against.

**Pitfalls.** The empirical estimator is noisy at small $q$ with a small sample — it
counts a shrinking number of joint tail events as $q \to 0$, so an estimate from a
16-year daily history at $q=0.01$ can be based on only a handful of joint exceedances.
Reporting at a slightly less extreme $q$ (0.10 here) trades off some purity for a more
stable estimate, and that tradeoff should be stated whenever the number is used, not
left implicit.
