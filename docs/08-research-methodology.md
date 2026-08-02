# 08 — Research methodology

This file explains the discipline that keeps a backtest, or any forecasting contest, from
lying to you. It's the one file worth reading even if you skip the maths in
[01](01-probability-and-distributions.md)-[07](07-extreme-value-theory.md) — it explains
*why* this whole research programme is run the way it is, and cites this repo's own real
history of getting it wrong and catching it.

---

### In-sample vs. out-of-sample

**In one sentence.** "In-sample" means evaluating a model on the same data it was fit
on — which will almost always make it look better than it really is, since any model can
partly memorize noise specific to that data; "out-of-sample" means evaluating on data the
model never saw during fitting, which is the only honest test of whether it generalizes.

**The maths.** No formula; a definitional distinction. If a model's parameters
$\hat{\theta}$ were estimated from data $D_{\mathrm{train}}$, evaluating its loss on
$D_{\mathrm{train}}$ itself is in-sample; evaluating on a disjoint $D_{\mathrm{test}}$ is
out-of-sample.

**Why it is here.** Every [rolling refit](02-estimation-and-fitting.md#rolling-trailing-window)
model in this repo is scored strictly out-of-sample: the forecast used at bar $t$ comes
from a fit made on data strictly before $t$, then compared against $t$'s own realized
value, which the fit never saw.

**Worked example.** A GARCH fit that reports its own training-window log-likelihood is
reporting an in-sample number; `dist_lib.rolling_garch_forecast`'s output, compared
against `rv_target` at each bar, is strictly out-of-sample — a materially harder, more
honest test.

**Pitfalls.** In-sample performance can look arbitrarily good with enough model
flexibility (see [overfitting](#overfitting)) and tells you almost nothing about real
predictive ability — every credible number in this research programme's write-ups is
out-of-sample, and any in-sample number that appears is labeled as such explicitly.

---

### Training / validation / test

**In one sentence.** The standard three-way split of data for model development: fit
parameters on **training** data, tune any remaining choices (which window size, which
family) on **validation** data, and get one final, honest performance read on **test**
data that was never touched during either of the first two steps.

**The maths.** No formula; a data-partitioning convention.
$D = D_{\mathrm{train}} \cup D_{\mathrm{val}} \cup D_{\mathrm{test}}$, disjoint, used in
strictly that order and never re-visited.

**Why it is here.** This repo's own version of this split is the
[rolling refit](02-estimation-and-fitting.md#rolling-trailing-window) + full pre-holdout
sample (playing the combined training/validation role, since model *choices* like which
rung wins are decided from it) + the frozen [holdout](#holdout-and-why-it-is-spent-by-use)
(playing the test role, spent once and only for a certified winner).

**Worked example.** Notebook 3's `cfg2_12h` frozen holdout run is exactly a "test set"
evaluation in this sense: a specific, already-chosen configuration run once against data
that played no role in choosing it.

**Pitfalls.** Using the "test" partition more than once, or using its results to go back
and change a modeling choice, silently converts it into a validation set in practice —
even though it's still labeled "test," it no longer carries the same honest guarantee.
This is precisely why the [holdout](#holdout-and-why-it-is-spent-by-use) in this repo is
described as "spent" the moment it's used even a single time.

---

### Holdout and why it is spent by use

**In one sentence.** A holdout is data set aside from the very start, never touched
during any model development or comparison — its entire value comes from never having
been peeked at, so using it even once for any purpose (even just "let's see how this
looks") permanently reduces its worth for every future decision.

**The maths.** No formula; the core logic is that any decision made *after* observing
holdout performance (even an innocuous-seeming one, like "let's also check this other
model") has effectively used the holdout's information, whether or not that information
was used to formally refit anything.

**Why it is here.** `HOLDOUT_START = research.HOLDOUT_START` (2025-07-01 in this repo)
is explicitly described as "spent once by notebook 3" — every notebook after it (4 and 5)
runs entirely on the pre-holdout rolling out-of-sample sample instead, and 5's own
runbook repeats the rule explicitly: "the holdout stays frozen... only a certified
Phase 6 application would touch the holdout, and only once."

**Worked example.** Notebook 4's own write-up notes explicitly that its numbers don't
depend on holdout purity "the way a return-prediction backtest's does" precisely because
Phase 3/4 there have thousands of scored rolling out-of-sample observations, and
deliberately avoids touching the holdout even though it could, in principle, have used it
for a second look.

**Pitfalls.** "We'll just take one quick look and won't change anything" is exactly how
holdouts get quietly spent without anyone intending to violate the rule — the discipline
in this repo is to simply never run holdout-touching code until a model has already been
fully certified by every other check, so the temptation never arises in the first place.

---

### Lookahead bias / leakage

**In one sentence.** Using information that would not actually have been available at
the time a forecast was supposedly made — the single most common and most dangerous way
a backtest or forecasting contest can silently overstate how good a model really is.

**The maths.** A forecast for bar $t$, $\hat{y}_t = g(x_1, \dots, x_k)$, leaks if any
$x_i$ was only knowable at some time after $t$. No formula catches this automatically —
it must be checked by tracing exactly which bars' data feed into each computed value.

**Why it is here — four real examples from this repo's own history, in the order they
were caught:**

1. **Notebook 4's HAR-RV same-bar target leak** (`make_har_features`, bug #3).
   The daily/weekly/monthly rolling-mean RV components were built without a `shift(1)`,
   so at the 1d interval (where the "daily" window is exactly 1 bar), the daily feature
   was *literally identical* to that bar's own `rv_target` — the model was regressing
   the target on itself. Caught because HAR-RV's QLIKE came back exactly `0.000000` at
   1d — an implausibly perfect score, the classic tripwire for this class of bug.
2. **Notebook 5's own §1a: GARCH-t degrees-of-freedom leak.** The variance forecast was
   properly causal (rolling, forward-filled), but the fitted Student-t degrees of
   freedom used to *score* it under its own distribution came from `fits[-1]` — the
   single *last* refit in the whole sample — applied to score every bar from the start
   of the evaluation period onward. Bar 500 was scored under a shape parameter estimated
   from data through bar 30,000. Fixed by building a causal, forward-filled `nu_path`
   exactly mirroring how the variance itself is already forward-filled.
3. **A specific cross-sectional leak this repo's design deliberately prevents** (not a
   bug that occurred, but a documented risk designed around): `panel_walk_forward_splits`
   splits a stacked multi-symbol panel by *timestamp*, not row position, specifically
   because splitting by row position could let one symbol's bar-$t$ row land in the
   training fold while another symbol's *same* bar $t$ lands in the test fold — the same
   instant in time, on two different sides of a supposedly clean train/test boundary.
4. **A proposed fix explicitly rejected for being a lookahead leak before it was ever
   implemented**: notebook 5's out-of-scope list rejects the range-estimator drift
   correction "as originally specified" — calibrating a multiplicative bias correction
   from a full-pre-holdout, fit-once excess measurement (Parkinson's range running 6-14%
   below the Brownian prediction) and applying that single constant to *rolling*
   forecasts would scale bar-$t$'s forecast using data from after $t$. Caught in review,
   never run.

**Worked example.** See example 1 and 2 above — both are the exact same underlying
mistake (a parameter or feature computed with knowledge of the future, applied to score
the past) at two different levels of the modeling stack (a plain OLS feature vs. a
distributional shape parameter).

**Pitfalls.** Lookahead bugs are dangerous precisely because they make results *look
better*, not obviously broken — the standard tripwire this repo uses throughout is
exactly that suspicion: any result that looks implausibly good (a QLIKE of 0.000000, an
IC well above a stated normal range, a forecast beating every alternative by an enormous
margin) should be investigated as a potential leak before being reported as a genuine
finding.

---

### Survivorship bias

**In one sentence.** A distortion that creeps in when your sample only includes things
that "survived" to be observable today — assets that failed, delisted, or disappeared
along the way are silently excluded, making the surviving sample look systematically
better (or just different) than the true, full population ever was.

**The maths.** No formula; a sample-construction concern. If $D_{\mathrm{observed}}
\subsetneq D_{\mathrm{true}}$, and the excluded elements weren't missing at random (they
failed *because* of something related to the very thing being studied), any statistic
computed on $D_{\mathrm{observed}}$ can be biased relative to the true population.

**Why it is here.** Notebook 3's own write-up flags this directly: its cross-sectional
universe (BTC, ETH, BNB, SOL, XRP, ADA, DOGE, ...) is chosen and backtested starting from
today's known-large-caps, list of assets that happen to still exist and be liquid in
2026 — coins that failed or were delisted along the way are absent from the sample by
construction.

**Worked example.** Notebook 3 explicitly notes this bias is "not zero" but chooses to
document it rather than attempt a correction, given the practical difficulty of
reconstructing a genuinely survivorship-free crypto universe going back to 2021.

**Pitfalls.** Survivorship bias tends to make cross-sectional strategies (buy the assets
that will turn out to have done well) look artificially better in backtest than they
would have in real time, since the backtest's own asset universe was implicitly chosen
using information about which assets turned out fine — a related but distinct concern
from ordinary [lookahead bias](#lookahead-bias-leakage), since it's about *which
assets* are in the sample, not about *which time periods'* data leaked into a forecast.

---

### Overfitting

**In one sentence.** When a model fits the specific noise in its training data so
closely that it no longer generalizes to new data — it looks great on the data it was
built from and disappoints on anything else.

**The maths.** No single formula; the general symptom is a large gap between
in-sample and out-of-sample performance, or (relatedly)
[overparameterization](02-estimation-and-fitting.md#overparameterization) relative to
the effective sample size.

**Why it is here.** Notebook 1's write-up names this explicitly for its own trend-following
sweep: "picked by sweeping 756 combinations (3 intervals x 3 loss functions x 3 test sizes
x 28 feature combos) and taking best Sharpe... this shows how easy it is to overfit a
backtest, not a strategy to trade" — an honest, self-critical acknowledgment rather than
reporting the swept-best result as a genuine finding.

**Worked example.** Choosing the single best-performing configuration out of 756 tried
combinations, evaluated on the same data used to choose it, is close to a textbook
overfitting setup — with enough combinations tried, *some* will look good by chance
alone, regardless of whether any of them represents a real, repeatable effect.

**Pitfalls.** Overfitting doesn't require a complicated model — it can happen just as
easily through *search* (trying many simple models and keeping the best-looking one)
as through genuine model complexity; both are addressed by the same underlying fix
(evaluating the chosen result out-of-sample, ideally on data untouched by the search
itself), which is exactly why this repo insists on frozen transfer checks and holdouts.

---

### Multiple-testing in research

**In one sentence.** The same [multiple-testing problem](03-statistical-inference.md#multiple-testing-problem)
from formal hypothesis testing, applied to the broader practice of research itself: if
you try enough models, features, or configurations, *some* will look good purely by
chance, even with no real effect anywhere — a research-design concern, not just a
statistical-test-correction concern.

**The maths.** Same underlying logic as the formal
[multiple-testing problem](03-statistical-inference.md#multiple-testing-problem): with
$m$ genuinely null configurations tried, expect roughly $m\alpha$ to look "significant"
by chance at threshold $\alpha$, whether or not you ever compute a formal p-value for
each one.

**Why it is here.** Notebook 3's Phase 6 deflated Sharpe calculation explicitly uses
**81** (the true total count of configs evaluated: 27 features x 3 intervals) as its
trial count — not just the handful that survived the IC screen — precisely because the
correction needs to account for the *entire search*, not just the subset that happened
to look interesting afterward.

**Worked example.** `config_log.jsonl` logging all 81 configs (not just the 34 that
survived screening) is exactly the record needed to make an honest multiple-testing
correction possible after the fact — without it, there'd be no way to know how many
"tries" a surviving result should really be judged against.

**Pitfalls.** Reporting only the models/features that "worked" without disclosing the
full search that produced them is the single easiest way a research programme
accidentally overstates its own findings — this repo's discipline of logging every
evaluated configuration, not just survivors, is the direct structural fix.

---

### Walk-forward analysis

**In one sentence.** A backtesting/evaluation scheme that repeatedly slides a
train/test split forward through time — fit on one period, test on the immediately
following period, then move both windows forward and repeat — so a model is validated
across many different historical periods rather than just one arbitrary split.

**The maths.** For fold $i$: train on $[t_i, t_i + w_{\mathrm{train}})$, test on
$[t_i+w_{\mathrm{train}}, t_i+w_{\mathrm{train}}+w_{\mathrm{test}})$, then advance $t_i$
and repeat — every test fold strictly follows its own training fold in time.

**Why it is here.** `research.walk_forward_splits` and its panel analogue
`panel_walk_forward_splits` implement exactly this, underlying notebook 2's and 3's
multi-fold evaluation — this repo's rolling-refit forecasting contests
([04](04-volatility-models.md), [05](05-regime-models.md)) are effectively a very
fine-grained, continuous version of the same idea (a "fold" every refit rather than a
handful of large folds).

**Worked example.** Notebook 2's write-up fixed a real bug in exactly this machinery:
`eval_model_performance` crashed when a fold happened to have zero losing trades (an
empty array's `.mean()` returning `None`, then an assertion failing) — "guaranteed to
happen somewhere across dozens of folds," a bug walk-forward analysis's own repetition
across many folds was what surfaced.

**Pitfalls.** More folds generally means more statistical confidence in the *average*
result across periods, but also more opportunities for an edge-case bug (like the
zero-losing-trades crash above) to surface — walk-forward analysis is valuable
specifically because it stress-tests code against a wide variety of real historical
conditions, not just one convenient split.

---

### Frozen transfer check

**In one sentence.** Testing whether a finding (a winning model, a regime pattern) that
was established on one asset also holds, *without any re-tuning*, on a separate set of
assets that played no role in developing it — the standard check in this research
programme for whether a result is a genuine, general pattern or just an artifact of one
particular dataset.

**The maths.** No formula; a re-application discipline: take the exact model/parameters
established on the primary asset, apply them unchanged to new assets, and compare
performance — any re-tuning on the transfer assets would defeat the entire purpose.

**Why it is here.** Used throughout notebooks 3-5: BTC is always the primary,
fully-explored asset; ETH/SOL/DOGE/BNB/XRP are the frozen transfer set, checked (usually
at a single interval, for compute reasons on this hardware) without any re-fitting of
model choices.

**Worked example.** Notebook 4's Phase 3 found HAR-RV the best-by-QLIKE rung on BTC, but
the frozen transfer check showed it *significantly* beating every other rung (per
all-pairs DM) only on ETH/SOL, not on BTC/BNB/XRP/DOGE — read as "not a stable winner,"
not as "HAR-RV wins 5/6," per the [stability vs. magnitude](#stability-vs-magnitude)
standard below.

**Pitfalls.** A transfer check that's allowed to re-tune anything (even something as
small as re-choosing a window size per symbol) is no longer testing generalization — it's
testing a *new*, per-symbol fitting exercise, which defeats the whole point of freezing
the model first.

---

### Stability vs. magnitude

**In one sentence.** This programme's explicit standard for what counts as a genuine
finding: a result that *replicates consistently* across symbols/intervals, even if
individually modest, is reported as the real finding — a result that's large on one
asset but doesn't replicate is reported as unstable, not as a headline win.

**The maths.** No formula; a reporting convention, stated explicitly and applied
consistently: "stability outranks magnitude."

**Why it is here.** Established in notebook 3 and repeated verbatim in every subsequent
notebook's own gating language (Gate C in `NEXT_RUN_PROMPT.md` requires a winner to
"reproduce on the frozen transfer symbols... a model that wins on BTC and 2 of 5 transfer
symbols is reported as 'not stable,' not as 'wins on BTC'").

**Worked example.** Notebook 4's Phase 4 regime finding ("regimes predict volatility, not
direction") is reported as the headline result specifically *because* it replicated at
every one of the 5 transfer symbols, even though it's a narrower, less dramatic-sounding
claim than "found a tradeable regime signal" would have been.

**Pitfalls.** It is tempting to lead a write-up with the single most impressive-looking
number found anywhere in a search, even if it doesn't replicate — this standard exists
specifically to resist that temptation, and every notebook in this repo's history has
been held to it consistently rather than selectively.

---

### Pre-declared gates and pre-registration

**In one sentence.** Deciding, in writing, *before* running any model, exactly what
result would count as a genuine "win" — so the bar can't be quietly moved after seeing
which numbers came out favorably.

**The maths.** No formula; a discipline of writing down $H_0$/$H_1$, the significance
level, and the exact decision rule in advance, then applying it mechanically once results
are in.

**Why it is here.** `NEXT_RUN_PROMPT.md` §3 writes down Gates A through E — the *exact*
criteria for a density winner, a tail-calibration winner, stability, and the Phase 6
trigger — before a single model in notebook 5 is fit, explicitly "before you see the
numbers."

**Worked example.** Gate B's "all six quantile levels, never rejected by any of three
tests" is a deliberately strict, pre-committed bar — `NEXT_RUN_PROMPT.md` explicitly
anticipates the likely honest outcome is that nothing clears it, and states in advance
that this is "a fine result," precisely so a disappointing outcome can't retroactively
tempt a loosening of the bar.

**Pitfalls.** The entire value of a pre-declared gate evaporates the moment it's
adjusted after seeing results ("well, it almost passed, let's call that close enough") —
`NEXT_RUN_PROMPT.md`'s own closing line makes this explicit: "do not break that record
now by relaxing a gate written down in §3 before the run."

---

### Sharpe ratio

**In one sentence.** The standard measure of risk-adjusted return: how much return a
strategy earned, per unit of volatility it took on to earn it — a strategy that earns
modest returns very smoothly can have a *better* Sharpe ratio than one earning larger
returns much more erratically.

**The maths.** $\mathrm{Sharpe} = \frac{\mathbb{E}[r] - r_f}{\sigma(r)}$ (often
annualized by multiplying by $\sqrt{\text{periods per year}}$), where $r_f$ is a
risk-free rate (frequently approximated as 0 for a rough comparison).

**Why it is here.** The headline metric in notebooks 1-2's trading backtests, and the
input to [deflated Sharpe](#deflated-sharpe) below, which corrects it for the number of
configurations tried.

**Worked example.** Notebook 1's swept trend-following config showed a pre-fee Sharpe of
8.89 (implausibly high — a red flag on its own) that fell to a much more modest,
still-positive post-fee number once the fee-accounting bugs were fixed — illustrating how
sensitive a raw Sharpe number is to getting the underlying P&L calculation right in the
first place.

**Pitfalls.** A Sharpe ratio computed on a small number of trades, or on returns that are
themselves heavy-tailed (as this whole research programme has repeatedly found crypto
returns to be), can be a noisy, unstable estimate — its own inference (a confidence
interval around it, or a corrected version like deflated Sharpe) needs the same care as
any other statistic built from fat-tailed, possibly-non-i.i.d. data.

---

### Fixed-notional vs. equity-path return

**In one sentence.** Two different ways to express a trading strategy's total return: *equity-path* (compounded) return assumes a single growing equity base where each trade's dollar P&L becomes part of the new base for future trades; *fixed-notional* return instead computes each trade's percentage return against *its own* entry-time equity and sums those percentages arithmetically, measuring "did we call the direction right" independent of how big the book had grown by any particular trade.

**The maths.** Given $n$ trades with realized P&L $\pi_i$ and entry-time equity $E_i$ (strictly *before* trade $i$ opens, to avoid lookahead), fixed-notional return is $R_{\text{fixed}} = \sum_{i=1}^{n} \frac{\pi_i}{E_i}$ — a simple arithmetic sum. Equity-path return uses a single running equity $\text{Equity}_t = \text{Equity}_{t-1} + \pi_t$, so $R_{\text{equity}} = \frac{\text{Equity}_n}{\text{Equity}_0} - 1$ — a compounded product. For a series of trades each with return $r$ (each contributing $+r$ to fixed-notional sum), equity-path final return is $(1+r)^n - 1 \approx nr$ for small $r$, but grows superlinearly as $r$ increases: two consecutive +10% trades give equity-path return of $(1.1)^2 - 1 = 21\%$, but fixed-notional sum of $10\% + 10\% = 20\%$.

**Why it is here.** Notebook 11a's backtest harness computes both (see `spread_lib11.book_metrics`): fixed-notional return via `ret_eq()` (sum of `realized_pnl / equity_at_open` per trade), and equity-path return via the daily equity curve's growth. They answer different questions: fixed-notional asks "how well did we predict direction/magnitude, normalized to entry capital," independent of sequence and compounding; equity-path asks "how much did an investor's money actually grow." Both are reported because a strategy can look great on fixed-notional (correctly predicting many trades) while delivering modest equity-path return if the wins came early and the capital sat flat, or vice versa.

**Worked example.** A backtest on a $1,000,000 starting equity, two trades:
- Trade 1: enters at $1,000,000 equity, wins $100,000 (10% fixed-notional return), ending equity $1,100,000.
- Trade 2: enters at $1,100,000 equity, wins $110,000 (10% fixed-notional return), ending equity $1,210,000.
- Fixed-notional sum: 10% + 10% = 20% total.
- Equity-path return: $1,210,000 / $1,000,000 − 1 = 21%.
The 1% difference is small here but compounds to larger gaps over many trades; the fixed-notional metric directly reflects call accuracy while equity-path reflects actual money growth.

**Pitfalls.** The equity snapshot used for a trade's fixed-notional return *must* be taken strictly *before* that trade's opening day (or opening bar), not on the same day or after any fills occur — a same-day post-fill snapshot would let that trade's own contribution leak into its own denominator, creating a lookahead bias where a trade's return is partially computed using information about its own outcome. This repo's `ret_eq` function guards this by construction (the backtest loop passes `entry_equity` pre-filled from the state machine, never a same-day recomputed value). Additionally, fixed-notional return is only meaningful for strategies that consistently trade across the sample — a strategy with only a few huge wins and many small losses can have a high fixed-notional sum while posting losses on a per-trade basis if the wins happened in a small number of outsized contracts that were then downsized, confounding "call accuracy" with "position sizing choices."

---

### Deflated Sharpe

**In one sentence.** A correction to the ordinary Sharpe ratio that accounts for how
many different strategies/configurations were tried before reporting the best one — the
more configurations searched, the more a raw Sharpe ratio needs to be "deflated"
(penalized) before it can be trusted as evidence of a real, non-chance effect.

**The maths.** Builds on the [multiple-testing](#multiple-testing-in-research) logic:
given $N$ independent trials, even a genuinely zero-skill process is expected to produce
a *maximum* observed Sharpe ratio well above zero purely by chance, growing (roughly)
with $\log N$. Deflated Sharpe computes a p-value for the observed best Sharpe *against
this chance-maximum benchmark*, rather than against a naive zero-skill benchmark that
ignores how many trials were run. It also incorporates the actual sample skewness and
kurtosis of the returns (not assumed values), since both distort a Sharpe ratio's own
sampling distribution.

**Why it is here.** Notebook 3's Phase 6 uses this on all **81** evaluated
configurations (see [multiple-testing in research](#multiple-testing-in-research) above)
— this repo's own commit history documents a specific correction to this calculation:
"real skew/kurtosis for deflated Sharpe" (rather than assuming a normal-return baseline),
motivated directly by every prior notebook's finding that crypto returns are nowhere
close to normal.

**Worked example.** A raw Sharpe of, say, 1.5 on the single best-performing config out of
81 tried might deflate to a much less impressive, possibly non-significant number once
properly corrected for both the search size and the fat-tailed nature of the underlying
returns — exactly the point of running the correction rather than reporting the raw
number alone.

**Pitfalls.** Deflated Sharpe still relies on the loss/return series' higher moments
being estimable at all — per the [Student-t](01-probability-and-distributions.md#student-t-distribution)
table's own boundary cases, a return series whose kurtosis is effectively infinite (a
real possibility given this programme's own tail-index findings) makes even the
corrected calculation's own precision harder to trust fully, a caveat worth carrying
forward rather than treating deflation as a complete fix for every distributional
problem.

---

### Information coefficient

**In one sentence.** A correlation between a predictive signal's ranking of assets and
those assets' actual subsequent returns — the standard way to grade a *cross-sectional*
prediction (does this feature correctly rank which assets will do better than others?)
as opposed to a time-series prediction.

**The maths.** $\mathrm{IC}_t = \mathrm{Corr}(\hat{y}_{i,t}, y_{i,t})$ across assets $i$
at a fixed time $t$ (usually Spearman rank correlation), then averaged across many $t$ to
get a mean IC, with significance assessed via a
[HAC t-statistic](03-statistical-inference.md#newey-west-hac-standard-errors) on the
time series of per-bar ICs.

**Why it is here.** This is the central metric of notebook 3's entire Phase 4 screen —
`cross_sectional_ic` and `panel_ic` compute exactly this, screening 27 candidate
features x 3 intervals (81 configs) for which ones genuinely rank assets correctly.

**Worked example.** Notebook 3 found `mean_reversion_1` the strongest, most stable
signal (mean IC +0.042 at 4h, NW t-stat 14.2, 93.8% of months positive) — a real,
non-trivial but not enormous cross-sectional ranking ability, explicitly compared against
a stated "normal range" of 0.01-0.03 and a 0.10 lookahead-suspicion tripwire (nothing in
the 81-config screen tripped it).

**Pitfalls.** As notebook 3 itself notes, some features (`hour_sin/cos`, `dow_sin/cos`)
come back exactly `NaN` for IC not because of a bug, but because they have zero
cross-sectional variance by construction (identical for every asset at a given
timestamp) — a structural fact about what cross-sectional IC can and can't measure, not
a broken calculation.

---

### Tail-premium factor

**In one sentence.** A hypothesized cross-sectional [information coefficient](#information-coefficient)
strategy that ranks assets by a fitted tail-shape parameter (e.g. Hansen skew-t's degrees
of freedom $\nu$) rather than by price momentum or level, on the theory that a riskier
(fatter-tailed) asset should carry a return premium to compensate holders for that risk —
the same logic behind an equity value/quality risk premium, applied to tail shape
specifically.

**The maths.** No new formula beyond [information coefficient](#information-coefficient)
and [dollar-neutral weights](#turnover): rank symbols each bar by a rolling-fitted
$\nu_t$ (or $|\lambda_t|$, or a predicted expected shortfall), long one extreme, short
the other, exactly like any other cross-sectional factor.

**Why it is here.** Notebook 7's Phase D built this directly from notebook 6's own
per-symbol Hansen skew-t machinery (the slowest-moving signal available in this repo's
whole toolkit, since a tail-shape parameter refits only every 30 days). The IC is
computed and checked for significance BEFORE any portfolio is built — the pre-declared
rule specifically to stop a non-significant IC from being backtested into a spurious
Sharpe.

**Worked example.** `src/results/7_alpha_generation.md`'s D2 factor (long high-$\nu$/
thin-tail, short low-$\nu$/fat-tail) found a significant but NEGATIVE full-sample IC
(NW t=-3.26) — the opposite sign from its own "long thin tails" hypothesis — yet its
top/bottom-quintile portfolio backtested net-Sharpe-positive at all four origin offsets.
Dropping the single symbol with the most leg-bars (FTTUSDT, whose collapse during the
FTX exchange failure a low-refit-count tail fit happened to place at the extreme) flipped
the sign to clearly negative at every offset — the portfolio's apparent edge was a
single-symbol artifact, not a genuine cross-sectional tail-premium effect, caught by
extending this whole programme's own "no single-asset finding" discipline from symbols
already known to dominate (BTC in EVT) to a symbol identified only by checking leg
composition after the fact.

**Pitfalls.** A significant cross-sectional IC and a profitable top/bottom-quantile
portfolio are NOT the same test — Spearman IC uses the full ranked cross-section every
bar, while a dollar-neutral book only trades the extremes, so the two can even disagree
in sign if the true relationship is non-monotonic or if one or two symbols' idiosyncratic
histories dominate a thin extreme leg (as small as 6 symbols out of 30 here). Always
check leg composition, not just the summary Sharpe, before crediting a cross-sectional
factor result — especially when the pre-declared universe includes an asset with an
unusual, low-data-quality history like a collapsed exchange token.

---

### Transaction costs

**In one sentence.** The real costs of actually executing a trade (exchange fees,
bid-ask spread crossed, slippage) — a backtest that ignores these will overstate a
strategy's real-world profitability, sometimes catastrophically.

**The maths.** No universal formula; this repo models fees as a fixed proportional cost
per unit of position *change* (not a flat per-bar charge), converted correctly to log
returns.

**Why it is here.** Notebook 1's write-up documents two real, serious bugs found here:
fees were originally charged every bar regardless of whether a trade actually occurred
(should only apply when position size changes), and the fee's log-return conversion used
`log(fee)` instead of the correct `log(1 - fee)` — the latter turning a small,
reasonable fee into an accounting wipeout (`log(0.0001) = -9.2` per "trade").

**Worked example.** Fixing both bugs flipped notebook 1's headline results from large
losses to large gains — not because the underlying models improved at all, but because
the P&L accounting had been badly wrong the whole time. The write-up is explicit that
this "doesn't fix the actual problem: no real edge found yet" — a corrected cost model is
a prerequisite for an honest evaluation, not evidence of a real strategy on its own.

**Pitfalls.** Getting transaction cost accounting wrong can distort results in *either*
direction (overcharging, as in bug #1/#2 above, made a real edge look like a loss;
undercharging or omitting costs entirely makes a strategy with no real edge look
profitable) — always sanity-check the magnitude of a fee-adjustment's effect against a
simple manual calculation before trusting a backtest's fee-adjusted numbers.

---

### Turnover

**In one sentence.** How often, and how much, a strategy actually changes its positions
— high turnover means frequent, often large position changes, which directly drives up
[transaction costs](#transaction-costs) regardless of how good the underlying signal is.

**The maths.** Commonly measured as the sum of absolute position changes over some
period, e.g. $\sum_t |w_t - w_{t-1}|$ for position weights $w_t$.

**Why it is here.** Notebook 5's Phase 6 (gated, likely not run) pre-declares turnover
cost as one of its four judged metrics (alongside Sharpe, max drawdown, and exceedance
count) for comparing the EVT risk-overlay strategy against buy-and-hold — a direct
acknowledgment that a strategy which trades often to react to a changing tail-risk
forecast pays for that reactivity in costs.

**Worked example.** A vol-targeting or risk-overlay strategy that rescales exposure every
single bar in response to a noisy, rapidly-changing forecast could have very high
turnover even if its underlying signal is genuinely useful — the net benefit only shows
up after subtracting the resulting transaction cost drag, which is exactly why it's
listed as one of Phase 6's judged metrics rather than an afterthought.

**Pitfalls.** A strategy can look attractive on gross (pre-cost) Sharpe while having
turnover high enough to erase the entire edge after realistic costs — notebook 1's own
history (Sharpe 8.89 pre-fee collapsing after correct fee accounting) is the concrete
illustration of exactly this risk, even though that specific case was a fee-accounting
bug rather than a genuine high-turnover strategy.

---

### Turnover budgeting

**In one sentence.** Deliberately capping or reducing how often a strategy trades —
treating turnover as a scarce resource to spend on genuine signal rather than an
automatic byproduct of recomputing a ranking every bar — as a distinct lever from the
signal itself.

**The maths.** No single formula; it's a design choice applied on top of an existing
position-construction rule, evaluated by comparing [turnover](#turnover) and net Sharpe
before/after the intervention on an *otherwise identical* signal (same predictions,
different trading mechanics).

**Why it is here.** Notebook 7's whole starting premise, stated directly in
`src/results/3_cross_sectional_ic.md` and `6_distribution_zoo.md`: every alpha attempt
in this research programme so far has been gross-profitable and net-negative, i.e. cost,
not signal absence, is the thing that keeps failing. Phase A tests three independent
turnover-budgeting mechanisms — [hysteresis bands](#hysteresis-no-trade-band), weight
quantization (rounding position sizes to a coarse grid so sub-grid rebalances never
trigger a trade), and rebalance throttling (recomputing positions only every $k$-th
bar) — against the *same frozen predictions* used in `3_cross_sectional_ic.md`'s
cfg2_12h, specifically so a Sharpe change can only be attributed to trading mechanics,
never to a re-fit signal.

**Worked example.** cfg2_12h's own headline turnover was ~578/year (offset 0); holding
the identical model's predictions fixed and rebalancing only every 6th 12h bar instead
of every bar cut turnover 71% and moved net Sharpe from roughly flat/negative to
consistently positive across all four origin offsets — but the bootstrap 95% CI on
excess return over basket still included zero at every offset, so the point-estimate
improvement did not clear the pre-declared bar for a genuine edge (Gate TC, see
`src/results/7_alpha_generation.md`).

**Pitfalls.** A turnover-budgeting change that also happens to alter *which* predictions
get acted on (e.g. retraining a model per intervention, or reusing an unseeded model fit)
confounds "cost reduction helped" with "a different, luckier signal got tested" — the
predictions must be generated once, from a fixed seed, and reused unchanged across every
turnover intervention for the comparison to mean anything.

---

### Hysteresis / no-trade band

**In one sentence.** A rule that only lets a position *enter* a portfolio leg once it
clears a threshold, but only lets it *exit* once it has fallen past a wider threshold —
so a prediction hovering right at the cutoff, flickering in and out of the top/bottom
ranks from noise alone, doesn't trigger a trade on every flicker.

**The maths.** With a cross-section of $n$ ranked symbols and a target leg size
$k_{\text{enter}} = \lfloor n \cdot \text{top\_frac} \rfloor$: a symbol enters a leg once
it ranks in the top/bottom $k_{\text{enter}}$, but a symbol already in that leg is only
dropped once it falls outside the wider $k_{\text{exit}} = \lfloor n \cdot
(\text{top\_frac} + \text{band}) \rfloor$ — between the two thresholds, it holds its
previous bar's position rather than being re-ranked from scratch. At $\text{band}=0$,
$k_{\text{exit}} = k_{\text{enter}}$ and the rule collapses to plain top-$k$/bottom-$k$
selection every bar (no memory effect at all).

**Why it is here.** `src/research/tmp/alpha_lib7.py`'s `hysteresis_weights` is Phase A's
first turnover-reduction mechanism, built specifically because most of cfg2_12h's
turnover was suspected to be "rank noise" (small, contribution-free shuffles near the
top/bottom cutoff) rather than genuine signal-driven trading. `band=0.0` is required to
reproduce `research.dollar_neutral_weights` exactly — the correctness check every other
band's number depends on (`tests/test_alpha_lib7.py`).

**Worked example.** A symbol sitting at rank 5 of 30 with `top_frac=0.2` (top 6 symbols
long) that drifts to rank 7 next bar would exit the long leg immediately under plain
top-$k$ selection; with `band=0.1` ($k_{\text{exit}}=9$), it stays long until it falls
to rank 10 or worse — avoiding an exit-then-possible-re-entry round trip if it drifts
back to rank 6 the bar after.

**Pitfalls.** A wider band trades a real thing (turnover) for a real cost (positions lag
the ranking more, potentially holding a symbol whose relative attractiveness has
genuinely changed, not just noisily fluctuated) — the fact that turnover falls
monotonically as the band widens (verified as a tripwire in
`tests/test_alpha_lib7.py`) says nothing on its own about whether net Sharpe improves;
that has to be checked separately, band by band.

---

### ATR-based position sizing and why stop width and position size are the same decision

**In one sentence.** When a position is sized so that a stop-loss hit costs exactly $X$ dollars (a fixed fraction of portfolio equity), the *position size* and the *stop distance* are mathematically linked — widen the stop and the position automatically halves to keep dollar risk constant, which means sweeping stop width alone is not isolating a single variable; any honest test must vary either stop width or sizing, not both independently.

**The maths.** Fixed-fractional risk sizing computes position quantity as:
$$\text{qty} = \left\lfloor \frac{\text{equity} \times \text{risk\_pct}}{\text{atr} \times \text{stop\_atr\_mult}} \right\rfloor$$
where $\text{atr}$ is a volatility scale (ATR, or spread-change volatility), $\text{stop\_atr\_mult}$ is the multiple of volatility at which a stop is placed (e.g., 6 ATRs away), and $\text{risk\_pct}$ is the portfolio fraction risked per trade (e.g., 3%). The dollar risk of a stop hit is:
$$\text{stop\_dollar\_loss} = \text{qty} \times \text{atr} \times \text{stop\_atr\_mult} \times (\text{point\_value or 1})$$
which equals $\text{equity} \times \text{risk\_pct}$ by construction — constant, independent of how $\text{stop\_atr\_mult}$ is chosen. If $\text{stop\_atr\_mult}$ doubles (wider stop), the denominator doubles, so $\text{qty}$ halves, and the dollar-at-risk product $\text{qty} \times \text{stop\_atr\_mult}$ stays flat.

**Why it is here.** Notebook 11a's Phase 2 backtest (see `spread_lib11.FixedFractionalSizing`) applies this sizing formula to each trade. When Phase 3 (risk gates) tests whether a wider or narrower stop improves profitability, *any* observed improvement or degradation comes from changing the stop distance *and* the resulting position size together — they are not independent levers. This means a gate that says "wider stops are better" might actually mean "smaller positions are better" or vice versa, and the gate's design must account for this coupling or else it's testing a compound effect, not a single policy choice.

**Worked example.** Institutional setup: $1,000,000 portfolio equity, $\text{risk\_pct} = 3\%$, ATR = $500, spread entry price ~$10,000:
- Config A: $\text{stop\_atr\_mult} = 6$ → $\text{qty} = \lfloor 30,000 / (500 \times 6) \rfloor = 10$ contracts, dollar risk = $10 \times 500 \times 6 = 30,000$ (exactly 3%).
- Config B: $\text{stop\_atr\_mult} = 12$ (double width) → $\text{qty} = \lfloor 30,000 / (500 \times 12) \rfloor = 5$ contracts, dollar risk = $5 \times 500 \times 12 = 30,000$ (still exactly 3%).
- The same $30,000 at-stop loss is distributed differently: 10 contracts × 6 ATRs vs. 5 contracts × 12 ATRs, but the tail-end behavior differs (fewer larger moves vs. more smaller ones hit the stop at different frequencies and sequences).

**Pitfalls.** Confusing stop width with position sizing is easy when a backtest's results show "wider stops performed better" — the causal mechanism might be "smaller positions survived more edge cases" or "fewer trades trigger, reducing bad entries," not "tighter stops were premature exits." To cleanly test stop-width effects, hold position *size* constant and vary only stop distance (using, e.g., $\text{qty} = \text{constant}$ and letting dollar risk vary), or pre-declare that both are coupled and measure the *compound* effect of "wider stop + smaller position" as a single, named configuration choice, not as an isolated stop-width variable.

---

### Maximum drawdown

**In one sentence.** The single worst peak-to-trough decline a strategy's cumulative
value ever experiences over the evaluated period — a measure of the deepest hole an
investor would have had to sit through, as opposed to average or typical volatility.

**The maths.** For cumulative return/value series $V_t$:
$\mathrm{MDD} = \max_{t} \left(\max_{s \le t} V_s - V_t\right)$ (often expressed as a
percentage of the running peak) — the largest drop from any prior high point to any
subsequent low point.

**Why it is here.** One of notebook 5's four pre-declared Phase 6 judgment metrics
(alongside Sharpe, exceedance count, and turnover), specifically because Phase 6's whole
proposed application is a *risk-management* overlay — its entire value proposition is
about reducing the depth of drawdowns during high-predicted-tail-risk periods, which
Sharpe alone (a ratio of average return to average volatility) doesn't directly capture.

**Worked example.** Two strategies can have identical Sharpe ratios while having very
different maximum drawdowns, if one experiences its volatility as many small, frequent
swings and the other as one rare, catastrophic decline — exactly the kind of tail-specific
risk this whole research programme's EVT machinery is built to characterize and, in
Phase 6, potentially manage against.

**Pitfalls.** Maximum drawdown is a single-worst-case statistic computed on one
particular historical sample — a longer or different sample could easily show a worse (or
better) drawdown purely by chance, especially for a fat-tailed asset like crypto; it
should be reported alongside the sample period it was measured over, not as if it were a
guaranteed ceiling on future losses.

---

### Buy-and-hold baseline

**In one sentence.** The simplest possible strategy — buy the asset once and hold it,
doing nothing else — used as the reference every more elaborate strategy in this
research programme must beat to be worth its added complexity and cost.

**The maths.** No formula; cumulative return is just the asset's own price return over
the period, no trading, no fees beyond the single entry (and, if applicable, exit).

**Why it is here.** Notebook 5's Phase 6 (gated) pre-declares its comparison set as
"unmodified buy-and-hold and... the identical overlay driven by a normal-innovation
GARCH" — buy-and-hold is the zero-effort floor any tail-aware overlay must clear, and the
normal-GARCH overlay is the "did the *tail-specific* modeling actually add anything"
comparison, isolating the EVT contribution specifically from the general idea of a
vol-based overlay at all.

**Worked example.** Any risk-overlay strategy that underperforms simple buy-and-hold
(even if it has, say, a smoother ride/lower drawdown) needs its trade-off made explicit —
lower risk at the cost of lower return is a legitimate choice, but not automatically a
"win," which is exactly why multiple metrics (Sharpe, drawdown, turnover) are compared
together rather than picking a single one that happens to favor the more complex
strategy.

**Pitfalls.** Buy-and-hold on a genuinely volatile, historically-appreciating asset
(like BTC over this repo's own sample period) can be a surprisingly hard baseline to beat
on raw return alone — this is exactly why notebook 5's proposed application is framed
explicitly as a risk-management comparison (drawdown, tail exceedances), not a pure
return-maximization contest, since that is the more honest question given what this
research programme has actually established (regimes and tails predict risk, not
return).

---

### The replication crisis in factor investing

**In one sentence.** A body of peer-reviewed literature finding that published
return-predicting factors/anomalies systematically perform worse after publication than
they did in the original sample — evidence, roughly in proportion to how much of the
decay is attributable to real capital trading the anomaly away versus how much is
attributable to the original discovery being statistical noise (an [overfitting](#overfitting)
and [multiple-testing](#multiple-testing-in-research) story at the scale of an entire
academic field, not one research programme).

**The maths.** No single formula; the shared empirical pattern across this literature is
a measured gap between in-sample and out-of-sample (or pre- vs. post-publication)
risk-adjusted return for the same nominal strategy — the same [in-sample vs.
out-of-sample](#in-sample-vs-out-of-sample) distinction this repo applies to every
rolling-refit model, applied instead to an entire literature's worth of published
strategies.

**Why it is here.** Notebook 9's external research review surveyed this literature
directly to adjudicate whether this programme's own eight-notebook run of nulls looks
like genuine market efficiency or an artifact of an unusually strict internal bar
(`src/results/9_external_research_review.md`, hypotheses (c) and (d)). The literature
itself does not agree with itself: McLean & Pontiff (2016, Journal of Finance) find
published factor returns 26% lower out-of-sample and 58% lower post-publication; Jensen,
Kelly & Pedersen (2023, Journal of Finance) find over 80% of factors remain significant
once construction is made careful and consistent across 93 countries — a genuine,
unresolved Tier 1-vs-Tier 1 disagreement, reported as such rather than averaged into a
false consensus, per this repo's own standard for [conflicting evidence](#pre-declared-gates-and-pre-registration).
Separately, Harvey, Liu & Zhu (2016, Review of Financial Studies) argue the literature's
own historical significance bar (t-stat > 2.0, uncorrected for the true scale of factor
search) is too *loose*, not too strict, once corrected for the true number of factors
tested (~316 at the time) — directly informing whether this repo's own [deflated
Sharpe](#deflated-sharpe) threshold (>0.95) is defensible.

**Worked example.** Notebook 8's own commodity carry result (net Sharpe 0.90-0.95 at
every origin offset, deflated Sharpe probability 0.997, excess-return-vs-basket CI
including zero — does not fire Gate AC) is exactly the kind of result this literature's
own findings help contextualize: strong on an absolute-Sharpe basis (comparable to Man
AHL's disclosed 0.86 institutional track record) but not shown, by this repo's own
stricter excess-vs-passive-basket criterion, to beat simply holding the underlying asset
class — the replication-crisis literature's broader lesson (real edges are usually
smaller, and decay faster, than a first look suggests) argues for caution before treating
even a Sharpe-0.9 result as obviously real, not for relaxing the bar that caught it.

**Pitfalls.** "Most published factors don't replicate" and "most published factors do
replicate with careful construction" are not actually the same finding measured two ways
— McLean-Pontiff and Jensen-Kelly-Pedersen differ in method (raw replication of the
original specification vs. a standardized, implementation-aware reconstruction across a
much larger sample), so the honest reading is that BOTH the "markets learn and arbitrage
away known effects" story and the "naive replication understates real effects" story have
real peer-reviewed support, and treating either one alone as the literature's settled
answer would be exactly the false-consensus error this repo's own methodology
(§ above, [pre-declared gates](#pre-declared-gates-and-pre-registration)) is built to
avoid.

---

### Regime-conditional backtesting and its multiple-testing cost

**In one sentence.** Splitting a backtest by a market regime (contango/backwardation,
high/low volatility, ...) and reporting the strategy's performance *within* the regime
that happens to look best is one of the easiest ways to manufacture a false edge, because
every candidate regime definition — and every candidate boundary within a definition — is
itself a free parameter that was implicitly searched over, even when only one is ever
shown.

**The maths.** No new formula beyond [deflated Sharpe](#deflated-sharpe)'s own $n_{trials}$
input — the point is what belongs *in* that count. If $k$ regime definitions (or regime
boundaries within one definition) are computed and the best-looking one is reported, the
deflated-Sharpe denominator must include all $k$, not just the one shown — exactly the
same accounting a cross-sectional factor search already requires, applied to a
conditioning variable instead of a signal.

**Why it is here.** Notebook 7's Gate TF (a term-structure-conditioned factor that cleared
its numeric bar and then turned out to be a single-symbol artifact once checked
individually) and notebook 10a/10b's own regime-gated spread hypothesis (NEXT_PROMPT.md
sec 4.1) are both instances of the same trap: "it works when I condition on X" is *the*
most common route from a true null to a false positive, precisely because the conditioning
variable is rarely pre-registered as rigorously as the primary signal is.

**Worked example.** Notebook 10a pre-declares, in advance and before any backtest, at most
three term-structure regime-definition variants (raw sign, a magnitude deadband, a
persistence requirement) and states which is primary and why — a structural argument (does
the definition actually create a no-trade zone distinct from unconditional trading?), not
a look at which variant's backtest Sharpe was highest. All three variants that are run
still enter the deflated-Sharpe count for Gate SPR, whether or not they end up being the
one reported as primary.

**Pitfalls.** Declaring a regime definition "in advance" is only a real safeguard if
*nothing about its choice* was informed by having already seen the strategy's own
performance under it — declaring one of three variants primary based on a *descriptive*
property of the conditioning variable itself (e.g. "this definition creates a genuine
no-trade zone and the others don't") is legitimate; declaring it primary because its
backtest happened to have the highest Sharpe, even if written down before the notebook
formally "ends," is not — the boundary is whether the choice used the target metric at
all, not merely whether it appears early in a document.
