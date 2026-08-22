# 006 — Does the Tail Result Generalise?

## The question

Notebook 005 changed the programme's question from "who forecasts variance best" (answer: nobody,
everything ties) to "who has the best-calibrated conditional tail" — and something finally won:

- **GARCH-t beat every other model on log score, significantly, at 1h, 4h and 12h.**
- **GARCH-EVT cleared every one of 36 tail-coverage tests at 12h.**
- **Every thin-tailed model significantly understated its own 1% expected shortfall, at every
  interval, with zero exceptions.**

But all three were tested on **one asset**. And the headline density win was never
cross-sectionally checked at all — the transfer testing there was scoped to daily bars, which is
precisely the one interval where that win did *not* fire on BTC.

This notebook is a generalisation test of that specific prior result, not a new contest. The
question is which of the three claims survive contact with five more symbols, three more quantile
levels, and a wider search over what "fat-tailed" can even mean — and where they don't survive,
whether the honest next question is "was Student-t ever the right family" rather than "try GARCH-t
harder".

Symbol set unchanged from notebooks 003–005: BTC, ETH, SOL, DOGE, BNB, XRP, at 1h, 4h, 12h and 1d.

## First: does notebook 005 reproduce?

Before extending anything, three published numbers were re-derived from the committed outputs and
asserted: GARCH-t's 12h log score (2.623), the count of models clearing all 36 coverage tests at
12h (exactly one, GARCH-EVT), and the verdict at daily bars (no significant winner).

**All three reproduced exactly.** The premise of this notebook holds.

---

## Claim 1 — does GARCH-t's density win generalise?

The identical eight-model contest — same competitors, same pairwise significance testing, same
multiple-testing correction — re-run on all five transfer symbols at 1h, 4h and 12h. The driver was
validated first by reproducing BTC's own committed numbers to full floating-point precision before
any transfer symbol was run.

| Interval | GARCH-t best on | GARCH-t **significant winner** on | Generalises? | Fat-tailed cluster holds? |
|---|---|---|---|---|
| 1h | 6/6 | **5/6** (including BTC) | **Yes** | Yes |
| 4h | 6/6 | 3/6 | No | Yes |
| 12h | 4/6 | 1/6 (BTC only) | No | Yes |

**It generalises at hourly bars — the strongest cross-sectional replication anywhere in this
notebook.** GARCH-t is the best-scoring model on every symbol at every interval tested, and at
hourly bars it significantly beats every competitor on five of six symbols.

The lone exception, XRP, is a power artefact rather than a real loss. XRP's HAR-log-RV forecast has
only 899 valid overlapping bars at hourly against roughly 33,000 for every other model, because
log-variance is undefined on an unusually large share of XRP's zero-return hourly bars. The
comparison against it therefore lacks power even though GARCH-t scores nearly double its log score
(3.59 against 1.61).

At 4h and 12h it does **not** generalise — GARCH-t remains the most frequently best model but loses
significance against at least one competitor on a majority of symbols.

This bounds rather than retracts notebook 005's claim. **"GARCH-t is the best density for BTC at
1h, 4h and 12h" survives as "GARCH-t is the best density for crypto" only at hourly bars.** At 4h
and 12h it downgrades to "usually best, not reliably significantly best" — the honest, narrower
claim.

The weaker *cluster* pattern — that the best model is always a fat-tailed-innovation or log-variance
model even when the specific winner varies — **holds without exception at every interval and every
symbol checked.** That is the single most consistently replicating finding in this whole research
programme.

---

## Claim 2 — is the expected-shortfall understatement universal?

The strongest single claim in notebook 005, tested properly for the first time across the full
grid: 10 models × 4 intervals × 6 symbols × 6 quantile levels = **1,440 expected-shortfall tests**,
with the multiple-testing correction applied across the whole grid at once. Thirteen cells had
fewer than 10 violations and were excluded from the primary percentages. The driver was validated
first against BTC's own committed statistic and p-value to full precision.

**The claim does not clear the pre-declared universality bar** (which required 90% of cells
positive, 60% significantly positive, and *zero* significantly negative, pooled across the 1%, 2.5%
and 5% lower-tail levels). Actual: 82.5% positive, 72.9% significantly positive, and **28
significantly negative cells** survive correction — not the single genuine counterexample the rule
anticipated, but 28 of them.

The pooled failure hides a clean, level-dependent structure that is the more interesting result:

| Quantile level | Fraction positive | Fraction significantly positive | Significantly negative cells |
|---|---|---|---|
| 1% | 96.5% | 88.0% | 1 |
| 2.5% | 85.2% | 80.3% | 3 |
| 5% | 66.0% | 50.7% | 24 |

**At the 1% level — the level notebook 005 actually tested — the finding is essentially universal
and would have cleared the bar on its own.** It degrades steadily moving toward the body of the
distribution, and by the 5% level roughly a third of thin-tailed cells are indistinguishable from
calibrated, or even mildly conservative.

The 28 significantly negative cells concentrate overwhelmingly in the HAR models (24 of 28), at the
5% level (24 of 28), and at hourly bars (18 of 28) — large-sample cells where even a practically
tiny negative statistic (−0.02 to −0.76, an order of magnitude smaller than the 1–3 range seen for
genuine violations) reaches significance on statistical power alone rather than practical
magnitude.

The upper tail is asymmetric and materially weaker. The 99% level looks like a milder mirror of 1%
(95.0% positive, 56.4% significantly so, zero negative), but the 95% level is close to a coin flip
(46.9% positive, 16 significantly negative). **Downside tail risk is understated far more
consistently and severely than upside** — genuinely a tail-side effect, not a symmetric one.

### Do fat-tailed models actually fix it?

This is the practically load-bearing question. Fraction of cells where the model's expected
shortfall is *not* significantly miscalibrated, lower tail, adequately powered cells only:

| Model | Pass rate |
|---|---|
| GARCH-EVT | **65.3%** |
| GJR-EVT | 51.4% |
| GARCH-t | 26.4% |
| GJR-t | 18.1% |

EVT-based tail modelling is a materially better answer than a plain Student-t GARCH to "how do I
avoid underestimating my downside" — but neither is a universal fix, and plain GARCH-t only clears
the bar about a quarter of the time.

**Verdict:** "thin-tailed models underestimate expected shortfall" survives as a **1%-level,
downside-specific** claim — exactly the level and side notebook 005 tested — not the fully
universal one a single-level, single-asset test implied.

---

## A wider search: four new distribution families

If Student-t keeps almost-but-not-quite winning, maybe Student-t was never the right family. Four
new innovation densities were added, each standardised to unit variance behind a common interface:

- **Generalised error distribution** — one shape parameter, nests the normal exactly, every moment
  always finite.
- **Normal-inverse Gaussian** — two shape parameters, and the first genuinely skewed family in the
  repo.
- **Johnson SU** — two shape parameters, with a closed-form quantile function.
- **Hansen skew-t** — nests the standardised Student-t exactly at zero skew, verified to machine
  precision.

154 unit tests across the four families, covering unit variance, quantile round-trips, nesting
checks and graceful failure on pathological input.

Each is fitted in two stages: reuse the existing, already-tested normal-innovation variance
recursion, then fit each family's shape parameters by maximum likelihood on the resulting
standardised residuals. Deliberately not a joint fit — that's flagged as the natural next lever
rather than done here. Then scored into the identical contest as before, on all six symbols and all
four intervals.

### Does anything beat GARCH-t?

The bar is the same one used for the transfer test above: beats GARCH-t significantly on at least
five of six symbols, including BTC.

| Interval | GED | NIG | Johnson SU | Hansen skew-t |
|---|---|---|---|---|
| 1h | No (1/6) | No (4/6) | **Yes (5/6)** | **Yes (5/6)** |
| 4h | No (2/6) | **Yes (6/6)** | **Yes (6/6)** | **Yes (6/6)** |
| 12h | No (3/6) | **Yes (5/6)** | **Yes (5/6)** | **Yes (5/6)** |
| 1d | No (2/6) | No (2/6) | No (2/6) | No (2/6) |

This was pre-declared as a coin flip and it was not one. **Three of four new families clear the bar
at 4h and 12h; two clear it at 1h.**

The generalised error distribution is the one consistent non-winner, despite its shape parameter
landing somewhere genuinely informative — a median of 0.98–1.10 across every interval, close to the
Laplace boundary. That confirms notebook 005's anxiety about near-boundary degrees of freedom was
as much about *peakedness* as raw tail weight. It just isn't enough extra flexibility on its own: a
model that can also fit tail weight wins more often than one that can only fit peakedness.

Margins are small in absolute terms (roughly 0.01–0.02 of log score) but consistent in sign across
nearly every symbol. This was checked not to be a data-reuse artefact by confirming the per-symbol
scores scale sensibly with each asset's own volatility level.

The fitted shape paths agree with each other in a way that is itself a finding: **the skew
parameters of the normal-inverse Gaussian and the Hansen skew-t are both consistently negative at
1h, 4h and 12h** (roughly −0.01 to −0.04), independently fitted in two different families — and
**both flip to essentially zero at daily bars.** Two unrelated parameters agreeing on sign and on
exactly which interval the skew vanishes at. The Hansen family's own fitted degrees of freedom sit
at 3.5–4.1, meaningfully higher than GARCH-t's typically near-boundary values, because a model with
a separate skew dial no longer has to force all the asymmetry into pure tail weight.

GJR variants of the new families were not fitted. Notebook 005 already established that GJR's extra
parameter costs more in rolling-refit noise than it buys, and re-running that cross for four more
families was judged a compute-budget item worth skipping. Stated explicitly as a scope decision.

**This makes a trading application eligible for the first time**, since a certified density winner
now exists together with cross-sectional stability at the same interval.

---

## Modelling the violation process itself

Every coverage test in notebook 005 treats Value-at-Risk violations as a binary sequence and asks
whether it looks independent. One test counts them; the other sees only a one-step-back
alternative. But notebook 004 already measured waiting-time shapes of 0.52–0.85, meaning violations
clump on *longer* scales than a two-state chain can detect. So the tests most likely to be **passed**
are exactly the ones least able to see the actual failure mode.

The 1% violation process is modelled here directly, two ways:

- **Weekly violation counts** — a Poisson null against a negative binomial alternative, with a
  boundary-corrected likelihood ratio test (the correct 50:50 mixture, not a naive chi-squared).
- **Durations between consecutive violations** — a geometric null against a discrete Weibull, where
  a shape below 1 means a falling hazard, i.e. clustering.

Both were validated on synthetic data before being trusted on real returns.

Across all 240 cells (10 models × 4 intervals × 6 symbols):

**127 of 240 cells (52.9%) reject the independent-Bernoulli null** — a bare majority rather than a
clean sweep, but real and above the pre-declared 50% bar. 107 of 238 valid count cells show
significant over-dispersion; 114 of 235 valid duration cells show significant clustering; and
**215 of 235 (91.5%) have a fitted shape below 1 even where not individually significant.** The
clustering *direction* is nearly universal even where power is too thin to certify it.

By model, rejection rates range from 42% to 71%. **EVT models do not show a uniformly cleaner
violation process than everything else** — GARCH-EVT rejects at 62%, GJR-EVT at 42%. That qualifies
rather than confirms notebook 005's calibration finding: a model can pass the standard coverage
tests and still have violations that cluster in a way those coarser tests cannot see.

---

## Getting the EVT models into the density contest

Notebook 005 never entered its EVT models in the log-score contest, because normalising a
GPD-tails-plus-empirical-body density proved too fiddly to trust.

Fixed here **structurally rather than iteratively**: the density is built as three separately
normalised pieces — a generalised Pareto tail on each side and a kernel-density body in the middle —
each scaled by its own known weight. The three weights sum to 1 by construction, so the spliced
whole integrates to exactly 1 without ever numerically hunting for a rescaling constant. Verified
to about 1e−4 in a synthetic check.

**Continuity at the two splice points is not enforced** — a genuinely separate and harder problem,
reported rather than hidden. The relative jump in density height at each threshold runs 20–33%
(mean ≈ 0.28) across every refit here. Every log score computed is still valid, since the density
is exactly normalised; only the visual smoothness at the seam is approximate.

Scored into the ten-model contest at 12h, 4h and 1d. **Hourly bars were dropped** after 30+ minutes
of CPU time with no sign of finishing on three symbols simultaneously — a compute-budget limit on
this hardware, handled by dropping the interval and saying so.

**On BTC at 12h, both EVT models decisively beat every other model in this notebook's entire zoo** —
the original eight and all four new families — tied only with each other (p = 0.156 between them;
every other comparison p < 0.001). The single cleanest density result this notebook produced.

**And it does not replicate.** EVT is the single best model on only 2 of 6 symbols at 12h, 1 of 6 at
4h and 2 of 6 at 1d, and significantly dominates every non-EVT model on only 2 of 6 (12h) and 0 of
6 at both other intervals. BTC is the only symbol where its dominance is both total and significant.

This is the same "spectacular on BTC alone, does not transfer" pattern the programme keeps
surfacing — on point forecasts, on daily density, on the 4h/12h transfer above, and now on EVT's own
density. Reported with the same discipline rather than letting the BTC number stand in for a
general claim.

Refit health: 85–100% of rolling refits produced a valid spliced density; the rest failed a guard
and were skipped cleanly rather than propagating junk.

---

## The trading application — gated, and it ran

For the first time in this programme, the gate fired. Following its mechanical letter rather than
the explicitly stated expectation that it probably wouldn't, the application ran.

Two adaptations to the pre-declared specification, stated rather than silently substituted:

1. The specification called for a **1-day** conditional Value-at-Risk overlay. No result fired at
   daily bars anywhere in this notebook, so the interval was substituted to **4h**, where the
   density result fired most robustly (all three winning families at 6 of 6 symbols).
2. The "best certified density" — the specification's own explicit alternative to an EVT-conditional
   overlay — is **GARCH-NIG**, the highest-scoring family on BTC at 4h.

This is the first time this notebook touches the frozen holdout period (2025-07-01 onward). Run
once, unchanged, with no retuning against the result.

BTC had a severe holdout year (buy-and-hold Sharpe −1.41, total log return −58.6%, matching notebook
003's basket Sharpe of −1.79 over the same window — broad crypto weakness, not anything specific to
this application):

| | Buy-and-hold | GARCH-NIG overlay (net) | GARCH-normal overlay (net) |
|---|---|---|---|
| Sharpe | −1.41 | −1.47 | −1.48 |
| Total log return | −0.586 | −0.559 | −0.562 |
| Max drawdown | −0.749 | −0.719 | −0.721 |
| 1% VaR exceedances (n = 2,185, expected 21.9) | — | **35** | **60** |

**No free lunch, but a real and qualified result.** The overlay improves raw total return and
maximum drawdown modestly over unmodified buy-and-hold, but does **not** improve risk-adjusted
Sharpe net of costs — the turnover from scaling exposure up and down costs enough that Sharpe comes
out fractionally worse than simply holding.

What the overlay does demonstrate cleanly: **the flexible-density model's own risk signal is
dramatically better calibrated than the normal one's, during exactly the year that mattered.** 35
realised 1% exceedances against GARCH-NIG's predictions versus 60 against GARCH-normal's, both
against 21.9 expected. That is the out-of-sample confirmation of the expected-shortfall finding
above: thin-tailed models understate downside risk, and flexible densities are a materially better
though still imperfect answer.

---

## Bugs and near-misses

No bugs in this notebook's new modelling code survived past its own unit tests — 154 across the four
new densities plus 19 more on the contest machinery, all passing. That is a different outcome from
notebooks 004 and 005, whose bugs were all caught by reading numbers after the fact rather than by
a test written in advance.

One near-miss worth recording: the spliced density's hourly interval blew the compute budget, at
30+ minutes with no sign of finishing on three symbols at once. It was caught by direct observation
of elapsed wall-clock time rather than by a timeout the code itself enforced, and handled by
dropping the interval and saying so.

---

## Bottom line

**Notebook 005's three headline claims survive in bounded, not universal, form.**

- **The density win** transfers cleanly to hourly bars (5 of 6 symbols including BTC), not to 4h or
  12h — where GARCH-t remains most-often-best but not reliably significantly best — and never to
  daily. "GARCH-t is the best density for crypto" is now a certified claim at exactly one interval.
- **The tail-calibration result** is qualified twice over. EVT models do not have a uniformly
  cleaner violation process than everything else even where their coverage tests pass. And EVT's
  spectacular BTC-at-12h dominance is BTC-specific and does not transfer.
- **The expected-shortfall understatement** — the strongest claim in the whole programme — holds up
  best of the three, but as a 1%-level, downside-specific claim rather than the fully universal one
  a single-level, single-asset test implied. It degrades materially by the 5% level and is
  meaningfully weaker on the upside.

**Two genuinely new certified findings** notebook 005 could not have produced: a wider distribution
search (normal-inverse Gaussian, Johnson SU, Hansen skew-t) **beats GARCH-t cross-sectionally at 4h
and 12h**; and **violations demonstrably do not look like a homogeneous Poisson process**, on a bare
but real majority of cells, which qualifies every coverage-test pass in the programme.

**A gated trading application ran for the first time** — notebooks 003, 004 and 005 either found no
gate satisfied or failed cost and robustness checks after running. This one clears its own gate
honestly and still finds **no risk-adjusted edge net of costs**: better drawdown control and
dramatically better tail-risk calibration, but not a better Sharpe.

Six notebooks in, that is the most consistent finding of all. Crypto's tails are real, extreme, and
increasingly well characterised by this programme's own machinery — and still, every time a
tradeable application has actually been built and tested, transaction costs and risk-adjusted return
have refused to validate an edge.

## What to test next

- **A joint rather than two-stage fit for the new families**, to check whether two-stage fitting is
  costing them log score relative to what a joint fit could achieve. The margins are real but
  small, and this is the natural lever if a bigger win exists.
- **Enforce continuity, not just normalisation, in the spliced density.** The 20–33% jump at each
  splice point is a quantified limitation that a jointly constrained fit could close, and BTC's
  dramatic 12h dominance makes this the highest-value place to spend that effort.
- **Work out why EVT's BTC dominance doesn't transfer.** Refit success rates and continuity gaps
  look similar across symbols, so it isn't an obvious data-quality artefact. Comparing each symbol's
  own tail-index estimates against BTC's might explain why BTC specifically rewards a
  semiparametric tail this much.
- **A genuinely joint regime-and-tail model** — deferred from notebooks 004 and 005, now doubly
  motivated. Both the cluster finding (fat-tailed or log-variance always wins, identity varies) and
  the skew-parameter finding (two independent parameters flipping sign at the same interval) look
  like they could be regime-linked rather than interval-linked.

*Notebook: `src/research/006_distribution_zoo.ipynb`. New terminology — the four distribution
families, the boundary likelihood-ratio test, duration-based coverage tests, and the spliced EVT
density — is defined in `docs/`.*
