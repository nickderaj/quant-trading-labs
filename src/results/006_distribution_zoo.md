# Does the Tail Result Generalize? - Results Summary

## What

This notebook tests whether notebook 5's three headline claims — GARCH-t's density win, GARCH-EVT's tail-coverage calibration, and the near-universal understatement of expected shortfall by thin-tailed models — survive contact with more symbols, more intervals, and a wider search over candidate distribution families. It also extends the "distribution zoo" itself, adding four new innovation-density families (GED, NIG, Johnson SU, Hansen skew-t), modeling the VaR violation process directly rather than only testing binary coverage, building a properly normalized spliced EVT density, and running a gated trading application on the frozen holdout.

## Why

Notebook 5's three claims were all derived from a single asset (BTC), and the strongest claim — GARCH-t's log-score win — was only cross-sectionally checked at one interval (1d), precisely where the win did not fire on BTC itself. Given this whole research programme's repeated pattern of results that look spectacular on BTC alone and fail to transfer, this notebook exists to determine which of notebook 5's claims are real, generalizable facts about crypto markets versus artifacts of testing one asset too narrowly.

## How

Using the frozen 6-symbol panel at 1h/4h/12h/1d, the notebook reran notebook 5's identical density contest across all transfer symbols (Phase 1), ran the Acerbi-Székely expected-shortfall test across a full 1,440-cell grid of models/intervals/symbols/quantile levels (Phase 2), fit four new innovation families and re-ran the contest (Phase 3), modeled violation counts and durations with dedicated statistical tests instead of only binary coverage tests (Phase 4), built a structurally normalized (rather than iteratively rescaled) spliced GPD-tails-plus-KDE-body EVT density (Phase 5), and — since a gate fired — ran a gated overlay application on the frozen holdout period (Phase 6).

## Results

GARCH-t's win transfers cleanly only at 1h (5/6 symbols significant), not at 4h/12h/1d. The expected-shortfall understatement finding holds up best of the three, but only as a 1%-level, downside-specific claim — it degrades materially toward the 5% level and the body of the distribution. EVT's spectacular BTC-only tail dominance does not replicate cross-sectionally, and EVT models do not show a uniformly cleaner violation process than other models even where they pass standard coverage tests. Two new findings emerged: NIG, Johnson SU, and Hansen skew-t all beat GARCH-t cross-sectionally at 4h/12h, and VaR violations show real, statistically detectable clustering in a bare majority (53%) of tested cells. The gated overlay application (GARCH-NIG vs. buy-and-hold on BTC's holdout year) improved raw return and drawdown modestly and produced a dramatically better-calibrated tail-risk signal, but did not improve net Sharpe after costs — consistent with this programme's recurring finding of no validated tradeable edge.

Notebook 5 changed this research programme's question from "who forecasts variance
best" (notebook 4: nobody, everything ties on QLIKE) to "who has the best-calibrated
conditional tail" - and something won: **GARCH-t beat every other model on log score,
significantly, at 1h/4h/12h on BTC** (Gate A), **GARCH-EVT cleared every one of 36
coverage tests at 12h on BTC** (Gate B), and **every thin-tailed model significantly
understated its own 1% expected shortfall, at every interval, with zero exceptions**
(the Acerbi-Székely result). But all three claims were tested on **one asset**
(BTC, and BTC alone) and, for Gate A's headline win, at **three of four intervals
without any cross-sectional check at all** - Phase 5 of notebook 5 was scoped to 1d
only, which is precisely the interval where Gate A did *not* fire on BTC.

This notebook is a generalization test of that specific prior result, not a new
contest. A reader arriving here cold should take away: **which of notebook 5's three
claims survive contact with five more symbols, three more quantile levels, and a wider
search over what "fat-tailed" can mean** - and where they don't survive, whether the
honest next question is "was Student-t ever the right family" rather than "try
GARCH-t harder."

Primary + frozen transfer set unchanged from notebooks 3-5: **BTCUSDT, ETHUSDT,
SOLUSDT, DOGEUSDT, BNBUSDT, XRPUSDT**, at 1h/4h/12h/1d. Machinery:
`src/research/tmp/dist_lib.py` and `dist_lib5.py` (notebooks 4/5's own, reused
unmodified), and the new `src/research/tmp/dist_lib6.py` (Gate-A/DM/BH machinery
factored out for reuse, four new innovation-density families under
`src/research/tmp/densities/`, the Phase 4 violation-process PMFs, and the Phase 5
spliced EVT density). Terminology defined from scratch, grounded in this repo's own
numbers, in `docs/` (start at `docs/README.md`); this notebook adds entries for GED,
NIG, Johnson SU, Hansen skew-t, the boundary likelihood-ratio test, duration-based
coverage tests, and the spliced EVT density.

## Phase 0 - Reproduction check

Before extending anything, `run_repro_check.py` re-derived three published notebook-5
numbers directly from the committed JSONs and asserted each: GARCH-t's 12h log score
(2.623), the count of models clearing all 36 coverage tests at 12h (exactly 1,
GARCH-EVT), and the Gate A verdict at 1d (no significant winner). **All three
reproduced exactly.** The premise of this notebook - that notebook 5's numbers are
correct - holds.

## Phase 1 - Transfer: does GARCH-t's win generalize?

The single highest-value follow-up notebook 5 itself flagged. Re-ran the identical
8-model density contest (same competitors, same all-pairs DM, same BH-adjusted
bootstrap p-values) on all five transfer symbols at 1h, 4h, 12h - the driver was
validated first by reproducing BTC's own committed 12h numbers to full float
precision before any transfer symbol was fanned out.

| interval | GARCH-t best on | GARCH-t **significant** winner on | Gate T fires? | cluster (fat-tailed/log-RV) holds? |
|---|---|---|---|---|
| 1h  | 6/6 | **5/6** (incl. BTC) | **Yes** | Yes |
| 4h  | 6/6 | 3/6 | No | Yes |
| 12h | 4/6 | 1/6 (BTC only) | No | Yes |

**Gate T fires at 1h - the strongest cross-sectional replication anywhere in this
notebook.** GARCH-t is the best-log-score model on every symbol at every interval
tested, and at 1h it significantly beats every competitor on 5 of 6 symbols including
BTC. The lone exception, XRP, is a genuine power artifact rather than a real loss:
XRP's HAR-log-RV forecast has only 899 valid overlapping bars at 1h against ~33,000
for every other model (log(rv) is undefined on an unusually large share of XRP's
zero-return 1h bars), so the DM test against it lacks power even though GARCH-t scores
nearly double HAR-log-RV's log score there (3.59 vs. 1.61).

At 4h and 12h, Gate T does **not** fire - GARCH-t remains the modal best model but
loses significance against at least one competitor on a majority of symbols. This
bounds, rather than retracts, notebook 5's own claim: **"GARCH-t is the best density
for BTC at 1h/4h/12h" only survives as "GARCH-t is the best density for crypto" at
1h.** At 4h/12h it downgrades to "usually best, not reliably significant best" - the
honest, narrower claim, per this whole research programme's own standard that
stability outranks magnitude.

The weaker "cluster" pattern notebook 5 found at 1d (the best model is always
fat-tailed-innovation or log-RV, even when the specific winner varies) **holds without
exception at every interval and every symbol checked here** - the single most
consistently-replicating finding in this whole research programme.

## Phase 2 - Is the ES-underestimation finding universal?

The strongest single claim in notebook 5, tested properly for the first time: the full
grid (10 models x 4 intervals x 6 symbols x 6 quantile levels = 1,440
Acerbi-Székely cells, BH-adjusted across the whole grid at once). 13 cells had under 10
violations and were excluded from the primary percentages (reported as a sensitivity
check in the committed JSON). The driver was validated first against BTC's own
committed 12h GARCH-t Z-statistic and p-value (0.6552012471862585 / p=0.0) to full
float precision.

**Gate U does not fire as pre-declared** (needs 90% positive / 60% significantly
positive / zero significantly negative, pooled across the 1%/2.5%/5% lower-tail
levels): 82.5% positive, 72.9% significantly positive, and **28 significantly-negative
cells** survive BH correction - not the single genuine counterexample the gate's rule
anticipated, but 28 of them, reported prominently rather than averaged away.

The pooled failure hides a clean, level-dependent structure that is the more
interesting result:

| level | frac. positive | frac. significantly positive | significantly negative cells |
|---|---|---|---|
| 1%   | 96.5% | 88.0% | 1  |
| 2.5% | 85.2% | 80.3% | 3  |
| 5%   | 66.0% | 50.7% | 24 |

**At 1% - the level notebook 5 actually tested - the finding is essentially universal
and would have cleared Gate U on its own.** It degrades steadily moving toward the
body, and at 5% roughly a third of thin-tailed-model cells are indistinguishable from
calibrated or even mildly conservative. The 28 significant-negative cells concentrate
overwhelmingly in HAR-RV/HAR-log-RV (24/28) at the 5% level (24/28) and at 1h (18/28) -
large-n cells where even a practically tiny negative Z (-0.02 to -0.76, an order of
magnitude smaller than the 1-3 range seen for genuine violations) reaches significance
on statistical power alone, not practical magnitude.

The upper tail is asymmetric and materially weaker: 99% (95.0% positive / 56.4%
significantly positive / 0 negative) looks like a milder mirror of 1%, but 95% is close
to a coin flip (46.9% positive, 16 significantly negative). **Downside tail risk is
understated far more consistently and severely than upside** - genuinely a tail-side,
not a magnitude-symmetric, effect.

**Gate U-fat** (the practically load-bearing question - do the fat-tailed/EVT models
actually fix this?): pass fractions (Z not significantly different from 0, lower-tail,
powered cells) are **GARCH-EVT 65.3%, GJR-EVT 51.4%, GARCH-t 26.4%, GJR-t 18.1%**.
EVT-based tail modelling is a materially better answer than a plain t-innovation GARCH
to "how do I avoid underestimating my downside" - but neither is a universal fix, and
plain GARCH-t only clears the bar about a quarter of the time.

**Bottom line for Phase 2**: "thin-tailed models underestimate expected shortfall"
survives as a 1%-level, downside-specific claim - the level and side notebook 5 tested
- not the fully universal one a single-level, single-asset test implied.

## Phase 3 - The wider distribution zoo

### New families

Four new innovation families, fanned out one subagent per family, each to its own file
under `src/research/tmp/densities/` with a fixed interface (`fit`/`logpdf`/`ppf`/`es`),
all standardized to unit variance:

- **GED** (generalized error) - one shape parameter, nests the normal exactly at
  kappa=2, every moment always finite.
- **NIG** (normal-inverse Gaussian) - two shape parameters, the first genuinely skewed
  family in this repo's zoo.
- **Johnson SU** - two shape parameters, closed-form quantile function.
- **Hansen skew-t** - nests the standardized Student-t exactly at lambda=0 (verified
  to ~1e-15, machine precision).

154 unit tests across the four families (unit variance, ppf/cdf round-trips, nesting
checks, `fit` returning `None` on pathological input), all passing.

### The contest: does anything beat GARCH-t?

Two-stage fit (`dist_lib6.fit_garch_zoo_two_stage`): reuse `dist_lib.fit_garch11`'s own
unchanged, already-tested normal-innovation variance recursion, then let each family's
own already-tested `fit` MLE the shape on the resulting standardized residuals -
deliberately not a joint fit (see the function's own docstring). Scored into the
identical log-score/all-pairs-DM/BH contest as Phase 1, all six symbols, all four
intervals.

**Gate P fires at 4h and 12h**, the same "beats GARCH-t significantly on >=5/6 symbols
including BTC" bar Gate T uses:

| interval | GED | NIG | Johnson SU | Hansen skew-t |
|---|---|---|---|---|
| 1h  | No (1/6) | No (4/6) | **Yes (5/6)** | **Yes (5/6)** |
| 4h  | No (2/6) | **Yes (6/6)** | **Yes (6/6)** | **Yes (6/6)** |
| 12h | No (3/6) | **Yes (5/6)** | **Yes (5/6)** | **Yes (5/6)** |
| 1d  | No (2/6) | No (2/6) | No (2/6) | No (2/6) |

This was pre-declared as a coin flip and it was not one: three of four new families
clear the bar at 4h and 12h, two do at 1h. **GED is the one consistent non-winner**,
despite its own shape parameter landing somewhere genuinely informative - median kappa
0.98-1.10 across every interval, close to the Laplace boundary, confirming that
notebook 5's fitted-t-df-near-2 anxiety was as much about peakedness as raw tail
weight. It just isn't, on its own, enough extra flexibility: a model that also gets to
fit tail weight (via its own df) wins more often than one that only gets to fit
peakedness. Margins are small in absolute log-score terms (~0.01-0.02) but consistent
in sign across nearly every symbol - verified this is not a data-reuse artifact by
checking the per-symbol scores scale sensibly with each asset's own volatility level.

Shape paths agree with each other in a way that is itself a finding: **NIG's beta and
Hansen's lambda are both consistently negative at 1h/4h/12h** (roughly -0.01 to -0.04),
independently fit in two different families, and **both flip to essentially zero at
1d** - two unrelated skew parameters agreeing on sign and on exactly which interval the
skew vanishes at. Hansen's own fitted nu sits at 3.5-4.1 (median), meaningfully higher
than GARCH-t's typically-near-boundary df, because a model with a separate skew dial no
longer has to force all the asymmetry into pure tail weight.

GJR variants of the new families were not fit - notebook 5 already established GJR's
extra parameter costs more in rolling-refit noise than it buys, and re-running that
cross for four more families was judged a compute-budget item worth skipping, stated
explicitly as a scope decision.

**This reopens Gate D.** Gate P's own firing criterion already requires cross-sectional
stability at the interval it fires; Gate D's "a density winner exists and is
cross-sectionally stable at the same interval" is satisfied at 4h and 12h (and 1h, via
Gate T). See Phase 6.

## Phase 4 - The violation process itself

Every coverage test in notebook 5 treats VaR violations as a binary sequence and asks
whether it looks i.i.d. Bernoulli - Kupiec counts them, Christoffersen independence
sees only a lag-1 alternative. Notebook 4 already measured gamma waiting-time shapes of
0.52-0.85 (violations clump on longer scales than a 2-state chain can see), so the
tests most likely to be *passed* are exactly the ones least able to see the actual
failure mode.

Modelled the 1% violation process directly, two ways, fit-once and descriptive (same
status as notebook 5's own Hill estimator): weekly violation counts (Poisson null vs.
negative binomial, a genuine MLE with a boundary-corrected LR test - the 50:50
chi2_0/chi2_1 mixture, not a plain chi2_1) and durations between consecutive violations
(geometric null vs. discrete Weibull, beta<1 meaning a falling hazard, i.e.
clustering). Both validated on synthetic data before trusting them on real returns.

Across all 10 models x 4 intervals x 6 symbols = 240 cells:

**Gate V fires**: 127/240 cells (52.9%) reject the i.i.d.-Bernoulli null - a bare
majority, not a clean sweep, but real and above the pre-declared 50% bar. 107/238 valid
count cells show significant overdispersion; 114/235 valid duration cells show
significant clustering, and **215/235 (91.5%) have a fitted beta<1 point estimate even
where not individually significant** - the clustering *direction* is nearly universal
even where power is too thin to certify it.

By model, rejection rates range from 42% (HAR-log-RV, range, GJR-normal, GJR-EVT) to
71% (trailing-std). **EVT models do not show a uniformly cleaner violation process than
everything else** - GARCH-EVT rejects at 62%, GJR-EVT at 42% - qualifying rather than
simply confirming notebook 5's Gate B: a model can pass Kupiec/Christoffersen and still
have violations that cluster in a way this coarser pair of tests cannot see.

## Phase 5 - A normalized EVT density

Notebook 5's own d8/d9 (GARCH-EVT, GJR-EVT) never entered the log-score contest -
normalizing a GPD-tails-plus-empirical-body density proved too fiddly to trust, and the
sanctioned fallback was "an honest partial entry beats a hand-waved density." Fixed
here **structurally, not iteratively**: `dist_lib6.fit_spliced_evt_density` builds the
density as three separately-normalized pieces (a GPD tail each side, a Gaussian-KDE
body in the middle), each scaled by its own known weight (k/n per tail, 1-2k/n
interior) - three weights that sum to 1 by construction, so the whole spliced density
integrates to exactly 1 without ever numerically hunting for a rescaling constant on
the spliced whole (verified to ~1e-4 in a synthetic check).

**Continuity at the two splice points is not enforced** - a genuinely separate, harder
problem, reported honestly rather than hidden: the relative jump in density height at
each threshold runs 20-33% (mean ~0.28) across every refit in this notebook. Every log
score computed is still valid (a genuine, exactly-normalized density); only the visual
smoothness at the seam is approximate.

Scored into the 10-model contest at 12h/4h/1d (**1h was dropped** after blowing 30+
minutes of CPU time with no sign of finishing on three symbols simultaneously - the
same sanctioned fallback Phase 1 used, for the same reason: compute budget on a
Raspberry Pi).

**On BTC at 12h, GARCH-EVT and GJR-EVT decisively beat every other model in this
notebook's entire zoo** - the original 8-model set and all four Phase 3 families - tied
only with each other (DM p=0.156 between them; every other pairwise comparison
p<0.001). The single cleanest Gate A result this notebook produced.

**It does not replicate cross-sectionally.** EVT is the single best model on only 2/6
symbols at 12h, 1/6 at 4h, 2/6 at 1d, and significantly dominates every non-EVT model on
only 2/6 (12h), 0/6 (4h), 0/6 (1d) - BTC is the only symbol where its dominance is both
total and significant. This is the same "spectacular on BTC alone, does not transfer"
pattern this whole research programme keeps surfacing (notebook 4's point-forecast
ranking, notebook 5's own 1d Gate A, Phase 1's own 4h/12h Gate T shortfall) - reported
with the same discipline rather than let the BTC number stand in for a general claim.

Refit health: 85-100% of rolling refits produced a valid spliced density (the rest
failed a tail/KDE guard and were skipped cleanly, same "null rather than propagate
junk" convention as every fitter in this repo).

## Phase 6 - Application (GATED - and it ran)

Gate D fired (Gate P at 4h/12h). Per the pre-declared rule this notebook's own gated
application therefore runs - following the mechanical letter of the gate rather than
the explicitly-stated expectation that it likely would not.

Two flagged adaptations to the pre-declared spec, stated explicitly rather than
silently substituted: (1) the spec says "1-day 1% conditional VaR"; no gate fired at 1d
anywhere in this notebook, so the interval was substituted to **4h**, where Gate D
fired most robustly (all three winning families at 6/6 symbols). (2)
"Best-certified-density-conditional" (the spec's own explicit alternative to
"EVT-conditional") is **GARCH-NIG**, the highest-log-score zoo family on BTC at 4h.

This is the first time this notebook touches the frozen holdout (2025-07-01 onward) -
run once, unchanged, no retuning against the result, the exact discipline notebook 3's
own Phase 7 holdout run used. `HOLDOUT_START` was used once before by notebook 3, for
an unrelated question (cross-sectional stock-picking); notebook 4's own "note on the
holdout" treats each notebook's gated application as a separate, once-only touch rather
than a shared, exhausted budget, and this run follows that precedent.

BTC had a severe holdout year (buy-and-hold Sharpe -1.41, total log return -58.6%,
matching notebook 3's own basket buy-hold Sharpe of -1.79 over the same window - broad
crypto weakness, not specific to this application):

| | buy-and-hold | GARCH-NIG overlay (net) | GARCH-normal overlay (net) |
|---|---|---|---|
| Sharpe | -1.41 | -1.47 | -1.48 |
| total log return | -0.586 | -0.559 | -0.562 |
| max drawdown | -0.749 | -0.719 | -0.721 |
| 1% VaR exceedances (n=2185, expected 21.9) | - | 35 | 60 |

**No free lunch, but a real, honest, qualified result.** The overlay improves raw total
return and max drawdown modestly over unmodified buy-and-hold, but does **not** improve
risk-adjusted Sharpe net of costs - turnover from scaling exposure up and down costs
enough that Sharpe comes out fractionally worse than simply holding. This matches this
whole research programme's consistent "no validated tradeable edge" finding, even where
the underlying tail-risk science is genuinely sound.

What the overlay does demonstrate cleanly: **GARCH-NIG's own risk signal is
dramatically better calibrated than GARCH-normal's during exactly the year that
mattered** - 35 realized 1% exceedances against GARCH-NIG's own predictions versus 60
against GARCH-normal's (both against 21.9 expected) - the practical, out-of-sample
confirmation of Phase 2's Gate U/Gate U-fat finding that thin-tailed models understate
downside risk and fat-tailed/flexible-density models are a materially better, though
still imperfect, answer.

## Bugs found

None specific to this notebook's own new modelling code survived past its own unit
tests (154 tests across the four new densities, 19 across `dist_lib6.py`'s Phase
4/5/6 machinery, all passing) - a different outcome from notebooks 4/5, whose bugs were
all caught by reading numbers after the fact rather than by a test written in advance.
Two things worth recording as near-misses rather than bugs:

1. **A duplicate background process accidentally killed mid-run** during Phase 3's
   family-implementation fan-out (two `pytest` invocations on the NIG test suite ended
   up racing for the same CPU cores under the 3-concurrent-agent cap; the redundant one
   was identified and killed cleanly, the surviving one finished and passed 46/46).
   Caught by checking `ps aux` against expected process counts before trusting a
   "completed" status, not by an assertion.
2. **The Phase 5 spliced density's 1h interval blew the compute budget** (30+ minutes
   with no sign of finishing on three symbols simultaneously) - caught by direct
   observation of elapsed wall-clock time against the sanctioned budget, not by a
   timeout the code itself enforced, and handled by dropping 1h and saying so, per this
   whole research programme's own "negative results / scope reductions are complete
   deliverables, not failures" standard.

## Bottom line

**Notebook 5's three headline claims survive in bounded, not universal, form.**

- **Gate A (GARCH-t's density win)**: transfers cleanly to 1h (5/6 symbols including
  BTC), not to 4h/12h (where GARCH-t remains modal-best but not reliably significant),
  and never to 1d. "GARCH-t is the best density for crypto" is now a certified claim -
  at exactly one interval.
- **Gate B/tail calibration (EVT works)**: qualified twice over in this notebook.
  Phase 4 shows EVT models do not have a uniformly cleaner violation process than
  everything else even where their coverage tests pass (Gate V). Phase 5 shows EVT's
  spectacular BTC-12h log-score dominance is BTC-specific and does not transfer
  cross-sectionally - the same pattern this research programme has now found on point
  forecasts (notebook 4), on 1d density (notebook 5), on 4h/12h transfer (this
  notebook's own Phase 1), and now on EVT's own density.
- **The ES-underestimation finding**: the strongest claim in the whole research
  programme, and it holds up best of the three - but as a 1%-level, downside-specific
  claim, not the fully universal one a single-level, single-asset test implied. It
  degrades materially by the 5% level and is meaningfully weaker on the upside.

**Two genuinely new, certified findings** notebook 5 could not have produced:
**Gate P** - a wider distribution search (NIG, Johnson SU, Hansen skew-t) beats
GARCH-t, cross-sectionally, at 4h and 12h - and **Gate V** - violations demonstrably do
not look like a homogeneous Poisson/Bernoulli process, on a bare but real majority of
cells, qualifying every coverage-test pass in this whole research programme.

**Gate D fired and Phase 6 ran** for the first time in this research programme -
notebooks 3, 4, and 5 either found no gate satisfied or (notebook 3's cross-sectional
signal) failed cost/robustness checks after running. This notebook's application
clears its own gate honestly and still finds **no risk-adjusted edge net of costs** -
better drawdown control and dramatically better tail-risk calibration, but not a better
Sharpe. Five notebooks into this research programme, that is now the single most
consistent finding of all: crypto's tails are real, extreme, and increasingly
well-characterized by this programme's own machinery, and still, every time a
tradeable application has actually been built and tested, transaction costs and
risk-adjusted return have refused to validate an edge.

## What to test next

- **A joint (not two-stage) MLE for the Phase 3 zoo families**, to check whether
  two-stage fitting is costing NIG/Johnson SU/Hansen skew-t log score relative to what
  a joint fit could achieve - Gate P's margins are already real but small, and a joint
  fit is the natural next lever if a genuinely bigger win is out there.
- **Enforcing continuity, not just normalization, in the spliced EVT density** - the
  20-33% relative jump at each splice point is a real, quantified limitation that a
  jointly-constrained GPD-scale/KDE-bandwidth fit could close, and BTC's own dramatic
  12h dominance makes this the highest-value place to spend that effort.
- **Why EVT's BTC dominance doesn't transfer** - Phase 5's own health diagnostics
  (refit success rate, continuity gap) look similar across symbols, so the transfer
  failure isn't an obvious data-quality artifact; a direct comparison of each symbol's
  own tail-index estimates (Hill, GPD xi) against BTC's might explain why BTC
  specifically rewards a semiparametric tail this much.
- **A genuinely joint regime + tail model** - still deliberately deferred from
  notebooks 4 and 5, now doubly motivated: Phase 1's own cluster finding (fat-tailed or
  log-RV always wins, identity varies) and Phase 3's own skew-parameter finding (NIG
  beta / Hansen lambda both flip sign at 1d) both look like they could be regime-linked
  rather than interval-linked per se.
