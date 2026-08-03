# Alpha Generation, Conditioned on What We Now Know About Risk - Results Summary

**The hypothesis this notebook tests: transaction cost, not signal absence, is what has
blocked every alpha attempt in this research programme so far.** Notebook 3 said it
outright ("the signal is real pre-cost, costs erase it"); notebook 6 said it again in
different words (the risk overlay's own signal was genuinely better calibrated and
*still* lost to buy-and-hold on Sharpe once turnover was charged). This notebook attacks
the cost problem directly, using the risk machinery notebooks 4-6 already built and
certified, before asking whether a genuinely different signal family survives where
price-based signals have not.

**The hypothesis does not survive its own most direct test.** Phase A held a
known-gross-profitable signal fixed and changed only its trading mechanics: turnover
fell as much as 71%/year and net Sharpe improved substantially at every origin offset,
and it still did not clear the pre-declared bootstrap-CI bar. If cost were the whole
story, the cheapest and most direct intervention - trade the identical signal less often
- should have been the one most likely to work. It came close and still fell short.

**All four pre-declared gates (TC, RG, CY, TF) come back null**, each for a different
and informative reason, detailed below. Per this notebook's own standard: six
independent, honestly-tested attempts at crypto alpha across this whole research
programme finding no tradeable edge is **evidence about the market, not evidence of a
failed research programme**. The risk findings of notebooks 4-6 remain exactly as valid
and useful as they were before this notebook ran, and that is said plainly here rather
than left to be inferred.

Universe: notebook 3's own frozen 30-symbol panel, unchanged. Machinery:
`src/research/tmp/alpha_lib7.py` (new - hysteresis/no-trade bands, weight quantization,
rebalance throttling, VaR-gating helpers), reusing `research.py`, `features.py`, and
`dist_lib6.py`/`densities/` unmodified. Terminology defined from scratch, grounded in
this repo's own numbers, in `docs/` (start at `docs/README.md`) - this notebook adds
entries for turnover budgeting, hysteresis/no-trade bands, carry/basis trades, and the
tail-premium factor.

## Phase 0 - Reproduction check

Before building anything on top of notebook 3's numbers, `run_repro_check7.py`
re-derived cfg2_12h's headline net/gross Sharpe (+0.42 / +1.32), its origin-shift flip to
-2.45 at offset 7, its realized annual fee drag (0.335-0.375%/yr across offsets), cfg1_4h
and cfg3_1d's negative net Sharpe at every offset, and the Phase 7 holdout Sharpe (-0.47
net / +0.74 gross) directly from the committed `backtest_results.json` and
`holdout_results.json`. **All reproduced to within 0.005.** The premise this notebook
builds on - that notebook 3's cost problem is real and correctly reported - holds.

## Phase A - Cut turnover on a known-gross-profitable signal

cfg2_12h's own OOS predictions were generated **once per origin offset**, with a fixed
seed (`research.set_seed(0)`), and never re-fit per intervention. This matters more here
than it might sound: `backtest_configs.py`'s model is unseeded, and notebook 3's own
inference-correction section already found that re-running cfg2_12h with unchanged code
flips its headline Sharpe from +0.42 to -1.22. Retraining per turnover variant would have
confounded "did the trading mechanics help" with "did this particular re-fit get lucky" -
exactly the ambiguity section 6's "changes only how the signal is traded, never what the
signal is" rule exists to prevent. One frozen signal per offset, four position-
construction variants tested against it.

Three interventions, plus one pre-declared combination:

- **A1 hysteresis / no-trade bands** (`alpha_lib7.hysteresis_weights`, bands
  `[0.05, 0.10, 0.15, 0.20]`, `0.0` reproducing `research.dollar_neutral_weights`
  exactly - the correctness check every other Phase A number depends on, verified in
  `tests/test_alpha_lib7.py` before trusting any live number).
- **A2 weight quantization** (round to the nearest 0.05 of gross).
- **A3 rebalance throttling** (`k in [2, 3, 6]`: recompute positions only every k-th
  12h bar, hold flat between).
- **A4 combined**: pre-declared mid-points (band=0.10, grid=0.05, k=3) tested once,
  before any individual result was seen.

### Results

| baseline (band=0, offset) | net Sharpe | turnover/yr |
|---|---|---|
| offset 0 | -0.180 | 492.5 |
| offset 7 | +0.134 | 486.1 |
| offset 14 | -0.040 | 488.5 |
| offset 21 | -0.343 | 487.0 |

(This baseline does not exactly match notebook 3's own logged cfg2_12h numbers -
expected, not a bug: notebook 3's model is unseeded, so a fresh, seeded run of the
identical code is methodologically identical but not bit-exact, the same phenomenon its
own inference-correction section documented.)

**Rebalance throttling is the effective lever.** At k=6, turnover falls 71%/year and net
Sharpe moves to consistently positive at every offset (+1.011, +1.053, +0.272, +0.587).
Hysteresis alone only cuts turnover 10-23% across its band grid - short of the
pre-declared 30% bar for a genuine intervention (0.05→11%, 0.10→16%, 0.15→20%, 0.20→23%).
Quantization at a 0.05 grid barely moves turnover at all (~0%): cfg2_12h's per-symbol
weights (top/bottom 6 of 30 symbols, gross 1.0) are already coarser than that grid, so
rounding to it changes almost nothing. The combined variant (A4) falls in between (55%
turnover reduction, net Sharpe +0.56/+0.90/+0.53/+0.37 across offsets).

| variant | turnover fall (avg. across offsets) | net Sharpe range across offsets |
|---|---|---|
| A1 hysteresis (best band, 0.20) | 22% | -0.31 to +0.08 |
| A2 quantize (grid 0.05) | ~0% | -0.29 to +0.26 |
| A3 throttle k=2 | 37% | -0.04 to +0.47 |
| A3 throttle k=3 | 51% | +0.49 to +0.89 |
| A3 throttle k=6 | 71% | +0.27 to +1.05 |
| A4 combined | 55% | +0.37 to +0.90 |

**Gate TC does not fire.** No variant's bootstrap 95% CI on excess return vs. basket
excludes zero, at any of the four origin offsets - despite point-estimate net Sharpe
reaching above 1.0 in the best throttled configuration. This is the outcome
`NEXT_PROMPT.md` wrote down in advance as the expected one: "turnover fell substantially
... net Sharpe improved somewhat ... whether it clears zero is genuinely unknown." It did
not clear zero. No net-above-gross-Sharpe tripwire fired at any of the 36
variant/offset cells checked.

Throttle k=6 (positive net Sharpe at all four offsets, 71% turnover reduction, the
largest genuine turnover intervention tested) is carried forward into Phase B as "the
best Phase A variant," per section 4's instruction, despite Gate TC not firing - Phase B
needs *a* baseline to gate, not a certified one.

## Phase B - Gate the signal on predicted tail risk

The first use of this programme's risk findings as an alpha input rather than a risk
report. Phase A's throttle-k6 book was gated on a causal 1% conditional VaR path from
**GARCH-NIG**, notebook 6's own best-certified density at 12h (Gate P), built with
`dist_lib6.rolling_garch_forecast_zoo` / `zoo_quantile_forecast` unchanged, fit per
symbol (30 symbols, ~3.2s/symbol for a full rolling refit sequence, benchmarked
directly). Two gating variants, both pre-declared:

- **B1 stand-down**: zero the whole book when the cross-sectional median predicted 1%
  VaR magnitude exceeds its own trailing 250-bar median by factor `k in [1.25, 1.5, 2.0]`.
- **B2 per-symbol tilt**: reuses `run_phase6_application.py`'s own `build_overlay_weight`
  unchanged, applied per symbol to shrink size before dollar-neutralizing.

Every comparison is gated vs. the **identical ungated** throttle-k6 book, on the same
signal, never against the raw notebook-3 baseline - otherwise Phase A's own improvement
would get miscredited to Phase B.

### Results

| offset | ungated (throttle k6) | B1 k=1.25 | B1 k=1.5 | B1 k=2.0 | B2 tilt |
|---|---|---|---|---|---|
| 0 | +1.011 | +0.919 (-0.092) | +0.928 (-0.083) | +1.007 (-0.004) | +0.924 (-0.087) |
| 7 | +1.053 | +0.829 (-0.224) | +0.904 (-0.149) | +1.070 (+0.017) | +1.006 (-0.047) |
| 14 | +0.272 | +0.020 (-0.252) | +0.003 (-0.269) | +0.256 (-0.016) | +0.326 (+0.054) |
| 21 | +0.587 | +0.678 (+0.091) | +0.639 (+0.052) | +0.583 (-0.004) | +0.589 (+0.002) |

(net Sharpe, delta vs. identical ungated in parentheses)

**Gate RG does not fire.** No gated variant improves net Sharpe by the pre-declared
≥0.20 bar over the identical ungated book at any origin offset - 12 of the 16
gated-vs-ungated deltas are actually negative, and B1's tighter thresholds (k=1.25/1.5)
hurt Sharpe by as much as -0.27. Drawdown does not improve consistently either (checked
per variant/offset; no clean pattern), so this is not even the "helps drawdown, hurts
Sharpe" result notebook 6's own Phase 6 application found - standing down during
high-predicted-tail-risk periods here costs more in forgone signal than it saves in
avoided drawdown.

This is a genuinely informative negative result, not a shrug: it directly answers the
open question `NEXT_PROMPT.md` posed in advance ("whether standing down during
high-tail-risk periods helps risk-adjusted return depends on whether high-tail-risk
periods are also high-expected-return periods"). On this book, they at least partially
are - qualifying the intuitive case for using notebook 6's risk science as a timing
overlay on top of an already-cost-reduced book. No net-above-gross-Sharpe tripwire fired
at any of the 16 cells checked.

## Phase C - Carry (funding rate) as a primary signal

Structurally different from every signal tested in this programme so far: a payment,
not a price prediction. Tested as a transparent, single-feature cross-sectional ranking
(deliberately not a fitted model, so a result can't be a fitting artifact and a null
can't be blamed on an under-trained net) at 4h/12h/1d, all four origin offsets, raw and
20-bar-z-scored.

**Sign convention, decided before any Phase C number was seen.** `NEXT_PROMPT.md`'s own
pseudocode passes `pred_col="funding_rate_zscore_20"` unmodified into
`dollar_neutral_weights`, which would long the highest-funding (payer) names - the
opposite of its own stated intent ("short the payers, long the receivers"). Perpetual
funding pays shorts when the rate is positive and longs when it is negative, so the
carry-consistent ranking is `pred_col = -funding_rate`. This also matches notebook 3's
own Phase 4 screening: raw `funding_rate`'s cross-sectional IC against forward return is
negative (-0.0095 at 4h). Both raw and z-scored variants use this corrected sign.

### Results

| interval | pred | best net Sharpe (any offset) | worst net Sharpe | gross range | turnover/yr |
|---|---|---|---|---|---|
| 4h | raw | -2.479 | -2.533 | -0.14 to -0.04 | ~965-971 |
| 4h | zscored | -3.396 | -3.629 | -0.33 to -0.11 | ~1260 |
| 12h | raw | -1.079 | -1.339 | +0.23 to +0.45 | ~674-681 |
| 12h | zscored | -2.188 | -2.336 | -0.37 to -0.25 | ~751 |
| 1d | raw | -0.733 | -0.990 | -0.02 to +0.25 | ~373-377 |
| 1d | zscored | -1.636 | -1.809 | -0.73 to -0.56 | ~406-408 |

**Every one of 24 base configs is net Sharpe negative.** Applying Phase A's own best
turnover intervention (throttle k=6) to the best carry variant - tested per section 4's
own instruction that carry should need it least - narrows the losses substantially
(e.g. 4h raw goes from -2.48 to -0.66 at offset 0) but flips the sign in only one
isolated cell across all 48 base+throttled configs (12h raw, offset 14, throttled:
+0.072). No bootstrap CI on excess return excludes zero at any of the 48 cells checked.

**Gate CY does not fire.** Funding-rate coverage came back clean (30/30 symbols, 100% of
panel rows have funding data - not the coverage-limited caveat the phase was built to
watch for), but a genuinely surprising finding replaces it: **at the same 12h interval,
realized turnover on the raw carry book (~674-681/year) is about 40% higher than
cfg2_12h's own turnover (~487-493/year, Phase A's seeded baseline), not lower**, despite
funding being a slow-moving payment. A rank-based top/bottom-20% book still churns when
funding rates cluster closely together cross-sectionally - "low turnover by
construction" turned out to be a claim about the underlying payment, not automatically
true of every way you might rank and trade it. Some gross Sharpes are genuinely positive
(up to +0.45 at 12h raw) - costs, not signal absence, are once again what erases the
edge, the same pattern every other phase in this notebook and every prior notebook in
this programme has found. No net-above-gross-Sharpe tripwire fired at any cell.

## Phase D - Tail shape as a cross-sectional factor

The slowest-moving signal available, built directly from a rolling GARCH-(Hansen
skew-t) fit per symbol at 4h (the interval Gate P fired most robustly for Hansen skew-t
in notebook 6 - 6/6 symbols significantly beating GARCH-t). IC computed **first**, before
any portfolio, per this notebook's own pre-declared rule.

### D1 - tail quality (long low |lambda|, short high |lambda|)

IC not significant (mean -0.0032, NW t=-1.43, 8766 periods). **Portfolio correctly
skipped** - backtesting it would only produce a spurious Sharpe on what the IC test
already says is noise.

### D2 - tail premium (long high nu/thin tails, short low nu/fat tails)

IC is significant but **negative** (mean -0.0089, NW t=-3.26) - the opposite sign from
its own "long thin tails" hypothesis. Read literally, this says high nu (thin tails)
predicts *lower* forward returns across the full 30-symbol cross-section. Yet the
top/bottom-quintile portfolio, built exactly as pre-declared (long the highest-nu
quintile, short the lowest), came back **net-Sharpe-positive at all four origin
offsets** (+0.402, +0.463, +0.395, +0.348) - which would technically clear Gate TF's raw
numeric bar.

**Investigated rather than taken at face value.** A significant full-sample IC and a
profitable top/bottom-quantile portfolio are not the same test - IC uses the whole
ranked cross-section every bar, the portfolio only trades the extremes (6 of 30 symbols
per leg). Checking leg composition found **FTTUSDT accounts for ~19% of the short leg's
bar-weight** - a symbol with only 6-7 successful GARCH refits over its entire history
(short listed window, most of it during the FTX exchange collapse), an unusually
unstable, thinly-identified nu estimate landing it repeatedly in the extreme low-nu
leg. **Excluding FTTUSDT flips net Sharpe to clearly negative at every offset**
(-0.338, -0.342, -0.450, -0.505). This is the same "spectacular on one symbol, does not
generalize" pattern this whole research programme has repeatedly found (BTC's own
EVT dominance in notebook 6) - here caught for the first time on a symbol that wasn't
already known to be special-cased, by checking leg composition after the fact rather
than assuming a passing number is a real finding. This single-symbol exclusion check is
now a standard part of `alpha_lib7`'s Phase D machinery, run automatically for every
factor/offset going forward.

### D3 - direct risk ranking (long low predicted-ES magnitude, short high)

IC significant (mean -0.0310, NW t=-9.17) but the portfolio is **net Sharpe negative at
every offset** (-1.552, -1.560, -1.600, -1.678), with or without FTT - a clean,
unambiguous null requiring no further investigation.

**Gate TF does not fire** for D1, D2, or D3. No net-above-gross-Sharpe tripwire fired at
any cell checked.

## Phase E - Holdout (GATED, did not run)

Per Gate H, the holdout runs only if at least one of TC/RG/CY/TF fired.
`run_phase_e_holdout.py` reads all four verdicts back out of the already-committed Phase
A-D JSONs and decides programmatically - never re-derives a fresh number, never lets a
subagent decide, per this notebook's own rule that gate decisions are the orchestrator's
alone.

**All four gates are null.** The holdout (2025-07-01 onward) was not touched: no
`load_universe_panel(allow_holdout=True)` call and no data past `research.HOLDOUT_START`
was read anywhere in this notebook. It remains available, unspent, for whichever future
notebook next has a fired gate to spend it on.

## Bugs found

Two instances of the same mistake, both in this notebook's own new driver code (not in
reused `research`/`dist_lib6`/`features` machinery): `run_phase_c_carry.py`'s and
`run_phase_d_tail_factor.py`'s own `fold_excess_returns` helper indexed into an
already-subsetted test panel using row positions computed by
`panel_walk_forward_splits` against the FULL panel - `polars.exceptions.OutOfBoundsError`
immediately, caught by running the script, not by a test. Fixed by passing the same full
frame the splits were computed on to both call sites. A loud, immediate failure rather
than a silent wrong number, but the same class of "index computed against frame X,
applied to frame Y" mistake worth watching for in future notebooks with similar
fold-then-subset patterns.

The more consequential catch was not a code bug: Phase D's D2 factor would have been
wrongly credited with firing Gate TF if its raw numbers had been trusted without
checking leg composition. Documented above and now built into `alpha_lib7`'s standard
Phase D robustness checks.

## Bottom line

**The hypothesis that transaction cost, not signal absence, has blocked every alpha
attempt in this programme does not survive this notebook's own most direct test.**
Phase A held a known-gross-profitable signal fixed and changed only its trading
mechanics - the cheapest, most targeted possible test of "is the cost problem solvable
at all." Turnover fell 71%/year, net Sharpe improved from roughly flat-to-negative to
consistently positive at every origin offset, and it still did not clear the
pre-declared bootstrap-CI bar. Cost reduction is real and mechanically reliable; it was
not, on its own, enough.

**Four gates, four nulls, four different and informative reasons - not four repeats of
the same finding:**

- **Gate TC** (turnover/cost beatability): the underlying gross edge in cfg2_12h isn't
  large enough for even a 71% turnover cut to produce a statistically distinguishable
  net edge.
- **Gate RG** (risk-gating helps): standing down during GARCH-NIG-predicted high-tail-
  risk periods mostly *hurt* net Sharpe rather than helping - a direct, informative
  answer to the open question notebook 6's own Phase 6 application result raised, not
  merely a repeat of it.
- **Gate CY** (carry survives costs): carry's own turnover turned out higher, not lower,
  than the price-based signal it was meant to contrast with - "low turnover by
  construction" is a claim about the payment, not about every way of ranking on it.
- **Gate TF** (tail factor ranks): the one factor that technically cleared the raw
  numeric bar was a single-symbol artifact, caught by checking leg composition rather
  than assumed away.

**Six notebooks now, and counting, have found no tradeable edge in liquid crypto majors
that survives its own pre-declared robustness bar, with costs charged honestly and gates
declared before any number was seen, every single time.** This is evidence about the
market - a claim about how hard cross-sectional and time-series edges are to extract
from liquid, well-arbitraged crypto majors, net of realistic execution costs - not
evidence of a failed research programme. The risk findings of notebooks 4-6 (crypto's
tails are fat, real, and increasingly well-characterized; thin-tailed models understate
1% expected shortfall almost universally; violations cluster; GARCH-NIG/Johnson SU/
Hansen skew-t genuinely beat GARCH-t cross-sectionally at 4h/12h) remain exactly as valid
and useful as they were before this notebook ran. Risk-modeling machinery and
alpha-generation machinery are not the same test, and one failing to produce a
tradeable strategy says nothing against the other's own, separately-certified findings.

## What to test next

- **A joint model of turnover and risk-gating**, rather than the sequential
  best-of-Phase-A-then-gate structure used here - Phase B's gating was only ever applied
  to a single, already-chosen throttle configuration; a joint search over (throttle k,
  gating k) might find an interaction this sequential design couldn't see, though it
  would also multiply the number of configs tried and correspondingly deflate any
  resulting Sharpe.
- **Why risk-gating hurt here specifically** - a direct check of whether cfg2_12h's own
  gross returns are higher, not lower, during GARCH-NIG's own high-predicted-VaR
  periods would confirm or rule out the "risk premium" mechanism this notebook's Phase B
  section speculated about, rather than leaving it as an inference from the net-Sharpe
  deltas alone.
- **A carry construction with genuinely lower turnover** - a slower rolling window on
  the funding z-score, or a wider no-trade band specifically tuned for carry's own
  update frequency (funding resets roughly every 8h, not every bar) - Phase C's own
  finding that a naive rank-based book churns despite a slow underlying signal suggests
  the construction, not the payment itself, is the fixable part.
- **A tail-premium factor with an explicit small-sample/short-history exclusion rule**,
  rather than a post-hoc single-symbol check - Phase D's own catch would have been
  structural rather than incidental if symbols with too few successful refits (e.g. a
  pre-declared minimum refit count) were excluded from factor construction before any
  number was seen, not after.
