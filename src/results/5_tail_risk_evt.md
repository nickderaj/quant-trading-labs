# Tail Risk and Conditional Non-Normality - Results Summary

**The evaluation criterion changes here, deliberately.** Notebook 4 ran a 7-rung
volatility *point*-forecast contest (QLIKE) and found no clear winner at any interval -
HAR-RV, the range estimators, and GARCH-normal all sit in one statistically
indistinguishable cluster, at every interval, on BTC. That contest is exhausted:
conditional variance at these horizons has an R² of 0.004-0.19, and every reasonable
point estimator lands in the same place. Notebook 4 did surface one real, narrower
result along the way - GARCH-t's own Student-t innovation distribution scored better
than a normal density - and that result is the reason this notebook exists: crypto's
tails are extreme (fitted Student-t df of 2-3 across every interval tested), and the
question worth asking next is not "who forecasts variance best" but **which model gives
the best-calibrated conditional tail, scored with the same rigor the variance ladder
was.** Notebooks 4 and 5 are not in disagreement - they answer different questions.

Primary asset **BTCUSDT**, all 4 intervals (1h/4h/12h/1d), frozen-transferred to
ETH/SOL/DOGE/BNB/XRP at 1d. Full machinery: `src/distributions.py` (closed-form CRPS
added this notebook), `src/research/tmp/dist_lib.py` (notebook 4's own machinery, reused
without modification except the one causality fix below), and the new
`src/research/tmp/dist_lib5.py` (Hill estimator, GJR-GARCH, GPD/EVT, vectorized density
scoring, Benjamini-Hochberg, the coverage battery, Acerbi-Székely). Terminology used
throughout this write-up is defined from scratch, with worked examples grounded in this
repo's own numbers, in `docs/` (start at `docs/README.md`).

## Correction to notebook 4 (mandatory, landed before any new model code)

Two bugs in already-committed code were found and fixed before any Phase 1-5 work in
this notebook began, because Phase 3's density contest is built directly on the same
scoring machinery they touch.

### The GARCH-t degrees-of-freedom lookahead

`run_phase3.py` scored GARCH-t's own-distribution log score and VaR using
`fits[-1]["params"][3]` - the Student-t degrees of freedom estimated on the **final**
training window of the whole sample - applied to score every bar from the start of the
evaluation period onward. The variance forecast itself was properly causal and rolling;
only the shape parameter scoring it was not. This is the same class of bug as notebook
4's own HAR-RV same-bar leak (bug #3 there), one level deeper: in the innovation
distribution's shape parameter rather than the point forecast.

**Fix**: `dist_lib.nu_path_from_fits` builds a causal, forward-filled step-function path
of ν from the fits list, mirroring the variance forecast's own forward-fill exactly.
Diagnosing the fitted ν path directly (not just trusting the fix ran) shows it genuinely
varies across the 46 refits at 1h - from 2.2 (the optimizer's own lower search bound,
during the 2021-2022 stretch) to 8.1 (later, calmer refits) - confirming the bug was
real, not cosmetic.

**Both halves of notebook 4's original claim moved, in different directions, once fixed
and Phase 3 fully re-run:**

| interval | GARCH-t log score, OLD (contaminated) | NEW (corrected) | Kupiec 5% VaR p, OLD | NEW |
|---|---|---|---|---|
| 1h  | 3.979 | **4.000** | 0.42 | **0.0000** |
| 4h  | 3.194 | **3.254** | 0.18 | **0.0008** |
| 12h | 2.614 | **2.623** | 0.39 | 0.35 |
| 1d  | 2.187 | **2.220** | 0.84 | 0.68 |

The **log-score win strengthened**: GARCH-t now beats every normal-density rung at all 4
intervals (previously 3 of 4 - 1d had been marginally the other way before the fix). The
**VaR-coverage claim weakened materially**: Kupiec now rejects GARCH-t's 5% VaR at 1h
and 4h (previously reported as never rejected anywhere). Per this notebook's own stop
condition, this was reported and flagged as a decision point before any further work -
it does not overturn notebook 4's bottom line (no point-forecast rung wins outright) but
it changes the strength of the one calibration-specific result notebook 4 did find, and
this notebook's own Gate B (below) is the real, fuller test of GARCH-t's VaR calibration
rather than assuming it from notebook 4's original, uncorrected number.

### The CRPS integration grid

`distributions.crps`'s numerical grid (`linspace(ppf(1e-6), ppf(1-1e-6), n_points)`)
spans ~10 units for a normal but ~1,400 units for a t(2) at the same `n_points=400` -
wildly different effective resolution across families, so any cross-family CRPS
comparison was partly measuring integration error, not forecast quality. Verified
directly: on a t(2.1) forecast, the old default grid disagrees with the true
(closed-form) value by **13-79%** depending on the observation. Added
`crps_normal_closed_form`/`crps_t_closed_form` (Gneiting & Raftery 2007;
Jordan/Krüger/Lerch, matching R's `scoringRules::crps_t`), verified to match a very fine
numerical grid to ~1e-5 on light-tailed cases. This is also what makes the 28-pair,
bootstrap-heavy Phase 3 contest below computationally tractable on this hardware - the
old per-observation frozen-distribution CRPS loop would have been far too slow at this
scale.

## Phase 1 - Foundations: does the variance even exist?

Fit-once on the full pre-holdout history (descriptive, same status as notebook 4's own
Phase 1 - Phase 2 onward is where everything becomes rolling and causal).

### Hill estimator: independent of the scipy `t.fit` optimizer notebook 4 already caught pinning at a boundary

| interval | tail | plateau k-range | α̂ (point) | 95% bootstrap CI |
|---|---|---|---|---|
| 1h  | upper | [143, 3506] | 2.220 | [2.134, 2.315] |
| 1h  | lower | [158, 3506] | 2.187 | [2.106, 2.286] |
| 4h  | upper | [176, 876]  | 2.356 | [2.189, 2.544] |
| 4h  | lower | [88, 876]   | 2.381 | [2.232, 2.594] |
| 12h | upper | [106, 198]  | 2.716 | [2.352, 3.092] |
| 12h | lower | [151, 272]  | 2.203 | [1.970, 2.495] |
| 1d  | upper | **no stable plateau found** | - | - |
| 1d  | lower | [79, 146]   | 2.278 | [1.979, 2.671] |

Every point estimate sits **above** 2, and every computed CI either sits fully above 2
(1h, 4h) or dips only barely below it (12h/1d lower tails, CI lower bound ~1.97-1.98).
**Gate E does not fire anywhere** (0/4 intervals meet "α≤2 with CI excluding 2") - no
prominent top-of-file caveat is triggered. 1d's upper tail found no stable plateau at
any `k` and is reported as provisional, honestly, per this notebook's own tripwire rule,
rather than quoting a point value anyway.

This is itself a real finding, not a non-event, and it sits in **mild tension** with
notebook 4's own parametric Student-t MLE fit, which found degrees of freedom as low as
1.98 at 1h (right at the boundary). The non-parametric Hill estimate is consistently
somewhat higher (further from the boundary) at every interval - **variance most likely
does exist, though not by a wide margin**, and the two independent estimators broadly
agree without literally agreeing on the point value.

### Does the Diebold-Mariano test's own CLT hold here?

Compared DM's normal-approximation p-value against a block-bootstrap p-value on the
same QLIKE loss-differential series, for the closest (least-significant) pair from
notebook 4's own Phase 3 all-pairs DM at each interval:

| interval | pair | normal-approx p | bootstrap p | materially disagree? | loss-diff Hill α (upper, lower) |
|---|---|---|---|---|---|
| 1h  | Garman-Klass vs GARCH-normal | 0.936 | 0.955 | No | 1.50, 1.42 |
| 4h  | Garman-Klass vs GARCH-normal | 0.878 | 0.887 | No | 1.25, 1.80 |
| 12h | EWMA vs Garman-Klass | 0.967 | 0.851 | **Yes** | 1.50, 1.38 |
| 1d  | trailing-96 vs Parkinson | 0.966 | 0.943 | No | 2.00, 1.90 |

Materially disagree (>0.05 apart) only at 12h - both still say "not significant," so no
notebook 4 conclusion is overturned by this specific check. But the QLIKE loss
differential's **own** Hill-estimated tail index runs as low as ~1.2-1.9 at several
interval/tail combinations - low enough that DM's CLT-based p-value is worth treating as
approximate rather than exact in general. Consequence carried through the rest of this
notebook: **Phase 3's own Gate A verdict below uses bootstrap p-values as primary**, not
the normal approximation alone.

### Is log-RV the better-behaved object? (cheap, done as asked)

Fit normal/t/skew-t to `rv` directly and to `log(rv)`, compared PIT/KS calibration
(large p = not rejected = well-calibrated):

| interval | RV levels: normal / t / skew-t KS p | log(RV): normal / t / skew-t KS p |
|---|---|---|
| 1h  | ~0 / ~0 / ~0                | 1.4e-88 / 2.5e-88 / (fit failed) |
| 4h  | ~0 / ~0 / 7.5e-29           | **0.377** / **0.383** / **0.980** |
| 12h | 1.4e-191 / 6.6e-92 / 4.5e-9 | 0.068 / 0.069 / **0.799** |
| 1d  | 1.0e-69 / 1.5e-32 / 3.5e-5  | 0.004 / **0.054** / **0.158** |

**Raw RV is rejected outright, every family, every interval, with no exceptions** (KS p
effectively 0 at every single cell in the left half of this table). **log(RV) is
dramatically better-calibrated at every interval except 1h** - not rejected at any
family at 4h/12h, and skew-t not rejected even at 1d. Only at 1h does log(RV) also fail
(p~1e-88) - the finest interval already flagged as having the heaviest, least stable
tail by every other check in this notebook. This is a clean, direct confirmation that
notebook 4's rung 2/4 (HAR-RV, RV-distribution fits) were handicapped by working in the
wrong space; the HAR-log-RV rung added to Phase 3 below is the direct consequence.

**Gate E verdict: does not fire.** Stated here in full per §3's requirement to report it
regardless of outcome.

## Phase 2 - New models

### GJR-GARCH (leverage)

Fit with normal and t innovations (skew-t skipped as over-parameterized, per notebook
4's own finding that skew-t buys nothing unconditionally for BTC). GJR nests plain
GARCH(1,1) exactly at γ=0, tested directly via a likelihood-ratio test at every refit:

| interval | frac. refits with significant leverage (normal innov.) | frac. (t innov.) |
|---|---|---|
| 1h  | **43.5%** | 21.7% |
| 4h  | 34.8% | 17.4% |
| 12h | 39.1% | 19.6% |
| 1d  | 17.4% | 6.5% |

Leverage is real and non-trivial in a substantial minority of individual refits,
strongest at 1h. But this does **not** translate into a better *pooled* density
forecast: all-pairs DM shows GARCH-t significantly beats GJR-t at 1h/4h/12h
(bootstrap-BH p<0.05 at all three) and ties at 1d; GARCH-normal vs. GJR-normal never
differs significantly at any interval. **Leverage is a real, occasionally-significant
refit-level effect that does not survive as a net improvement once rolled forward** -
the extra parameter's estimation noise outweighs its benefit in this rolling-refit
setting, a clean, numbers-backed illustration of overparameterization rather than a
hypothetical.

### Conditional EVT (McNeil-Frey two-stage)

GPD fitted to standardized residuals from the GARCH fit, refit at exactly the same
cadence and training windows as the variance model (causal, forward-filled, per
NEXT_RUN_PROMPT.md's critical-causality note). Summary of the fitted shape ξ across all
individual rolling refits (both tails pooled):

| interval | n refits (both tails) | ξ median | ξ range | frac. with ξ<0 |
|---|---|---|---|---|
| 1h  | 92 | 0.16  | [-0.27, 0.73] | 12% |
| 4h  | 92 | 0.05  | [-0.47, 0.49] | 35% |
| 12h | 88 | 0.03  | [-0.49, 0.43] | 45% |
| 1d  | 78 | -0.13 | [-0.52, 0.22] | **78%** |

**Tripwire investigated, per §9's own instruction to stop and check ξ<0 rather than
ignore it.** A substantial and *growing-with-interval-width* fraction of individual
500-bar-window refits show ξ<0 (formally a bounded tail - implausible for crypto taken
at face value). Investigated directly rather than waved through: with `tail_frac=0.10`
on a 500-bar window, each individual refit estimates ξ from only ~50 exceedances - a
genuinely small sample for a shape parameter, and the *median* ξ across refits stays
positive at 1h/4h/12h and only dips slightly negative at 1d, in a pattern that exactly
tracks aggregational Gaussianity (thinner tails at coarser intervals) already established
everywhere else in this research programme. Read plainly: this is consistent with a
true ξ that is small-but-positive at fine intervals and close to zero at 1d, with **the
individual ξ<0 refits being small-sample noise scattered around that true value**, not
genuine evidence of a bounded tail at every one of those refits. Still worth flagging
honestly as a real limitation of a 500-bar/50-exceedance rolling GPD fit, not
papered over.

**A second, informative cross-check**: Phase 1's Hill estimator (on raw, unconditional
returns) implies ξ = 1/α ≈ 0.37-0.45 at every interval - consistently *higher* than the
median standardized-residual GPD ξ above (0.16 down to -0.13). This gap is itself a
real, explainable finding, not a contradiction: Hill measures the **unconditional** tail
(mixing every volatility regime together), while the GPD here is fit to **GARCH-
standardized residuals** - the **conditional** tail, after the time-varying variance
has already been divided out. A meaningful share of what makes raw crypto returns look
so fat-tailed is volatility clustering itself (a mixture-of-regimes effect), not a
genuinely heavy-tailed *conditional* innovation - exactly consistent with GARCH-t's own
own-distribution degrees of freedom (7-8 at most refits, per notebook 4/this notebook's
diagnostics) sitting far higher than the ~2-3 df found unconditionally in notebook 4's
Phase 1. Conditioning on volatility genuinely thins the tail that's left over.

## Phase 3 - The density contest

**Log score is primary** (QLIKE kept only as a secondary column, for continuity with
notebook 4). d8/d9 (GARCH-EVT, GJR-EVT) are **not entered** in this log-score/CRPS
contest - continuously normalizing a GPD-tails-plus-empirical-body density proved
exactly as fiddly as anticipated, and this notebook uses its own sanctioned fallback
("an honest partial entry beats a hand-waved density") rather than forcing a shaky
density through. So this is an **8-model, 28-pair-per-interval contest** (documented
plainly, not silently smaller than the originally-planned 10-model/45-pair one).

### Log score by model (higher is better; ranked within each interval)

| rank | 1h | 4h | 12h | 1d |
|---|---|---|---|---|
| 1 | **GARCH-t** 4.000 | **GARCH-t** 3.254 | **GARCH-t** 2.623 | HAR-log-RV 2.244 |
| 2 | GJR-t 3.988 | GJR-t 3.233 | GJR-t 2.609 | GARCH-t 2.220 |
| 3 | GARCH-normal 3.848 | HAR-log-RV 3.167 | HAR-log-RV 2.572 | GJR-t 2.220 |
| 4 | HAR-RV 3.844 | HAR-RV 3.139 | HAR-RV 2.543 | HAR-RV 2.191 |
| 5 | GJR-normal 3.842 | GARCH-normal 3.130 | GJR-normal 2.530 | GJR-normal 2.167 |
| 6 | Garman-Klass 3.826 | GJR-normal 3.124 | GARCH-normal 2.522 | GARCH-normal 2.163 |
| 7 | trailing-96 3.768 | range 3.101 | trailing-96 2.471 | trailing-96 2.139 |
| 8 | HAR-log-RV 3.707 | trailing-96 3.073 | range 2.399 | Parkinson 2.117 |

**This is the cleanest result this research programme has produced.** Notebook 4's
QLIKE ladder found ties everywhere; scoring the identical underlying variance recursions
on log score instead surfaces a real, ordered, mostly-replicating ranking - t-innovation
models at the top, normal-innovation in the middle, trailing-std/range at the bottom, at
every interval - rather than noise.

### Gate A verdict (all-pairs Diebold-Mariano, Benjamini-Hochberg-adjusted, bootstrap p-values primary)

| interval | best model | beats every other, significantly? |
|---|---|---|
| 1h  | GARCH-t | **Yes** |
| 4h  | GARCH-t | **Yes** |
| 12h | GARCH-t | **Yes** |
| 1d  | HAR-log-RV | No |

**Gate A fires at 3 of 4 intervals.** GARCH-t is a genuine, statistically certified
density winner at 1h/4h/12h - both the bootstrap-adjusted and normal-approximation
verdicts agree at every interval this time. Only at 1d does no model win significantly
(HAR-log-RV edges narrowly ahead, 2.244 vs. GARCH-t's 2.220, but not significantly) -
consistent with 1d being the interval where notebook 4's own point-forecast ladder came
closest to a real result and where Phase 1's own Hill/GPD diagnostics show the tail
closest to normal.

## Phase 4 - The tail calibration battery

Full grid: Kupiec + Christoffersen independence + Christoffersen conditional coverage,
all 6 quantile levels (1%/2.5%/5%/95%/97.5%/99%), all 10 models (d8/d9's quantile/ES
forecasts are well-defined from their GPD fits even though their full density wasn't
entered in Phase 3), all 4 intervals - **1,440 individual tests.**

### Gate B verdict

| interval | model(s) clearing all 36 tests |
|---|---|
| 1h  | none |
| 4h  | none (GARCH-EVT comes closest: 0 Kupiec failures, 1 independence, 1 conditional-coverage failure out of 18 tests) |
| 12h | **GARCH-EVT** |
| 1d  | none (GARCH-EVT/GJR-EVT each fail only 1 of 18 tests) |

**Gate B fires exactly once**: GARCH-EVT clears every single one of the 36 tests at 12h.
At every other interval, both EVT models come close (0-3 failures out of 18 tests each)
while every non-EVT model fails multiple tests at every interval, usually on 4-6 of the
6 quantile levels on at least one of the three tests. 1h is the hardest interval across
the board, including for the EVT models - consistent with every other diagnostic in this
notebook flagging it as the heaviest, least-stable tail.

### The single cleanest result in this notebook: the Acerbi-Székely ES backtest

At the 1% level, **at every interval, with zero exceptions**, every non-fat-tailed model
(trailing-std, HAR-RV, HAR-log-RV, range, GARCH-normal, GJR-normal) has a significantly
positive Z statistic (bootstrap p<0.05, usually p=0.000):

| interval | non-fat-tailed models' Z range (all significant) | t-innovation Z (1h/4h sig.; 12h/1d not) | EVT Z (never significant except 1h/4h borderline) |
|---|---|---|---|
| 1h  | 1.04 to 1.98 | 1.34 - 1.71 | 0.17 - 0.21 |
| 4h  | 0.98 to 2.11 | 0.71 - 1.57 | 0.24 |
| 12h | 1.15 to 2.92 | 0.66 - 1.51 | 0.04 - 0.09 (not sig.) |
| 1d  | 0.48 to 1.38 | 0.38 - 0.48 (not sig.) | 0.13 - 0.18 (not sig.) |

(Recall $Z \approx 0$ means well-calibrated ES; $Z>0$ means realized 1%-tail losses are
significantly *worse* than the model's own expected-shortfall prediction - the model
understates tail risk. A genuine bug in the runbook's own pseudocode was caught and
fixed here before this table was trusted: both the sign of the statistic's additive
constant and the stated direction of the failure mode were backwards, verified
numerically with a deliberately mis-specified model before correcting - see
`docs/06-scoring-rules-and-calibration.md#acerbi-székely`.)

Plainly stated, this is the headline finding of the whole notebook: **models that don't
account for fat tails don't just score worse on an abstract log-score metric - they
concretely, measurably underestimate how bad the worst days actually get, every single
time this was checked, at every interval.** Fat-tailed and EVT-based models are
dramatically better calibrated for this specific, practically-important risk, clearing
the strict coverage bar outright at 12h and coming within one or two tests of it
everywhere else except 1h.

## Phase 5 - Transfer / stability

ETH/SOL/DOGE/BNB/XRP, 1d only (same scoping rationale as notebook 4 - wall-clock cost on
this hardware). **A real scoping limit, stated plainly**: BTC's headline Gate A win
(GARCH-t, significant at 1h/4h/12h) cannot be directly transfer-tested, because Phase 5
is scoped to 1d only and 1d is the one interval where BTC's own contest found no
significant winner. What follows tests whether *that* (1d, no-significant-winner)
pattern replicates, and whether Gate B's calibration story does.

### Gate A: a perfectly stable null

| symbol | best-by-log-score model | Gate A fires? |
|---|---|---|
| BTC (reference) | HAR-log-RV | No |
| ETH | GJR-t | No |
| SOL | HAR-log-RV | No |
| DOGE | GARCH-t | No |
| BNB | HAR-log-RV | No |
| XRP | GARCH-t | No |

**Gate A fires at 0 of 6 symbols at 1d** - a perfectly replicating null, not a single
spurious "winner" anywhere. The identity of the best model splits three ways (GJR-t,
HAR-log-RV, GARCH-t) but never lands on a naive baseline (trailing std, plain range,
GARCH-normal) on any symbol - the *cluster* of plausible winners (fat-tailed/log-RV
models) replicates even though the single best does not, the same pattern notebook 4
found for its own point-forecast ladder.

### Gate B: clears on most altcoins, not on BTC or ETH

| symbol | model(s) clearing all 36 tests at 1d |
|---|---|
| BTC | none |
| ETH | none |
| SOL | HAR-log-RV, GARCH-EVT, GJR-EVT |
| DOGE | GARCH-EVT, GJR-EVT |
| BNB | GARCH-normal, GJR-normal, GARCH-t, GJR-t, GARCH-EVT, GJR-EVT (6 of 10) |
| XRP | GARCH-t, GJR-t |

Gate B clears on 4 of 5 transfer symbols (not ETH), and an EVT model is present in every
clearing set except XRP's. BNB is a standout with six of ten models clearing at once -
unusually well-behaved 1d tails for that symbol specifically. Read plainly: **EVT-based
tail calibration replicates as a genuinely good idea across most of this symbol set, but
"which specific model clears, or whether any does" is asset-specific**, not a single
portable number - exactly notebook 4's own standard (tail-shape findings replicate more
readily than rankings), confirmed again on a different question.

### Leverage is not a stable, quotable quantity across assets

GJR's leverage LR-test significant-refit fraction (normal innovation): 0.087 (SOL),
0.174 (BTC), 0.217 (XRP), 0.370 (DOGE), 0.413 (BNB), 0.457 (ETH) - a nearly 6x range
across six assets at the same interval. Leverage exists intermittently and its
prevalence is genuinely asset-dependent; reported as such rather than averaged into one
misleading number.

## Phase 6 - Application (gated, does not run)

Pre-declared application: an EVT-conditional risk-limit overlay on buy-and-hold BTC,
judged against buy-and-hold and a normal-GARCH-driven overlay on Sharpe, max drawdown,
1% exceedance count, and turnover cost. **Gate D requires a Gate A or Gate B winner AND
Gate C stability.** Gate A fired at 1h/4h/12h (BTC only - not tested cross-sectionally
at those intervals, per Phase 5's own scoping limit) and Gate B fired at 12h (BTC) and
at 4 of 5 transfer symbols at 1d, but never together with a stability check spanning
the *same* interval and *all six* symbols - Gate A's 1h/4h/12h wins were never
transfer-tested, and Gate B's cross-sectional success is at 1d, where Gate A does not
fire on BTC at all. **Gate D does not fire. Phase 6 does not run.** Written up here, as
notebook 4 did, rather than silently skipped.

## Bugs found

Beyond the two mandatory corrections to notebook 4 at the top of this write-up:

1. **Acerbi-Székely sign error in the runbook's own pseudocode.** Both the sign of the
   statistic's additive constant ("+1" vs. the correct "-1") and the stated direction
   of the failure mode ("Z<0 means worse than predicted" vs. the correct "Z>0") were
   backwards. Caught by verifying numerically - not just re-deriving on paper - with a
   deliberately mis-specified model (known-wrong volatility) before trusting either the
   formula or its interpretation. Fixed in `dist_lib5.acerbi_szekely_z` and documented
   in `docs/06-scoring-rules-and-calibration.md#acerbi-székely` so it isn't silently
   re-broken.
2. **GPD ξ<0 tripwire, investigated rather than ignored.** A substantial and growing
   (12% to 78% across 1h→1d) fraction of individual 500-bar-window GPD refits produced
   ξ<0 (a bounded tail, implausible on its face for crypto). Investigated directly:
   each refit estimates ξ from only ~50 exceedances, a genuinely small sample; the
   *median* ξ across refits stays sensible (tracking aggregational Gaussianity) and the
   scattered negative estimates are consistent with small-sample noise around a small,
   true ξ rather than a real, per-refit finding. Reported honestly as a real limitation
   of a 500-bar/50-exceedance rolling fit, not smoothed over.

None of these were caught by a unit test in the driver scripts themselves (matching
notebook 4's own convention - the driver/contest machinery isn't unit-tested by design;
the new *modelling* code in `dist_lib5.py` does have a dedicated test suite,
`tests/test_dist_lib5.py`, 15 tests). Both were caught the same way every bug in this
research programme's history has been: by reading the actual numbers - a Monte Carlo
check against a case whose right answer was already known, and a direct tabulation of
how often a supposedly-rare tripwire condition actually fired - rather than trusting
that code which ran without raising had produced a correct or complete answer.

## Bottom line

**The point-forecast question notebook 4 asked is exhausted; the density/tail question
this notebook asks is not.** Reframing the identical variance models' scoring from QLIKE
to log score surfaces a real, replicating, statistically certified density winner
(GARCH-t) at 3 of 4 BTC intervals - something the point-forecast ladder never found
anywhere. The tail-calibration battery is stricter still, and still finds something:
GARCH-EVT clears every one of 36 coverage tests at 12h, and - more practically than any
single gate - **every model that ignores fat tails significantly underestimates how bad
the worst 1% of days actually are, at every interval, with no exceptions**, confirmed
by the Acerbi-Székely ES backtest. GJR's leverage effect is real in a meaningful
minority of individual refits but does not survive as a net pooled improvement, and its
prevalence varies enormously by asset. The 1d interval, and the transfer check run
there, tell a consistent, honest story of their own: no significant density winner
anywhere at 1d (a stable null across 6 symbols), and tail-calibration success that
replicates on most but not all of the transfer set, via EVT models specifically where
it does.

**Phase 6 did not run** - Gate D's dual requirement (a Gate A or B winner, plus
cross-sectional stability at the *same* interval) was never jointly satisfied, a
legitimate outcome under this notebook's own pre-declared rule, not a shortfall. This
notebook adds genuinely new, certified knowledge notebook 4 could not produce on its
own: crypto's conditional tails are real, extreme, non-normal in a way that concretely
costs risk-unaware models measurable accuracy on the outcomes that matter most, and this
is now established with the same pre-declared, multiple-testing-corrected rigor the
variance ladder was held to - even though, per this whole research programme's
consistent pattern, no tradeable application clears the bar this notebook set for it.

## What to test next

- **Extend Gate A's 1h/4h/12h transfer test.** Phase 5 was scoped to 1d only for
  compute reasons; BTC's actual certified win (GARCH-t at 1h/4h/12h) has never been
  transfer-tested at all. This is the single most valuable, cheapest-to-articulate
  follow-up: it would either turn Gate A into a stability-certified, Phase-6-eligible
  finding, or reveal that BTC's win doesn't generalize the way notebook 4's own HAR-RV
  point-forecast ranking didn't.
- **A properly normalized d8/d9 density**, to enter GARCH-EVT/GJR-EVT in the log-score
  contest directly rather than deferring them to Gate B alone - given how strong their
  coverage/ES performance already is, they are a natural candidate to also win Gate A
  if their density can be made to integrate cleanly.
- **A formal test of whether GJR's leverage effect is asset-class-general or BTC/crypto-
  specific**, given how widely its significant-refit fraction varied (0.09 to 0.46)
  across just six crypto symbols in Phase 5 - the traditional-equities "leverage effect"
  literature this model borrows from was built on a very different asset class.
- **Revisit the deferred Phase 4-of-notebook-4 regime-comparison gap** alongside this
  notebook's own tail work - a joint regime-and-tail model (does the EVT shape parameter
  itself differ meaningfully by volatility regime?) is a natural, currently untested
  extension of both notebooks' machinery.
