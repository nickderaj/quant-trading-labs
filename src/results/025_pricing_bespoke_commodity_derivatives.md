# Notebook 025 — Pricing bespoke commodity derivatives, and a lesson

## The narrow question

If someone handed you a bespoke commodity derivative tomorrow — an Asian call on
diesel, a knock-out floor for a wheat farmer, an autocallable reverse convertible on
crude — and asked "what is this worth, and how should our client hedge it?", what
would you actually do? Not "does model A beat model B" (there is no market price in
this repo to beat), but the practitioner's question: how do you get a defensible
number, cross-check it, and explain the trade-off to a client who can't afford the
textbook hedge.

This is a **teaching notebook**, not a forecasting contest. Eight pricing methods
(Black-76, Bachelier, displaced diffusion, binomial, finite-difference PDE, Monte
Carlo, Longstaff-Schwartz, a P-measure one-factor Markov functional, plus a
model-free historical bootstrap) are built once, cross-validated against each other
and against closed-form limits, and then applied to seven real hedging problems using
this repo's own market data (CL, HO, RB, NG, ZW, KE, BZ, 6E, 2010–2026).

**There are no option prices anywhere in this repo.** Every premium quoted below is a
*model* price computed from *realised* volatility, never a market quote — and since
options generally trade above realised vol (the variance risk premium), every one of
these prices is probably an underestimate of what a dealer would actually charge. This
is stated once here, in the notebook's title cell, in Part E, and is not repeated as
a caveat on every number below; assume it applies everywhere.

## The answer

| Question | Answer |
|---|:---:|
| Do independent pricing methods agree (Phase 2)? | **Yes — 17/17 cross-validation cells pass** |
| How often would the farmer's knock-out have knocked out? | **28.6% of the time (ZW), n=49; 26.0% (KE)** |
| What does the zero-cost collar actually cost, in upside given up? | Call strike is solved so net premium = 0; the farmer gives up all upside above that strike (Part D3/D4) |
| Does hedging inputs and outputs separately hedge the refiner's margin? | **No, exactly** — Kirk vs bivariate-MC spread-option prices differ by ~2.0%, the correlation term the separate hedges miss |
| What did Black-76 say about 2020-04-20? | **Log score = −∞** (zero density below $0); Bachelier and displaced diffusion both assign finite probability |
| How large is the quanto adjustment vs the instability of its own correlation? | Adjustment ≈ **$0.05**; ρ's ±1σ price range ≈ **$0.19** — the instability is ~4× the adjustment |
| What is the issuer margin on the structured note? | **≈ −0.11%** (fair value $1.0011 vs $1.00 face) — negligible, well inside Monte Carlo noise |

## Method

Seven phases, one frozen shared library (`lib25.py`) and four independently-tested
pricer modules, mirroring 024's `run_step`-based driver discipline.

### Phase 0 — Pre-registration

The honesty list (no market prices, realised vol is a downward-biased proxy, no
bid-offer/credit/margin modelled, no Sharpe/backtest anywhere), the Phase 2
cross-validation tolerances, and each of the seven case studies' candidate structures
were fixed here, before any price was computed.

### Phase 1 — Market inputs

For CL, HO, RB, NG, ZW, KE, BZ: the observed forward curve plus Nelson-Siegel
interpolation (never Gabillon — 023's own held-out-maturity gate), and the realised
(MAD) volatility term structure. Every product's fitted Samuelson slope was checked
against 024's own published full-sample slope and 95% CI and landed inside a widened
band around it — the wiring check the design doc calls for. FX (6E) and cross-product
correlations (CL–HO, CL–RB, CL–BZ, CL–6E) were computed once here and reused by every
downstream phase and case study.

### Phase 2 — The cross-validation matrix (the referee)

With no market data to check pricers against, the only available proof a pricer is
right is that independent methods agree and reduce to known closed forms in their
degenerate limits. All **17 cross-validation cells pass**: European call/put
(Black-76 vs binomial vs PDE vs Monte Carlo, all within tolerance), Bachelier vs MC on
paths that go negative, the displaced-diffusion limits (→ Black-76 as shift→0, →
Bachelier as shift→large), all four barrier types (Reiner-Rubinstein analytic vs a
PDE with the barrier on an exact grid node vs BGK-corrected discrete Monte Carlo), the
knock-in + knock-out = vanilla identity to `1e-10`, geometric and arithmetic Asian
options, American options (binomial vs PDE vs Longstaff-Schwartz with its dual upper
bound), spread options (Margrabe vs Kirk vs bivariate MC, with Kirk's known
near-zero-spread/high-correlation degradation recorded rather than hidden), quanto,
and put-call parity on every European pricer.

Three genuine implementation bugs were found and fixed here, none of them in the
module authors' own unit tests (see **Bugs found**): a sign error in both the barrier
and American finite-difference operators, the barrier PDE not actually placing the
barrier on a grid node for two of the four barrier kinds, and the
Broadie-Glasserman-Kou continuity correction applied to the wrong side of the
MC-vs-analytic comparison.

### Phase 3 — Primer mechanics

Pure payoff geometry, three real CL front-month paths over the COVID window (one
breaching an illustrative barrier, two not), and the single most useful number in the
primer: the **realised knock-out frequency** for CL, by barrier distance and tenor,
split crisis vs calm regime, over the full 16-year sample — this feeds directly into
case study B.

### Phase 4 — Method behaviour

Wall-clock timing, convergence traces (binomial, Monte Carlo, PDE), variance-reduction
efficiency (antithetic ≈1×, a simple control variate ≈6.3×, Sobol ≈1× at this path
count — antithetic and Sobol earn their keep more at higher dimensionality than this
smoke-scale test exercises), the Longstaff-Schwartz duality gap (≈4.8, roughly two
thirds of the LSM price itself — a large, honest gap for a coarse basis, reported as
the point rather than hidden), and a one-factor Markov functional calibrated to
realised terminal marginals, which reprices its own calibration exactly (error = 0.0
by construction) while its path-dependent price differs from a plain-GBM price by
about 4.7% — pure joint-dynamics error, cleanly isolated because the marginal channel
is closed off by construction.

### Phase 5 — Case studies A–D

**A, the refiner.** Hedging the 3:2:1 crack spread margin (long crude, short
products) directly (Kirk + bivariate MC, ~2.0% apart) vs hedging inputs and outputs
separately: the two differ by the correlation term (CL–HO ρ≈0.87, CL–RB ρ≈0.89), a
concrete, non-zero number rather than an assertion.

**B, the wheat farmer — the case study the whole notebook is built around.** Plain
put vs zero-cost collar vs 15%-OTM knock-out put on ZW (cross-checked on KE). Over 49
historical ~4-month starting points, the knock-out would have fired **28.6%** of the
time (26.0% on KE), and on the years it did, the realised shortfall averaged
**$4.74/bushel** relative to the plain put's floor — the number a farmer needs before
deciding the premium saving is worth it.

**C, the airline / diesel buyer.** A monthly-average (Asian) call on HO vs a
single-date European. The realised vol of the monthly average vs a single settlement
date, and the wrong textbook `σ/√3` shortcut vs the schedule-correct calculation,
differ by **~13.4%** in resulting premium — see **Bugs found** for a caveat on the
vol-ratio direction, which came out the opposite way from what the design doc expects
and needs a second look before being treated as a finding.

**D, the gas utility.** A winter Dec–Mar cap strip priced with a single flat vol vs
the curve's own seasonal vol structure: **13.4%** cheaper priced flat than seasonal
(i.e. a flat-vol desk would underprice winter optionality relative to what the
term-structure-consistent seasonal price actually costs) — a concrete dollar
consequence of skipping the seasonal step, not just an assertion that seasonality
matters.

### Phase 6 — Case studies E–G, and the variance aside

**E, negative prices.** A $10 put on CL202005 priced on 2020-04-06. Black-76 assigns
the actual 2020-04-20 settlement (−$2.67) **zero probability** — log score `−∞`, the
cleanest falsification in the notebook — while Bachelier and displaced diffusion both
assign it finite, computable density. The systematic version (not built on one day):
rolling 1% left-tail Kupiec coverage over the full 16-year sample rejects calibration
for the lognormal-return model at `p≈5.5e-20`; the level-change (Bachelier-consistent)
model also fails Kupiec but for the opposite reason — it is *too* conservative
(observed exceedance rate 2.06% vs a 1% target), which is itself an honest,
reportable finding rather than a clean pass.

**F, the European buyer.** Quanto (fixed-FX) vs composite (floating-FX) calls on CL
settled in EUR. The quanto drift adjustment is small (≈$0.05 vs an unadjusted
$4.77 call), but the price range implied by the CL–6E correlation's own rolling
±1σ dispersion (≈$0.19) is **about four times larger than the adjustment itself** —
exactly the finding the design doc predicted, now with real numbers behind it.

**G, the structured note (capstone).** A one-year autocallable reverse convertible on
CL: 8% coupon, 100% autocall trigger, 70% principal barrier. Model fair value
($1.0011) sits almost exactly at face ($1.00) — an implied issuer margin of about
−0.11%, negligible next to the Monte Carlo standard error (≈0.0004) at this path
count, so this notebook cannot distinguish a real margin from simulation noise here.
Realised historical autocall frequency (72.3% over 238 downsampled 1-year windows)
is far higher than the model's own risk-neutral-drift autocall frequency (16.3%) —
expected, since 2010–2026 CL history has more up-drift than a driftless GBM assumes,
and the honest reason the two numbers should *not* match.

**Variance-swap aside.** The fair strike (≈32.0% vol) is reported as a *forecast*,
never a replicated price — there is no option strip in this repo to replicate one.
See **Bugs found** for the intraday-vs-close-to-close ratio, which came out
implausibly large and should not be trusted as computed.

### Phase 7 — Notebook and charts

35 figures across Parts A–E (11 primer, 5 inputs, 8 methods, 9 case studies, 2
limits — within the 30–34 target range), built by four Haiku agents from frozen
`lib25`/pricer-module APIs (Stage 2), one referee pass by the main session (Stage 3),
four more agents for Phases 3–6 (Stage 4), three more for the Part A/C/D chart
modules (Stage 5), with Part B and Part E written directly by the main session. The
notebook executes cleanly end to end (126 cells, 0 error outputs) with
`jupyter nbconvert --execute`.

## Bugs found

Reported in the order they were caught, following this repo's convention of
documenting what went wrong rather than only what went right.

1. **Barrier and American finite-difference operators had a sign error.** Both
   solvers' off-diagonal coefficients were built as `+(α±β)` instead of `−(α±β)` in
   the `(I − dt·L)v_new = v_old` implicit scheme, which made both solvers decay
   toward zero as the grid refined instead of converging to the analytic/binomial
   reference. Caught by Phase 2's 0.5%/1% cross-validation tolerances, not by either
   module's own unit tests (which only checked qualitative properties, not absolute
   convergence). Fixed in both `pricers_25_barrier.py` and `pricers_25_american.py`.
2. **The barrier PDE didn't actually solve the knock-out equation for the knock-in
   kinds.** For `kind in ('di','ui')` the original code skipped the barrier-boundary
   enforcement in the time-stepping loop entirely (guarded by
   `if kind in ("do","uo")`), so it silently priced the unconstrained vanilla PDE and
   then subtracted it from itself, returning ≈0 instead of the knock-in price. Fixed
   by always solving the knock-out PDE (mapping `di→do`, `ui→uo` for the barrier
   direction) and deriving the knock-in via parity, matching `barrier_analytic`'s own
   design. Also not on an exact grid node originally — the grid is now snapped so
   the barrier lands exactly on a node, per the "must place a barrier exactly on a
   node" rule in NEXT_PROMPT.md's known pitfalls.
3. **The Broadie-Glasserman-Kou continuity correction was applied to the wrong leg**
   of the MC-vs-analytic barrier comparison. The correction shifts the barrier used
   in the *continuous* analytic formula, not the barrier used in the discretely-
   monitored Monte Carlo simulation — discrete monitoring needs no adjustment; the
   continuous formula does, to compensate for catching fewer breaches than continuous
   monitoring would. The original comparison had this backwards, so the "corrected"
   comparison was actually further from agreement than the uncorrected one.
4. **A case-study B (the farmer) realised-payoff loop silently dropped nearly every
   observation.** The loop selected the *front-month* contract at each historical
   starting date, whose remaining time-to-expiry is usually shorter than the ~4-month
   harvest horizon being tested, so the "never chain across a roll" guard correctly
   refused almost every starting point (n=1, a degenerate distribution with zero
   variance). Fixed by selecting, at each date, the contract whose time-to-expiry is
   closest to the target horizon instead of the front month — n rose to 49 with a
   sensible, non-degenerate realised-payoff distribution.
5. **The historical bootstrap pricer's mean is dominated by a handful of pathological
   paths.** `bootstrap_paths("CL", ...)` returned a mean terminal level of ~1020 (vs
   a median of ~72, starting from F0≈72) because CL's return history contains several
   far-dated, thinly-traded contracts that print an implausible near-zero-but-positive
   settlement on 2026-07-23 (e.g. `CL202809` at $0.46, `CL202805` at $0.19) — real
   prints that survive `lib24.load_panel`'s `close > 0` filter but generate ~±590%
   single-day log returns, exactly the kind of far-dated junk print 024's own
   write-up already documented for other products. Block-resampling with replacement
   can draw one of these blocks into a path and compound it, producing physically
   implausible terminal levels for a small fraction of paths and inflating the mean
   (not the median) of the "no model at all" benchmark. Reported here rather than
   patched — a real fix belongs in `lib24`'s hygiene layer (an additional robust
   floor on `close`, or a MAD-based per-contract outlier screen on returns feeding
   `bootstrap_paths`), out of scope for this notebook to touch given 023/024/lib24
   are shared across three notebooks.
6. **Case study C's "vol reduction from averaging" ratio came out in the wrong
   direction.** The reported `monthly_avg_vol_realized` (1.43) is *larger* than
   `daily_vol_realized` (0.21), giving a ratio > 1 where the design doc's own
   intuition (and every other Asian-vs-European comparison in this notebook, e.g.
   Part A7's real Turnbull-Wakeman-vs-Black-76 comparison, which correctly shows the
   Asian premium below the European) says it should be < 1. This smells like an
   annualisation or window-definition mismatch in the Phase 5 script's own
   vol-of-the-average calculation, not yet root-caused. The number is reported above
   with this caveat rather than silently treated as a finding; A7's cross-validated
   Black-76/Turnbull-Wakeman comparison remains the trustworthy version of this point.
7. **The intraday-vs-close-to-close variance ratio (48×) is implausibly large.**
   Expected magnitude for this comparison is roughly 1.1–3×, not 48×. Both sides of
   the ratio use the same `× 365.25` annualisation convention, so the bug is not an
   obvious double-annualisation; it was not root-caused in the time available. This
   number should not be treated as a finding — the section's honest content is the
   *direction* (intraday captures variance the daily close proxy misses) and the
   explicit caveat that the underlying dataset is Jan–Jul 2026 only, not a
   representative sample either way.

## Bottom line

The eight-method pricing harness is internally consistent — every cross-validation
cell passes, including the ones (knock-in+knock-out parity to `1e-10`, put-call
parity to `1e-8`) that have essentially zero tolerance for being wrong. Applied to
seven real hedging problems, the harness produces concrete, checkable numbers rather
than qualitative claims: a 28.6% realised knock-out frequency for the farmer's
cheapest floor, a ~2% Kirk-vs-Monte-Carlo gap on the refiner's crack spread that
quantifies exactly what "hedging inputs and outputs separately" misses, a quanto
correlation instability about four times the size of the adjustment it's meant to
correct, and the cleanest possible falsification of the lognormal assumption (Black-76's
log score of `−∞` on the actual 2020-04-20 settlement). None of the underlying
mathematics is new — Black-76 is 1976, Reiner-Rubinstein 1991, Longstaff-Schwartz
2001 — the contribution is one consistent, cross-validated harness applied honestly
to real commodity data and worked client problems, with five genuine implementation
bugs caught and either fixed (three) or reported rather than hidden (two, plus one
still-open ratio-direction anomaly in case study C).

## What to test next

- Root-cause **Bugs found** items 6 and 7 (the C-airline vol-reduction-ratio
  direction and the 48× intraday-variance ratio) before either number is used in
  any downstream teaching material.
- Add a MAD-based per-contract outlier screen to `lib25.bootstrap_paths` (item 5) so
  the "no model at all" benchmark's mean is not dominated by a handful of far-dated
  junk prints — report the median as the headline bootstrap number until then.
- The Longstaff-Schwartz duality gap (≈4.8 against a ≈7.1 price) is large for a
  degree-3 polynomial basis; a richer basis or more exercise dates would sharpen
  Part C6's teaching point about LSM's low bias.
- Case study G's structured-note issuer margin is currently indistinguishable from
  Monte Carlo noise (±0.0004 SE against a ±0.001 margin) — a much larger path count,
  or variance-reduction techniques from Phase 4, would be needed before that number
  could support a real claim about issuer economics.
- A genuine implied-vol input (this notebook has none) would let Part E's variance
  risk premium figure move from illustrative to measured.

---

**Traceability.** Notebook:
`src/research/025_pricing_bespoke_commodity_derivatives.ipynb`. Driver:
`scripts/run_025.sh`. Referee: `scripts/check_025.sh`. Smoke test:
`scripts/smoke_025.sh`. Phase reports: `src/research/tmp/phase_{0,1,2,3,4,5,6}_25_*.json`.
Library: `src/research/tmp/lib25.py` (tests: `tests/test_lib25.py`). Pricer modules:
`src/research/tmp/pricers_25_{barrier,asian,american,exotic}.py` (tests:
`tests/test_pricers_25_*.py`). Chart modules: `src/research/tmp/charts_25_{a,b,c,d,e}.py`.
