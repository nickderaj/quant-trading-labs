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

One convention worth stating once, because it changes how the case-study numbers
read: every realised payoff below is the payoff of the **hedge leg on its own**, not
of the hedged position. A collar showing a large negative mean has not lost the
client money — its short option is giving back gains on a physical position that is
not drawn. Compare the structures against each other, not against zero.

## The answer

| Question | Answer |
|---|:---:|
| Do independent pricing methods agree (Phase 2)? | **Yes — 17/17 cross-validation cells pass** |
| How often would the farmer's knock-out have knocked out? | **28.6% of windows (ZW), n=49; 26.0% (KE)** — but only **24.5%** cost anything |
| What does that knock-out cost when it costs something? | **≈116 cents/bushel** (≈$1.16/bu) of lost floor, to save **55%** of the premium |
| Does hedging inputs and outputs separately hedge the refiner's margin? | **No, exactly** — Kirk vs bivariate-MC spread-option prices agree to ~2.0%, and that price is the correlation term the separate hedges miss |
| How much does averaging actually save the airline? | Vol of the monthly average is **89%** of a single fixing's; the Asian prices **41% below** the European |
| Is the gas utility's winter premium about seasonality? | **No** — flat vol overprices by **15.5%** (Samuelson); winter-delivery conditioning moves it **−0.5%** |
| What did Black-76 say about 2020-04-20? | **Log score = −∞** (zero density below $0); Bachelier and displaced diffusion both assign finite density |
| How large is the quanto adjustment vs the instability of its own correlation? | Adjustment ≈ **$0.05**; ρ's ±1σ price range ≈ **$0.19** — the instability is ~4× the adjustment |
| Does the autocallable's model match history? | **Closely, observation by observation** — 65.4% model vs 72.3% realised, the whole gap at the first observation |
| What is the issuer margin on the structured note? | **≈ −0.11%**, which is **inside Monte Carlo noise** (SE ≈ 0.0004) — not a measurable margin |

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

`lib25.vol_term_structure` also takes a `delivery_months` argument, which restricts
the fit to contracts delivering in a given set of months. That is how 024 Phase 5
measures seasonality, and case D needs it — see below for why the notebook previously
had no way to express a seasonal vol at all.

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

Four genuine implementation bugs were found and fixed here, none of them in the
module authors' own unit tests (see **Bugs found**): a sign error in both the barrier
and American finite-difference operators, the barrier PDE not actually placing the
barrier on a grid node for two of the four barrier kinds, the
Broadie-Glasserman-Kou continuity correction applied to the wrong side of the
MC-vs-analytic comparison, and a degenerate control-variate measurement that reported
an impossible convergence rate.

### Phase 3 — Primer mechanics

Pure payoff geometry, three real CL front-month paths over the COVID window (one
breaching an illustrative barrier, two not), and the single most useful number in the
primer: the **realised knock-out frequency** for CL, by barrier distance and tenor,
split crisis vs calm regime, over the full 16-year sample — this feeds directly into
case study B.

### Phase 4 — Method behaviour

Wall-clock timing, convergence traces (binomial, Monte Carlo, PDE), variance-reduction
efficiency, the Longstaff-Schwartz duality gap (≈4.8, roughly two thirds of the LSM
price itself — a large, honest gap that the degree-3 polynomial basis sets, not the
path count), and a one-factor Markov functional calibrated to realised terminal
marginals, which reprices its own calibration exactly (error = 0.0 by construction)
while its path-dependent price differs from a plain-GBM price by about 4.7% — pure
joint-dynamics error, cleanly isolated because the marginal channel is closed off by
construction.

On variance reduction there are two separate experiments, and they answer different
questions. Phase 4 uses the **terminal price** as a control for a **European** call:
a weak control, worth ≈**6.3×** in variance. Phase 2's convergence chart uses the
**geometric Asian** as a control for the **arithmetic Asian**: a near-perfect control,
worth ≈**690×**. Antithetic and Sobol earn ≈1× on a smooth one-dimensional European,
which is the honest result at this scale rather than a failure — both earn their keep
at higher dimensionality than this test exercises. Every leg converges at the
1/√N rate; a control variate moves the level of the error, never its rate.

### Phase 5 — Case studies A–D

**A, the refiner.** Hedging the 3:2:1 crack spread margin (long crude, short
products) directly (Kirk + bivariate MC, ~2.0% apart) vs hedging inputs and outputs
separately: the two differ by the correlation term (CL–HO ρ≈0.87, CL–RB ρ≈0.89), a
concrete, non-zero number rather than an assertion. Every structure is re-struck at
each historical window's own crack level.

**B, the wheat farmer — the case study the whole notebook is built around.** Plain
put vs zero-cost collar vs 15%-OTM knock-out put on ZW (cross-checked on KE), each
re-struck at every historical start date's own forward. Over 49 historical ~4-month
windows the barrier was touched **28.6%** of the time (26.0% on KE) — but only
**24.5%** of windows were knock-outs that cost anything; on the other two the plain
put would have expired worthless regardless. Averaged over the costly ones the farmer
gave up **≈116 cents/bushel** of protection, about **$1.16/bu** (ZW is quoted in
cents). The trade is that shortfall, at roughly a one-in-four chance, against a
**55%** premium saving — 9 cents for the knock-out against 20 for the plain put.

**C, the airline / diesel buyer.** A monthly-average (Asian) call on HO vs a
single-date European, with the collar built the right way round for a *buyer* (long
the cap, short the floor). The realised vol of the monthly average is **89%** of the
vol of a single settlement date, and the Asian consequently prices **41% below** the
equivalent European. The wrong textbook `σ/√3` shortcut misprices it by **−2.1%**
against the schedule-correct calculation — small on this schedule, but a shortcut
with no reason to be small in general.

**D, the gas utility.** A winter Dec–Mar cap strip, priced three ways to separate two
effects that are easy to conflate: one flat front-month vol for all four months, the
unconditional Samuelson term structure, and a term structure fitted only to
winter-delivery contracts. Quoting the strip off a single front-month vol
**overprices it by 15.5%** against the maturity-aware curve. Conditioning on winter
delivery — genuine seasonality, the thing this case study is nominally about — moves
it by **−0.5%**. NG's winter vol premium is real at the front of the curve (the
winter/summer ratio is 1.17 at a 15-day tenor, 1.09 at 45 days) but changes sign
repeatedly across the 180–365 day tenors this strip actually spans, so the fitted
seasonal and unconditional term structures are indistinguishable. **The Samuelson
effect does essentially all the work; the seasonal effect is not measurable at this
horizon.** The knock-out cap, at a barrier that can actually be touched (90% of each
month's forward, hit in 30% of historical windows), saves **8.3%** of the premium.

### Phase 6 — Case studies E–G, and the variance aside

**E, negative prices.** A $10 put on CL202005 priced on 2020-04-06. Black-76 assigns
the actual 2020-04-20 settlement (−$2.67) **zero probability** — log score `−∞`, the
cleanest falsification in the notebook — while Bachelier (−189.1) and displaced
diffusion (−233.3) both assign it finite, computable density. The systematic version
(not built on one day): rolling 1% left-tail Kupiec coverage over the full 16-year
sample rejects **both** families, in opposite directions. The lognormal-return model
never breaches its own 1% VaR at all (0 exceedances in 4168 days, `p≈5.5e-20`), so
its left tail is far too *wide* — too conservative. The level-change
(Bachelier-consistent) model breaches on **2.06%** of days against a 1% target
(`p≈1.6e-9`), roughly twice too often, so its tail is too *narrow*. Failing a
coverage test by being too cautious and failing it by being too aggressive are
different diagnoses with different fixes.

**F, the European buyer.** Quanto (fixed-FX) vs composite (floating-FX) calls on CL
settled in EUR. The quanto drift adjustment is small (≈$0.05 vs an unadjusted
$4.77 call), but the price range implied by the CL–6E correlation's own rolling
±1σ dispersion (≈$0.19) is **about four times larger than the adjustment itself** —
exactly the finding the design doc predicted, now with real numbers behind it.

**G, the structured note (capstone).** A one-year autocallable reverse convertible on
CL: 8% coupon, 100% autocall trigger, 70% principal barrier, quarterly observations.
The model puts the probability of autocalling at any observation at **65.4%**; CL's
own 2010–2026 history gives **72.3%** over 238 downsampled one-year windows. Broken
out by observation the two track each other closely — model 47.5% / 11.9% / 6.0% /
0.0%, history 53.8% / 11.8% / 6.7% / 0.0% — with the whole gap at the first
observation, which is what you would expect from a model priced under a driftless
risk-neutral measure against history carrying oil's actual drift. Model fair value
($1.0011) sits almost exactly at face ($1.00), an implied issuer margin of about
−0.11%, which is **smaller than the Monte Carlo standard error around it**
(≈0.0004): this notebook cannot distinguish a real margin from simulation noise here,
and does not claim to.

**Variance-swap aside.** The fair strike (≈32.0% vol) is reported as a *forecast*,
never a replicated price — there is no option strip in this repo to replicate one.
Its QLIKE is deliberately **not** compared against the naive baseline as if that were
a contest: the strike is one constant fixed on the valuation date and scored over the
whole sample, while the baseline re-reads yesterday's realised variance every day, so
a daily-updating forecast beating a static one says nothing about either.

On intraday vs close-to-close variance: over the 170 common dates (Jan–Jul 2026 only,
a seven-month window and not a representative sample), the **ratio of means is 0.83**
and the **median of the per-day ratios is 1.78**. Those point in opposite directions,
and that disagreement is the honest content: the typical day has more intraday
variance than its close-to-close proxy captures, while the sample's total variance is
dominated by a handful of large close-to-close moves. Neither number supports a
general claim at this sample size.

### Phase 7 — Notebook and charts

35 figures across Parts A–E (11 primer, 5 inputs, 8 methods, 9 case studies, 2
limits). NEXT_PROMPT.md states a target of 30–34 figures but its own per-part counts
sum to 35; the per-part counts were treated as binding. All seven case studies run
through the identical five-step template the spec asks for. The notebook executes
cleanly end to end (126 cells, 0 error outputs) with `jupyter nbconvert --execute`.

## Bugs found

Reported in the order they were caught, following this repo's convention of
documenting what went wrong rather than only what went right. Every item below is
fixed unless it says otherwise.

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
4. **The control-variate leg of the convergence chart was measuring floating-point
   noise, not a standard error.** Phase 2 called `asian_mc(..., control_variate=True)`
   with a single averaging index (the terminal point). Over one fixing the geometric
   and arithmetic averages are the same number, so `beta = Cov/Var` is exactly 1, the
   adjusted payoff collapses to a constant array, and the reported "SE" is round-off:
   it came out as exactly `se_plain / n_paths` at every path count, giving a log-log
   slope of −1.5. No Monte Carlo estimator can converge faster than −0.5, so the chart
   was claiming something impossible and captioning it as a ~1000× variance reduction.
   Fixed by averaging over the whole path, which is the payoff a geometric control
   variate is actually a control for; the honest figure is ≈690× on the arithmetic
   Asian, against ≈6.3× for Phase 4's much weaker terminal-price control on a
   European. The pricer itself was correct — only the call site was wrong.
5. **A case-study B (the farmer) realised-payoff loop silently dropped nearly every
   observation.** The loop selected the *front-month* contract at each historical
   starting date, whose remaining time-to-expiry is usually shorter than the ~4-month
   harvest horizon being tested, so the "never chain across a roll" guard correctly
   refused almost every starting point (n=1, a degenerate distribution with zero
   variance). Fixed by selecting, at each date, the contract whose time-to-expiry is
   closest to the target horizon instead of the front month — n rose to 49 with a
   sensible, non-degenerate realised-payoff distribution.
6. **Case study B then priced sixteen years of history against a single 2017 strike.**
   All three structures' realised payoffs used the valuation-date forward `F` (446.18)
   while the knock-out barrier floated with each window's own `F_t` — so the barrier
   was re-struck and the options were not. That made the plain put far out of the
   money in every high-price year (mean payoff 2.5 with a median of 0) and gave the
   collar a −142.6 mean that was really a statement about where wheat traded in 2017
   rather than about the structure. Case A had it right all along (`crack_t`, not a
   frozen `crack_0`), which is what made this visible. Fixed by re-striking all three
   structures at each window's own forward; the collar's mean moves from −142.6 to
   +8.5 and the knock-out shortfall from 4.7 to 116 cents.
7. **The farmer's shortfall was reported in dollars but computed in cents.** ZW is
   quoted in cents per bushel, so `max(F - F_T, 0)` is a cent figure; the write-up
   and the D4 chart annotation both printed it with a dollar sign, overstating it
   a hundredfold. Fixed, and the report now carries an explicit `quote_units` field
   so the next reader does not have to infer it. The same pass added the distinction
   between knock-outs that cost something (12 of 14) and knock-outs on years the put
   would have expired worthless anyway (2 of 14), which is the number the decision
   actually turns on.
8. **Cases C and D hedged in the wrong direction.** Both compute
   `payoff_1 = -(F_T - F_t)` for their futures leg — but the airline buys diesel and
   the utility buys gas, so both are *long* futures and their hedge pays
   `+(F_T - F_t)`. The sign convention was correct for case A's short crack position
   and appears to have been copied across. Every realised-payoff number and both
   charts for these two cases read the hedge backwards. Fixed.
9. **Case C's Asian collar was a seller's collar, and its realised payoff was
   neither.** The airline buys diesel and needs a cap, so its collar is long the call
   and short the put. It was priced long-put/short-call (the wheat farmer's collar,
   the wrong side), while the realised-payoff loop computed
   `max(F_T − 0.95F_t, 0) − max(F_T − 1.05F_t, 0)` — a bull call spread, matching
   neither the pre-registration nor the priced structure. All three now agree.
10. **Case C's "vol reduction from averaging" ratio came out above 1**, which is
    impossible — averaging cannot add variance. Root-caused to three compounding
    errors in one calculation: (a) `lib24.mad_vol` annualises with `√252` but was
    being fed monthly observations, over-annualising by ~4.6×; (b) the "monthly
    return" was `log(last close / first close)` over a group spanning *all* contracts
    in that month, with the two sides sorted differently, so numerator and denominator
    were often different contracts — a roll, not a return; and (c) it measured a
    month-end-over-month-start return rather than the vol of the *average*, which is
    the only quantity an Asian option cares about. Rewritten to stay inside one
    contract, use the mean of the month's closes, and annualise from monthly
    observations. The ratio is now **0.888**, and the notebook cross-checks it against
    the Asian-vs-European premium ordering, which agrees.
11. **Case D's "seasonal vol" contained no seasonality.** The seasonal leg used
    `inputs["sigma"](T_i)` — the unconditional Samuelson maturity power law, which has
    no month-of-year term in it at all — and compared it to a flat front-month vol.
    The resulting gap is real but it is the Samuelson effect wearing a seasonal label.
    The write-up compounded this by reporting the direction backwards: the seasonal
    leg (0.721) is *cheaper* than the flat leg (0.832), not dearer, so a flat-vol desk
    **over**prices this strip rather than underpricing it. Fixed by adding
    `delivery_months` to `lib25.vol_term_structure` and pricing all three legs
    separately. The honest result is that the seasonal effect is **−0.5%** at these
    tenors against the Samuelson effect's **15.5%** — NG's winter vol premium is a
    front-of-curve phenomenon that does not survive to 6–12 months. The raw
    winter/summer bucket ratios are now stored in the report and plotted, so the claim
    can be read off the data.
12. **Case D's knock-out cap was a no-op sold as a saving.** A down-and-out call
    struck at the forward with its barrier at 0.7×F can only knock out in states where
    the call is already far out of the money: the analytic saving was 2.4e-07 of
    premium, and the realised-payoff array was byte-identical to the plain cap because
    the barrier was tested once against `F_T` rather than monitored along the path
    (and a call that finishes in the money necessarily passes an `F_T ≥ 0.7·F_t`
    test). The pre-registered structure therefore returned a null result, and the
    previous write-up dropped it silently instead of reporting it. Fixed by monitoring
    the path and by sweeping the barrier rather than asserting one level; the headline
    is now a 90% barrier saving 8.3% of premium and touched in 30% of historical
    windows, with the full ladder — including the degenerate 0.7 case — in the report.
13. **Case G compared a probability against a per-observation average.** The model's
    autocall probability was computed as `autocall_frequency_by_obs[:-1].sum() /
    len(obs_idx)`. The per-observation frequencies are disjoint events, so they sum to
    the total probability; dividing that sum by the number of observation dates turned
    65.4% into 16.3%. The previous write-up built its capstone finding on the
    resulting 72.3%-vs-16.3% gap and explained it with oil's historical drift. With
    the arithmetic fixed the comparison is 72.3% vs 65.4%, the two line up closely
    observation by observation, and the whole remaining gap sits at the first
    observation — a much smaller effect that the drift explanation actually fits.
14. **Figure D9 drew a Gaussian and labelled it the realised payoff distribution.**
    The chart called `np.random.normal(mean, std)` and histogrammed the result. The
    real distribution is nothing like a normal: p25, median and p75 are all the same
    number (most windows autocall and pay exactly the same redemption amount) with a
    long left tail to 0.30. Its right panel then spread the total autocall frequency
    uniformly across the four observation dates — the same divide-by-observation-count
    error as item 13, independently reproduced in the chart layer, hiding a sharply
    front-loaded profile. Phase 6 now stores the actual payoff values and the realised
    per-observation timing, and the figure plots both against the model.
15. **The intraday-vs-close-to-close variance ratio (48×) was a mean of per-day
    ratios.** Days that close near where they opened have a near-zero close-to-close
    variance in the denominator, and a handful of those dominate the average. The
    ratio of means — the estimator that answers the question being asked — is 0.83,
    and the median per-day ratio is 1.78. The previous write-up flagged the 48× as
    implausible but kept trusting its *direction*, which its own two reported means
    already contradicted. Fixed; all three statistics are now reported, with the
    mean-of-ratios retained under a name that says not to use it.
16. **The historical bootstrap pricer's mean is dominated by a handful of pathological
    paths.** *(Reported, not fixed.)* `bootstrap_paths("CL", ...)` returned a mean
    terminal level of ~1020 (vs a median of ~72, starting from F0≈72) because CL's
    return history contains several far-dated, thinly-traded contracts that print an
    implausible near-zero-but-positive settlement on 2026-07-23 (e.g. `CL202809` at
    $0.46, `CL202805` at $0.19) — real prints that survive `lib24.load_panel`'s
    `close > 0` filter but generate ~±590% single-day log returns, exactly the kind of
    far-dated junk print 024's own write-up already documented for other products.
    Block-resampling with replacement can draw one of these blocks into a path and
    compound it, producing physically implausible terminal levels for a small fraction
    of paths and inflating the mean (not the median) of the "no model at all"
    benchmark. Reported here rather than patched — a real fix belongs in `lib24`'s
    hygiene layer (an additional robust floor on `close`, or a MAD-based per-contract
    outlier screen on returns feeding `bootstrap_paths`), out of scope for this
    notebook to touch given 023/024/lib24 are shared across three notebooks. Report
    the median as the headline bootstrap number until then.
17. **The referee only ever checked Phase 2.** `scripts/check_025.sh` printed the
    cross-validation matrix and nothing else, which is how items 8–15 all shipped
    while the banner read 17/17 PASS: every one of them lives in Phase 5, Phase 6, or
    the chart layer, and nothing looked there. The script now also asserts the
    invariants those bugs violated — that every Monte Carlo leg converges at the
    1/√N rate, that a buyer's cap is positive only where its long futures leg is,
    that the vol of an average sits below the vol of a single fixing and the Asian
    below the European, that a knock-out is cheaper than the vanilla by an amount
    worth quoting and has a barrier that can actually be touched, that the seasonal
    and maturity legs are reported separately, that case B declares its quote units,
    that the autocall frequencies sum to a probability and complement the
    never-autocalled rate, and that the variance ratio is a ratio of means.

## Bottom line

The eight-method pricing harness is internally consistent — every cross-validation
cell passes, including the ones (knock-in+knock-out parity to `1e-10`, put-call
parity to `1e-8`) that have essentially zero tolerance for being wrong. Applied to
seven real hedging problems, the harness produces concrete, checkable numbers rather
than qualitative claims: a 28.6% barrier-touch rate for the farmer's cheapest floor
of which only 24.5% cost anything, a ~2% Kirk-vs-Monte-Carlo agreement on the
refiner's crack spread that prices exactly what "hedging inputs and outputs
separately" misses, a quanto correlation instability about four times the size of the
adjustment it's meant to correct, an autocallable whose model and history agree
observation by observation, and the cleanest possible falsification of the lognormal
assumption (Black-76's log score of `−∞` on the actual 2020-04-20 settlement).

Two of the case studies now report a *negative* result where they previously reported
a finding, and both are more useful for it. The gas utility's winter cap premium turns
out to be almost entirely a Samuelson effect rather than a seasonal one — the seasonal
term is −0.5% against 15.5% — and the structured note's issuer margin is smaller than
the Monte Carlo noise around it, so the notebook says it cannot measure it.

None of the underlying mathematics is new — Black-76 is 1976, Reiner-Rubinstein 1991,
Longstaff-Schwartz 2001 — the contribution is one consistent, cross-validated harness
applied honestly to real commodity data and worked client problems, with seventeen
implementation and reporting bugs caught, sixteen fixed and one documented as out of
scope. The largest lesson is item 17: a referee that checks only the part of the
pipeline you were worried about will certify the rest of it as sound.

## What to test next

- Add a MAD-based per-contract outlier screen to `lib25.bootstrap_paths` (item 16) so
  the "no model at all" benchmark's mean is not dominated by a handful of far-dated
  junk prints — report the median as the headline bootstrap number until then.
- The Longstaff-Schwartz duality gap (≈4.8 against a ≈7.1 price) is large for a
  degree-3 polynomial basis; a richer basis or more exercise dates would sharpen
  Part C6's teaching point about LSM's low bias.
- Case study G's structured-note issuer margin is currently indistinguishable from
  Monte Carlo noise (±0.0004 SE against a ±0.001 margin) — a much larger path count,
  or the variance-reduction techniques from Phase 4, would be needed before that
  number could support a real claim about issuer economics.
- Case D's seasonal result is a null at 6–12 month tenors on a 756-day causal window.
  Re-running it on the front of the curve, where 024 finds the winter premium
  concentrated, would turn the null into a positive result about *where* seasonality
  is priceable — a better teaching point than either version of this case study has
  made so far.
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
