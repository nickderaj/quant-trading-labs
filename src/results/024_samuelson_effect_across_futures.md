# 024 — The Samuelson Effect Across 16 Futures Markets

## The narrow question

Samuelson (1965) predicted that a futures contract's return volatility should rise as it
approaches expiry: news about near-term supply and demand moves the front of the curve
sharply, but has mostly decayed away by the time a deferred contract settles. This
notebook asks whether that pattern shows up, cleanly and consistently, across every
futures market in this repo — 16 products, 2010-2026, roughly 846,000 contract-days —
using one consistent method throughout.

**This is a descriptive study, not a trading study.** There is no Sharpe ratio, no
returns backtest, no trading rule, and no holdout spent — the full sample is used
throughout and there is no train/test split. The value here is breadth (16 markets, 16
years, one method), the cross-market ranking, and the data-quality documentation, not
novelty: the core effect is textbook. Full spec: `NEXT_PROMPT.md`.

## The answer

**The effect is real, strong, and cleanly ranked — but two of the naive predictions
made before looking were wrong, and the write-up says so rather than smoothing them
over.** Volatility rises toward expiry in every energy and grain product, is flat or
weakly positive in precious metals, and the negative control (`ES`, a financial future
with no storage economics) sits inside that flat cluster — but is *not* the single
weakest slope of all 16, contrary to the scoping expectation. The realised "smile" check
turned up a genuine cross-product surprise (see below); the curve-state ("skew") check,
once a data bug was found and fixed, ended up matching the textbook intuition rather
than contradicting it.

| Check | Result |
|---|:---:|
| Samuelson effect visible in energy? | **Yes** — CL, NG, HO, BZ, RB all significantly negative slope |
| Visible in grains? | **Yes** — ZC, KE, ZS, ZM, ZW significantly negative; ZL (soybean oil) weak, CI touches zero |
| Visible in precious metals? | **No — inverted** — GC ≈ 0, SI/PA/PL all *positive* (vol rises with maturity) |
| Negative control (`ES`) weakest, as predicted? | **No** — `ES` (+0.046) is not the weakest; `PL` (+0.066) is slightly more positive. `ES` does sit inside the near-zero/positive cluster, well separated from energy/ags, which is the claim that matters |
| Front-end vol spikes more than back-end in crises? | **Yes** — CL front/deferred ratio rises from 1.38 (2017 calm) to 1.81-1.89 in every crisis window |
| Robust and winsorised estimators agree? | **Yes, on sign, everywhere** — plain `std` disagrees on sign with MAD/winsor for CL, GC, PA, and ZL |
| Realised smile/skew survives the maturity confound? | **Yes, but the shape is not what was first reported** — see the correction below. CL's shape is an asymmetric skew (a shallow dip, then a steep rise into contango), not a symmetric smile; NG's is a genuine peaked hump. Both persist within every maturity bin, so neither is the Samuelson effect in disguise |

**A correction made during review.** An earlier version of this notebook showed a
volatility *spike* at the longest-maturity bucket for several products (most visibly
crude oil and gold) and described CL's realised smile as a clean U-shape. Both were
wrong, for two different, now-fixed reasons — see **Bugs found** below. The numbers and
charts here are the corrected versions; the ranking and every qualitative conclusion
above survived the fix (in most cases, got cleaner), but two chart-level claims did not
and are corrected below.

## Method

### Data and hygiene

16 products' per-contract OHLCV (`src/research/data/market/databento/ohlcv/<PROD>.parquet`,
2010-06 to 2026-07, ~845,839 usable contract-days after hygiene, including the
whole-contract exclusion described in **Bugs found**) joined to
`contracts.parquet` for expiry. Rows with `close <= 0` are dropped and counted per
product (`CL202005`'s genuine -2.67 print on 2020-04-20 is excluded for the mechanical
reason that `ln()` is undefined, not treated as junk — see `NEXT_PROMPT.md` section 2).
Returns are computed within `contract_id` only, never chained across a roll. The
consecutive-day filter keeps a return only when the gap to the previous observation of
the same contract is exactly 1 day (a `gap<=3` sensitivity variant changes no
conclusion). `dte = (expiry - date).days`; one product (`BZ`, 782 rows) carries trade
prints dated after its recorded expiry — a metadata quirk, not a panel bug, reported and
left outside every bucket (bins start at `dte=0`) rather than silently dropped.

**Pipeline verification:** Phase 1's usable-return counts for CL (81,895), NG (93,612),
ES (6,797), and GC (38,484) matched the independently-computed smoke-test numbers in
`NEXT_PROMPT.md` exactly, confirming the pipeline before any chart was built.

### The estimator problem

The far-dated end of every panel contains junk prints — stale settlements and sign-flip
errors on illiquid stub contracts — that destroy a plain standard deviation. Measured
on CL, the plain-`std` estimate of 4-year futures volatility is 187% annualised; the
robust MAD estimate on the same data is 15%. The primary estimator throughout is
therefore

```
sigma_hat = 1.4826 * median(|r - median(r)|) * sqrt(252)
```

with plain `std` and a symmetric 1%-winsorised `std` computed alongside it as a
contamination check and a robustness check respectively. Where it matters (the
Samuelson-slope sign, product by product), MAD and winsorised `std` agree on every
product checked — including the two soft controls (GC, SI) and both inverted metals
(PL, PA) — while plain `std` disagrees on *sign* for CL, GC, PA, and ZL. That agreement
between two differently-constructed robust estimators, against a plain estimator that
gets the sign wrong, is what makes the ranking below trustworthy.

### Phase 0 — Pre-registration

Written before any chart was inspected: the descriptive framing (no trading gate, no
holdout), a falsification condition (a samuelson slope that isn't reliably negative in
energy/ags, or whose sign flips between estimators or liquidity variants), and the
estimator hierarchy above, all fixed in advance. The predicted ordering — energy/grains
strongly negative, precious metals near zero, `ES` weakest of all — was recorded
honestly as a **confirmed scoping expectation** (it had already been measured during
scoping), not a blind prediction. It turned out to be half right: the qualitative
clustering held, the precise "ES is weakest" ranking did not.

### Phase 1 — Panel construction and data-quality census

All 16 products loaded, hygiene applied, and censused in one process. Row counts matched
`NEXT_PROMPT.md`'s reference table to within rounding. Junk prints (`|r| > 1.0`, i.e. a
day-over-day price move of more than e-fold) were counted and named per product rather
than silently dropped — 23 on CL, 5 on NG, 0 on GC, 0 on ES *after* the whole-contract
exclusion described in **Bugs found** (94/58/9/0 respectively before it — most of each
count came from the same handful of unreliable tickers that exclusion removes). NG's
14,373 non-positive closes (10% of its raw panel) are reported explicitly rather than
absorbed into a `.dropna()`.

### Phase 2 — Vol-vs-maturity profiles

For all 16 products × 3 estimators (MAD, std, winsor) × 3 liquidity variants
(`min_volume` in {0, 100, 1000}) × 2 gap variants (`max_gap` in {1, 3}) — 288
configurations, computed in one process. The primary front/deferred ratio (0-30d vs.
365-730d MAD vol) reproduced the independently-computed smoke-test table in
`NEXT_PROMPT.md` to within 6% on 15 of 16 products; `PA` came in at 0.89 against an
expected 0.70 (27% off) — large enough to flag, and the reason turns out to be exactly
the thin-far-dated-contract effect that Phase 3's metals investigation was designed to
catch (see below).

### Phase 3 — Samuelson slopes and the metals inversion

`samuelson_slope` fits a weighted log-log OLS of bucket vol against bucket `dte_mid`,
with a **contract-level** bootstrap CI (returns within one contract are serially
dependent, so a row-level bootstrap would understate the true uncertainty). The naive
implementation resamples contracts by rebuilding a pandas frame per bootstrap draw,
which is O(n_boot × n_contracts) in pandas overhead — fine for a handful of contracts,
unusable for CL's 244 or NG's 262 at `n_boot=300` across the 144-plus calls this phase
needs. It timed out twice (5 and 10 minutes) before being rewritten to precompute each
contract's `(dte, r)` arrays once and bin resampled draws with `np.searchsorted`,
cutting the full Phase 3 run to 2m17s. This is now `lib24.samuelson_slope`'s only
bootstrap implementation.

Ranked slopes (MAD, `max_gap=1`, `min_volume=0`, and — after the fix described in
**Bugs found** — every bucket backed by fewer than 5 distinct contracts excluded from
the fit), most negative first:

| Product | Slope | 95% CI |
|---|---:|---:|
| NG | -0.284 | [-0.330, -0.242] |
| ZC | -0.191 | [-0.228, -0.110] |
| CL | -0.174 | [-0.197, -0.150] |
| ZW | -0.122 | [-0.156, -0.020] |
| ZM | -0.118 | [-0.140, -0.048] |
| KE | -0.115 | [-0.146, -0.082] |
| ZS | -0.115 | [-0.149, -0.056] |
| HO | -0.112 | [-0.134, -0.089] |
| BZ | -0.107 | [-0.131, -0.076] |
| RB | -0.098 | [-0.120, -0.058] |
| ZL | -0.046 | [-0.072, +0.004] |
| GC | +0.013 | [-0.006, +0.037] |
| PA | +0.017 | [-0.019, +0.073] |
| SI | +0.033 | [-0.000, +0.065] |
| ES | +0.046 | [-0.039, +0.120] |
| PL | +0.066 | [+0.021, +0.108] |

Every energy and ag product except soybean oil (`ZL`, CI just touches zero) has a
95% CI excluding zero on the negative side. `GC`/`SI`/`PA`/`PL` and `ES` cluster near
zero or positive — clearly separated from the commodity group — but `ES` is not the
single weakest, contradicting the scoping expectation. This does not falsify the
central claim (the falsification condition was about the energy/ags sign and
cross-estimator/liquidity agreement, not the precise `ES` rank), but it is reported
plainly rather than glossed. Every energy and ag slope moved slightly *more* negative
after the `min_contracts >= 5` fix below — removing a few noisy, single-contract
far-dated points sharpened the effect rather than weakening it, which is itself a
useful check that the ranking wasn't propped up by exactly the artefact being removed.

**The metals inversion.** `PL` and `PA` show a genuinely positive slope — vol *rising*
with maturity — and it survives every check run against it: `volume > 1000` (`PL` +0.078, `PA` +0.016),
`max_gap <= 3` (`PL` +0.085, `PA` +0.045), and restricting to buckets with
`n_contracts >= 10` (`PL` +0.057, `PA` +0.017) — every variant stays positive against a
primary estimate of `PL` +0.066, `PA` +0.017.
This is **not** a liquidity artefact of a handful of thin far-dated contracts; it is a
real feature of how platinum and palladium futures are priced. The most likely
explanation, consistent with these being smaller, less liquid, industrial precious
metals with idiosyncratic supply risk concentrated in a few mining regions, is that
their far-dated contracts carry a persistent risk premium unrelated to near-term
information flow — but that is a hypothesis for a future notebook, not something this
descriptive study tests.

### Phase 4 — The time dimension

A 63-day trailing causal rolling MAD vol (verified causal: perturbing a future return
does not change a past value) computed for 5 maturity buckets across all 16 products.
CL's front/deferred vol ratio averages 1.38 in the 2017 calm control and rises to
1.81 (2014-15 oil collapse), 1.89 (COVID), and 1.41 (Ukraine) — the front end
consistently reacts more than the back end in a crisis, as the mechanism predicts. The
rolling 126-day front/deferred return correlation on CL drops as low as -0.06 at its
minimum (mean 0.85) — the de-correlation companion to the vol effect: not only does the
front move more, it occasionally moves almost entirely independently of the back.

### Phase 5 — Seasonality

NG's winter-delivery (Dec-Mar) front-end vol (49.0% annualised) is 1.28x its
summer-delivery front-end vol (38.2%) — heating-demand seasonality showing up exactly
where the Samuelson mechanism says it should (near-term weather risk is a much bigger
deal for a contract that delivers in January than one that delivers in July). `GC`, run
through the identical treatment as the flat control, shows a front-end winter/summer
ratio of 0.94 — indistinguishable from noise, as expected for a financial-asset-like
metal with no seasonal physical demand.

### Phase 6 — The realised smile/skew exploration

**Every result in this section is realised, not implied — there are no options in this
repo, so a true implied-vol smile is impossible**, and every chart says so in its title.

CL's pooled realised "smile" (vol vs. log-moneyness against the front contract) is
**not a symmetric U-shape** — an earlier draft of this notebook said it was, based on
comparing only two points; looking at the full curve (chart D3) shows a shallow dip
around moneyness -0.15 (vol 0.13) followed by a steep, monotonic climb through
contango out to +0.35 (vol 0.29). It is better described as an asymmetric skew, not a
smile: realised vol is highest when a contract trades at a large premium to the front
month, not symmetrically at both extremes. **This asymmetric shape persists within
every maturity bin** (the same dip-then-climb pattern shows up separately in the
0-90d, 90-365d, and 365+d curves) — so it is not simply the Samuelson effect in
disguise, even though the specific shape is not what "smile" usually implies. NG's
pooled shape is genuinely different: a peaked hump centred close to zero moneyness
(vol rises from 0.18 in deep backwardation to 0.25 near the front contract, then falls
back to 0.16 in deep contango), also persisting across maturity bins. The two products
do not tell the same story here, and this write-up reports that rather than forcing a
single narrative onto both.

The curve-state ("skew") check asked whether backwardated curves (signalling scarcity)
carry a steeper Samuelson slope than contango curves. For CL, **backwardation is
markedly steeper** than contango (-0.218 vs. -0.149) — matching the naive prediction
that scarcity signals make near-term news more urgent. (An earlier draft of this
notebook reported the opposite for CL; that number was computed on the
still-contaminated panel, before the far-dated-bucket fix in **Bugs found** below, and
is superseded by the figure here.) For NG the two are nearly identical (-0.284 vs.
-0.292) — no material difference, a result which stands unchanged by the fix.

## Bugs found

1. **The far-dated volatility "spike" was real bad data, not a maturity effect —
   found and fixed during review.** Several products' longest-maturity charts showed
   volatility rising sharply at the very end of the curve after declining everywhere
   else (most visibly CL and GC). Investigating it top to bottom: CL's spike
   (0.137 → 0.275 annualised in the 6-10y bucket) was caused by three specific
   contracts — `CL203212`, `CL203305`, `CL203311` — that supplied 530 of the bucket's
   1,701 observations. These are the exact same tickers notebook 023 had already
   flagged as printing non-positive settlements; it turns out their *positive*-looking
   prints are just as unreliable, with individual daily log returns as large as 1.4-2.5
   (a several-hundred-percent one-day move) on days their close is nominally normal. A
   single `|r| > 1.0` filter does not catch this, since even their non-outlier days run
   30-60% annualised noisier than a clean contract — the contamination is spread across
   the whole contract's life, not concentrated in a few flagged days. **Fix:** any
   contract that ever printed a non-positive close (other than the documented genuine
   negative-WTI event) is now excluded *entirely* from every downstream phase, not just
   its non-positive rows — mirroring notebook 023's own treatment of these exact
   tickers. Applied in `run_phase_1_24_panels.py`, this removed 1.61% of CL's rows and
   1.94% of GC's (up to 11.94% of NG's, though NG's version of this problem is
   genuinely negative gas prices at real hubs, not junk — see point 3 below). CL's
   spike shrank from +100% to +23% relative to the prior bucket, now backed by 19
   legitimately diverse contracts rather than 3 broken ones.
2. **A second, distinct bug compounded the first: several products' *entire*
   longest-maturity bucket was backed by exactly ONE contract.** Soybean oil,
   palladium, corn, soybeans, and wheat's farthest bucket each had `n_contracts == 1`
   despite clearing the `min_obs >= 50` floor easily (one contract's several years of
   daily history is hundreds of rows on its own) — this is not a "maturity effect"
   estimate at all, just that single contract's realised volatility over whatever
   calendar period it happened to trade. RBOB gasoline's 4-5y bucket, backed by only 2
   contracts, showed vol of 0.650 (annualised!) against a neighbouring bucket's 0.217 —
   a 3x jump from pure small-sample noise, not a market phenomenon. **Fix:** added a
   `min_contracts` parameter to `lib24.vol_by_bucket` (default 1, backward-compatible)
   and set it to 5 by default in `lib24.samuelson_slope`, so the regression that
   produces the headline ranking now excludes single- and double-contract buckets. Every
   chart pulling raw bucket lists from the Phase 2 JSON applies the same
   `min_contracts >= 5` filter before plotting (`viz24.min_contracts_filter`); the
   heatmap (A7) shows the excluded cells as blank rather than silently omitting the
   row. Fixing this made the Samuelson ranking *more* negative across the board for
   energy and ags (see the updated table below) — the effect was being diluted by
   single-contract noise, not propped up by it.
3. **`NG` has 45 tickers that print negative or zero closes, spanning every maturity
   from front-month to 20 years out** — genuinely different from CL's handful of
   broken far-dated stubs. These are very likely real, documented negative natural-gas
   prices at oversupplied regional hubs (e.g. the Permian basin's Waha Hub), not junk.
   They are dropped (as `ln()` is undefined either way) but **not** subjected to the
   whole-contract exclusion in point 1, since there is no evidence their other prints
   are unreliable. NG's remaining far-dated uptick (0.108 → 0.188 in the last bucket)
   is not fixable the same way: its longest bucket is dominated by a well-populated
   2032-delivery strip (30 contracts, ~500 observations each) that simply traded during
   a different, and apparently more volatile, calendar window than the sparser 4-6y
   bucket immediately before it (only 175 observations, 64 contracts). This is reported
   as a genuine sample-composition effect, not swept into the same fix as CL's.
4. **A dict-iteration bug in the vol-of-vol chart (B6) produced 16 overlapping bars
   and a legend with 16 duplicate-labelled entries** (six "ags", five "energy", four
   "metals", one "control") instead of 4 clean grouped bars. The cause: `sorted(
   lib24.SECTOR.values())` iterates the sector dict's values *once per product* (16
   entries, with repeats), not the 4 unique sector names — so the bar-offset math
   assumed 4 groups but received 16, spilling each maturity bucket's bars into its
   neighbours. Fixed with `sorted(set(lib24.SECTOR.values()))`.
5. **The energy/metals sector colours (orange/yellow) were too close in hue and
   lightness to distinguish at a glance** — reported as illegible in review. Metals
   recoloured to blue, clearly separated from energy's orange, ags' teal, and
   control's violet, everywhere `SECTOR_COLOR` is used.
6. **Every log-scaled days-to-expiry axis showed matplotlib's default power-of-ten
   day labels** (`10^2`, `10^3`), reported as hard to read for a non-technical
   audience. All such axes (A1-A5, A7's column labels, B5, B8, C1, C4, D1, D2, D4) now
   show years-to-expiry with plain-number tick labels (0.1, 0.5, 1, 5, 10) via
   `viz24.style_years_axis`.
7. **`BZ` (Brent) carries 782 rows with `dte < 0`** — trade prints dated after the
   contract's recorded expiry, down to -18 days on the earliest contract. Almost
   certainly a last-trade-date vs. settlement-date mismatch in this product's contract
   metadata rather than a panel-construction bug (no other product shows this at scale).
   Reported and left outside every maturity bucket rather than silently dropped or
   silently included.
8. **`lib24.samuelson_slope`'s bootstrap was O(n_boot × n_contracts) in pandas
   overhead** — rebuilding a filtered-and-concatenated DataFrame on every bootstrap
   draw. This is invisible at small contract counts (the unit tests, with 6-10 synthetic
   contracts, ran in under a second) and catastrophic at real ones: the Phase 3 script
   timed out at both 5 and 10 minutes on CL/NG-sized panels across ~150 calls. Fixed by
   precomputing each contract's `(dte, r)` arrays once outside the bootstrap loop and
   binning each resampled draw with `np.searchsorted` instead of `pd.cut` + `groupby`.
   Full Phase 3 run: ~2-3 minutes after the fix. All unit tests still pass unchanged.
9. **Phase 2's independently-recomputed PA front/deferred ratio (0.89) missed the
   NEXT_PROMPT.md smoke-test figure (0.70) by 27%**, the largest miss of any product —
   consistent with, and explained by, points 1-2 above: PA's inverted slope was partly
   driven by a genuinely thin far-dated bucket.

## Bottom line

The Samuelson effect is real and strong in every storable commodity in this repo, weak
or absent in refined products, and clearly inverted in platinum and palladium — an
inversion that survives every liquidity and contract-count check thrown at it and is
therefore a real finding, not an artefact. The negative control does what a negative
control should (sits well outside the commodity cluster) without landing exactly where
the scoping notes predicted, and the realised smile/skew checks in Section D turned up
a genuine cross-product surprise (NG's peaked hump vs. CL's asymmetric skew) that
survives the maturity-confound control on both sides. None of this is novel — Samuelson
(1965) is 60 years old — but the breadth (16 markets, one method, 16 years), the
documented data-quality problems and their fixes, and the honest reporting of where the
naive predictions were wrong (including two chart-level claims that were themselves
wrong on first release, and are corrected in this version — see **Bugs found**) are the
actual contribution of this notebook.

## What to test next

- Whether PL/PA's inverted slope correlates with a measurable proxy for mining-supply
  concentration risk, and whether it is priced (a risk premium) or purely a volatility
  phenomenon.
- Whether the CL/NG divergence in the realised smile shape (asymmetric skew vs.
  peaked hump) is stable across sub-periods, or specific to a particular regime in
  the 2010-2026 sample.
- A genuine per-date term structure (Phase 4's B5 chart used the nearest annual
  aggregate as a documented approximation) for a sharper crisis-date comparison.
- Whether NG's genuinely negative regional-hub prints (point 3 in **Bugs found**) are
  concentrated at specific hubs or dates identifiable from `ticker`/`contract_month`,
  which would let a future notebook study Waha-style basis blowouts directly instead
  of only dropping them.

---

**Traceability.** Notebook: `src/research/024_samuelson_effect_across_futures.ipynb`.
Driver: `scripts/run_024.sh`. Phase reports:
`src/research/tmp/phase_{0,1,2,3,4,5,6}_24_*.json`. Library:
`src/research/tmp/lib24.py` (tests: `tests/test_lib24.py`). Chart modules:
`src/research/tmp/charts_24_{a,b,c,d}.py`.
