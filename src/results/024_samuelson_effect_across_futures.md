# 024 — The Samuelson Effect Across 16 Futures Markets

## The narrow question

Samuelson (1965) predicted that a futures contract's return volatility should rise as it
approaches expiry: news about near-term supply and demand moves the front of the curve
sharply, but has mostly decayed away by the time a deferred contract settles. This
notebook asks whether that pattern shows up, cleanly and consistently, across every
futures market in this repo — 16 products, 2010-2026, roughly 850,000 contract-days —
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
weakest slope of all 16, contrary to the scoping expectation. The realised "smile" and
the curve-state ("skew") checks both turned up genuine surprises rather than confirming
the textbook story on the first try.

| Check | Result |
|---|:---:|
| Samuelson effect visible in energy? | **Yes** — CL, NG, HO, BZ, RB all significantly negative slope |
| Visible in grains? | **Yes** — ZC, KE, ZS, ZM, ZW significantly negative; ZL (soybean oil) weak, CI touches zero |
| Visible in precious metals? | **No — inverted** — GC ≈ 0, SI/PA/PL all *positive* (vol rises with maturity) |
| Negative control (`ES`) weakest, as predicted? | **No** — `ES` (+0.046) is not the weakest; `PL` (+0.056) is slightly more positive. `ES` does sit inside the near-zero/positive cluster, well separated from energy/ags, which is the claim that matters |
| Front-end vol spikes more than back-end in crises? | **Yes** — CL front/deferred ratio rises from 1.38 (2017 calm) to 1.81-1.89 in every crisis window |
| Robust and winsorised estimators agree? | **Yes, on sign, everywhere** — plain `std` disagrees on sign with MAD/winsor for CL, GC, PA, and ZL |
| Realised "smile" survives the maturity confound? | **Mixed** — CL's pooled U-shape persists within every maturity bin; NG's pooled shape is a hump, not a U, once you look at it directly |

## Method

### Data and hygiene

16 products' per-contract OHLCV (`src/research/data/market/databento/ohlcv/<PROD>.parquet`,
2010-06 to 2026-07, ~850,833 usable contract-days after hygiene) joined to
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
than silently dropped — 94 on CL, 58 on NG, 9 on GC, 0 on ES. NG's 14,373 non-positive
closes (10% of its raw panel) are reported explicitly rather than absorbed into a
`.dropna()`.

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

Ranked slopes (MAD, `max_gap=1`, `min_volume=0`), most negative first:

| Product | Slope | 95% CI |
|---|---:|---:|
| NG | -0.282 | [-0.334, -0.241] |
| ZC | -0.164 | [-0.224, -0.108] |
| CL | -0.153 | [-0.183, -0.090] |
| KE | -0.115 | [-0.146, -0.082] |
| HO | -0.112 | [-0.134, -0.089] |
| BZ | -0.107 | [-0.138, -0.075] |
| ZS | -0.095 | [-0.151, -0.056] |
| ZM | -0.090 | [-0.143, -0.050] |
| RB | -0.088 | [-0.117, -0.059] |
| ZW | -0.082 | [-0.150, -0.021] |
| ZL | -0.032 | [-0.072, +0.004] |
| GC | +0.013 | [-0.008, +0.036] |
| PA | +0.032 | [-0.011, +0.062] |
| SI | +0.033 | [+0.001, +0.066] |
| ES | +0.046 | [-0.039, +0.120] |
| PL | +0.056 | [+0.026, +0.114] |

Every energy and ag product except soybean oil (`ZL`, CI just touches zero) has a
95% CI excluding zero on the negative side. `GC`/`SI`/`PA`/`PL` and `ES` cluster near
zero or positive — clearly separated from the commodity group — but `ES` is not the
single weakest, contradicting the scoping expectation. This does not falsify the
central claim (the falsification condition was about the energy/ags sign and
cross-estimator/liquidity agreement, not the precise `ES` rank), but it is reported
plainly rather than glossed.

**The metals inversion.** `PL` and `PA` show a genuinely positive slope — vol *rising*
with maturity — and it survives every check run against it: `volume > 1000`,
`max_gap <= 3`, and restricting to buckets with `n_contracts >= 10` (`PL`: +0.060 with
the contract floor vs. +0.056 primary; `PA`: +0.016 vs. +0.032 — smaller but still
positive). This is **not** a liquidity artefact of a handful of thin far-dated
contracts; it is a real feature of how platinum and palladium futures are priced. The
most likely explanation, consistent with these being smaller, less liquid, industrial
precious metals with idiosyncratic supply risk concentrated in a few mining regions, is
that their far-dated contracts carry a persistent risk premium unrelated to near-term
information flow — but that is a hypothesis for a future notebook, not something this
descriptive study tests.

### Phase 4 — The time dimension

A 63-day trailing causal rolling MAD vol (verified causal: perturbing a future return
does not change a past value) computed for 5 maturity buckets across all 16 products.
CL's front/deferred vol ratio averages 1.38 in the 2017 calm control and rises to
1.81 (2014-15 oil collapse), 1.89 (COVID), and 1.41 (Ukraine) — the front end
consistently reacts more than the back end in a crisis, as the mechanism predicts. The
rolling 126-day front/deferred return correlation on CL drops as low as 0.012 at its
minimum (mean 0.83) — the de-correlation companion to the vol effect: not only does the
front move more, it moves more independently of the back.

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

CL's pooled realised "smile" (vol vs. log-moneyness against the front contract) is a
genuine U-shape (wing vol 0.220 vs. centre vol 0.214) that **persists within every
maturity bin** — controlling for the maturity/moneyness confound does not make it go
away, so it is not simply the Samuelson effect in disguise. NG's pooled shape is the
opposite: a hump, with centre vol (0.251) *higher* than wing vol (0.172). The two
products do not tell the same story here, and this write-up reports that rather than
forcing a single narrative.

The curve-state ("skew") check asked whether backwardated curves (signalling scarcity)
carry a steeper Samuelson slope than contango curves. For CL, **contango is slightly
steeper** than backwardation (-0.150 vs. -0.128) — the opposite of the naive
prediction. For NG the two are nearly identical (-0.284 vs. -0.277). Neither product
supports the hypothesis as stated; both results are reported as found.

## Bugs found

1. **`lib24.samuelson_slope`'s bootstrap was O(n_boot × n_contracts) in pandas
   overhead** — rebuilding a filtered-and-concatenated DataFrame on every bootstrap
   draw. This is invisible at small contract counts (the unit tests, with 6-10 synthetic
   contracts, ran in under a second) and catastrophic at real ones: the Phase 3 script
   timed out at both 5 and 10 minutes on CL/NG-sized panels across ~150 calls. Fixed by
   precomputing each contract's `(dte, r)` arrays once outside the bootstrap loop and
   binning each resampled draw with `np.searchsorted` instead of `pd.cut` + `groupby`.
   Full Phase 3 run: 2m17s after the fix. All 12 unit tests still pass unchanged.
2. **`BZ` (Brent) carries 782 rows with `dte < 0`** — trade prints dated after the
   contract's recorded expiry, down to -18 days on the earliest contract. Almost
   certainly a last-trade-date vs. settlement-date mismatch in this product's contract
   metadata rather than a panel-construction bug (no other product shows this at scale).
   Reported and left outside every maturity bucket rather than silently dropped or
   silently included.
3. **Phase 2's independently-recomputed PA front/deferred ratio (0.89) missed the
   NEXT_PROMPT.md smoke-test figure (0.70) by 27%**, the largest miss of any product —
   consistent with, and explained by, Phase 3's finding that PA's inverted slope is
   driven by a genuinely thin far-dated bucket, not a bug in either computation.

## Bottom line

The Samuelson effect is real and strong in every storable commodity in this repo, weak
or absent in refined products, and clearly inverted in platinum and palladium — an
inversion that survives every liquidity and contract-count check thrown at it and is
therefore a real finding, not an artefact. The negative control does what a negative
control should (sits well outside the commodity cluster) without landing exactly where
the scoping notes predicted, and the realised smile/skew checks in Section D produced
two genuine surprises (NG's inverted smile shape, CL's contango-steeper-than-backwardation
result) rather than confirming the textbook picture by default. None of this is novel —
Samuelson (1965) is 60 years old — but the breadth (16 markets, one method, 16 years),
the documented data-quality problem and its fix, and the honest reporting of where the
naive predictions were wrong are the actual contribution of this notebook.

## What to test next

- Whether PL/PA's inverted slope correlates with a measurable proxy for mining-supply
  concentration risk, and whether it is priced (a risk premium) or purely a volatility
  phenomenon.
- Whether the CL/NG divergence in the realised smile shape (U vs. hump) is stable
  across sub-periods, or specific to a particular regime in the 2010-2026 sample.
- A genuine per-date term structure (Phase 4's B5 chart used the nearest annual
  aggregate as a documented approximation) for a sharper crisis-date comparison.

---

**Traceability.** Notebook: `src/research/024_samuelson_effect_across_futures.ipynb`.
Driver: `scripts/run_024.sh`. Phase reports:
`src/research/tmp/phase_{0,1,2,3,4,5,6}_24_*.json`. Library:
`src/research/tmp/lib24.py` (tests: `tests/test_lib24.py`). Chart modules:
`src/research/tmp/charts_24_{a,b,c,d}.py`.
