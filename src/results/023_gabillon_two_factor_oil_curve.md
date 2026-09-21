# 023 — The Gabillon (1991) Two-Factor Model on 16 Years of WTI

## The narrow question

Gabillon (1991) prices an entire crude futures curve from just two state variables on a
given day: the extrapolated spot `S` and a long-term price anchor `L`. Given those two
numbers and three constant parameters (`beta`, and a volatility combination `nu`), the
model (eq. 26) reproduces the whole strip. The paper tested this on 423 trading days of
1990-91 WTI data and reported in-sample RMSE of roughly 0.1-0.5 $/bbl in calm periods,
rising to ~1.5 $/bbl in the crisis months it studied (April-May 1990, January-February
1991).

This notebook asks whether that claim holds on this repo's 16-year, per-contract WTI
panel (2010-06-06 to 2026-07-28) — and, specifically, whether the model prices maturities
it was **not** fitted on. That held-out-maturity test is the model's entire stated
purpose (extrapolating past the liquid front of the curve) and the paper never ran it.

This is a pricing-accuracy study, not a returns backtest. No Sharpe ratio appears
anywhere in this notebook.

## The answer

**The pre-registered gate fails, on both the train sample (Phase 6) and the frozen
holdout (Phase 9).** Gabillon's model (26) prices the curve better than a flat-forward
benchmark by a wide margin, and gets the cross-sectional shape of the curve
approximately right — but it loses the **paired**, day-by-day comparison against the
best of four purely statistical benchmarks (cubic spline, Nelson-Siegel, 2-factor PCA,
previous-day-carried-forward) on both samples, decisively on the holdout.

| Check | Result |
|---|:---:|
| Beats flat forward (train, held-out maturity)? | **Yes**, by a wide margin (median 1.82 vs. 7.75 $/bbl) |
| Beats the best of 4 statistical benchmarks, paired, train sample (Phase 6)? | **No** — CI [+1.84, +2.62] $/bbl, wrong side of zero |
| Beats the best of 4 statistical benchmarks, paired, holdout (Phase 9)? | **No, badly** — CI [+22.0, +28.0] $/bbl |
| Volatility identity (eq. 23) independently confirms the fit? | **No** — predicted vol collapses with maturity; realised vol does not |
| Seasonality limitation (Gabillon's own, §8) reproduces? | **Yes, cleanly** — NG worst, BZ best, in the predicted order |

## Method

### Data and hygiene

`src/research/data/market/databento/ohlcv/CL.parquet` (109,758 rows, 2010-06-06 to
2026-07-28) joined to `contracts.parquet` on `contract_id` for `tau = (expiry -
date).days / 365.25`. Three liquidity variants tested throughout: `all`, `volume > 0`
(identical on this panel — every retained row already has positive volume), and
`volume > 100`.

**The real negative-price event.** `CL202005` settled at **-2.67 on 2020-04-20** (volume
102,083 — the single highest-volume row in the whole non-positive-price set). This is
genuine market history that the model is structurally incapable of representing (`ln`
of a negative price is undefined). It is excluded from every fit and recorded here, not
silently dropped.

**Deviation from the pre-registered description: more junk than expected.** NEXT_PROMPT.md
named one junk stub contract (`CL203212`, Dec 2032) with "at least 8" non-positive rows.
The actual panel has **390** non-positive-price rows across **seven** far-dated stub
tickers (`CL202707`, `CL202903`, `CL202904`, `CL203012`, `CL203212`, `CL203305`,
`CL203311`), all illiquid scattered sign-flips on far-dated contracts, none of which is
the 2020-04-20 event. All are filtered unconditionally (`ln` is undefined for every one),
documented and reported rather than silently absorbed into the liquidity-variant choice.

**Minimum curve width.** Days with fewer than 8 distinct contracts are dropped. Retained:
4,891 days on `all`/`vol_gt_0`, 4,197 days on `vol_gt_100` (mean 22.2 and 16.6
contracts/day respectively).

### Spot proxy and L estimators (Phase 2)

`S` from the two nearest contracts per the prescribed linear extrapolation. Three `L`
estimators, per NEXT_PROMPT.md: (a) joint least squares (paper §5.1, both monthly-fixed
and daily-free), (b) long-end extrapolation (paper §5.2), (c) a plain linear Kalman
filter on `(ln S, ln L)`.

**The paper's own tension reproduces, more sharply than the paper itself reported.** On
this 16-year sample:

| Estimator | `sigma_S` (ann.) | `sigma_L` (ann.) | `rho` | Notes |
|---|---:|---:|---:|---|
| Paper method (a) | 46.4% | 17.6% | 0.21 | reference |
| Paper method (b), raw / filtered | — | 58.1% / 33.7% | 0.42 | reference |
| (a) joint LS, monthly-fixed | 44.7% | 98.3% | -0.03 | 3,472 train days |
| (a) joint LS, daily-free | 44.7% | 773% | 0.02 | numerically unstable |
| (b) long-end extrapolation | 44.7% | 2,441% | 0.00 | valid only 21.7% of days |
| (c) Kalman filter | 23.1% | 3.2% | 0.11 | **the only stable estimator** |

Both (a)-daily and (b) explode into economically absurd `sigma_L` even after a
pragmatic bound (`ln(L/S)` clipped to ±log 5) was added to `fit_L_joint_ls` to stop raw
float overflow — a bug caught and fixed during Phase 2 (see Bugs Found). Per-day
identification of `L` from a single day's curve is close to unidentified whenever
`beta` is small (the `(1-B(tau))` blend weight vanishes for every observed maturity),
and a 16-year sample hits that regime often enough to dominate the noise budget. The
long-end estimator's second-derivative proxy (paper's own "arbitrary" characterisation)
is satisfiable on only 21.7% of days; the rest carry forward the last valid value.

**Gabillon considered his own method (b) "unsatisfactory... too volatile, too
correlated" even though it fit better than (a).** That tension reproduces exactly, and
worse, on this longer sample: the Kalman filter — the modern fix the paper didn't
have — is the only estimator whose `sigma_L`/`rho` sit anywhere near the paper's own
figures. It was carried forward as the primary configuration for every subsequent phase.

### Model fit and benchmark race (Phases 3-4)

Per-day `beta` fit by 1-D least squares given the estimator's `(S, L)`; `nu` from a
causal, trailing 63-day rolling realised-vol estimate (shifted one day to avoid
lookahead). Across the three liquidity variants, the Kalman config gives median train
RMSE **0.94-0.95 $/bbl** — in the right order of magnitude versus the paper's own
0.1-1.5 $/bbl range. `joint_ls_daily` (median RMSE up to ~1,900 $/bbl) and `longend`
(~2-3 $/bbl, one-fifth coverage) were dropped from Phases 4 onward as numerically unsound
survivors of the Phase-3 diagnostic — both remain charged in the 24-config
multiple-testing count.

In-sample, full-curve RMSE (Phase 4, informational only): cubic spline and
Nelson-Siegel are near-interpolants through the very points scored here (spline RMSE
~1e-14 by construction) — not a meaningful comparison. The decisive test is held-out
maturity.

### Model (28), the short-term shock (Phase 5)

Adds `theta`, `eta` for a decaying convenience-yield shock. Fit within the pre-registered
crisis windows (2014-15 collapse, 2020 COVID, 2022) and a 150-day calm-period control.
2022 falls entirely inside the holdout and correctly returns zero fitted days here
(deferred to Phase 9's discipline). Model (28) improves in-sample RMSE on 83-99% of days
in every window it could be fit — **including the calm control (94%)**. This is expected
and not informative on its own: two extra free parameters fit within the same day's data
will almost always reduce in-sample error. No held-out claim is made for model (28)
anywhere in this notebook; it plays no role in the primary gate.

### Phase 6 — the primary gate, train sample

Fit on `tau <= 2y`, predict `tau` in `(2y, 9y]`, score against actual settlement.
3,146 days scored (Kalman config, `all` variant).

| | Gabillon (26) | Flat forward | Cubic spline | Nelson-Siegel | PCA (2-factor) |
|---|---:|---:|---:|---:|---:|
| Median RMSE ($/bbl) | **1.82** | 7.75 | 77.6 (extrapolation blow-up) | 2.66 | 2.59 |

Median RMSE alone makes Gabillon look competitive against NS/PCA. **It is not,
paired.** Each benchmark's median above is its own aggregate across all days; the fair
comparison is Gabillon's RMSE against the *best* of the four benchmarks *on that same
day*:

- Paired gap (Gabillon − best-of-4-benchmarks), point estimate: **+2.21 $/bbl**
- 95% bootstrap CI: **[+1.84, +2.62]** — entirely on the side of Gabillon losing
- Gabillon wins the daily head-to-head on **38.7%** of days

By year, Gabillon's median RMSE is fairly stable (1.1-2.9 $/bbl in 10 of 12 years,
spiking to 8.9 in 2012). By curve state: median RMSE 1.21 $/bbl in backwardation (871
days) vs. 2.12 $/bbl in contango (2,275 days) — the model handles backwardated curves
somewhat better, consistent with the paper's own note that model (26) alone struggles
precisely in the steep-contango-against-backwardated-long-end regime that motivates
model (28).

**Leave-one-contract-out**, full strip, 250 sampled train days (5,689 observations):
RMSE 4.70 $/bbl, median absolute error 0.73 $/bbl — noticeably worse than the
tau<=2y-fit/predict-far protocol above, since a randomly dropped single contract can sit
anywhere on the curve including thin, noisy far-dated points.

### Phase 7 — volatility identity (eq. 23), an independent check

Using the fitted `(sigma_S, sigma_L, rho, beta)` from the Kalman config (train medians:
`beta`=0.407, `sigma_S`=23.1%, `sigma_L`=3.2%, `rho`=0.107), the price-implied
`sigma_F(tau)` was compared against realised volatility by maturity bucket — a check the
price fit never sees.

| Maturity bucket | Realised vol (ann.) | Model-implied vol | Gap |
|---|---:|---:|---:|
| 0-1y | 29.7% | 19.0% | 10.7pp |
| 1-2y | 29.3% | 12.8% | 16.5pp |
| 2-4y | 36.0% | 7.4% | 28.6pp |
| 4-9y | 27.4% | 3.5% | 23.9pp |

**Clean falsification.** Realised volatility is nearly flat across the whole curve
(27-36% in every bucket); the model's implied term structure of volatility collapses
sharply with maturity (19% down to 3.5%). The model gets the day-to-day cross-sectional
*shape* of the curve approximately right (Phase 6) while getting the *volatility term
structure* badly wrong — exactly the "fits prices but misses the vol curve, so it's
fitting noise" pattern `docs/08-research-methodology.md` names as a reason to distrust a
good-looking price fit.

### Phase 8 — seasonality, cross-product

Same held-out-maturity protocol applied to BZ (Brent, WTI-like control), HO (heating
oil, moderate seasonality), RB (RBOB gasoline, moderate), NG (natural gas, strong
seasonality). RMSE normalised to each product's own price level (raw units differ:
$/bbl for BZ, $/gallon for HO/RB, $/mmbtu for NG) as **relative RMSE = median
RMSE / median price**, since $/bbl and $/gallon numbers are not directly comparable.

| Product | Seasonality | Days scored | Relative RMSE |
|---|---|---:|---:|
| BZ | none (control) | 1,362 | **1.55%** |
| HO | moderate | 1,079 | 1.64% |
| RB | moderate | 357 | 5.11% |
| NG | strong | 2,867 | **6.75%** |

**Gabillon's own stated limitation reproduces cleanly and in the predicted order.**
"Petroleum products with strong seasonal patterns cannot be efficiently described by
this couple of state variables" (Gabillon's own words) — BZ fits best, the two
moderately-seasonal products sit in the middle, NG is worst by more than 4x. A seasonal
convenience-yield term was **not implemented** — a scope cut given the time/compute
budget for this notebook, flagged below rather than silently dropped.

### Phase 9 — the holdout, spent once

2022-01-01 onward, touched here for the first and only time, on the certified
configuration only (`all` variant, Kalman `L`, model 26). 1,364 days scored.

| | Gabillon (26) | Flat forward | Cubic spline | Nelson-Siegel | PCA |
|---|---:|---:|---:|---:|---:|
| Median RMSE, overall ($/bbl) | 11.35 | 19.11 | 58.18 | **2.88** | 25.11 |

Gabillon still beats flat forward, but not by nearly the margin it managed on train
(11.35 vs. 19.11, a much smaller gap than 1.82 vs. 7.75). **Nelson-Siegel alone beats
Gabillon by 4-9x in every holdout year except a NS-specific numerical blow-up in early
2026** (median NS RMSE 543 that quarter — a curve-fit instability in NS itself, not a
Gabillon result, but real and worth flagging).

- Paired gap (Gabillon − best-of-4-benchmarks): point estimate **+24.9 $/bbl**
- 95% bootstrap CI: **[+22.0, +28.0]** — an order of magnitude worse than the train-sample gap
- Gabillon wins the daily head-to-head on **25.1%** of holdout days (down from 38.7% on train)
- By year: 2022 median RMSE 2.67 (best year, still loses to NS's 2.48), degrading to
  **60.6 in 2025** — the model's worst holdout year by far.

**`PRE_REGISTERED_GATE_PASS: false`.** The pre-registered pass rule (median RMSE beats
the best benchmark by >=10% on a paired comparison, CI excluding zero in Gabillon's
favour) fails outright: the point estimate and full CI sit on the wrong side of zero.

## Bugs found

1. **`fit_L_joint_ls` float overflow via near-unidentified `L`.** At small `beta`, the
   `(1-B(tau))` blend weight vanishes for every observed maturity, making the closed-form
   `ln L` estimate numerically unstable — observed values up to **1e62** before a fix. A
   pragmatic bound (`ln(L/S)` clipped to ±log 5, i.e. `L` within a factor of 5 of `S`,
   an economically generous band for a long-term oil price anchor) was added and is
   documented in `lib23.py`. Caught by `run_phase_2_23_SL.py`'s own `RuntimeWarning:
   overflow encountered in scalar divide` output before any downstream phase consumed
   the corrupted values.
2. **`fit_L_longend`'s second-derivative proxy hitting `fprime == 0` and
   `a2 * t_far1 > 50`.** Both produce `inf`/`nan` propagation into `exp()` without the
   original code's guard catching them (`fpp >= 0` alone was insufficient). Fixed with
   an explicit `fprime == 0` check and an exponent cap; caught the same way, via
   `RuntimeWarning` on the first full run.
3. **No bug affected the certified (Kalman) configuration's reported numbers** — both
   fixes above were in the two configurations Phase 3 independently identified as
   numerically unsound and dropped from further use.

## Bottom line

**Gabillon's two-factor curve model captures real information — it prices the curve far
better than a flat-forward guess — but it does not clear the bar against cheap,
purely-statistical curve interpolators (Nelson-Siegel, 2-factor PCA) on the test that
matters: held-out maturity, scored paired against the best alternative each day.** That
result held on the pre-holdout train sample (Phase 6) and held far more decisively on
the frozen 2022-2026 holdout (Phase 9), where Nelson-Siegel alone beat Gabillon by
4-9x in every year but one. The volatility identity (Phase 7) independently confirms the
same verdict from a different angle: the model gets the curve's cross-sectional shape
roughly right while getting its volatility term structure badly wrong, the textbook
signature of a price fit that is partly fitting noise rather than structure.

The one place the paper's own qualitative claims **did** reproduce cleanly: its
complaint about noisy `L` estimation (worse here than in the original paper, and fixed
only by reaching for a tool — the Kalman filter — the paper didn't have), and its own
stated limitation about seasonal products (NG fits 4x worse, relative to price level,
than the WTI-like control).

This is a null result on the primary economic question (does the model out-predict
cheaper alternatives at held-out maturities) and a positive replication of two of the
paper's own secondary claims. Reported exactly as it came out, per this repo's standing
discipline (`docs/08-research-methodology.md`).

## What to test next

- **The seasonal convenience-yield extension (Phase 8's natural next step) was not
  implemented.** Given how cleanly the plain model degrades on NG, a seasonal term is
  the most promising concrete lead this notebook surfaces — but it needs its own
  pre-registration, not a same-notebook tack-on after seeing the Phase 8 numbers.
- **The Kalman `L` estimator's own hyperparameters (`q_S`, `q_L`, `r_obs`) were set once,
  by inspection, not tuned or cross-validated.** A proper likelihood-based calibration
  might tighten the fit further, though it would need to be done on train data only and
  charged as additional trials.
- **Model (28) was never given a fair, held-out test.** Phase 5 only shows the
  unsurprising in-sample improvement from two extra free parameters. A version of the
  Phase 6/9 protocol scored on model (28) instead of (26) is the natural follow-up, with
  its own pre-registration (this notebook's holdout is now spent and cannot be reused).
- **The `joint_ls_daily` and `longend` estimators were abandoned after Phase 3's
  diagnostic, not repaired.** A regularised or robust variant of either (e.g. a
  bounded-influence M-estimator, or shrinkage toward the Kalman-filtered `L`) might
  recover the paper's own reported stability without the Kalman filter's own
  approximations (a single global `beta` used for its observation matrix).
- **The Nelson-Siegel benchmark's 2026 blow-up (median RMSE 543 in one quarter) was
  observed but not diagnosed.** It is very likely a `lambda` mis-specification at an
  unusual curve shape rather than a real result — worth a five-minute look before this
  benchmark is trusted again in a future notebook.

*Notebook: `src/research/023_gabillon_two_factor_oil_curve.ipynb`. Driver:
`scripts/run_023.sh`. All numbers above are traceable to JSON artefacts in
`src/research/tmp/phase_{0..9}_23_*.json`. The holdout was touched exactly once, in
Phase 9, and every reported number above from that phase reflects that single run.*
