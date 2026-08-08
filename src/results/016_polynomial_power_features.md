# Polynomial Power Features on BTC's Lag-1 Return

## What

This notebook extends the single-feature linear model from notebook 001 by replacing it with three-feature polynomial power combinations of the same lag-1 return: `x^p1 + x^p2 + x^p3`. Eight different power combinations were tried (cubic/quintic/high-odd/even-only/mixed-sign/negative-power), each walk-forward validated on BTC 1h bars (2021-07-01 to 2025-07-01, 42 folds, 180d train / 30d test rolling). All results net of realistic fees.

## Why

Notebook 001's own single-split result was flagged as untrustworthy (a disguised constant bet, no walk-forward validation). This notebook first reproduces that same single-lag linear model properly — walk-forward validated on 4 years of BTC 1h bars instead of one split — as its own baseline, which comes out net negative (-3.13 sharpe after fees). It then asks: does adding polynomial flexibility (higher powers, negative powers, pure-even combinations) reveal a real signal the plain line was missing, or just multiply the ways to find noise? Every decision (which powers to try, the `safe_power` EPS floor for negative powers, fee assumptions) was set up front before looking at results.

## How

Built on the walk-forward infrastructure from notebook 002: 42 rolling train/test folds, per-fold weight/degenerate-bet inspection, buy-and-hold/always-long/short/flat/random baselines, deflated Sharpe correction for 8 trials, and bootstrap 95% CI on excess return (best combo net minus buy-and-hold, per bar). Negative-power combos were guarded against numerical blow-up with a `safe_power` function that floors `|x|` at `EPS = 1e-4` before raising to a negative power, and with NaN-to-null conversion before selecting a best combo and computing sharpe/CI.

## Results

The best of 8 power combos — `high_odd` powers (7, 5, 3) — achieved sharpe -0.971 net of fees: better than the plain lag-1 linear baseline's -3.134 and better than all other polynomial attempts (second place: even_only at -1.412), but still deeply negative and not validated. Position mix was reasonable (long 42.59%, short 51.98%, flat 5.43%, zero degenerate folds), ruling out a disguised constant directional bet — the specific failure mode this task asked to check for. Deflated Sharpe probability of a true edge: 0.06% (best-of-8 correction on an already near-zero per-period sharpe of -0.01037). Bootstrap 95% CI on excess return over buy-and-hold: [-0.00018, 0.00001], includes zero — can't reject no edge.

What actually separates the best combo from the rest isn't signal quality: gross (pre-fee) sharpe was uncorrelated with net sharpe across the sweep, and the combo with the best *gross* sharpe of all nine configs tried (`mixed_one_neg` at +1.252) had one of the worst net results (-3.040). The real driver was turnover — see below.

Negative-power combinations were numerically usable (did not blow up or produce widespread NaN). All 8 combos produced finite Sharpe values despite the known risk of x→0 division. The `safe_power` EPS floor and NaN-to-null safeguards held without triggering edge-case failures.

## Power combinations tried

| combo | powers | sharpe_net | sharpe_gross | degenerate_folds |
|---|---|---|---|---|
| high_odd | (7, 5, 3) | -0.971 | +0.734 | 0 / 42 |
| even_only | (6, 4, 2) | -1.412 | -0.605 | 1 / 42 |
| quintic_quad_linear | (5, 2, 1) | -3.033 | -0.397 | 2 / 42 |
| mixed_one_neg | (3, 1, -1) | -3.040 | +1.252 | 2 / 42 |
| all_negative | (-1, -2, -3) | -3.048 | +0.284 | 0 / 42 |
| cubic_quad_linear | (3, 2, 1) | -3.086 | +0.169 | 2 / 42 |
| quintic_cubic_linear | (5, 3, 1) | -3.245 | +0.599 | 4 / 42 |
| mixed_neg_odd | (1, -1, -3) | -4.320 | -0.346 | 2 / 42 |

Degenerate-fold counts stayed low across the board (0-4 out of 42, i.e. 0-9.5%): `high_odd` and `all_negative` had zero, `quintic_cubic_linear` the most at 4. There's no correlation between degenerate-fold count and net sharpe rank either — the model being genuinely feature-responsive in most folds didn't translate into a usable net edge for any combo.

## Baseline comparisons (BTC 2021-07-01 to 2025-07-01)

| baseline | sharpe | compound_return |
|---|---|---|
| Buy & hold | +0.522 | +213.04% |
| Always long | +0.522 | +213.04% |
| Always short | -0.522 | -68.05% |
| Always flat | +0.000 | — |
| Random (200 seeds, mean) | -0.013 | — |
| Lag-1 linear (baseline) | -3.134 | -99.68% |
| Best power combo (high_odd) | -0.971 | -82.89% |

Both the baseline linear model and the best power combo lost more than 80% net of fees. Buy-and-hold's +213% and sharpe 0.522 is the hurdle neither comes close to clearing. Even `high_odd`'s -0.971 sharpe, the best net result in the notebook, is still deep in the negative regime — well outside the random baseline's 90% range of [-0.812, +0.841].

## Turnover and the overtrading trap

The gap between gross and net numbers is the most important thing in this notebook, and it's driven almost entirely by trade frequency, not signal quality:

| config | sharpe_gross | sharpe_net | no_trades / 30,240 bars | frac_time_in_market |
|---|---|---|---|---|
| Baseline (lag-1 linear) | +0.234 | -3.134 | 29,486 | 97.51% |
| high_odd (best net) | +0.734 | -0.971 | 28,597 | 94.57% |
| mixed_one_neg (best gross) | +1.252 | -3.040 | 30,197 | 99.86% |
| mixed_neg_odd (worst net) | -0.346 | -4.320 | 30,173 | 99.78% |

`mixed_one_neg` had the best gross sharpe of any of the 9 configs tried (+1.252, beating even buy-and-hold's +0.522) — and one of the worst net results, because it was in the market and flipping position on 99.86% of bars. `high_odd`, the eventual net winner, didn't win on signal quality (its own gross sharpe of +0.734 was worse than three other combos); it won because its no-trade zone actually filtered something, leaving it in the market 94.57% of the time instead of ~99.7-99.9% like every other config. Averaged across the baseline's stitched OOS window, that difference in trade frequency is worth roughly 4,465 round-trip-equivalent position changes per year (from `cost_summary`'s `turnover_per_year`) against a per-trade cost of `taker_fee + slippage = 0.0004` — costs that compound multiplicatively bar over bar across 3.45 years of stitched out-of-sample trading into the near-total losses in the table above.

The practical lesson: `EDGE_THRESHOLD = 2x taker_fee`, the no-trade zone notebooks 1 and 2 established, was calibrated for a single raw-lag feature. It does not automatically transfer to power-transformed features, whose predicted-return scale (and therefore how often `|y_pred|` crosses the threshold) changes with the transform. A model can look better before fees and still lose more after them if that flexibility mainly translates into more frequent, smaller-edge trades.

## Per-fold analysis: best combo (high_odd, powers 7/5/3)

All 42 folds marked "responsive", indicating no degenerate (constant-direction) folds:

- Folds with strong sharpe (|sharpe| > 4): 0, 3, 4, 12, 15, 19, 22, 26, 31, 32, 34, 36, 38, 40 — mix of positive and negative, no directional consistency across folds.
- Folds with sharpe near zero (|sharpe| < 1): 7, 8, 9, 10, 14, 16, 20, 27, 29, 30 — model had no edge in these windows.
- Per-fold sharpe range: +7.656 (fold 22) to -7.458 (fold 34).

Each fold showed a responsiveness verdict (`responsive (|bias|=..., sign flips)`) confirming the linear model's weight was genuinely driven by the features, not overridden by the bias term. Bias magnitudes stayed below the max feature contribution per fold (the template check from notebook 2), confirming the model wasn't a disguised constant bet in any individual fold.

Position mix across all 42 folds: 42.59% long, 51.98% short, 5.43% flat. The model did trade both directions, ruling out "always short" or "always long" as the failure mode. More short than long, but not overwhelmingly (51.98% vs 42.59% — a preference, not a constant bet).

## Safe power design and numerical stability

`EPS = 1e-4` was set before looking at any results, chosen to lie below the typical smallest non-zero 1h BTC log-return move. For negative powers, `safe_power(x, p)` floors `|x|` at EPS before exponentiation, preserving sign (so odd negative powers stay sign-preserving, even ones stay positive). 

All 8 combos, including the most numerically aggressive (`all_negative: (-1, -2, -3)` and `mixed_neg_odd: (1, -1, -3)`), produced finite sharpe values without numerical blow-up. No widespread NaN production; NaN-to-null conversion in the feature engineering was applied as a safeguard but did not appear to be needed. The `safe_power` floor held up under real data.

## Deflated Sharpe and statistical significance

**Best combo per-period sharpe (net):** -0.01037  
**n_trials:** 8 (power combos)  
**n_obs:** 30,240 (bars in stitched OOS trades)  
**Skew:** +0.062 (near-symmetric)  
**Kurtosis:** 17.310 (fat-tailed, as expected for crypto)

**Deflated Sharpe probability: P(true edge > 0 | best of 8 trials): 0.06%**

Even with best-of-8 correction, the probability that there is a true positive edge is indistinguishable from what noise's best-of-8 looks like. The per-period sharpe is already close to zero (-0.01037), so the multiplication by the deflation factor compresses an already-weak signal further.

## Bootstrap confidence interval on excess return

**95% CI (best combo net return - buy&hold return per bar):** [-0.00018, 0.00001]  
**Includes zero:** Yes → can't reject no real edge.

The interval is tight (centered near zero, ±0.0001 log return per bar) and straddles zero, meaning the best combo's per-bar excess return over buy-and-hold is statistically indistinguishable from zero across the 30,240 bars of out-of-sample data.

## Follow-up: recalibrating the no-trade threshold and minimum hold period

**What was tested:** The original `EDGE_THRESHOLD = 0.0006` (2 × taker_fee) used in sections 1–8 does not include slippage; the true round-trip cost of a full position flip is `2 × (taker_fee + slippage) = 0.0008`. Section 9 tested whether fixing that, and adding a minimum-hold period between position changes, could turn any net Sharpe positive. For three configs (baseline_lag1, high_odd, mixed_one_neg), each model was trained once with an explicit per-config seed (123, 124, 125) to capture raw per-bar predictions, then a 7 × 6 grid (threshold multiples: 0.5x, 1.0x, 2.0x, 3.0x, 5.0x, 10.0x, 20.0x; min-hold periods: 0, 3, 6, 12, 24, 48 bars) was swept purely by re-deriving position/cost from fixed predictions—no retraining. This isolates the effect of the trading rule from the model fit.

**OOS predictions captured:**
- baseline_lag1: 30,240 OOS predictions (seed=123)
- high_odd: 30,240 OOS predictions (seed=124)
- mixed_one_neg: 30,240 OOS predictions (seed=125)

**Did any grid point achieve positive net Sharpe?** Yes—one. **Best follow-up config:**
- Config: baseline_lag1
- Threshold: 0.004 (5.0× round-trip cost, vs. original 0.75×)
- Min-hold: 48 bars
- sharpe_net: +0.238
- compound_return_net: +53.67%
- frac_time_in_market: 92.92%
- no_position_changes: 454

**Reproducibility cross-check** (original threshold 0.0006, min_hold=0):

| config | sharpe_net | compound_return_net | frac_time_in_market |
|---|---|---|---|
| baseline_lag1 | -3.134 | -99.68% | 97.51% |
| high_odd | -1.429 | -92.70% | 97.38% |
| mixed_one_neg | -4.165 | -99.95% | 99.70% |

baseline_lag1 matches the original section 3 result exactly (-3.134), confirming the seeding reproduces correctly. high_odd and mixed_one_neg differ from the original sweep table (which used shared RNG across all 9 configs) due to independent seeding here, but both are in the same ballpark and sign, validating the approach.

**Deflated Sharpe and bootstrap CI for the best follow-up config:**
- n_trials: 126 (3 configs × 42 grid points each)
- n_obs: 30,240
- Skew: +0.255
- Kurtosis: 17.492
- **Deflated Sharpe P(true edge > 0 | best of 126 trials): 1.51%**
- **Bootstrap 95% CI on per-bar excess return (best follow-up config net - buy&hold): [-0.00010, 0.00008]**
- Includes zero → can't reject no edge over buy-and-hold.

**Per-config impact of recalibration:**

| config | original sharpe_net | best sharpe_net | threshold_mult | min_hold |
|---|---|---|---|---|
| baseline_lag1 | -3.134 | +0.238 | 5.0× | 48 |
| high_odd | -1.429 | -0.012 | 1.0× | 48 |
| mixed_one_neg | -4.165 | -0.096 | 5.0× | 48 |

All three configs' best results occurred at min_hold=48 bars, but min-hold and threshold are not interchangeable levers — isolating each on `baseline_lag1` (verified independently, not from the notebook's printed grid) shows they don't contribute equally:

| threshold_mult | min_hold=0 | min_hold=48 |
|---|---|---|
| 1.0x | sharpe_net -3.140 | sharpe_net -0.539 |
| 2.0x | sharpe_net -3.113 | sharpe_net -0.584 |
| 3.0x | sharpe_net -3.132 | sharpe_net **+0.117** |
| 5.0x | sharpe_net -3.161 | sharpe_net **+0.238** |

At `min_hold=0`, raising the threshold from 1x to 5x the true round-trip cost does essentially nothing — sharpe_net stays pinned around -3.1 to -3.2 regardless. The minimum-hold period is what does almost all of the work, taking baseline_lag1 from catastrophic (-3.13) to merely losing (-0.49 to -0.58) at any threshold from 0.5x-2x. But minimum-hold alone isn't sufficient to cross into positive territory — that only happens once the threshold is also pushed to 3x-5x the true cost *on top of* the 48-bar hold. The two levers compound rather than substitute for each other.

**Verdict on follow-up findings:** Recalibrating the no-trade threshold to the true round-trip cost and adding a minimum-hold period yielded a positive result: baseline_lag1 at +0.238 net Sharpe, +53.67% compound return, tradeable on 92.92% of bars. This reverses the original configuration's -3.134 sharpe — a 3.37-unit improvement. However, the improvement is fragile in two ways. Statistically, the deflated Sharpe probability of 1.51% (best of 126 trials) is weak validation, and the bootstrap CI on excess return over buy-and-hold still includes zero — not distinguishable from noise. Economically, +53.67% compound over 3.45 years is far below buy-and-hold's +213.04% over the same window (see the baseline-comparisons table above) — a positive Sharpe here is not the same as a strategy worth trading over just holding BTC. The result is better than the original — and better than every power combo in sections 1-8 — but not yet defensible as a tradeable edge, and not better than doing nothing. The core lesson stands: the key to reducing (not eliminating) the damage in this domain is controlling turnover, and a minimum-hold rule controls it far more than threshold recalibration alone does.

## Bottom line

Polynomial power features on a single lag-1 return do not reveal a hidden signal in the original sweep (sections 1–8). The best combo (`high_odd`, powers 7/5/3) beats the lag-1 linear baseline's sharpe of -3.134 and improves to -0.971, but still loses >80% after fees and sits outside the random baseline's 90% range. Deflated Sharpe (0.06%) and bootstrap CI (includes zero) both reject the hypothesis of a true edge.

The follow-up (section 9) found that the original `EDGE_THRESHOLD = 0.0006` was calibrated below the true round-trip cost, and that a minimum-hold period does most of the work of fixing the overtrading (alone, it takes baseline_lag1 from -3.13 to roughly -0.5; threshold recalibration alone, without a minimum hold, does essentially nothing). The best follow-up result — baseline_lag1 at 5.0× true cost with a 48-bar minimum hold — achieved +0.238 net Sharpe and +53.67% compound return, a 3.37-unit improvement over the original baseline. That reversal is real, but two things keep it from being a finding: statistically, deflated Sharpe (1.51%, best of 126 trials) and a bootstrap CI on excess return that includes zero both say it's not distinguishable from noise; economically, +53.67% compound is still far below buy-and-hold's +213.04% over the same window, so even taken at face value it's a worse use of capital than doing nothing.

The more useful finding is *why* every config lost so much in the original sweep: turnover, not signal quality. Gross sharpe was uncorrelated with net sharpe across the 9 power combos — the best gross performer (`mixed_one_neg`, +1.252) was one of the worst net performers, because it traded on 99.86% of bars. `high_odd` won on net sharpe not because its fit was better (its own gross sharpe of +0.734 trailed three other combos) but because it was the only config whose no-trade zone meaningfully filtered trades, at 94.57% time-in-market versus ~99.7-99.9% everywhere else. The `EDGE_THRESHOLD = 2x taker_fee` no-trade zone that worked for notebooks 1 and 2's raw single-lag feature does not automatically transfer to power-transformed features: it needs recalibrating to the transformed prediction's own scale, or a config can look strictly better before fees and still lose more after them. The follow-up confirms this: threshold recalibration alone wasn't enough; minimum-hold periods were needed to keep turn-over low enough for net results to be close to break-even.

Negative-power combos were numerically stable; the `safe_power` design worked as intended.

The walk-forward machinery reconfirms the core finding from notebook 2: BTC return prediction at 1h frequency with simple features, under realistic fees and robust validation, doesn't yield a tradeable edge. Neither more parameters (polynomial transform), nor exotic powers (negative exponents), nor sign-agnostic combinations (even-only), nor recalibrated no-trade thresholds and minimum-hold rules change that conclusion. Turnover control is necessary but not sufficient to turn an unprofitable model profitable; the underlying signal must also exist, and it does not.
