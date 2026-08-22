# 016 — Polynomial Power Features on Bitcoin's Previous Return

## The question

Notebook 001 fitted a straight line through one feature — the previous bar's return — and its
apparently profitable result turned out to be a disguised constant bet on a single train/test split.

This notebook asks whether the problem was the *shape* of the model. Instead of one raw feature, it
uses three powers of the same feature: x^p₁ + x^p₂ + x^p₃. Eight power combinations are tried,
spanning cubic, quintic, high-odd, even-only, mixed-sign and negative exponents.

Does polynomial flexibility reveal a real signal the plain line was missing, or just multiply the
ways to find noise?

Everything is walk-forward validated on Bitcoin hourly bars from 2021-07-01 to 2025-07-01 — 42
rolling folds, 180-day training, 30-day testing — and all results are net of realistic fees. Every
decision (which powers, the numerical safeguard for negative powers, the fee assumptions) was fixed
before looking at any result.

The same single-feature linear model is first re-run *properly* as the baseline, walk-forward
validated over four years rather than one split. **It comes out at −3.134 Sharpe net of fees.**

## The answer

**No hidden signal.** The best of eight combinations reaches −0.971 — better than the baseline and
better than every other polynomial attempt, but still deeply negative and not validated.

**But the more interesting finding is *why* everything lost so much: turnover, not signal quality.**

## The eight combinations

| Combination | Powers | Net Sharpe | Gross Sharpe | Degenerate folds |
|---|---|---:|---:|---:|
| **High odd** | (7, 5, 3) | **−0.971** | +0.734 | 0 / 42 |
| Even only | (6, 4, 2) | −1.412 | −0.605 | 1 / 42 |
| Quintic–quadratic–linear | (5, 2, 1) | −3.033 | −0.397 | 2 / 42 |
| Mixed, one negative | (3, 1, −1) | −3.040 | **+1.252** | 2 / 42 |
| All negative | (−1, −2, −3) | −3.048 | +0.284 | 0 / 42 |
| Cubic–quadratic–linear | (3, 2, 1) | −3.086 | +0.169 | 2 / 42 |
| Quintic–cubic–linear | (5, 3, 1) | −3.245 | +0.599 | 4 / 42 |
| Mixed negative odd | (1, −1, −3) | −4.320 | −0.346 | 2 / 42 |

Degenerate folds — where the bias term dominates and the position stops depending on the features —
stayed low everywhere (0 to 4 out of 42). There's no correlation between degeneracy count and net
Sharpe rank either. The models were genuinely feature-responsive in most folds, and it didn't help.

## Against the benchmarks

| Benchmark | Sharpe | Compound return |
|---|---:|---:|
| **Buy and hold** | **+0.522** | **+213.04%** |
| Always long | +0.522 | +213.04% |
| Always short | −0.522 | −68.05% |
| Always flat | 0.000 | — |
| Random (200 seeds, mean) | −0.013 | — |
| Single-feature linear baseline | −3.134 | −99.68% |
| Best power combination | −0.971 | −82.89% |

Both the baseline and the best combination **lost more than 80% net of fees.** Buy-and-hold's +213%
is the hurdle neither comes close to, and even the best result sits well outside the random
baseline's 90% range of [−0.812, +0.841].

## The real finding: turnover, not signal quality

This is the most important part of the notebook. Gross Sharpe is **uncorrelated** with net Sharpe
across the sweep:

| Configuration | Gross | Net | Non-trading bars (of 30,240) | Time in market |
|---|---:|---:|---:|---:|
| Baseline (single feature) | +0.234 | −3.134 | 29,486 | 97.51% |
| High odd (best net) | +0.734 | **−0.971** | 28,597 | **94.57%** |
| Mixed one negative (best gross) | **+1.252** | −3.040 | 30,197 | 99.86% |
| Mixed negative odd (worst net) | −0.346 | −4.320 | 30,173 | 99.78% |

**The combination with the best gross Sharpe of all nine configurations tested — +1.252, beating even
buy-and-hold — produced one of the worst net results**, because it was in the market and flipping
position on 99.86% of bars.

And the eventual net winner didn't win on signal quality: its gross Sharpe of +0.734 trailed three
other combinations. **It won because its no-trade zone actually filtered something**, leaving it in
the market 94.57% of the time instead of the 99.7–99.9% everywhere else.

That difference is worth roughly 4,465 round-trip-equivalent position changes a year against a
per-trade cost of 4 basis points — costs that compound bar over bar across 3.45 years of stitched
out-of-sample trading into the near-total losses above.

**The practical lesson:** the no-trade threshold established in notebooks 001 and 002, set at twice
the taker fee, was calibrated for a *single raw feature*. It does not automatically transfer to
power-transformed features, whose predicted-return scale — and therefore how often the prediction
crosses the threshold — changes with the transform. **A model can look better before fees and still
lose more after them, if that flexibility mainly translates into more frequent, smaller-edge trades.**

## The best combination, fold by fold

All 42 folds are responsive, with no constant-direction degeneracy anywhere. Bias magnitudes stayed
below the maximum feature contribution in every fold.

Per-fold Sharpe ranges from +7.656 to −7.458, with 14 folds showing a strong result in either
direction and 10 near zero — **no directional consistency across folds.**

Position mix across all 42 folds: **42.59% long, 51.98% short, 5.43% flat.** The model traded both
directions, ruling out "always long" or "always short" as the failure mode. More short than long, but
that is a preference, not a constant bet.

## Numerical stability of negative powers

The floor for negative exponents was set at 1e−4 before looking at any results, chosen to sit below
the typical smallest non-zero hourly move. The safeguard floors the absolute value before
exponentiating while preserving sign, so odd negative powers stay sign-preserving and even ones stay
positive.

**All eight combinations produced finite results**, including the most numerically aggressive ones,
with no widespread undefined values. The design worked as intended.

## Statistical verdict

**Deflated Sharpe:** at a per-period net Sharpe of −0.01037 across 30,240 bars, with skew +0.062 and
kurtosis 17.310, the probability of a true positive edge given the best of 8 trials is **0.06%** —
indistinguishable from what noise's best-of-8 looks like.

**Bootstrap 95% interval on excess return over buy-and-hold, per bar:** **[−0.00018, +0.00001]** —
tight, centred near zero, and containing zero.

---

## Follow-up: recalibrating the threshold and adding a minimum hold

The original threshold was set at twice the taker fee — but the true round-trip cost of a full
position flip is twice the taker fee **plus slippage**, i.e. 33% higher. Would fixing that, and adding
a minimum hold period between position changes, turn anything positive?

Three configurations were trained once with explicit per-configuration seeds to capture raw
predictions, then a 7 × 6 grid was swept purely by re-deriving positions and costs from those fixed
predictions — **no retraining.** That isolates the trading rule from the model fit.

The reproducibility cross-check at the original settings confirms the baseline matches exactly
(−3.134), so the seeding reproduces correctly.

**One grid point achieved a positive net Sharpe:**

- The **baseline single-feature model**, at 5× the true round-trip cost, with a **48-bar minimum
  hold**.
- Net Sharpe **+0.238**, compound return **+53.67%**, in the market 92.92% of bars, with just 454
  position changes.

That reverses the original −3.134 — a 3.37-unit improvement.

### Which lever actually did the work?

All three configurations' best results occurred at the 48-bar minimum hold, but the two levers are
not interchangeable. Isolating each on the baseline:

| Threshold multiple | No minimum hold | 48-bar minimum hold |
|---|---:|---:|
| 1× | −3.140 | −0.539 |
| 2× | −3.113 | −0.584 |
| 3× | −3.132 | **+0.117** |
| 5× | −3.161 | **+0.238** |

**With no minimum hold, raising the threshold from 1× to 5× does essentially nothing** — Sharpe stays
pinned around −3.1 regardless.

**The minimum hold does almost all the work**, taking the baseline from catastrophic to merely losing
at any threshold. But it isn't sufficient on its own to cross into positive territory — that only
happens once the threshold is *also* pushed to 3–5× the true cost **on top of** the hold period. **The
two levers compound rather than substitute.**

### And it still isn't a finding

The improvement is fragile in two ways.

**Statistically**, the deflated Sharpe probability is **1.51%** at the honest count of 126 trials
(3 configurations × 42 grid points), and the bootstrap interval on excess return over buy-and-hold,
**[−0.00010, +0.00008]**, still contains zero.

**Economically**, +53.67% compound over 3.45 years is far below buy-and-hold's +213.04% over the same
window. **A positive Sharpe here is not the same as a strategy worth trading over just holding the
asset.**

---

## Bottom line

Polynomial power features on a single previous return do not reveal a hidden signal. The best
combination improves on the baseline — from −3.134 to −0.971 — but still loses more than 80% after
fees, sits outside the random baseline's range, and fails both the deflation and the interval test.

**The more useful finding is why every configuration lost so much: turnover, not signal quality.**
Gross Sharpe was uncorrelated with net Sharpe across all nine configurations. The best gross
performer was one of the worst net performers, because it traded on 99.86% of bars. The net winner
won because its no-trade zone was the only one that meaningfully filtered anything.

A no-trade threshold calibrated for one feature's scale does not transfer to a transformed feature's
scale. A configuration can look strictly better before fees and still lose more after them.

The follow-up confirms this and sharpens it: **threshold recalibration alone does nothing; a minimum
hold period does almost all the work of controlling turnover** — and even then, the one positive
result is statistically indistinguishable from noise and economically worse than doing nothing.

**Turnover control is necessary but not sufficient.** The underlying signal must also exist, and here
it does not. Neither more parameters, nor exotic powers, nor sign-agnostic combinations, nor
recalibrated trading rules change the conclusion notebook 002 reached: Bitcoin return prediction at
hourly frequency with simple features, under realistic fees and robust validation, does not yield a
tradeable edge.

*Notebook: `src/research/016_polynomial_power_features.ipynb`.*
