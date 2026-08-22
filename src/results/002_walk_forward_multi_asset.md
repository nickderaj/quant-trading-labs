# 002 — Walk-Forward Validation Across Multiple Assets

## The question

Notebook 001 produced apparently profitable results from a single train/test split on one symbol
and one year of data, and flagged that those numbers could not be trusted. Does any of that
apparent edge survive proper validation?

## What was done

The same linear return-prediction approach, put through every check notebook 001 said was
missing:

- **Walk-forward testing** over five years instead of a single split.
- **An origin-shift check** — the identical configuration, with only the start date of the fold
  grid moved by 0, 7, 14 and 21 days.
- **A timeframe sweep** — 28 combinations of bar interval and train/test window length.
- **A structured grid search** — 54 pre-declared combinations of features, loss function,
  no-trade threshold and model form.
- **Weight inspection on every fold**, to detect the constant-bet degeneracy that explained
  notebook 001's headline result.
- **A properly annualised buy-and-hold comparison** per symbol.
- **Deflated Sharpe** across all 122 configurations searched in the notebook.
- **Bootstrap confidence intervals** on excess return over buy-and-hold.
- **A frozen-configuration test** applying the BTC winner unchanged to ETH, DOGE, SOL, XRP and BNB.

A later correction re-ran the deflated Sharpe and the bootstrap using the return series' real
skew and kurtosis and a block bootstrap, instead of assuming normality and independence.

## Data

Notebooks 001a/001b downloaded raw trades and aggregated them into bars at load time. One
symbol-year of ticks is about 12GB; five years across six symbols would run to hundreds of
gigabytes. This notebook switched to pre-aggregated kline files instead — tens of kilobytes per
symbol-month.

The two feeds were reconciled before the switch: open and close match to about 1e-6 relative
difference; high, low and volume differ by up to roughly 0.1%, most likely from differing
trade-type inclusion between feeds. Not a blocker.

Two data bugs surfaced and were fixed. The kline CSV schema was being inferred, and it failed on
months where the volume column's early rows happened to look integer-valued and a later row
carried a decimal — fixed with explicit schema overrides so nothing is guessed. And a fold
containing zero losing trades returned `None` from a `.mean()` call and tripped an assertion,
which was guaranteed to happen somewhere across dozens of folds.

SOL and XRP are each missing the same 120 hours between February and April 2022. This is a real
gap in the exchange's own archive for those two symbols, not a downloader fault — about 0.27% of
bars, noted and not patched.

## Results

**No validated edge was found.**

### The origin-shift check alone is nearly decisive

Same configuration (BTC hourly, 180-day train, 30-day test, rolling), only the fold grid's start
date moved:

| Grid start offset | Sharpe |
|---|---|
| 0 days | −1.29 |
| 7 days | +0.07 |
| 14 days | +0.0001 |
| 21 days | −0.62 |

Nothing else changed. The sign flips twice from shifting the calendar alignment by up to three
weeks. A result that only appears at one arbitrary alignment is not a result.

### Timeframe sweep

28 combinations of bar interval (1h, 4h, 12h, 1d) against train/test window (from 90/30 up to
365/180, both rolling and anchored), using a single mean-reversion feature. Sharpe ranged from
+1.06 (daily bars, 90/30 rolling) down to −0.79, with no dominant setting. Whatever wins is
winning because it happened to suit this particular grid.

### Grid search

54 pre-declared combinations: features (mean-reversion / momentum / combined) × loss
(MSE / L1 / Huber) × no-trade threshold (0 / one taker fee / round trip) × model form (linear /
tanh-bounded linear), on BTC 12-hour bars, 180-day train, 90-day test, rolling.

Winner: momentum features, Huber loss, one-taker-fee threshold, tanh model. **Sharpe 0.76**,
compounding to +411% across 17 folds, with 0% of folds flagged degenerate.

### Weight inspection

Every fold of both the baseline and the winning configuration was checked for whether the
absolute bias term exceeded the largest contribution the feature itself ever made in that fold.
When it does, the position never really depends on the feature — it is a constant long or short,
which is exactly what notebook 001's "+34%" turned out to be.

Result: **0% degenerate**, everywhere. The weights were genuinely responsive to the feature in
every fold. That does not establish an edge; it rules out one specific way of faking one.

### Buy-and-hold, measured honestly

Five-year BTC buy-and-hold: Sharpe 0.21, +75% compound. Year by year, though:

| Year | Sharpe |
|---|---|
| 2021 | +0.98 |
| 2022 | −1.64 |
| 2023 | +2.14 |
| 2024 | +1.42 |
| 2025 | −0.18 |
| 2026 (partial) | −1.70 |

Rolling one-year compound return ranges from −71% to +166% purely on start date, and is positive
in 61% of windows. Any single-period buy-and-hold comparison is exactly as fragile as the
strategy number it is supposed to discipline.

Because log returns add over time, dividing total log return by the number of years gives a true
geometric average rate rather than an arithmetic mean of noisy annual percentages:

| Symbol | Avg annual | Avg monthly | Sharpe |
|---|---|---|---|
| BNB | +13.7% | +1.08% | 0.21 |
| SOL | +17.2% | +1.33% | 0.16 |
| BTC | +11.9% | +0.94% | 0.21 |
| XRP | +9.6% | +0.76% | 0.11 |
| ETH | −5.7% | −0.49% | −0.08 |
| DOGE | −21.6% | −2.01% | −0.27 |

The strategy's own out-of-sample return under the same treatment: **+0.29% per month** over its
50.2-month span, against BTC buy-and-hold's **+0.94% per month**. Simply holding the asset earned
more than three times the rate. Not close.

Always-long, always-short, always-flat and random baselines were also run. Across 200 random
seeds the random baseline had mean Sharpe −0.02, standard deviation 0.48, and a 90% range of
[−0.80, +0.78]. The winning configuration's 0.76 sits just outside that range on its own — but
see the deflation below for why that isn't enough.

Fold by fold, the winning configuration beat buy-and-hold in **9 of 17** out-of-sample folds. A
coin flip.

### Deflated Sharpe and bootstrap

122 configurations were searched in total across the notebook (28 timeframe + 16 origin-shift +
54 grid + 24 symbol × interval). Reporting the best of many noisy trials inflates it even with
zero true edge, because what's being reported is the maximum of N draws rather than one draw.
Correcting for that:

**P(true Sharpe > 0 | best of 122 trials) = 0.69%.**

That is not "unlikely" — it is indistinguishable from what the best of 122 pure-noise trials
looks like.

The bootstrap 95% confidence interval on per-fold excess return (strategy minus buy-and-hold) is
**[−0.19, +0.14]**, which contains zero.

### Across six symbols

The winning configuration was frozen on BTC and applied unchanged to the other five, with no
per-symbol retuning (which would simply reintroduce the overfitting notebook 001b already
demonstrated). It beat its own buy-and-hold on BTC, SOL, DOGE and ETH (excess Sharpe +0.64, +0.49,
+0.39, +0.20) and lost on BNB and XRP (−0.08, −0.61). The symbol × interval Sharpe heatmap is a
checkerboard with no consistent sign per symbol across intervals — precisely the noise signature
the heatmap was built to detect.

Cross-symbol correlation of fold returns is mostly weak (|r| < 0.5; BTC–ETH 0.01, BTC–SOL 0.35,
DOGE anti-correlated with most at −0.15 to −0.35). At fold granularity this is not one leveraged
crypto-beta bet wearing six tickers.

## Correcting the inference assumptions

Both the deflated Sharpe and the bootstrap above relied on assumptions that don't hold for crypto
returns: normality (skew 0, kurtosis 3) for the first, and independent resampling for the second.
Both were re-run with real inputs, and the notebook was re-executed end to end. It reproduced
bit-for-bit — the retrained winning configuration's Sharpe of 0.07 and every downstream number
matched the prior run exactly, confirming the seeding and deterministic-algorithm settings
genuinely pin this notebook down.

**Real moments of the winning configuration's own return series:** skew **+0.059** (essentially
symmetric), kurtosis **8.56** (excess kurtosis +5.56 over the normal's 3 — heavily fat-tailed, as
expected).

**Deflated Sharpe, before and after** (same Sharpe 0.07, 122 trials, 3,060 observations):

| Assumption | Skew | Kurtosis | P(true Sharpe > 0) |
|---|---|---|---|
| Normal | 0 | 3 | 0.69% |
| Real moments | +0.059 | 8.56 | 0.69% |

Unchanged at this precision. Near-zero skew means the skew term in the standard-error correction
contributes almost nothing, and at a per-period Sharpe this close to zero the kurtosis term
cannot move the headline. The correction matters when skew is large or the raw Sharpe is larger —
notebook 003 contains a case where it does.

**Bootstrap, independent versus block**, on fold-level excess return over the same 17 folds and
the same seed:

| Method | Block length | 95% CI |
|---|---|---|
| Independent resampling | 1 (n/a) | [−0.190, +0.135] |
| Block bootstrap (auto length) | 1 | [−0.190, +0.135] |

Identical. The autocorrelation-based rule selected block length 1 for this 17-point series — no
significant autocorrelation was detectable at fold level, so the block bootstrap degenerates to
the independent case. This does not mean autocorrelation is irrelevant in general (bar-level
returns are visibly autocorrelated through volatility clustering); it means this particular
17-point summary series doesn't show it strongly enough for the heuristic to lengthen the block.

Both conclusions stand unchanged.

## Bottom line

No validated edge — the same conclusion as notebook 001, but now demonstrated quantitatively
rather than asserted. A result that looks good on raw Sharpe, or on beating a directional
baseline, falls apart under origin-shift robustness, deflation for the number of trials, a
bootstrap interval on excess return, fold-by-fold comparison, cross-asset generalisation, and a
fair per-month comparison against simply holding the asset.

What carries forward is the machinery: walk-forward splitting and running, model description and
degeneracy checks, deflated Sharpe, and the buy-and-hold, constant and random baselines. The bar
any future result has to clear is all of the above — not a good headline Sharpe.

*Notebook: `src/research/002_walk_forward_multi_asset.ipynb`.*
