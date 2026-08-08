# Walk-Forward Multi-Asset - Results Summary

## What

This notebook takes the linear return-prediction approach from notebook 1 and subjects it to rigorous validation: walk-forward testing over 5 years instead of one split, an origin-shift robustness check, a timeframe sweep, a large "try different things" grid search, weight inspection for degenerate constant bets, honest buy-and-hold benchmarking, deflated Sharpe correction for multiple testing, bootstrap confidence intervals, and a frozen-config test across 6 symbols.

## Why

Notebook 1 found apparently profitable results from a single train/test split on one symbol/year, but explicitly flagged that these numbers weren't trustworthy without walk-forward validation, baseline comparisons, and protection against overfitting via many-config search. This notebook exists to apply exactly those checks and see whether any of notebook 1's apparent edge survives.

## How

Switched from tick-aggregated Binance data to pre-aggregated klines for tractability across 5 years and 6 symbols, then ran: an origin-shift check (same config, fold grid shifted by 0/7/14/21 days), a 28-combo timeframe/window sweep, a 54-combo "try different things" grid (features x loss x threshold x model), a weight/bias degenerate-bet check on every fold, a properly annualized buy-and-hold comparison per symbol, deflated Sharpe probability across all 122 configs searched, bootstrap CIs on excess return, and application of the frozen winning config unchanged across ETH/DOGE/SOL/XRP/BNB. A later inference correction re-ran the deflated Sharpe and bootstrap CI using real return moments (skew/kurtosis) and block bootstrapping instead of normal/i.i.d. assumptions.

## Results

No validated edge was found. The origin-shift check alone showed Sharpe flipping sign twice from moving the fold grid by three weeks (-1.29 to +0.07 to +0.0001 to -0.62). The best grid-search config (momentum features, huber loss, tanh model) had Sharpe 0.76, but deflated Sharpe probability of a true edge given 122 trials searched was only 0.69%, and its bootstrap CI on excess return over buy-and-hold included zero. It beat buy-and-hold in only 9 of 17 out-of-sample folds (a coin flip), and its own OOS monthly return was less than a third of BTC buy-and-hold's average monthly rate. Applied unchanged across 6 symbols, it beat buy-and-hold on some and lost on others with no consistent pattern. The later inference correction (real skew/kurtosis, block bootstrap) left all conclusions unchanged. Bottom line: no tradeable edge, but reusable walk-forward/deflated-Sharpe/baseline machinery for future notebooks.

Notebook 1 fit a straight line on one train/test split of one symbol, one year of data. This notebook walk-forward validates over 5 years, shifts the fold grid to check it isn't an artifact, benchmarks against buy-and-hold properly, inspects every fitted model's weights for the "it's just a constant bet" failure, and runs the same frozen config across 6 symbols.

## Data source switch

Notebooks 1 and 2 download raw Binance trades and aggregate into bars at load time - one symbol-year of ticks is ~12GB, 5 years of 6 symbols that way would be hundreds of GB and hours of downloading. Switched to Binance's pre-aggregated klines files instead, tens of KB per symbol-month. Checked they agree with the tick-aggregated bars first: close/open match to ~1e-6 relative difference, high/low/volume differ by up to ~0.1% (probably differing trade-type inclusion between the two feeds), not a blocker. Klines used throughout.

Found a bug along the way: klines CSV schema inference chokes on months where the volume column's early rows happen to look integer valued, then fails when a later row has a decimal. Fixed with explicit schema_overrides so parsing doesn't guess. Also fixed a crash in eval_model_performance - a fold with zero losing trades returned None from .mean() and the assert blew up, guaranteed to happen somewhere across dozens of folds.

SOL and XRP are each missing the same 120 hours in Feb-Apr 2022 - a real gap in Binance's own archive for those two symbols, not a downloader bug. ~0.27% of bars, noted and not corrected for.

## Origin shift

Same baseline config (BTC 1h, 180d train / 30d test, rolling), only the fold grid's start date shifted by 0/7/14/21 days:

- offset 0: sharpe -1.29
- offset 7: sharpe +0.07
- offset 14: sharpe +0.0001
- offset 21: sharpe -0.62

Nothing else changed. Sign flips twice just from moving where the grid starts by up to three weeks. A result that only looks good on one specific calendar alignment isn't a result.

## Timeframe sweep

28 combos of bar interval (1h/4h/12h/1d) x train/test window (90/30 up to 365/180, rolling and anchored), single mean-reversion feature. Sharpe ranged from +1.06 (1d bars, 90d/30d rolling) down to -0.79. No dominant setting - whatever wins is winning because it happened to fit this particular grid, not because that timeframe is structurally better.

## Try different things grid

54 pre-declared combos: features (mean-reversion / momentum / combined) x loss (mse/l1/huber) x threshold (0 / 1x taker fee / round trip) x model (linear / tanh-bounded linear), BTC 12h bars, 180d/90d rolling. Winner: momentum features, huber loss, 1x taker threshold, tanh model. Sharpe 0.76, compound +411% over 17 folds, 0% degenerate.

## Weight/bias check

Every fold of both the baseline and the winning grid config got inspected: is |bias| bigger than the max the feature's own contribution ever reaches in that fold. If so the position never actually depends on the feature, it's a constant long or short bet wearing a model costume - exactly what notebook 1's "always short" result turned out to be.

This time: 0% degenerate across every section, baseline and winning config both. The linear/tanh weights were genuinely responsive to the feature in every fold checked. Doesn't mean there's an edge, just rules out this specific failure mode.

## Buy and hold - per year is noisy, wide average is not

Full 5yr BTC buy and hold: sharpe 0.21, +75% compound. But year by year:

- 2021: sharpe +0.98
- 2022: sharpe -1.64
- 2023: sharpe +2.14
- 2024: sharpe +1.42
- 2025: sharpe -0.18
- 2026 (partial): sharpe -1.70

Rolling 1yr compound return ranges -71% to +166% depending purely on start date, positive in 61% of windows. Any single-period B&H comparison is exactly as fragile as the strategy numbers it's meant to benchmark against.

Fix: since log returns are additive over time, total_log_return / years gives a true geometric average rate, not an arithmetic mean of noisy yearly percentages. BTC average annual: +11.84%. Average monthly: +0.94%. Same treatment across all six:

| symbol | avg annual | avg monthly | sharpe |
|---|---|---|---|
| BNB | +13.7% | +1.08% | 0.21 |
| SOL | +17.2% | +1.33% | 0.16 |
| BTC | +11.9% | +0.94% | 0.21 |
| XRP | +9.6% | +0.76% | 0.11 |
| ETH | -5.7% | -0.49% | -0.08 |
| DOGE | -21.6% | -2.01% | -0.27 |

Strategy's own OOS return put through the same treatment: average monthly +0.29% over its 50.2 month span, against BTC buy-and-hold's +0.94%/month over its full history. Buy and hold averaged more than 3x the strategy's monthly rate. Not close.

Also checked always-long/short/flat and random (200 seeds): random baseline sharpe mean -0.02, std 0.48, 90% range [-0.80, +0.78]. The winning grid config's 0.76 sits outside that range on its own - but see deflated sharpe below for why that's not enough.

Fold by fold, the winning BTC config beat buy-and-hold in 9 of 17 out-of-sample folds. Coin flip.

## Deflated sharpe

122 configs searched total across the notebook (28 timeframe + 16 origin shift + 54 grid + 24 symbol x interval). Picking the best of many noisy trials inflates its sharpe even with zero real edge, since you're reporting the max of N draws not one draw. Deflated sharpe corrects for this: P(true sharpe > 0 | best of 122 trials) = 0.69%. Not "unlikely" - indistinguishable from what noise's best-of-122 looks like.

Bootstrap 95% CI on strategy-minus-buy&hold excess return per fold: [-0.19, +0.14]. Includes zero, can't reject no real edge.

## Multi-asset

Winning config frozen from BTC, applied unchanged to ETH/DOGE/SOL/XRP/BNB - no per-symbol retuning, that would just reintroduce the overfitting notebook 2 already showed. Beat its own buy-and-hold on BTC/SOL/DOGE/ETH (excess sharpe +0.64/+0.49/+0.39/+0.20), lost on BNB/XRP (-0.08/-0.61). Symbol x interval sharpe heatmap is a checkerboard, no consistent sign per symbol across intervals - the noise signature the heatmap was built to catch.

Cross-symbol correlation of fold returns mostly weak (|r| < 0.5, BTC-ETH 0.01, BTC-SOL 0.35, DOGE anti-correlated with most at -0.15 to -0.35). Not just one leveraged crypto-beta bet wearing six tickers, at least at fold granularity.

## Bottom line

No validated edge found. Same conclusion as notebook 1, now with the tools to show it quantitatively instead of asserting it: a result that looks good by raw sharpe or beating a directional baseline falls apart under origin-shift robustness, deflated sharpe, bootstrap CI on excess return, fold-by-fold comparison, cross-asset generalization, and a fair per-month comparison against just holding the asset.

The walk-forward machinery (research.walk_forward_splits, walk_forward_run, describe_linear_model, deflated_sharpe_prob, the buy-and-hold/constant/random baselines) is reusable for whatever gets tried next. The bar it needs to clear is all of the above, not a good headline sharpe.

## Inference correction

`deflated_sharpe_prob` and the fold-excess-return bootstrap CI above both used unexamined assumptions: normal returns (skew=0, kurtosis=3) for the former, i.i.d. resampling for the latter. Neither holds for crypto strategy returns. Re-ran both with real inputs, notebook re-executed end to end (fully reproducible - the retrained winning config's Sharpe, 0.07, and every downstream number matched the prior run bit-for-bit, confirming `research.set_seed(123)` plus `torch.use_deterministic_algorithms` actually pins this notebook down).

**Real moments of the winning config's own per-period return series**: skew = **+0.059** (essentially symmetric), kurtosis = **8.56** (fisher=False; excess kurtosis +5.56 over the normal's 3 - heavily fat-tailed, as expected for crypto).

**Deflated Sharpe, old vs new** (same sharpe=0.07, n_trials=122, n_obs=3060):

| | skew | kurtosis | P(true Sharpe > 0) |
|---|---|---|---|
| old (normal assumption) | 0 | 3 | 0.69% |
| new (real moments) | +0.059 | 8.56 | 0.69% |

Unchanged at this precision. The near-zero skew means the skew term in `deflated_sharpe_prob`'s standard-error correction contributes almost nothing, and at a per-period Sharpe this close to zero the kurtosis term's effect on the deflation is too small to move the headline number. The correction matters more when skew is large or the raw Sharpe is larger - see notebook 3's cfg1_4h below for a case where it does.

**Bootstrap CI, old (i.i.d.) vs new (block)**, fold-level excess return (strategy − buy&hold), same 17 folds, same seed/n_boot:

| | block length | 95% CI |
|---|---|---|
| old (`bootstrap_ci`, i.i.d.) | 1 (n/a) | [-0.190, +0.135] |
| new (`block_bootstrap_ci`, auto block length) | 1 | [-0.190, +0.135] |

Identical. `research._auto_block_length`'s ACF-based rule picked block length 1 for this 17-observation fold-excess-return series - no significant autocorrelation was detected at the fold level, so the block bootstrap degenerates to the i.i.d. case here. This doesn't mean autocorrelation isn't a real concern in general (bar-level returns are visibly autocorrelated via volatility clustering); it means this particular fold-level summary series, with only 17 points, doesn't show it strongly enough for the heuristic to pick a longer block.

**Conclusion unchanged.** Both the normal-assumption and real-moment deflated Sharpe sit at 0.69% - indistinguishable from noise's best-of-122 either way - and the bootstrap CI still includes zero either way. "No validated edge" stands.
