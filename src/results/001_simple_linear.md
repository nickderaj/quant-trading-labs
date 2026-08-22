# 001 — Simple Linear Models

## The question

Can the simplest possible model — a straight line — predict Bitcoin's next-bar return well
enough to trade on the sign of that prediction?

This is the first notebook in the programme. Its real job is less about finding an edge and more
about building the basic machinery (fee accounting, position logic, backtest plumbing) and
proving it works, because a broken cost model can turn a losing strategy into an apparently
winning one and quietly poison everything built on top of it.

## What was done

Two notebooks, both on Bitcoin:

- **001a** fits a linear model on mean-reversion features (the return some number of bars ago,
  expecting it to reverse).
- **001b** fits a linear model on trend-following features and sweeps 756 configurations
  (3 bar intervals × 3 loss functions × 3 test-set sizes × 28 feature combinations).

Both predict the next bar's return and take a position equal to the sign of that prediction. One
train/test split, one symbol, roughly one year of hourly data.

## Three bugs found and fixed

These were real defects in the backtest code, not modelling choices, and all three are now fixed
in `src/research.py`.

**1. Fees charged on every bar instead of only when the position changed.** The original code
deducted a full round-trip fee every single bar, even when the position had been held unchanged
for months. Fixed so the fee scales with how much the position actually moved that bar — zero if
it didn't move.

**2. The fee formula used `log(fee)` where it should have used `log(1 - fee)`.** With a 1bp fee,
`log(0.0001) = -9.2` in log-return terms. That isn't a transaction cost, it's a total wipeout of
the account on every trade.

**3. The model could never sit out.** Taking `sign(prediction)` forces a position on every bar,
including when the prediction is indistinguishable from zero. Added a no-trade band: if the
predicted edge is smaller than the round-trip cost of acting on it, go flat instead.

## Results

Fixing the fee bugs flipped the headline numbers from large losses to large gains — and neither
gain is real.

**001a, mean reversion.** Before the fixes, hourly single-lag models lost 15–25% with Sharpe
between −1.5 and −2.5, though those after-fee numbers were meaningless given the broken fee code.
After the fixes, the best model returned **+34% net of fees**.

Inspecting the fitted weights explains why: the bias term dominates and the feature weight barely
contributes, so the position is effectively a constant short held across the whole window. Bitcoin
fell during that window, so the number came out positive. It is a directional bet wearing a model
costume, not a signal.

The 6-hour and 12-hour multi-feature configurations still showed Sharpe between 2.0 and 2.7, but
on roughly 182 trades from a single train/test split — far too small a sample to mean anything.

**001b, trend following.** Before the fixes, the best swept configuration showed a pre-fee Sharpe
of 8.89 that became −23.23% after the (broken) fee deduction. After the fixes, the same
configuration returned **+59.10% net**, with the no-trade band visibly skipping bars.

That configuration was chosen as the best Sharpe out of 756 combinations. Selecting the maximum
of hundreds of noisy trials inflates it even when no edge exists at all. This demonstrates how
easily a backtest overfits; it does not demonstrate a strategy.

## Bottom line

- The three fixes were genuine bugs, and correcting them moved results dramatically.
- Correct fee arithmetic does not create an edge. None was found here.
- Both apparently profitable results have a mundane explanation: one is a constant directional
  bet, the other is the best of hundreds of overfit configurations on one asset, one split, one
  year.
- Nothing here was validated out of sample, across assets, or against a buy-and-hold benchmark.

## What would make numbers like these trustworthy

This list is the direct agenda for notebook 002:

- Walk-forward testing across many splits, not one.
- Comparison against buy-and-hold and always-flat benchmarks.
- A penalty for the number of configurations tried (deflated Sharpe).
- More than one symbol and more than one year.
- Weight inspection on every fit, to catch constant-bet degeneracy before trusting any Sharpe.

*Notebooks: `src/research/001a_mean_reversion_single_asset_ML.ipynb`,
`src/research/001b_trend_following_single_asset_ML.ipynb`.*
