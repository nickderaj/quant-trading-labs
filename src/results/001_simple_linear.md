# Simple Linear Models - Results Summary

Both notebooks fit a straight line to predict BTC next return, trade on the sign. Found 3 bugs in fee/trading logic. Fixed them, reran everything. Numbers changed a lot. Still don't trust these as real strategies.

## The 3 bugs (now fixed in research.py)

1. Fees charged every bar, not just when you trade.
   - Old code charged the round trip fee on every bar, even when position held for months.
   - Fixed: fee now based on how much position actually changed that bar (0 if unchanged).

2. Fee math bug: add_tx_fees_log used log(fee) instead of log(1 - fee).
   - log(0.0001) = -9.2 per trade. Not a fee, a wipeout.
   - Fixed the formula.

3. Models couldn't sit out.
   - Old code always forced a trade (sign(prediction)), even when prediction was basically zero/noise.
   - Added a no-trade zone: if predicted edge is smaller than round trip fee, don't trade. Position goes flat instead.

## Notebook 1 - Mean Reversion

- Before fixes: single lag 1h models lost money (-15% to -25%), Sharpe -1.5 to -2.5. Fee code broken so after fee numbers meaningless.
- After fixes: the always short model (lag 3 feature) nets +34% after fees over test window.
- Model weights show bias term dominates, feature weight barely matters. It's basically a constant short bet the whole window. BTC fell during the test window so the number is positive.
- 6h/12h multi feature configs still show positive Sharpe (2 to 2.7) but only ~182 trades from one single train/test split. Sample too small.

## Notebook 2 - Trend Following

- Before fixes: best swept config showed Sharpe 8.89 pre fee, but after (broken) fees lost -23.23%.
- After fixes: same config nets +59.10% after (correct) fees, no-trade filter visibly skipping some bars.
- Model was picked by sweeping 756 combinations (3 intervals x 3 loss functions x 3 test sizes x 28 feature combos) and taking best Sharpe. Same warning as before still applies: this shows how easy it is to overfit a backtest, not a strategy to trade.

## Bottom line

- The 3 fixes were real bugs, results flipped from big losses to big gains.
- Fixing fee math doesn't fix the actual problem: no real edge found yet.
- Both winning results are either a constant directional bet, or the best pick out of hundreds of overfit combinations on one asset, one split, one year of data.
- None of this validated out of sample, cross asset, or against buy and hold baseline.

## What would make these numbers trustworthy

- Walk forward testing, many splits not one.
- Compare against buy and hold and always flat baselines.
- Penalize for how many configs were tried (deflated Sharpe).
- Test on more than 1 symbol / 1 year.
- Check model isn't just a constant directional bet, inspect weights before trusting Sharpe.
