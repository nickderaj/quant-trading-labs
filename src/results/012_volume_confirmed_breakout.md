# 012 — Does Breakout Volume Actually Confirm the Move?

## The question

A widely believed piece of trading folklore says a breakout backed by heavy volume is more likely to
follow through than one on quiet volume. That claim had never been tested in this repo.

This notebook tests it directly. The breakout detector from notebook 011d is generalised to be
symmetric long/short, and then run two ways over the same instruments and the same period: **gated**
on the breakout bar's volume exceeding a multiple of its own trailing median, and **ungated** as a
control.

A second, deliberate goal: build the **best-powered book in this programme**. Every prior null could
in principle be dismissed as underpowered, so this one pools 88 instruments across three asset
classes into a single backtest while holding the trial count to an honest 12.

**The result: volume confirmation does not earn its keep.** Net Sharpe is positive at every offset
and the cost-stress effect is real and correctly signed — but the volume filter's point estimate
moves the **wrong** direction against its own control, and the interval is too wide to save it
either way.

## The rule, and how its thresholds were fixed

Two binding constraints, both honoured:

**One parameter set for every instrument, expressed scale-free.** The prior-run and base-width
thresholds are expressed as multiples of each instrument's own average true range, not as
percentages of price. That lets the same frozen numbers transfer across asset classes with wildly
different price levels and volatility regimes — Bitcoin, corn futures and a broad equity ETF — without
needing a second parameter set.

**Thresholds frozen once, from a window disjoint from the backtest window.** For each of the 88
instruments independently, its own first three years of history are held out as calibration-only.
The three thresholds come from declared percentiles of that pooled calibration sample:

| Parameter | Rule | Value |
|---|---|---|
| Base width | 30th percentile of base range in volatility units | **2.8** |
| Prior run | 70th percentile of prior-run return in volatility units | **4.5** |
| Volume multiple | 75th percentile of volume-to-trailing-median ratio | **1.39** |

The percentiles were chosen once from the calibration pool's own shape — **never by trying several
values and keeping the best.** Between 66,645 and 75,581 calibration bars fed these three numbers,
pooled across all asset classes, before any of them traded a dollar.

## A data extension this required

The existing continuous-futures builder returns only closing prices and drops open, high, low and
volume entirely — fine for spread and return work, useless for a breakout rule.

A new builder was added, reusing the identical roll schedule and front-month selection logic but
joining the full bar instead of the close alone, and back-adjusting open, high and low by the same
additive offset already used for close. Raw per-contract prices jump at every roll, and the breakout
rule needs a gap-free level series in a way the existing machinery never did.

**Volume is deliberately not back-adjusted.** It is a traded quantity, not a price, and an additive
shift would be meaningless. It is left exactly as flagged — discontinuous at rolls, valid only
relative to its own trailing within-contract history. Both roll-related volume hazards are handled
the same way: a trailing volume window is disqualified from firing on the volume leg (never on the
ungated control) whenever it would straddle a roll or land on the roll bar itself.

## The universe

| Asset class | Instruments | Regime reference | Gate open |
|---|---:|---|---:|
| Crypto perpetuals | 30 (including two effectively delisted) | Bitcoin, 1 confirmation | 54.1% |
| Commodity equities/ETFs | 42 (27 proxies excluded for unreliable volume) | Three broad index ETFs, 2 of 3 | 57.3% |
| Futures | 16 products, front-month continuous | Energy, metal and equity-index products, 2 of 3 | 43.9% |

The futures regime reference is a new construction, structurally identical to the equity vote and
**not fitted to this rule's performance**.

Costs reuse the crypto convention uniformly across all three classes — the same disclosed-not-measured
assumption notebook 011d made. This repo has no established futures- or equity-specific cost model,
and inventing three different ones would itself be another undeclared parameter.

## Results

| Offset | Sharpe (1×) | (2×) | (3×) | Gated trades |
|---|---:|---:|---:|---:|
| 0 | **+0.1150** | +0.0939 | +0.0743 | 406 |
| 7 | **+0.1150** | +0.0939 | +0.0743 | 406 |
| 14 | **+0.1151** | +0.0940 | +0.0743 | 406 |
| 21 | **+0.1152** | +0.0940 | +0.0744 | 406 |

Net Sharpe is positive at every offset, so the first requirement clears.

**But that check turns out to be nearly vacuous here, and that is disclosed rather than presented as
a clean pass.** Look at the trade count: identical at 406 across all four offsets, with Sharpe
matching to three decimal places. That's a mechanical consequence of stacking the calibration
exclusion on top of the offset convention — each instrument's first three years are already excluded
from trading, so trimming a further 0 to 21 bars off the front of an already-long remaining series
changes essentially nothing.

Notebook 011d's version of this check was informative because it had no calibration exclusion sitting
in front of it. This one doesn't carry the same evidentiary weight. **A future notebook wanting this
check to bite should apply the offset *before* the calibration split, not after.**

### The volume filter moves the wrong way

The gated book returns **+27.4%** pooled, against the ungated control's **+45.6%** on the same
1,105-trade book. That isn't merely an insignificant improvement — **the point estimate is negative.**

The paired bootstrap of gated minus ungated returns, over 98 quarterly blocks, gives a difference of
**−18.2 percentage points, 95% interval [−148.9, +87.0]**. Wide enough to contain both a real
improvement and a real deterioration, so no directional claim survives — but the point estimate
offers no support for "volume confirmation helps" either.

### And the deflation is nowhere close

The deflated Sharpe probability, at 12 trials against 406 trades, is **0.064** — far below the 0.95
bar, despite this notebook pulling both available levers: pooling 88 instruments into one book, and
holding the trial count as low as it can honestly go.

It's also ineligible for the fundable flag on independent grounds: maximum drawdown of **−42.6%**,
well outside the 25% bound.

**Cost stress is real and correctly signed**, as everywhere else in this programme: the paired
comparison between 1× and 3× costs gives **−16.1 percentage points, interval [−18.7, −13.7]**.
Statistically distinguishable from zero. This null is not an artefact of an underpriced cost
assumption.

## The pooling is not balanced, and that's part of the finding

| Asset class | Gated trades | Ungated trades |
|---|---:|---:|
| Crypto | 8 | 15 |
| Equities/ETFs | **383** | **936** |
| Futures | 15 | 154 |

Equities supply **94% of the gated book's trades** and 85% of the ungated control's.

The headline sample size is real — 406 trades honestly clears more of the deflation bar than any
single-asset-class book in this programme has managed. But it is overwhelmingly **an equity result
wearing a three-asset-class label**, not an even pooling. Whatever the basket-level Sharpe says, it
is mostly describing behaviour on 42 commodity-sector equities and ETFs, with crypto and futures
each contributing too few trades to move the verdict either way.

Stated plainly rather than left to be inferred from the table.

## Bottom line

Both available levers — raise the sample size, lower the trial count — were pulled here, and the
answer is still a clean null. But it fails for a **third distinct reason**, different from the two
nulls in notebook 011d.

There, one failed because sample size alone killed it, and the other because it was the wrong sign
outright. Here, Sharpe is positive everywhere, the sample is the best-powered this programme has
built, and cost stress behaves exactly as a real trading rule's should — **and the one thing this
notebook actually set out to test comes back not supported.** The point estimate moves the wrong
direction and the interval is too wide to save it.

That is a publishable answer in its own right. "Does breakout volume confirm the move?" was a
genuinely untested question here, and it has now been tested honestly at a sample size an order of
magnitude larger than anything upstream of it. The answer is not "we couldn't tell" — it is **"the
mechanism does not pay for itself"**, a sharper and more useful negative than most of the
sample-size-limited nulls that came before it.

The temptation to test a second pattern — a golden cross, a pennant, a moving-average crossover — was
declared out of scope up front and stays out of scope. One test, no pattern zoo.

*Notebook: `src/research/012_volume_confirmed_breakout.ipynb`. The holdout remains untouched.*
