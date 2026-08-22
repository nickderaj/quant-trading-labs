# 003 — Cross-Sectional Signal Screening Across 30 Crypto Symbols

## The question

Notebook 002 found no edge in single-asset models, and every number it reported was gross of
transaction costs. This notebook changes both things at once:

- **Charge real costs everywhere.** Nothing here is reported gross without saying so.
- **Trade breadth instead of depth.** Rather than searching many single-asset configurations and
  backtesting whichever wins — a method notebook 002 showed overfits badly — screen a 30-symbol
  panel on rank correlation first, then backtest only the handful of signals that survive
  screening.

The point is measurement power (30 symbols instead of 1) and honesty about cost.

## The cost model, and which bar intervals it kills

Costs are charged per bar as `turnover × (taker fee + slippage)`, where turnover is the absolute
change in position from the previous bar. The position before the first bar counts as zero, so
the initial entry is charged too. The charge is applied as a log-return drag, so it compounds the
same way returns do.

Before anything else, a sanity check: simulate an always-in-market ±1 position that flips sign
with 40% probability each bar (a rough stand-in for a mediocre but real signal), 100,000 bars, at
3bp taker fee plus 1bp slippage.

| Bar interval | Annual drag (taker only) | Annual drag (taker + slippage) |
|---|---|---|
| 1h | 713% | 1,535% |
| 4h | 70% | 102% |
| 12h | 19% | 26% |
| 1d | 9% | 12% |

These are far larger than a back-of-envelope estimate would suggest, because that estimate
annualises a per-bar cost linearly while the real thing compounds. At hourly bars — 8,760 per
year — a fee that looks trivial per bar explodes geometrically over a year of turnover. The exact
multiplier depends on the flip-rate assumption, but the ranking across intervals doesn't.

**Decision: hourly bars are dropped from this run entirely.** Screening and backtesting proceed
at 4h, 12h and 1d only.

## The universe, and its remaining bias

30 USDT-margined perpetual futures, deliberately including 2021-era coins that later died or were
delisted, rather than picking today's most liquid 30 and backtesting them from 2021 (which would
be pure survivorship bias):

BTC, ETH, BNB, SOL, XRP, ADA, DOGE, DOT, AVAX, MATIC, LINK, LTC, ATOM, UNI, ETC, XLM, ALGO, VET,
FIL, TRX, EOS, AAVE, SAND, MANA, AXS, THETA, NEAR, FTM, LUNA, FTT.

**The residual bias is real and is not corrected for.** This list was chosen by hindsight — which
coins are remembered as having mattered and died — not by reconstructing the exchange's actual
listed-symbol history at each past date. A trader in 2021 would not have known FTM would survive
and LUNA wouldn't. This is smaller than top-30-today survivorship bias, but it isn't zero.

Nothing is forward- or back-filled across a listing or delisting boundary, so the panel is ragged
by construction:

| Symbol | First bar | Last bar | What happened |
|---|---|---|---|
| LUNA | 2021-07-01 | 2022-05-13 | The Terra/UST collapse — died as intended, the canonical case this universe was built to include |
| MATIC | 2021-07-01 | 2024-09-11 | Perpetual delisted around the POL migration |
| EOS | 2021-07-01 | 2025-05-21 | Perpetual delisted |
| FTT | 2022-04-15 | 2026-06-30 | Listed nine months after the panel starts; never delisted (it stayed listed through the FTX collapse, it just collapsed in price) |
| the other 26 | 2021-07-01 | 2026-06-30 | Full coverage |

Notebook 002's known archive gap persists: SOL and XRP are both missing 2022-03-01 to 2022-04-03.
Requiring at least 10 symbols present before computing any cross-sectional rank absorbs this
cleanly — 28 of 30 symbols are still available at every bar in that window.

Panel sizes with the holdout excluded: 252,053 rows at 4h, 84,036 at 12h, 42,033 at 1d.

The holdout period begins 2025-07-01 and is enforced in the data loader itself, not by
discipline: any request reaching past that date raises an error unless a flag is explicitly
passed, which happens in exactly one place in the whole run.

## The features

Every feature is computed per symbol and is strictly causal — rolling windows trail, momentum uses
past shifts only, and the panel is partitioned by symbol before any window is applied so a
rolling window can never pick up a different symbol's bars. The one deliberately non-causal
function is the forward return, which is the prediction target and must never appear on the
feature side.

1. **Order flow** — taker-buy ratio, signed order-flow imbalance, average trade size, plus
   rolling z-scores of imbalance, trade size and trade count. All derived from columns already
   present in the cached bar data that notebooks 001 and 002 never touched.
2. **Seasonality** — hour of day and day of week, encoded as sine/cosine pairs so a linear model
   doesn't treat hour 23 and hour 0 as far apart.
3. **Realised volatility** — rolling standard deviation of log returns at three windows (8, 24
   and 96 bars), volatility-of-volatility, and a short-over-long volatility regime ratio.
4. **Momentum and mean reversion** — cumulative log return over 1, 4 and 12 bars, and its
   negation. Carried over from notebook 002 as the baseline to beat.
5. **Funding rate** — joined onto bars with a backward as-of join, so a bar only ever sees
   funding that was published at or before its own close. Downloaded for all 30 symbols across
   the full period.

All features also have cross-sectionally demeaned and standardised variants, computed per
timestamp across the panel, so a fitted weight means the same thing regardless of an individual
symbol's scale.

Two tests guard this. **Causality under truncation:** recompute every feature on a truncated
history; every row that still exists must produce an identical value. If truncating the future
changes a past value, that feature was using the future. **A lookahead tripwire:** on synthetic
random-walk data with no implanted signal, no feature's correlation with next-bar return may
exceed 0.10 — notebook 001b's implausible Sharpe of 8.89 was exactly this kind of bug.

## How signals are screened

The screening metric is the **information coefficient**: the per-timestamp Spearman rank
correlation between a feature and the forward return, taken across symbols.

This is the right metric for a dollar-neutral book because it never compares one timestamp to
another. BTC and ETH moving together within a bar is absorbed inside that bar's single
correlation rather than counted as two independent observations.

Two corrections matter for the significance test:

- A feature built from a W-bar rolling window makes the IC series itself autocorrelated out to
  roughly W lags, so the naive standard error understates the noise. All t-statistics use a
  Newey-West (HAC) correction with the lag set to roughly each feature's own lookback.
- When correlations are stacked over every (symbol, bar) row rather than per timestamp, the
  rank products are first averaged within each timestamp — so a heavily populated bar doesn't
  outweigh a thin one, and symbols within a bar aren't treated as independent — before the same
  HAC machinery is applied.

Stability is tracked alongside magnitude: rolling mean IC, a per-year breakdown, and the fraction
of months with positive mean IC. Stability outranks magnitude for deciding what to trust.

### A bug found by running the screen

The zero-variance guard — skip a bar if the prediction or target has no cross-sectional spread —
tested for standard deviation *exactly* equal to zero. Floating-point summation leaves a residual
around 1e-16 on a genuinely constant cross-section, so the guard silently never fired and the bar
leaked through as a missing value instead of being skipped, corrupting the mean IC and t-statistic
for any affected feature. Fixed with a tolerance rather than exact equality. It was caught by the
screen producing an all-missing result for hour-of-day at daily bars, not by a unit test.

### A structural finding: seasonality is invisible to this method

Hour-of-day and day-of-week come back undefined at every interval, and that is correct rather
than broken. They are properties of the *timestamp*, identical for every symbol at a given bar, so
they have exactly zero cross-sectional variance by construction. Cross-sectional IC only measures
whether a feature ranks symbols correctly *against each other*; a market-wide seasonal effect, if
one exists, is invisible to it by design. It would show up in a directional strategy, not a
dollar-neutral one.

### What survived

**81 configurations were evaluated** (27 features × 3 intervals), and every one was logged, since
the deflation calculation later depends on the true count. **34 survived** a filter of
|t| > 3 together with a consistent sign across years. Momentum and mean reversion at a given
window are exact sign-flips of the same feature, so they always survive or fail together and are
listed once:

| Signal family | Best interval | Mean IC | t-stat (HAC) | % positive months | Notes |
|---|---|---|---|---|---|
| Mean reversion, 1 bar | 4h | +0.042 | 14.2 | 93.8% | Strongest and most stable found; survives at all three intervals |
| Realised vol (negative) | 4h | −0.038 | −11.4 | 8–17% | High recent volatility predicts lower forward return; survives at all three |
| Vol-of-vol, 96 bar (negative) | 4h | −0.028 | −8.9 | 12.5% | Survives at 4h and 12h |
| Mean reversion, 4 bar | 4h | +0.035 | 12.1 | 93.8% | Survives at all three |
| Mean reversion, 12 bar | 4h | +0.029 | 10.0 | 95.8% | Survives at all three |
| Funding rate (negative) | 4h | −0.0095 | −4.4 | 31.3% | Weak; survives at 4h only |

Everything else — order-flow imbalance, taker-buy ratio, average trade size, the volatility regime
ratio, the funding-rate z-score — fails at every interval on either significance, sign
consistency, or both.

The largest absolute mean IC anywhere in the 81 was 0.073, so **nothing tripped the 0.10 lookahead
tripwire**. The surviving families sit at 0.03–0.06, above the 0.01–0.03 range that's typical but
well clear of the tripwire. Both are well-documented effects (short-horizon mean reversion and
bid-ask bounce; a volatility-regime effect) rather than a red flag — but a signal this size is
exactly the case where the backtest decides whether it survives costs or is simply too fast and
too thin to trade.

## Portfolio construction

The pieces, each applied once and left alone rather than tuned per backtest:

- **Panel-aware walk-forward splitting.** Folds are computed over unique timestamps, not raw row
  positions. Splitting a stacked panel by row position would put some symbols' bar-*t* rows in
  training and others from the same instant in test.
- **A volatility-normalised target.** The model predicts forward return divided by that bar's
  realised volatility, so training loss stops being dominated by high-volatility eras (2022) at
  the expense of everything else.
- **Continuous, volatility-targeted sizing** rather than `sign(prediction)`. Because the model
  predicts a vol-normalised quantity, the prediction is already in units of "move per unit of that
  bar's own volatility", so clipping it to [−1, 1] is a natural risk cap — never bet more than a
  one-sigma-equivalent move — and scaling by target-vol over current-vol keeps realised risk
  roughly constant across symbols and regimes.
- **Dollar-neutral weights.** Each bar, rank symbols by prediction and take the top and bottom
  20% as the long and short legs. Because both legs only bet on relative ranking, whatever is
  common to the whole cross-section that bar — crypto beta — is stripped out. Weights within each
  leg are proportional to the vol-targeted size, normalised so the long leg sums to +0.5 gross and
  the short to −0.5, then each symbol is capped at 25% of the book. Capping can only shrink gross
  exposure, so no separate total-gross step is needed.

## Backtest configurations, declared before running

All three use one pooled linear model across all symbols (not one per symbol), fitted on
cross-sectionally standardised features, with a volatility-normalised target, 20% long/short
fractions, gross exposure 1.0, a 25% per-symbol cap, 4bp taker fee plus 1bp slippage, and rolling
walk-forward folds of roughly one year training to one quarter testing.

| # | Interval | Features |
|---|---|---|
| 1 | 4h | Mean reversion 1/4/12, realised vol 8/24/96, vol-of-vol 96, funding rate |
| 2 | 12h | Mean reversion 1/4/12, realised vol 8/24/96, vol-of-vol 96 |
| 3 | 1d | Mean reversion 1/4/12, realised vol 8/24/96, vol-of-vol 96 |

Each is also re-run at fold-grid start offsets of 0, 7, 14 and 21 days. Every one of those 12
evaluations counts as a separate trial for the deflation, alongside the 81 from screening.

### Results

The log ended with **95 trials, not 93** — two extra entries came from a debugging run. They were
counted anyway, since a fit-and-evaluate that happened is a trial that happened, bug or not.

Annualised Sharpe, net of costs:

| Config | Offset 0d | Offset 7d | Offset 14d | Offset 21d | Gross (offset 0) |
|---|---|---|---|---|---|
| 4h | −1.94 | −4.20 | −3.91 | −3.10 | +0.95 |
| 12h | **+0.42** | −2.45 | −1.12 | −0.96 | +1.32 |
| 1d | −0.29 | −1.15 | −1.66 | −0.45 | +0.43 |

The same origin instability notebook 002 found. The 12h configuration is the only positive
headline result, and it drops to −2.45 from moving the fold grid one week. The 4h and 1d
configurations are negative at every offset.

Crucially, **every configuration's gross Sharpe is positive** (0.43 to 1.32). The screened signal
is real enough to appear before costs; costs are what kill it. That is precisely the failure mode
the cost model was built to expose.

**Against baselines** (offset 0, each against its own basket):

| Config | Strategy net | Basket buy-and-hold | Always-short basket | Random ranking (200 seeds), mean / p90 |
|---|---|---|---|---|
| 4h | −1.94 | −0.06 | +0.06 | −10.40 / −9.55 |
| 12h | +0.42 | +0.05 | −0.05 | −3.51 / −2.79 |
| 1d | −0.29 | +0.00 | −0.00 | −1.61 / −0.93 |

The random baseline runs through the identical portfolio construction and cost model with random
instead of model-based ranking — a much stronger null than a coin flip. The real strategy beats it
comfortably every time. The model *is* doing something; it just isn't enough to profit net of
costs, except transiently at one configuration and offset. The basket's own Sharpe is near zero
over these windows, unlike the full-history buy-and-hold numbers in notebook 002 — the fair
comparison for a dollar-neutral book is a diversified basket, not a single levered directional
bet.

**Degeneracy:** 0 of 48 fold-fits collapsed to a constant directional bet. That failure mode is
ruled out; it doesn't rescue the Sharpe.

**Bootstrap 95% intervals on per-fold excess return over the basket** (offset 0): 4h
[−0.35, +0.07], 12h [−0.21, +0.30], 1d [−0.34, +0.28]. All three contain zero. Fold-by-fold win
rates against the basket were 27%, 45% and 60% — only the daily configuration beat its basket in a
majority of folds, and its Sharpe is still negative.

**Deflated Sharpe**, using the true 95-trial count: for the 12h configuration, the only positive
headline result, **P(true Sharpe > 0 | best of 95 trials) = 3.4%**. That is higher than notebook
002's 0.69%, but combined with the origin-shift sign flip and an interval containing zero, it
doesn't change the conclusion.

**Realised turnover cost** came in far below the illustrative worst case: roughly 1.5–2.1% a year
at 4h, 0.33–0.37% at 12h, 0.17–0.21% at 1d. A volatility-targeted, ranked long/short book trades
much less than a bar-by-bar sign-flipping strategy. Costs still flip the 4h and 1d configurations
from gross-positive to net-negative.

### A bug found during backtesting

Dividing forward return by realised volatility divides by exactly zero whenever a symbol's
trailing volatility is 0 — a frozen or pinned price. The clearest case is LUNA's final bars after
the Terra collapse, where the symbol stayed listed at a near-zero, barely-moving price. This
produced infinite training targets that corrupted training silently until every fold Sharpe came
back undefined. Fixed by dropping bars with essentially zero realised volatility (2.7–2.8% of
rows), which carry no usable vol-normalised signal anyway.

## The holdout, spent once

The 12h configuration — the highest headline net Sharpe — was run completely unchanged on a panel
extended through 2026-07-01. Same features, same hyperparameters, same fold grid, same training
length. Reusing the same grid means the fold boundaries simply continue forward; only folds whose
*entire* test window falls at or after 2025-07-01 were evaluated.

That yields **three folds fully inside the holdout**, covering 2025-08-17 to 2026-05-17 (546
bars). The grid's fixed 91-day step doesn't tile the holdout year exactly, so a stretch at each
end falls outside any complete fold. Reported as-is rather than re-gridded to force full coverage.

| Metric | Value |
|---|---|
| Sharpe, net of costs | **−0.47** |
| Sharpe, gross | +0.74 |
| Basket buy-and-hold, same window | −1.79 |
| Random ranking (200 seeds), mean / p90 | −4.17 / −2.75 |
| Folds beating the basket | 2 of 3 |
| Degenerate fits | 0 of 3 |
| Bootstrap 95% CI, excess return vs basket | [−0.17, +0.50] |

The holdout year is **negative net of costs**, consistent with everything above. Gross is positive
again (+0.74) — the same "signal is real pre-cost, costs erase it" pattern as every other result
here. The basket itself had a rough year and the random baseline was worse still, so the strategy
did beat both naive alternatives in relative terms; it simply didn't clear zero after paying for
itself. The interval on excess return contains zero, as everywhere else, though the point estimate
happens to be positive this time — a reminder of how wide these intervals are on three folds.

## Reproducibility check, and what it revealed

Two inference assumptions were revisited: normality in the deflated Sharpe, and independent
resampling in the bootstrap.

Reproducing the backtest turned up a defect in the analysis itself. The deflated Sharpe values
above were computed by hand from the logged Sharpe and observation counts, and the per-bar net
return series the moment corrections need was never saved. Worse, the model fitting was never
seeded, so re-running an identical configuration is methodologically identical but not a bit-exact
replay — unlike notebook 002, which reproduced exactly.

**Re-running the three configurations produced different headline numbers.** Same code, same data,
same folds, different random initialisation:

| Config | Original net Sharpe | Rerun net Sharpe |
|---|---|---|
| 4h | −1.94 | −2.32 |
| 12h | **+0.42** | **−1.22** |
| 1d | −0.29 | −1.16 |

The 12h configuration — the only one ever net-positive at its headline setting — **flips negative
on a fresh run of the exact same configuration.** It had already failed the origin shift, the
bootstrap interval and the deflation; now it also fails to reproduce its own headline number. That
is a second, independent route to the same conclusion.

**Deflated Sharpe under real moments**, 95 trials throughout:

| Config | Skew | Kurtosis | Normal assumption, original Sharpe | Real moments, rerun Sharpe |
|---|---|---|---|---|
| 4h | +6.25 | 189.3 | 0.0000005% | 0.00000050% |
| 12h | +0.26 | 11.7 | 3.4% | 0.00032% |
| 1d | −0.44 | 10.1 | 0.15% | 0.00065% |

The 4h configuration's kurtosis of 189 reflects a handful of extreme fold-level blowups in a book
rebalanced every four hours. Nothing here is normally distributed, and the deflated probability is
unmeasurably small under either assumption.

Those two columns confound two separate effects — the moment correction and the rerun's worse
starting Sharpe. Isolating the moment correction alone, holding the rerun's Sharpe fixed: 4h moves
from 0.0000000105% to 0.0000005%, 12h from 0.00030% to 0.00032%, 1d from 0.00071% to 0.00065%.
Real moments move the number by less than an order of magnitude in every case, and the direction
isn't consistent — it depends on the sign of skew relative to the sign of the Sharpe, not on a
uniform "fat tails always look worse".

**Bootstrap intervals**, original versus rerun. The block-length heuristic selected length 1 for
all three — the same finding as notebook 002, since these 10–11-point fold series don't show enough
autocorrelation to warrant longer blocks, so the block and independent bootstraps agree:

| Config | Original | Rerun |
|---|---|---|
| 4h | [−0.35, +0.07] | [−0.30, −0.03] |
| 12h | [−0.21, +0.30] | [−0.28, +0.12] |
| 1d | [−0.34, +0.28] | [−0.33, +0.17] |

The 4h interval no longer contains zero — but it is entirely *negative*, meaning this run is
confident the configuration underperforms its basket. The opposite of an edge.

## Bottom line

**No validated edge, holdout included.** Every stage told the same story: a statistically real if
modest mean-reversion and volatility signal exists in the cross-sectional panel and is
gross-profitable at every interval and in the holdout year, but transaction costs consistently
erase it; the one configuration that cleared costs at its headline setting didn't survive a
one-week shift in the fold grid, didn't survive deflation for search width, and didn't even
survive being run again; and the holdout, spent once with no retuning, came back at −0.47.

The two methodological fixes — breadth through a cross-sectional book, and honest costs — were
both real improvements. Neither produced a tradeable edge.

## What to test next

- **Funding rate at scale.** It only weakly and narrowly survived screening here (4h only,
  |t| = 4.4). Carry is historically the most robust crypto signal, and a dedicated study using
  the full funding history with proper carry construction — funding-weighted basis trades rather
  than one cross-sectional feature among many — could do considerably better.
- **Slower rebalancing.** The 4h and 1d configurations were gross-positive and lost to costs; the
  middling-turnover 12h one came closest. Deliberately lower-turnover variants (rebalance every N
  bars, or widen the no-trade band) are more promising than more intervals.
- **Seasonality, measured properly.** It is structurally invisible to cross-sectional IC. If it is
  real, it needs a directional backtest, which is a different validation path entirely.
- **Regime-conditioning.** Both surviving families are volatility-related. Whether they are
  specifically a high-volatility phenomenon rather than a stable year-round effect is answerable
  directly from the per-year breakdown already logged.
- **Maker execution.** All costs here assume taker fees. If maker fills are realistic at this
  turnover, the economics could look different — worth quantifying rather than assuming.

*Notebook: `src/research/003_cross_sectional_ic.ipynb`.*
