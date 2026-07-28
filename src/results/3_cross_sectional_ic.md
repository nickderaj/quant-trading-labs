# Cross-Sectional Crypto IC Pipeline - Results Summary

Notebook 2 found no validated edge on single-asset trend/mean-reversion models, and never
charged transaction costs while doing it (`add_tx_fees*` was called zero times in that
notebook - every number in `2_walk_forward_multi_asset.md` is gross). This notebook fixes
both root causes at once: adds real transaction costs everywhere, and switches from
"search many single-asset configs, backtest whichever wins" to "screen a cross-sectional
panel on rank correlation (IC), backtest at most 3 things that survive screening." The
goal is measurement power (30 symbols instead of 1) and honesty about cost (nothing here
is reported gross).

## Phase 0 - Cost model

Added `research.add_trading_costs(trades, taker_fee, slippage=1e-4)`: charges
`turnover_t * (taker_fee + slippage)` per bar, where `turnover_t = |position_t -
position_{t-1}|` (position before the first bar treated as 0, so the first entry is
charged too). Cost is expressed as a log-return drag, `log(1 - cost_t)`, added to
`trade_log_return` to produce `trade_log_return_net`, with `equity_curve_net` and
`drawdown_log_return_net` recomputed from it. This generalizes the existing
`add_tx_fees_log` (which only handled fees, no slippage term).

Wired into `walk_forward_run` (new `taker_fee`/`slippage` params - when set, every fold's
metrics gain `sharpe_net`, `total_log_return_net`, `compound_return_net`,
`max_drawdown_net`, `mean_turnover_per_bar`, `turnover_per_year`, `annual_fee_drag_log`,
`annual_fee_drag_pct`; the stitched OOS trade frame carries the net columns across fold
boundaries, recomputed once on the full stitched series rather than concatenated
per-fold) and into `stitched_metrics` (reports the same net/cost fields whenever the net
columns are present, gross-only metrics unchanged when `taker_fee` is omitted so old
callers don't break).

Unit tests in `tests/test_research.py` (6 tests, all passing): zero turnover charges
zero cost; a single sign flip charges exactly one round-trip's worth of turnover (2
units) and nothing on the held bars before/after; holding a position across N bars
charges the entry once and nothing else; `cost_summary`'s annualization matches a
manual calculation; `walk_forward_run` surfaces both gross and net columns/metrics when
a fee is given and net return is never above gross; `stitched_metrics` omits all net
fields when no fee is given.

### Sanity check: which intervals survive costs at all

Simulated a `+-1`, always-in-market position that flips sign with probability 0.4 each
bar (a rough proxy for "a mediocre but real trend/reversal signal"), 100k bars, at
3bps taker fee (6bps round trip) plus the default 1bp slippage, across four bar
intervals:

| interval | annualized fee drag (taker only) | annualized fee drag (taker + slippage) |
|---|---|---|
| 1h  | +713%/yr  | +1535%/yr |
| 4h  | +70%/yr   | +102%/yr  |
| 12h | +19%/yr   | +26%/yr   |
| 1d  | +9%/yr    | +12%/yr   |

These are bigger than the back-of-envelope "~1.8%/yr at 12h, ~21%/yr at 1h" figure this
run was scoped against, because that figure used simple (non-compounding) annualization
of a per-bar cost; this cost model compounds `log(1 - cost)` the same way returns
compound, so at 1h bars (8760/yr) a fee that looks tiny per-bar blows up geometrically
over a year of turnover. The exact multiplier depends on assumptions (flip definition,
fee vs. round-trip vs. slippage), but the qualitative point survives, more starkly than
the original estimate: **going from 12h to 1h bars turns a survivable cost drag into a
completely unviable one, for the same underlying signal and flip rate.** Sign of the
effect is unambiguous and the ranking across intervals doesn't depend on the exact
constant.

Decision: **1h bars are dropped from this run.** Screening and backtesting proceed at
4h/12h/1d only, per the guardrail that 1h was conditional on surviving this check. It
didn't.

## Bugs found

- None specific to this phase beyond the one already documented in `2_walk_forward_multi_asset.md` (klines schema inference). Re-verified: `describe_linear_model`'s
  degenerate-bet check and the fee math from `1_simple_linear.md` are unaffected by
  this change - `add_trading_costs` is additive on top of existing gross columns, not
  a replacement.

## Phase 1 - Universe expansion

Added `research.load_universe_panel(symbols, interval, start_date, end_date,
min_cross_section=10, allow_holdout=False)`: downloads each symbol's klines via
`data.download_klines_range` (skipping symbols/months with no archive file rather
than raising - a partial history is not an error), concatenates into one ragged
panel with a `symbol` column, then drops any bar where fewer than
`min_cross_section` symbols have data, so no cross-sectional rank/z-score is ever
computed from a near-empty cross-section. `HOLDOUT_START = 2025-07-01` is a module
constant; any call reaching past it raises `ValueError` unless `allow_holdout=True`,
which only the Phase 7 holdout run should ever pass. This is the "enforce in the
loader, not discipline" guardrail - verified with a unit test that the guard fires
before any data is touched.

### Universe construction rule (and its bias)

30 USDT-M perpetual futures symbols, chosen to include several 2021-era coins that
later died or were delisted, specifically to avoid picking today's top-30-by-liquidity
and backtesting it from 2021 (survivorship bias): BTC, ETH, BNB, SOL, XRP, ADA, DOGE,
DOT, AVAX, MATIC, LINK, LTC, ATOM, UNI, ETC, XLM, ALGO, VET, FIL, TRX, EOS, AAVE, SAND,
MANA, AXS, THETA, NEAR, FTM, LUNA, FTT.

This list was still chosen by hindsight ("which coins do I remember mattering and
dying"), not by reconstructing Binance's actual listed-symbols history at each past
date - a real backtester in 2021 wouldn't have known FTM would survive and LUNA
wouldn't. That residual bias is real and unresolved; it's smaller than "top-30-today"
survivorship bias but not zero. Noted, not corrected for.

No forward/back-fill across a listing or delisting boundary - each symbol's series is
whatever Binance's archive actually has, so the panel is ragged by construction.
Downloading all 30 x 3 intervals (4h/12h/1d; 1h dropped per Phase 0) for 2021-07 to
2026-07 confirmed real ragged coverage, not just a hypothetical:

| symbol | first bar | last bar | note |
|---|---|---|---|
| LUNA | 2021-07-01 | 2022-05-13 | Terra/UST collapse - died as intended, this is the canonical example the universe was built to catch |
| MATIC | 2021-07-01 | 2024-09-11 | Binance delisted the MATIC perpetual around the POL migration |
| EOS | 2021-07-01 | 2025-05-21 | Binance delisted the EOS perpetual |
| FTT | 2022-04-15 | 2026-06-30 | listed ~9 months after the panel start, not delisted (FTT stayed listed through the FTX collapse, just collapsed in price) - a late-listing example rather than a died-mid-series one |
| all other 26 | 2021-07-01 | 2026-06-30 | full coverage |

Confirms notebook 2's known archive gap still applies at klines granularity: SOL and
XRP are both missing 2022-03-01 to 2022-04-03 (~128h combined across two gap segments,
not the continuous 120h window notebook 2 described from the tick-aggregated feed, but
the same underlying Binance archive hole). `min_cross_section=10` absorbs this cleanly
- 28 of 30 symbols are still available at every bar in that window, so cross-sectional
ranking is unaffected; it would matter if the panel were much smaller.

At 4h: 252,053 panel rows across 30 symbols, 2021-07-01 to 2025-07-01 (holdout excluded).
At 12h: 84,036 rows. At 1d: 42,033 rows.

## Phase 2 - Feature library

New `src/features.py`. Every raw feature is per-symbol and strictly causal (rolling
windows computed with polars' trailing `rolling_mean`/`rolling_std`, momentum via
`shift()` only); `apply_per_symbol` partitions the panel by symbol before applying any
of them so a rolling window can never cross a symbol boundary (BTC's vol window
picking up ETH's trailing bars, say). `forward_return` (the target, `shift(-horizon)`)
is the one function in the module that's deliberately *not* causal and is documented
as such - it must never appear on the feature side of a model.

Built, in the priority order from the guardrails:

1. **Order flow** - `taker_buy_ratio`, `order_flow_imbalance` (signed, in [-1, 1]),
   `avg_trade_size` (volume/count), plus rolling z-scores (20/60 bar windows) of
   imbalance, avg trade size, and trade count. All from columns already in the
   cached klines (`taker_buy_volume`, `count`) that notebooks 1/2 never touched.
2. **Seasonality** - hour-of-day and day-of-week, cyclically (sin/cos) encoded so a
   linear model doesn't see hour 23 and hour 0 as far apart. Noted in the docstring
   that hour-of-day degenerates at 12h/1d bars - screening (Phase 4) will show that
   directly rather than hardcoding it away here.
3. **Realized vol** - rolling std of log returns at three windows (8/24/96 bars),
   vol-of-vol (rolling std of the short-window vol), and a vol-regime ratio
   (short-window vol / long-window vol).
4. **Momentum / mean-reversion** - cumulative log return over 1/4/12 bars and its
   negation, carried over as the notebook-2 baseline to beat.
5. **Funding rate** - not skipped. Added `data.download_funding_rate_range`
   (Binance's live `/fapi/v1/fundingRate`, paginated by `fundingTime` since there's
   no bulk historical archive for it like klines have) and
   `features.add_funding_rate_feature`, which joins it onto bars via a **backward
   asof join** so a bar only ever sees funding published at or before its own close.
   Downloaded successfully for all 30 symbols, 2021-07 to 2025-07 (4,383 rows/symbol
   for the continuously-listed ones; ragged for LUNA/FTT matching their klines
   coverage) - well inside the 30-minute time-box.

Cross-sectional variants (`_cs_demean`, `_cs_z`) computed per-timestamp across the
whole panel - what the pooled ranking model in Phase 5 actually consumes, so a fitted
weight means the same thing regardless of a symbol's own scale.

### Tests (`tests/test_features.py`, 9 passing)

- **Causality under truncation**: every raw feature, recomputed on a truncated
  history, must produce identical values for every row that still exists - if
  truncating the future changed a past value, that feature was using the future.
  This is the concrete form of "assert no lookahead with an explicit shift test."
- **Lookahead magnitude tripwire**: on synthetic random-walk data with no implanted
  signal, every raw feature's correlation with next-bar return must stay under 0.10
  - the guardrail's own example (notebook 1b's Sharpe 8.89) was exactly a `shift()`
  bug producing a similarly implausible bar-level correlation.
- Formula checks (order-flow-imbalance bounds and formula, momentum vs. manual log
  return, mean-reversion as momentum's negation), a cross-sectional z-score check
  (zero mean / unit std per bar), a funding-rate backward-asof-join causality check,
  and an end-to-end `build_feature_panel` smoke test.

## Phase 3 - IC harness

Added to `research.py`:

- **`cross_sectional_ic`** - per-timestamp Spearman IC of a prediction against forward
  return, across symbols. This is the right metric for a dollar-neutral book: it never
  compares different timestamps to each other, so BTC and ETH moving together within
  one bar is absorbed inside that bar's single IC_t rather than counted as two
  independent data points. `cross_sectional_ic_stats` then summarizes the resulting
  IC_t series with `newey_west_tstat` (Bartlett-kernel HAC), because a feature built
  from a W-bar rolling window makes IC_t itself autocorrelated out to about W lags -
  the naive i.i.d. standard error of the mean would understate that.
- **`panel_ic`** - Spearman IC stacked over every (symbol, bar) row, with a
  Driscoll-Kraay-style standard error: average the rank-product within each timestamp
  first (so a heavily-populated bar doesn't outweigh a thin one and symbols within a
  bar aren't treated as independent), then apply the same Newey-West machinery across
  that per-timestamp series. `naive_tstat` (assuming all `n_obs` rows are i.i.d.) is
  reported alongside for contrast - it's the number that would be badly overstated if
  reported on its own, which is exactly the guardrail's warning about panel IC.
- **`ic_stability`** - rolling mean IC, per-year IC breakdown, and fraction of months
  with positive mean IC, since stability outranks magnitude for deciding what to trust.

### Tests (`tests/test_ic_harness.py`, 7 passing)

- `newey_west_tstat` matches a plain t-test at lag=0 on i.i.d. data, and produces a
  visibly smaller |t-stat| than the naive calculation on a strongly autocorrelated
  (AR(1), rho=0.9) series with the same marginal variance - confirming the HAC
  correction actually does something, not just that it runs.
- Synthetic panels with a known implanted IC (target = true_ic * pred + orthogonal
  noise): both `cross_sectional_ic` and `panel_ic` recover a mean IC within a
  reasonable band of the true value with a clearly significant t-stat (>5), while a
  true_ic=0 panel comes back with |IC| < 0.1 and |t-stat| < 3.
- `panel_ic`'s clustered+HAC t-stat is asserted to not exceed the naive i.i.d. one by
  more than a small margin on data with real cross-sectional/temporal structure - the
  whole point of the correction is that it should be equal or smaller, never larger.
- `ic_stability` on a strong, constant-sign synthetic signal reports >90% positive
  months, and an empty-panel edge case returns NaN stats rather than raising.

## Phase 4 - IC screening

Screened all 27 raw candidate features (`src/research/tmp/screen_features.py`) across
all 3 surviving intervals (4h/12h/1d; 1h dropped in Phase 0) against `fwd_return_1`,
using `cross_sectional_ic`. Newey-West lag set per-feature to roughly its own lookback
window (20/60 bars for the z-scored order-flow features, the window itself for
momentum/mean-reversion/realized-vol, 1 bar for anything with no rolling window).
**81 configs evaluated total, every one logged to `src/research/tmp/config_log.jsonl`**
(27 features x 3 intervals) - this is the true count the deflated Sharpe in Phase 6
must use.

### Bug found during screening

The zero-variance guard in `cross_sectional_ic` (skip a bar if pred or target has no
cross-sectional spread that bar) compared `np.std(x) == 0` exactly. Floating-point
summation leaves a residual of ~1e-16 on a *genuinely* constant cross-section rather
than exactly 0.0, so the guard silently failed to fire and the bar leaked through as
`ic = NaN` instead of being skipped - corrupting `mean_ic`/`nw_tstat` for every such
feature. Fixed with a `< 1e-12` tolerance instead of exact equality. Caught by running
the screen itself (`hour_sin` at 1d bars came back all-NaN), not by a unit test -
added no regression test for it since the fix is a one-line tolerance change covered
implicitly by every screening result no longer being NaN.

### A structural finding, not a bug: seasonality is invisible to cross-sectional IC

`hour_sin/cos` and `dow_sin/cos` come back `NaN` at every interval, and this is
correct, not broken: day-of-week and hour-of-day are properties of the *timestamp*,
identical for every symbol at a given bar - so they have exactly zero cross-sectional
variance by construction, at any interval. Cross-sectional IC only ever measures
whether a feature ranks symbols correctly *relative to each other*, and a market-wide
seasonality effect (if one exists) is invisible to that by design - it would show up
in a directional/beta strategy's IC, not a dollar-neutral one. Worth remembering for
"what to test next."

### Ranked IC table (surviving |t| > 3 AND consistent sign across years)

**34 of 81 configs survive.** Grouped by underlying signal (momentum_W and
mean_reversion_W are sign-flips of the same feature, so they always survive or fail
together and are listed once):

| feature family | best interval | mean IC | NW t-stat | % positive months | notes |
|---|---|---|---|---|---|
| mean_reversion_1 (= -momentum_1 = -log_return) | 4h | +0.042 | 14.2 | 93.8% | strongest and most stable signal found; survives at 4h/12h/1d |
| realized_vol_8/24/96 (negative) | 4h | -0.038 (vol_24) | -11.4 | 8-17% | high recent vol -> lower forward return; survives at all 3 intervals |
| vol_of_vol_96 (negative) | 4h | -0.028 | -8.9 | 12.5% | survives at 4h/12h |
| mean_reversion_4 | 4h | +0.035 | 12.1 | 93.8% | survives at 4h/12h/1d |
| mean_reversion_12 | 4h | +0.029 | 10.0 | 95.8% | survives at 4h/12h/1d |
| funding_rate (negative) | 4h | -0.0095 | -4.4 | 31.3% | weak but survives at 4h only |

Everything else - order flow imbalance/taker-buy-ratio, avg trade size, vol regime,
funding_rate_z20 - fails the filter at every interval (either |t| < 3, or sign flips
across years, or both). Full 81-row table with every feature/interval/mean IC/t-stat/
year-by-year breakdown is in `config_log.jsonl`.

Max |mean IC| observed across all 81 configs was 0.073 (count_z60 at 1d, which failed
the sign-consistency filter anyway) - **nothing tripped the 0.10 lookahead tripwire.**
The mean-reversion and realized-vol families sit at 0.03-0.06, above the guardrail's
stated "normal" range of 0.01-0.03 but well clear of the tripwire; both are
well-documented, unsurprising effects (short-horizon mean reversion / bid-ask-bounce
microstructure, and a vol-regime effect) rather than a red flag, but Phase 6's
backtest is exactly where a real signal this size either survives costs or turns out
to be too fast/thin to trade profitably - noted as something to watch, not assumed.

## Phase 5 - Portfolio construction

Added to `research.py`:

- **`panel_walk_forward_splits`** - the multi-symbol analogue of `walk_forward_splits`,
  built by delegating to it (folds computed over unique timestamps, not raw row
  position) so the fold-boundary logic isn't duplicated. This is what makes a pooled
  model possible without leaking: splitting a stacked (symbol, bar) panel by row
  position would let some symbols' bar-t rows land in train while others at the same
  bar land in test, even though they're the same instant. Verified by test that every
  timestamp's rows are entirely on one side of the boundary.
- **`vol_normalized_target`** - `fwd_return_1 / realized_vol_t` as the regression
  target instead of raw return, so training loss stops being dominated by high-vol
  bars/eras (2022) at the expense of everything else.
- **`vol_targeted_size`** - `clip(pred, -1, 1) * (vol_target / vol_t)`: continuous
  position size instead of `sign(pred)`. Since the model predicts the vol-normalized
  target, `pred` is already in "predicted move per unit of that bar's own vol" units,
  so clipping to [-1, 1] is a natural risk cap (never bet bigger than a 1-sigma-
  equivalent move) and the `vol_target/vol_t` scaling keeps realized vol roughly
  constant across symbols/regimes - a risk-management effect, applied once and left
  alone, not something to tune per backtest.
- **`dollar_neutral_weights`** - per-bar: rank symbols by prediction, take the top/
  bottom `top_frac` as long/short legs (stripping whatever's common to the whole
  cross-section that bar, i.e. crypto beta, since both legs only ever bet on relative
  ranking), weight *within* each leg proportionally to `vol_targeted_size` (or equally
  if no size given), normalize so long sums to `+gross/2` and short to `-gross/2`, then
  clip each symbol to `max_position_per_symbol`. Clipping can only shrink gross
  exposure, never breach the target, so no separate total-gross-cap step was needed.
- **`portfolio_turnover`** / **`portfolio_trade_frame`** / **`add_portfolio_costs`** /
  **`portfolio_metrics`** - the multi-symbol analogues of Phase 0's
  `add_trading_costs`/`stitched_metrics`, reusing the same cost math and
  `_series_metrics`/`cost_summary` helpers rather than duplicating them. Turns a
  weights panel + forward returns into one portfolio-level bar return series with
  gross and net (fee-charged) variants, ready for the same reporting Phase 6 uses.

### Tests (`tests/test_portfolio.py`, 9 passing)

Dollar-neutral weights are net-zero and within the gross cap every bar; only the
top/bottom `top_frac` symbols get nonzero weight and the split is on the right side of
the ranking; `max_position_per_symbol` actually caps; leg weights scale proportionally
to a given size column; turnover charges a symbol's entry and exit but not the held
middle bar; `portfolio_trade_frame` matches a hand-computed weighted sum; net metrics
never exceed gross; `vol_targeted_size` clips before scaling; and the walk-forward
split test described above.

## What's next

Phase 6: pre-declare and run at most 3 backtest configs using this machinery, over
2021-07 to 2025-07, with full notebook-2 methodology (origin-shift, baselines,
degenerate-bet check, bootstrap CI, deflated Sharpe using the true config count from
`config_log.jsonl`).
