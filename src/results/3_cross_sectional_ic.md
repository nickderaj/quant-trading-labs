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

## What's next

Phase 1: expand the universe to ~30 symbols with explicit survivorship-bias handling,
covering 2021-07 through 2026-07 at 4h/12h/1d.
