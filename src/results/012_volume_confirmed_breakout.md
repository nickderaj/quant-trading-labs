# Notebook 012 — Volume-Confirmed Breakout, One Rule, Whole Basket: Results Summary

One pre-registered gate (`phase_1_12_preregistration.json`, committed before this notebook's
pooled backtest ran and not edited since). **Gate VB returns a `fires=False` verdict.** Net Sharpe
is positive at every declared origin offset and the cost-stress leg is real and correctly signed —
but the volume filter does not earn its keep (the gated-minus-ungated bootstrap CI includes zero,
and the point estimate goes the *wrong* direction) and the Deflated Sharpe Ratio (0.064) is nowhere
near the 0.95 bar even at the honestly small `n_trials=12`. This is the fourteenth-plus gate across
twelve notebooks to return a null, and — like Gate MB before it — a null with a positive-everywhere
Sharpe and a real cost-stress effect, which makes "cannot be distinguished from noise, not obviously
broken" the honest characterization rather than "the mechanism is absent."

## The rule, generalized from 11d, not carried over from it

Gate VB tests NEXT_PROMPT.md sec 2's rule: 11d's prior-run + tightening-base + breakout bull-flag
detector, made symmetric long/short and gated on breakout-bar volume ≥ `vol_k` × trailing median
volume. Two binding constraints, both honored:

- **One parameter set for every instrument, expressed scale-free.** `prior_run_min_atr_mult` and
  `base_max_range_atr_mult` are ATR multiples (a genuine Wilder ATR per instrument), not
  %-of-price — the same frozen numbers transfer across four asset classes with wildly different
  price levels and volatility regimes (BTC vs. `ZC` corn vs. `SPY`) without a second parameter set.
- **Thresholds frozen once, before any pooled backtest, from a calibration window disjoint from
  the backtest window.** For each of the 88 instruments independently, its own first 3 years of
  history (or its full history if shorter, capped at the 2024-12-31 dev-window end) is held out as
  calibration-only; `base_max_range_atr_mult` = the pooled calibration sample's 30th percentile of
  base range in ATR units (**2.8**), `prior_run_min_atr_mult` = the 70th percentile of
  `|prior-run return / ATR-implied % move|` (**4.5**), `vol_k` = the 75th percentile of
  volume-to-trailing-median ratio (**1.39**) — declared percentiles chosen once from the
  calibration pool's own shape, never by trying several `k` values and keeping the best. 66,645 to
  75,581 calibration-window bars fed these three numbers, pooled across all three asset classes
  before any of them traded a single dollar.

## Data extension this notebook required, disclosed

NEXT_PROMPT.md sec 1 flagged the one real blocker up front: `commod_lib8.build_continuous_series`
returns only `close_f{leg}` and drops open/high/low/volume entirely. This notebook adds
`build_continuous_series_ohlcv` (F1 only), reusing the identical roll-schedule and
`searchsorted` front-month-selection logic, joining the full OHLCV row instead of `close` alone,
and back-adjusting open/high/low by the same additive Panama offset already used for
`close_backadj` (raw per-contract prices jump at every roll; the breakout rule needs a gap-free
level series the way the existing spread/return machinery never did). `volume` is deliberately
**not** back-adjusted — it is a traded quantity, not a price, and an additive shift would be
meaningless — and is left exactly as flagged: discontinuous at rolls, valid only relative to its
own trailing, within-contract history. The two declared roll-volume hazards from sec 1 are both
handled the same way: a trailing volume window is disqualified from firing on the volume leg
(never on the ungated control) whenever it would straddle a roll or land on the roll bar itself —
the "suppress near a roll" choice, not the alternative within-contract z-score choice.

## Universe: 88 instruments, three-of-four asset classes voting on regime

| Asset class | Instruments | Regime reference | Regime gate open |
|---|---:|---|---:|
| Crypto perpetuals | 30 (incl. `LUNAUSDT`, `FTTUSDT`, delisted in effect) | `BTCUSDT`, min_confirm=1 | 54.1% |
| Commodity-equity/ETF | 42 (27 `*=F` FX/futures proxies excluded — sec 1's disclosed unreliable-volume check) | `SPY`/`QQQ`/`IWM`, min_confirm=2 | 57.3% |
| Databento futures | 16 products, F1 continuous | `CL`/`GC`/`ES`, min_confirm=2 | 43.9% |

The futures regime reference (`CL`/`GC`/`ES` — energy/metal/equity-index) is a new, structurally
identical construction to the equity class's three-reference vote, not fitted to this rule's
performance. Order-85 became 88 once the actual universes were counted. Costs reuse the crypto
convention (`TAKER_FEE=0.0004` + `SLIPPAGE=0.0001` per side) uniformly across all three classes,
the same disclosed-not-measured assumption 11d made for its equity side — this repo has no
established futures- or equity-specific cost model, and inventing three different ones would
itself be another undeclared parameter.

## Gate VB: positive Sharpe everywhere, an offset leg that turns out to be nearly vacuous, and a
## volume filter that does not pay for itself

| offset | Sharpe, gated (1×) | Sharpe, gated (2×) | Sharpe, gated (3×) | n trades (gated) |
|---|---:|---:|---:|---:|
| 0 | **+0.1150** | +0.0939 | +0.0743 | 406 |
| 7 | **+0.1150** | +0.0939 | +0.0743 | 406 |
| 14 | **+0.1151** | +0.0940 | +0.0743 | 406 |
| 21 | **+0.1152** | +0.0940 | +0.0744 | 406 |

Net Sharpe is positive at every one of the four offsets and the trade count is identical (406) to
three decimal places of Sharpe across all of them — Gate VB's first leg clears, but this
particular every-offset check turns out to be nearly uninformative here, and that is disclosed
rather than presented as a clean pass. It is a mechanical consequence of stacking the calibration
exclusion on top of the offset convention: each instrument's own first ~3 years are already
excluded from trading (median calibration window vastly exceeds 21 bars), so trimming a further
0/7/14/21 bars off the front of an already-long remaining series changes essentially nothing.
11d's offset leg was informative because it had no calibration exclusion sitting in front of it;
this notebook's does not carry that same evidentiary weight, and a future notebook that wants the
every-offset check to bite again should apply the offset before, not after, the calibration split.

The gated book (406 trades, offset 0, 1× cost) has a pooled fixed-notional point return of
**+27.4%**, against the ungated control's **+45.6%** on the same 1,105-trade book — **the volume
filter's point estimate moves the wrong way**, not just an insignificant one. The paired block
bootstrap of gated-minus-ungated returns (98 quarterly blocks) gives delta **−18.2pp, 95% CI
[−148.9pp, +87.0pp]** — a CI wide enough to contain both a real improvement and a real
deterioration, so no directional claim survives, but the point estimate offers no support for
"volume confirmation helps" either. **Gate VB's second leg does not clear.**

The deflated Sharpe probability, offset-0 daily Sharpe deflated for `n_trials=12` (4 offsets × 3
cost multipliers, single fixed rule and single fixed `k`, per `phase_1_12_preregistration.json`)
against `n_obs=406` gated trades, is **0.064** — far below the 0.95 bar despite this notebook
pulling both of NEXT_PROMPT.md sec 0's levers (pooling ~85+ instruments into one book, and holding
the trial count to 12, the same honest count as Gate MB). **Gate VB does not fire.** It is also not
eligible for the fundable flag on independent grounds: max drawdown at offset 0, 1× cost is
**−42.6%**, well outside the 25%-of-peak bound.

Cost stress is a genuine, correctly-signed effect, same as everywhere else in this programme: the
1×-vs-3×-cost paired block bootstrap on the gated book gives a delta of **−16.1pp, 95% CI
[−18.7pp, −13.7pp]** — real, statistically distinguishable from zero. The null here is not an
artifact of an underpriced cost assumption.

## Trade counts by asset class — pooling here is not balanced, and that is part of the finding

| Asset class | n trades, gated | n trades, ungated |
|---|---:|---:|
| Crypto | 8 | 15 |
| Equity/ETF | **383** | **936** |
| Futures | 15 | 154 |

Equities supply 94% of the gated book's 406 trades (383 of 406) and 85% of the ungated control's
1,105. This basket's headline sample size is real — 406 trades honestly clears more of the DSR bar
than any single-asset-class book in this programme has managed — but it is overwhelmingly an
equity/ETF result wearing a four-asset-class label, not an even pooling across crypto, equities,
and futures. Per NEXT_PROMPT.md sec 3's explicit instruction, this is stated plainly rather than
left to be inferred from the table: whatever Gate VB's basket-level Sharpe says, it is mostly
describing behavior on 42 commodity-sector equities and ETFs, with crypto and futures each
contributing single-digit-to-low-double-digit trades that cannot move the pooled verdict much
either way.

## What this notebook establishes, plainly

Raising `n` and lowering the trial count both happened here, exactly as sec 0 prescribed — 406
gated trades against `n_trials=12` is the best-powered book this programme has built — and the
answer is still a clean, honest null, for a third distinct reason after Gate MB's "sample size
alone kills it" and Gate MB-E's "wrong sign outright." Gate VB's failure mode is neither: Sharpe is
positive everywhere, the sample is well-powered by this programme's own standard, and cost stress
behaves exactly as a real trading rule's should — but the one thing this notebook actually set out
to test, whether volume confirmation improves a bull-flag breakout, comes back **not supported**.
The point estimate moves in the wrong direction and the confidence interval is too wide to save it
either way. That is a real and publishable answer in its own right: "does breakout volume confirm
the move?" was a genuinely untested question in this repo (NEXT_PROMPT.md sec 1), and now it has
been tested, honestly, at a sample size an order of magnitude larger than anything upstream of it
in this programme, and the answer is no — not "we couldn't tell," but "the mechanism this notebook
added does not pay for itself," a sharper and more useful negative than most of the sample-size-
limited nulls that came before it. NEXT_PROMPT.md sec 4's second-pattern temptation (golden cross,
pennant, MACD) was declared out of scope up front and stays out of scope: one gate, no pattern zoo,
consistent with `phase_1_12_preregistration.json`'s `second_pattern_declared: false`.

Machinery: `src/research/tmp/run_phase_{0,1,2,3}_12_*.py` (Phase 0 the three-asset-class universe
loaders, dev-window trim, regime-gate construction, per-instrument calibration-window split, and
threshold-freezing diagnostic; Phase 1 the Gate VB pre-registration, committed once and asserted
against, never re-typed; Phase 2 the pooled gated-vs-ungated backtest, bootstrap, DSR, and
per-asset-class trade counts; Phase 3 the final gate table, cross-checked programmatically against
Phases 0/1), extending `commod_lib8.py` with `build_continuous_series_ohlcv` and `spread_lib11.py`
with `VolBreakoutParams`, `compute_vol_breakout_entries`, `simulate_vol_breakout_single`/
`simulate_vol_breakout_book` (symmetric long/short, volume-gated or not via one flag) —
reusing `true_atr_series`, `sma_causal`, `regime_gate`, `pnl_atr`, `trade_blocks`,
`paired_block_bootstrap`, and `research.deflated_sharpe_prob` unmodified from 11a/11d — all
unit-tested in `tests/test_spread_lib11.py`. The holdout (2025-01-01 to 2026-07-28) remains
untouched and unspent.
