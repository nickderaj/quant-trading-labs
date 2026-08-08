# Notebook 11d — Momentum/Breakout Transfer: Results Summary

## What

This notebook tests two pre-registered gates for a breakout-trading rule transferred from the external programme's qualitative sketch (prior-run, base-consolidation, breakout-entry, ATR-stop, MA-trail-exit, BTC-regime-gate): Gate MB on a 30-symbol crypto perpetuals universe (including delisted-in-effect names LUNAUSDT and FTTUSDT) and Gate MB-E on a 69-ticker commodity-equity/ETF universe.

## Why

The external programme never built or published a complete breakout system of its own — only a single cost-study result (breakout gross mean R positive, momentum gross mean R negative) and a qualitative description. Since there was no specific external parameter set to adopt as a prior (unlike Gate BF in 11b), this notebook needed to fix its own single trading-rule specification and test whether it can be supported by data this repo actually holds, rather than reproducing an external result that doesn't exist in quantitative form.

## How

A single breakout rule is declared once (25% trailing prior run, 10-bar tightening-range base, breakout above prior day's base high, 2x-ATR stop, 20-day-SMA trail exit, fixed-fractional sizing, fail-closed regime gate) and swept only along the two pre-registered axes (cost multiplier for Gate MB: 1x/2x/3x; offset for both gates) to avoid inflating the pre-registered n_trials (12 for MB, 4 for MB-E). Crypto costs reuse this repo's established convention; equity costs reuse the same convention as an explicit, disclosed assumption since this repo has no established equity cost model. Both gates are evaluated with every-offset Sharpe checks, block-bootstrap noise-floor CIs, and DSR against the pre-registered trial counts, on the development window only.

## Results

Both gates return fired=False. Gate MB shows positive Sharpe at every offset (+0.17 to +0.31) — the first leg clears — but the noise floor is far too wide (95% CI [−52.7%, +100.1%] on fixed-notional return) and DSR is only 0.055 against the 12-trial bar; the crypto history (3.5 years, 42 trades) simply cannot support a confident verdict either way, though a genuine but small cost-stress effect (−1.48pp per cost-multiplier step) is not the reason for the null. Gate MB-E fails outright and unambiguously: net Sharpe is negative at every offset (−0.022 to −0.027), and the universe's structural survivorship bias means it was never eligible for a fundable verdict regardless. The direction of the gap (crypto marginally promising, equities clearly negative) is consistent with the rule's fixed thresholds being implicitly calibrated to crypto's volatility regime rather than equities'.

Two pre-registered gates (`phase_6_11a_results.json`, committed before this notebook ran and not
edited since). **Both return a fired=False verdict.** Gate MB (30 crypto perpetuals, including
the two delisted-in-effect names, `LUNAUSDT` and `FTTUSDT`) clears its every-offset positive-Sharpe
leg but fails both the noise-floor CI and the DSR leg. Gate MB-E (69-ticker commodity-equity
universe, survivorship-unknown by construction) is net negative at every offset and fails outright.
DSR trial counts total **16** across the two gates (12 + 4), matching `phase_6_11a_results.json`'s
pre-registered breakdown exactly (asserted programmatically in each phase runner, not re-typed by
hand). The external programme's own equity momentum work has zero market evidence (NEXT_PROMPT.md
sec 7: Phase 3 measured zero eligible securities); this notebook does not attempt to reproduce
their strategy, only to test what this repo's own data can actually support.

## The breakout rule and why it is a single fixed prior, not a sweep

Unlike Gate BF/VA/RE in 11b, the external programme never built or published a Kullamagi-style
breakout system — NEXT_PROMPT.md sec 7 hands over only a single quantitative result (the Phase −1
cost study: breakout gross mean R 1.157 over 87 trades, net 1×/2×/3× 0.986/0.815/0.644, vs.
momentum's negative gross mean R over 44 trades) and a qualitative sketch ("prior run, base
consolidation, tightening range, breakout entry, ATR/percentage stop, MA-trail exit, BTC regime
gate"). There is no external parameter set to take as a prior the way Gate BF took `c < −0.5` from
a specific, out-of-sample-supported measurement. This notebook therefore had to fix its own single
rule, declared once and swept only along the two axes `phase_6_11a_results.json` already commits
to (cost multiplier for Gate MB, offset for both) — sweeping the rule's own thresholds would
inflate n_trials beyond the pre-registered 12/4 and was not done:

- **Prior run**: cumulative return ≥ +25% over a trailing 40-bar window ending at the base's start.
- **Base**: a 10-bar rolling high/low range ≤ 15% of price (the "tightening range").
- **Breakout**: today's close breaks above *yesterday's* base high (all three conditions evaluated
  as of bar *t*'s close; entry fills at bar *t*+1's open — a daily-bar backtest cannot fill on the
  same bar its own signal is confirmed without lookahead, so, like every other engine in this
  programme, this one waits one bar).
- **Stop**: entry − 2.0 × true ATR(14) (a genuine Wilder ATR here — these are single-instrument
  OHLC series, unlike the spread notebooks' honestly-mislabeled "ATR" which is really a std of
  daily changes because spreads have no OHLC). Checked before the trail exit each bar, same
  declared ordering as 11b's spread engine.
- **Exit**: close < 20-bar SMA → exit at next bar's open. No time stop.
- **Sizing**: fixed-fractional, 2% of book equity risked per trade against the stop distance,
  capped at 20% of equity notional and 3× leverage.
- **Regime gate, fail-closed**: crypto requires `BTCUSDT`'s 10-day SMA > its 20-day SMA (both
  shift(1)); equity requires at least 2 of {SPY, QQQ, IWM} in the same configuration. A reference
  symbol missing a date counts as not-confirming, never as an omission from the vote.

Phase 0's raw-signal diagnostic (`phase_0_11d_results.json`) confirms the rule is neither
degenerate nor vacuous before any backtest ran: 71 raw breakout signals across the 30 crypto
symbols, 1,193 across the 69 equity symbols, with the regime gate open 54.1% / 58.8% of the time
respectively — plausible base rates for a trend-following filter, not a coin flip or a filter that
never opens.

## Cost model and the equity-side assumption, disclosed

Crypto costs reuse this repo's own established convention (`TAKER_FEE=0.0004`,
`SLIPPAGE=0.0001`, notebook 3/7/8's `add_trading_costs`/`add_portfolio_costs` bps), applied per
side, stressed at 1×/2×/3×. **This repo has no established equity cost model** — commodity-sector
equities and ETFs are not this repo's usual instrument class — so Gate MB-E reuses the identical
5bps-per-side convention rather than inventing a new number. This is disclosed as an assumption,
not a measured cost, and is more likely too cheap than too dear for the less-liquid single-name
tickers in the 69 (`DINO`, `DE`, `DINO` vs. `SPY`/`QQQ` are not the same liquidity class); Gate
MB-E's net-negative result is therefore not an artifact of an inflated cost estimate.

## Holdout discipline

Both gates run on the development window only. Equities use their full available history
(1993–2024 for the oldest tickers) through 2024-12-31; crypto is bounded below by each symbol's
own Binance kline history (2021-07 onward) and above by 2024-12-31, same cutoff. Neither gate
touches 2025-01-01 → 2026-07-28. Sec 8's holdout-contamination disclosure is specific to
commodity-spread strategies (the external programme's held-out numbers, read during this design,
overlap that window) and explicitly does not apply to 11d's crypto work — the external momentum
programme produced no held-out numbers of its own to have read. The equity side inherits the same
general repo-wide holdout discipline as every other notebook, for the same reason as always: the
window stays unspent so it remains available for a future notebook that actually needs it.

## Gate MB: positive Sharpe everywhere, indistinguishable from noise everywhere

| offset | Sharpe (1×) | Sharpe (2×) | Sharpe (3×) |
|---|---:|---:|---:|
| 0 | **+0.173** | +0.166 | +0.159 |
| 7 | **+0.174** | — | — |
| 14 | **+0.241** | — | — |
| 21 | **+0.311** | — | — |

Net Sharpe is positive at every one of the four offsets — the first leg of Gate MB's `fires_if`
clears. Offset-0, 1× book: 42 trades over the crypto universe's full dev-window history, 24
stop-exits / 18 trail-exits, max drawdown **−35.5%**, fixed-notional return **+9.30%**
($1,000,000 → $1,088,852 equity-path), return/drawdown **+0.25**. `LUNAUSDT` fired exactly one
breakout signal in its 312-bar pre-collapse history (2021-07 to 2022-05-13) and that single trade
lost **−$20,182** — the crash is visible in the stop-exit fill (min(open, stop) correctly prices
the gap, not the pre-crash stop level) rather than silently absent from the book. `FTTUSDT`, whose
Binance kline data this repo happens to hold through 2024-12-31 despite FTX's November 2022
collapse, never fired a single raw breakout signal in 992 bars and contributes zero trades either
way — its inclusion changes nothing, but it was not excluded to get there.

The noise floor is the binding constraint. Bootstrapping the offset-0, 1×-cost book's own
fixed-notional return (block bootstrap, quarterly blocks, 2,000 draws) gives point **+9.30%**,
95% CI **[−52.7%, +100.1%]** — a **±76.4pp** half-width on a 42-trade, 3.5-year book. The CI
contains zero by a wide margin: **Gate MB's second leg does not clear.** The deflated Sharpe
probability, using the offset-0 daily Sharpe deflated for `n_trials=12` (3 cost multipliers × 4
offsets, per `phase_6_11a_results.json`) against `n_obs=42` trades, is **0.055** — far below the
0.95 bar. **Gate MB does not fire.** It is also not eligible for the fundable flag on the same
grounds (DSR) even setting the CI leg aside — the 3.5-year crypto history this repo holds simply
does not contain enough trades to clear a 12-trial deflation, the same "properly powered test"
shortfall this programme has published before (Gate SP, DSR 0.562; Gate RE, DSR unreachable at
n=36).

Cost stress is a genuine but small effect, not the reason for the null: the 1×-vs-3×-cost paired
block bootstrap (same resampled quarters cancel the shared price path) gives a delta of
**−1.48pp, 95% CI [−2.37pp, −0.70pp]** — real, statistically distinguishable from zero, and
economically small relative to the ±76.4pp noise floor that actually governs the verdict. Tripling
transaction costs cannot explain why this book's null is a null; sample size can.

## Gate MB-E: net negative at every offset, no ambiguity to adjudicate

| offset | Sharpe (1×) |
|---|---:|
| 0 | **−0.022** |
| 7 | −0.022 |
| 14 | −0.022 |
| 21 | −0.027 |

401 trades, 133 stop-exits / 268 trail-exits, offset-0 fixed-notional return **−19.6%**, max
drawdown **−32.0%**, return/drawdown **−0.61**. Net Sharpe is negative at every offset — Gate
MB-E's first leg fails outright, so the noise-floor CI ([−111.1%, +81.1%], point −19.6%) and the
DSR (**0.140** at `n_trials=4`) are reported for completeness but are moot. **Gate MB-E does not
fire**, and per `phase_6_11a_results.json`'s own pre-registration it was never eligible for the
institutionally-fundable flag regardless of outcome — a survivorship-unknown universe (this
repo's 69 yfinance tickers are current listings only, structurally incapable of holding a name
that later delisted) cannot support an absolute-performance claim, the same caveat NEXT_PROMPT.md
sec 7 attaches to the external programme's own legacy proxy registry.

The direction of the gap is itself informative and matches the Phase −1 cost study's own prior
(§7's cited table: breakout net-of-cost positive, momentum net-of-cost negative, same universe and
window). This notebook's version of the breakout rule reverses sign on the equity side while
holding on the crypto side — the same rule, same regime-gate logic, same cost convention, applied
to a slower-moving, lower-beta commodity-equity/ETF universe instead of crypto perpetuals. That is
consistent with a mechanism whose edge (if it exists on the crypto side at all, which the DSR says
this sample cannot establish) depends on the underlying instrument's volatility and trend
persistence matching the rule's fixed thresholds — a 25% prior run and a 15%-of-price base width
are calibrated implicitly to crypto's volatility regime, not equities'. Re-fitting those thresholds
per-universe was not done here (it would inflate n_trials beyond the pre-registered 4), and is the
natural next step if this notebook's own null is to be pursued further.

## What this notebook establishes, plainly

Both gates are honest nulls, and the two nulls fail for different, informative reasons. Gate MB's
Sharpe is positive everywhere and its cost-stress delta is real and small — the mechanism is not
obviously broken, but 3.5 years of crypto history and 42 trades cannot clear a 12-trial DSR bar,
and the noise floor is wide enough (±76.4pp) that no verdict beyond "cannot be distinguished from
noise on this sample" is honest. Gate MB-E fails on its own terms before the statistics get a
chance to matter — net Sharpe is negative at every offset on a universe this notebook's own
governing document (NEXT_PROMPT.md sec 7) says was never eligible for a fundable verdict in the
first place. Consistent with sec 0.3's programme-wide pattern and sec 11's honest-null discipline:
a null reported with full rigour — every-offset check, paired block bootstrap, noise floor, DSR
against the pre-registered trial count, three-way risk gate — remains a success, even when (Gate
MB) the point estimates look promising and even when (Gate MB-E) they plainly do not.

Machinery: `src/research/tmp/run_phase_{0,1,2,3}_11d_*.py` (Phase 0 universe loading, dev-window
trim, regime-gate construction, and the raw-signal firing-rate diagnostic; Phase 1 Gate MB; Phase
2 Gate MB-E, importing Phase 0/1's loaders and `build_book` unchanged rather than duplicating
them; Phase 3 the final gate table, cross-checked programmatically against
`phase_6_11a_results.json`), extending `spread_lib11.py` with `true_atr_series`, `sma_causal`, `regime_gate`,
`BreakoutParams`, `compute_breakout_entries`, `simulate_breakout_single`/`simulate_breakout_book`,
and `breakout_book_metrics` — reusing `pnl_atr`, `trade_blocks`, `paired_block_bootstrap`, and
`noise_floor` unmodified from 11a — all unit-tested in `tests/test_spread_lib11.py`. The holdout
(2025-01-01 to 2026-07-28) remains untouched and unspent.
