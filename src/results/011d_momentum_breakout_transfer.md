# 011d — A Breakout Rule, Tested on Crypto and on Commodity Equities

## The question

The other research programme never built a complete breakout system. It handed over one quantitative
result — a cost study showing breakout trades gross-profitable and momentum trades gross-unprofitable
over the same window — and a qualitative sketch: a prior run, a consolidation base with a tightening
range, an entry on the breakout, a volatility-based stop, a moving-average trailing exit, and a
market-regime gate.

There is no external parameter set to adopt as a prior, unlike the sign-flip test in notebook 011b,
which took a specific threshold from a specific out-of-sample measurement. So this notebook fixes
**its own single rule** and asks whether the data this repo actually holds can support it.

Two universes:

- **30 crypto perpetuals**, including two names that effectively delisted — the Terra token and the
  FTX exchange token.
- **69 commodity-sector equities and ETFs**, a universe that is survivorship-unknown by
  construction.

**Both come back null**, for different and informative reasons. Trial counts total 16 (12 and 4),
matching the pre-registration exactly.

## The rule, declared once

Sweeping the rule's own thresholds would inflate the trial count beyond what was pre-registered, so
the rule is fixed and swept only along the two committed axes: cost multiplier (crypto only) and
origin offset.

- **Prior run** — cumulative return of at least +25% over a trailing 40-bar window ending where the
  base starts.
- **Base** — a 10-bar rolling high-low range no wider than 15% of price. The tightening range.
- **Breakout** — today's close breaks above *yesterday's* base high. All three conditions are
  evaluated as of the bar's close, and entry fills at the *next* bar's open. A daily-bar backtest
  cannot fill on the same bar its own signal is confirmed without lookahead, so this engine waits
  one bar, like every other engine here.
- **Stop** — entry price minus twice the 14-bar average true range. This is a genuine true range,
  since these are single-instrument series with real high/low data — unlike the spread notebooks,
  where the equivalent quantity is honestly labelled but is really a standard deviation of daily
  changes, because spreads have no high or low.
- **Exit** — close below the 20-bar moving average, exiting at the next bar's open. No time stop.
- **Sizing** — fixed-fractional, risking 2% of book equity per trade against the stop distance,
  capped at 20% of equity notional and 3× leverage.
- **Regime gate, fail-closed** — crypto requires Bitcoin's 10-day average above its 20-day average;
  equities require at least two of three broad index ETFs in the same configuration. **A reference
  symbol missing a date counts as not confirming, never as an abstention.**

A raw-signal diagnostic confirms the rule is neither degenerate nor vacuous before any backtest ran:
71 raw breakout signals across the 30 crypto symbols and 1,193 across the 69 equity symbols, with
the regime gate open 54.1% and 58.8% of the time respectively. Plausible base rates for a
trend-following filter, not a coin flip and not a filter that never opens.

## Costs, and a disclosed assumption

Crypto costs reuse this repo's established convention, applied per side and stressed at 1×, 2× and
3×.

**This repo has no established equity cost model** — commodity-sector equities and ETFs are not its
usual instrument class — so the equity test reuses the identical per-side convention rather than
inventing a new number. That is disclosed as an assumption, not a measurement, and it is **more
likely too cheap than too expensive** for the less liquid single-name tickers in that universe. The
equity result being net-negative is therefore not an artefact of an inflated cost estimate.

Both tests run on the development window only. The holdout is untouched.

## Crypto: positive Sharpe everywhere, indistinguishable from noise everywhere

| Offset | Sharpe (1× cost) | (2×) | (3×) |
|---|---:|---:|---:|
| 0 | **+0.173** | +0.166 | +0.159 |
| 7 | **+0.174** | — | — |
| 14 | **+0.241** | — | — |
| 21 | **+0.311** | — | — |

Net Sharpe is positive at all four offsets — the first requirement clears.

At the baseline setting: 42 trades over the full available history, 24 stop exits and 18 trailing
exits, maximum drawdown **−35.5%**, fixed-notional return **+9.30%**, return-to-drawdown **+0.25**.

Two notes on the delisted names. The Terra token fired exactly one breakout signal in its 312-bar
pre-collapse history, and that single trade lost **$20,182** — the crash is visible in the stop
fill, which correctly prices the gap rather than the pre-crash stop level, instead of being silently
absent from the book. The FTX token, whose data this repo happens to hold past the exchange's
collapse, never fired a single raw signal in 992 bars and contributes nothing either way. Its
inclusion changes nothing — but it was not excluded to get there.

**The noise floor is the binding constraint.** Bootstrapping the baseline book's own fixed-notional
return, with quarterly blocks and 2,000 draws, gives a point estimate of **+9.30%** and a 95%
interval of **[−52.7%, +100.1%]** — a **±76.4 percentage point** half-width on a 42-trade,
3.5-year book. The interval contains zero by a wide margin.

The deflated Sharpe probability at 12 trials against 42 trades is **0.055**, far below the 0.95 bar.

**It does not fire.** The 3.5 years of crypto history this repo holds simply does not contain enough
trades to clear a 12-trial deflation — the same power shortfall published before for spread mean
reversion (0.562) and the reentry grid (unreachable at 36 trials).

**Cost stress is a genuine but small effect, and it is not the reason for the null.** The paired
comparison between 1× and 3× costs — where the same resampled quarters cancel the shared price path —
gives a difference of **−1.48 percentage points, interval [−2.37, −0.70]**. Real, statistically
distinguishable from zero, and economically tiny relative to the ±76.4-point noise floor that
actually governs the verdict. **Tripling transaction costs cannot explain this null; sample size
can.**

## Commodity equities: net negative at every offset

| Offset | Sharpe |
|---|---:|
| 0 | **−0.022** |
| 7 | −0.022 |
| 14 | −0.022 |
| 21 | −0.027 |

401 trades, 133 stop exits and 268 trailing exits, fixed-notional return **−19.6%**, maximum
drawdown **−32.0%**, return-to-drawdown **−0.61**.

Net Sharpe is negative at every offset, so the first requirement fails outright and the noise-floor
interval and deflated probability are reported for completeness but are moot.

**It was never eligible for a fundable verdict regardless of outcome.** The 69 tickers are current
listings only, structurally incapable of including a name that later delisted, and a
survivorship-unknown universe cannot support an absolute-performance claim.

## Why the two universes differ, and what that suggests

**The direction of the gap is itself informative**, and matches the original cost study's own
finding that breakout trades were net-of-cost positive where momentum trades were not.

The same rule, the same regime-gate logic and the same cost convention hold up on the crypto side
and reverse sign on the equity side. That is consistent with a mechanism whose edge — if it exists
on the crypto side at all, which the deflation says this sample cannot establish — depends on the
underlying instrument's volatility and trend persistence matching the rule's **fixed thresholds**.
A 25% prior run and a 15%-of-price base width are calibrated implicitly to crypto's volatility
regime, not to equities'.

Re-fitting those thresholds per universe was not done, since it would inflate the trial count beyond
what was pre-registered. It is the natural next step if this null is to be pursued.

## Bottom line

Two honest nulls, failing for different and informative reasons.

**Crypto:** Sharpe is positive everywhere and the cost-stress effect is real and small — the
mechanism is not obviously broken. But 3.5 years and 42 trades cannot clear a 12-trial deflation
bar, and the noise floor is wide enough that no verdict beyond "cannot be distinguished from noise
on this sample" is honest.

**Equities:** it fails on its own terms before the statistics get a chance to matter, on a universe
that was never eligible for a fundable verdict in the first place.

A null reported with full rigour — every-offset check, paired bootstrap, noise floor, deflation
against the pre-registered trial count, and the three-way risk view — remains a result, both when
the point estimates look promising and when they plainly do not.

*Notebook: `src/research/011d_momentum_breakout_transfer.ipynb`. The holdout remains untouched. The
reduced-independence disclosure from notebook 011a applies to commodity spreads only and does not
apply here — the corresponding crypto programme produced no held-out numbers.*
