# 011c — Can You Predict Which Spread Trades Will Stop Out?

## The question

Notebook 011a's trade-shape analysis found two things: entry extremity does not separate winners
from losers, and the catastrophic tail comes almost entirely from stop exits.

That raises an obvious idea — one the other research programme proposed but never built. If the
damage is concentrated in stop-outs, can a classifier trained on what's knowable **at entry** predict
which trades will stop out, so those trades can simply be skipped?

The honest prior, informed by that same analysis, was **no**: if entry extremity doesn't
discriminate, the true separator is probably the first post-entry price response, which isn't
knowable at entry by definition. This notebook tests that prior properly rather than assuming it.

**The criterion, fixed in advance:** a walk-forward out-of-sample AUC above 0.60 on the stop-exit
label, **and** a book that suppresses its top-decile predicted-loss entries beating the unsuppressed
book on all three risk measures at every offset. Four trials — four origin offsets, one pre-declared
feature set, one model class, nothing swept.

**It does not fire.**

## The trade log and its features

Notebook 011a's control book isn't stored, it's regenerated — so the identical pre-declared trading
rule was re-run to reproduce it exactly: **57 trades, 23 stop exits, 34 clean exits**, matching the
earlier counts.

Fifteen features are attached to each trade, all computed **strictly from data at or before that
trade's own entry bar**, using a 252-day trailing window:

entry z-score magnitude · carry ratio (with real financing rates) · realised-volatility percentile ·
stationarity test statistic · rolling half-life · half-life sub-period stability · a full-sample
in-band flag · roll-window proximity · 60-day leg correlation · variance-ratio statistics at two
horizons · Hurst exponent · spread level within its own trailing range · 5-day and 20-day pre-move
in volatility units.

55 of 57 trades have every feature finite. The earliest two predate enough history for some rolling
statistics and are **dropped rather than imputed** — imputing would quietly change what "known at
entry" means.

## The classifier clears its bar — and the bar alone shouldn't be trusted

No walk-forward *classification* infrastructure existed here; the repo's existing machinery is
regression-only. The fold-splitting logic is reused unchanged — an anchored expanding window,
exactly the no-lookahead discipline established back in notebook 002 — but applied over **trades
rather than daily bars**, since 55 trades over 14.5 years is this classifier's native sampling unit.
One new primitive was added for the ranking metric.

The model is logistic regression with strong regularisation, chosen because 15 features against
roughly 30-trade training folds is a genuine more-features-than-data regime.

Four origin offsets, each with its own anchored grid (train on 30 trades, test on 5, step 5):

| Offset | Folds | Out-of-sample trades | AUC |
|---|---:|---:|---:|
| 0 | 5 | 25 | **0.757** |
| 1 | 4 | 20 | **0.667** |
| 2 | 4 | 20 | **0.738** |
| 3 | 4 | 20 | **0.707** |

All four clear the 0.60 bar on the criterion's literal text. **This leg fires.**

**But look closer.** The per-fold values are exactly the small handful a 5-trade test fold can
produce — 1.0, 0.75, 0.5, 0.25, 0.0, with no room for anything else. Inspecting the actual predicted
probabilities behind a perfect-scoring fold shows values clustered in **0.41–0.56** — barely
distinguishable from a coin flip in absolute terms, correctly *ranked* only by tiny margins.

Bootstrapping the already-collected out-of-sample predictions (resampling trades, not refitting)
gives a 95% interval that **includes 0.5 at every single offset**:

| Offset | Point estimate | 95% interval |
|---|---:|---|
| 0 | 0.757 | [0.487, 0.960] |
| 1 | 0.667 | [0.389, 0.920] |
| 2 | 0.738 | [0.458, 0.956] |
| 3 | 0.707 | [0.374, 0.944] |

**The leg fires by the pre-registered criterion's literal text, and that verdict is reported
honestly — but the same scrutiny this programme applies to every other small-sample result says this
point estimate is not distinguishable from chance on 20–25 out-of-sample trades.** Disclosed as a
fragility caveat on a leg that mechanically passes, rather than smoothed over.

## The suppression book fails outright

At each offset, the top decile of that offset's own out-of-sample predicted stop probability is
vetoed at entry — two or three trades per offset. Trades predating any out-of-sample prediction are
left untouched, since suppressing them would require a prediction that doesn't exist without
lookahead.

The veto is checked **last**, after every other suppression, regime and reentry condition, so it can
only remove an entry the rule would otherwise take, never add one.

| Offset | Vetoed | Sharpe | Max drawdown | Return/drawdown | Trades | Beats control? |
|---|---:|---|---|---|---|---|
| 0 | 3 | −0.1646 → −0.1635 | −1.877% → −1.877% | −0.580 → −0.576 | 57 → 57 | **Yes** |
| 1 | 2 | −0.1646 → −0.1591 | −1.877% → −1.877% | −0.580 → −0.561 | 57 → 57 | **Yes** |
| 2 | 2 | −0.1646 → −0.1939 | −1.877% → −1.877% | −0.580 → −0.680 | 57 → 56 | **No** |
| 3 | 2 | −0.1646 → −0.1639 | −1.877% → −1.878% | −0.580 → −0.578 | 57 → 57 | **Yes** |

Three of four offsets show a marginal improvement — Sharpe moving a few thousandths, well within any
noise floor measured anywhere else in this programme. One offset makes the book *worse*.

The every-offset standard is the operative convention everywhere else in this programme, and the
criterion's text doesn't relax it, so all four must clear. **They don't, and the criterion's own
requirement that both legs pass means this does not fire** — regardless of the already-qualified
first leg.

### A mechanical finding worth recording

At one offset, vetoing two entries removed only **one** net trade from the final book (57 → 56), not
two. Suppressing a specific entry bar doesn't guarantee the underlying z-score crossing never trades
at all — only that *this* bar's attempt is skipped. If the crossing persists past the vetoed bar, a
later entry can still open.

A "delete this entry" filter is **not perfectly throughput-neutral** in this engine — echoing the
same surprise found with the sign-flip test in notebook 011b.

## Bottom line

The result is more nuanced than a clean null: a point-estimate AUC comfortably above the bar at
every offset, on a genuinely walk-forward, no-lookahead classifier — but one a direct bootstrap check
shows is not reliably distinguishable from chance at this sample size, and whose practical payoff
fails at one of four offsets regardless.

Reported together rather than separately: a well-qualified near-miss on prediction, an outright miss
on the thing that would matter.

That is consistent with the prior — **the catastrophic tail is not predictable at entry time** — and
it directly supports keeping the stop-loss rule rather than trying to avoid the trades that trigger
it.

*Notebook: `src/research/011c_entry_time_loss_classifier.ipynb`. The holdout remains untouched; its
reduced independence for commodity-spread strategies, disclosed in notebook 011a, is unchanged.*
