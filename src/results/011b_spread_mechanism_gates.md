# 011b — Do the Outside Programme's Trading Mechanisms Actually Help?

## The question

Notebook 011a reproduced a second research programme's spread-trading work and found that its
headline performance doesn't reproduce here — but that its *trade shape* does. So the natural next
question is whether its distinctive **mechanisms** are portable, or whether its apparent edge came
from universe and parameter choices fitted to its own data.

Seven tests, all with criteria fixed before this notebook ran:

1. **The structured trading rule** — discrete trades with entry/exit thresholds and volatility-based
   stops, against this repo's own simpler continuous position.
2. **The same rule with the stop disabled** — to isolate how much of any gap the stop explains.
3. **Sign-flipping mild-backwardation trades** — reverse the signal in one specific curve state.
4. **Per-spread version of the same** — does it help on individual spreads even if not in aggregate?
5. **The stationarity screen** — does filtering the universe help or hurt?
6. **A volatility-adaptive stop** — scale the stop distance with recent realised volatility.
7. **A reentry-parameter sweep** — a full 36-cell grid over the gated-reentry mechanism.

**Every comparison is internal.** Structured against continuous, sign-flipped against
unconditional, screen-inclusive against screen-exclusive, adaptive stop against static — never a
validation of the other programme's absolute numbers, which notebook 011a already found diverge
materially.

Trial counts total 65, cross-checked line by line against the pre-registration. Every count matches
exactly; none was shrunk.

**All seven come back null.** The holdout remains untouched.

## The most important result: the structure makes things worse, not better

The pre-declared trading rule — entry and exit thresholds, per-spread volatility stops,
fixed-fractional sizing, suppression filters, gated reentry — run on the five live spreads under
this repo's costs, produces a **negative** Sharpe at every origin offset: −0.165, −0.185, −0.180,
−0.209.

The continuous benchmark from notebook 010b, restricted to the **same five spreads** and the **same
costs**, runs **+0.552 to +0.555** at every offset.

The paired bootstrap interval on the difference is **[−0.520, −0.124]**, excluding zero in the
direction that says the discrete, stopped packaging is **worse**.

**This fails in the opposite direction from the hypothesis.** The plain continuous, always-on
position — the thing the hypothesis called strictly inferior to their mechanism — outperforms their
own pre-declared mechanism by a wide, bootstrapped margin on this data and these costs.

### Disabling the stop sharpens the finding rather than muddying it

Removing only the stop, keeping every other piece of the discrete packaging, turns the structured
book's Sharpe **positive** (+0.304, 107 trades, no stop exits by construction). Better than the
stopped version — but **still significantly below the continuous benchmark** (paired interval
[−0.468, −0.092], still excluding zero in the same direction).

So the stop is *a* drag, but not the *whole* story. The remaining gap between the stop-disabled
discrete book (+0.304) and the continuous book (+0.552) is attributable to the entry and exit
thresholds and the trade-boundary discipline themselves.

The hypothesis going in was "their mechanisms are better than ours; our statistics are better than
theirs". On this repo's data and costs, **the packaging is a net negative at every layer tested**,
and the plainest possible always-on position remains the best-performing book here.

## Sign-flipping mild backwardation makes the book monotonically worse

Tested on the 16-spread calendar universe — a declared, non-cherry-picked pooled list —
sign-flipping only the entries opened in the mild-backwardation bucket, against three assumptions
about storage cost, all reported:

| Storage assumption | Sharpe by offset | Trades | Versus unconditional |
|---|---|---:|---|
| Unconditional (no flip) | 0.236 / 0.186 / 0.188 / 0.182 | 436 | — |
| Low | 0.200 / 0.139 / 0.142 / 0.132 | 378 | Worse at every offset |
| **Mid (headline)** | −0.146 / −0.208 / −0.206 / −0.216 | 378 | Worse, and sign-flipped |
| High | −0.185 / −0.248 / −0.245 / −0.256 | 377 | Worse still |

**The degradation is monotone in the storage assumption**, and even the low assumption never
exceeds the unconditional book at any offset.

There's also a mechanical finding worth recording. Trade count drops about 13% (378 against 436),
independent of the storage assumption. A pure sign flip is **not throughput-neutral** in this
engine, because flipping direction changes which trades get stopped out — adverse excursion is
measured relative to direction. Worth knowing for any future proposal that reverses rather than
filters a signal.

The per-spread version finds only 4 of 16 spreads individually improved, short of its own bar.

### A sign convention, resolved

Notebook 011a flagged that this repo's carry-ratio primitive evaluates positive in backwardation and
negative in contango — the opposite of the other programme's description. Verified again here across
the full calendar universe: applying their bucket boundaries requires negating this repo's raw ratio
first. The corrected mapping is used throughout, so the result above is not an artefact of the sign
confusion.

## The stationarity screen is essentially inert

The screen-inclusive book (4 spreads) and the screen-exclusive book (adding 3 more, resolving both
named cross-programme disagreements) are, to three decimal places, **the same book** at every offset
— Sharpe −0.165 against −0.164, with a paired interval of [−0.0002, +0.00005], a needle's width
around zero.

The reason is that the three added spreads contribute almost no trades under the pre-declared risk
rule (see below). The screen's presence or absence is close to observationally void for this
parameterisation.

**The screen does not survive** — but on a technicality: it barely matters either way, not that
dropping it helps. That distinction is worth stating plainly. **This notebook cannot currently
distinguish whether the screen is good, bad, or simply inert** under this trading rule and universe.

## The volatility-adaptive stop: a genuine near-miss

Scaling the stop distance between 0.75× and 1.25× against a rolling realised-volatility percentile
improves **both** Sharpe (−0.111 to −0.157 across offsets, against the control's −0.165 to −0.209)
**and** maximum drawdown (−1.63% against −1.88%) at every single offset. Directionally exactly what
was anticipated.

But the improved Sharpe never crosses zero, and the paired interval [−0.004, +0.013] still straddles
it.

Unlike the sign-flip result, this is a genuinely close, informative near-miss rather than a
reversal. The mechanism helps on both axes that matter, just not by enough — on a universe where the
control itself is Sharpe-negative at every offset anyway.

## The reentry sweep: an unreachable bar at its honest denominator

The full 36-cell grid's best non-baseline cell improves on the baseline at every offset (−0.114 /
−0.134 / −0.129 / −0.160 against −0.165 / −0.185 / −0.180 / −0.209), and its paired interval
[−0.000004, +0.0095] sits a hair's width from excluding zero on the positive side.

But the best cell's own Sharpe is still negative at every offset, so the "positive at every offset"
requirement fails outright — and the deflated probability at 36 trials is **0.0138**, nowhere near
the 0.95 bar.

**The 36-trial count is reported in full and not reduced because the grid turned out unreachable.**
That unreachability, at its honest denominator, is itself the finding — the same tradition as
notebook 010b's spread mean-reversion result.

## Two disputed spreads are barely tradeable at all

Neither of the two spreads at the centre of the cross-programme disagreement is meaningfully
tradeable under this notebook's pre-declared parameterisation, and the reason is **mechanical rather
than statistical**.

The single-name cap of 12% of an assumed $1,000,000 starting equity limits notional per contract to
$120,000. But one spread's median per-contract notional is **$196,275** — inside the cap on only
0.2% of development-window bars — and the other's is **$147,870**, inside the cap on 9.2% of bars.
Position size floors to zero almost everywhere regardless of the risk parameter.

This was confirmed directly by re-running both at five times the default risk setting, with **zero
change in trade count** (0 trades at both settings for one; 5 trades at both for the other). **The
single-name cap, not the risk parameter, is the binding constraint** — and this notebook does not
relax it to work around something that is itself part of the pre-declared rule.

The consequence is that the two programmes' conflicting claims about one of these spreads cannot be
adjudicated from this side; that disagreement lives entirely inside data this repo doesn't have.

What *is* established independently is a different, prior question: under the pre-declared sizing
rule and a plausible $1M capital base, both spreads are barely tradeable at all — which is itself a
reason their standalone framing may not transfer cleanly between programmes with different assumed
capital bases.

## Reconciling notebook 010b's drawdown figure

Notebook 010b reported a maximum drawdown of −5.41 in continuously compounded log units, which
exponentiates to ≈ −99.55% of peak. **That reading is confirmed correct, not a bug.**

The capital-bounded alternative was also built: equity evolves by adding simple returns on a fixed
notional, with an absorbing floor so an account that hits zero stays there rather than reviving when
a cumulative sum later recovers.

**Sharpe and the deflated probability are unchanged** by the recomputation (1.16478 against the
published 1.16466 — a 1e−4 relative difference from non-deterministic rolling internals, not from
methodology; both round to the same three significant figures).

But the fixed-notional reading is different **in kind**, not just degree: the account is wiped out
at bar 49 of 4,488 and never recovers, giving a drawdown of −100.0% outright — worse than the
already-extreme compounded reading.

This is not a contradiction; it is a concrete illustration of *why* the two conventions diverge this
much. A daily-rebalanced, always-reinvested portfolio's real equity path is genuinely closer to the
compounded reading — that is how a live account marks itself to market daily, which is why it
survives past bar 49. The fixed-notional convention this repo's episodic single-trade bookkeeping
uses everywhere else produces a qualitatively different, more pessimistic answer when applied to a
strategy that re-levers every day.

Both are reported as labelled hypothetical recomputations of an already-published result, neither
replacing it.

## All seven books on the three-way risk view

| Test | Sharpe | Max drawdown | Return/drawdown | Fires |
|---|---:|---:|---:|---|
| Structured, stopped | −0.165 | −1.88% | −0.57 | No |
| Structured, stop disabled | +0.304 | −1.83% | +1.81 | No |
| Sign-flipped (mid storage) | −0.146 | −5.80% | −0.44 | No |
| Screen-inclusive | −0.165 | −1.88% | −0.57 | No |
| Volatility-adaptive stop | −0.111 | −1.63% | −0.62 | No |
| Best reentry cell | −0.114 | −1.57% | −0.61 | No |

No book clears a Sharpe of 0.5, so the fundable flag is moot for all seven. None is even in range
of the tradeable-alpha bar, let alone the fundable one.

Had any of these cleared an interval on Sharpe alone without this table, that would have been the
wrong basis for a verdict. None did, so the table is confirmatory rather than load-bearing — but it
is reported in full regardless.

## Bottom line

The stakes here were higher than a routine extension. The structured trading rule was **the first
proposition in this programme with a specific, mechanically identified reason to expect a positive
result**, and the sign-flip was **the first with independent out-of-sample support from a separate
codebase**.

Both came back null against a properly paired control — and both came back null in the
*informative* direction: **the other book's advantage is universe selection and parameterisation
fitted on its own data, not a portable mechanism.** That is exactly the distinction this repo's
methodology is positioned to make and theirs is not.

The continuous, un-stopped, unstructured position this programme has been implicitly comparing
against all along remains, on this data and these costs, the best-performing book here.

*Notebook: `src/research/011b_spread_mechanism_gates.ipynb`. The holdout remains untouched; its
reduced independence for commodity-spread strategies specifically, disclosed in notebook 011a, is
unchanged.*
