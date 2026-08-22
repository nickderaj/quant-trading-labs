# 010b — Five Spread and Sizing Strategies, Backtested

## What was tested

Five distinct strategies, each with its firing criteria fixed in advance by notebook 010a's
pre-registration, plus one data-provenance question:

1. **Unconditional spread mean reversion** — does the mechanism notebook 009's probe found survive
   full transaction costs?
2. **Regime-gated spread mean reversion** — does only trading when the term structure is in a
   definite state improve on that?
3. **The Brent–WTI regime question** specifically — is the regime effect real, or a single-spread
   artefact, and does it depend on which leg's curve defines the regime?
4. **Volatility-scaled carry** — does risk-weighting positions close the gap notebook 008's carry
   left open?
5. **Blended multi-lookback momentum** — does averaging momentum horizons beat any single one?
6. **Is the cached crypto data spot or perpetuals?** — a precondition for a future cash-and-carry
   test.

Every backtest charges **one round-turn cost per leg**, so a three-leg spread pays three, not one.
All use four origin offsets and are judged on a block-bootstrap interval plus a deflated Sharpe
probability against an honestly counted cumulative trial count.

A separate **fundable-flag** standard, introduced by notebook 009's recommendation, is also
reported: net Sharpe above 0.5 at every offset, deflated probability above 0.95, **and** a literal
bounded drawdown. It sits alongside, not instead of, the tradeable-alpha criterion.

The holdout is untouched throughout — nothing here is a certified winner, so there is no legitimate
reason to spend it.

## The headline

**Five strategies, five nulls — and the two most informative near-misses in this programme's
history.**

| Strategy | Passes? | Fundable flag? | The number behind it |
|---|---|---|---|
| **Mean reversion, inter-commodity** | **No** | No | Net Sharpe 0.42 at every offset; deflated probability 0.562 (8 trials); interval on net return [−2.2e−5, +2.0e−4] does not exclude zero |
| **Mean reversion, calendar** | **No** | No | Net Sharpe 0.50–0.51; deflated probability 0.680; interval [+1.9e−7, +6.8e−5] **does** exclude zero, but the deflation alone kills it |
| **Regime gating** | **No** | No | Gated Sharpe exceeds unconditional at every offset (0.426 vs 0.423) but by a margin too small to matter; deflated probability 0.484 (12 trials); neither interval excludes zero |
| **Brent–WTI regime check** | **No** | — | Brent–WTI itself does *not* show the effect under the primary definition (0.604 vs 0.614); 3 of 6 other eligible spreads do, clearing the "at least 3 others" bar — but the binding Brent–WTI clause is not met |
| **Volatility-scaled carry** | **No** | **No — on drawdown alone** | Net Sharpe **1.16–1.23** (up from 0.90–0.95), deflated probability **0.9997** — both clear the fundable bar — but the excess-over-basket interval [−0.0022, +0.0098] still includes zero, and the drawdown is ≈99.6% of peak equity |
| **Blended momentum** | **No** | No | Net Sharpe **negative** at every offset (−0.015 to −0.033); deflated probability 0.025 (20 trials) |
| **Is the crypto data spot?** | **Resolved: no** | — | Every download path is a perpetual-futures endpoint, and every cached symbol has a matching funding file — funding only exists for perpetuals |

**No strategy fires. None clears the fundable flag either** — and the closest, volatility-scaled
carry, fails specifically and only on the drawdown bound.

## First: does notebook 010a reproduce?

Fifteen assertions against the committed pre-registration — spread counts, the taxonomy split, the
two cointegration exclusions, the deadband-is-primary declaration, all five trial counts, and the
data-provenance resolution. All passed before any backtest ran.

---

## Unconditional spread mean reversion

The rule was fixed in advance and is not re-derived here: a 60-day rolling z-score of the spread's
own value; position equals the negative z-score, clipped to ±2 and halved; one round-turn cost per
leg, summed across legs; roll-window rows excluded from both signal and P&L; equal-weighted across
each taxonomy group's cointegration-eligible spreads (7 inter-commodity, 16 calendar).

**Both books show a real, positive, cost-surviving net Sharpe at every origin offset** —
inter-commodity at 0.42, calendar at 0.50–0.51. Genuinely better than a coin flip, and directionally
exactly what notebook 009's cheap probe predicted.

**Neither clears the deflation bar** at the honestly counted 8 trials: 0.562 and 0.680. The
calendar book's interval on net return does exclude zero on its own, but the deflation alone is
enough to keep it from firing.

That is a clean illustration of why the deflation bar exists. A Sharpe this size, found after
screening this many configurations, is not yet distinguishable from what an unlucky search would
produce by chance.

---

## Regime gating

Same rule, same universe, with the position zeroed on any day the term-structure regime is not in a
definite state. The deadband definition is primary; raw sign and a persistence requirement are
secondary and both counted toward the 12-trial total.

**Cross-spread verdict: the direction is genuinely and consistently supportive, but it isn't
tradeable.**

Across all four origin offsets, the gated book's net Sharpe exceeds the unconditional book's — a
small but perfectly consistent margin (0.426–0.427 against 0.423–0.424). **That consistency is worth
something: it is not what four independent coin flips produce.**

But the absolute margin is too small for either interval — gated against zero, or gated minus
unconditional — to clear zero, and the deflated probability at 12 trials is 0.484. Essentially no
better than random search would produce this often.

**A real but too-small-to-trade directional signal, not a tradeable improvement.**

Both secondary definitions corroborate the same picture. Raw sign shows the gated book *not*
exceeding the unconditional one at all — exactly the structural weakness that got it demoted from
primary in advance, since it restricts almost no days. The persistence variant shows gating
exceeding unconditional at every offset, similar in spirit to the deadband.

### The Brent–WTI question: genuinely unresolved, and reported as such

Under the **primary** definition committed to in advance — Brent's own curve alone — Brent–WTI's
gated Sharpe (0.604) is *lower* than its unconditional Sharpe (0.614). The prior does not show up
for this spread under the definition this notebook was bound to.

Under the **secondary** both-legs-agree definition, also pre-declared and run once, the picture
flips: gated Sharpe of 0.694 clears unconditional 0.614 by a real margin. That echoes notebook
010a's descriptive finding that Brent–WTI's half-life drops from 79 days pooled to 9.5 days
specifically when both curves agree on backwardation.

**The honest reading: Brent–WTI's regime effect, if it exists, needs *both* legs to confirm the
state — one leg's curve is not enough.** The binding criterion here is evaluated on the primary
definition, as it must be to mean anything, and correctly does not fire on that basis.

A future notebook with its own fresh pre-registration is the legitimate way to test the
both-legs-agree definition as primary. Not a retroactive edit here.

This is not the same as notebook 007's single-symbol artefact. There, one thinly-traded symbol was
carrying an entire result. Here, the sensitivity is to a *definitional* choice with a clear economic
reading, which is itself informative.

---

## Volatility-scaled carry — the most consequential result here

The only change from notebook 008's carry book is the position-sizing rule: inverse
20-day-realised-volatility weighting instead of equal weighting within each leg.

**It fires on two of three fundable criteria and fails hard on the third.**

Net Sharpe lifts from 0.90–0.95 to **1.16–1.23**, and the deflated probability from 0.997 to
**0.9997**. Both comfortably inside the fundable flag's own bars.

But the excess-over-basket interval is essentially unchanged in shape — [−0.0022, +0.0098], still
including zero. **Volatility scaling reshapes risk *within* the carry book; it does not change
whether carry beats a passive commodity basket.** That is a question about the factor's exposure to
broad commodity beta, not about how that exposure is risk-weighted internally.

And separately, the drawdown corresponds to roughly **99.6% of peak equity**, nowhere near the
25% bound declared in advance — **failing the fundable flag on a completely independent criterion
from the one that fails the tradeable-alpha test.**

Reported exactly as required: *fundable-looking on absolute Sharpe and deflated Sharpe, not shown to
beat passive exposure to the same asset class, and not institutionally fundable either once drawdown
is actually checked.* Never rounded up to "essentially a pass".

### A note on that drawdown figure

This repo reports drawdown as a cumulative log-return quantity, matching every other notebook's
convention, rather than a literal percentage. Converting to a percentage of peak equity for the
first time in this programme — because the fundable flag requires a literal bound — turns a
log-drawdown of −5.41 into ≈99.6% of peak.

That is an artefact of an 18-year, high-turnover, continuously-compounding backtest convention that
was never previously asked to answer a literal percentage question. It's reported honestly rather
than converted to a friendlier number, and it is itself a finding about construction: a
**capital-bounded** equity curve, rather than a continuously reinvested one, would be needed to make
this comparison fair to the strategy if the flag is to be checked routinely.

---

## Blended momentum — an unambiguous null

An equal-weighted blend of notebook 008's four momentum lookbacks is net-**negative** at every
offset (−0.015 to −0.033). Sign-consistent, but consistently the wrong sign — diluting the weak
positive one-month and twelve-month Sharpes with the larger negative three-month and six-month ones,
exactly as notebook 008's per-lookback breakdown predicted.

The deflated probability of 0.025 at 20 trials confirms there is no signal being missed. Not a
near-miss.

---

## The data-provenance question

**Resolved: the cached crypto data is entirely perpetual futures, not spot.**

Every download path in the repo points at a perpetual-futures endpoint. No spot host appears
anywhere in the downloader. And every cached symbol has a matching funding-rate file — a genuine
spot series would have none.

The cash-and-carry test is deferred with this note, and **no proxy was built.** Treating the
perpetual's own mark price as a stand-in for spot would manufacture a spread mechanically guaranteed
to look small — that measures the construction, not the opportunity.

---

## Bugs and discipline notes

No new bugs in this notebook's machinery. One discipline note worth recording in the same spirit:

The volatility-scaled carry book's raw log-drawdown, taken at face value, would have silently passed
an eyeball check — −5.41 reads like a moderate number next to notebook 008's own −7.52. Only
converting it explicitly, for the first time this programme has needed a literal percentage bound,
revealed the ≈99.6% figure. Caught by taking the declared bound literally rather than reading an
existing metric's magnitude as "probably fine because it's smaller than last time".

## Bottom line

Five honest nulls, with considerably more texture than a flat "nothing works":

- **Spread mean reversion genuinely survives cost** at a positive Sharpe on both taxonomy groups. It
  simply hasn't cleared the multiple-testing bar — and the honest trial count, not a shrunk one, is
  why.
- **Regime gating points the right direction at every offset**, too faintly to trade.
- **The Brent–WTI result is a genuinely open question**, sensitive to the leg definition in a way
  that is itself informative rather than a mere artefact.
- **Volatility-scaled carry is the most important negative result here**, because it dissociates
  three previously conflated ideas — absolute Sharpe, deflated Sharpe, and a literal drawdown bound —
  and shows a strategy can clear the first two decisively while failing the third just as
  decisively. That discrimination is exactly what the two-flag reporting standard was built for.
- **Blended momentum closes cleanly**, with no ambiguity.

Nothing reaches the holdout.

*Notebook: `src/research/010b_spread_strategies.ipynb`.*
