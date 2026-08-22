# 011a — Reproducing an Outside Spread-Trading Programme

## Purpose

A second, independent research programme has been working on commodity spread trading, with its own
codebase, its own data pipeline and its own reported results. It is a potential source of validated
strategy ideas and parameter priors.

But it had recently gone through a major data correction — a contract-substitution bug that
corrupted 12.6 years of its spread series and changed three of its six strategy verdicts. So before
trusting any of its numbers, this notebook independently verifies them.

Four things get done here:

1. **Reproduce its half-life measurements** on this repo's own independently built data.
2. **Reproduce its control-book backtest** under this repo's own, stricter cost model.
3. **Rebuild and calibrate its stationarity screen**, both the old version and the new stricter one.
4. **Replicate its trade-shape analysis** on the reproduced book.

And then pre-register everything the follow-on notebooks (011b, 011c, 011d) will test, before any
of those backtests exist.

**This notebook is descriptive and infrastructural only — no verdicts, no strategy conclusions.**

## Four findings

### The half-life corroboration is real, and it closes the other programme's top open item

That programme's data correction changed two of its verdicts specifically because a half-life had
been measured on corrupted data — its flagship Brent calendar spread measured 1.7–4.7 days
corrupted against 28–73 days (mean 59.5) corrected. Its own highest-priority follow-up was to
remeasure half-life on the corrected series as a standalone check.

This repo already had that measurement, built independently from a different data vendor with no
code shared between the two programmes at any point. It's re-derived here and then reproduced a
second way, via a from-specification reimplementation calling the same underlying regression on the
same roll-window-excluded series.

| Spread | Half-life measured here |
|---|---|
| Brent calendar | 42.7 days |
| Brent–WTI | 79.3 days |
| Corn–wheat | 45.4 days |
| Bean–corn | 118.2 days |
| KC–Chicago wheat | 113.4 days |

**Every one lands inside the corrected ranges and nowhere near the corrupted ones.** A genuine
cross-programme validation, obtained without either side ever reading the other's data pipeline.

### The reported control-book Sharpe does not reproduce

The other programme's pre-declared trading rule was re-run in full — entry and exit z-score
thresholds, per-spread volatility-based stops, fixed-fractional risk sizing, volatility suppression
filters, a cooldown and gated-reentry mechanism, and a backwardation-only regime gate on the Brent
calendar spread — on the five live spreads, over this repo's development window, under two cost
models: this repo's own (materially more conservative) and a reimplementation of theirs.

| Metric | Theirs (2014–2023) | Ours, our costs | Ours, their costs |
|---|---:|---:|---:|
| Fixed-notional return | +85.1% | −1.1% | −1.1% |
| Equity-path return | +122.4% | −1.1% | −1.1% |
| Sharpe | 0.889 | **−0.16** | **−0.16** |
| Max drawdown | −7.30% | −1.9% | −1.9% |
| Number of trades | 333 | 57 | 57 |

**A material divergence on every axis, reported honestly rather than tuned away.** It was flagged
in advance as a live possibility given the data-corruption history.

The candidate explanations are recorded without isolating one:

- A development window that is longer and differently dated than their 2014–2023 tuning window.
- An independently built spread series from a different vendor.
- Reimplementation from specification rather than from their code — including an *approximate*
  p-value for the stationarity test, interpolated between three tabulated critical values rather
  than computed from the exact distribution.
- A documented simplification in book construction: five independently sized per-spread books
  pooled by dollar P&L, rather than one shared-equity risk engine enforcing real joint exposure and
  daily drawdown caps.

**The practical consequence: every downstream comparison against this control book is internal** —
structured versus unconditional, sign-flipped versus unconditional, screen-inclusive versus
screen-exclusive — never a validation of their absolute reported numbers. The noise floor on this
book (a 95% interval half-width of a few percentage points around a near-zero point estimate) states
plainly what any internal comparison can resolve.

### The new screen is strict enough to reject their own flagship spread

Both versions of their stationarity screen were rebuilt.

**The old screen fails its own random-walk check completely.** It tested stationarity of the 30-day
deviation from a 30-day rolling mean. Run on 20 synthetic pure random walks, it flags **all 20** as
stationary, with test statistics matching the astronomically small p-values that programme reported
for the same construction.

The reason is structural: detrending a random walk against its own rolling mean manufactures a
bounded, spuriously stationary residual almost by construction. A screen built this way carries no
information about genuine mean reversion.

**The new screen is properly calibrated but very strict.** It requires a stationarity test on the
level, *and* a variance ratio at two horizons, *and* a Hurst exponent below 0.5, *and* half-life
stability — the full-sample half-life inside a 3–60-day band, with at least three of four contiguous
sub-periods also in band.

Calibrated against 500 seeded random walks it comes in close to its nominal 5% false-positive rate
at both variance-ratio horizons (5.8% and 4.6%), so that component is not miscalibrated. But applied
to all 30 spreads here it passes only **8**, against the old screen's 23.

Strikingly, the Brent calendar spread — one of their own five live spreads, with a clean stationarity
rejection and the 42.7-day half-life corroborated above — **fails on the variance-ratio leg alone**
(z = +1.71, positive rather than the required negative).

**This is a real finding, not a screen bug.** That spread's daily changes show short-horizon
*positive* autocorrelation — momentum at the five-day horizon — layered on top of genuine
long-horizon mean reversion. The two diagnostics measure different things and can legitimately
disagree.

Whether the stricter screen earns its place in the trading universe, or is itself an example of a
mechanism that improves quality only by deletion, is left as a question for the follow-on notebook
to decide rather than assumed here.

### The trade-shape corroboration is the strongest evidence the mechanism is real

Their trade-pattern analysis was replicated on this repo's own 57-trade reproduction. Despite the
aggregate P&L divergence, the *shape* corroborates closely:

| | Here (n = 57) | Theirs (n = 9,545 pooled) |
|---|---:|---:|
| Does entry extremity separate winners from losers? | No (2.13 vs 2.23 median) | No (2.06 vs 1.95 median) |
| Stop-exit fraction, best half vs worst half | 0% / 82% | 0% / 85% |
| Loss-to-win size asymmetry | 2.09× | ≈3.4× |

An independently built book, roughly **170× smaller**, under a materially stricter cost model, still
reproduces the central pattern: **entry extremity does not discriminate winners from losers, and the
catastrophic tail comes almost entirely from stop exits**, while the bulk of trades exit cleanly
when the spread normalises.

That the shape survives even where the magnitude diverges is meaningful. It suggests the underlying
mechanism — a small number of adverse continuations doing most of the damage — is a genuine property
of this trading-rule family rather than an artefact of one programme's parameterisation or data.

**This is the empirical basis for notebook 011c's entry-time loss classifier** — worth building
properly, even though the honest prior informed by exactly this pattern is that entry-time features
will not predict the tail.

---

## What is pre-registered for the follow-on notebooks

Ten tests across three notebooks, with their firing criteria fixed before any backtest.

**Trial counts total 85.** The largest single component is a 36-cell grid (3 × 3 × 4) for the
reentry-mechanism test, and it is not to be reduced even if it proves unreachable — an unreachable
grid is itself the finding.

**Two competing trading universes are fixed here** — the cointegration-passing screen from notebook
010a versus the fuller eligible universe — with the resolution of two specific cross-programme
disagreements left to a paired comparison rather than assumed.

**Positioning data is recorded as a hard gap.** This repo's cache holds only crude oil, not the
grain series the other programme's proposal needs. It is out of scope for every downstream test, and
is not proxied.

**A holdout disclosure.** The holdout period is untouched here and remains unspent, but its
independence for commodity-spread strategies specifically is **reduced**: the other programme's own
held-out window overlaps it, and their held-out numbers were read while designing this notebook.
Every future write-up touching the holdout for the spread work must carry that disclosure. The
crypto momentum work is unaffected, since the corresponding crypto programme produced no held-out
numbers.

## One discrepancy flagged, not resolved

The carry-ratio primitive, implemented literally to specification, evaluates to approximately −1 at
the deep-contango boundary under this repo's sign convention — the opposite of the "+1 at the
contango ceiling" description in the same specification.

The function is implemented literally and the discrepancy is documented in place rather than
silently corrected. It affects nothing computed here, and is left for the first notebook that
actually consumes it to resolve against the other programme's own live output.

*Notebook: `src/research/011a_methodology_transfer_and_reproduction.ipynb`.*
