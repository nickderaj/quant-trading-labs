# 018 — The Crypto Perpetual Funding Basis Trade

## The trade

Buy the spot asset, short the perpetual future on the same asset. The two legs cancel, so the
position has essentially no exposure to the price. What it collects is the **funding payment** —
the 8-hourly cash flow perpetual futures use to keep their price tethered to spot.

**This is the only structurally non-directional trade this research programme has attempted.** Every
prior test bet, in one way or another, that some feature predicted the direction of a price or a
spread. This one bets that a cash flow is positive more often than it costs to collect.

Notebook 009 shortlisted it with strong evidence — by analogy to the roughly $4 trillion Treasury
cash-and-carry basis trade — but parked it because the repo's cached data hadn't been verified to
include a spot price series. That verification is done here: the exchange's spot, futures and
premium-index archives all serve native 8-hourly data going back to the start of this repo's crypto
sample.

## What had to be true, and what happened

| Question | Result | The number |
|---|:---:|---|
| **Does the mechanism exist?** Positive carry net of basis drift, before costs | **Yes** | Pooled mean gross paired return 4.30e−05 per period, HAC t = 3.27 |
| **Is it tradeable?** The timed book survives costs | **No** | Net Sharpe +0.577 (clears 0.5) **but** the bootstrap interval on net return includes zero **and** the deflation is 0.186 against a 0.95 bar |
| **Does timing add value?** Timed beats always-on | **No** | The interval on the difference includes zero, though the point estimate strongly favours timing |
| **Is it genuinely neutral?** Not a disguised long | **Yes** | Beta of 0.0005 to the crypto basket and 0.0016 to Bitcoin, both far inside ±0.10 |
| **Fundable on absolute performance?** | **No** | Sharpe clears; deflation fails |
| **Holdout access?** | **Not granted** | Requires both tradeability and timing value; neither fired |

**A genuine, statistically significant funding carry exists in this data, and the hedge genuinely
works — but it is not a tradeable strategy by this notebook's own pre-declared bar.**

## Everything fixed before any data was fetched

Round-turn cost of 34 basis points. A 15-day target hold. Entry and exit carry thresholds, with the
exit at half the entry — a hysteresis band. A 7-day exponentially-weighted half-life for the carry
estimate. A cap of 10 simultaneous positions. A $5 million per day liquidity floor on **both legs
independently**. And 18 trials — 3 books × 4 origin offsets, plus 6 ablations — never revised. Every
configuration actually run was one of the 18 declared upfront.

Two obvious objections were also answered in advance, before any result existed:

**"Notebook 007 already tested funding carry and found nothing."** That test was a cross-sectional,
perpetuals-only, dollar-neutral book betting that *funding predicts price direction*. Its failure mode
— rank churn producing 674–681 round trips a year — doesn't apply to a per-symbol, spot-plus-perp,
delta-neutral position with no cross-section to rank against. What *does* transfer is notebook 007's
own prescribed fix: a no-trade band tuned to funding's 8-hourly cadence, which is exactly the
hysteresis used here — built fresh rather than by reusing the cross-sectional machinery, which would
have reimported the failure mode.

**"Isn't this just crypto beta with extra steps?"** That's the neutrality check, and it passes
cleanly.

## Data

126 of 128 candidate symbols have a usable spot leg. The two that don't are a capacity finding, not
an error.

The holdout window was fetched in the same pass, into a **separate cache directory that no
development-phase loader can read.** The panel loader structurally guards past the holdout start date,
and exactly one script ever names the holdout path.

## Does the mechanism exist? Two bugs found first

**A units error in the break-even calculation.** The first implementation divided the round-turn cost
by a daily rate where it needed a per-period rate, landing on **1,133 periods** instead of **34**.
Caught because the persistence check came back at 0.1% — implausibly low against the prior.

**A handful of bad bars distorting two pooled statistics.** The premium-index cross-check's pooled
correlation came back at **−0.08**, materially disagreeing with expectation, even though the
per-symbol distribution was healthy (median 0.744).

Traced to two symbols. One had a perpetual price frozen at a stale value for several days — verified
directly against the cached bars, with volume at exactly zero — while spot kept moving, producing a
computed basis of **+275%**. The other is a real 2022 collapse where dividing by a near-zero spot
price produces values over 100×.

Both are excluded from the *descriptive* checks by a stated sanity bound, with the exclusion count
reported (5,066 of 461,298 observations, touching 14 symbols) — **not** excluded from the backtest
universe. After exclusion, the premium-index correlation recovers, and the funding-versus-basis
identity check's correlation rises **from 0.48 to 0.994**.

With both fixed:

**The mechanism exists.** Pooled mean gross paired return of 4.30e−05 per period, HAC t = 3.27, on
461,298 observations.

**Funding dominates the basis-change term, on both magnitude and significance.** Funding's pooled mean
is highly significant (t = 20.5); the basis-change term's mean is **not distinguishable from zero**
(t = 0.33). Exactly as predicted: the basis term is mean-reverting and roughly zero-mean — noise around
the funding drift, not a competing source of return.

**The claimed 2025 funding decay does not yet show up here.** The development window only reaches mid
2025, and pooled funding by year is actually *higher* in 2024 (mean 1.10e−04, t = 45.6) than in 2023
(3.60e−05, t = 6.1), with 2022 slightly negative. This is reported as a genuine limit rather than
smoothed over: the interesting comparison sits inside the holdout, which was never spent.

**Funding regimes persist long enough to matter, but not by a wide margin.** 44% of above-threshold
carry runs last at least 34 periods, the break-even hold. Real and exploitable, but not overwhelming.

## The backtest

| Book | Gross Sharpe | Net Sharpe | Max drawdown | Turnover/year |
|---|---:|---:|---:|---:|
| **Timed** | 2.66 | **+0.577** | −8.6% | 56.9 |
| Always on | 0.21 | −0.415 | −25.0% | 15.2 |
| Cash | 0.0 | 0.0 | 0.0% | 0 |

All four origin offsets agree to three or more decimal places. That check is **vacuous** for a
fixed-parameter, non-refitting design like this one — the same pattern notebooks 012 and 013 found —
and is disclosed rather than presented as robustness.

**Tradeability fails on the interval and the deflation, not the Sharpe.** Net Sharpe clears the 0.5
bar at every offset. But the 95% interval on net return is [−1.31e−05, +7.02e−05], which includes
zero, and the deflation is 0.186 at the honest count of 18 trials.

That deflation figure carries a pre-registered caveat: **this repo's deflation estimator is
known-harsh for exactly this kind of trial family** — near-identical origin offsets producing
near-identical Sharpes. See the addendum at the end for how that was resolved.

**Timing fails despite a large point-estimate gap.** The timed book's +0.577 against always-on's
−0.415 is nearly a full Sharpe point, and the mechanism makes sense: timing only holds positions when
carry clears the threshold, so its **gross** Sharpe is 12× always-on's. But turnover is *higher* for
timing (56.9 against 15.2 a year, since always-on barely changes membership), so the improvement has
to clear a real cost hurdle — and the paired interval on the difference still includes zero.

Read plainly: **timing looks like it helps a great deal, and the data cannot yet rule out that it
doesn't.**

**Neutrality passes cleanly.** Beta of 0.0005 to the crypto basket and 0.0016 to Bitcoin — two orders
of magnitude inside the bound, at every offset. Confirmed independently below.

### Why the deflation inputs are so extreme: a concentration finding

Rather than accepting the extreme skew (−11.5) and kurtosis (817) at face value, they were traced.

The timed book holds a median of 10 symbols — the cap — and is at the cap 54% of bars. But it is down
to **a single symbol 5.4% of bars.** The equal-weight-among-qualifiers construction has **no
diversification floor below the position cap.**

The five worst single-bar returns are dominated by exactly this: one symbol alone on one day at
−5.7%, and another alone or nearly alone across four bars at −0.6% to −2.0%.

Both trace to a **real perpetual-market liquidity collapse**, verified directly against the cached
bars: open, high, low and close all identical with zero volume, for several consecutive days. One is
an exchange rebrand transition already documented in notebook 013; the other a genuine multi-day
zero-volume stretch.

**During a zero-volume stretch the short perpetual leg cannot actually be hedged** — there is no price
discovery to hedge against — and the pre-registered 30-day trailing **median** liquidity screen is slow
to catch a sudden collapse. It does catch one of them, one bar *after* the worst loss.

This is a structural property of a median-based screen combined with an equal-weight book with no
floor, **not a code defect** — and it is deliberately **not fixed here** by changing the frozen
liquidity floor, lookback or construction rule. That would be exactly the kind of after-the-fact
tuning the pre-registration forbids. It is reported as a capacity and robustness finding.

## Controls and ablations

| Exhibit | Result | Finding |
|---|---|---|
| **No hysteresis** | Net Sharpe +0.577 → **−0.726**, turnover 56.9 → 95.4/yr | The band is genuinely load-bearing — notebook 007's own prescribed fix matters exactly as hypothesised |
| **Perpetual leg only** (no hedge) | Beta −0.897 to basket, −1.103 to Bitcoin | **Confirms the hedge, not luck, removes beta.** Max drawdown on the unhedged variant is **−99.6%** — a near-total wipeout showing why the hedge matters practically, not just statistically |
| **Excluding the two collapsed tokens** | +0.577 → **+0.562** | The headline does not depend on either collapse |
| **Cost sensitivity** (0/17/34/51bp) | Sharpe 2.66 / 1.62 / 0.577 / −0.450 | Crosses zero between 34 and 51bp — **break-even is around 43–44bp**, about 1.3× today's retail rate. A lower institutional fee tier would clear it by a wider margin |
| **Leverage 3× and 5×** | Sharpe 0.52 / 0.45 — but the 1% expected shortfall on the perpetual leg's own 8-hour return is **13.9%** | At 3× and 5×, that is **42% and 70% of deployed capital in a single bad 8-hour period.** The levered Sharpes should not be read as investable without this attached |
| **By year** | +7.56 (2021 H2), −0.19 (2022), +1.72 (2023), +0.67 (2024) | The 2021 figure is a small sample during crypto's highest-funding era and shouldn't be over-read. 2022–2024 show a real if noisy decline |

## The holdout was not spent

Access required both the tradeability and timing checks to fire. Neither did.

The holdout runner — the only file that names the holdout directory or reads past the cutoff for this
notebook — was invoked once, deliberately, to verify that it refuses correctly. It printed the access
block read back from the stored results, declined, and exited with an error **without ever loading the
holdout data.** Verified, not merely asserted.

## Bugs found

Two real bugs, both caught by suspicious numbers before being trusted:

1. **The break-even units error**, caught because a persistence fraction came back implausibly low.
   Fixed; the fraction is now 44%, matching the prior.
2. **Two symbols' price-feed artefacts distorting pooled statistics**, fixed by making the per-symbol
   distribution the primary comparison and excluding implausible bars from pooled statistics, with the
   count disclosed rather than silently absorbed.

**One near-miss that wasn't a code bug.** A naive read of the extreme skew and kurtosis could have been
reported as "the estimator is unreliable here" and left at that. Tracing it instead found the real,
structural, disclosable concentration finding above — rather than hand-waving the number away or,
worse, silently patching the liquidity screen to make it disappear.

**Also disclosed, not a bug:** a join suffix parameter does nothing unless column names already
collide. Three call sites relied on it and would have computed beta against the wrong column — caught
by the scripts erroring loudly rather than producing a silent wrong number. Fixed by renaming
explicitly before joining.

## Bottom line

**The mechanism is real.** A statistically significant funding carry exists in this repo's own data,
driven by funding rather than basis drift, exactly as claimed.

**The trade is not demonstrably tradeable by this notebook's bar**, and the holdout stays unspent.

But the verdict is more informative than a flat null. Net Sharpe clears the absolute 0.5 bar at every
offset, and timing shows a large, economically sensible edge over always-on — 12× the gross Sharpe.
Neither the interval on net return nor the paired interval on timing's value-add can yet rule out
zero.

**And the neutrality check passes cleanly, confirmed two independent ways** — direct beta measurement,
and the unhedged ablation's beta collapsing to −1 when the hedge is removed.

This is the first test in this programme's thirty-one-test history where a **structurally different
mechanism — a cash flow rather than a forecast — shows up statistically significant before costs and
survives its own most direct validity check.** It simply does not clear the higher bar of being a
demonstrably tradeable edge net of realistic costs and multiple-testing correction. The Sharpe of
1.2–2.4 the pre-registration expected going in was not met; the honest figure is 0.577, inside the
wider "plausibly under 1.0" band the same document flagged as the realistic floor.

**Real reasons for caution about extrapolating even the point estimate:**

- A liquidity floor with no diversification requirement leaves the book concentrated in a single name
  5.4% of the time, and the worst outcomes in the whole backtest come from exactly that meeting a
  liquidity collapse the screen is too slow to catch.
- **Reverse carry was never tested, by design.** The strategy as built goes flat, not short, in
  negative-funding regimes — which caps upside relative to external reports that don't appear to
  impose this constraint.
- Leverage, the only way to make this size-competitive, carries a 1% expected shortfall on a single
  8-hour period equal to 42–70% of deployed capital on the unhedged leg alone.

## What to test next

- **A diversification floor** — a minimum symbol count below which the book stands down entirely rather
  than concentrating. This directly addresses the concentration finding, but it is a new
  pre-registerable design choice for a future notebook, not a retroactive edit here.
- **A faster liquidity screen**, such as a same-day zero-volume veto layered on the trailing median.
  This closes the specific detection lag found here, at the cost of another parameter and a
  corresponding increase in the trial count.
- **A lower-turnover carry construction** — a slower smoothing, or a wider band derived from a longer
  target hold — might clear the interval this design misses by a small margin. Any such change needs
  its own pre-registration.
- **Fee-tier sensitivity as a standing exhibit.** The trade clears zero around 43–44bp against a
  current retail round turn of 34bp. Worth tracking against actual institutional fee schedules rather
  than re-deriving per notebook.

---

## Addendum — re-scored under notebook 017

Notebook 017 set out to test the caveat this document recorded: that this repo's deflation estimator
likely over-penalises a near-identical-offset trial family, because it builds its benchmark from the
sampling error of a single Sharpe rather than the observed spread of the trials actually run.

**It confirmed the defect is real** — by simulation, at 20,000 replications across 756 grid cells, the
current estimator's false-positive rate collapses toward zero as inter-trial correlation rises. Exactly
the pattern this result was suspected of.

**It did not find an adoptable repair.** Two candidates pass calibration cleanly but lose real
detection power in the independent-trials case — a "no free lunch" clause built into that notebook's
pre-registration specifically to catch that trade-off. A third is powerful enough but badly
miscalibrated, over-firing in roughly two-thirds of null cases. Per the pre-registered rule, none was
adopted and the estimator was left unchanged.

**That would ordinarily leave this case open. It isn't**, because of a second, independent finding.
This book's own stored inputs — sample skew −11.5 and kurtosis 816.9, a genuinely extreme regime rather
than a stylised example — **cap what *any* spread-based repair could ever have produced for this exact
result at 0.83**, below the 0.95 bar. That ceiling holds for every candidate evaluated, adopted or
not.

In other words, **this result's deflation leg was never actually contingent on which repair, if any,
was settled on.** Even in the counterfactual where the source paper's own repair had passed both
checks, the corrected value could not have exceeded 0.83.

**The practical outcome is unchanged.** Every verdict stands exactly as recorded, and the holdout
remains unspent — access requires both the tradeability and timing checks, neither of which fires, and
neither of which has a leg an estimator change could affect. Tradeability's failure is independently
sealed by its bootstrap interval; the timing check carries no deflation leg at all.

The asterisk this document carried against the deflation figure is removed — not because the
underlying worry was unfounded, since it wasn't, but because this book's own numbers settle the
question independently of it. **No correction to this estimator, however it eventually gets fixed, can
move this result.**

Nothing above this section has been edited. The trial count stays 18, no backtest was re-run, and the
notebook was appended to rather than re-executed.

*Notebook: `src/research/018_funding_basis_trade.ipynb`. Notebook 017's own write-up:
`src/results/017_deflated_sharpe_correction.md`.*
