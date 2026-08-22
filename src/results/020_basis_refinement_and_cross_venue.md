# 020 — Refining the Basis Trade, and a Cross-Venue Funding Spread

## Two independent attempts

Notebook 018's funding basis trade is the closest thing to real alpha this repo has found in
thirty-one tests. It failed on two counts: it couldn't clear the tradeability bar, and it couldn't
show that its timing rule added value.

This notebook tests two independent, pre-registered responses.

### Attempt one: fix the construction

Notebook 018's own diagnostic explained *why* its return distribution was so extreme — skew of −11.65
and kurtosis of 849. The book was equal-weighted among however many symbols currently qualified, with
a cap of 10 above but **no floor below**. It held a single symbol on 5.4% of bars, and the five worst
single-bar losses were exactly those bars, each coinciding with a verified perpetual-market liquidity
collapse the trailing-median screen was too slow to catch.

Notebook 018 correctly refused to patch that after the fact. This is the pre-registered place to test
the fix:

- **A diversification floor** — require at least 3 qualifying symbols, and stand down entirely into
  cash rather than concentrate below that.
- **A slower, lower-turnover carry** — double the smoothing half-life, with thresholds re-derived from
  a 30-day rather than 15-day target hold.

**The hypothesis:** a floor removes the one- and two-symbol bars producing the extreme moments. Better
moments raise the deflation mechanically, and fewer catastrophic single-bar losses tighten the
interval. Both failing legs are therefore addressable **by construction, not by estimator repair.**

### Attempt two: a structurally different trade

The **spread** between two exchanges' funding rates on the same underlying. Short the
expensive-funding venue's perpetual, long the cheap one's. No spot leg needed, and cheaper to trade
(25bp round turn against 34bp) because two perpetual legs hedge each other.

**The hypothesis:** a funding *spread* between venues is a more arbitrage-like return driver than a
funding *level*, with lower directional and basis exposure and therefore possibly a better-behaved
return distribution — at the cost of a shorter usable history and a smaller universe.

## Results

| Question | Verdict | The number |
|---|:---:|---|
| **Refinement: does the mechanism survive the change?** | **Yes** | Pooled mean gross return 1.13e−04/period, HAC t = 11.71 |
| **Refinement: is it tradeable?** | **Yes** | Net Sharpe **3.887–3.889** at every offset, interval [2.06e−05, 6.98e−05] excludes zero, deflation **0.9999997** |
| **Refinement: does it beat notebook 018's own book?** | **No** | The paired interval [−2.20e−05, +5.62e−05] includes zero |
| **Refinement: genuinely neutral?** | **Yes** | Beta 0.0022 to basket, 0.0020 to Bitcoin |
| **Refinement: fundable on absolute performance?** | **Yes** | Sharpe 3.89, deflation 0.9999997 |
| **Cross-venue: does the mechanism exist?** | **No** | Pooled mean **−2.21e−05**/period, t = −6.73 — significant, but **negative** |
| **Cross-venue: is it tradeable?** | **No** | Net Sharpe 0.354, interval includes zero, deflation 0.082 |
| **Cross-venue: does the spread beat the level?** | **No** | The interval favours **the single-venue book** |
| **Cross-venue: genuinely neutral?** | **Yes** | Beta 0.0007 to basket, 0.0009 to Bitcoin |
| **Holdout access** | **Not granted** for either | Each requires a conjunction; neither is satisfied |

**The refinement works exactly as hypothesised and is dramatically better than notebook 018's book on
any absolute reading — and still doesn't clear its own paired-comparison bar.**

**The cross-venue hypothesis is not supported.**

## Everything fixed before a single byte was fetched

The floor value of 3 is derived from notebook 018's own diagnostic — the smallest integer eliminating
both the one- and two-symbol tails. The slow-carry constants are an exact 2× scaling of notebook
018's own ratio. (One consequence: the slow entry threshold equals notebook 018's *exit* threshold to
the last digit — an expected coincidence of the scaling, not a bug.)

The trial budget was itemised in advance and totals 32.

## Data

The refinement needed no new data — notebook 018's cache is reused unchanged.

The cross-venue trade needed a new fetcher for the second exchange. **Its kline endpoint silently
rejects the 8-hour interval** — returning a success code with an empty result and no error. Worked
around by fetching native 4-hour bars and aggregating pairs into buckets aligned with the first
exchange's.

One genuine asymmetry, **disclosed rather than "fixed" on either side**: the second exchange publishes
quote-denominated volume directly, which is better than the first exchange's, where it has to be
approximated from price times volume because the bulk archive drops the native field.

**Universe:** intersecting notebook 018's 126 spot-usable symbols with the second exchange's 725
perpetuals gives **93 symbols** — slightly above the expected range of 60–90, recorded honestly rather
than trimmed. Funding intervals across the intersection: 85 symbols match the first exchange's
cadence and 8 are on a shorter one, and both divide the bucket evenly, so no symbol needed excluding.

The second exchange's published taker fee **could not be independently confirmed** through its public
market-data API — no endpoint exposes it. Disclosed as unconfirmed rather than silently assumed.

All 93 symbols completed in both windows with zero truncation and zero errors, in about 20 minutes
against a 90-minute budget. Resampling and aggregation deliberately do **not** happen at fetch time —
they live in tested, pinned functions, so the fetcher's only job is a faithful, resumable copy of the
raw response.

## The reproduction tripwire

Before anything else, the refined library was proved to reduce **exactly** to notebook 018 when the
floor is set to 1 — both synthetically and, more importantly, element-wise identical on the real
panel.

And notebook 018's stored net Sharpe was reproduced **to the full stored precision, absolute
difference 0.0.**

## Probing both mechanisms without costs

### The refinement

| Variant | Single-symbol bars | Gross skew | Gross kurtosis |
|---|---:|---:|---:|
| Notebook 018 baseline (reproduced) | 5.44% | −11.65 | 849 |
| **Floor only** | **0.00%** | **+0.53** | **18.7** |
| Slow carry only | — | −10.66 | 529 |
| **Both** | **0.00%** | **+0.34** | **16.1** |

Reproducing notebook 018's 5.4% figure to within 0.4 percentage points was a free tripwire, and it
passed. **Most of the improvement comes from the floor, not the slower carry**, though combining them
is best.

The mechanism survives cleanly: pooled mean gross return of 1.13e−04 per period at HAC t = 11.71.

### The cross-venue spread

The raw pooled statistic is significant but **negative** — pooled, the second exchange's funding runs
*higher* than the first's, not the other way round. A genuine, disclosed finding rather than a bug,
confirmed by checking the decomposition.

The funding-spread term dominates the price-divergence term both in magnitude (2.20e−05 against
1.00e−07) and in significance (t = −15.53 against 0.05) — **exactly as notebook 018 found for its own
single-venue decomposition.** Funding is the drift and price divergence is noise around it, on both
mechanisms.

551 bars were excluded from the descriptive statistics under notebook 018's own sanity bound, reused
verbatim, touching two symbols — consistent with the earlier finding that a handful of feed-artefact
bars can dominate a pooled statistic. Never excluded from the backtest universe.

Spread persistence: 32% of above-threshold runs clear the break-even hold, against 44% for the
single-venue trade. Plausible, somewhat weaker persistence.

## A prediction, written down before the backtest ran

Reusing the device from notebook 019, and using only the no-cost numbers above:

| | Predicted skew | Predicted kurtosis | Sharpe used | Counterfactual deflation | Clears 0.95? |
|---|---:|---:|---:|---:|---|
| Refinement | +0.339 | 16.08 | 0.5766 (notebook 018's own) | **0.154** | No |
| Cross-venue | +0.168 | 51.20 | 0.854 (the single-venue comparator's) | **0.346** | No |

**The prediction: neither headline book would clear the deflation bar on improved moments alone. Each
would need an actual Sharpe substantially above its counterfactual baseline.**

Both predictions were falsifiable, and **both resolved correctly.** The refinement's actual Sharpe of
3.89 did substantially exceed notebook 018's, and it cleared. The cross-venue book's 0.354 did *not*
exceed the single-venue comparator's 0.854 — it came in lower — and it failed.

**The prediction machinery worked exactly as intended, and its qualitative call was right in both
directions.**

## The books

| Variant | Net Sharpe | Max drawdown | Turnover/year | Note |
|---|---:|---:|---:|---|
| Notebook 018, reproduced | 0.577 | −8.6% | 56.9 | Matches exactly |
| Floor only | 2.757 | — | — | |
| Slow carry only | 1.352 | — | — | |
| **Refinement (both)** | **3.887–3.889** | **−3.4%** | **44.2** | Offsets agree to 3 decimals |
| Refinement, always on | −0.415 | — | — | Timing still matters |
| **Cross-venue** | **0.354** | −3.5% | 58.1 | Below the 0.5 bar |
| Cross-venue, slow carry | 0.961 | −2.4% | 47.6 | Notably stronger — but not the pre-registered headline |
| Single-venue on the same universe | 0.854 | −8.3% | 53.4 | **Beats the cross-venue book outright** |
| Cross-venue, always on | −1.869 | — | — | |

**The concentration diagnostic tells the story.** Median symbols held is unchanged at the cap of 10,
but the worst single-bar loss drops from **−5.7%** (one symbol alone) to **−0.38%** — a 10-symbol bar
that still included that same problem symbol. **The floor kept the book diversified through exactly
the kind of event that broke notebook 018.** The book sits in cash 5.2% of bars, which is the floor's
cost, and it's cheap.

The cross-venue book holds a median of 5 symbols — well below its cap, so it rarely fills up — and
sits in cash 19.1% of bars. Substantially more stand-down, consistent with a thinner, more marginal
opportunity set.

## Ablations

**The refinement is robust.** Varying the floor between 2 and 5 barely moves it (3.884 to 3.920) —
reported as robustness, not grounds to retune, and the headline stays at the pre-registered value.
Excluding the two collapsed tokens barely moves it either (3.985), which is a **direct contrast with
notebook 018**, whose worst bars were driven by exactly that kind of single-symbol event. Removing the
hysteresis band costs about half the Sharpe.

Cost sensitivity **never crosses the 0.5 bar within the tested range** — still 0.812 at 51bp, 1.5× the
real cost. The true break-even is somewhere beyond that, reported as a bound rather than extrapolated.

**The cross-venue book is fragile.** Removing the hysteresis band is deeply negative (−3.679, a far
larger relative collapse than the refinement's equivalent — sign-flipping without a band is
expensive). Excluding its own top two contributing symbols **flips it negative** (−0.278), a real
concentration finding that its median-5-held diagnostic alone doesn't reveal. The single-leg
neutrality control is negative and structurally unlike the book's own return, which is evidence
against a disguised directional bet.

And the decisive number: **its interpolated break-even cost is 24.95bp against a real cost of 25bp.**
**The book operates almost exactly at its own break-even.** That single figure explains the weak
headline directly.

## The holdout was not spent

Neither mechanism satisfied its conjunction. The holdout runner was invoked anyway, to *demonstrate*
the fence rather than merely assert it: it read the access block from the stored results, found both
false, and exited with an error **without constructing any path into either holdout directory.**

Verified by search: the two literal holdout directory strings appear in exactly one file across the
whole notebook's codebase.

## Bugs found

**One real bug**, caught by the fencing check before it was ever exercised: a stray, unused holdout
directory constant declared in the library alongside its development counterpart for symmetry, and
never referenced. It caused no wrong number — it was dead code — but it violated the invariant that
exactly one file may name a holdout directory. Removed.

**One near-miss, not a bug.** The paired comparison failing despite the refinement's dramatically
higher absolute Sharpe looked, on first read, like a possible computation error. Investigating by
comparing interval widths showed the baseline's own width (≈8.3e−05) and the difference series' width
(≈7.8e−05) are close — consistent with the difference series **inheriting most of the baseline's
volatility bar by bar.** A real statistical property of a paired comparison against a noisy baseline,
not a bug.

## Bottom line

**The refinement confirms its hypothesis at every level the notebook could test it, and is on any
absolute reading a dramatically better book.** The mechanism survives, tradeability clears, neutrality
holds, the fundable flag fires, the deflation clears 0.95 by six nines, and the worst single-bar loss
drops by more than an order of magnitude.

**But the pre-registered bar for "the refinement adds value" is a stricter, paired question** — does
the difference from notebook 018's own book clear an interval — **and that does not fire.** The holdout
stays exactly as unspent as notebook 018 left it. Under a conjunction requirement, that is the correct
outcome, not a partial-credit situation.

**The cross-venue trade does not support its hypothesis.** The mechanism is real in a narrow,
direction-following sense — a significant positive sign-following return exists — but the headline book
is weak, sits almost exactly at its own break-even cost, and **loses outright to the much simpler
single-venue trade on the identical universe.** A clean, informative negative result rather than a
data-quality artefact: every symbol fetched cleanly, the reproduction tripwire passed exactly, and
every robustness check behaves as expected.

**The central methodological result is that the pre-registered prediction worked, in both
directions.** The counterfactual deflation computed before any backtest ran correctly anticipated that
neither headline book would clear the bar on improved moments alone — and then correctly anticipated
exactly what *would* need to happen for each. Both conditional predictions resolved correctly.

That closes this notebook cleanly on both mechanisms **without needing the estimator question at
all.** Notebook 018's hinge — that its deflation leg was never contingent on the estimator but on the
return distribution — is now validated directly: fixing the distribution cleared the deflation, and the
higher paired bar still wasn't met.

## What to test next

- **The paired-comparison gap is the sharpest open question here.** The refined book is unambiguously
  better in absolute terms; the open question is purely **statistical power on the paired
  comparison.** A longer paired series, or a variance-reduction technique on the difference series —
  such as a coupled bootstrap conditioning on the shared market-regime component both books are
  exposed to — could sharpen this without touching the frozen construction.
- **The slow-carry cross-venue variant scored 0.961, nearly 3× the pre-registered headline**, and was
  not itself gated as a headline. That is exactly the kind of after-the-fact-visible number this
  repo's discipline forbids adopting retroactively — but it is a legitimate, pre-registerable
  candidate headline for a future notebook starting fresh.
- **A joint floor-plus-slow-carry cross-venue construction**, mirroring exactly what worked for the
  refinement. The slow variant already outperforms, and the refinement's own result shows the two
  changes **compounding rather than merely adding.**
- **Why does restricting to the intersected universe make the plain single-venue trade so much
  stronger** (0.854 against notebook 018's full-universe 0.577)? A survivorship or liquidity story —
  the second exchange's listings skewing toward larger, more liquid names — is plausible but untested.
  Worth its own descriptive pass before building more machinery on top of it.

*Notebook: `src/research/020_basis_refinement_and_cross_venue.ipynb`. Notebook 018's numbers,
verdicts and text stand exactly as recorded; nothing there was edited.*
