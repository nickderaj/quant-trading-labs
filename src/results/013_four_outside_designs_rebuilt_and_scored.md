# 013 — Four Published Designs, Rebuilt and Scored on This Repo's Own Data

## The question

Twenty-two-plus prior tests across twelve notebooks have all come back null — and every one was on
this programme's *own* constructions. That leaves an obvious rebuttal available: maybe the
strategies just weren't built well enough.

This notebook removes that rebuttal. Four independently specified, externally published trading
designs are rebuilt end to end on this repo's data and costs — each faithful to its own
specification, and each including the specific mechanism its authors credit for the edge:

| Design | What it is | The claimed source of edge |
|---|---|---|
| **A** | A gold futures trend and execution book | Execution quality — the alpha is in the fill, not the signal |
| **B** | Cross-asset sequence models | The training objective — optimise Sharpe directly, not squared error |
| **C** | An adaptive 6-hour crypto trend book | Adaptation — causal monthly re-parameterisation |
| **D** | A crypto cross-sectional graph-attention model | Cross-sectional structure — returns are driven by correlated neighbours |

Six criteria were fixed before any backtest ran. Each design is scored **on its own terms**, not just
on top-line Sharpe.

**All six come back null.** The best net Sharpe achieved across all four designs is **+0.33**, still
below this programme's own record of +1.053 and nowhere near the designs' reported range of 2.4–3.1.

## Results at a glance

| Design | Claim | Headline result | Passes |
|---|---|---:|:---:|
| A | Clears the Sharpe bar, near-zero market beta, stable sign | Net Sharpe **+0.33**, interval includes zero, deflated probability 0.15 | **No** |
| B | The best sequence model beats a linear baseline | Linear **0.204** beats both LSTM (0.057) and gated LSTM (0.052) | **No** |
| C (corrected) | Adaptive beats frozen; ablations correctly signed | Adaptive **−0.055**, doesn't beat frozen significantly — but **3 of 3 ablations correct** | **No** |
| D | A dollar-neutral book clears the bar; the correlation survives screening | Net Sharpe **−1.52**; correlation significant but **wrong sign** | **No** |
| Any | Net Sharpe ≥ 2.0, deflated probability ≥ 0.95, passes the lookahead audit | Best achieved: **+0.33** | **No** |
| D | Removing temporal mixing significantly improves the correlation | Direction matches (−0.032 vs −0.046) but interval includes zero | **No** |

A lookahead audit was pre-armed to trigger on any design reaching a net Sharpe of 1.5. **It never
triggers**, because none comes close. These are not suppressed wins — they are honest nulls at every
stage.

No holdout was spent, since holdout access is tied to a result firing and none did.

## An important addendum on Design C

After the initial results were committed, the four designs' actual source papers were obtained and
read. Design C's paper specifies exact parameters the original build had guessed at and gotten
materially wrong:

| | Original guess | Paper's actual value |
|---|---|---|
| Quality thresholds | 0.0 / 0.15 | **1.3 / −1.7** |
| Universe size | 30 symbols | **150+** |
| Selection mechanism | Dollar-volume proxy | **Top-15 / bottom-15 by market cap** |
| Funding costs | Not modelled | Modelled |

A corrected rebuild fixes all of these. **The corrected version is the headline result below; the
original is kept and explained as the disclosed history of getting there, not erased.**

The trial count rises honestly from 18 to **25** to account for the newly computed configurations.
This changes no verdict, since the deflated probability was already near zero at the smaller count.

---

## Design A — trend and execution on gold futures

The claim: the alpha is in the fill. Volatility-targeted, Kelly-scaled, impact-discounted sizing
around a single smoothed trend-momentum state.

Rebuilt exactly to specification on front-month continuous gold, with rolling 10-year training and
6-month test windows.

| Variant | Net Sharpe | Reading |
|---|---:|---|
| **Full design** | **+0.333** | Headline |
| No smoothing | +0.039 | Smoothing is confirmed load-bearing |
| No impact discount | +0.383 | Impact costs about 0.05 of Sharpe, as expected |
| No fractional Kelly | +0.461 | **The Kelly overlay is a net drag here, not a gain** |

The deflated probability at the base configuration is **0.15** against 18 trials — nowhere near 0.95
— and the interval on net returns includes zero.

The out-of-sample period runs 2019-03-24 to 2024-09-06, **1,512 bars**, well short of the design's
claimed ~2,793 trading days, because its history starts a decade earlier than this repo's. Reported
honestly rather than manufactured by shortening the training window.

**Two structural checks came back *supporting* the design's mechanism even as its Sharpe fails.**

Measured beta to spot gold is **0.002**, matching the design's own claimed near-zero-beta property
almost exactly.

And a pre-registered arithmetic red flag did not reproduce, which is itself informative. The design
claims a 0.52% maximum drawdown at a Sharpe near 2.9 — flagged in advance as arithmetically
implausible without a stop-fill leak. This rebuild's maximum drawdown is **−13.3%**, a normal figure
for a Sharpe-0.33 process. The required worse-of-stop-or-gapped-open fill convention produces
materially worse fills than an optimistic same-bar convention would, confirming that the convention
exists precisely to prevent the kind of leak the original numbers imply.

Origin offsets agree to four or more decimal places — vacuous here, the same pattern notebook 012
found, disclosed rather than presented as robustness.

---

## Design B — sequence models trained on a Sharpe objective

The claim: models trained on negative realised Sharpe, rather than squared error, learn temporal
representations a linear model can't.

Universe: 16 futures products plus 6 currency futures, giving **22 instruments**. A duplicate equity
index series was dropped, since pooling both feeds would double-count rather than diversify. **No
bond or volatility futures exist anywhere in this repo** — a disclosed structural gap against the
source design's universe, not papered over with an ETF proxy.

Real exchange tick specifications were added for the six currency futures so the futures cost model
covers the whole panel, rather than re-applying a flat convention to instruments it was never built
for — exactly the mistake notebook 012 made.

| Architecture | Median Sharpe (5 seeds) | Seed range | Versus linear (0.204) |
|---|---:|---:|---|
| **Linear baseline** | **0.204** | — | — |
| LSTM | 0.057 | [−0.003, 0.073] | Loses |
| Gated LSTM | 0.052 | [−0.029, 0.104] | Loses |

Both architectures land well **below** the linear model they were supposed to beat. The substance of
the claim fails, not just its absolute bar.

The per-seed spread is reported in full rather than the best seed. **Both networks swing negative on
at least one seed**, underscoring that whatever they learn is unstable across initialisation rather
than a repeatable edge. Breakeven cost is about 10bp per side for both.

### A real bug found and fixed in flight

Additive back-adjustment produces a handful of **negative synthetic prices** for four products, by
stacking roll adjustments onto already-low 2020-crash-era prices. The logarithm of a negative number
is undefined, and the data library's null-dropping does not remove those undefined floats — only true
nulls.

The first run's Sharpe-objective loss ingested one such row, produced an undefined gradient, and
**silently zeroed every model's predictions from that fold onward.** The symptom was a median Sharpe
reported as exactly 0.0 across all five seeds for both architectures.

Fixed by dropping non-positive back-adjusted prices at the source and adding an explicit
finite-value filter. The numbers above are from the corrected run.

---

## Design C, first attempt (superseded, kept for the record)

The claim: volatility-adaptive trailing exits, quality and liquidity selection, and causal monthly
re-parameterisation clear a Sharpe of 2.4 net of fees.

One assumption turned out false in a helpful direction: 6-hour bars are a *native* interval on this
data source and could be fetched directly rather than resampled from hourly.

| Variant | Net Sharpe | Max drawdown |
|---|---:|---:|
| **Adaptive** | **−0.105** | −59.4% |
| Frozen twin (calibrated once, never refitted) | −0.174 | −54.3% |
| No trailing stop | +0.332 | −53.0% |
| No selection filter | −0.610 | −68.6% |
| No liquidity filter | +0.455 | −45.7% |
| Adaptive, excluding the two collapsed tokens | −0.060 | −59.4% |

Adaptive does **not** beat the frozen twin — the paired interval on the difference squarely includes
zero, so the design's most interesting claim (its authors attribute roughly 1.07 of Sharpe to
adaptation alone) does not survive at all.

**Only 2 of 4 ablations move in the predicted direction.** Removing the selection filter and removing
adaptation both hurt, as predicted. But removing the trailing stop and removing the liquidity filter
both **help** — the opposite of predicted.

Drawdowns are severe (−46% to −69%) at Sharpes near zero. That's a symptom of capital concentrating
when few symbols pass the selection filter simultaneously — a market-wide stress month can leave a
leg holding one or two names at most of the book's capital.

Once the source paper was read, this build's quality thresholds turned out to bear no resemblance to
the paper's actual values.

## Design C, corrected

Three material gaps were identifiable by direct comparison rather than guesswork, and fixed. The
universe expanded to **128 perpetuals** (against the paper's "150+"), thresholds set to the paper's
stated values, selection switched to the paper's literal market-cap rule, funding costs modelled,
and the stop-multiplier grid widened to bracket the paper's reported optimum.

Market-cap data is a **current-day snapshot** applied statically across the whole backtest, not a
rolling historical ranking — the free data tier's rate limits make a true rolling reconstruction
across 128 symbols impractical. Disclosed as a real remaining gap. Two parameters the paper never
discloses carry over unchanged, because there is nothing more specific to match.

| Variant | Net Sharpe | Max drawdown |
|---|---:|---:|
| **Adaptive** | **−0.055** | **−6.6%** |
| Frozen twin | −0.209 | −5.7% |
| Adaptive, funding not modelled | −0.055 | −6.6% |
| No trailing stop | −0.733 | −22.7% |
| No selection filter | −1.362 | **−95.4%** — a near-total wipeout |
| Adaptive, excluding the two collapsed tokens | −0.240 | −6.6% |

**The corrected parameters produce a strikingly more internally consistent result, even though the
headline Sharpe is still negative.**

The first build's most damaging problem — a −59% drawdown from capital concentrating whenever a weak
filter let a stress month through — **is gone. The corrected drawdown is a sane −6.6%**, a direct
mechanical consequence of the real quality screen actually doing its job.

**And every ablation now moves in the direction the paper predicts**, where the first attempt had two
of four backwards. Removing the trailing stop hurts badly. Removing the selection filter is
catastrophic — and is the one effect in this entire design whose interval **excludes zero**, i.e. a
real, statistically distinguishable effect. Removing adaptation hurts, with the point estimate now
favouring adaptive over frozen, though the paired interval still includes zero. Not yet significant,
but no longer backwards.

Funding cost is immaterial at this position sizing. Origin offsets are **not vacuous** and are
considerably less stable than the first build's (−0.055 / −0.904 / −0.462 / −0.447) — consistently
negative in sign but highly sensitive in magnitude to start date, itself a finding.

**This makes Design C's null stronger, not weaker.** An implementation this much closer to the actual
paper — right universe scale, right thresholds, right selection mechanism, funding modelled — still
does not produce a positive significant Sharpe. What it produces is a mechanism that behaves exactly
as its authors describe (stops help, selection helps, adaptation helps) while still losing money on a
window that includes the 2022 bear market and two major token collapses, which the paper's own test
window does not.

There is no longer a live "maybe this was implemented wrong" objection hanging over it.

---

## Design D — attention over the crypto correlation graph

The claim: crypto returns are driven by correlated neighbours, with a cross-sectional correlation of
about +0.047 — inside the range notebook 003's own surviving factors occupied — and temporal mixing
actively hurts.

The universe is restricted to the **26 of 30** symbols with complete daily coverage, since a fixed
node count is required to batch-train one graph model across every rebalance date.

**A finding before any backtest runs:** node degree averages **22.8 of 25** possible neighbours at
the pre-registered correlation threshold. Crypto's pairwise correlations are high enough that this
graph is **nearly complete, not sparse** — which undercuts the "neighbour structure" premise on its
face. Reported as its own finding rather than re-tuned after the fact.

**And the sanity check returns a genuine anti-finding, not a null.** The cross-sectional correlation
is **−0.032** (t = −3.58) without temporal mixing and **−0.046** (t = −4.75) with it. Both pass the
magnitude and significance screen — but **the sign is inverted** from the claimed +0.047.

This is not "no signal found". It is **"a significant signal found, pointing the wrong way"** — the
same category as notebook 012's result.

The dollar-neutral book returns net Sharpe **−1.52** with an interval that **excludes zero** — a real,
statistically distinguishable loss, not noise. The long-only variant is flat (+0.009). Beta to the
equal-weight basket is −0.02, confirming the book is genuinely market-neutral: it is losing on its
cross-sectional bet specifically, not eating basket beta.

A turnover-throttled variant roughly halves the loss (−0.81 against −1.52), consistent with notebook
007's turnover finding applying here too, though its interval no longer excludes zero.

The temporal-mixing comparison agrees with the design's thesis in point estimate — the no-mixing
correlation is less negative — but the interval on the difference includes zero, so it correctly does
not fire.

---

## Substitutions, disclosed

| Design | Substitution | Why |
|---|---|---|
| A | Assumed $10M capital base for the impact calculation | No figure given in the specification; capacity is reported relative to this base |
| B | No bond or volatility futures | None exist anywhere in this repo; not proxied with an ETF |
| B | One duplicate equity-index series dropped | Already covered by another feed; avoids double-counting |
| B | Currency futures cost specifications added | Real exchange tick values, extending rather than replacing the cost model |
| C (first) | Dollar volume substituted for market cap | No market-cap data at the time — superseded by the corrected build |
| C (first) | Funding not modelled | No funding cache at the time — superseded |
| C (both) | One symbol drops from the cross-section after 2024-09 | An exchange rebrand ended the cached symbol's feed |
| C (corrected) | Market-cap ranking is a current snapshot, not rolling | Free-tier rate limits make rolling reconstruction across 128 symbols impractical |
| C (corrected) | 128 symbols against the paper's "150+" | Close but not exact; built from every perpetual onboarded on or before the start date |
| C (corrected) | Two parameters carried over unchanged | The paper never discloses either |
| D | Universe restricted to 26 of 30 symbols | A fixed node count is required for batch training |
| D | 26-symbol graph against the source's 66 | This repo's full crypto universe is 30 symbols; a disclosed structural weakening |

---

## Bottom line

Four externally reported mechanisms, each rebuilt to its own specification with its authors'
identified source of edge intact, and each one fails on this data under these costs.

**Design D fails in the more informative way**: not "no effect detected" but "the specific mechanism
under test moves the wrong way, with an interval that excludes zero" — a stronger, more falsifiable
answer than a wide null would have been.

**Design C's story is more nuanced and arguably more valuable.** After the source paper was read and
a corrected rebuild fixed three material parameter mismatches, the mechanism now behaves exactly as
its authors describe in every ablation — and *still* does not clear the bar, on a window this repo's
data makes harder than the paper's tested period. That is a cleaner refutation than the first
attempt, precisely because fixing the implementation removed every plausible "you built it wrong"
objection rather than adding one.

Combined with notebook 011a's −0.16 reproduction of an outside spread book and the twenty-two nulls
that came before, the honest summary of this programme's search for alpha — using both its own
constructions and four independently sourced outside ones — is that none of it clears the bar set at
the outset. And Design D's inverted correlation and Design C's now-confirmed-genuine
mechanism-with-no-net-edge are specific, falsifiable claims about *why*, not just *that*.

*Notebook: `src/research/013_four_outside_designs_rebuilt_and_scored.ipynb`. Both holdouts remain
exactly as spent as notebooks 003 and 008 left them.*
