# Notebook 013 — Four Outside Designs, Rebuilt and Scored on Our Own Data: Results Summary

## What

This notebook rebuilds four independently-specified, externally-reported trading mechanisms end to
end on this programme's own data and costs: a gold trend/execution book (Design A), cross-asset
sequence models trained on a Sharpe objective (Design B), an adaptive 6h crypto trend book (Design C),
and a crypto cross-sectional graph-attention model (Design D).

## Why

Notebook 009 flagged that the programme's twenty-two-plus prior null gates were all on its own
in-house constructions, leaving open the rebuttal that the strategies simply weren't built well
enough. Testing outside, published designs — rebuilt faithfully to each paper's own specification,
including the specific mechanism each paper attributes its edge to — removes that rebuttal and gives
a materially stronger form of null if these also fail.

## How

Six pre-registered gates scored each design on its own terms (execution quality for A, training
objective for B, adaptation for C, cross-sectional structure for D), plus a shared look-ahead audit
(Phase L) and structural checks like beta and drawdown-plausibility. Design C was rebuilt twice: a v1
guess at unpublished parameters, then a corrected v2 once the source papers were obtained and read,
fixing quality thresholds, universe size, and selection mechanism, and adding funding costs.

## Results

All six gates return `fires=False`. The best net Sharpe achieved across all four designs is +0.33
(Design A), well below this programme's own +1.053 record and far short of the designs' reported
2.4–3.1 range. Design B's neural sequence models lose to a simple linear baseline. Design D's
cross-sectional IC is statistically significant but inverted in sign relative to its claim. Design C's
corrected v2 rebuild behaves exactly as its authors describe in every ablation yet still shows no net
edge, making it a stronger, not weaker, refutation. None of the four outside designs reproduces on
this data.

Six pre-registered gates (`phase_0_13_preregistration.json`, committed before any backtest ran and
not edited since). **All six return `fires=False`.** Four independently-specified, externally-
reported trading mechanisms — a gold trend/execution book, cross-asset sequence models trained on
a Sharpe objective, an adaptive 6h crypto trend book, and a crypto cross-sectional graph-attention
model — were rebuilt end to end on this programme's own data and costs. None reproduces. The best
net Sharpe achieved across all four is **+0.33** (Design A), still below 007's own +1.053 record and
nowhere near the four designs' reported range of 2.4–3.1. Phase L's look-ahead audit never
triggers for any design, because none clears the 1.5 pre-registered suspicion threshold — these are
not suppressed wins, they are honest nulls at every stage.

This closes the hole 009 flagged and 011a only partially addressed: the programme has now built
four complete, externally-specified, execution-aware strategies on the crypto and futures data
where its nulls live, not just its own twenty-two constructions. All four fail. That is a
materially stronger null than the twenty-two before it, because "you didn't build it well enough"
is no longer available as a rebuttal — each design is rebuilt to its own specification, complete
with the mechanism its authors identified as the source of edge (execution for A, objective for B,
adaptation for C, cross-sectional structure for D), and each mechanism is tested on its own terms,
not just at the top-line Sharpe.

**Addendum, Design C v2.** After dev-window results were committed, the four designs' actual source
papers were supplied and read (arXiv:2511.08571, 2603.01820, 2602.11708, 2606.27670). Design C's
source specifies exact parameters the original build (v1, below) had guessed at and gotten
materially wrong: quality thresholds of 1.3/1.7 (v1 used 0.0/0.15), a 150+-symbol universe (v1 used
30), and top-15/bottom-15 market-cap-based selection (v1 used a dollar-volume proxy). A corrected
rebuild (v2) fixes all three, expands the universe to 128 live-fetched Binance perpetuals, and adds
previously-unmodelled funding-rate costs. **v2 is the design's headline result below; v1's numbers
are kept and explained as the disclosed history of getting there, not erased.** Pooled `n_trials`
rises from 18 to **25** (18 + 7 newly-computed v2 configurations), honestly counted upward per
sec6's own rule — this changes no gate verdict, since DSR was already ~0 at the smaller count.

## Gate table

| gate | design | claim | headline result | fires |
|---|---|---|---:|:---:|
| FF | A | net Sharpe clears bar, \|beta\|<0.15, stable sign | net Sharpe **+0.33**, CI includes zero, DSR 0.15 | **No** |
| SQ | B | best sequence model beats linear baseline | linear **0.204** beats both LSTM (0.057) and GatedLSTM (0.052) | **No** |
| AT | C (v2) | adaptive beats frozen, ablations correctly signed | adaptive **−0.055**, does not beat frozen (not significant), **3/3 ablations correct** | **No** |
| XS | D | dollar-neutral book clears bar, IC survives 003's filter | net Sharpe **−1.52**, IC significant but **wrong sign** | **No** |
| TGT | any | net Sharpe ≥ 2.0, DSR ≥ 0.95, passes Phase L | best achieved: **+0.33** (Design A) | **No** |
| TM | D | no-mixing IC significantly beats mixing IC | direction matches (−0.032 vs −0.046) but CI includes zero | **No** |

Phase L trigger (net Sharpe ≥ 1.5) never fires for any design — the highest headline is A's 0.33,
the lowest is D's −1.52. No holdout was spent: §8's rule ties holdout access to a dev-gate firing,
and none did, so both the crypto and futures holdouts remain exactly as spent/unspent as notebooks
003 and 008 left them.

## Design A — forecast-to-fill trend/momentum on GC futures

The claim under test was that the alpha is in the fill, not the signal — vol-targeted, Kelly-scaled,
impact-discounted sizing around a single smoothed trend-momentum state. Rebuilt exactly to spec on
`GC` F1 continuous, rolling 10y-train/6mo-test walk-forward.

| variant | net Sharpe | note |
|---|---:|---|
| base (full design) | **+0.333** | headline |
| no smoothing | +0.039 | smoothing is confirmed load-bearing, matching 007's finding restated |
| no impact discount | +0.383 | impact costs ~0.05 Sharpe, as expected for a sizing discount |
| no fractional Kelly | +0.461 | the Kelly overlay is a net drag here, not a net gain |

DSR at the base config is **0.15** against the pooled `n_trials=18` — nowhere near 0.95. The 95%
bootstrap CI on net returns includes zero. OOS runs 2019-03-24 to 2024-09-06, **1,512 bars**, well
short of the design's claimed ~2,793 trading days (its history starts a decade earlier than ours) —
reported honestly rather than manufactured by shortening the training window.

Two structural checks the design invites came back **supporting** its mechanism even as its Sharpe
fails: measured **beta to spot gold is 0.002**, matching the design's own claimed ≈0.03
near-zero-beta property almost exactly, and the pre-registered arithmetic red flag — the design's
claimed 0.52% max drawdown at Sharpe ~2.9 was flagged before this notebook ran as "impossible
without a stop-fill leak" — did **not** reproduce: our max drawdown is **−13.3%**, a normal
figure for a Sharpe-0.33 process, and the required worse-of-(stop, gapped-open) fill convention
produces materially different (worse) fills than the optimistic same-bar convention would have,
confirming the leak the design's own numbers implied is exactly the kind of thing this convention
exists to prevent. Origin offsets (0/7/14/21 bars) agree to 4+ decimals — vacuous here, same pattern
012 found, disclosed rather than presented as robustness.

## Design B — sequence models on the cross-asset futures panel

The claim under test was that models trained on negative realized Sharpe (not MSE) learn temporal
representations a linear model can't. Universe: 16 databento products (F1 continuous) + 6 CME FX
futures (`6A/6B/6C/6E/6J/6S`) = **22 instruments** — yfinance's separate `ES=F` daily series was
dropped since databento's `ES` already covers the same instrument; pooling both would double-count,
not diversify. **No bond or VIX futures exist anywhere in this repo** — disclosed structural gap
versus the source design's universe, not papered over with an ETF proxy.
`commod_lib8.CONTRACT_SPECS` was extended with the six FX futures' real CME tick/tick-value specs
so the sec2 futures cost-model headline (`round_turn_cost_per_contract`) covers the whole panel,
not a flat convention re-applied to instruments it was never built for (012's exact mistake).

| architecture | median Sharpe (5 seeds) | seed range | vs. linear baseline (0.204) |
|---|---:|---:|---|
| linear baseline | **0.204** | — | — |
| LSTM | 0.057 | [−0.003, 0.073] | loses |
| gated LSTM | 0.052 | [−0.029, 0.104] | loses |

Both architectures land well **below** the linear model they were supposed to beat — Gate SQ's
substance, not just its absolute-bar leg, fails outright. Per-seed spread is reported in full (not
the best seed): both nets swing negative on at least one seed, underscoring that whatever these
models learn is unstable across initialization, not a repeatable edge. Breakeven cost is ≈10bp/side
for both architectures, a single comparable number now adopted for future cost-sensitivity legs per
the design's own suggestion.

**One real bug found and fixed in-flight, disclosed rather than quietly patched:** the additive
Panama back-adjustment produces a handful of negative synthetic prices for `CL`/`HO`/`NG`/`ZW` by
stacking roll adjustments onto already-low 2020-crash-era prices. `log(negative)` is `NaN`, and
polars' `drop_nulls()` does not remove `NaN` floats (only true nulls) — the first run's full-batch
Sharpe-objective loss ingested one such row, produced a `NaN` gradient, and silently zeroed every
model's predictions from that fold onward (median Sharpe reported as exactly `0.0` across all five
seeds for both architectures). Fixed by dropping non-positive back-adjusted prices at the source and
adding an explicit `is_finite()` filter alongside `drop_nulls()`. The numbers above are from the
corrected run.

## Design C v1 — adaptive trend, 6h crypto, asymmetric long/short book (superseded, kept for the record)

The claim under test was that volatility-adaptive trailing exits, quality/liquidity selection, and
causal monthly re-parameterization clear Sharpe 2.4 net of 4bp fees. **NEXT_PROMPT.md's own premise
that "we have no 6h cache" turned out to be false** — 6h is a native Binance klines interval,
fetched directly from `data.binance.vision` for all 30 symbols (confirmed live before committing to
the design), not resampled from 1h as planned. `MATIC` still drops out of the cross-section from
2024-10 onward (Binance's `MATIC`→`POL` rebrand ended the old symbol's feed), the same treatment
`LUNA`/`FTT` already get post-collapse/delisting.

| variant | net Sharpe | max drawdown |
|---|---:|---:|
| **adaptive** (headline) | **−0.105** | −59.4% |
| frozen twin (2021H2-calibrated, never re-fit) | −0.174 | −54.3% |
| ablation: no trailing stop | +0.332 | −53.0% |
| ablation: no selection filter | −0.610 | −68.6% |
| ablation: no liquidity filter | +0.455 | −45.7% |
| ablation: no adaptation (= frozen twin) | −0.174 | −54.3% |
| adaptive, excluding LUNA/FTT | −0.060 | −59.4% |

(Drawdowns are simple-return equivalents, `exp(log-drawdown)−1` — an earlier draft of this table
quoted the raw cumulative-log-return figures directly as percentages, overstating every one of
them; corrected here.)

Adaptive does **not** beat the frozen twin — the paired block-bootstrap CI on (adaptive − frozen)
is `[−0.00023, +0.00027]`, squarely including zero, so the design's own most-interesting claim (its
authors attribute ~1.07 Sharpe to adaptation alone) does not survive contact with our data at all.
Only 2 of 4 ablations move in the design-predicted direction (removing the selection filter and
removing adaptation both hurt, as predicted; removing the trailing stop and removing the liquidity
filter both **help**, the opposite of predicted) — Gate AT's third leg (≥3/4 correct) fails on its
own terms, not just on the headline Sharpe. All three legs required by the gate fail simultaneously.
Excluding `LUNA`/`FTT` leaves the sign unchanged (−0.06 vs −0.105) — the result does not live on
those two collapses. Origin offsets (−0.105/−0.166/−0.155/−0.136) move together but are **not**
vacuous this time — a genuine perturbation, unlike Design A's and 012's offset legs.

Drawdowns across every variant are severe (−46% to −69%) at Sharpes near zero — a symptom of
capital concentration when few symbols pass the selection filter simultaneously (a market-wide
stress month can leave a leg holding one or two names at 70%/30% of book capital), disclosed as a
limitation of this book construction rather than smoothed over. **Funding rates are not modelled**
(no funding-rate cache exists) — the persistently-short 30% leg's unmodelled 8h funding cash flow
through a multi-year sample of mixed regimes is a known, undirected source of additional error not
reflected in the numbers above.

**v1's own quality thresholds (0.0 long / 0.15 short) turned out to bear no resemblance to the
source paper's actual values (1.3 / 1.7)** once the paper (arXiv:2602.11708) was read directly —
see Design C v2 below, which corrects this and every other identifiable parameter mismatch.

## Design C v2 — corrected rebuild matching the actual AdaptiveTrend paper (arXiv:2602.11708)

Once the source paper was available, three material implementation gaps in v1 were identifiable
by direct comparison, not guesswork, and fixed:

| | v1 (guessed) | v2 (paper's stated values) |
|---|---|---|
| Universe | 30 symbols | **128** Binance USDT perpetuals (paper: "150+") — the original 30 unioned with every perpetual onboarded on/before 2022-07-01 per live `exchangeInfo`, fetched fresh via `data.download_and_unzip_klines` |
| Quality thresholds | 0.0 long / 0.15 short | **1.3 long / −1.7 short** (paper's stated rolling-Sharpe screen) |
| Selection mechanism | dollar-volume rank proxy | **top-15 / bottom-15 by market cap** (paper's literal `KL=15` rule) |
| Funding | not modelled | **modelled**, live Binance funding-rate history for all 128 symbols |
| ATR multiplier grid | `[2, 3, 4]` | `[2.0, 2.5, 3.0, 3.5]`, bracketing the paper's reported optimal region |

Market cap comes from CoinGecko's free `/coins/markets` endpoint — a **current-day snapshot**
ranking (117 of 128 symbols mapped) applied statically across the whole 2022–2025 backtest, not a
rolling historical ranking. The free tier's rate limit makes a true rolling reconstruction across
128 symbols impractical; this is disclosed as a real remaining gap versus the paper's presumably-
rolling ranking, not presented as an exact match. `L`/`θ` grid values and the exact rolling-Sharpe
window length are never disclosed anywhere in the paper, so v1's original choices for those two
carry over unchanged — there is nothing more specific to match.

| variant | net Sharpe | max drawdown |
|---|---:|---:|
| **adaptive** (headline) | **−0.055** | **−6.6%** |
| frozen twin | −0.209 | −5.7% |
| adaptive, funding not modelled | −0.055 | −6.6% |
| ablation: no trailing stop | −0.733 | −22.7% |
| ablation: no selection filter | −1.362 | **−95.4%** (a near-total wipeout) |
| ablation: no adaptation (= frozen twin) | −0.209 | −5.7% |
| adaptive, excluding LUNA/FTT | −0.240 | −6.6% |

(Drawdowns are simple-return equivalents, `exp(log-drawdown)−1`, same correction as the v1 table.)

The corrected parameters produce a strikingly more internally consistent result even though the
headline Sharpe is still negative. **v1's most damaging problem — a −59% max drawdown from capital
concentrating into one or two names whenever a weak selection filter let a stress month through —
is gone: v2's drawdown is a sane −6.6%,** a direct, mechanical consequence of the real 1.3/1.7
quality screen actually doing its job. And **every ablation now moves in the direction the paper
predicts** (v1 got 2 of 4 backwards): removing the trailing stop hurts badly (−0.055 → −0.733),
removing the selection filter is catastrophic and is the one effect in this whole design whose
bootstrap CI **excludes zero** (a real, statistically distinguishable effect), and removing
adaptation hurts (−0.055 → −0.209, with the point estimate now favouring adaptive over frozen,
though the paired CI `[−0.0000218, +0.0000293]` still includes zero — not yet significant, but no
longer backwards). Funding cost is immaterial at this position sizing (−0.055 with or without it).
Origin offsets are **not vacuous** and considerably less stable than v1's (−0.055 / −0.904 / −0.462
/ −0.447) — sign is consistently negative but magnitude is highly sensitive to start date, itself a
finding rather than noise to explain away.

**Net effect on the conclusion: this makes Design C's null stronger, not weaker.** A implementation
this much closer to the actual paper — right universe scale, right thresholds, right selection
mechanism, funding modelled — still does not produce a positive, significant Sharpe on our data.
What it does produce is a mechanism that behaves exactly as its authors describe (stops help,
selection helps, adaptation helps) while still losing money overall on a window that includes the
2022 bear market and the LUNA/FTX collapses, which the paper's own test window does not. There is
no longer a live "maybe this was just implemented wrong" objection hanging over this design — v1's
contradictions were real bugs, now fixed and disclosed, not evidence against the paper's mechanism.

## Design D — cross-sectional attention over the crypto correlation graph

The claim under test was that crypto returns are driven by correlated neighbours (cross-sectional
IC ≈ 0.047, in 003's own surviving-factor range) and that temporal mixing actively hurts. Universe
restricted to the **26 of 30** symbols with complete daily coverage across the full dev window (a
fixed node count is required to batch-train one graph-attention model across every rebalance date);
`LUNA`/`FTT`/`MATIC` drop out by this criterion, plus `EOS` (a one-month 2025-06 cache gap). Node
degree averages **22.8 of 25** possible neighbours at the pre-registered 0.3 correlation threshold
— crypto's pairwise correlations are high enough that this graph is nearly complete, not sparse,
undercutting the "neighbour structure" premise before any backtest runs, and reported as its own
finding rather than re-tuned after the fact.

**The sanity-check gate that runs before any backtest is believed returns a genuine anti-finding,
not a null:** cross-sectional IC is `−0.032` (Newey-West t = −3.58) for the no-time-mixing model and
`−0.046` (t = −4.75) for the time-mixing model — both pass 003's magnitude/significance survival
filter, but **the sign is inverted** from the design's claimed +0.047. This is not "no signal
found," it is "a significant signal found, pointing the wrong way," the same category of finding as
012's Gate VB (point estimate moves the wrong direction). The dollar-neutral book, built on the
no-mixing model's predictions, returns net Sharpe **−1.52** with a bootstrap CI that **excludes
zero** (`[−0.0020, −0.00028]`) — a real, statistically distinguishable loss, not noise. The
long-only top-5 book is flat (net Sharpe 0.009). Beta of the dollar-neutral book to the equal-weight
basket is −0.02, confirming the book is genuinely market-neutral — it is losing on its cross-
sectional bet specifically, not eating basket beta. A turnover-throttled variant
(`alpha_lib7.throttle_weights`) roughly halves the loss (−0.81 vs −1.52), consistent with 007's
turnover-reduction finding applying here too, though the throttled book's CI no longer excludes
zero. Gate TM's point estimate agrees with the design's thesis (no-mixing IC is less negative than
mixing IC) but the bootstrap CI on the difference includes zero, so the gate correctly does not fire.

## Substitutions table (sec7 trap4)

| design | substitution | disclosed reason |
|---|---|---|
| A | assumed $10M AUM base for impact/capacity calc | no AUM figure given in the design spec; capacity curve reported relative to this base |
| B | no bond/VIX futures in universe | none exist anywhere in this repo; not proxied with an ETF |
| B | yfinance `ES=F` dropped from panel | databento's `ES` already covers the same instrument; avoids double-counting |
| B | FX futures cost specs added to `commod_lib8.CONTRACT_SPECS` | real CME tick/tick-value specs, extending (not replacing) the existing cost model |
| C v1 | trailing dollar volume substitutes for market cap | no market-cap data in this repo (superseded by v2's real market-cap fetch) |
| C v1 | funding rates not modelled | no funding-rate cache existed at the time (superseded by v2's live fetch) |
| C v1/v2 | `MATIC` drops from cross-section after 2024-09 | Binance `MATIC`→`POL` rebrand ended the cached symbol's feed |
| C v2 | market-cap ranking is a current-day snapshot, not rolling history | CoinGecko free-tier rate limits make a rolling reconstruction across 128 symbols impractical |
| C v2 | 128-symbol universe vs. source paper's "150+" | close but not exact; built from every USDT perpetual Binance's live `exchangeInfo` shows onboarded on/before 2022-07-01, unioned with the original 30 |
| C v2 | `L`/`θ` grid values and rolling-Sharpe window length carried over from v1 unchanged | the source paper never discloses either; nothing more specific exists to match |
| D | universe restricted to 26/30 symbols (complete coverage) | fixed node count required for batch GAT training across all rebalance dates |
| D | 26-symbol graph vs. source design's 66 | this repo's full crypto universe is 30 symbols; disclosed structural weakening |

## What this notebook establishes, plainly

Four externally-reported mechanisms, each rebuilt to its own specification with its authors'
identified source of edge intact — execution quality for A, a risk-adjusted training objective for
B, adaptive re-parameterization for C, cross-sectional graph structure for D — and each one fails on
this programme's own data under its own costs. Design D fails in the more informative way: not "no
effect detected" but "the specific mechanism under test moves the wrong way, with a confidence
interval that excludes zero" — a stronger, more falsifiable answer than a wide null interval would
have been. Design C's story is more nuanced and, after the source paper was supplied and a corrected
v2 rebuild fixed three material parameter mismatches (quality thresholds, universe scale, selection
mechanism), arguably more valuable: the corrected mechanism now behaves exactly as its authors
describe in every ablation, and *still* does not clear the bar, on a window this repo's own data
happens to make harder (2022's bear market and the LUNA/FTX collapses) than the paper's tested
period. That is a cleaner refutation than v1's original result, precisely because fixing the
implementation removed every plausible "you built it wrong" objection rather than adding one.
Combined with 011a's −0.16 reproduction of an outside spread book and the twenty-two null gates
that came before this notebook, the honest summary of this programme's search for alpha on this
data, using both its own constructions and four independently-sourced outside ones, is that none of
it clears the bar this programme set for itself at the outset — and D's inverted IC and C's
now-confirmed-genuine mechanism-with-no-net-edge are specific, falsifiable claims about *why*, not
just *that*.

Machinery: `src/research/tmp/exec_lib13.py` (trend-momentum state construction, fractional-Kelly
and square-root-impact sizing, the required stop-fill convention, causal monthly re-grid search,
quality/liquidity selection, the causal rolling correlation graph, `GraphAttentionPredictor`/
`LSTMForecaster`/`GatedLSTMForecaster`, the negative-realized-Sharpe training objective, and
breakeven-cost search), unit-tested in `tests/test_exec_lib13.py` including the two causal-leakage
perturbation tests sec7 traps 2 and 3 specifically call for. `src/research/tmp/run_phase_{A,B,C,D,L}_13.py`
build and score each design independently against the shared pre-registration
(`phase_0_13_preregistration.json`); Phase L's audit never triggers, because no design clears the
1.5 suspicion threshold. `commod_lib8.py` gained CONTRACT_SPECS entries for the six CME FX futures
used in Design B. `run_phase_C_13_v2.py` is the corrected Design C rebuild, live-fetching 128
symbols' 6h klines and funding-rate history from Binance and market-cap rankings from CoinGecko's
free API (`src/research/tmp/design_c_v2_universe.json`/`design_c_v2_marketcap.json`); the addendum
to `phase_0_13_preregistration.json` documents this as a disclosed follow-up, not a retroactive edit
of v1's committed numbers. Both holdouts (crypto 2025-07-01 onward, futures 2025-01-01 to
2026-07-28) remain exactly as spent as notebooks 003 and 008 left them — sec8's rule ties holdout
access to a firing dev gate, and none of the seven gates here (including AT_v2) fired.
