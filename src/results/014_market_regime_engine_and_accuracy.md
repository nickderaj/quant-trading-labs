# 014 — Porting the Daily Market-Regime Engine, and Scoring Its Accuracy

## The question

A regime engine already runs in production, shipping labels every morning across 11 market sectors
and six dimensions — trend, volatility, risk, credit, yield curve, term structure and carry. Its
accuracy has never been independently measured.

This notebook ports it module-for-module into this repo and then scores it against two independent
ground-truth sources.

**Unlike every prior notebook here, a positive result was the hoped-for outcome.** The engine already
ships. The goal was to establish whether a future notebook could safely condition a trading strategy
on these labels.

Six criteria, fixed before the historical panel was ever built or plotted:

| Check | Question | Result |
|---|---|:---:|
| **No lookahead** | Does the pipeline ever use future data? | **Passes** |
| **Port fidelity** | Does the ported engine match the source? | **Passes** |
| **Structural soundness** | Are any labels degenerate — stuck, or flipping like noise? | **Passes** |
| **Crisis accuracy** | Do labels beat a naive baseline on known historical episodes? | **Fails** |
| **Mechanical accuracy** | Do labels beat a naive baseline against forward-realised outcomes? | **Fails** |
| **Reaction speed** | Is the median lag identifying a crisis onset within 21 trading days? | **Fails — 27 days** |

**The engine is structurally sound and its labels do not demonstrably beat naive baselines.** Half
the checks failing is a real, actionable finding about production, not a design flaw in the study.

This notebook authorises no trading and spends no holdout.

---

## The port, and what it cost to trust it

The regime engine is ported module for module — configuration, scoring, registry, engine,
aggregation, alignment, transitions, evaluation, prediction, and all six dimension modules, plus the
reporting layer — rewired onto this repo's own data through a single new adapter that is the sole
conversion boundary between the two data libraries. 41 ported tests pass; linting and type checking
are clean.

Every claimed data source exists: 27 daily bar symbols (the equity index going back to 2000, 6,531
rows), 8 macroeconomic series, one positioning file, and five futures curves.

**Fidelity was established two ways, both stronger than the anticipated fallback.** The source
environment turned out to be reachable, so configuration hashes were confirmed identical on all four
configurations **and** a genuine end-to-end run — same synthetic fixture, source engine against ported
engine, in two separate Python environments — produced **byte-identical scores and labels**. That is
the strongest port evidence this programme has produced.

**The lookahead check is the single most valuable thing the port brings across.** It was run as a
hard gate at four different truncation depths against every one of the 27 real symbols — not just
synthetic fixtures — each wired with its actual curve, macro and positioning inputs exactly as the
builder uses them. All 27 passed, plus an optional crude-positioning path. A failure anywhere would
have halted the notebook.

### Two structural limitations that carry through everything below

- **The curve slope is computed across three months, not production's twelve**, because this repo's
  curve files have only three legs. The term-structure and carry dimensions here therefore do not
  transfer to production one-for-one.
- **The broad commodities basket's term structure and carry are a 5-symbol aggregate, not 20.** Only
  five legs have a curve file here; the rest drop out via the coverage rule.

---

## Does it behave like a regime model at all?

43 sector-and-dimension pairs evaluated on transition matrices, regime durations, label stability and
expected remaining duration.

**Two are disqualified, both real findings:**

**Commodities trend** shows 90.24% single-label occupancy — essentially always "sideways". A
20-symbol pooled basket score rarely moves far enough from zero to leave the neutral band. No single
trend regime dominates a diversified commodity book for long. This was one of the failure modes named
in advance.

**Macro credit** has only 7.90% coverage of its sector's history. **This was not anticipated.** The
two credit-spread series this repo caches only start in **July 2023** — thirty-three years later than
the volatility index or the yield-curve series. The introspection done during the port recorded these
dates but did not flag them as a gap, and the gap only became visible when coverage was checked
against each sector's *full* history rather than the panel's already-filtered row count. An early
version of that check made exactly that mistake and silently reported 100% coverage everywhere before
being corrected.

The remaining 41 pairs all show full coverage within their available window, flip rates between 0.001
and 0.041 — well under the 0.10 threshold, so the smoothing is doing real work rather than producing
noise — and no other dimension stuck on a single label.

---

## Accuracy, source one: known historical crisis episodes

A frozen table of eight named historical episodes, expanded into concrete (sector, dimension,
expected label) targets — so the COVID crash's "all volatility" expands to every basket with a
volatility dimension. Scored by balanced accuracy against three baselines: persistence (yesterday's
label), a one-step Markov forecast, and the historical class prior.

| Sector / dimension | Observations | Balanced accuracy | Best baseline |
|---|---:|---:|---|
| Oil products / carry | 295 | 0.932 | Markov (tied) |
| Oil products / term structure | 295 | 0.919 | Persistence (tied) |
| Macro / yield curve | 400 | 0.825 | Persistence, +0.0025 (p = 0.53) |
| Precious / volatility | 42 | 0.429 | Markov (tied) |
| Oil products / volatility | 42 | 0.286 | Markov (tied) |
| FX / volatility | 42 | 0.238 | Markov (tied) |
| **Macro / risk** | 567 | **0.353** | Persistence (tied) |
| Natural gas / volatility | 42 | 0.167 | Markov (tied) |
| Base metals / volatility | 42 | 0.119 | Markov (tied) |
| Softs / volatility | 42 | 0.119 | Markov (tied) |
| Soy complex / volatility | 42 | 0.000 | Persistence (tied) |
| Grains / volatility | 42 | 0.000 | Persistence (tied) |
| Meats / volatility | 42 | 0.000 | Persistence (tied) |
| Commodities / volatility | 42 | 0.048 | Markov (tied) |

Note the macro risk dimension at 0.353 — **worse than chance for a three-state label.**

### A structural caveat, discovered here rather than assumed

**Thirteen of the fourteen "best baseline" rows are exact ties.** Within a single episode's short
window, the engine's own flip rate is very low, so persistence and one-step Markov are frequently
*literally identical* to the engine's own label that day. The comparison becomes a constant-zero
series by construction.

That makes engine-versus-persistence comparisons within short windows structurally weak evidence
either way — a finding about this test's power, not about the engine.

The class-prior baseline has the opposite problem: episodes are chosen *because* they are historically
unusual, so a dimension's modal label over its whole history is almost never the episode's expected
label. That guarantees near-zero scores regardless of engine quality, which is why it never appears
as "best baseline" — it is mechanically the worst one.

Neither comparison is very informative here. The raw accuracy numbers are more informative read
directly, and the second source below — much larger samples, no dependence on a single short window —
carries the real evidentiary weight.

### How fast does it react?

Signed trading days from episode onset to the first correct label:

- Financial crisis: **+15**
- COVID crash (risk): **+20**
- Hiking cycle (risk): **−21** (early)
- Euro crisis: **censored** — risk never flipped in-window
- Taper tantrum: **−21** (early)
- Hiking cycle (yield curve): **+110** — the inversion label took five and a half months to register
- Oil glut: **censored**
- Energy backwardation: **−18** (early) and **+20** on carry
- COVID volatility across baskets: **+24 to +40** — the minimum-dwell requirement and smoothing meant
  "extreme" mostly arrived one to two months into a crash that peaked in weeks

**Median across the 14 scored pairs: 27 trading days**, against a 21-day threshold. Missed mainly by
the yield-curve inversion's extreme lag and the volatility dimensions' uniformly slow reaction.

---

## Accuracy, source two: forward-realised mechanical labels

Targets with no human judgment, over each sector's full available history: volatility terciles, the
sign of forward return, the sign of the yield-curve series, and the sign of the front-month curve
spread. **The last two are contemporaneous consistency checks, not forecasts** — the underlying series
is literally what the indicator is built from.

| Check | Observations | Balanced accuracy |
|---|---:|---:|
| Yield curve against the yield-curve series' sign | 4,889 | 0.981 |
| Term structure against curve-spread sign (oil, gas) | 1,180 / 1,513 | 0.879 / 0.871 |
| Term structure against curve-spread sign (other baskets) | 307–1,501 | 0.428–0.679 |
| Volatility against forward 21-day volatility tercile (9 baskets) | 5,924–6,166 | 0.538–0.657 |
| Trend against forward 63-day return sign (8 baskets) | 1,405–2,786 | **0.393–0.599** |

**The two highest numbers are close to tautological.** The yield-curve dimension is built 50% from
the very series it is checked against, and the curve-slope indicator *is* a transform of the spread
it's compared to. High accuracy here mostly validates the scoring arithmetic, not the regime concept.

Volatility, at 0.54–0.66, is a genuinely independent and unremarkably moderate result.

**Trend against the sign of the forward 63-day return is the genuinely independent test with the
widest coverage, and it is weak-to-null everywhere.** Two baskets score *below* a coin flip (0.393 and
0.394), two more sit near it, and only one clears 0.60.

That is consistent with this programme's 22-plus prior null findings on trend-following edge —
extended here from "can you trade it" to **"does the label even describe the forward direction".**

**None of the 39 trials, from either source, clears the multiple-testing-corrected significance
threshold against its best baseline.**

---

## Verdict, per dimension

| Dimension | Safe to condition a strategy on? | Why |
|---|---|---|
| Term structure, carry (oil, gas) | **No, without further work** | Highest raw accuracy (85–93%), but the confirming check is close to tautological |
| Yield curve | **No, without further work** | Same tautology problem — built from and checked against the same series |
| Volatility (all baskets) | **No** | Moderate mechanical accuracy, but failed to reach "extreme" during COVID in most baskets, and reacted 24–40 days late where it did |
| Trend (all baskets) | **No** | Weak-to-null against forward returns; the commodities version is additionally disqualified as degenerate |
| Risk (macro) | **No** | 35.3% accuracy against four named crises — worse than chance for a three-state classifier; missed two of four entirely |
| Credit (macro) | **Not scoreable** | A real data gap starting in 2023, not a code defect |

**No dimension is validated strongly enough, independent of near-tautological checks, for a follow-up
notebook to condition on without further work.**

The two highest-accuracy dimensions would need a genuinely independent confirming test — one that
doesn't share an input series with the dimension's own indicators — before that changes. And trend and
risk, the two dimensions most likely to matter for a regime-conditional strategy, look **actively
unreliable** over the episodes and horizons tested here.

Stated without hedging: **production has been shipping labels every morning that pass every
structural check and fail every accuracy check against two independent ground-truth sources.**

The concrete next step this notebook identifies is building an independent mechanical check that
doesn't share an input series with the dimension it validates — not "trade it".

---

## Substitutions and disclosed limitations

| Item | Substitution | Reason | Effect |
|---|---|---|---|
| Curve slope horizon | 3 months, not production's 12 | This repo's curves have only three legs | Term structure and carry aren't comparable to production one-for-one |
| Commodities basket curve dimensions | 5-symbol aggregate, not 20 | Only five legs have a curve file | A narrower aggregate than the basket's other dimensions |
| A dependency-declaration bug in the risk dimension | **Preserved bug-for-bug** | This notebook scores what production actually runs | The risk dimension is quietly reweighted onto five indicators whenever positioning data is absent — which is always, here — rather than skipping the indicator |
| Credit coverage | Scored on 2023 onward only | The underlying cache starts there | Disqualified as unscoreable |
| Indicator primitives | Ported fresh rather than delegated to repo-local equivalents | No repo-local equivalents exist with matching schema and semantics | None — confirmed byte-identical via the end-to-end fidelity check |
| Episode-table baselines | Whole-window predictions, not day-by-day evolving forecasts | Episode windows are only weeks to months | Weak discriminating power, disclosed rather than hidden |
| Representative sector price | First symbol in each basket | No cross-sectional composite index exists for these baskets | A genuine simplification; a weighted composite might score differently |

## Scope

One notebook, one results file, one pre-registration frozen before any panel was built and never
edited. Infrastructure lives in durable modules with their own tests, not scratch files.

No strategy was built, no Sharpe was computed, no holdout was touched. No weight, band or window was
tuned to make any check pass. No new configuration was added — the universe is exactly production's,
copied verbatim.

*Notebook: `src/research/014_market_regime_engine_and_accuracy.ipynb`. The historical panel (224,878
rows) is persisted so no future notebook needs to recompute it.*
