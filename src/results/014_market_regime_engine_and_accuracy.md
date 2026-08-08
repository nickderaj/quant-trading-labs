# Notebook 014 — Port the Daily Market-Regime Engine, and Score Its Accuracy: Results Summary

## What

This notebook ports the production daily market-regime engine (the one behind the live 7am cron
report) module-for-module into this repo, then scores the historical accuracy of its regime labels
across 11 sectors and six dimensions (trend, volatility, risk, credit, yield_curve, term_structure,
carry) against two independent ground-truth sources.

## Why

The engine already ships labels every morning in production, but its accuracy had never been
independently measured against this repo's own data and testing discipline. Unlike every prior
notebook, a gate *firing* here was the hoped-for outcome, since the goal was to determine whether a
future notebook could safely condition a trading strategy on these labels, or whether the labels —
despite being structurally sound — fail to demonstrably beat naive baselines.

## How

The port was verified bit-identical to the live source engine on synthetic fixtures and via matching
config hashes, then run over the full historical panel for every symbol. A hard lookahead check ran
across all 27 symbols at four truncations. Accuracy was scored two ways: balanced accuracy against a
frozen 8-episode table of known historical crisis periods (with persistence/Markov/class-prior
baselines) and against forward-realized mechanical labels over each sector's full history, both at a
Bonferroni-corrected significance threshold across 39 trials, plus a median label-lag check at named
crisis onsets.

## Results

Three of six gates fire: the port is faithful (bit-identical replication, NL/RC) and structurally
sound (no degenerate labels, RS). But the three accuracy-focused gates do not fire: no dimension beats
its naive baseline at Bonferroni-corrected significance (RA, RM both null across all 39 trials), and
median crisis-detection lag is 27 trading days, slower than the pre-registered 21-day bar (RL fails).
The two highest raw-accuracy dimensions (yield_curve, term_structure/carry) turn out to be close to
tautological, built from and checked against overlapping input series. The verdict: no dimension in
this port is validated strongly enough for a future notebook to condition a strategy on without
further, genuinely independent work.

Six pre-registered gates (`phase_0_14_preregistration.json`, committed before Phase 1 ever built or
plotted the historical panel, and not edited since). **NL, RS, and RC fire. RA, RM, and RL do not.**
The production regime engine (`../ultron/apps/trading-labs`'s 7am cron report, ported here verbatim
into `src/regime/`) is structurally sound — no lookahead anywhere in the pipeline, a faithful port
bit-identical to the source on synthetic fixtures, no dimension stuck in a degenerate single-label
or noise-flip state — but its labels do not demonstrably beat naive baselines at a
Bonferroni-corrected significance level against either of two independent ground-truth sources, and
its median lag identifying a named crisis episode's onset (27 trading days) is slower than the
21-day pre-registered threshold. Unlike every prior notebook in this programme, a gate *firing* here
was the hoped-for outcome (the engine already ships every morning); half the gates not firing is a
real, actionable finding about production, not a design flaw in this study.

This notebook authorizes no trading and spends no holdout. Both remain untouched (crypto
2025-07-01, futures 2025-01-01 → 2026-07-28); every Phase 3 comparison below is truncated to dates
before 2025-01-01.

## Gate table

| gate | claim | headline result | fires |
|---|---|---|:---:|
| NL | `no_lookahead_check` passes at every truncation, every sector | all 27 symbols pass at truncations (1,5,21,63); opt-in oil_products COT path also passes | **Yes** |
| RA | episode-table balanced accuracy beats class-prior, Bonferroni-corrected | 0/14 sector×dimension pairs significant at α=0.05/39 | No |
| RM | mechanical-label balanced accuracy beats persistence/Markov, Bonferroni-corrected | 0/25 pairs significant at α=0.05/39 | No |
| RL | median label lag at episode onset ≤ 21 trading days | **27** trading days (n=14 scored, 4 censored/never-matched) | No |
| RS | no scored dimension >90% single-label occupancy or >1 flip/10 bars | 2/43 pairs disqualified in Phase 2 (excluded, not scored); 0 violations among the 41 scored | **Yes** |
| RC | port fidelity: ported engine matches source where checkable | config hashes identical (4/4 configs) *and* a true end-to-end run against the live source engine (`uv run --project ../ultron/libs/finance`) is bit-identical on `macro_default`/`commodity_default` synthetic fixtures | **Yes** |

## Phase 0 — Port, and what it cost to trust it

`src/regime/` is a module-for-module port of `ultron_finance.regime` (engine: config, scoring,
registry, engine, aggregate, align, transitions, evaluation, forecast_eval, prediction, six
dimension modules) and `common.regime_report` (report layer: universe, builder, charts), rewired
onto this repo's own parquet via a single new adapter, `regime/loaders.py`, that is the sole
polars-to-pandas conversion boundary. 41 ported/adapted tests pass; ruff and mypy are clean.

Every data source NEXT_PROMPT.md §3 claimed exists does: 27 daily bar symbols
(`ES=F` from 2000-09-18, 6531 rows), 8 FRED series, the one CFTC COT file this repo has
(067651 = crude oil light sweet, 2006-01-03 → 2026-07-07), and five futures curves
(`cl/gc/hg/ng/si`, 2018-01-02 →). Two known gaps were confirmed as disclosed in advance
(§3.3/§3.4) and one more was found in Phase 2, not Phase 0 (below).

Port fidelity was established two ways, both stronger than the fallback NEXT_PROMPT.md anticipated
("if the source cannot be executed here"): the source venv *was* reachable (`../ultron/libs/finance`
has its own `uv` lock), so `config_hash()` equality was confirmed on all four YAMLs *and* a genuine
end-to-end run — same synthetic fixture, source engine vs. ported engine, in two separate Python
environments — produced byte-identical scores and labels for `macro_default` and
`commodity_default`. This is the strongest evidence this programme has produced for any port to
date.

`no_lookahead_check` — the single most valuable thing this port brings across — was run as a hard
gate at truncations `(1, 5, 21, 63)` against every one of the 27 real symbols in the universe (not
just synthetic fixtures), each wired with its actual curve/macro/COT inputs exactly as the builder
uses them. All 27 passed, plus the opt-in `oil_products` crude-COT wiring path (CL=F). A failure
anywhere would have halted the notebook; none did.

## Phase 1 — The historical panel

`build_regime_report` was run over every symbol's full available history (not a single `as_of`),
producing 11 sectors (Macro + 9 baskets — `Commodities`, `FX`, `oil products`, `natgas`,
`soy complex`, `grains`, `softs`, `precious`, `base metals`, `meats`) with zero build errors.
Persisted long-format to `src/research/data/market/research/regime_panel.parquet`
(224,878 rows) so no future notebook needs to recompute it. `history_start` was **not** overridden
from the production default (2019-01-01) in the universe config — it turned out not to matter,
because `regime/loaders.py` reads each parquet's full history unconditionally regardless of that
field; the panel already spans back to each symbol's real start (2000-era for most, 1997 for
`GC=F`/`SI=F`/`PL=F`/`PA=F`), so 2008 and 2011 were in scope for Phase 3 without any code change.

Two disclosed, structural limitations carry through every later phase, exactly as flagged in
advance:

- **`curve_slope` is called with `far="close_f3"`**, not production's default `far="close_f12"` —
  this repo's curves have only three legs. `term_structure`/`carry` here are therefore scored on a
  **3-month slope**, not production's 12-month one; their numbers do not transfer to production
  one-for-one.
- **The Commodities basket's `term_structure`/`carry` are a 5-symbol aggregate, not 20** — only
  `CL/NG/GC/SI/HG=F` have a curve file in this repo; the other 15 legs are permanently `curve=None`
  and drop out via the coverage rule.

## Phase 2 — Does it behave like a regime model at all?

43 sector×dimension pairs evaluated (`transition_matrix`, `regime_durations`, `label_stability`,
`expected_remaining_duration`). **2 disqualified, both real findings, one of them new:**

- **`Commodities / trend`** — 90.24% single-label occupancy (essentially always "sideways"). A
  20-symbol pooled basket score rarely moves far enough from zero to leave the sideways band; no
  single trend regime dominates a diversified commodity book for long. This was one of the three
  failure modes named in advance.
- **`Macro / credit`** — 7.90% coverage of the sector's full history. **Not anticipated in Phase 0**:
  `BAMLH0A0HYM2` and `BAMLC0A0CM` (the two credit-spread FRED series this repo caches) only start
  **2023-07-17**, thirty-three years later than `VIXCLS` (1990) or `T10Y2Y` (1976). Phase 0's own
  data introspection recorded these dates but did not flag them as a gap; the gap only became
  visible once Phase 2's coverage check was run against each sector's *full* history length rather
  than the panel's own (already-filtered) row count — an early version of the Phase 2 script made
  exactly that mistake and silently reported 100% coverage everywhere before being corrected.

Both are excluded from every Phase 3 comparison. The remaining 41 pairs all show 100% coverage
within their available window, flip rates between 0.001–0.041 (well under the 0.10 threshold — the
hysteresis is doing real work, not producing noise), and no other single-label occupancy above
90%.

## Phase 3 — Accuracy against independently-known regime periods

39 sector×dimension×ground-truth-source trials, honestly counted (14 episode-table + 25
mechanical-label), Bonferroni threshold α = 0.05/39 = 0.00128.

### (a) Episode table

The frozen 8-episode table, expanded to concrete (sector, dimension, expected label) targets —
COVID crash's "all volatility" expands to every basket with a `volatility` dimension. Balanced
accuracy against the episode window, plus three baselines reusing the already-ported, tested
`regime.prediction` functions (`persistence`, one-step `markov_forecast`, expanding
`prior_forecast`).

| sector / dimension | n_obs | balanced accuracy | best baseline | note |
|---|---:|---:|---|---|
| oil products / carry | 295 | 0.932 | markov (tied) | |
| oil products / term_structure | 295 | 0.919 | persistence (tied) | |
| Macro / yield_curve | 400 | 0.825 | persistence, +0.0025 (p=0.53, ns) | only non-tied row |
| precious / volatility | 42 | 0.429 | markov (tied) | COVID only |
| oil products / volatility | 42 | 0.286 | markov (tied) | COVID only |
| FX / volatility | 42 | 0.238 | markov (tied) | COVID only |
| **Macro / risk** | 567 | **0.353** | persistence (tied) | worse than chance for a 3-state label |
| natgas / volatility | 42 | 0.167 | markov (tied) | COVID only |
| base metals / volatility | 42 | 0.119 | markov (tied) | COVID only |
| softs / volatility | 42 | 0.119 | markov (tied) | COVID only |
| soy complex / volatility | 42 | 0.000 | persistence (tied) | COVID only |
| grains / volatility | 42 | 0.000 | persistence (tied) | COVID only |
| meats / volatility | 42 | 0.000 | persistence (tied) | COVID only |
| Commodities / volatility | 42 | 0.048 | markov (tied) | COVID only |

**A structural caveat, discovered here rather than assumed in advance:** 13 of 14 "best baseline"
rows above are exact *ties* — the baseline's hit rate exactly equals the engine's (`Macro /
yield_curve` is the lone exception, and its +0.0025 edge isn't significant either, p=0.53). Within a
single named
episode's short window, the engine's own flip rate is low (Phase 2: 0.001–0.041), so persistence
(yesterday's label) and one-step Markov are very often *literally identical* to the engine's own
label that day; the block-bootstrap difference is a constant-zero series with p=1.0 by construction.
That makes engine-vs-persistence/Markov comparisons within short episode windows structurally weak
evidence either way — not a finding about the engine, a finding about this particular test's power.
The class-prior baseline has the opposite problem: episodes are chosen because they are historically
*unusual*, so a sector/dimension's modal label over its whole history is almost never the episode's
expected label, guaranteeing class-prior scores near zero regardless of engine quality (this is why
class-prior never appears as "best baseline" above — it's the worst one, mechanically). Neither
comparison is very informative for source (a) alone; the raw balanced-accuracy numbers above are
more informative read directly than via either baseline, and source (b) below — much larger
samples, no dependence on a single short window — carries the real evidentiary weight for RM.

**Lead-lag** (signed trading days from episode onset to first correct label, search window
episode-start − 21 to episode-end; `None` = never matched): GFC +15, COVID crash (risk) +20, Hiking
cycle (risk) **−21** (early), Euro crisis **censored** (risk never flipped to risk_off in-window),
Taper tantrum −21 (early), Hiking cycle (yield_curve) **+110** (very late — the inversion label took
five and a half months to register), Oil glut (term_structure) censored, Energy backwardation −18
(early), Energy backwardation (carry) +20, COVID volatility across baskets +24 to +40 (COVID's
`min_dwell=5`+hysteresis meant "extreme" mostly arrived 1–2 months into a crash that peaked in
weeks). Median across the 14 scored pairs: **27 trading days** — RL's 21-day threshold, missed
mainly by the yield_curve inversion's extreme 110-day lag and the volatility dimensions' uniformly
slow COVID detection.

### (b) Mechanical labels

Forward-realized, no-human-judgment targets over each sector's full available history (truncated
before 2025-01-01): vol terciles, sign of forward return, sign of `T10Y2Y`, sign of the f1−f2 curve
spread — the last two are **contemporaneous consistency checks**, not forecasts (the underlying
series is literally what its indicator is built from).

| check | n_obs (range) | balanced accuracy (range) |
|---|---:|---:|
| `yield_curve` vs. `T10Y2Y` sign (Macro) | 4,889 | 0.981 |
| `term_structure` vs. f1−f2 sign (oil products, natgas) | 1,180 / 1,513 | 0.879 / 0.871 |
| `term_structure` vs. f1−f2 sign (Commodities, base metals, precious) | 307–1,501 | 0.428–0.679 |
| `volatility` vs. forward 21d vol tercile (9 baskets) | 5,924–6,166 | 0.538–0.657 |
| `trend` vs. forward 63d return sign (8 baskets) | 1,405–2,786 | **0.393–0.599** |

`yield_curve` (98.1%) and oil products/natgas `term_structure` (87–88%) are the highest raw
numbers in this notebook, but they are close to tautological rather than independent confirmation:
`macro.yield_curve` is built 50% from `T10Y2Y` itself, and `ts.curve_slope` *is* a transform of the
f1−f2 spread. High accuracy here mostly validates the scoring/banding arithmetic, not the regime
concept. `volatility` (0.54–0.66) is a genuinely independent, moderate, unremarkable result.
**`trend` vs. the sign of the forward 63-day return is the genuinely independent test with the
widest spread of sectors, and it is weak-to-null everywhere** — `precious` (0.393) and `meats`
(0.394) are *below* a coin flip, `oil products` (0.467) and `grains` (0.506) near it, only `FX`
(0.599) clears 0.60. Consistent with this programme's now 22+ prior null findings on trend-following
edge, extended here from "can you trade it" to "does the label even describe the forward direction."

None of the 39 trials — episode table or mechanical — clears the Bonferroni-corrected significance
threshold against its best baseline. RA and RM both report null.

## Phase 4 — Gates

See the gate table at the top. NL, RS, RC fire (the port is faithful and structurally sound); RA,
RM, RL do not (the labels don't demonstrably beat naive baselines, and react too slowly to named
crises). Per NEXT_PROMPT.md's asymmetry note, this is reported without hedging: **production has
been shipping labels every morning that pass every structural check but fail every accuracy check
against two independent ground-truth sources.**

## Phase 5 — Verdict and handoff

Per-dimension, not a single verdict:

| dimension | trustworthy for a future notebook to condition on? | why |
|---|---|---|
| `term_structure`, `carry` (oil products, natgas) | **No, without further work** | highest raw accuracy (85–93%), but the confirming mechanical check is close to tautological; needs an independent test before this changes |
| `yield_curve` (Macro) | **No, without further work** | same tautology problem — built from and checked against the same series |
| `volatility` (all baskets) | **No** | moderate mechanical accuracy (0.54–0.66) but failed to reach "extreme" during COVID in most baskets (0.0–0.29 against the episode table); reacted 24–40 trading days late where it did |
| `trend` (all baskets) | **No** | weak-to-null against forward 63-day returns (0.39–0.60); `Commodities/trend` additionally disqualified in Phase 2 |
| `risk` (Macro) | **No** | 35.3% accuracy against four named crisis episodes' `risk_off` label — worse than chance for a three-state classifier; missed 2 of 4 crises entirely within-window (Euro crisis censored, low overall accuracy) |
| `credit` (Macro) | **Not scoreable** | disqualified in Phase 2 — a real 2023-only FRED data gap, not a code defect |

**No dimension in this port is validated strongly enough, independent of near-tautological
mechanical checks, for a follow-up notebook to condition a strategy on it without further work.**
The two highest-accuracy dimensions would need a genuinely independent confirming test (not one
built from the same input series) before that changes. `trend` and `risk` — the two dimensions
most likely to matter for a regime-conditional strategy — look actively unreliable during the
episodes and horizons tested here.

This notebook authorizes no trading and spends no holdout. Both remain untouched (crypto
2025-07-01, futures 2025-01-01 → 2026-07-28) for whatever comes next. If a follow-up notebook wants
to condition on `term_structure`/`carry` or `yield_curve`, it should first build an independent
mechanical check that doesn't share an input series with the dimension's own indicators — that is
the concrete, actionable next step this notebook identifies, not "trade it."

## Substitutions and disclosed limitations

| item | substitution | reason | effect |
|---|---|---|---|
| `curve_slope(far=...)` | `"close_f3"` not production's default `"close_f12"` | this repo's curves have only 3 legs | `term_structure`/`carry` scored on a 3-month slope, not a 12-month one; not comparable to production 1:1 |
| Commodities basket `term_structure`/`carry` | 5-symbol aggregate (CL/NG/GC/SI/HG), not 20 | only 5 of 20 legs have a curve file | basket-level term_structure/carry numbers are a narrower aggregate than the basket's other 3 dimensions |
| `macro.cot_noncomm` `requires={"macro"}` bug | preserved bug-for-bug | NEXT_PROMPT.md §3.3: this notebook scores what production actually runs | `risk` dimension is quietly reweighted onto 5 indicators whenever `cot=None` (always, in this repo — no E-MINI S&P 500 COT series exists here) rather than skipped |
| `Macro / credit` FRED coverage | scored on 2023-07-17 → present only (7.9% of history) | `BAMLH0A0HYM2`/`BAMLC0A0CM` cache starts there | disqualified in Phase 2; not scored in Phase 3 |
| indicator/stats/carry primitives | ported fresh into `regime/indicators.py`, not delegated to a repo-local equivalent | no repo-local pandas equivalents exist (`src/features.py`/`research.py` are polars, different schema and semantics) | none — confirmed bit-identical to source via the Phase 0 end-to-end fidelity check |
| Episode-table baselines | persistence/Markov/class-prior computed as whole-window predictions from `regime.prediction`, not day-by-day evolving forecasts | short (weeks-to-months) episode windows | see Phase 3(a) caveat — weak discriminating power for RA specifically, disclosed rather than hidden |
| representative sector price (mechanical `trend`/`volatility` checks) | first symbol in each basket's `symbols_used` list | no cross-sectional composite index exists for these baskets | a genuine simplification — a volume- or liquidity-weighted composite might score differently; not built here given scope |

## Scope discipline confirmed

- One notebook (`src/research/014_market_regime_engine_and_accuracy.ipynb`, 35 cells, executes
  end-to-end with zero error cells), one results file (this one), one pre-registration JSON
  (`phase_0_14_preregistration.json`, frozen before Phase 1, never edited), infrastructure in
  `src/regime/` (durable, not `src/research/tmp/` scratch), tests in `tests/test_regime_*.py`.
- No strategy was built, no Sharpe was computed, no holdout was touched. Both holdouts (crypto
  2025-07-01, futures 2025-01-01 → 2026-07-28) remain exactly as the ledger left them after
  notebook 013.
- `sweep.py` was not ported and no weight/band/window was tuned to make any gate fire.
- No crypto config was added; the universe is exactly `configs/universe.yaml`'s daily
  futures-and-macro scope, copied verbatim from production.
- The cron, the database, and Telegram were not ported — this repo has none of the three and gains
  none here.
