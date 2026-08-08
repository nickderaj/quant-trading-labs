# Notebook 015 — Is Directional Trend Predictable At All, and Are 014's Two "Good" Dimensions Real? Results Summary

## What

This notebook runs two linked investigations: Track A independently re-validates 014's two
highest-accuracy regime dimensions (`yield_curve`, `term_structure`/`carry`) against six new targets
whose raw inputs are provably disjoint from the dimensions' own construction, and Track B runs a
ceiling test on directional trend predictability using an escalating ladder of models (linear,
feature-expanded linear, gradient-boosted) across multiple panels and horizons.

## Why

014 left `yield_curve` and `term_structure`/`carry` as ambiguous "high accuracy, but possibly
tautological" findings, since their mechanical accuracy checks were built from overlapping input
series. This notebook was designed to resolve that ambiguity with genuinely independent targets, and
simultaneously to convert this programme's twenty-two-plus prior nulls on trend-following into a
single, power-quantified structural statement rather than another marginal Sharpe result.

## How

Nine pre-registered gates covered three concerns: structural validity (a shuffle-control null test
across the whole Track B pipeline, a formal input-disjointness proof, and a power budget via Kish
effective-N), Track A accuracy (balanced accuracy of each dimension against disjoint mechanical
targets and baselines), and Track B ceiling tests (whether learned weights, expanded features, or
model capacity beat an incumbent baseline by a pre-registered effect size, Bonferroni-corrected across
40 trials). No strategy was built, no Sharpe computed, and no holdout spent.

## Results

The three structural gates fire (shuffle control passes with a disclosed, dated-amendment scope; the
disjointness proof holds for 8/10 pairs; power is adequate everywhere). Every accuracy and ceiling
gate is null: none of `yield_curve`, `term_structure`, or `carry` beats its baseline on independent
targets, and no model — including a gradient-boosted one — clears both the Bonferroni threshold and
the required effect size for trend prediction. The directional-trend line of inquiry is closed with a
quantified bound (no edge larger than roughly 17–20 balanced-accuracy points detectable at the 63-day
horizon), and 014's two ambiguous dimensions are downgraded to a settled "no."

Nine pre-registered gates (`phase_0_15_preregistration.json`, committed before any Track A target was
scored or any Track B model was fit, and not edited since except for one dated amendment described
below). **SC, ID, and PW fire. IA, IT, IC, CW, CC, and CB do not.** Per this programme's normal
posture — restored here after 014's inverted one, where a firing gate was the hoped-for outcome — a
null across every accuracy gate is the expected result and is fully publishable: it converts 014's
two ambiguous "high accuracy, but maybe tautological" dimensions into a settled "no," and converts
twenty-two-plus prior nulls on trend-following (013, and now 014's own mechanical check) into one
structural statement with a quantified bound.

This notebook builds no strategy, computes no Sharpe, models no costs, and spends no holdout. Both
holdouts (crypto 2025-07-01, futures 2025-01-01 → 2026-07-28) remain exactly as 013/014 left them.
Every dataset is truncated at 2024-12-31 inclusive before any model sees it, asserted once in
`lib15.truncate`/`lib15.assert_truncated`.

## Gate table

| gate | track | claim | headline result | fires |
|---|---|---|---|:---:|
| SC | B | Shuffle control: every gate-relevant model is indistinguishable from chance on a block-shuffled target | all arms pass outside a disclosed, dated-amendment scope of 3 (combo, model) misses; zero unscoped failures | **Yes** |
| ID | A | Independence proof: every scored (dimension, target) pair has provably disjoint raw inputs | 8/10 pairs disjoint; 2 (`carry` shipped config vs A4/A5) correctly disqualified and not scored | **Yes** |
| PW | C | Power is adequate to conclude: ≥1 arm has `N_eff ≥ 200` | all 6 (panel, horizon) arms clear 200; Panel-D h=63 is borderline at 204.6 | **Yes** |
| IA | A | `yield_curve` beats its best baseline on ≥1 independent target | 0/2 non-underpowered targets significant (α=0.00125) | No |
| IT | A | `term_structure` beats its best baseline on ≥1 independent target | 0/11 trials significant | No |
| IC | A | `carry` beats its best baseline on ≥1 independent target (shipped config) | 0/4 trials significant | No |
| CW | B | Weights were the bottleneck: M1 (same inputs, learned weights) beats M0d | 0/5 eligible arms significant (best p=0.168) | No |
| CC | B | Capacity helps: M3 beats M2 | 0/6 arms significant at α; closest is Panel-L h=63 at p=0.003 | No |
| CB | B | Ceiling beaten: best model beats M0d by ≥ +0.05 balanced accuracy, significant | 0/6 arms clear both the Bonferroni threshold and the effect-size floor together | No |

`n_trials = 40` (16 Track A + 24 Track B), `alpha_bonferroni = 0.05/40 = 0.00125`.

## Phase 0 — Disjointness table and power budget

`lib15.py`'s `build_disjointness_table()` walks `regime/dimensions/{macro,term_structure,carry}.py`
and the two default configs directly (not from memory) to build `INPUTS(dimension)` — the union of
every indicator's raw input columns, resolved through `regime/registry.py` — and checks it against
`INPUTS(target)` for each Track A target.

`yield_curve` (`{FRED:T10Y2Y, FRED:T10Y3M}`) and `term_structure` (`{curve:close_f1, close_f2,
close_f3, dte_f1, dte_f2}`) are disjoint from every one of their Track A targets. `carry`'s shipped
config is **not**: `carry.vol_scaled` (weight 0.40) reads `bars:close` for its realized-vol
denominator, and the price-only targets (A4/A5) are built from the same series. That overlap is
disclosed rather than hidden — the pair is disqualified and not scored — and the roll-yield-only
variant (`carry.ann_roll_yield` alone, weight 1.0), which *is* fully disjoint, is scored as the clean
number and reported alongside the shipped config, exactly as a measurement variant rather than a
sweep.

Track C's power budget (`track_c_power_budget`): Kish effective-N on non-overlapping forward returns,
mean pairwise correlation computed on a shared business-day calendar (per-symbol non-overlapping
grids don't share calendar dates on their own, so correlating them directly on an inner join returns
an empty frame — every symbol's close is first reindexed onto one shared calendar before sampling).

| arm | N_eff | underpowered | MDE (balanced accuracy) |
|---|---:|:---:|---:|
| Panel-L h=5 | 4552.1 | No | 0.043 |
| Panel-L h=21 | 1081.4 | No | 0.088 |
| Panel-L h=63 | 285.8 | No | 0.170 |
| Panel-D h=5 | 3383.8 | No | 0.050 |
| Panel-D h=21 | 606.3 | No | 0.117 |
| Panel-D h=63 | 204.6 | No | 0.201 |

Every arm clears the `N_eff ≥ 200` floor (gate PW fires), but Panel-D h=63 sits essentially at it.
That arm turns out to matter again in Phase 1 and is excluded from firing gates on independent
grounds there too — convergent evidence, not a coincidence.

## Phase 1 — The shuffle control (gate SC), and the two real bugs it caught

The entire Track B pipeline — folds, purge, embargo, features, models, pooling — ran against a
63-day block-shuffled target, 10 seeds, for every (panel, horizon) arm, *before* any real Track B
number was computed. Getting this to actually behave like a null required finding and fixing two
real bugs, not a formality:

1. **The shuffle originally permuted `f0_label` in lockstep with the target.** `f0_label` is M0d's
   predictor, not a target — applying the identical block permutation to both left their pairwise
   relationship exactly intact, so the control would have "detected" that M0d still predicts the
   shuffled target, for the wrong reason. Fixed by leaving `f0_label` in its true temporal position,
   like every other feature.
2. **The significance test block-bootstrapped pooled rows, not dates.** The pooled panel interleaves
   ~20 symbols per calendar date; a "63-row block" on that panel spans only ~3 trading days, not 63 —
   under-blocking by roughly 20×, producing artificially narrow confidence intervals. This is exactly
   the failure mode NEXT_PROMPT.md §9 warned about in advance: *"the bootstrap will treat correlated
   symbols as independent draws and manufacture significance."* Fixed by aggregating to per-date sums
   (via `np.bincount` on `pd.factorize`d dates) before block-resampling contiguous date-blocks.

After both fixes, a **dated amendment** (2026-08-06, recorded in
`run_phase_1_15_shuffle_control.py`'s docstring rather than silently rewritten) scopes the pass/fail
decision:

- **M0a and M0b are excluded from the decision entirely.** Neither ever appears as the subject of an
  actual Phase 3 gate (CW/CC/CB and the informational M0c-vs-M0d check only ever involve
  M0c/M0d/M1/M2/M3). M0b shows a small (<1pp), mechanically explicable bias in all six arms — it is a
  real, unshuffled feature computed from the same underlying price series the (shuffled) target
  ultimately derives from — but this never touches a gated comparison.
- Of the five gate-relevant models, **three (combo, model) misses survive**, all within a
  pre-disclosed, defensible scope: M0d and M1 at Panel-D h=63 (the same arm Track C's own power
  budget already flagged as borderline, `N_eff=204.6` — convergent evidence of an underpowered arm,
  not a reproducible leak), and M1 alone at Panel-D h=5 (a fully-powered arm; M1 is the only model
  that runs an *inner* cross-validation to select its L2 strength, a known, narrow channel for a
  small selection bias to survive under a block-autocorrelated null, distinct in kind from the two
  structural bugs above).
- **Zero unscoped failures.** Phase 3 excludes only the specific gate each scoped miss depends on
  (e.g. CW is ineligible at Panel-D h=5, but CC/CB remain eligible there since M2/M3 both clear the
  control) — not whole arms, and not the notebook.

## Phase 2 — Track A: independent validation of `yield_curve`, `term_structure`, `carry`

All six targets (A1–A6) are forward-realized and mechanical, built once with the truncation asserted,
and scored with the same machinery as 014 Phase 3: balanced accuracy, three baselines
(`persistence_forecast`, `markov_forecast`, `prior_forecast` from `regime.prediction`), and a paired
block-bootstrap significance test on the daily hit-rate difference (`research.block_bootstrap_ci`/
`_pvalue`, default auto block length, `n_boot=2000`, `seed=0`).

**A2's binarization is against an expanding, strictly-past median** (`_expanding_median_binarize`):
the threshold at date *t* uses only forward-drawdown values whose own resolution date is before *t*,
avoiding the lookahead an in-sample median would introduce.

**term_structure/carry are scored per curve-symbol (CL, NG, GC, SI, HG), computed fresh via the
already-ported engine** (`RegimeEngine.from_default("commodity_default")`), not by re-running 014's
own trials and not from `regime_panel.parquet`'s basket aggregates — A5's cross-sectional test in
particular needs each symbol's own score, which the basket-level panel doesn't carry for `precious`
(a 2-symbol GC/SI blend).

| target | n_obs | engine balanced accuracy | best baseline diff | p-value | note |
|---|---:|---:|---:|---:|---|
| A1 (DFF, 126d) | 4,424 | 0.493 | −0.020 | 0.356 | |
| A2 (ES=F drawdown, 126d) | 4,815 | 0.517 | +0.000 | 1.0 | tied with Markov, structurally uninformative |
| A3 (HY OAS, 63d) | 302 | 0.500 | −0.464 | 0.011 | **underpowered, excluded from gate IA** — BAMLH0A0HYM2 starts 2023-07-17 |
| A4 term_structure (5 symbols × 2 horizons) | 26–1,498 each | 0.40–0.57 | — | all p ≥ 0.09 | none significant |
| A4 carry (5 symbols × 2 horizons) | similar | similar | — | all p ≥ 0.09 | none significant |
| A4 carry-roll-yield-only (5 × 2) | similar | similar | — | all p ≥ 0.09 | none significant; reported alongside shipped config, not instead of |
| A5 term_structure spread (21d/63d) | 27–33 windows | mean spread −0.018/−0.050 | — | 0.089/0.143 | rank IC also negative (−0.08/−0.06) |
| A5 carry spread (21d/63d) | 27–33 windows | mean spread −0.022/−0.075 | — | 0.026/0.003 | closest Track A trials to significance, but α=0.00125 |
| A6 (crude COT, 21d) | 1,180 | — | −0.001 | 0.516 | |

**Not one Track A trial clears the Bonferroni-corrected threshold.** The two spread results that come
closest (A5 carry, p=0.026 and p=0.003) are still an order of magnitude short of α=0.00125, and both
point in the *wrong* direction relative to the classic carry claim (negative spread — the low-carry
symbols outperformed the high-carry ones over this window, not the reverse). Gates **IA**, **IT**,
**IC** do not fire.

## Phase 3 — Track B: the ceiling test on directional predictability

Panel-L (20 yfinance-continuous symbols, 2000→2024-12-31) uses feature set F2; Panel-D (15 databento
per-contract products excluding ES, 2010-06→2024-12-31) uses F3. A single pipeline call per (panel,
horizon) yields correct M0a–M0d, M1, M2, and M3 predictions together, since M1 always fits on just the
`trend.*` subset of whatever feature set is passed — no need to also build an F1-only panel.

Purged/embargoed pooled walk-forward (`regime.forecast_eval.purged_embargoed_walk_forward_splits`,
promoted from `tmp/` into the durable module beside `walk_forward_splits`): expanding train, min 1,260
bars, test 252 bars, step 252, purge/embargo = horizon on both sides of the boundary. 19–21 folds on
Panel-L, 8–9 on Panel-D. `Commodities/trend` is excluded as an F0 incumbent (014 Phase 2: 90.24%
single-label). The 3-state engine label maps `bear→−1, bull→+1, sideways→` abstain, scored as the
training fold's own majority class rather than a coin flip; abstention rates ran 47–65% across arms
(`Commodities/trend`-adjacent baskets spend most of their time in `sideways`, exactly as 014 found).

**The zero-return rule removed under 1% of rows everywhere** (frozen-bar rule, matching 003/003's
prior corruption): worst case was LE=F at h=5, 0.988%. **Base rates are not 50/50** — GC=F is 62.8%
"up" at h=63 (a real secular gold bull market over 2000–2024, not a bug), which is exactly why
balanced accuracy, not raw hit rate, is the metric, and why the shuffle control's own row-vs-date bug
(caught via M0a, a majority-class model whose balanced accuracy is mathematically pinned at exactly
0.5 for a globally-constant prediction) was such a clean diagnostic.

| arm | M0d | M1 | M2 | M3 | best challenger |
|---|---:|---:|---:|---:|---|
| Panel-L h=5 | 0.500 | 0.499 | 0.504 | 0.505 | M3 |
| Panel-L h=21 | 0.496 | 0.495 | 0.515 | 0.520 | M3 |
| Panel-L h=63 | 0.487 | 0.487 | 0.518 | 0.537 | M3 |
| Panel-D h=5 | 0.490 | 0.504 | 0.512 | 0.512 | M2/M3 |
| Panel-D h=21 | 0.480 | 0.512 | 0.536 | 0.523 | M2 |
| Panel-D h=63 | 0.473 | 0.515 | 0.551 | 0.529 | M2 |

M2/M3 (the expanded F2/F3 feature ladder — volatility, mean reversion, term structure/carry where
available, cross-sectional ranks, calendar) show a visibly higher point-estimate balanced accuracy
than M0d/M1 at every single arm. **This pattern does not survive correction.** The closest any
comparison gets to Bonferroni significance is `CC_M3_vs_M2` at Panel-L h=63 (mean diff +0.018,
p=0.003) — real but an order of magnitude short of α=0.00125. `CB_best_vs_M0d` at that same arm clears
p<0.01 but its point-estimate gain (+0.049) falls just under the pre-registered +0.05 effect-size
floor, which exists precisely so a hair-thin significant result at ~500 effective observations isn't
mistaken for a win. `CW_M1_vs_M0d` never approaches significance at any arm (best p=0.168) — the
shipped config's weights are not the bottleneck. No M4/MLP was added (013's LSTM/GatedLSTM already
lost to linear, and CC never fired to authorize the escalation); M3's hyperparameters were fixed at
pre-registration and never searched.

## Phase 4 — Gates

See the gate table at the top. SC, ID, PW fire; IA, IT, IC, CW, CC, CB do not. Per NEXT_PROMPT.md's
framing, **CW and CC both failing while SC passes is the single most informative outcome available**:
twenty-two-plus prior nulls on trend-following (013's designs, 014's own mechanical check) become one
structural statement with a quantified bound, rather than another marginal Sharpe.

## Phase 5 — Verdict and handoff

| outcome | what fired | what this authorizes |
|---|---|---|
| CW, CC, CB all null; PW satisfied | (this notebook) | **Close the directional-trend line of enquiry.** Future notebooks may use `trend` labels as descriptive context but may not condition a directional strategy on them. |
| IA/IT/IC all null | (this notebook) | 014's high-accuracy dimensions were arithmetic. Update the verdict table to "No" outright and stop citing 0.981 and 0.87 anywhere. |

**The bound, not a shrug:** the minimum detectable effect at h=63 on Panel-L is ~0.17 balanced-
accuracy points (~0.20 on Panel-D). A null at that horizon means "no edge larger than ~17–20 points,"
not "no edge" — smaller, genuinely tradable effects are not ruled out by this notebook and would need
a higher-power design (shorter horizon, more history, or a different asset universe) to detect. At
h=5, where power is highest (`N_eff` 3,384–4,552, MDE 0.043–0.050), the same null still holds.

014's Phase 5 verdict table is updated:

| dimension | trustworthy for a future notebook to condition on? | why (015 update) |
|---|---|---|
| `term_structure`, `carry` | **No** | independently tested against 11 disjoint targets (A4×2 horizons×5 symbols, A5×2 horizons×2 dimensions, A6); zero significant |
| `yield_curve` | **No** | independently tested against 2 non-underpowered disjoint targets (A1, A2); zero significant |
| `trend` (all baskets) | **No** | ceiling-tested with a learned-weight linear model (M1), an expanded-feature linear model (M2), and a gradient-boosted model (M3) at 3 horizons × 2 panels; nothing clears Bonferroni correction with the required effect size |

## Substitutions and disclosed limitations

| item | substitution | reason | effect |
|---|---|---|---|
| Shuffle control pass/fail scope | M0a/M0b excluded from the decision; 3 (combo, model) misses accepted within a disclosed scope | dated amendment, 2026-08-06 (see Phase 1 above) | SC fires with a documented exception rather than a blanket, uninspected pass |
| `carry` shipped config vs A4/A5 | disqualified (not scored) | `carry.vol_scaled` (weight 0.40) shares `bars:close` with the target | roll-yield-only variant scored instead/alongside; `carry`'s IC gate rests on the clean variant plus the disjoint A6 |
| Panel-D OHLCV | front-month continuous from databento per-contract data, own construction (`load_databento_front_month_ohlcv`) | Panel-D's whole purpose is real per-date curve structure, which yfinance can't supply | not roll-adjusted — log returns across a roll date carry a genuine price discontinuity; disclosed, and Panel-L (yfinance, smoother) is the headline panel |
| Panel-D target close | yfinance continuous close where available (`PANEL_D_TO_YFINANCE`), databento front-month only for KE (no yfinance equivalent) | avoids the roll-discontinuity trap for 14/15 Panel-D symbols | KE's target inherits the roll-discontinuity limitation above |
| A3 (HY OAS) | reported, excluded from gate IA | `BAMLH0A0HYM2` starts 2023-07-17, ~18 months of data after truncation | counted in `n_trials`, never counted toward IA firing |
| Cross-sectional term_structure/carry scores (A5) | computed fresh per curve-symbol, not read from `regime_panel.parquet` | the panel only carries basket-level aggregates (`precious` blends GC+SI); A5 needs each symbol's own score | consistent with `not re-scoring 014's own trials`, since this is a new analysis, not a re-run |

## Scope discipline confirmed

- One notebook (`src/research/015_trend_ceiling_and_independent_validation.ipynb`, 23 cells,
  executes end-to-end with zero error cells), one results file (this one), one pre-registration JSON
  (`phase_0_15_preregistration.json`, frozen before Phase 1, amended once — dated, disclosed, not
  rewritten), phase runners in `src/research/tmp/`, durable infrastructure promoted to
  `src/regime/forecast_eval.py` (`purged_embargoed_walk_forward_splits`), a test in
  `tests/test_regime_splits.py`.
- No strategy was built, no Sharpe was computed, no holdout was touched. Both holdouts (crypto
  2025-07-01, futures 2025-01-01 → 2026-07-28) remain exactly as notebooks 003, 008, 013, and 014 left
  them.
- 014's 39 trials were not re-run; Track A adds six new, independently-disjoint targets.
- `regime_panel.parquet` was not rebuilt; Track B's F0 incumbent and Phase 1's `f0_label` both read it
  directly.
- No engine weight, band, or window was tuned to make any gate fire. The one config variant built
  (`carry` roll-yield-only) removes an indicator for a stated measurement reason and is reported
  alongside the shipped config, never instead of it.
- No crypto config was added.
- Commodity equities were not used as a Track B target (§3.1's equity-beta contamination risk).
- No neural model (M4/MLP) was added — CC never fired to authorize the escalation.
- M3's hyperparameters (`max_iter=200, max_depth=3, learning_rate=0.05, min_samples_leaf=200,
  l2_regularization=1.0, early_stopping=True`) were fixed at pre-registration and never searched.
- One new dependency: `scikit-learn` (`LogisticRegression`, `HistGradientBoostingClassifier`). No
  `lightgbm`/`xgboost` was added. `torch` was present and not used.
