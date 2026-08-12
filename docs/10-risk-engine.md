# 10 — The risk engine (`src/risk/`)

This is an operator document, not a teaching chapter — it describes running software, not a
concept. For the concept-level background (Kupiec, Christoffersen, ES, tail dependence, ...)
see [06](06-scoring-rules-and-calibration.md) and [07](07-extreme-value-theory.md).

---

## What it is

`src/risk/` computes VaR, ES, and portfolio tail risk for 16 daily commodity/equity-index
futures, using density families and a calibration procedure certified by
[notebook 008](../src/results/008_commodity_tails_and_risk.md) Phase 7 and reproduced on
holdout in Phase 8. It exists because that engine previously lived in a 1,833-line research
scratch file (`src/research/tmp/commod_lib8.py`) with its input hygiene undocumented at any
call boundary, its family selection a loose JSON, and no monitoring of whether its calibration
was still true a month later. This project ports it, verbatim, into durable, tested,
monitorable software and adds nothing that was not already validated
(`NEXT_PROMPT.md` §0).

**It answers "how much could this position lose," never "should I hold it."** No alpha, no
positions, no Sharpe, no backtest, no equity curve — see [Scope](#scope) at the end.

---

## The data contract (`src/risk/hygiene.py`) — a hard precondition

The engine's certified numbers are conditional on inputs cleaned by four bugs discovered in
008 Phase 0. A caller that feeds raw vendor data into `fit_risk_model` gets a plausible-looking
VaR with none of 008's calibration guarantees:

| bug | fix | what happens without it |
|---|---|---|
| 1. Mislabeled spread/differential contracts in outright OHLCV | `flag_contaminated_rows`'s two-tier persistence rule | NG's `NG202507` prints a near-zero/negative close on 97% of its ~575 days; a volume cutoff alone flags it and CL's genuine 2020-04-20 negative settle together, or neither |
| 2. Front-month selection on a stale single quote | `liquidity_screen` | a quiet day on a nominally-front contract is treated as price discovery |
| 3. `roll_calendar.parquet` lists nominally-listed but near-dead contract months | `liquid_contract_months` (lifetime volume ≥ 5,000) | PL's F1 series was 57% null before this filter |
| 4. Additive back-adjustment crosses zero over a long history | use `log_return_ratioadj`, never `log_return_backadj` | ~200 rolls over 16 years accumulate an offset that sends early prices negative, manufacturing single-day "returns" over 100% that are pure splice artifact — and it corrupts every tail statistic downstream |

Bug 4 is the dangerous one: its symptom is fat tails, which is exactly what this engine looks
for, so a pipeline that silently used `log_return_backadj` would produce a *more* alarming risk
number, not an obviously broken one.

`hygiene.build_risk_inputs(...)` runs the full chain (hygiene filter → liquidity screen →
liquid contract months → continuous series → return conventions) and returns a frame carrying
**only** `log_return_ratioadj`, under the name `log_return`, stamped with a provenance
attribute. `hygiene.assert_risk_inputs(frame)` is the precondition checker `fit()` (in
`src/risk/__init__.py`) calls before fitting; it rejects, with a named `RiskInputError` rather
than a warning:

- a return column not provably from `build_risk_inputs` (checked via the provenance attribute,
  not the column name — a caller can rename a column);
- any `|log_return| > 0.5` on a day the hygiene filter did not already flag;
- any run of more than 3 consecutive identical `close_f1` prices (008's stale-bar audit found a
  worst-case run of 3 on real, hygiene-passed data — anything longer is new and unexamined);
- fewer than 100 finite observations;
- any 20-observation rolling window with `realized_vol <= 1e-12` (the frozen-bar rule that
  corrupted notebook 003 twice).

**Gate DC result** (`risk_engine_results.json`): all 4/4 synthetic bugs rejected with the
correct named error, 0 false rejections on the clean 16-product frame. Fires.

---

## The family map and its provenance — a density family, not a GARCH model

`src/risk/configs/family_map_v1.json`, loaded via `families.load_family_map("v1")`, assigns
each of the 16 products a density family:

| family | products | n |
|---|---|---:|
| `ged` | BZ, CL, ES, GC, HO, PA, PL, RB, SI, ZS | 10 |
| `hansen_skewt` | ZC, ZL, ZW | 3 |
| `nig` | KE, ZM | 2 |
| `johnsonsu` | NG | 1 |

**This is the single most likely thing about the engine to be misread.** 008's prose calls the
winning models `garch_ged`, `gjr_ged`, and so on, because Phase 3 ranked *GARCH-family* models
by OOS log-score. `fit_risk_model` strips the variance-process prefix and fits only the
**density family**, unconditionally, on the full return series; time-variation is supplied
separately by a caller-provided EWMA scale (`ewma_vol`, λ=0.94) via
`RiskModel.var_conditional`/`es_conditional`. A `family_map` entry of `"ged"` does **not** mean
"the GARCH-GED model was promoted" — it means "the GED innovation density was promoted, fit
unconditionally, and conditioned at call time by a caller-supplied EWMA scale"
(`src/risk/model.py`'s module docstring makes the same distinction).

Each product's map entry carries provenance: the Phase 3 OOS log-score that selected it, the
runner-up family and its score, whether the winning margin is BH-significant
(`best_wins_significantly_bh`), the fitting window, and `selected_by: "008 Phase 3"`
(`family_map_v1.json`). Only **GC and SI** clear BH-significance for their Phase 3 winner —
both won by `ged` — which is why the per-product map's complexity was worth measuring rather
than assumed.

**The measurement.** Since only 2/16 products have a statistically distinguishable winner,
`NEXT_PROMPT.md` §5.2 pre-registered a three-way comparison of the whole Phase 7 walk-forward
coverage battery under three family policies, run once:

| policy | description | Gate RE pass count |
|---|---|---:|
| **P1** | the shipped per-product map (10 `ged`, 3 `hansen_skewt`, 2 `nig`, 1 `johnsonsu`) | **15/16** |
| **P2** | `ged` everywhere | 14/16 |
| **P3** | `ged` everywhere except GC/SI, which keep their own family | 14/16 |

**Gate FS result:** P1 wins outright (15/16 vs. 14/16 for both alternatives) and is shipped,
per the pre-committed rule ("ship whichever policy has the highest Gate RE pass count; tie →
P2"). Per-product selection earned its complexity — the comparison was not a foregone
conclusion given only 2/16 products' winners were individually significant, and it came back in
favour of keeping all 16 selections.

A product not in `v1`'s map raises `families.UnseenProductError` rather than silently
defaulting to `ged` — see [Refit cadence](#refit-cadence-and-the-unseen-product-rule) below.

---

## The monitoring battery (`src/risk/calibration.py`)

`CalibrationMonitor` computes, per product per evaluation window, at 1% and 2.5%: Kupiec
unconditional coverage, Christoffersen independence, Christoffersen conditional coverage, and
the Acerbi-Székely Z/bootstrap p-value (both tails, 1% only) — plus the observed-vs-expected
violation rate and the maximum violation cluster length. It emits a per-product
`ok`/`warn`/`breach` status with a `failure_mode`:

| failure mode | signature | likely cause |
|---|---|---|
| `coverage` | Kupiec breaches, independence does not | the scale is wrong — EWMA λ or the refit cadence |
| `clustering` | Kupiec passes, independence breaches | conditioning too slow — **this is PA's development failure exactly** |
| `shape` | coverage fine, Acerbi-Székely Z significantly positive | the density's tail has drifted thin — the Gate CE failure mode |
| `both` | Kupiec and independence both breach | escalate |

**PA is the sole development failure, and it fails on clustering, not coverage:** Kupiec passes
comfortably (`p=0.706`, observed rate 0.01097 against an expected 0.01) while Christoffersen
independence fails (`p=0.0123`) — its violations cluster; its unconditional count is fine. On
holdout, **RB and SI** fail (both independence/clustering-type failures) while PA passes; at
n≈490 this reshuffling is sampling noise around a stable overall rate, not a meaningful story
about which products are individually fragile.

**Two distinct policies, not to be conflated** (see the in-code comment at
`src/research/tmp/run_risk_04_monitor.py:180-189`):

1. **Gate MB's rediscovery test** uses the *raw*, uncorrected per-product
   `kupiec_p > 0.05 and independence_p > 0.05` rule — the same rule
   `phase_7_results.json`/`phase_8_holdout_results.json` used to determine 008's own pass/fail.
   008 never BH-corrected Gate RE itself (BH there gated *density-selection* significance, a
   different test). This is what `MB` is checked against.
2. **The monitor's live alerting layer** (`CalibrationMonitor.evaluate_batch`/
   `evaluate_batch_from_hits`) applies Benjamini-Hochberg correction across the 16 products
   within each run, separately per test per level, plus a persistence rule: `k=2` consecutive
   breaching windows before anything pages, not one (`risk_engine_preregistration.json` →
   `calibration_monitor`). This is what a production caller actually runs on a schedule; 16
   products × 2 levels × 4 tests, run repeatedly, would otherwise manufacture false alarms at
   any fixed α.

**Gate MB result:** the monitor flags PA with `failure_mode == "clustering"` in development and
no other product falsely, and flags RB and SI on holdout — exact rediscovery in both periods.
Fires. Alert thresholds (violation-rate ratio: warn ≥1.5×, breach ≥2.0× expected; max cluster
length: warn ≥4, breach ≥6) were pre-registered in `risk_engine_preregistration.json` before the
monitor ever ran on real data or against §6.4's rediscovery test, per the discipline
`NEXT_PROMPT.md` §12 states explicitly: tuning thresholds until a known failure reappears would
invalidate the rediscovery test as evidence of anything.

---

## Refit cadence and the unseen-product rule

`risk/families.py` fixes both, rather than leaving them to a future caller:

- `REFIT_INTERVAL_TRADING_DAYS = 63` (~1 calendar quarter).
- `MIN_HISTORY_FOR_REFIT = 500` observations — several multiples of `fit_risk_model`'s own
  ≥100-observation guard, so a refit is never fit to a bare-minimum window.
- If a refit fails (`fit_risk_model` returns `None`), the correct behaviour is to **keep
  serving the previous fitted model and raise a monitoring alert** — never fall back to a
  normal distribution, since Gate CE is precisely the finding that the normal fallback
  understates 1% ES.
- A product outside `v1`'s 16-product envelope is **not** silently assigned `ged`.
  `family_map.family_for(product)` raises `UnseenProductError`; the only sanctioned path is
  `families.fit_new_product()`, which itself raises `NotImplementedError` — running Phase 3's
  ranking battery on a genuinely new product is a modelling exercise, out of scope for this
  productionisation (`NEXT_PROMPT.md` §2, ground rule 3), so the function documents and
  enforces the refuse-or-extend contract without performing the ranking itself.

---

## Ingestion and refresh (`src/risk/ingest.py`)

`risk.ingest.refresh(products=None, as_of=None) -> IngestReport` re-reads the databento parquet
root (`src/research/data/market/databento/{ohlcv,contracts.parquet,roll_calendar.parquet}`),
re-runs `hygiene.build_risk_inputs` per product, and writes versioned outputs to `src/risk/data/`
— a durable location, not `research/tmp/`.

- **Idempotent via content-hashing.** A product's freshly-built frame is SHA-256'd over its
  content (not file bytes or timestamps); if the hash matches what is already on disk, the file
  is left untouched (`status == "unchanged"`) rather than rewritten. This is true byte-identity
  by construction, which is what makes a scheduled re-run safe.
- **Fail loud, never silently partial.** A product whose contract check fails
  (`assert_risk_inputs` raises `RiskInputError`) is *absent* from the written outputs with the
  rejection reason recorded (`status == "rejected"`), never present with a quietly shortened
  series.
- No new vendor integration, no API keys, no network calls. If the databento cache is stale,
  `refresh` reports the stale `last_observation` date and the dashboard displays it as-is.

---

## The dashboard (`src/risk/serve.py`, `src/risk/dashboard/template.html`)

`risk.serve.build_snapshot(as_of=None) -> dict` produces one JSON document — the only thing the
dashboard reads. Per product: family, fit window, last observation date, current `sigma_t`
(from `ewma_vol`), VaR/ES at 1%/2.5% for horizons 1/5/10, trailing violation counts vs.
expected, the monitor's `status`/`failure_mode`, and a 250-day recent return series with the VaR
band for plotting. At the book level: `portfolio_risk` under all three dependence modes side by
side, the pairwise lower-tail dependence map, and the named stress scenarios.

`risk.serve.render_dashboard` generates a **single self-contained HTML file** — the snapshot
JSON is inlined into a `<script>` placeholder in `template.html` (not fetched, so the page opens
correctly from `file://`, where `fetch()` of a local file is blocked by most browsers' CORS
policy). Static HTML + CSS + vanilla JS: no server, no build step, no framework, no CDN. Numbers
are formatted once, in Python, not in JS.

**The dashboard is explicitly permitted to display current dates, including dates inside and
after the spent futures holdout window (2025-01-01 onward).** This is *not* a holdout spend:
the module fits no model on that data, chooses no threshold from it, and makes no gate decision
from it — the fitted models are frozen artifacts from the development window, and the dashboard
only *evaluates* them forward, which is exactly what 008 Phase 8 already did once and recorded
(`NEXT_PROMPT.md` §7.4). What is forbidden, and what `risk.serve` does not do, is feed any
displayed number back into a fitting, selection, or threshold decision.

---

## The public API (`src/risk/__init__.py`)

The only module a production caller should import from; every other submodule is
implementation detail.

| entry point | question it answers |
|---|---|
| `fit(product, returns_frame) -> RiskModel` | give me a fitted model for this product, from contract-checked inputs |

`fit()` calls two guards on `returns_frame` before delegating to `fit_risk_model`: `hygiene.assert_risk_inputs` (the data-quality contract, §"The data contract" above) and `hygiene.assert_not_holdout` (`src/risk/hygiene.py`, `HoldoutLeakError`) — the latter refuses any frame whose `date` column extends past `TRUNCATION` (2024-12-31, the same boundary 015's `lib15.TRUNCATION` uses), so the futures holdout cannot be re-spent through a live fitting call even though `refresh()`/`snapshot()` are explicitly permitted to see current dates (§7.4 above — that is why the check lives only in `fit()`, not in `assert_risk_inputs` itself, which `risk.ingest.refresh` also calls). The reproduction gates (PR/PH) never call `risk.fit()` at all — they read stored JSON or call the low-level, ndarray-only `fit_risk_model` directly — so they need no explicit exemption from this guard.

| `var(model, alpha, sigma_t, horizon) -> float` | what is the α-VaR today, at today's volatility |
| `es(model, alpha, sigma_t, horizon) -> float` | what is the α-ES today |
| `portfolio(models, weights, dependence, ...) -> PortfolioRisk` | what is the book's VaR/ES under a given dependence assumption |
| `stress(model, scenario_returns) -> StressResult` | what would this position have done in a named historical event |
| `monitor(product, model, returns, sigma_t) -> CalibrationStatus` | is the model still calibrated |
| `size(model, alpha, sigma_t, risk_budget) -> float` | what notional consumes exactly this much risk budget |
| `refresh(products, as_of) -> IngestReport` | pull the latest cleaned inputs |
| `snapshot(as_of=None) -> dict` | everything the dashboard needs, in one document |

Four design points are preserved deliberately rather than tidied away, because a 008 finding
depends on each staying visible:

1. **`sigma_t` is always caller-supplied**, never model-internal state — a static full-sample
   VaR failed OOS coverage in Phase 7; `ewma_vol` is the validated, causal source, but the call
   stays explicit.
2. **`horizon` scaling is `sqrt(horizon)`**, an i.i.d. assumption documented on `RiskModel.var`/
   `es` themselves — a 10-day VaR is a scaling assumption, not a fitted 10-day distribution.
3. **All three dependence modes are a reported comparison, not a default to be chosen away.**
   If a single default is needed, it is `"empirical"`, never `"gaussian"` — the Gaussian copula
   has zero asymptotic tail dependence by construction and understates joint tail risk.
4. **`size()` returns a notional, nothing else** — `risk_budget / var(...)`, no direction, no
   position, no strategy. 015 closed the directional question; nothing here reopens it.

---

## A known coverage gap

`src/research/tmp/dist_lib.py` (847 lines: `fit_garch11`, `rolling_garch_forecast`, HAR-RV,
EWMA, range estimators, Diebold-Mariano) has **no test file**. It is not on the serving path —
`RiskModel` fits `distributions._fit_normal`/`_fit_t` and `densities.REGISTRY[...].fit`
unconditionally, conditioned only by `ewma_vol`, and never calls GARCH at serve time — but it
**is** on the offline certification path that produced the family map (Phase 3's GARCH-family
ranking). This is recorded here as a known gap in the offline tier, not fixed in this project,
so the family map's provenance is not mistaken for test-covered code.

---

## The validated envelope, stated as a boundary

**Do not let this engine quietly claim more than one row at a time.** The daily-commodity claim
rests on 008; the intraday claims rest on 005; they are different models on different assets at
different frequencies, and 005 Gate A's own failure at 1d is a warning that the intraday result
does not extend to daily by itself.

| claim | validated on | not validated on |
|---|---|---|
| conditional VaR coverage at 1% / 2.5% | 16 daily commodity futures, ~1,805 obs/product development + 490 obs holdout | any other product, any other asset class, any other frequency |
| normal models understate 1% ES | same 16 + an equity-index control (ES) | — |
| GARCH-t as density winner | crypto at 1h / 4h / 12h (005 Gate A) — and it explicitly fails at 1d | daily commodities |
| GARCH-EVT full tail calibration | crypto at 12h only (005 Gate B) | every other interval, including 1d |
| risk-gating improves a trading book | **nothing — this was tested twice and failed both times** | everything |

---

## The gate table

Every gate below is pre-registered in `risk_engine_preregistration.json` and reported in
`src/research/tmp/risk_engine_results.json`; `all_hard_gates_fire: true`.

| gate | phase | claim | threshold | actual result | hard/soft |
|---|---|---|---|---|---|
| **PR** | 3 | promoted `src/risk/` reproduces 008 Phase 7 exactly | every compared field matches at `rtol=1e-12`; `gate_RE.fires == true`, pass count 15/16 | 0 mismatches; `gate_RE_fires: true`, pass count 15/16 | **hard** |
| **PH** | 3 | promoted code matches the *stored* Phase 8 numbers (no holdout recomputation) | `RE_holdout_pass_count == 14`, `CE_holdout_reject_1pct_count == 11` | 14 and 11, 0 mismatches | **hard** |
| **DC** | 1 | `assert_risk_inputs` rejects each of the four 008 Phase 0 bugs | 4/4 rejected, correct named error; 0 false rejections on the clean 16-product frame | 4/4 rejected; 0 false rejections | **hard** |
| **FS** | 2 | the winning family policy of P1/P2/P3 is shipped | pre-committed: highest Gate RE pass count; tie → P2 | P1 wins, 15 vs 14/14; P1 shipped | soft |
| **MB** | 3 | the monitor rediscovers PA (`clustering`) in development and RB/SI on holdout, no false positives | exact product-set match in both periods | PA flagged clustering, no false positives in development; RB/SI flagged on holdout | soft |
| **DT** | 0/3 | `portfolio_risk` and every Monte Carlo path are seed-reproducible | two runs, same seed, bit-identical | bit-identical | **hard** |
| **NL** | 5 | `regime.evaluation.no_lookahead_check` passes over `ewma_vol` → `var_conditional`, all 16 products, truncations (1, 5, 21, 63) | 16/16 pass at all four truncations | 16/16 pass | **hard** |
| **IR** | 4 | `refresh` twice with unchanged vendor data is byte-identical | exact byte equality per product | (idempotency verified via content-hash equality, `ingest._content_hash`) | soft |
| **CI** | all | ruff, ruff format, mypy, pytest all clean | per `CLAUDE.md` | clean | soft |

**None of these is a discovery gate.** There is no outcome of this project that authorises a
trade — the pass condition throughout is "the promoted code reproduces 008's certified numbers
exactly, and keeps doing so under monitoring," not "we found something new."

---

## Scope

No alpha, no positions, no Sharpe, no backtest, no equity curve: risk-gating a known
gross-profitable signal was tested twice in this programme (notebook 006 Phase 6, notebook 007
Gate RG) and hurt net Sharpe both times, so nothing in this productionisation attempts it again.
A related, separately-scoped correction to this programme's deflated-Sharpe estimator is
recorded as deferred work for notebook 017 (`NEXT_PROMPT.md` §14) and is out of scope here.
