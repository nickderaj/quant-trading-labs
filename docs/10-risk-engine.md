# 10. The risk engine (`src/risk/`)

This is an operator document. It describes running software, not a concept. For background
on the statistics involved (Kupiec, Christoffersen, Expected Shortfall, tail dependence), see
[06](06-scoring-rules-and-calibration.md) and [07](07-extreme-value-theory.md).

---

## What it is

`src/risk/` computes Value-at-Risk (VaR), Expected Shortfall (ES), and portfolio tail risk for
16 daily commodity and equity-index futures. The density families and calibration procedure it
uses come from [notebook 008](../src/results/008_commodity_tails_and_risk.md), which found and
holdout-tested them.

The engine used to live inside a single 1,833-line research script, with no documented input
requirements, a family selection stored as a loose JSON file, and no way to check whether its
calibration was still accurate months later. This module is that engine, rebuilt as durable,
tested, monitored software. Nothing here is new: every number the engine produces traces back
to something already validated in the research notebooks.

It answers "how much could this position lose", not "should I hold it." See
[Scope](#scope) at the end for what it does not do.

---

## The data contract (`src/risk/hygiene.py`)

The engine's calibration only holds if the input data has been through four cleaning steps,
all discovered during notebook 008's own investigation into why its early numbers looked wrong:

| problem | fix | what happens without it |
|---|---|---|
| Mislabeled spread/differential contracts appear in outright OHLCV data | `flag_contaminated_rows`'s two-tier persistence rule | NG's `NG202507` prints a near-zero or negative close on 97% of its roughly 575 days; a volume cutoff alone flags it and CL's genuine 2020-04-20 negative settle together, or flags neither |
| Front-month selection lands on a stale single quote | `liquidity_screen` | a quiet day on a nominally front-month contract gets treated as real price discovery |
| `roll_calendar.parquet` lists contract months that were nominally listed but never really traded | `liquid_contract_months` (lifetime volume of at least 5,000) | PL's F1 series was 57% null before this filter existed |
| Additive back-adjustment crosses zero over a long history | use `log_return_ratioadj`, never `log_return_backadj` | roughly 200 rolls over 16 years accumulate an offset that sends early prices negative, manufacturing single-day "returns" over 100% that are pure splice artifact, corrupting every tail statistic downstream |

The last one is the dangerous bug: its symptom is fat tails, which is exactly what this engine
looks for. A pipeline that silently used `log_return_backadj` would produce a more alarming
risk number, not an obviously broken one.

`hygiene.build_risk_inputs(...)` runs the full chain (hygiene filter, liquidity screen, liquid
contract months, continuous series construction, return conventions) and returns a frame
carrying only the ratio-adjusted return, under the column name `log_return`, stamped with a
marker that proves where it came from. `hygiene.assert_risk_inputs(frame)` is the check that
`fit()` (in `src/risk/__init__.py`) runs before fitting anything. It rejects, with a named
`RiskInputError` rather than a warning:

- a return column that can't be proven to come from `build_risk_inputs` (checked via the
  stamp, not the column name, since a caller can rename a column);
- any single-day return larger than 50% that the hygiene filter did not already flag;
- any run of more than 3 identical closing prices in a row (the longest run seen on real,
  cleaned data is 3, so anything longer is unexamined);
- fewer than 100 finite observations;
- any 20-day rolling window with realized volatility at or below 1e-12 (a frozen, stale block
  of data, not an individually quiet day).

All four synthetic versions of the original bugs are correctly rejected with the right named
error, and the clean 16-product data passes with zero false rejections. This is gate DC.

---

## The family map: a density family, not a GARCH model

`src/risk/configs/family_map_v1.json`, loaded via `families.load_family_map("v1")`, assigns
each of the 16 products a density family:

| family | products | count |
|---|---|---:|
| `ged` | BZ, CL, ES, GC, HO, PA, PL, RB, SI, ZS | 10 |
| `hansen_skewt` | ZC, ZL, ZW | 3 |
| `nig` | KE, ZM | 2 |
| `johnsonsu` | NG | 1 |

This is the part most likely to be misread. Notebook 008's own write-up calls the winning
models `garch_ged`, `gjr_ged`, and so on, because that phase ranked GARCH-family models by
out-of-sample log-score. `fit_risk_model` strips the variance-process prefix and fits only the
density family, unconditionally, on the full return series. Time variation is supplied
separately by a caller-provided EWMA scale (`ewma_vol`, lambda 0.94) through
`RiskModel.var_conditional` and `es_conditional`. A family map entry of `"ged"` does not mean
the GARCH-GED model was promoted; it means the GED innovation density was promoted, fit
unconditionally, and conditioned at call time by a caller-supplied EWMA scale
(`src/risk/model.py`'s module docstring makes the same point).

Each entry in the map carries its own provenance: the out-of-sample log-score that selected
it, the runner-up family and its score, whether the winning margin is statistically
significant after correcting for multiple comparisons, the fitting window, and which notebook
phase selected it. Only GC and SI have a statistically significant winner, and both were won
by `ged`.

Because only 2 of 16 products have an individually significant winner, a three-way comparison
was pre-registered and run once, refitting the whole walk-forward coverage test under three
different family policies:

| policy | description | pass count |
|---|---|---:|
| P1 | the shipped per-product map (10 `ged`, 3 `hansen_skewt`, 2 `nig`, 1 `johnsonsu`) | 15/16 |
| P2 | `ged` everywhere | 14/16 |
| P3 | `ged` everywhere except GC and SI, which keep their own family | 14/16 |

P1 won outright and is what ships, following the rule fixed before the comparison was run:
ship whichever policy has the highest pass count, and use the simplest policy (P2) on a tie.
Per-product family selection earned its complexity here. The outcome was not a foregone
conclusion given how few individual winners were significant, and it came back in favor of
keeping all 16 selections. This is gate FS.

A product that isn't in the v1 map raises `families.UnseenProductError` rather than silently
defaulting to `ged`. See [Refit cadence](#refit-cadence-and-the-unseen-product-rule) below.

---

## The monitoring battery (`src/risk/calibration.py`)

`CalibrationMonitor` computes, per product and per evaluation window, at the 1% and 2.5%
levels: Kupiec unconditional coverage, Christoffersen independence, Christoffersen conditional
coverage, and the Acerbi-Szekely Z and bootstrap p-value (both tails, 1% only), plus the
observed-versus-expected violation rate and the longest run of consecutive violations. It
reports a per-product status of `ok`, `warn`, or `breach`, with a failure mode attached:

| failure mode | signature | likely cause |
|---|---|---|
| `coverage` | Kupiec breaches, independence does not | the scale is wrong: EWMA lambda or the refit cadence needs attention |
| `clustering` | Kupiec passes, independence breaches | conditioning is too slow to react; this is exactly how PA failed in development |
| `shape` | coverage is fine, but Acerbi-Szekely Z turns significantly positive | the density's tail has drifted thin |
| `both` | Kupiec and independence both breach | escalate |

PA is the only development failure, and it fails on clustering, not coverage: Kupiec passes
comfortably (p = 0.706, an observed rate of 0.01097 against an expected 0.01), while
Christoffersen independence fails (p = 0.0123). Its violations cluster; its overall count is
fine. On the holdout period, RB and SI fail instead while PA passes. At a sample size around
490, this reshuffling looks like sampling noise around a stable overall rate, not a meaningful
statement about which specific products are fragile.

Two related but distinct policies are worth keeping separate:

1. The gate that checks the monitor rediscovers these known failures uses the same raw,
   uncorrected per-product rule (`kupiec_p > 0.05` and `independence_p > 0.05`) that the
   original notebook used to determine pass or fail. The notebook never applied a multiple-
   comparison correction to this particular test; that correction was applied elsewhere, to
   density-selection significance.
2. The monitor's actual alerting layer (`CalibrationMonitor.evaluate_batch` and
   `evaluate_batch_from_hits`) applies a Benjamini-Hochberg correction across the 16 products
   within each run, separately per test and per level, plus a persistence rule: two
   consecutive breaching windows before anything pages, not one. Sixteen products at two
   levels and four tests, run repeatedly, would otherwise manufacture false alarms at any
   fixed significance level.

The monitor correctly flags PA with failure mode `clustering` in development and nothing else
falsely, and correctly flags RB and SI on the holdout period: an exact rediscovery in both
periods. This is gate MB. Alert thresholds (violation-rate ratio: warn at 1.5x expected,
breach at 2.0x; longest violation run: warn at 4, breach at 6) were fixed before the monitor
ever ran on real data or against this rediscovery test. Tuning thresholds until a known failure
reappears would make the rediscovery test meaningless as evidence of anything.

---

## Refit cadence and the unseen-product rule

`risk/families.py` fixes both of these rather than leaving them to a future caller:

- Refit interval: 63 trading days, about one calendar quarter.
- Minimum history for a refit: 500 observations, several multiples of `fit_risk_model`'s own
  minimum of 100, so a refit is never fit to a bare-minimum window.
- If a refit fails, the correct behavior is to keep serving the previous fitted model and
  raise a monitoring alert, never to fall back to a normal distribution. Normal models are
  known to understate 1% ES.
- A product outside the 16-product envelope is not silently assigned `ged`.
  `family_map.family_for(product)` raises `UnseenProductError`. The only sanctioned path is
  `families.fit_new_product()`, which itself raises `NotImplementedError`: running the full
  ranking process on a genuinely new product is a modelling exercise, not something this
  module does on its own. The function documents and enforces the refuse-or-extend contract
  without performing the ranking itself.

---

## Ingestion and refresh (`src/risk/ingest.py`)

`risk.ingest.refresh(products=None, as_of=None) -> IngestReport` re-reads the databento
parquet data (`src/research/data/market/databento/{ohlcv,contracts.parquet,
roll_calendar.parquet}`), re-runs `hygiene.build_risk_inputs` per product, and writes the
results to `src/risk/data/`, a durable location separate from any scratch directory.

- It's idempotent via content hashing. A product's freshly built frame is hashed over its
  content, not file bytes or timestamps. If the hash matches what's already on disk, the file
  is left untouched rather than rewritten. This makes a scheduled re-run safe by construction.
- It fails loud, never silently partial. A product whose contract check fails is absent from
  the written output, with the rejection reason recorded, rather than present with a quietly
  shortened series.
- There's no new vendor integration, no API keys, no network calls. If the underlying data
  cache is stale, `refresh` reports the stale last-observation date and the dashboard shows
  it as-is.

---

## The dashboard (`src/risk/serve.py`, `src/risk/dashboard/template.html`)

`risk.serve.build_snapshot(as_of=None) -> dict` produces one JSON document: the only thing the
dashboard reads. Per product, it includes the family, fit window, last observation date,
current volatility (`sigma_t`, from `ewma_vol`), VaR and ES at 1% and 2.5% for 1, 5, and 10-day
horizons, trailing violation counts against expectation, the monitor's status and failure mode,
and a 250-day recent return series with the VaR band for plotting. At the book level it
includes portfolio VaR and ES under all three dependence assumptions side by side, the pairwise
lower-tail dependence map, and the named stress scenarios.

`risk.serve.render_dashboard` generates a single self-contained HTML file. The snapshot JSON is
inlined directly into the page rather than fetched separately, so it opens correctly from a
local file, where fetching a local JSON file is blocked by most browsers. It's static HTML,
CSS, and vanilla JavaScript: no server, no build step, no framework, no CDN. Numbers are
formatted once, in Python, not in JavaScript.

The dashboard is explicitly allowed to show current dates, including dates inside and after
the futures holdout window that begins 2025-01-01. This is not a holdout spend: the dashboard
fits no model on that data, chooses no threshold from it, and makes no gate decision from it.
The fitted models are frozen artifacts from the development window, and the dashboard only
evaluates them forward, which is the same thing the original holdout test already did once and
recorded. What's forbidden, and what `risk.serve` does not do, is feed any displayed number
back into a fitting, selection, or threshold decision.

---

## The public API (`src/risk/__init__.py`)

The only module a production caller should import from. Everything else is implementation
detail.

| entry point | question it answers |
|---|---|
| `fit(product, returns_frame) -> RiskModel` | give me a fitted model for this product, from contract-checked inputs |
| `var(model, alpha, sigma_t, horizon) -> float` | what is the alpha-VaR today, at today's volatility |
| `es(model, alpha, sigma_t, horizon) -> float` | what is the alpha-ES today |
| `portfolio(models, weights, dependence, ...) -> PortfolioRisk` | what is the book's VaR/ES under a given dependence assumption |
| `stress(model, scenario_returns) -> StressResult` | what would this position have done in a named historical event |
| `monitor(product, model, returns, sigma_t) -> CalibrationStatus` | is the model still calibrated |
| `size(model, alpha, sigma_t, risk_budget) -> float` | what notional consumes exactly this much risk budget |
| `refresh(products, as_of) -> IngestReport` | pull the latest cleaned inputs |
| `snapshot(as_of=None) -> dict` | everything the dashboard needs, in one document |

`fit()` runs two checks on `returns_frame` before handing it to `fit_risk_model`:
`hygiene.assert_risk_inputs`, the data-quality contract described above, and
`hygiene.assert_not_holdout` (`src/risk/hygiene.py`, raises `HoldoutLeakError`). The second
check refuses any frame whose date column extends past 2024-12-31, the boundary where the
futures holdout begins, so the holdout can't be spent again through a live fitting call. This
check lives only in `fit()`, not in `assert_risk_inputs` itself, because `risk.ingest.refresh`
also calls `assert_risk_inputs` and is explicitly allowed to see current dates for the
dashboard.

Four design choices are kept deliberately visible rather than smoothed over, because they
matter to what the numbers mean:

1. `sigma_t` is always supplied by the caller, never held as internal model state. A static,
   full-sample VaR failed out-of-sample coverage during development; `ewma_vol` is the
   validated, causal source, but the call stays explicit rather than hidden behind a default.
2. Horizon scaling uses the square root of the horizon, which assumes returns are independent
   and identically distributed. That assumption is documented on `RiskModel.var` and `.es`
   directly: a 10-day VaR is a scaling assumption, not a distribution separately fit to 10-day
   returns.
3. All three dependence modes are reported as a comparison, not collapsed into a single
   default. If one default is needed, it's `"empirical"`, never `"gaussian"`: the Gaussian
   copula has zero asymptotic tail dependence by construction and understates joint tail risk.
4. `size()` returns a notional amount and nothing else: `risk_budget / var(...)`. No
   direction, no position, no strategy.

---

## A known coverage gap

`src/research/tmp/dist_lib.py` (847 lines: `fit_garch11`, `rolling_garch_forecast`, HAR-RV,
EWMA, range estimators, Diebold-Mariano) has no test file. It isn't on the serving path:
`RiskModel` fits `distributions._fit_normal` / `_fit_t` and `densities.REGISTRY[...].fit`
unconditionally, conditioned only by `ewma_vol`, and never calls GARCH at serve time. But it is
on the offline path that produced the family map, since the original GARCH-family ranking used
it. This is recorded here as a known gap in the offline tier, not something fixed by this
module, so the family map's provenance isn't mistaken for test-covered code.

---

## The validated envelope

This engine should never be read as claiming more than one row of this table at a time. The
daily-commodity claim rests on notebook 008; the intraday claims rest on notebook 005. They're
different models, on different assets, at different frequencies, and notebook 005's own result
failing at the 1-day interval is a warning that the intraday finding does not extend to daily
data by itself.

| claim | validated on | not validated on |
|---|---|---|
| conditional VaR coverage at 1% / 2.5% | 16 daily commodity futures, about 1,805 observations per product in development plus 490 in holdout | any other product, asset class, or frequency |
| normal models understate 1% ES | the same 16 products plus an equity-index control (ES) | |
| GARCH-t as the best density | crypto at 1h, 4h, and 12h intervals, and it explicitly fails at 1 day | daily commodities |
| full GARCH-EVT tail calibration | crypto at the 12h interval only | every other interval, including 1 day |
| risk-gating improves a trading book | nothing: this was tested twice and failed both times | everything |

---

## The gate table

Every gate here is pre-registered in `risk_engine_preregistration.json` and reported in
`src/research/tmp/risk_engine_results.json`. All hard gates currently pass.

| gate | claim | threshold | current result | hard or soft |
|---|---|---|---|---|
| PR | promoted `src/risk/` reproduces the original walk-forward results exactly | every compared field matches to a relative tolerance of 1e-12; pass count 15/16 | 0 mismatches, pass count 15/16 | hard |
| PH | promoted code matches the stored holdout numbers, with no holdout recomputation | 14 products pass coverage, 11 reject ES calibration at 1% | matches, 0 mismatches | hard |
| DC | the data contract rejects each of the four known bugs | 4 of 4 rejected with the correct named error, 0 false rejections on the clean 16-product frame | 4/4 rejected, 0 false rejections | hard |
| FS | the winning family policy is shipped | pre-committed rule: highest pass count wins, ties go to the simpler policy | P1 wins, 15 vs. 14/14, and is shipped | soft |
| MB | the monitor rediscovers the known development and holdout failures, with no false positives | exact match on both product sets | PA flagged with clustering in development, no false positives; RB and SI flagged on holdout | soft |
| DT | `portfolio_risk` and every Monte Carlo path are seed-reproducible | two runs with the same seed produce bit-identical output | bit-identical | hard |
| NL | the volatility-to-VaR path has no lookahead | passes for all 16 products at truncations of 1, 5, 21, and 63 days | 16/16 pass | hard |
| IR | refreshing twice with unchanged source data produces identical output | exact byte equality per product | verified via content-hash equality | soft |
| CI | linting, formatting, type checking, and tests are all clean | standard project checks | clean | soft |

None of these is a discovery gate. There's no outcome of this project that authorizes a trade.
The pass condition throughout is that the promoted code reproduces the original certified
numbers exactly, and keeps doing so under ongoing monitoring.

---

## Scope

This engine estimates risk. It doesn't take positions, size trades, or generate a P&L series.
Using its output to gate a trading signal was tested twice earlier in this research programme
and hurt risk-adjusted returns both times, so this module doesn't attempt that. A separate,
unrelated correction to this research programme's Sharpe-ratio estimator is planned as future
work and is out of scope here.
