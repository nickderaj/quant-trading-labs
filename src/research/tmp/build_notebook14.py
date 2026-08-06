import json


def md(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


cells = []

cells.append(
    md("""\
# Notebook 014 — Port the Daily Market-Regime Engine, and Score Its Accuracy
""")
)

cells.append(
    code("""\
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "tmp")
import json

TMP = "tmp"

def load(name):
    with open(f"{TMP}/{name}") as f:
        return json.load(f)

prereg = load("phase_0_14_preregistration.json")
print("Gates pre-registered:", list(prereg["gates"].keys()))
print("Episode table entries:", len(prereg["episode_table"]))
print("Holdouts untouched:", prereg["scope"]["holdouts_untouched"])
""")
)

cells.append(
    md("""\
## Phase 0 — Ground truth, port verification, pre-registration

`src/regime/` is a module-for-module port of `../ultron/libs/finance/src/ultron_finance/regime/`
(engine) and `../ultron/apps/trading-labs/common/regime_report/` (report layer), rewired onto this
repo's parquet data via `src/regime/loaders.py`. Before any scoring runs, this phase verifies three
things: every data source NEXT_PROMPT.md sec3 claims exists actually does (with row counts and date
ranges), the port reproduces the source engine's scores where both can be executed, and the ported
`no_lookahead_check` passes as a **hard gate** on every symbol in the universe. All three, plus the
frozen episode table and gate definitions, are committed to `phase_0_14_preregistration.json`
before Phase 1 ever builds or plots the historical panel.
""")
)

cells.append(
    code("""\
intro = prereg["ground_truth_phase0_findings"]
print(f"Bars: {len(intro['bars'])} symbols")
for sym in sorted(intro["bars"]):
    row = intro["bars"][sym]
    print(f"  {sym:8s} {row['rows']:6d} rows  {row['first']} -> {row['last']}")
print(f"\\nFRED: {len(intro['fred'])} series")
for series in intro["fred"]:
    row = intro["fred"][series]
    print(f"  {series:14s} {row['rows']:6d} rows  {row['first']} -> {row['last']}")
print(f"\\nCOT: {intro['cot']}")
print(f"\\nCurves: {len(intro['curves'])} symbols")
for sym in intro["curves"]:
    row = intro["curves"][sym]
    print(f"  {sym:8s} {row['rows']:6d} rows  {row['first']} -> {row['last']}")
""")
)

cells.append(
    code("""\
print("Disclosed gaps:")
for gap in intro["disclosed_gaps"]:
    print(f"- {gap}")
""")
)

cells.append(
    md("""\
### Port-fidelity check

Both `config_hash()` equality on all four YAMLs *and* a true end-to-end run of the source engine
(via its own `uv` environment in `../ultron/libs/finance`) against this repo's ported engine, on
identical synthetic fixtures, were attempted.
""")
)

cells.append(
    code("""\
fidelity = prereg["port_fidelity"]
print("Method:", fidelity["method"])
print("config_hash equal:", fidelity["config_hash_equal"])
print("End-to-end scores/labels equal:", fidelity["end_to_end_equal"])
""")
)

cells.append(
    md("""\
### `no_lookahead_check` — hard gate

Run at truncations `(1, 5, 21, 63)` for every symbol in the universe (macro index + every
commodity/FX leg across every basket), using each symbol's actual wiring (curve where this repo has
one, macro/COT only for the macro sector). A failure anywhere halts the notebook -- there is nothing
to score if the engine peeks. The opt-in oil_products COT wiring path is gated separately since it
changes `RegimeInputs.cot` for CL=F only.
""")
)

cells.append(
    code("""\
nl = prereg["no_lookahead_gate"]
print("All symbols passed:", nl["all_passed"])
n_pass = sum(1 for v in nl["per_symbol"].values() if v["passed"])
print(f"{n_pass}/{len(nl['per_symbol'])} symbols passed")
failures = {k: v for k, v in nl["per_symbol"].items() if not v["passed"]}
if failures:
    print("FAILURES:", failures)
print("oil_products COT opt-in path:", nl["oil_products_cot_opt_in"])
""")
)

cells.append(
    md("""\
### Frozen episode table (ground truth source (a))

Dated, publicly-known regime episodes, drafted and frozen here before any label was ever plotted --
see NEXT_PROMPT.md sec6 Phase 3 for the discipline this protects.
""")
)

cells.append(
    code("""\
import pandas as pd

episodes = pd.DataFrame(prereg["episode_table"])
episodes[["episode", "start", "end", "sector", "dimensions"]]
""")
)

cells.append(
    md("""\
## Phase 1 — Build the historical panel

`build_regime_report` run over each symbol's full available history (`as_of=None`), producing a
daily score/label frame per sector x dimension for every sector in `configs/universe.yaml` (Macro +
9 baskets). Persisted long-format to
`src/research/data/market/research/regime_panel.parquet` so later phases -- and any future notebook
-- load it rather than recompute it.
""")
)

cells.append(
    code("""\
phase1 = load("phase_1_14_results.json")
print("Panel rows:", phase1["panel_rows"], "->", phase1["panel_path"])
print("Errors:", phase1["errors"])
for sector in phase1["sectors"]:
    print(
        f"  {sector['name']:14s} ({sector['kind']:6s}) "
        f"{len(sector['dimensions'])} dims, {sector['rows']} rows, "
        f"{sector['first']} -> {sector['last']}, "
        f"used={sector['symbols_used']} skipped={sector['symbols_skipped']}"
    )
""")
)

cells.append(
    md("""\
### Snapshot heatmap and 6-month history (most recent as-of)
""")
)

cells.append(
    code("""\
from IPython.display import Image, display

display(Image(filename=phase1["charts"]["snapshot"]))
""")
)

cells.append(
    code("""\
if phase1["charts"]["history"]:
    display(Image(filename=phase1["charts"]["history"]))
""")
)

cells.append(
    md("""\
### Full-history label ribbon

One horizontal band per sector x dimension, coloured by which label was active, spanning the whole
sample, with the frozen episode table (grey shading) overlaid. First the sectors the episode table
actually names (legible at full width), then every sector.
""")
)

cells.append(
    code("""\
display(Image(filename=phase1["charts"]["ribbon_episode_sectors"]))
""")
)

cells.append(
    code("""\
display(Image(filename=phase1["charts"]["ribbon_full"]))
""")
)

cells.append(
    md("""\
## Phase 2 — Descriptive: does it behave like a regime model at all?

Per sector x dimension: transition matrix, regime durations, `label_stability` (flip rate, average
spell, labelled coverage), `expected_remaining_duration`. Three failure modes were named in advance
(NEXT_PROMPT.md sec6 Phase 2) and checked mechanically here, not by eye: >90% single-label
occupancy, a flip rate high enough that hysteresis isn't binding, and a dimension NaN for most of
the sample. Any dimension tripping one of these is disqualified from Phase 3 scoring.
""")
)

cells.append(
    code("""\
phase2 = load("phase_2_14_results.json")
stats = pd.DataFrame(phase2["per_sector_dimension"])
stats[
    ["sector", "dimension", "n_bars", "pct_of_sector_history_covered", "flip_rate", "avg_duration",
     "max_single_label_occupancy", "disqualified"]
].sort_values(["sector", "dimension"])
""")
)

cells.append(
    code("""\
print(f"{len(phase2['disqualified'])} sector x dimension pairs disqualified from Phase 3:")
for item in phase2["disqualified"]:
    print(f"  {item['sector']} / {item['dimension']}: {item['reasons']}")
""")
)

cells.append(
    md("""\
`Macro / credit` (7.9% coverage) is a real, previously-undisclosed data gap surfaced here, not in
Phase 0: `BAMLH0A0HYM2`/`BAMLC0A0CM` (the two credit-spread FRED series) only start 2023-07-17 in
this repo's cache, 33 years later than `VIXCLS`/`T10Y2Y`. `Commodities / trend` fails the occupancy
check at 90.24% -- essentially always "sideways" for a 20-symbol pooled basket, i.e. no single trend
regime dominates a diversified commodity book long enough to move the aggregate score outside the
sideways band. Both are excluded from every Phase 3 comparison below.
""")
)

cells.append(
    md("""\
## Phase 3 — Accuracy against independently-known regime periods

Two ground-truth sources, both frozen in Phase 0: (a) the hand-labelled episode table, scored as
balanced accuracy of the engine's causal label against three baselines (persistence, one-step
Markov, expanding class-prior, all reusing the already-ported `regime.prediction` functions) within
each episode's date window; (b) mechanical, forward-realized-quantity labels (vol terciles, sign of
forward return, sign of `T10Y2Y`, sign of the f1-f2 curve spread) scored the same way over the whole
available history. Both holdouts are untouched: every series is truncated to dates before
2025-01-01 before any target or baseline is built.
""")
)

cells.append(
    code("""\
phase3 = load("phase_3_14_results.json")
print("n_trials:", phase3["n_trials"])
print("Excluded (disqualified in Phase 2):", phase3["disqualified_excluded"])
""")
)

cells.append(
    md("""\
### (a) Episode table
""")
)

cells.append(
    code("""\
rows = []
for key, v in phase3["episode_table"]["per_sector_dimension"].items():
    if v.get("excluded"):
        rows.append({"sector_dimension": key, "n_obs": None, "balanced_accuracy": None,
                      "vs_best_baseline": v["reason"]})
        continue
    vb = v.get("vs_best_baseline", {})
    rows.append({
        "sector_dimension": key,
        "n_obs": v["n_obs"],
        "balanced_accuracy": round(v["engine"]["balanced_accuracy"], 3),
        "vs_best_baseline": f"{vb.get('baseline')} diff={vb.get('mean_hit_rate_diff'):+.4f} p={vb.get('pvalue'):.4f}" if vb else None,
    })
pd.DataFrame(rows)
""")
)

cells.append(
    md("""\
**A structural caveat, discovered here rather than assumed in advance:** within a single named
episode's short window, the engine's own flip rate is low (Phase 2), so the persistence and
one-step-Markov baselines are very often *literally identical* to the engine's own label that day --
they inherit whatever the engine said yesterday, which is usually what it says today too. That
makes "engine vs. persistence/Markov" comparisons within episode windows structurally weak evidence
either way. The class-prior baseline has the opposite problem: episodes are chosen because they are
historically unusual, so the sector/dimension's *modal* label over its whole history is almost never
the episode's expected label, guaranteeing class-prior scores near zero regardless of whether the
engine is any good. Neither baseline comparison is very informative for source (a) alone; the
mechanical-label comparisons in (b) below, with much larger `n_obs` and less structural dependence on
the same short window, carry the real evidentiary weight.
""")
)

cells.append(
    code("""\
lags = phase3["episode_table"]["lead_lag_days"]
import numpy as np
print(f"Lead-lag across {len(lags)} scored episode/dimension pairs (trading days, negative = early):")
print(sorted(round(l) for l in lags))
print(f"Median: {np.median(lags):.1f} trading days")
""")
)

cells.append(
    md("""\
### (b) Mechanical labels
""")
)

cells.append(
    code("""\
rows = []
for key, v in phase3["mechanical_labels"].items():
    if v.get("insufficient_data"):
        continue
    vb = v.get("vs_best_baseline", {})
    rows.append({
        "sector_dimension_check": key,
        "n_obs": v["n_obs"],
        "balanced_accuracy": round(v["engine"]["balanced_accuracy"], 3),
        "hit_rate": round(v["engine"]["hit_rate"], 3),
        "vs_best_baseline": f"{vb.get('baseline')} diff={vb.get('mean_hit_rate_diff'):+.4f} p={vb.get('pvalue'):.4f}" if vb else None,
    })
pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
""")
)

cells.append(
    md("""\
`Macro / yield_curve` vs. contemporaneous `T10Y2Y` sign (98.1% balanced accuracy) and
`.../term_structure` vs. the f1-f2 spread sign (87-89% for oil products/natgas) are close to
tautological rather than independent confirmations: `macro.yield_curve` is built directly from
`T10Y2Y` (50% of its weight), and `ts.curve_slope` *is* a transform of the f1-f2 spread. High
accuracy there mostly validates the scoring/banding arithmetic, not the regime concept. `trend` vs.
the sign of the forward 63-day return is the genuinely independent test, and it is weak-to-null
across every basket (balanced accuracy 0.39-0.60, i.e. coin-flip to worse-than-coin-flip) --
consistent with this programme's twenty-two-plus prior null findings on trend-following edge.
""")
)

cells.append(
    md("""\
## Phase 4 — Pre-registered gates

Unlike every alpha-gate notebook before this one, **a gate firing (passing) is the desired
outcome** here -- the engine already ships every morning in production, so a null means production
has been shipping noise. Reported without hedging, whichever way it lands. RA/RM apply Bonferroni
correction across `n_trials=39` (14 episode-table + 25 mechanical-label sector/dimension pairs).
""")
)

cells.append(
    code("""\
phase4 = load("phase_4_14_results.json")
for gate_id, g in phase4.items():
    status = "FIRES" if g["fires"] else "null"
    print(f"{gate_id:3s} [{status:5s}] {g['claim']}")
""")
)

cells.append(
    md("""\
## Phase 5 — Verdict and handoff

**NL and RC fire**: the port is faithful (bit-identical to the source engine on synthetic fixtures,
config hashes match, `no_lookahead_check` passes on real data at every truncation for all 27
symbols) and structurally sound (no lookahead anywhere in the pipeline). **RS fires**: excluding the
two Phase-2-disqualified dimensions, nothing scored is degenerate (single-label-stuck or flip-noise).

**RA, RM, and RL do not fire.** No sector/dimension pair beats its baselines at a Bonferroni-corrected
significance level, in either ground-truth source. Median label lag at episode onset was 27 trading
days, worse than the 21-day pre-registered threshold. Read together with Phase 3's numbers, the
honest per-dimension picture is:

- **`term_structure` and `carry` (oil products, natgas)**: highest raw accuracy (85-93%), but the
  mechanical check that produced the higher numbers is close to tautological (the indicator *is* a
  transform of the same spread the check compares against). Not independent confirmation.
- **`yield_curve` (Macro)**: similarly high but similarly tautological (built from `T10Y2Y`, checked
  against `T10Y2Y`'s own sign).
- **`trend` (every basket)**: weak-to-null against forward 63-day returns (0.39-0.60 balanced
  accuracy) -- consistent with this programme's prior findings that trend-following lacks
  demonstrated edge here.
- **`risk` (Macro)**: 35.3% accuracy against four named crisis episodes' `risk_off` label -- worse
  than chance for what is nominally a three-state classifier. The engine largely failed to identify
  three of four historical crises as risk-off.
- **`volatility` (every basket) during COVID**: mostly failed to reach "extreme" during the
  2020-02-01 to 2020-04-01 window (several sectors scored 0.0-0.29 balanced accuracy against the
  episode table), though the broader mechanical vol-tercile check (much larger sample) shows more
  reasonable 0.54-0.66 balanced accuracy outside that one acute episode.
- **`Macro / credit` and `Commodities / trend`**: disqualified in Phase 2 (a real 2023-only FRED
  data gap, and near-total single-label occupancy respectively) -- not scored at all, not
  trustworthy by construction.

**No dimension here is validated strongly enough, independent of near-tautological mechanical
checks, for a follow-up notebook to condition a strategy on it without further work.** The two
highest-accuracy dimensions (`term_structure`/`carry`, `yield_curve`) would need an independent
(non-tautological) mechanical check before that changes. `trend` and `risk` look actively unreliable
during named crisis/trend episodes on this data. This notebook authorizes no trading and spends no
holdout -- both remain untouched (crypto 2025-07-01, futures 2025-01-01 -> 2026-07-28) for whatever
comes next.
""")
)

with open("src/research/014_market_regime_engine_and_accuracy.ipynb", "w") as f:
    json.dump(
        {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.12"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        },
        f,
        indent=1,
    )
print(
    f"written src/research/014_market_regime_engine_and_accuracy.ipynb ({len(cells)} cells)"
)
