"""Notebook 024 builder -- a narrative over the phase JSONs. Loads results and
renders them; does not re-run any fit.

Usage (from repo root): uv run python src/research/tmp/build_notebook24.py
"""

from __future__ import annotations

import json as _json
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TMP = REPO_ROOT / "src" / "research" / "tmp"
OUT_PATH = REPO_ROOT / "src" / "research" / "024_samuelson_effect_across_futures.ipynb"


def cid() -> str:
    return uuid.uuid4().hex[:8]


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cid(),
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cid(),
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


cells: list[dict] = []

cells.append(
    md("""\
# Notebook 024 — The Samuelson effect across 16 futures markets

The Samuelson (1965) hypothesis says futures return volatility rises as a contract
approaches expiry: information about near-term supply and demand decays by the time a
deferred contract expires, so the front of the curve reacts to news the back barely
feels. This notebook is a **descriptive, visual atlas** of that effect across every
futures market in this repo (16 products, 2010-2026) -- it fits no trading rule,
computes no Sharpe ratio, and spends no holdout.

Full spec: `NEXT_PROMPT.md`. Full narrative and numbers:
`src/results/024_samuelson_effect_across_futures.md`.
""")
)

cells.append(
    code("""\
import json

TMP = "tmp"


def load(name):
    with open(f"{TMP}/{name}") as f:
        return json.load(f)


prereg = load("phase_0_24_preregistration.json")
prereg["statement"]
""")
)

cells.append(
    md(
        "## Phase 0 — Pre-registration\n\n"
        "This is a descriptive study: no holdout spent, no trading gate. The "
        "falsification condition, the estimator hierarchy, and the hygiene rules "
        "were fixed before any chart was looked at."
    )
)
cells.append(code('prereg["falsification_condition"]'))
cells.append(code('prereg["predicted_ordering_confirmed_scoping_expectation"]'))

cells.append(
    md(
        "## Phase 1 — Panel construction and data-quality census\n\n"
        "850,833 usable rows across 16 products after dropping `close <= 0` "
        "(this is the single most important section of the underlying "
        "prompt -- the far-dated end of every panel contains junk prints that "
        "destroy a plain standard deviation; see Phase 2 for the MAD-vs-std "
        "comparison)."
    )
)
cells.append(
    code("""\
phase1 = load("phase_1_24_panel_report.json")
for p in ["CL", "NG", "ES", "GC"]:
    d = phase1["products"][p]
    print(f"{p}: raw_rows={d['raw_rows']}, close<=0={d['close_le_0_count']}, "
          f"usable(gap1,vol0)={d['usable_returns_variants']['gap1_vol0']['n']}, "
          f"junk(|r|>1)={d['junk_prints_gt_1_log_return']['count']}")
""")
)
cells.append(
    md(
        "These four match the smoke-test numbers in `NEXT_PROMPT.md` section 1 exactly "
        "(CL usable=81,895; NG=93,612; ES=6,797; GC=38,484) -- the pipeline is verified "
        "correct before any chart is built."
    )
)

cells.append(
    code("""\
import sys

sys.path.insert(0, "tmp")
import matplotlib.pyplot as plt
import viz24 as vz

plt.rcParams["figure.facecolor"] = vz.SURFACE
plt.rcParams["savefig.facecolor"] = vz.SURFACE

phase2 = load("phase_2_24_profiles.json")
phase3 = load("phase_3_24_slopes.json")
phase4 = load("phase_4_24_timeseries.json")
phase5 = load("phase_5_24_seasonality.json")
phase6 = load("phase_6_24_smile_skew.json")
""")
)

cells.append(
    md(
        "## Section A — the effect exists\n\n"
        "Seven figures over Phase 2's vol-vs-maturity profiles (16 products x 3 "
        "estimators x 3 liquidity variants x 2 gap variants) and Phase 3's Samuelson "
        "slopes."
    )
)

section_a = [
    (
        "A1",
        "CL, the hero chart: robust vol vs. days-to-expiry with a fitted power law.",
    ),
    (
        "A2",
        (
            "CL, three estimators overlaid -- the std line detonating at the long end "
            "*is* the data-quality finding from section 2 of the prompt."
        ),
    ),
    (
        "A3",
        "All 16 products, small multiples, shared log-x, panel borders tinted by sector.",
    ),
    (
        "A4",
        (
            "All 16 overlaid, each normalised to its own 365-day vol -- the metals "
            "crossing above 1.0 at the long end is the inversion, visible."
        ),
    ),
    (
        "A5",
        (
            "Energy / metals / ags sector panels, with ES as a dashed grey reference "
            "in every one."
        ),
    ),
    (
        "A7",
        "Heatmap: product x maturity bucket, normalised to each product's 365-day vol.",
    ),
]
for fig_id, caption in section_a:
    cells.append(md(f"**{fig_id}.** {caption}"))
    cells.append(
        code(f"""\
import charts_24_a as ca

fig, ax = ca.fig_{fig_id}({{"profiles": phase2, "slopes": phase3}})
plt.show()
""")
    )

cells.append(
    md(
        "## Phase 3 — Samuelson slopes and the metals inversion\n\n"
        "Ranked slopes (mad estimator, `max_gap=1`, `min_volume=0`), with "
        "contract-level bootstrap CIs."
    )
)
cells.append(
    code("""\
for r in phase3["ranking_primary"]:
    print(f"{r['product']:4s} slope={r['slope']:+.4f}  ci=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]")
""")
)
cells.append(
    md(
        "**Finding that deviates from the scoping expectation:** `ES` was expected to "
        "be the single weakest (most positive) slope of all 16 products -- it is not. "
        "`PL` is slightly more positive than `ES`, and `PA`/`SI` sit close behind. `ES` "
        "does land firmly inside the near-zero/positive cluster, well separated from "
        "the strongly negative energy/ags group, which is the qualitative claim that "
        'matters -- but the precise "weakest of all" ranking was not exactly right, '
        "and the write-up says so."
    )
)
cells.append(md("**A6.** Ranked Samuelson slopes with bootstrap CIs, by sector."))
cells.append(
    code("""\
fig, ax = ca.fig_A6({"profiles": phase2, "slopes": phase3})
plt.show()
""")
)
cells.append(
    md(
        "**The metals-inversion investigation.** `PL` and `PA` show a positive "
        "(inverted) Samuelson slope. Does it survive tighter liquidity and a "
        "contract-count floor, or is it a handful of thin far-dated contracts?"
    )
)
cells.append(
    code("""\
for p in ["PL", "PA"]:
    inv = phase3["metals_inversion"][p]
    print(p, "verdict:", inv["verdict"])
    print("  primary slope:", round(inv["primary"]["slope"], 4),
          " vol>1000:", round(inv["vol1000"]["slope"], 4),
          " gap<=3:", round(inv["gap3"]["slope"], 4))
    print("  365-730d bucket:", inv["bucket_365_730_stats"])
phase3["metals_inversion_contract_floor"]
""")
)

cells.append(
    md(
        "## Section B — how short-dated vol moves against long-dated vol over time\n\n"
        "Eight figures over Phase 4's 63-day causal rolling vol, crisis-vs-calm "
        "windows, and the front/deferred return correlation."
    )
)
section_b = [
    (
        "B1",
        (
            "CL: rolling front vs. deferred vol, 2010-2026, crisis windows shaded -- "
            "the chart that answers the notebook's core question."
        ),
    ),
    ("B2", "CL: the front/deferred ratio as its own series."),
    ("B3", "The same two-panel treatment for NG."),
    ("B4", "Six products' front/deferred ratio on one axis."),
    (
        "B5",
        (
            "Term structure on selected calm and crisis dates (via nearest annual "
            "aggregate -- see the code comment)."
        ),
    ),
    ("B6", "Vol-of-vol by maturity bucket, sector means."),
    (
        "B7",
        (
            "Rolling 126-day correlation between front and deferred daily returns, "
            "CL and NG -- the de-correlation companion effect."
        ),
    ),
    ("B8", "CL annual small multiples, 2010-2026: is the effect stable or episodic?"),
]
for fig_id, caption in section_b:
    cells.append(md(f"**{fig_id}.** {caption}"))
    cells.append(
        code(f"""\
import charts_24_b as cb

fig, ax = cb.fig_{fig_id}({{"timeseries": phase4}})
plt.show()
""")
    )
cells.append(
    code("""\
for name, stats in phase4["products"]["CL"]["crisis_vs_calm"].items():
    print(name, {k: round(v, 3) for k, v in stats.items()})
""")
)
cells.append(
    md(
        "Front-end vol spikes materially more than deferred vol in every crisis "
        "window relative to the 2017 calm control, and the CL/NG front-deferred "
        "return correlation drops sharply in at least one window (see the printed "
        "`corr_min` above B7) -- the de-correlation companion to the Samuelson effect."
    )
)

cells.append(
    md(
        "## Section C — seasonality\n\n"
        "Four figures over Phase 5: NG's winter/summer split, the grains against "
        "the harvest calendar, calendar-month heatmaps, and GC as the flat control."
    )
)
section_c = [
    ("C1", "NG: winter-delivery vs. summer-delivery contracts."),
    ("C2", "Grains (ZC/ZS/ZW/KE) vol by delivery month."),
    ("C3", "Calendar month x maturity-bucket heatmap, NG and CL."),
    ("C4", 'GC as the flat control -- the "nothing here" chart, which is the point.'),
]
for fig_id, caption in section_c:
    cells.append(md(f"**{fig_id}.** {caption}"))
    cells.append(
        code(f"""\
import charts_24_c as cc

fig, ax = cc.fig_{fig_id}({{"seasonality": phase5}})
plt.show()
""")
    )
cells.append(
    code("""\
ng_winter = phase5["ng_seasonality"]["winter"][0]["vol"]
ng_summer = phase5["ng_seasonality"]["summer"][0]["vol"]
gc_winter = phase5["gc_control"]["winter"][0]["vol"]
gc_summer = phase5["gc_control"]["summer"][0]["vol"]
print(f"NG front-end vol: winter={ng_winter:.3f} summer={ng_summer:.3f} "
      f"ratio={ng_winter / ng_summer:.2f}")
print(f"GC front-end vol: winter={gc_winter:.3f} summer={gc_summer:.3f} "
      f"ratio={gc_winter / gc_summer:.2f}")
""")
)

cells.append(
    md(
        "## Section D — the realised smile/skew exploration\n\n"
        "**Every chart below is realised, not implied.** There are no options in "
        "this repo, so a true implied-vol smile is impossible -- these are realised "
        "analogues built from settlement prices, and are labelled that way "
        "everywhere. Five figures over Phase 6."
    )
)
section_d = [
    ("D1", "Realised skewness by maturity bucket, all 16 products, sector-coloured."),
    ("D2", "Realised excess kurtosis by maturity bucket, log-y."),
    (
        "D3",
        (
            'The realised "smile": vol vs. log-moneyness, pooled and by maturity bin '
            "(the maturity/moneyness confound control)."
        ),
    ),
    ("D4", "Vol-vs-dte split by curve state: contango vs. backwardation, CL and NG."),
    ("D5", "Front vs. deferred return densities, CL and NG, log-y."),
]
for fig_id, caption in section_d:
    cells.append(md(f"**{fig_id}.** {caption}"))
    cells.append(
        code(f"""\
import charts_24_d as cd

fig, ax = cd.fig_{fig_id}({{"smile_skew": phase6}})
plt.show()
""")
    )
cells.append(
    md(
        "**Confound check:** CL's pooled realised smile is a genuine U-shape that "
        "persists within each maturity bin -- not just the Samuelson effect in "
        "disguise. NG's pooled shape is the opposite (a hump, not a U) once "
        "maturity is not controlled for, which is itself worth noting rather than "
        "forcing into the same story as CL.\n\n"
        "**Curve-state check:** the naive expectation was that backwardated curves "
        "(signalling scarcity) would show a steeper Samuelson slope than contango "
        "curves. For CL, contango is *slightly steeper* than backwardation "
        "(-0.150 vs. -0.128) -- the opposite of the naive prediction. For NG the two "
        "are nearly identical (-0.284 vs. -0.277). The write-up reports this "
        "honestly rather than reaching for the expected answer."
    )
)

cells.append(
    md(
        "## Bottom line\n\n"
        "See `src/results/024_samuelson_effect_across_futures.md` for the full "
        "write-up."
    )
)

nb = {
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
}

OUT_PATH.write_text(_json.dumps(nb, indent=1))
print(f"wrote {OUT_PATH}")
