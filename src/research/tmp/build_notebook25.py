"""Notebook 025 builder -- a narrative over the phase JSONs and chart
modules. Loads results and renders them; does not re-run any pricing.

Usage (from repo root): uv run python src/research/tmp/build_notebook25.py
"""

from __future__ import annotations

import json as _json
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TMP = REPO_ROOT / "src" / "research" / "tmp"
OUT_PATH = (
    REPO_ROOT / "src" / "research" / "025_pricing_bespoke_commodity_derivatives.ipynb"
)

sys.path.insert(0, str(TMP))
import charts_25_a as ca
import charts_25_b as cb
import charts_25_c as cc
import charts_25_d as cd
import charts_25_e as ce

phase5 = _json.loads((TMP / "phase_5_25_cases_abcd.json").read_text())
phase6 = _json.loads((TMP / "phase_6_25_cases_efg.json").read_text())


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


def explain(module, fig_id: str) -> dict:
    entry = module.CAPTIONS[fig_id]
    return md(
        f"**What this shows:** {entry['what']}\n\n**Does it match intuition?** {entry['intuition']}"
    )


cells: list[dict] = []

# --------------------------------------------------------------------------- #
# Title cell -- the four honesty caveats, up front and unmissable
# --------------------------------------------------------------------------- #

cells.append(
    md("""\
# Notebook 025 -- Pricing bespoke commodity derivatives, and a lesson

**The question this notebook answers:** if someone handed you a bespoke commodity
derivative tomorrow and asked "what is this worth, and how should our client hedge?",
what would you actually do?

It is a **teaching notebook**, not a "model A beats model B" study, structured in five
parts: **A** the primer (what each instrument *is*), **B** the inputs (where the
numbers come from), **C** the methods (how to actually price each instrument, eight
ways, cross-checked against each other), **D** seven worked hedging case studies, and
**E** honest limits.

### Four things to hold in mind on every page

1. **There are no option prices in this repo. None.** Every premium shown below is a
   *model* price computed from *realised* volatility, never a market quote.
2. **Realised vol is a biased proxy for implied vol.** Options generally trade *above*
   realised volatility (the variance risk premium), so every premium here is probably
   an **underestimate** of what a dealer would actually charge.
3. **No bid-offer, credit, funding, or margin cost is modelled anywhere here.** The gap
   between a model price and a client quote is illustrated once, in Part E, as a
   labelled waterfall -- never elsewhere.
4. **No Sharpe ratio. No returns backtest. No trading rule.** Where a case study shows
   "what would have happened," it is a *realised payoff computation* -- the payoff of a
   defined structure against recorded settlement prices -- never a strategy backtest.

None of the pricing mathematics is new: Black-76 is 1976, Reiner-Rubinstein 1991,
Longstaff-Schwartz 2001. The contribution is the *teaching*: one consistent harness,
real commodity data, cross-validated pricers, and worked client problems.

Full spec: `NEXT_PROMPT.md`. Full narrative and numbers:
`src/results/025_pricing_bespoke_commodity_derivatives.md`.
""")
)

cells.append(
    code("""\
import json

TMP = "tmp"


def load(name):
    with open(f"{TMP}/{name}") as f:
        return json.load(f)


prereg = load("phase_0_25_preregistration.json")
for line in prereg["honesty_list"]:
    print("-", line)
""")
)

cells.append(
    md(
        "## Phase 0 -- pre-registration\n\n"
        "A pedagogical study: no holdout spent, no trading gate. The Phase 2 "
        "cross-validation tolerances and the seven case studies' candidate structures "
        "were fixed here, before any price was computed."
    )
)
cells.append(code('prereg["case_studies_fixed_in_advance"]'))

cells.append(
    md(
        "## Phase 1 -- market inputs\n\n"
        "The forward curve (Nelson-Siegel interpolation, never Gabillon -- 023's "
        "gate), and the realised vol term structure, checked against 024's published "
        "Samuelson slopes before anything downstream runs."
    )
)
cells.append(
    code("""\
phase1 = load("phase_1_25_inputs.json")
for p, d in phase1["products"].items():
    fit = d["samuelson_fit_full_sample"]
    print(f"{p}: slope={fit['slope']:.3f}  within widened band of 024 = {d['within_widened_band_of_024']}")
""")
)

cells.append(
    code("""\
import sys

sys.path.insert(0, "tmp")
import matplotlib.pyplot as plt
import viz25 as vz

plt.rcParams["figure.facecolor"] = vz.SURFACE
plt.rcParams["savefig.facecolor"] = vz.SURFACE

# This notebook executes with cwd = src/research/ (its own directory), but
# lib24/lib25's data-directory constants are repo-root-relative strings --
# point them at the repo root explicitly so every chart module's direct
# lib24.load_panel / lib25.fx_series / lib25.discount_curve calls resolve.
import pathlib

import lib24
import lib25

_repo_root = pathlib.Path.cwd().parent.parent
lib24.DATA_DIR = str(_repo_root / "src/research/data/market/databento")
lib25.DATA_DIR = str(_repo_root / "src/research/data/market/databento")
lib25.FRED_DIR = str(_repo_root / "src/research/data/market/fred")
lib25.FX_DIR = str(_repo_root / "src/research/data/market/yfinance/daily")

phase2 = load("phase_2_25_crossval.json")
phase3 = load("phase_3_25_primer.json")
phase4 = load("phase_4_25_methods.json")
phase5 = load("phase_5_25_cases_abcd.json")
phase6 = load("phase_6_25_cases_efg.json")
""")
)

# --------------------------------------------------------------------------- #
# Part A -- the primer (11 figures)
# --------------------------------------------------------------------------- #

cells.append(
    md(
        "## Part A -- the primer\n\n"
        "What each instrument *is*, taught through pictures: payoff diagrams, real "
        "price paths with barriers drawn on them, premium-vs-strike curves."
    )
)
for i in range(1, 12):
    fig_id = f"a{i}"
    cells.append(md(f"### A{i}"))
    cells.append(
        code(f"""\
import charts_25_a as ca

fig, ax = ca.fig_{fig_id}(phase3)
plt.show()
""")
    )
    cells.append(explain(ca, fig_id))

# --------------------------------------------------------------------------- #
# Part B -- the inputs (5 figures)
# --------------------------------------------------------------------------- #

cells.append(
    md(
        "## Part B -- the inputs\n\n"
        "Where the numbers come from: the forward curve, the volatility term "
        "structure, rates, FX. Grounded in this repo's own 023/024 results."
    )
)
for i in range(1, 6):
    fig_id = f"b{i}"
    cells.append(md(f"### B{i}"))
    cells.append(
        code(f"""\
import charts_25_b as cb

fig, ax = cb.fig_{fig_id}()
plt.show()
""")
    )
    cells.append(explain(cb, fig_id))

# --------------------------------------------------------------------------- #
# Part C -- the methods (8 figures)
# --------------------------------------------------------------------------- #

cells.append(
    md(
        "## Part C -- the methods\n\n"
        "*How you actually price it* -- eight methods, from closed form to Monte "
        "Carlo to Markov functional, each cross-checked against the others. "
        "**This is the only pass/fail phase in the notebook** -- the proof the "
        "pricers are sound."
    )
)
cells.append(
    code("""\
for c in phase2["matrix"]:
    status = "PASS" if c["passed"] else "FAIL"
    print(f"{status}  {c['name']}  ({c['tolerance']})")
print()
print("ALL PASSED:", phase2["all_passed"])
""")
)
for i in range(1, 9):
    fig_id = f"c{i}"
    cells.append(md(f"### C{i}"))
    cells.append(
        code(f"""\
import charts_25_c as cc

fig, ax = cc.fig_{fig_id}(None)
plt.show()
""")
    )
    cells.append(explain(cc, fig_id))

# --------------------------------------------------------------------------- #
# Part D -- the case studies (9 figures)
# --------------------------------------------------------------------------- #

cells.append(
    md(
        "## Part D -- the case studies\n\n"
        "Seven real hedging problems, each worked end to end through the same "
        "five-step template: **1.** the exposure, **2.** candidate structures, "
        "**3.** the prices (with the model-price caveat), **4.** what actually "
        "happened over history (a realised-payoff distribution, not a backtest), "
        "**5.** the trade-off."
    )
)

_case_letters = {
    "A_refiner": ("A", "the refiner"),
    "B_farmer": ("B", "the wheat farmer"),
    "C_airline": ("C", "the airline / diesel buyer"),
    "D_gas_utility": ("D", "the gas utility"),
}
for key, (letter, title) in _case_letters.items():
    case = phase5[key]
    lines = [f"### Case {letter} -- {title}\n", f"**Exposure:** {case['exposure']}\n"]
    for s in case["structures"]:
        name = s.get("name", "structure")
        prem = s.get("premium")
        prem_str = f"{prem:.3f}" if isinstance(prem, int | float) else "n/a"
        lines.append(f"- **{name}**: premium = {prem_str}")
    cells.append(md("\n".join(lines)))

for key in ("E_negative_prices", "F_european_buyer", "G_structured_note"):
    case = phase6[key]
    letter = {
        "E_negative_prices": "E",
        "F_european_buyer": "F",
        "G_structured_note": "G",
    }[key]
    cells.append(
        md(
            f"### Case {letter}\n\nSee the figures below and the write-up for the full narrative."
        )
    )

for i in range(1, 10):
    fig_id = f"d{i}"
    cells.append(md(f"#### D{i}"))
    cells.append(
        code(f"""\
import charts_25_d as cd

fig, ax = cd.fig_{fig_id}(None)
plt.show()
""")
    )
    cells.append(explain(cd, fig_id))

# --------------------------------------------------------------------------- #
# Part E -- limits (2 figures)
# --------------------------------------------------------------------------- #

cells.append(
    md(
        "## Part E -- what a real desk adds that this notebook does not\n\n"
        "Honest limits: the variance risk premium this notebook cannot see, and "
        "everything between a model price and a client quote. Also explicitly out "
        "of scope: swing/optionality on delivery volume (needs a multi-exercise "
        "stochastic-control solver and would double this notebook)."
    )
)
for i in range(1, 3):
    fig_id = f"e{i}"
    cells.append(md(f"### E{i}"))
    cells.append(
        code(f"""\
import charts_25_e as ce

fig, ax = ce.fig_{fig_id}()
plt.show()
""")
    )
    cells.append(explain(ce, fig_id))

# --------------------------------------------------------------------------- #

cells.append(
    md(
        "## Bottom line\n\n"
        "See `src/results/025_pricing_bespoke_commodity_derivatives.md` for the full "
        "write-up, the method-vs-method agreement table, and what to test next."
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
