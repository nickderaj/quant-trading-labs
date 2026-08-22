"""Notebook 021 builder (NEXT_PROMPT.md sec 9, deliverable 1). A narrative
over the phase JSONs -- loads results and renders them, does not re-run any
backtest.

Usage (from repo root): uv run python src/research/tmp/build_notebook21.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TMP = REPO_ROOT / "src" / "research" / "tmp"
OUT_PATH = REPO_ROOT / "src" / "research" / "021_rc3_power_and_data_quality.ipynb"


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


cells = []

cells.append(
    md("""\
# Notebook 021 — Is RC-3 Blocked by Data Artifacts, or by Statistical Power?

020 found a refined basis book (`A3`) that unambiguously beats 018's own construction on every
*absolute* measure (net Sharpe 3.89 vs. 0.58, DSR 0.9999997) and clears its own tradeability gate
(RC-2) cleanly — but the *paired* comparison against 018's own frozen reproduction (RC-3) did not
clear its bootstrap CI: `[-2.20e-05, +5.62e-05]`, point estimate favouring the refined book but the
CI still straddling zero.

Two candidate explanations, with different fixes: **data artifacts** in the baseline's own return
series (fixable by a disclosed, independently-defined exclusion applied to the comparison only) or
**statistical power** (fixable only by more paired history). 021 asks that one question, with one
pre-registered answer per branch.

Full narrative and numbers: `src/results/021_rc3_power_and_data_quality.md`.
""")
)

cells.append(
    code("""\
import json
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "tmp")

TMP = "tmp"


def load(name):
    with open(f"{TMP}/{name}") as f:
        return json.load(f)


prereg = load("phase_0_21_preregistration.json")
tripwire = load("phase_0_21_tripwire.json")
catalogue = load("phase_1_21_catalogue.json")
results = load("phase_3_21_results.json")

print(
    "n_trials:",
    prereg["n_trials_itemisation"]["total"],
    "(no DSR consumer -- see prereg)",
)
""")
)

cells.append(
    md("""\
## Pre-registration (Phase 0) + reproduction tripwire

The exclusion rule has no free parameter: a panel bar `(symbol, T)` is flagged iff, on the perp leg,
`open == high == low == close AND volume == 0`. No halo, no minimum run length, no widening — the
bare signature covers both of 018's own documented events. Excluded book-return timestamps are the
union over flagged `(symbol, T)` of `{T-1, T}`, restricted to bars where **`A0`** (never `A3`)
carried non-zero weight in that symbol.

Before anything else ran, the notebook reproduced 020's own RC-3 diff CI from the two stored returns
parquets, to confirm nothing upstream had moved.
""")
)

cells.append(
    code("""\
print(json.dumps(prereg["constants"]["exclusion_rule"], indent=2))
print()
print(
    "reproduction tripwire matches 020's stored RC-3 CI:",
    tripwire["matches_020_stored_value"],
)
print("tripwire CI:", tripwire["bootstrap_ci_95"])
""")
)

cells.append(
    md("""\
## Phase 1 — The mechanical catalogue

Scanning all 128 full-range cached perp files with the bare signature took well under a second and
flagged a large number of symbol-bars — but most of that is post-delisting dead-feed tail, not the
events of interest. `run_phase_1_21_catalogue.py` names no symbol anywhere in its own code (verified
by grep, not just asserted); PW-1's coverage check runs in Phase 3, where it is free to name both
symbols explicitly.
""")
)

cells.append(
    code("""\
print("files scanned:", catalogue["n_files_scanned"])
print("total flagged symbol-bars:", catalogue["total_flagged_symbol_bars"])
print("symbols flagged:", catalogue["n_symbols_flagged"])
print()
print(
    "gate PW-1 (detector soundness):", json.dumps(results["gate_PW1"], indent=2)
)
""")
)

cells.append(
    md("""\
## Phase 3 — Diff CIs with/without exclusion, MDE, placebo

Both CIs, side by side, both bar counts (symbol-bars and book-return bars), and the placebo
percentile that decides whether the exclusion's effect on the CI is specific to the flagged bars or
just generic outlier-trimming from removing *any* same-size subset.
""")
)

cells.append(
    code("""\
print("excluded-bar counts:", json.dumps(results["exclusion"], indent=2))
print()
print(
    "diff CI without exclusion:",
    json.dumps(results["diff_ci_without_exclusion"], indent=2),
)
print()
print(
    "diff CI with exclusion:",
    json.dumps(results["diff_ci_with_exclusion"], indent=2),
)
print()
print("gate PW-2 (data-quality-corrected RC-3):", results["gate_PW2"])
""")
)

cells.append(
    md("""\
## Power: MDE and years required

Computed both with and without the exclusion, closed-form from each CI's implied standard error.
""")
)

cells.append(
    code("""\
print(json.dumps(results["power"], indent=2))
print()
print(
    "gate PW-3 (adequately powered):", json.dumps(results["gate_PW3"], indent=2)
)
""")
)

cells.append(
    md("""\
## PW-4 — The placebo control

Draws 200 random same-size exclusions from the 3,840-bar series (seed 0) and compares means only.
If the flagged exclusion's mean diff doesn't clear the 95th percentile of that placebo distribution,
the CI move under exclusion is not attributable to the flagged bars specifically.
""")
)

cells.append(
    code("""\
print(json.dumps(results["placebo"], indent=2))
print()
print(
    "gate PW-4 (surgical, not a reshape):",
    json.dumps(results["gate_PW4"], indent=2),
)
print()
print(
    "a3 immunity diagnostic (not used for exclusion):",
    json.dumps(results["a3_immunity_diagnostic"], indent=2),
)
""")
)

cells.append(
    md("""\
## Bottom line

021's own branch verdict:
""")
)

cells.append(
    code("""\
print(results["branch"])
""")
)

cells.append(
    md("""\
PW-1 fires cleanly — the mechanical detector, blind to both symbol names, independently rediscovers
018's own two documented events. PW-2 nominally fires too: excluding the 22 book-return bars where
`A0` held a frozen-feed symbol (0.57% of the series) moves the diff CI to `[1.02e-05, 3.10e-05]`,
clearing zero. But PW-4's placebo control catches it: the flagged-exclusion mean sits just inside the
95th percentile of 200 same-size random exclusions, meaning the CI move is generic outlier-trimming
from dropping *any* 22 bars, not something specific to the flagged bars. **Branch (b): statistical
power, not data quality, is the binding constraint.** Without exclusion, detecting an effect this
size at 80% power would need roughly 27.6 years of paired history against the 3.50 years the paired
series actually has.

No holdout access is granted under any outcome (021's own pre-registered policy, sec 6). Full detail:
`src/results/021_rc3_power_and_data_quality.md`.
""")
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.13",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print(f"written {OUT_PATH}")
