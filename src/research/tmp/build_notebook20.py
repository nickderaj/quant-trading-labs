"""Notebook 020 builder (NEXT_PROMPT.md sec 10). A narrative over the phase
JSONs -- loads results and renders them, does not re-run any backtest.

Usage (from repo root): uv run python src/research/tmp/build_notebook20.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TMP = REPO_ROOT / "src" / "research" / "tmp"
OUT_PATH = REPO_ROOT / "src" / "research" / "020_basis_refinement_and_cross_venue.ipynb"


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
# Notebook 020 — Refined Basis Construction + Cross-Venue Funding Dispersion

018 found a real, significant funding-carry mechanism (FA-1) with a genuinely delta-neutral hedge
(FA-4), but failed the tradeability bar (FA-2's bootstrap-CI and DSR legs) and the "does timing add
value" bar (FA-3) — and 018's own concentration diagnostic traced the DSR's harsh verdict to a real
construction flaw: an equal-weight book with no diversification floor, occasionally (5.4% of bars)
concentrated in a single symbol during exactly the bars a slow liquidity screen cannot yet see a real
perp-market liquidity collapse. 017/019 between them closed off "fix the DSR estimator" as a route to
FA-2 — the estimator defect is real but no repair is simultaneously calibrated and powerful, and 018's
own sample moments (skew −11.5, kurtosis 816.9) cap any dispersion-based repair below 0.95 anyway. The
DSR leg was never contingent on the estimator; it was contingent on the return distribution.

This notebook tests two independent, pre-registered fixes to that construction, one per mechanism:

- **Mechanism A** — the same Binance basis trade, refined with (i) a diversification floor (stand
  down entirely below `N_MIN=3` symbols) and (ii) a slower, lower-turnover carry (half-life 42
  instead of 21, thresholds re-derived from a 30-day rather than 15-day target hold).
- **Mechanism B** — a structurally different trade: the *spread* between Binance's and Bybit's
  funding rates on the same underlying, short the expensive-funding venue's perp, long the cheap
  one's, no spot leg needed.

Full narrative and numbers: `src/results/020_basis_refinement_and_cross_venue.md`.
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


prereg = load("phase_0_20_preregistration.json")
mech = load("phase_3_20_mechanism.json")
pred = load("phase_3_20_prediction.json")
results = load("phase_4_20_results.json")
ablations = load("phase_5_20_ablations.json")
holdout = load("phase_6_20_holdout.json")

print("n_trials:", prereg["n_trials_itemisation"]["total"])
""")
)

cells.append(
    md("""\
## Pre-registration (Phase 0)

Every constant, gate, and the 32-trial `n_trials` budget were frozen in `phase_0_20_preregistration.json`
before a single byte of Bybit data was fetched. `N_MIN=3` was derived from 018's own concentration
diagnostic (the smallest integer eliminating both the 1- and 2-symbol tails); the slow-carry
constants scale 018's half-life and target hold by exactly 2x, preserving 018's own ratio.
""")
)

cells.append(
    code("""\
print(json.dumps(prereg["constants"]["mechanism_a_new"], indent=2))
print(json.dumps(prereg["constants"]["mechanism_b_new"], indent=2))
""")
)

cells.append(
    md("""\
## Phase 3 — Mechanism probes (no cost model, no Sharpe)

**Mechanism A**: adding the diversification floor alone (A1) collapses the single-symbol-bar
fraction from 5.4% to exactly 0.0%, and gross skew/kurtosis go from −11.65/849 (A0, reproducing
018) to +0.53/18.7. Combining the floor with the slow carry (A3) improves moments further still
(+0.34/16.1). H-A is confirmed at the moments level.

**Mechanism B**: the raw, undirected pooled funding-spread statistic (XD-1, a fixed short-Binance/
long-Bybit direction across the whole panel) is significantly *negative* — Bybit's funding runs
higher than Binance's on average, pooled. But the strategy's own direction-following return (taking
whichever side the sign of the spread favours, symbol by symbol, bar by bar) is significantly
*positive* (mean 7.15e-05/period, Newey-West t=10.8). These are different, both correct, statistics;
the distinction is disclosed explicitly below.
""")
)

cells.append(
    code("""\
for k in ("A0", "A1", "A2", "A3"):
    d = mech["mechanism_a"][k]["symbols_held_distribution"]
    m = mech["mechanism_a"][k]["gross_moments"]
    print(f"{k}: median_held={d['median']:.0f} frac_1sym={d['frac_1_symbol']:.4f} "
          f"skew={m['skew']:.2f} kurt={m['kurtosis_non_excess']:.2f}")

print()
print("XD-1 (undirected):", mech["mechanism_b"]["gate_xd1"])
print("B0 gross (direction-following):", mech["mechanism_b"]["b0_gross_pooled_mean"],
      mech["mechanism_b"]["b0_gross_newey_west_t"])
""")
)

cells.append(
    md("""\
## Phase 3b — The pre-registered prediction

Before Phase 4 ran: if A3 merely matched 018's net Sharpe (0.5766) but carried A3's own predicted
(much better) moments, the counterfactual DSR was only **0.154** — nowhere near 0.95. The prediction
was explicit: better moments alone would not be enough; A3 would need to substantially beat 018's
Sharpe, not just carry better moments, for RC-2's DSR leg to clear. The same held for B0 against
`B_single`'s Sharpe (0.854): counterfactual DSR 0.346.
""")
)

cells.append(
    code("""\
print(json.dumps(pred["counterfactual_dsr_A3"], indent=2))
print(json.dumps(pred["rc2_dsr_leg_prediction"], indent=2))
print(json.dumps(pred.get("counterfactual_dsr_B0"), indent=2))
""")
)

cells.append(
    md("""\
## Phase 4 — The books

A3's actual net Sharpe came in at **3.89** — not because the mean return changed much (A0's gross
total return 0.434 vs A3's 0.435, essentially identical) but because eliminating the catastrophic
single-symbol tail bars cut the standard deviation 4x (0.0014 → 0.00035). That absolute jump is what
the prediction said would be needed, and it happened — RC-2 fires (DSR=0.99999973). RC-3, testing
the *paired* difference against 018's own reproduction, does not: the diff series still inherits most
of A0's own volatility bar-by-bar, and its bootstrap CI still includes zero even though the point
estimate favours the refined book. RC-4 (neutrality) fires cleanly.

Mechanism B's headline (B0) comes in weak: net Sharpe 0.354, below the 0.5 bar, DSR 0.082. It also
loses head-to-head against `B_single` (the plain single-venue trade restricted to the same 93-symbol,
Bybit-intersected universe), which itself scores a healthy 0.854 — so XD-3 (spread beats level) fails
in the *wrong* direction: the level, not the spread, wins on this universe. XD-4 (neutrality) still
fires. **H-B is not supported.**
""")
)

cells.append(
    code("""\
for key in ("A0_0", "A1_0", "A2_0", "A3_0", "A_alwayson_0", "B0_0", "B1_0", "B_single_0", "B_alwayson_0"):
    m = results["cells"][key]["metrics"]
    print(f"{key:14s} net Sharpe={m['sharpe_net']:7.4f}  net MDD={m['max_drawdown_net']:8.4f}  "
          f"turnover/yr={m['annualized_turnover']:6.1f}")

print()
for g in ("gate_RC2", "gate_RC3", "gate_RC4", "gate_FUND_A", "gate_XD2", "gate_XD3", "gate_XD4", "gate_FUND_B"):
    print(g, results[g]["fires"])
print("holdout_access:", results["holdout_access"])
""")
)

cells.append(
    md("""\
## Phase 5 — Ablations

A3 is robust: `N_MIN` in {2,5} barely moves the Sharpe (3.88, 3.92 vs 3.89 headline), excluding
LUNA/FTT barely moves it either (3.99) — the floor already prevents dependence on any one name. Cost
sensitivity never crosses the 0.5 bar within the tested 0–51bp range (still 0.81 at 51bp, 1.5x the
real cost) — no computable break-even, i.e. a break-even beyond the tested range. Removing hysteresis
costs about half the Sharpe (1.75 vs 3.89).

B0 is fragile in every direction it's cut: no-hysteresis is deeply negative (−3.68, worse than A's
equivalent ablation — sign-flipping without a band is much more expensive than simple entry/exit),
excluding its own top-2 contributing symbols (TRBUSDT, DOTUSDT) flips it negative (−0.28), and its
interpolated break-even (24.95bp) sits almost exactly at its actual 25bp cost — explaining the weak
headline number directly. The one-venue-leg-only neutrality control is negative and unlike B0's own
return (−0.59 vs 0.35), evidence against a disguised single-leg bet, but that's the only ablation that
goes B0's way.
""")
)

cells.append(
    code("""\
for k, v in ablations["cells"].items():
    print(f"{k:24s} net Sharpe={v['metrics']['sharpe_net']:7.4f}")
print()
print("A3 break-even:", ablations["a3_breakeven"])
print("B0 break-even:", ablations["b0_breakeven"])
""")
)

cells.append(
    md("""\
## Phase 6 — Holdout

Neither mechanism unlocked (Mechanism A needs RC-2 AND RC-3; RC-3 did not fire. Mechanism B needs
XD-2 AND XD-3; neither fired). `run_phase_6_20_holdout.py` was invoked once anyway, per sec 9's own
"demonstrate the fence" rule, and refused correctly (exit 1) without constructing any path into a
holdout directory.
""")
)

cells.append(
    code("""\
print(json.dumps(holdout, indent=2))
""")
)

cells.append(
    md("""\
## Bottom line

**Mechanism A: H-A confirmed at the construction level, RC-2 and FUND-A fire, but the holdout stays
locked because RC-3 does not.** The diversification floor does exactly what the concentration
diagnostic predicted — it eliminates catastrophic single-symbol tail risk and, this time, pushes the
absolute risk-adjusted return far enough (net Sharpe 3.89) to clear the DSR bar cleanly. But 020's own
pre-registered bar for "does the refinement add value" is a *paired* comparison against 018's own
frozen construction, and that comparison — noisier than either book's own Sharpe suggests, because it
inherits much of 018's own volatility bar-by-bar — cannot yet rule out zero. **Mechanism B: H-B is not
supported.** The cross-venue spread trade is real (XD-1's direction-following statistic is
significant) but weak and fragile, and loses outright to the plain single-venue trade on the same
restricted universe.

Full detail, every disclosed asymmetry, and what to test next: `src/results/020_basis_refinement_and_cross_venue.md`.
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
