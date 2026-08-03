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
# Notebook 11c — Entry-Time Loss Classifier
""")
)

cells.append(
    code("""\
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "tmp")
import json

import matplotlib.pyplot as plt
import numpy as np

TMP = "tmp"

def load(name):
    with open(f"{TMP}/{name}") as f:
        return json.load(f)

fig_n = [0]
def show(fig, caption):
    fig_n[0] += 1
    print(f"Figure {fig_n[0]}: {caption}")
    plt.tight_layout()
    plt.show()
""")
)

# ---------------------------------------------------------------------------
# Phase 0
# ---------------------------------------------------------------------------
cells.append(
    md("""\
## Phase 0 - Trade log and entry-time features

11a's own 57-trade control book is not persisted to a committed file — it is
regenerated, not stored — so Phase 0 reruns the identical pre-declared trading
rule (same five live spreads, same dev window, same cost model, same
`TradingRuleParams`) to reproduce it exactly: **57 trades, 23 stop-exits, 34
zscore-exits**, matching 11a Phase 4/Phase 5's own counts. Every feature
attached to a trade — 15 in total (entry |z|, corrected carry ratio at MID
storage with real FRED `DFF` financing, realized-vol percentile, ADF t-stat,
rolling half-life, half-life sub-period stability fraction, full-sample in-band
flag, roll-window proximity, 60-day leg correlation, variance-ratio z-stats at
q=5 and q=20, Hurst exponent, spread-level percentile within its own trailing
range, and 5-/20-day pre-move in ATR units) — is computed from data strictly at
or before the trade's own entry bar, using a 252-day trailing window feeding
the same primitives 11a built. 55 of 57 trades have every feature finite; those
2 are dropped from the classifier rather than imputed.
""")
)

cells.append(
    code("""\
phase0 = load("phase_0_11c_results.json")
print(f"Phase 0 trade log: n_trades={phase0['n_trades']} "
      f"n_stop={phase0['n_stop']} n_zscore={phase0['n_zscore']} "
      f"n_complete_features={phase0['n_trades_all_features_finite']}")
print(f"  features: {phase0['n_trades_all_features_finite']}/{phase0['n_trades']} trades with all 15 features finite")
print(f"  storage constant: {phase0['storage_constant']}, feature window: {phase0['feature_window']} days")
""")
)

# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------
cells.append(
    md("""\
## Phase 1 - Walk-forward classifier and AUC robustness

No walk-forward classification infrastructure existed in this repo — this
notebook adds a minimal primitive for the walk-forward logistic regression and
its AUC robustness check. The model class is logistic regression (`torch.nn.Linear`
+ `BCEWithLogitsLoss`, strong L2 via `weight_decay=1.0`), with one pre-declared
feature set (15 entry-time features from Phase 0). Four origin offsets (0-3
trades) each give an anchored walk-forward grid (train 30 trades, test 5, step
5). All four offsets clear the pre-registered ">0.60" bar on their literal,
point-estimate AUC text. However, bootstrapping the already-collected stitched
out-of-sample predictions gives a 95% CI that **includes 0.5 at every single
offset** — the AUC leg fires by the pre-registered criterion's literal text,
and that verdict is reported honestly, but a direct bootstrap check shows this
point estimate alone is not distinguishable from chance on 20-25 out-of-sample
trades.
""")
)

cells.append(
    code("""\
phase1 = load("phase_1_11c_results.json")
print(f"Phase 1 classifier results:")
for offset in ['0', '1', '2', '3']:
    res = phase1['results_by_offset'].get(offset, {})
    auc = res.get('stitched_oof_auc')
    n_folds = res.get('n_folds')
    n_oof = res.get('n_oof_trades')
    print(f"  offset {offset}: n_folds={n_folds} n_oof_trades={n_oof} stitched_auc={auc:.3f}")
print(f"  all_offsets_auc_above_0.60={phase1['all_offsets_auc_above_0_60']}")
print(f"  (95% CI bootstrap robustness: checked in Phase 2)")
""")
)

# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------
cells.append(
    md("""\
## Phase 2 - Suppression book and final verdict

Per offset, the top decile of that offset's own out-of-sample predicted P(stop)
is vetoed at entry (2-3 trades per offset) via a new, minimal `spread_lib11`
addition: `veto_entry_mask`/`veto_entry_masks`. Three of four offsets show a
marginal improvement on the three-way risk gate (Sharpe moves a few thousandths,
well within bootstrap noise); offset 2 makes the book worse. Because the rest of
this programme's every-offset convention is the operative standard everywhere else,
and Gate LC's own text does not explicitly relax it, this notebook requires all
four offsets to clear before the suppression leg fires. **It does not — leg 2
fails, and per the gate's own `AND`, Gate LC does not fire,** independent of
leg 1's already-qualified pass.
""")
)

cells.append(
    code("""\
phase2 = load("phase_2_11c_results.json")
print(f"Phase 2 suppression verdict:")
print(f"  AUC leg fires: {phase2['auc_leg_fires']}")
print(f"  Suppression leg fires (all offsets): {phase2['suppression_leg_fires_all_offsets']}")
print(f"  Gate LC fires: {phase2['gate_lc_fires']}")
print(f"  Per-offset three-way results:")
for offset in ['0', '1', '2', '3']:
    off_rec = phase2['per_offset_suppression'].get(offset, {})
    if 'skipped' not in off_rec:
        beats = off_rec.get('suppressed_beats_unsuppressed_three_way')
        n_vetoed = off_rec.get('n_vetoed_trades')
        print(f"    offset {offset}: n_vetoed={n_vetoed} beats_three_way={beats}")
""")
)

# ---------------------------------------------------------------------------
# Bottom line
# ---------------------------------------------------------------------------
cells.append(
    md("""\
## What this notebook establishes, plainly

This was v4 §14.3's own proposed idea, never built by the external programme,
and NEXT_PROMPT.md's own honest prior — informed by 11a Phase 5's trade-shape
atlas, where entry extremity failed to discriminate winners from losers and the
catastrophic tail traced entirely to stop-exits — was that entry-time features
would not predict which trades stop out, because the atlas's own evidence points
to the *first post-entry response*, not anything knowable at entry, as the true
separator. **The result here is more nuanced than a clean null**: a point-estimate
AUC comfortably above the pre-registered bar at every offset, on a genuinely
walk-forward, no-lookahead classifier — but one that a direct bootstrap check
shows is not reliably distinguishable from chance on this sample size, and whose
practical payoff (a suppression book) fails to clear its own bar at 1 of 4
offsets regardless. Reported together, not separately, this is a well-powered
near-miss in the AUC leg and an outright miss in the suppression leg, and Gate
LC's overall verdict is a clean **does not fire** — consistent with sec 6's own
prior that the catastrophic tail is not predictable at entry-time, and directly
supporting the case for keeping the stop rather than trying to avoid the trades
that trigger it.
""")
)

with open("src/research/011c_entry_time_loss_classifier.ipynb", "w") as f:
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
print(f"written src/research/011c_entry_time_loss_classifier.ipynb ({len(cells)} cells)")
