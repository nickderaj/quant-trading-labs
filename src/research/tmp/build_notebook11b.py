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
# Notebook 11b — Spread Mechanism Gates

Seven pre-registered gates testing the external repo's proposed mechanisms
against this repo's own data and cost model. All seven return fired=False,
establishing that the discrete-trade structure, backwardation-regime flipping,
ADF screening, vol-adaptive stops, and reentry-gating do not deliver tradeable
alpha on this repo's reproducing implementation and capital assumptions. DSR trial counts
total 65 across the seven gates, cross-checked line-by-line against 11a's
Phase 6 pre-registration, every count matching exactly, none shrunk.
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
## Phase 0 - Gate TS and TS-S: Discrete-trade structure vs continuous

The pre-declared trading rule (entry/exit z-thresholds, per-spread ATR stops,
fixed-fractional sizing, suppression filters, gated reentry), run on the five
live spreads under this repo's costs, produces negative Sharpe at every
origin offset. The continuous Gate SP book on the same five spreads and same
costs runs +0.552 to +0.555 -- a large, one-sided gap. **Gate TS does not
fire.** The stop-disabled variant shows the stop is a drag but not the whole
story, confirming the packaging itself is the net negative at every layer
tested. Gate TS-S does not fire by construction.
""")
)

cells.append(
    code("""\
phase0 = load("phase_0_11b_results.json")
ts = phase0['gate_TS']
ts_s = phase0['gate_TS_S']
print(f"Gate TS: structured book Sharpe at offsets 0/7/14/21 = "
      f"{ts['structured_by_offset']['offset_0']['sharpe']:.3f} / "
      f"{ts['structured_by_offset']['offset_7']['sharpe']:.3f} / "
      f"{ts['structured_by_offset']['offset_14']['sharpe']:.3f} / "
      f"{ts['structured_by_offset']['offset_21']['sharpe']:.3f}")
print(f"         paired CI (structured - continuous): "
      f"[{ts['paired_bootstrap']['delta_ci'][0]:.3f}, {ts['paired_bootstrap']['delta_ci'][1]:.3f}]")
print(f"         DSR: {ts['deflated_sharpe_prob']:.4f}, fires: {ts['fires']}")
print()
ts_s_m = ts_s['stop_disabled_metrics_offset0']
print(f"Gate TS-S (stop-disabled): Sharpe = {ts_s_m['sharpe']:.3f}, "
      f"n trades = {ts_s_m['n_trades']}")
print(f"           paired CI (stop-disabled - continuous): "
      f"[{ts_s['paired_bootstrap_stop_disabled']['delta_ci'][0]:.3f}, "
      f"{ts_s['paired_bootstrap_stop_disabled']['delta_ci'][1]:.3f}]")
print(f"           fires: {ts_s['fires']}")
""")
)

# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------
cells.append(
    md("""\
## Phase 1 - Gate BF and BF-X: Backwardation-regime sign flip

Sign-flip ONLY the entries opened in the mild-backwardation bucket
(0 < carry_ratio < 0.5, corrected for this repo's sign convention --
verified empirically on brent_calendar: backwardation rows have c in
[+0.01, +11.13], contango rows in [-6.94, -0.01], entirely consistent with
the sign flip). Universe: Gate SP's own calendar 16-spread group, non-cherry-
picked. Three storage constants tested (LOW/MID/HIGH); MID is the pre-
declared headline. **Gate BF does not fire:** flipping the signal hurts
at every offset, and the trade count drops 13% (378 vs 436, the throughput-
collapse Gate BF was supposed to avoid). Gate BF-X (per-spread improvement,
headline storage) finds only 4 of 16 spreads individually improved, short of
its own bar, and does not fire.
""")
)

cells.append(
    code("""\
phase1 = load("phase_1_11b_results.json")
bf = phase1['gate_BF']
bf_x = phase1['gate_BF_X']
print("Gate BF results (Calendar universe, 16 spreads):")
print(f"  unconditional (no flip) offset 0: Sharpe = "
      f"{phase1['unconditional_by_offset']['offset_0']['sharpe']:.3f}, "
      f"n trades = {phase1['unconditional_by_offset']['offset_0']['n_trades']}")
for storage in ['low', 'mid', 'high']:
    by_storage = bf['by_storage'][storage]
    offset_0 = by_storage['by_offset']['offset_0']
    print(f"  {storage.upper()} storage:          Sharpe = {offset_0['sharpe']:.3f}, "
          f"n trades = {offset_0['n_trades']}")
print(f"  MID storage paired CI (flipped - unconditional): "
      f"[{bf['paired_bootstrap_headline']['delta_ci'][0]:.3f}, "
      f"{bf['paired_bootstrap_headline']['delta_ci'][1]:.3f}]")
print(f"  Gate BF fires: {bf['fires']}")
print()
print(f"Gate BF-X (per-spread wins at MID storage): {bf_x['n_spreads_improved']}/16 improved")
print(f"Gate BF-X fires: {bf_x['fires']}")
""")
)

# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------
cells.append(
    md("""\
## Phase 2 - Gate SCR: ADF screen vs no screen

Screen-inclusive (4 ADF-passing live spreads) and screen-exclusive
(same four plus kc_chicago_wheat, gc_cal_m2m3, es_calendar) books are,
to three decimal places, the same book at every offset. This is because
the three excluded spreads contribute almost no trades under the pre-
declared risk rule -- the screen's presence or absence is observationally
void for this parameterization. **Gate SCR does not KEEP the screen:**
sec 4.2's bar ("a screen that cannot beat its own absence does not survive")
is failed on a technicality -- the screen barely matters either way, which
is itself worth stating plainly: this notebook cannot currently distinguish
whether the ADF screen is good, bad, or simply inert under this specific
trading rule and universe.
""")
)

cells.append(
    code("""\
phase2 = load("phase_2_11b_results.json")
print("Gate SCR results (Screen vs no screen):")
inc_o0 = phase2['screen_inclusive_by_offset']['offset_0']
exc_o0 = phase2['screen_exclusive_by_offset']['offset_0']
print(f"  screen-inclusive Sharpe at offsets 0/7/14/21 = "
      f"{inc_o0['sharpe']:.3f} / "
      f"{phase2['screen_inclusive_by_offset']['offset_7']['sharpe']:.3f} / "
      f"{phase2['screen_inclusive_by_offset']['offset_14']['sharpe']:.3f} / "
      f"{phase2['screen_inclusive_by_offset'].get('offset_21', {}).get('sharpe', 0):.3f}")
print(f"  screen-exclusive Sharpe at offsets 0/7/14/21 = "
      f"{exc_o0['sharpe']:.3f} / "
      f"{phase2['screen_exclusive_by_offset']['offset_7']['sharpe']:.3f} / "
      f"{phase2['screen_exclusive_by_offset']['offset_14']['sharpe']:.3f} / "
      f"{phase2['screen_exclusive_by_offset'].get('offset_21', {}).get('sharpe', 0):.3f}")
print(f"  paired CI (inclusive - exclusive) at offset 0: "
      f"[{phase2['paired_bootstrap']['delta_ci'][0]:.5f}, "
      f"{phase2['paired_bootstrap']['delta_ci'][1]:.5f}]")
print(f"  Gate SCR fires: {phase2['fires']}")
""")
)

# ---------------------------------------------------------------------------
# Phase 3
# ---------------------------------------------------------------------------
cells.append(
    md("""\
## Phase 3 - Gate VA: Vol-adaptive stop

Scaling `stop_atr_mult` by 0.75×–1.25× against a rolling realized-vol
percentile improves both Sharpe (−0.111 to −0.157 across offsets, vs control's
−0.165 to −0.209) and max drawdown (−1.63% vs control's −1.88%) at every
single offset -- directionally exactly what sec 2.1's reopened, corrected-data
finding anticipated. But the improved Sharpe never crosses zero, and the
paired CI still straddles it. **Gate VA does not fire.** This is a genuinely
close, informative near-miss: the mechanism helps on both axes, just not
by enough to clear a Sharpe-positive-every-offset bar that the underlying
book was never going to clear anyway.
""")
)

cells.append(
    code("""\
phase3 = load("phase_3_11b_results.json")
ctrl_o0 = phase3['control_by_offset']['offset_0']
va_o0 = phase3['va_by_offset']['offset_0']
print("Gate VA results (Vol-adaptive stop):")
print(f"  control Sharpe at offsets 0/7/14/21 = "
      f"{ctrl_o0['sharpe']:.3f} / "
      f"{phase3['control_by_offset']['offset_7']['sharpe']:.3f} / "
      f"{phase3['control_by_offset']['offset_14']['sharpe']:.3f} / "
      f"{phase3['control_by_offset'].get('offset_21', {}).get('sharpe', 0):.3f}")
print(f"  VA Sharpe at offsets 0/7/14/21 = "
      f"{va_o0['sharpe']:.3f} / "
      f"{phase3['va_by_offset']['offset_7']['sharpe']:.3f} / "
      f"{phase3['va_by_offset']['offset_14']['sharpe']:.3f} / "
      f"{phase3['va_by_offset'].get('offset_21', {}).get('sharpe', 0):.3f}")
print(f"  control max drawdown = {phase3['control_max_drawdown']:.2%}")
print(f"  VA max drawdown = {phase3['va_max_drawdown']:.2%}")
print(f"  paired CI (VA - control): "
      f"[{phase3['paired_bootstrap']['delta_ci'][0]:.4f}, "
      f"{phase3['paired_bootstrap']['delta_ci'][1]:.4f}]")
print(f"  Gate VA fires: {phase3['fires']}")
""")
)

# ---------------------------------------------------------------------------
# Phase 4
# ---------------------------------------------------------------------------
cells.append(
    md("""\
## Phase 4 - Gate RE: Reentry-grid sweep

Sweeps the gated-reentry validity thresholds -- `half_life_max` in {30, 45, 60}
days and `adf_pmax` in {0.05, 0.10, 0.20} -- against the pre-declared baseline
(45, 0.10): a genuine 3×3×4 = 36-trial grid, counted in full per sec 9's explicit
instruction. The best-performing non-baseline cell improves on baseline at every
offset but never achieves Sharpe > 0. The resulting DSR at n_trials=36 is 0.0138,
nowhere near the 0.95 bar. **Gate RE does not fire.** The full n_trials=36 count
is reported even though unreachable -- that unreachability, at its honest denominator,
is itself the finding.
""")
)

cells.append(
    code("""\
phase4 = load("phase_4_11b_results.json")
baseline_key = phase4['baseline_key']
best_key = phase4['best_non_baseline_key']
baseline_o0 = phase4['grid'][baseline_key]['by_offset']['offset_0']
best_o0 = phase4['grid'][best_key]['by_offset']['offset_0']
print("Gate RE results (Reentry-grid sweep, 3×3×4 = 36 trials):")
print(f"  baseline (45d half_life, 0.10 adf_pmax) Sharpe at offset 0 = "
      f"{baseline_o0['sharpe']:.3f}")
best_cell_rec = phase4['grid'][best_key]
hl_val = best_cell_rec['half_life_max']
pmax_val = best_cell_rec['adf_pmax']
print(f"  best non-baseline cell: half_life_max={hl_val:.0f}d, adf_pmax={pmax_val}")
print(f"  best cell Sharpe at offset 0 = {best_o0['sharpe']:.3f}")
print(f"  best cell paired CI (best - baseline): "
      f"[{phase4['paired_bootstrap_best_vs_baseline']['delta_ci'][0]:.6f}, "
      f"{phase4['paired_bootstrap_best_vs_baseline']['delta_ci'][1]:.6f}]")
print(f"  DSR at n_trials=36: {phase4['deflated_sharpe_prob_best']:.4f}")
print(f"  Gate RE fires: {phase4['fires']}")
""")
)

# ---------------------------------------------------------------------------
# Phase 5
# ---------------------------------------------------------------------------
cells.append(
    md("""\
## Phase 5 - Sec 4.3: Standalone es_calendar / gc_cal_m2m3

Runs each of the two named cross-repo-conflict spreads alone under the pre-
declared trading rule at TWO risk_pct levels: the default (0.03) and a
drawdown-matched level tuned so the standalone book's max drawdown equals
Gate TS's control book's max drawdown. This does NOT adjudicate the external
repo's own internal contradiction for gc_cal_m2m3 (their v4 claims +0.56/+0.62
mean ATR; their atlas manifest shows -0.04 over 224 pooled trades) -- that
disagreement lives inside their own data. What this phase CAN do is report
this repo's own independently measured per-trade edge and Sharpe for both
spreads standalone, under this repo's stricter cost model. **Key finding:**
both spreads are trade-starved under the pre-declared capital assumptions
(max_single_name_pct=12% caps notional per contract, and both spreads' median
notional exceeds this cap on most bars), making the drawdown-matched comparison
not a meaningful edge-vs-edge test for either spread.
""")
)

cells.append(
    code("""\
phase5 = load("phase_5_11b_results.json")
print("Sec 4.3 Standalone diagnostics:")
print(f"Control max drawdown target (from Gate TS): {phase5['control_max_drawdown_target']:.2%}")
print()
for spread_name in ['es_calendar', 'gc_cal_m2m3']:
    rec = phase5['per_spread'][spread_name]
    cap = rec['single_name_cap_diagnostic']
    default_m = rec['default_metrics']
    matched_m = rec['drawdown_matched_metrics']
    print(f"{spread_name}:")
    print(f"  median notional per contract: ${cap['median_notional_per_contract']:,.0f} "
          f"(single-name cap: ${cap['single_name_cap_dollars']:,.0f}, "
          f"frac of bars within cap: {cap['frac_bars_notional_within_cap']:.3f})")
    print(f"  n trades at default risk_pct (0.03): {default_m['n_trades']}, Sharpe: {default_m['sharpe']}")
    print(f"  n trades at risk_pct=0.15 (5x stress): {rec['risk_pct_0.15_metrics']['n_trades']}")
    print(f"  drawdown-matched risk_pct: {rec['drawdown_matched_risk_pct']:.4f}, "
          f"n trades: {matched_m['n_trades']}, Sharpe: {matched_m['sharpe']}")
    print()
print(phase5['note'])
""")
)

# ---------------------------------------------------------------------------
# Phase 6
# ---------------------------------------------------------------------------
cells.append(
    md("""\
## Phase 6 - Gate VS drawdown reconciliation and three-way risk summary

10b's Gate VS (vol-scaled carry) reported a continuously-compounded log-return
cumsum max drawdown of -5.41 (log units), i.e. exp(-5.41)−1 ≈ -99.6% of peak
-- an artifact of compounding an 18-year daily log-return series without any
capital bound, not a real capital-at-risk number. This phase rebuilds Gate VS's
offset-0 daily return series into a capital-bounded, fixed-notional equity curve
(same fixed-notional convention spread_lib11.ret_eq uses everywhere else in
notebooks 10b/11a/11b) with an absorbing floor. Sharpe and DSR are UNCHANGED by
this recomputation (verified below); only the drawdown number differs.

The three-way risk gate (Sharpe, max drawdown, return/drawdown) across all seven
11b books at offset 0: no book clears Sharpe > 0.5, so the institutionally-fundable
flag is moot for all seven. Consistent with sec 5's worked example: the three-way
table is confirmatory rather than load-bearing.
""")
)

cells.append(
    code("""\
phase6 = load("phase_6_11b_results.json")
vs = phase6['gate_VS_drawdown_reconciliation']
print("Gate VS drawdown reconciliation:")
print(f"  10b Gate VS published max_dd (log-return cumsum): {vs['published_max_drawdown_net_log_units']:.2f}")
print(f"  Fixed-notional recomputation (absorbing floor): {vs['max_drawdown_fixed_notional_convention_pct']:.1%}")
print(f"  10b Sharpe published (net offset 0): {vs['published_sharpe_net_offset0']:.5f}")
print(f"  10b Sharpe recomputed from returns: {vs['recomputed_sharpe_net_offset0']:.5f}")
print(f"  Sharpe unchanged: {vs['sharpe_unchanged']}")
print()
print("Three-way risk gate summary (offset 0):")
print(f"{'gate':<10s} {'Sharpe':>10s} {'max dd':>10s} {'return/dd':>10s} {'fires':>8s}")
for gate_name, gate_rec in phase6['three_way_risk_gate_summary'].items():
    if 'sharpe' in gate_rec:
        print(f"{gate_name:<10s} {gate_rec['sharpe']:>10.3f} {gate_rec['max_drawdown']:>10.2%} "
              f"{gate_rec['return_over_drawdown']:>10.2f} {str(gate_rec['fires']):>8s}")
    else:
        print(f"{gate_name:<10s} {'--':>10s} {'--':>10s} {'--':>10s} {str(gate_rec.get('fires', '--')):>8s}")
""")
)

# ---------------------------------------------------------------------------
# Phase 7
# ---------------------------------------------------------------------------
cells.append(
    md("""\
## Phase 7 - Final gate table: cross-checked against 11a pre-registration

Final gate table, cross-checked verbatim against 11a's Phase 6 pre-registration
(`phase_6_11a_results.json`), per this notebook's own added rule that 11b's gate
table must match that pre-registration exactly -- no gate claim reworded, no n_trials
count changed or shrunk. DSR trial counts total 65 across the seven gates.
""")
)

cells.append(
    code("""\
phase7 = load("phase_7_11b_results.json")
print(f"Gate table final cross-check:")
print()
print(f"{'gate':<10s} {'verdict':>10s} {'n_trials_11b':>12s} {'n_trials_prereg':>15s} {'match':>8s}")
print("-" * 60)
for gate_name in ['TS', 'TS-S', 'BF', 'BF-X', 'SCR', 'VA', 'RE']:
    if gate_name in phase7['n_trials_actual']:
        n_11b = phase7['n_trials_actual'][gate_name]
        n_pre = phase7['n_trials_preregistered'].get(gate_name, 0)
        fires = phase7.get('fires', {}).get(gate_name, False)
        verdict = "fires" if fires else "no fire"
        match = n_11b == n_pre
        print(f"{gate_name:<10s} {verdict:>10s} {n_11b:>12d} {n_pre:>15d} {str(match):>8s}")
print("-" * 60)
print(f"{'TOTAL':<10s} {' ':>10s} {phase7['n_trials_11b_total']:>12d}")
print()
print(f"All gates match pre-registration: {phase7['n_trials_match_prereg_exactly']}")
print(f"Any gate fires: {phase7['any_gate_fires']}")
""")
)

# ---------------------------------------------------------------------------
# Bottom line
# ---------------------------------------------------------------------------
cells.append(
    md("""\
## Bottom line

Seven pre-registered gates testing the external repo's proposed mechanisms:
discrete-trade structure vs continuous, backwardation-regime flipping, ADF
screening, vol-adaptive stops, reentry-gating, standalone conflicts
diagnostics, and drawdown convention reconciliation. **All seven return
fired=False,** establishing that none of these mechanisms deliver tradeable
alpha on this repo's reproducing implementation and capital assumptions.

Gate TS was "the first proposition in this programme with a specific,
mechanically-identified reason to expect a positive result," and Gate BF
"the first with independent out-of-sample support from a separate codebase."
Both came back null against a properly paired control -- and both came back
null in the *informative* direction: **the external book's advantage is
universe selection and parameterization choices fitted on their own data, not
a portable mechanism.** The continuous, un-stopped, un-structured position
this whole programme has been implicitly comparing against since Gate SP
remains, on this repo's data and costs, the best-performing book in this
notebook.
""")
)

with open("src/research/11b_spread_mechanism_gates.ipynb", "w") as f:
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
    f"written src/research/11b_spread_mechanism_gates.ipynb ({len(cells)} cells)"
)
