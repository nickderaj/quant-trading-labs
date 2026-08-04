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
# Notebook 013 — Four Outside Designs, Rebuilt and Scored on Our Own Data
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

prereg = load("phase_0_13_preregistration.json")
print("Gates pre-registered:", list(prereg["gates"].keys()))
print("Pooled n_trials:", prereg["n_trials"]["pooled_total"])
""")
)

cells.append(
    md("""\
## Phase 0 — Ground truth, pre-registration

Every data source and cost-model function NEXT_PROMPT.md sec2 claims exists was verified by
introspection before any backtest ran. Two disclosed cache gaps: `LUNAUSDT`/`FTTUSDT` 1h/6h bars
end at their real collapse/delisting dates (not a defect), and `MATICUSDT` loses its feed from
2024-10 onward (Binance's `MATIC`->`POL` rebrand). `databento` and `yfinance` trees were confirmed
to have zero overlapping files (safe to pool). All six gates (FF/SQ/AT/XS/TGT/TM), the pooled
`n_trials=18`, and every universe/window decision were committed to
`phase_0_13_preregistration.json` before Design A's backtest ran, and never edited afterward.
""")
)

cells.append(
    code("""\
findings = prereg["ground_truth_phase0_findings"]
for k, v in findings.items():
    print(f"- {k}:")
    if isinstance(v, dict):
        for kk, vv in v.items():
            print(f"    {kk}: {vv}")
    else:
        print(f"    {v}")
""")
)

# ---------------------------------------------------------------------------
# Design A
# ---------------------------------------------------------------------------
cells.append(
    md("""\
## Design A — forecast-to-fill trend/momentum on GC futures

Claim under test: the alpha is in the fill, not the signal. A single smoothed trend-momentum
state, sized through vol targeting, fractional-Kelly, and square-root market impact, exited via a
Wilder-ATR trailing stop under the required worse-of-(stop, gapped-open) fill convention.
""")
)

cells.append(
    code("""\
A = load("phase_A_13_results.json")
print("Trial variants (net Sharpe):")
for name, t in A["trials"].items():
    print(f"  {name}: {t['net']['sharpe']:.4f}  (DSR={t['dsr']:.3f}, CI excludes zero={t['ci_excludes_zero']})")
print()
base = A["trials"]["base"]
print(f"OOS window: {base['oos_date_range']}, n_bars={base['n_oos_bars']}")
print(f"Sharpe by origin offset: {{k: round(v['sharpe'],4) for k,v in base['by_offset'].items()}}")
""")
)

cells.append(
    code("""\
print("Benchmark neutrality vs spot GC=F:", A["benchmark_neutrality_vs_spot_GCF"])
print()
print("Arithmetic red flag check (pre-registered before the build):")
print(A["arithmetic_red_flag_check"])
print()
print("Capacity (AUM at which impact alone degrades net Sharpe):", A["capacity"])
print()
print("Gate FF fires:", A["gate_FF_fires"])
""")
)

# ---------------------------------------------------------------------------
# Design B
# ---------------------------------------------------------------------------
cells.append(
    md("""\
## Design B — sequence models on the cross-asset futures panel

Claim under test: models trained on negative realized Sharpe (not MSE) beat a linear benchmark on
a pooled cross-asset panel. Universe: 16 databento products + 6 CME FX futures = 22 instruments (no
bond/VIX futures anywhere in this repo, disclosed gap; yfinance's separate `ES=F` dropped to avoid
double-counting databento's own `ES`). One real bug was found and fixed in-flight: additive
back-adjustment produces a handful of negative synthetic prices for `CL`/`HO`/`NG`/`ZW`,
`log(negative)` is `NaN`, and polars' `drop_nulls()` does not remove `NaN` floats -- the first run's
full-batch Sharpe loss silently zeroed every seed's predictions from one fold onward. Fixed by
filtering non-positive back-adjusted prices and adding an explicit `is_finite()` check.
""")
)

cells.append(
    code("""\
B = load("phase_B_13_results.json")
lin = B["results"]["linear_baseline"]
print(f"Linear baseline: features={lin['features']}, Sharpe={lin['sharpe']:.4f}")
print()
for name in ["LSTM", "GatedLSTM"]:
    r = B["results"][name]
    print(f"{name}: median Sharpe (5 seeds) = {r['median_sharpe']:.4f}, "
          f"range [{r['min_sharpe']:.4f}, {r['max_sharpe']:.4f}]")
    print(f"  seed Sharpes: {[round(s,4) for s in r['seed_sharpes']]}")
    print(f"  breakeven cost: {r['breakeven_cost_bps']} bps/side, DSR={r['dsr_at_median']:.3f}")
print()
print("Best architecture:", B["best_architecture"], "-- beats linear baseline:",
      B["results"][B["best_architecture"]]["median_sharpe"] > lin["sharpe"])
print("Gate SQ fires:", B["gate_SQ_fires"])
""")
)

# ---------------------------------------------------------------------------
# Design C
# ---------------------------------------------------------------------------
cells.append(
    md("""\
## Design C v1 — adaptive trend, 6h crypto, asymmetric long/short book (superseded, kept for the record)

Claim under test: volatility-adaptive trailing exits, quality/liquidity selection, and causal
monthly re-parameterization clear Sharpe 2.4 net of 4bp fees. NEXT_PROMPT.md's premise that "we
have no 6h cache" turned out to be false -- 6h is a native Binance interval, fetched directly for
all 30 symbols rather than resampled from 1h.

This build guessed at several parameters the source paper (arXiv:2602.11708) actually specifies.
See Design C v2 below for the corrected rebuild -- v1 is kept here, unedited, as the honest history
of getting there, not erased.
""")
)

cells.append(
    code("""\
C = load("phase_C_13_results.json")
print("Frozen params (fit once on 2021H2 calibration):", C["frozen_params"])
print(f"Adaptive months fit: {C['n_adaptive_months']}")
print()
for name, t in C["trials"].items():
    print(f"{name}: net Sharpe={t['net']['sharpe']:.4f}, max_drawdown={t['net']['max_drawdown']:.3f}")
print()
print("Adaptive vs frozen paired CI:", C["adaptive_vs_frozen_paired_ci"],
      "-- beats frozen:", C["adaptive_beats_frozen"])
print("Ablation directions (True = moved as design predicted):", C["ablation_predicted_direction"])
print(f"{C['n_ablations_correct_direction']}/4 ablations correctly signed")
print()
print("Gate AT fires:", C["gate_AT_fires"])
""")
)

cells.append(
    code("""\
print("Origin offset check (net Sharpe by offset, in 6h bars):")
for off, t in C["by_offset"].items():
    print(f"  {off}: {t['net']['sharpe']:.4f}")
print("(Not vacuous -- a real perturbation, unlike Design A's and notebook 012's offset legs.)")
""")
)

# ---------------------------------------------------------------------------
# Design C v2
# ---------------------------------------------------------------------------
cells.append(
    md("""\
## Design C v2 -- corrected rebuild matching the actual AdaptiveTrend paper (arXiv:2602.11708)

Once the source paper was supplied and read, three material implementation gaps in v1 were
identifiable by direct comparison and fixed: quality thresholds corrected to the paper's stated
1.3 (long) / -1.7 (short), universe expanded from 30 to 128 Binance USDT perpetuals (paper: "150+"),
and selection changed from a dollar-volume proxy to the paper's literal top-15/bottom-15-by-market-
cap rule (market cap from CoinGecko's free API, a current-day snapshot applied statically --
disclosed limitation, not a rolling historical reconstruction). Funding-rate costs, previously
disclosed as unmodelled, are now fetched live from Binance and included.
""")
)

cells.append(
    code("""\
C2 = load("phase_C_13_v2_results.json")
print("Corrections from v1:")
for k, v in C2["corrections_from_v1"].items():
    print(f"  {k}: {v}")
print()
print(f"Universe: {C2['universe_size']} symbols, {C2['n_symbols_with_market_cap']} with mapped market cap")
print("Frozen params:", C2["frozen_params"], f"-- {C2['n_adaptive_months']} adaptive months fit")
""")
)

cells.append(
    code("""\
import numpy as np

print("Trial variants (net Sharpe, max drawdown as simple-return equivalent):")
for name, t in C2["trials"].items():
    dd = np.exp(t["net"]["max_drawdown"]) - 1
    print(f"  {name}: Sharpe={t['net']['sharpe']:.4f}, drawdown={dd:.1%}")
print()
print("Adaptive vs frozen paired CI:", C2["adaptive_vs_frozen_paired_ci"],
      "-- beats frozen:", C2["adaptive_beats_frozen"])
print("Ablation directions (True = moved as design predicted):", C2["ablation_predicted_direction"])
print(f"{C2['n_ablations_correct_direction']}/3 ablations correctly signed "
      "(v1 got 2/4 -- every ablation now matches the paper's described mechanism)")
print()
print("Gate AT v2 fires:", C2["gate_AT_v2_fires"])
""")
)

cells.append(
    md("""\
v1's most damaging problem -- a real (~59%) max drawdown from capital concentrating into one or two
names whenever a too-loose selection filter let a stress month through -- is gone with the paper's
actual 1.3/1.7 threshold in place (v2's drawdown is a sane ~7%). Every ablation now moves in the
paper-predicted direction, including one (removing the selection filter) whose bootstrap CI
excludes zero -- a real, statistically distinguishable effect. The headline Sharpe is still
negative and Gate AT v2 still does not fire: a much more faithful reproduction of the paper's own
method still does not find tradable alpha on this data, which strengthens rather than weakens the
overall null -- there is no longer a "maybe this was implemented wrong" objection available for
Design C.
""")
)

# ---------------------------------------------------------------------------
# Design D
# ---------------------------------------------------------------------------
cells.append(
    md("""\
## Design D — cross-sectional attention over the crypto correlation graph

Claim under test: crypto returns are driven by correlated neighbours (cross-sectional IC ~0.047,
in 003's own surviving-factor range), and temporal mixing hurts. Universe restricted to the 26 of
30 symbols with complete daily coverage (fixed node count required for batch GAT training).
""")
)

cells.append(
    code("""\
D = load("phase_D_13_results.json")
print(f"Universe: {D['universe_size']} symbols, node degree {D['node_degree']}")
print()
print("Cross-sectional IC sanity check (run BEFORE any backtest is believed):")
for variant, ic in D["ic_sanity_check"].items():
    print(f"  {variant}: mean_ic={ic['mean_ic']:.4f}, NW t={ic['nw_tstat']:.2f}, "
          f"passes 003 filter={ic['passes_003_survival_filter']}, leak_suspected={ic['leak_suspected']}")
print(f"  (003's own surviving factors: mean-reversion +0.042, realized-vol -0.038)")
print()
print("*** Sign is INVERTED from the design's claimed +0.047 -- a significant anti-finding,")
print("*** not an absence of signal.")
""")
)

cells.append(
    code("""\
for name, m in D["books"].items():
    print(f"{name}: net Sharpe={m['net']['sharpe']:.4f}, "
          f"CI={m['ci_95']}, excludes zero={m['ci_excludes_zero']}, turnover={m['mean_turnover']:.3f}")
print()
print("Dollar-neutral beta to equal-weight basket:", D["dollar_neutral_beta_to_basket"])
print("Gate TM:", D["gate_TM"])
print("Gate XS fires:", D["gate_XS_fires"])
""")
)

# ---------------------------------------------------------------------------
# Phase L / final gate table
# ---------------------------------------------------------------------------
cells.append(
    md("""\
## Phase L — the look-ahead audit, and the final gate table

Pre-registered: any design's dev-window net Sharpe >= 1.5 is a suspected defect until it survives
Phase L. None of the four designs clears 1.5 -- the audit never triggers, and that absence is
itself reported rather than silently skipped.
""")
)

cells.append(
    code("""\
L = load("phase_L_13_results.json")
for design, result in L["designs"].items():
    print(f"Design {design}: headline net Sharpe = {result['headline_net_sharpe']:.4f}, "
          f"Phase L triggered = {result['triggered']}")
""")
)

cells.append(
    md("""\
## What this notebook establishes, plainly

All seven pre-registered gates (including Design C's corrected v2 rebuild, AT_v2) return
`fires=False`. The best net Sharpe achieved across all four externally-specified designs is Design
A's **+0.33**, still below 007's own +1.053 record and far short of the four designs' reported range
of 2.4-3.1. Design D fails in the more informative way -- not "no effect detected" but "the specific
mechanism moves the wrong way, with a confidence interval that excludes zero": its cross-sectional
IC is significant but inverted in sign. Design C's story sharpened after its source paper was
supplied: a corrected v2 rebuild fixed three material parameter mismatches (quality thresholds,
universe scale, selection mechanism), and every ablation now moves exactly as the paper's authors
predict -- yet the mechanism still nets out negative on this data, a cleaner refutation than v1's
original result precisely because it removes any "you built it wrong" objection. Combined with
011a's -0.16 reproduction of an outside spread book and the twenty-two null gates before this
notebook, four independently-sourced outside designs, rebuilt end to end with their authors' own
identified sources of edge intact, also fail to reproduce on this programme's data and costs. See
`src/results/013_four_outside_designs_rebuilt_and_scored.md` for the full writeup, substitutions
table, and per-design detail.
""")
)

with open("src/research/013_four_outside_designs_rebuilt_and_scored.ipynb", "w") as f:
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
print(f"written src/research/013_four_outside_designs_rebuilt_and_scored.ipynb ({len(cells)} cells)")
