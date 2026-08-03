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
# Notebook 012 — Volume-Confirmed Breakout, One Rule, Whole Basket
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
""")
)

# ---------------------------------------------------------------------------
# Phase 0
# ---------------------------------------------------------------------------
cells.append(
    md("""\
## Phase 0 — Universe, regime gates, and frozen thresholds

Pooled basket: 30 crypto perpetuals, 42 commodity-equity/ETF tickers (the
27 `*=F` FX/futures proxies excluded -- their yfinance volume is unreliable,
NEXT_PROMPT.md sec 1), and 16 databento futures products with OHLCV carried
through the roll via a new `commod_lib8.build_continuous_series_ohlcv`
(the existing `build_continuous_series` drops open/high/low/volume
entirely). 88 instruments total.

Thresholds are frozen once, before any pooled backtest runs, from each
instrument's own first 3 years of history (or its full history if
shorter) -- never re-tuned after seeing a result. All three are
scale-free (ATR multiples or a volume-ratio percentile), not %-of-price,
so a single set of numbers transfers across four very different asset
classes.
""")
)

cells.append(
    code("""\
phase0 = load("phase_0_12_results.json")
print(f"Instruments: {phase0['n_instruments']} (total {phase0['total_instruments']})")
print(f"Regime gate open fraction: {phase0['regime_ok_frac']}")
print(f"Delisted crypto present: {phase0['delisted_crypto_present']}")
print(f"Frozen thresholds: {phase0['thresholds']['base_max_range_atr_mult']=}, "
      f"{phase0['thresholds']['prior_run_min_atr_mult']=}, "
      f"{phase0['thresholds']['vol_k']=}")
print(f"Calibration bars pooled: {phase0['thresholds']['n_calibration_bars']}")
""")
)

# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------
cells.append(
    md("""\
## Phase 1 — Gate VB pre-registration

Committed before the pooled backtest ran; Phase 2/3 assert against this
file programmatically rather than re-typing its criteria.
""")
)

cells.append(
    code("""\
prereg = load("phase_1_12_preregistration.json")["gates"]["VB"]
print(f"n_trials: {prereg['n_trials']}")
print(f"fires_if: {prereg['fires_if']}")
""")
)

# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------
cells.append(
    md("""\
## Phase 2 — Gate VB: pooled book, volume-gated vs. the identical ungated control

The control is the byte-identical breakout rule (same signals, stops,
exits, costs, bars, regime gate) with only the volume condition switched
off -- the mechanism isolation NEXT_PROMPT.md sec 3 asks for.
""")
)

cells.append(
    code("""\
phase2 = load("phase_2_12_results.json")
print("Sharpe (volume-gated, 1x cost) by offset:")
for off, s in phase2["sharpes_gated_1x_by_offset"].items():
    print(f"  {off}: {s:.4f}")
print(f"Positive every offset: {phase2['positive_every_offset']}")
print()
gvu = phase2["gated_vs_ungated_bootstrap_offset0_1x"]
print(f"Gated-minus-ungated delta (offset 0, 1x): point {gvu['delta_point']:.4f}, "
      f"95% CI [{gvu['delta_ci'][0]:.4f}, {gvu['delta_ci'][1]:.4f}], "
      f"excludes zero: {gvu['delta_excludes_zero']}")
print()
cs = phase2["cost_stress_1x_vs_3x_gated_offset0"]
print(f"Cost stress (1x vs 3x, gated, offset 0): delta {cs['delta_point']:.4f}, "
      f"95% CI [{cs['delta_ci'][0]:.4f}, {cs['delta_ci'][1]:.4f}], "
      f"correctly signed: {phase2['cost_stress_correctly_signed']}")
print()
print(f"Deflated Sharpe prob (n_trials={phase2['n_trials']}): {phase2['deflated_sharpe_prob']:.3f}")
print(f"DSR fires: {phase2['dsr_fires']}")
print(f"Gate VB fires: {phase2['gate_fires']}")
print(f"Fundable flag: {phase2['fundable_flag']}")
print()
print(f"n trades gated (offset 0, 1x): {phase2['n_trades_gated_offset0_1x']}")
print(f"n trades ungated (offset 0, 1x): {phase2['n_trades_ungated_offset0_1x']}")
print(f"Trade counts by asset class, gated: {phase2['trade_counts_by_asset_class_gated']}")
print(f"Trade counts by asset class, ungated: {phase2['trade_counts_by_asset_class_ungated']}")
print(f"Max drawdown (offset 0, 1x, gated): {phase2['max_drawdown_offset0_1x_gated']:.4f}")
""")
)

# ---------------------------------------------------------------------------
# Phase 3
# ---------------------------------------------------------------------------
cells.append(
    md("""\
## Phase 3 — Final gate table, cross-checked against pre-registration
""")
)

cells.append(
    code("""\
phase3 = load("phase_3_12_results.json")
print(f"Gate VB fires: {phase3['fires']}")
print(f"Fundable: {phase3['fundable']}")
print("Legs:")
for k, v in phase3["legs"].items():
    print(f"  {k}: {v}")
""")
)

cells.append(
    md("""\
## What this notebook establishes, plainly

Gate VB does not fire. Net Sharpe is positive at every offset on the
volume-gated book (+0.115 at 1x cost, essentially flat across offsets 0/7/
14/21 -- an artifact of stacking a small origin-offset shift on top of a
per-instrument calibration exclusion that already removes each
instrument's first ~3 years, so shifting the remaining start by a further
0/7/14/21 bars leaves an identical 406-trade book at 3-decimal precision;
see the results markdown for the full disclosure). But the volume filter
does not earn its keep: the gated-minus-ungated paired bootstrap CI
includes zero and the point estimate goes the wrong way (gated book's
pooled return is *lower* than the ungated control's, not higher), and the
Deflated Sharpe Ratio is far below the 0.95 bar even at the honestly small
n_trials=12. Cost stress is correctly signed and real, same as every
other notebook in this programme -- the null is not explained by an
inflated cost assumption. Trade counts are reported per asset class
because pooling here is not balanced: equities supply the large majority
of trades, and that imbalance is part of the finding, not hidden by it.
""")
)

with open("src/research/012_volume_confirmed_breakout.ipynb", "w") as f:
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
print(f"written src/research/012_volume_confirmed_breakout.ipynb ({len(cells)} cells)")
