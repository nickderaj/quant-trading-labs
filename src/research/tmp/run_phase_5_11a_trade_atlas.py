"""11a Phase 5: the trade-shape atlas (NEXT_PROMPT.md sec 3 Phase 5).
Replicates their `pattern-summary.md` analysis on our own Phase 4 control
book's trades. Descriptive only -- no gate verdict.

Regenerates the same book Phase 4 built (deterministic, ~seconds) rather
than depending on Phase 4's JSON, since Phase 4 does not persist the full
trade list (kept out of the committed JSON to stay small; this script is
the one place that needs trade-level granularity).

Writes phase_5_11a_results.json.
"""

import json
import sys

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import commod_lib8 as C8
import numpy as np
import polars as pl
import spread_lib11 as S11

SPREAD_DIR = "src/research/data/market/spreads"
OUT_PATH = "src/research/tmp/phase_5_11a_results.json"
DEV_END = "2024-12-31"

LIVE_SPREADS = [
    "brent_wti",
    "brent_calendar",
    "corn_wheat",
    "bean_corn",
    "kc_chicago_wheat",
]
STOP_ATR_OVERRIDES = {"brent_calendar": 4.0, "kc_chicago_wheat": 12.0}
REGIME_REQUIREMENTS = {"brent_calendar": "backwardation"}


def main() -> None:
    frames = {}
    for name in LIVE_SPREADS:
        df = pl.read_parquet(f"{SPREAD_DIR}/{name}.parquet")
        df = df.filter(pl.col("date") <= pl.lit(DEV_END).str.to_date()).sort("date")
        frames[name] = df

    p = S11.TradingRuleParams()
    params = {n: p for n in LIVE_SPREADS}
    book = S11.simulate_book(
        frames,
        params,
        STOP_ATR_OVERRIDES,
        REGIME_REQUIREMENTS,
        C8.round_turn_cost_per_contract,
    )
    trades = book["trades"]
    n = len(trades)

    pnl_atr_vals = np.array([t["pnl_atr"] for t in trades])
    median = np.median(pnl_atr_vals)
    top_group = [t for t, v in zip(trades, pnl_atr_vals) if v >= median]
    worst_group = [t for t, v in zip(trades, pnl_atr_vals) if v < median]

    def frac_stop(group):
        return (
            sum(1 for t in group if t["exit_reason"] == "stop") / len(group)
            if group
            else float("nan")
        )

    def median_abs_entry_z(group):
        return (
            float(np.median([abs(t["entry_z"]) for t in group]))
            if group
            else float("nan")
        )

    stop_trades = [t for t in trades if t["exit_reason"] == "stop"]
    zscore_trades = [t for t in trades if t["exit_reason"] == "zscore"]

    out: dict = {
        "n_trades": n,
        "entry_extremity": {
            "top_group_median_abs_entry_z": median_abs_entry_z(top_group),
            "worst_group_median_abs_entry_z": median_abs_entry_z(worst_group),
            "discriminates": abs(
                median_abs_entry_z(top_group) - median_abs_entry_z(worst_group)
            )
            > 0.3,
        },
        "stop_exit_fraction": {
            "top_group": frac_stop(top_group),
            "worst_group": frac_stop(worst_group),
        },
        "mae_mfe_by_outcome": {
            "winners_median_mae_atr": float(
                np.median([t["mae_atr"] for t in zscore_trades])
            )
            if zscore_trades
            else None,
            "winners_median_mfe_atr": float(
                np.median([t["mfe_atr"] for t in zscore_trades])
            )
            if zscore_trades
            else None,
            "losers_median_mae_atr": float(
                np.median([t["mae_atr"] for t in stop_trades])
            )
            if stop_trades
            else None,
            "losers_median_mfe_atr": float(
                np.median([t["mfe_atr"] for t in stop_trades])
            )
            if stop_trades
            else None,
        },
        "exit_reason_profile": {
            "stop": {
                "n": len(stop_trades),
                "mean_pnl_atr": float(np.mean([t["pnl_atr"] for t in stop_trades]))
                if stop_trades
                else None,
            },
            "zscore": {
                "n": len(zscore_trades),
                "mean_pnl_atr": float(np.mean([t["pnl_atr"] for t in zscore_trades]))
                if zscore_trades
                else None,
            },
        },
        "loss_win_asymmetry_ratio": (
            abs(
                float(np.mean([t["pnl_atr"] for t in stop_trades]))
                / float(np.mean([t["pnl_atr"] for t in zscore_trades]))
            )
            if stop_trades
            and zscore_trades
            and np.mean([t["pnl_atr"] for t in zscore_trades]) != 0
            else None
        ),
        "their_reference": {
            "entry_extremity_discriminates": False,
            "stop_exit_fraction_top": 0.0,
            "stop_exit_fraction_worst": 0.85,
            "winners_mae_mfe": [-0.31, 9.17],
            "losers_mae_mfe": [-6.78, 0.22],
            "exit_reason_profile": {
                "stop": {"n": 698, "mean_atr": -6.94},
                "zscore": {"n": 6837, "mean_atr": 2.05},
            },
        },
        "_dev_window_end": DEV_END,
        "_note": (
            "Our book has 57 trades total (Phase 4) vs their 333 (tune) / 9,545 pooled "
            "(trade-atlas, both books far larger than ours) -- every statistic here is "
            "reported on a much smaller sample and top/worst-group splits at n=57 median "
            "should be read as directionally suggestive, not independently powered."
        ),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(
        f"Phase 5: n={n}, top/worst |entry_z| median={out['entry_extremity']['top_group_median_abs_entry_z']:.2f}/"
        f"{out['entry_extremity']['worst_group_median_abs_entry_z']:.2f}, "
        f"stop-exit frac top/worst={out['stop_exit_fraction']['top_group']:.2f}/{out['stop_exit_fraction']['worst_group']:.2f}, "
        f"loss:win asymmetry={out['loss_win_asymmetry_ratio']}"
    )


if __name__ == "__main__":
    main()
