"""Notebook 020, Phase 5 (NEXT_PROMPT.md sec 7/sec 8): the thirteen
ablations, itemised in sec 7 -- 7 on Mechanism A's headline (A3), 6 on
Mechanism B's headline (B0). Evaluated at offset 0 only (018's own
convention, sec 8: "the offsets are vacuous").

Cost-sensitivity ablations test three explicit round-turn levels each; the
break-even point reported alongside them is an INTERPOLATION between those
three measured points, not a fourth tested configuration (018's convention,
keeps n_trials exact).

Usage: uv run python src/research/tmp/run_phase_5_20_ablations.py [--variant NAME --out PATH]
       uv run python src/research/tmp/run_phase_5_20_ablations.py --collate --cells DIR --out PATH
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import scipy.stats as st

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "research" / "tmp"))

import basis_lib18 as bl18
import basis_lib20 as bl20

import research

SCRATCH_DIR = REPO_ROOT / "scratch" / "020"
ANNUALIZED_RATE = research.sharpe_to_annualized_rate("8h")

A3_COST_LEVELS_BP = {"cost_0bp": 0.0, "cost_17bp": 17.0, "cost_51bp": 51.0}
B0_COST_LEVELS_BP = {"cost_0bp": 0.0, "cost_12.5bp": 12.5, "cost_37.5bp": 37.5}

A3_VARIANTS = [
    "A3_no_hysteresis",
    "A3_nmin2",
    "A3_nmin5",
    "A3_cost_0bp",
    "A3_cost_17bp",
    "A3_cost_51bp",
    "A3_excl_luna_ftt",
]
B0_VARIANTS = [
    "B0_no_hysteresis",
    "B0_one_venue_leg_only",
    "B0_cost_0bp",
    "B0_cost_12.5bp",
    "B0_cost_37.5bp",
    "B0_excl_top2",
]
ALL_VARIANTS = A3_VARIANTS + B0_VARIANTS


def _json_default(o: object) -> object:
    if isinstance(o, np.floating | np.integer):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _apply_costs_at_round_turn_bp(
    trade_frame: pl.DataFrame, round_turn_bp: float
) -> pl.DataFrame:
    """Same log-cost accounting as bl18.apply_two_leg_costs/bl20.apply_xvenue_costs,
    parameterised by an arbitrary round-turn bp (one-way cost = round_turn_bp/2).
    """
    cost_frac = (round_turn_bp / 2.0) * 1e-4
    return (
        trade_frame.with_columns(
            (1 - cost_frac * pl.col("turnover")).log().alias("cost_log_return")
        )
        .with_columns(
            (pl.col("trade_log_return") + pl.col("cost_log_return")).alias(
                "trade_log_return_net"
            )
        )
        .with_columns(
            pl.col("trade_log_return_net").cum_sum().alias("equity_curve_net")
        )
        .with_columns(
            (pl.col("equity_curve_net") - pl.col("equity_curve_net").cum_max()).alias(
                "drawdown_log_return_net"
            )
        )
    )


def _metrics_and_summary(costed: pl.DataFrame, label: str) -> dict[str, Any]:
    metrics = bl18.book_metrics(costed, ANNUALIZED_RATE, label)
    net = costed["trade_log_return_net"].drop_nulls().to_numpy()
    return {
        "metrics": metrics,
        "n_obs": len(net),
        "skew": float(st.skew(net, nan_policy="omit")),
        "kurtosis_non_excess": float(st.kurtosis(net, fisher=False, nan_policy="omit")),
    }


def _interpolate_breakeven(
    levels_bp: dict[str, float], sharpes: dict[str, float]
) -> dict[str, Any]:
    """Linear interpolation of the round-turn bp at which net Sharpe crosses
    0.5 (FA2_SHARPE_BOUND), across the three measured cost points. Reported
    as interpolated, never as a tested configuration (sec 8).
    """
    pairs = sorted((levels_bp[k], sharpes[k]) for k in levels_bp)
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    target = 0.5
    for i in range(len(xs) - 1):
        if (ys[i] - target) * (ys[i + 1] - target) <= 0 and ys[i] != ys[i + 1]:
            frac = (target - ys[i]) / (ys[i + 1] - ys[i])
            breakeven = xs[i] + frac * (xs[i + 1] - xs[i])
            return {
                "breakeven_round_turn_bp": breakeven,
                "interpolated": True,
                "in_range": True,
            }
    return {
        "breakeven_round_turn_bp": None,
        "interpolated": True,
        "in_range": False,
        "note": "Sharpe does not cross 0.5 within the three measured cost points",
    }


def build_variant(name: str) -> dict[str, Any]:
    if name.startswith("A3"):
        panel = pl.read_parquet(
            SCRATCH_DIR / f"panel_dev_binance_hl{bl20.SLOW_CARRY_HALF_LIFE}.parquet"
        )
        if name == "A3_no_hysteresis":
            weights = bl20.build_book_weights_v2(
                panel,
                timed=True,
                n_min=bl20.N_MIN,
                theta_in=bl20.THETA_IN_SLOW,
                theta_out=bl20.THETA_IN_SLOW,
            )
            tf = bl20.book_trade_frame(panel, weights, "fwd_paired_return_1", 0)
            costed = bl18.apply_two_leg_costs(tf)
        elif name in ("A3_nmin2", "A3_nmin5"):
            n_min = 2 if name == "A3_nmin2" else 5
            weights = bl20.build_book_weights_v2(
                panel,
                timed=True,
                n_min=n_min,
                theta_in=bl20.THETA_IN_SLOW,
                theta_out=bl20.THETA_OUT_SLOW,
            )
            tf = bl20.book_trade_frame(panel, weights, "fwd_paired_return_1", 0)
            costed = bl18.apply_two_leg_costs(tf)
        elif name.startswith("A3_cost_"):
            bp = A3_COST_LEVELS_BP[name.replace("A3_", "")]
            weights = bl20.build_book_weights_v2(
                panel,
                timed=True,
                n_min=bl20.N_MIN,
                theta_in=bl20.THETA_IN_SLOW,
                theta_out=bl20.THETA_OUT_SLOW,
            )
            tf = bl20.book_trade_frame(panel, weights, "fwd_paired_return_1", 0)
            costed = _apply_costs_at_round_turn_bp(tf, bp)
        elif name == "A3_excl_luna_ftt":
            filtered = panel.filter(~pl.col("symbol").is_in(["LUNAUSDT", "FTTUSDT"]))
            weights = bl20.build_book_weights_v2(
                filtered,
                timed=True,
                n_min=bl20.N_MIN,
                theta_in=bl20.THETA_IN_SLOW,
                theta_out=bl20.THETA_OUT_SLOW,
            )
            tf = bl20.book_trade_frame(filtered, weights, "fwd_paired_return_1", 0)
            costed = bl18.apply_two_leg_costs(tf)
        else:
            raise ValueError(name)
        result = _metrics_and_summary(costed, name)
        result["mechanism"] = "A"
        return result

    if name.startswith("B0"):
        panel = pl.read_parquet(
            SCRATCH_DIR / f"panel_dev_xvenue_hl{bl20.CARRY_EWMA_HALF_LIFE}.parquet"
        )
        if name == "B0_no_hysteresis":
            weights = bl20.build_xvenue_book_weights(
                panel,
                timed=True,
                n_min=bl20.N_MIN_XV,
                theta_in=bl20.THETA_IN_XV,
                theta_out=bl20.THETA_IN_XV,
            )
            tf = bl20.book_trade_frame(panel, weights, "fwd_xvenue_paired_return_1", 0)
            costed = bl20.apply_xvenue_costs(tf)
        elif name.startswith("B0_cost_"):
            bp = B0_COST_LEVELS_BP[name.replace("B0_", "")]
            weights = bl20.build_xvenue_book_weights(
                panel,
                timed=True,
                n_min=bl20.N_MIN_XV,
                theta_in=bl20.THETA_IN_XV,
                theta_out=bl20.THETA_OUT_XV,
            )
            tf = bl20.book_trade_frame(panel, weights, "fwd_xvenue_paired_return_1", 0)
            costed = _apply_costs_at_round_turn_bp(tf, bp)
        elif name == "B0_one_venue_leg_only":
            weights = bl20.build_xvenue_book_weights(
                panel,
                timed=True,
                n_min=bl20.N_MIN_XV,
                theta_in=bl20.THETA_IN_XV,
                theta_out=bl20.THETA_OUT_XV,
            )
            # Neutrality control: what if the position were taken via ONLY
            # the Binance leg's price return (no Bybit leg, no funding)?
            # If this alone reproduces B0's real return, the strategy is a
            # disguised single-leg directional bet, not a genuine spread.
            single_leg_panel = panel.with_columns(
                (
                    -(
                        pl.col("binance_close")
                        / pl.col("binance_close").shift(1).over("symbol")
                    ).log()
                )
                .shift(-1)
                .over("symbol")
                .alias("fwd_xvenue_paired_return_1")
            )
            tf = bl20.book_trade_frame(
                single_leg_panel, weights, "fwd_xvenue_paired_return_1", 0
            )
            costed = bl20.apply_xvenue_costs(tf)
        elif name == "B0_excl_top2":
            base_weights = bl20.build_xvenue_book_weights(
                panel,
                timed=True,
                n_min=bl20.N_MIN_XV,
                theta_in=bl20.THETA_IN_XV,
                theta_out=bl20.THETA_OUT_XV,
            )
            base_tf = bl20.book_trade_frame(
                panel, base_weights, "fwd_xvenue_paired_return_1", 0
            )
            base_costed = bl20.apply_xvenue_costs(base_tf)
            contrib = (
                base_weights.join(
                    panel.select("datetime", "symbol", "fwd_xvenue_paired_return_1"),
                    on=["datetime", "symbol"],
                    how="inner",
                )
                .with_columns(
                    (pl.col("weight") * pl.col("fwd_xvenue_paired_return_1")).alias(
                        "contribution"
                    )
                )
                .group_by("symbol")
                .agg(pl.col("contribution").sum().alias("total_contribution"))
                .sort("total_contribution", descending=True)
            )
            top2 = contrib.head(2)["symbol"].to_list()
            filtered = panel.filter(~pl.col("symbol").is_in(top2))
            weights = bl20.build_xvenue_book_weights(
                filtered,
                timed=True,
                n_min=bl20.N_MIN_XV,
                theta_in=bl20.THETA_IN_XV,
                theta_out=bl20.THETA_OUT_XV,
            )
            tf = bl20.book_trade_frame(
                filtered, weights, "fwd_xvenue_paired_return_1", 0
            )
            costed = bl20.apply_xvenue_costs(tf)
            result = _metrics_and_summary(costed, name)
            result["mechanism"] = "B"
            result["excluded_symbols"] = top2
            base_metrics = research._series_metrics(
                base_costed["trade_log_return_net"], ANNUALIZED_RATE, "base"
            )
            result["base_net_sharpe_for_comparison"] = float(base_metrics["sharpe"])
            return result
        else:
            raise ValueError(name)
        result = _metrics_and_summary(costed, name)
        result["mechanism"] = "B"
        return result

    raise ValueError(name)


def cmd_run(args: argparse.Namespace) -> None:
    result = build_variant(args.variant)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    print(f"{args.variant}: net Sharpe={result['metrics']['sharpe_net']:.4f}")


def cmd_collate(args: argparse.Namespace) -> None:
    cells_dir = Path(args.cells)
    cells: dict[str, Any] = {}
    for name in ALL_VARIANTS:
        path = cells_dir / f"{name}.json"
        if path.exists():
            with open(path) as f:
                cells[name] = json.load(f)

    out: dict[str, Any] = {"cells": cells}

    if all(f"A3_cost_{k}" in cells for k in ("0bp", "17bp", "51bp")):
        a3_sharpes = {
            f"cost_{k}": cells[f"A3_cost_{k}"]["metrics"]["sharpe_net"]
            for k in ("0bp", "17bp", "51bp")
        }
        out["a3_breakeven"] = _interpolate_breakeven(A3_COST_LEVELS_BP, a3_sharpes)

    if all(f"B0_cost_{k}" in cells for k in ("0bp", "12.5bp", "37.5bp")):
        b0_sharpes = {
            f"cost_{k}": cells[f"B0_cost_{k}"]["metrics"]["sharpe_net"]
            for k in ("0bp", "12.5bp", "37.5bp")
        }
        out["b0_breakeven"] = _interpolate_breakeven(B0_COST_LEVELS_BP, b0_sharpes)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=_json_default)
    print(f"Wrote {out_path} ({len(cells)}/{len(ALL_VARIANTS)} ablations)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant")
    parser.add_argument("--out", required=True)
    parser.add_argument("--collate", action="store_true")
    parser.add_argument("--cells")
    args = parser.parse_args()
    if args.collate:
        cmd_collate(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
