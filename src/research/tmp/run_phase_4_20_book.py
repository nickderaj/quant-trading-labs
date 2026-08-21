"""Notebook 020, Phase 4 (NEXT_PROMPT.md sec 8): the books.

Runs ONE cell per invocation (--variant NAME --offset N --out PATH), or
collates already-computed cells (--collate --cells DIR --out PATH).

DSR call shape copies run_phase_4_18_backtest.py lines 220-234 exactly:
per-period Sharpe, n_obs=len(returns), skew/kurtosis via scipy with
fisher=False (sec 6's implementation notes -- the single easiest place to
produce a plausible-looking wrong number is passing an annualized Sharpe or
excess kurtosis here).
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
N_TRIALS = 32
BENCHMARK_BETA_BOUND = 0.10
FA2_SHARPE_BOUND = 0.5
DSR_BOUND = 0.95
ANNUALIZED_RATE = research.sharpe_to_annualized_rate("8h")


def _json_default(o: object) -> object:
    if isinstance(o, np.floating | np.integer):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _load_binance_panels() -> dict[int, pl.DataFrame]:
    return {
        hl: pl.read_parquet(SCRATCH_DIR / f"panel_dev_binance_hl{hl}.parquet")
        for hl in (bl20.CARRY_EWMA_HALF_LIFE, bl20.SLOW_CARRY_HALF_LIFE)
    }


def _load_xvenue_panels() -> dict[int, pl.DataFrame]:
    return {
        hl: pl.read_parquet(SCRATCH_DIR / f"panel_dev_xvenue_hl{hl}.parquet")
        for hl in (bl20.CARRY_EWMA_HALF_LIFE, bl20.SLOW_CARRY_HALF_LIFE)
    }


def _b_single_panel() -> pl.DataFrame:
    cache_path = SCRATCH_DIR / "panel_dev_b_single.parquet"
    if cache_path.exists():
        return pl.read_parquet(cache_path)
    xvenue_symbols = bl20.load_xvenue_universe()
    panel, _manifest = bl18.load_basis_panel(
        symbols=xvenue_symbols, start_date=bl18.DEV_START, end_date=bl18.DEV_END
    )
    featured = bl18.add_trade_features(panel)
    featured.write_parquet(cache_path)
    return featured


def _basket_and_btc_benchmark() -> tuple[pl.DataFrame, pl.DataFrame]:
    """The repo-wide equal-weight crypto basket + BTC benchmark, from the
    full 126-symbol Binance panel -- shared by both mechanisms' beta gates
    (RC-4/XD-4), per sec 6's "the equal-weight crypto basket" as one fixed
    construct.
    """
    hl21 = pl.read_parquet(
        SCRATCH_DIR / f"panel_dev_binance_hl{bl20.CARRY_EWMA_HALF_LIFE}.parquet"
    )
    basket = research.equal_weight_basket_returns(
        hl21, target_col="fwd_perp_return_1"
    ).rename({"trade_log_return": "trade_log_return_basket"})
    btc = hl21.filter(pl.col("symbol") == "BTCUSDT").select(
        "datetime", pl.col("fwd_perp_return_1").alias("btc_return")
    )
    return basket, btc


def _compute_beta(
    costed: pl.DataFrame, basket: pl.DataFrame, btc: pl.DataFrame
) -> tuple[float, float]:
    beta_frame = (
        costed.select("datetime", "trade_log_return_net")
        .join(basket, on="datetime", how="inner")
        .join(btc, on="datetime", how="inner")
    )
    beta_basket = bl20.ols_beta(
        beta_frame["trade_log_return_net"].to_numpy(),
        beta_frame["trade_log_return_basket"].to_numpy(),
    )
    beta_btc = bl20.ols_beta(
        beta_frame["trade_log_return_net"].to_numpy(),
        beta_frame["btc_return"].to_numpy(),
    )
    return beta_basket, beta_btc


def _cash_frame(all_times: pl.Series, offset: int) -> pl.DataFrame:
    frame = pl.DataFrame(
        {
            "datetime": all_times,
            "trade_log_return": [0.0] * len(all_times),
            "turnover": [0.0] * len(all_times),
        }
    )
    if offset > 0 and offset < len(all_times):
        cutoff = all_times[offset]
        frame = frame.filter(pl.col("datetime") >= cutoff)
    return frame


def build_cell(variant: str, offset: int) -> tuple[dict[str, Any], pl.DataFrame]:
    binance_panels = None
    xvenue_panels = None

    def get_binance():
        nonlocal binance_panels
        if binance_panels is None:
            binance_panels = _load_binance_panels()
        return binance_panels

    def get_xvenue():
        nonlocal xvenue_panels
        if xvenue_panels is None:
            xvenue_panels = _load_xvenue_panels()
        return xvenue_panels

    if variant == "A0":
        panel = get_binance()[bl20.CARRY_EWMA_HALF_LIFE]
        weights = bl20.build_book_weights_v2(
            panel, timed=True, n_min=1, theta_in=bl20.THETA_IN, theta_out=bl20.THETA_OUT
        )
        tf = bl20.book_trade_frame(panel, weights, "fwd_paired_return_1", offset)
        costed = bl18.apply_two_leg_costs(tf)
        mechanism, headline = "A", False
    elif variant == "A1":
        panel = get_binance()[bl20.CARRY_EWMA_HALF_LIFE]
        weights = bl20.build_book_weights_v2(
            panel,
            timed=True,
            n_min=bl20.N_MIN,
            theta_in=bl20.THETA_IN,
            theta_out=bl20.THETA_OUT,
        )
        tf = bl20.book_trade_frame(panel, weights, "fwd_paired_return_1", offset)
        costed = bl18.apply_two_leg_costs(tf)
        mechanism, headline = "A", False
    elif variant == "A2":
        panel = get_binance()[bl20.SLOW_CARRY_HALF_LIFE]
        weights = bl20.build_book_weights_v2(
            panel,
            timed=True,
            n_min=1,
            theta_in=bl20.THETA_IN_SLOW,
            theta_out=bl20.THETA_OUT_SLOW,
        )
        tf = bl20.book_trade_frame(panel, weights, "fwd_paired_return_1", offset)
        costed = bl18.apply_two_leg_costs(tf)
        mechanism, headline = "A", False
    elif variant == "A3":
        panel = get_binance()[bl20.SLOW_CARRY_HALF_LIFE]
        weights = bl20.build_book_weights_v2(
            panel,
            timed=True,
            n_min=bl20.N_MIN,
            theta_in=bl20.THETA_IN_SLOW,
            theta_out=bl20.THETA_OUT_SLOW,
        )
        tf = bl20.book_trade_frame(panel, weights, "fwd_paired_return_1", offset)
        costed = bl18.apply_two_leg_costs(tf)
        mechanism, headline = "A", True
    elif variant == "A_alwayson":
        panel = get_binance()[bl20.SLOW_CARRY_HALF_LIFE]
        weights = bl20.build_book_weights_v2(panel, timed=False, n_min=1)
        tf = bl20.book_trade_frame(panel, weights, "fwd_paired_return_1", offset)
        costed = bl18.apply_two_leg_costs(tf)
        mechanism, headline = "A", False
    elif variant == "A_cash":
        panel = get_binance()[bl20.CARRY_EWMA_HALF_LIFE]
        all_times = panel["datetime"].unique(maintain_order=True).sort()
        tf = _cash_frame(all_times, offset)
        costed = tf.with_columns(
            pl.col("trade_log_return").alias("trade_log_return_net")
        )
        mechanism, headline = "A", False
    elif variant == "B0":
        panel = get_xvenue()[bl20.CARRY_EWMA_HALF_LIFE]
        weights = bl20.build_xvenue_book_weights(
            panel,
            timed=True,
            n_min=bl20.N_MIN_XV,
            theta_in=bl20.THETA_IN_XV,
            theta_out=bl20.THETA_OUT_XV,
        )
        tf = bl20.book_trade_frame(panel, weights, "fwd_xvenue_paired_return_1", offset)
        costed = bl20.apply_xvenue_costs(tf)
        mechanism, headline = "B", True
    elif variant == "B1":
        panel = get_xvenue()[bl20.SLOW_CARRY_HALF_LIFE]
        weights = bl20.build_xvenue_book_weights(
            panel,
            timed=True,
            n_min=bl20.N_MIN_XV,
            theta_in=bl20.THETA_IN_XV_SLOW,
            theta_out=bl20.THETA_OUT_XV_SLOW,
        )
        tf = bl20.book_trade_frame(panel, weights, "fwd_xvenue_paired_return_1", offset)
        costed = bl20.apply_xvenue_costs(tf)
        mechanism, headline = "B", False
    elif variant == "B_single":
        panel = _b_single_panel()
        weights = bl18.build_book_weights(panel, timed=True)
        tf = bl18.book_trade_frame(panel, weights, offset)
        costed = bl18.apply_two_leg_costs(tf)
        mechanism, headline = "B", False
    elif variant == "B_alwayson":
        panel = get_xvenue()[bl20.CARRY_EWMA_HALF_LIFE]
        weights = bl20.build_xvenue_book_weights(panel, timed=False, n_min=1)
        tf = bl20.book_trade_frame(panel, weights, "fwd_xvenue_paired_return_1", offset)
        costed = bl20.apply_xvenue_costs(tf)
        mechanism, headline = "B", False
    elif variant == "B_cash":
        panel = get_xvenue()[bl20.CARRY_EWMA_HALF_LIFE]
        all_times = panel["datetime"].unique(maintain_order=True).sort()
        tf = _cash_frame(all_times, offset)
        costed = tf.with_columns(
            pl.col("trade_log_return").alias("trade_log_return_net")
        )
        mechanism, headline = "B", False
    else:
        raise ValueError(f"unknown variant {variant!r}")

    metrics = bl18.book_metrics(costed, ANNUALIZED_RATE, variant)
    net_returns = costed["trade_log_return_net"].drop_nulls().to_numpy()
    ci_lo, ci_hi = research.block_bootstrap_ci(net_returns)

    skew = float(st.skew(net_returns, nan_policy="omit"))
    kurt = float(st.kurtosis(net_returns, fisher=False, nan_policy="omit"))

    basket, btc = _basket_and_btc_benchmark()
    beta_basket, beta_btc = _compute_beta(costed, basket, btc)
    beta_ok = bool(
        np.isfinite(beta_basket)
        and np.isfinite(beta_btc)
        and abs(beta_basket) < BENCHMARK_BETA_BOUND
        and abs(beta_btc) < BENCHMARK_BETA_BOUND
    )

    concentration = None
    if variant in ("A0", "A1", "A3", "B0"):
        held_col = weights.filter(pl.col("weight") != 0)
        n_held = held_col.group_by("datetime").agg(pl.len().alias("n_held"))
        all_bars = weights.select("datetime").unique()
        n_held = all_bars.join(n_held, on="datetime", how="left").with_columns(
            pl.col("n_held").fill_null(0)
        )
        counts = n_held["n_held"].to_numpy()
        worst = costed.sort("trade_log_return_net").head(5)
        worst_detail = []
        for row in worst.iter_rows(named=True):
            held_syms = weights.filter(
                (pl.col("datetime") == row["datetime"]) & (pl.col("weight") != 0)
            )["symbol"].to_list()
            worst_detail.append(
                {
                    "datetime": str(row["datetime"]),
                    "trade_log_return_net": row["trade_log_return_net"],
                    "symbols_held_at_decision": held_syms,
                }
            )
        concentration = {
            "n_symbols_held_median": float(np.median(counts)),
            "n_symbols_held_mean": float(np.mean(counts)),
            "frac_bars_single_symbol": float(np.mean(counts == 1)),
            "frac_bars_zero_symbols": float(np.mean(counts == 0)),
            "worst_5_bars": worst_detail,
        }

    result: dict[str, Any] = {
        "variant": variant,
        "offset": offset,
        "mechanism": mechanism,
        "headline": headline,
        "metrics": metrics,
        "n_obs": len(net_returns),
        "net_return_skew": skew,
        "net_return_kurtosis_non_excess": kurt,
        "bootstrap_ci_95": [ci_lo, ci_hi],
        "ci_excludes_zero": bool(ci_lo > 0),
        "beta_basket": beta_basket,
        "beta_btc": beta_btc,
        "beta_ok": beta_ok,
        "sharpe_gt_0.5": bool(metrics["sharpe_net"] > FA2_SHARPE_BOUND),
        "concentration_diagnostic": concentration,
    }
    return result, costed


def cmd_run(args: argparse.Namespace) -> None:
    result, costed = build_cell(args.variant, args.offset)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    returns_path = out_path.with_suffix(".returns.parquet")
    costed.select("datetime", "trade_log_return", "trade_log_return_net").write_parquet(
        returns_path
    )
    result["net_return_series_path"] = str(returns_path)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    print(
        f"{args.variant} offset={args.offset}: net Sharpe={result['metrics']['sharpe_net']:.4f} "
        f"n_obs={result['n_obs']} ci=[{result['bootstrap_ci_95'][0]:.2e},{result['bootstrap_ci_95'][1]:.2e}]"
    )


def _dsr_for_headline(
    cells: dict[str, dict[str, Any]], variant: str, offsets: list[int]
) -> dict[str, Any]:
    sharpes = []
    per_offset_returns = {}
    for o in offsets:
        key = f"{variant}_{o}"
        c = cells[key]
        sharpes.append(c["metrics"]["sharpe_net"])
        per_offset_returns[o] = np.array(
            pl.read_parquet(c["net_return_series_path"])[
                "trade_log_return_net"
            ].drop_nulls()
        )
    best_idx = int(np.argmax(sharpes))
    best_offset = offsets[best_idx]
    best_returns = per_offset_returns[best_offset]
    best_sharpe_annualized = sharpes[best_idx]
    best_sharpe_per_period = best_sharpe_annualized / ANNUALIZED_RATE
    skew = float(st.skew(best_returns, nan_policy="omit"))
    kurt = float(st.kurtosis(best_returns, fisher=False, nan_policy="omit"))
    dsr = research.deflated_sharpe_prob(
        best_sharpe_per_period,
        n_trials=N_TRIALS,
        n_obs=len(best_returns),
        skew=skew,
        kurtosis=kurt,
    )
    return {
        "best_offset": best_offset,
        "best_sharpe_net_annualized": best_sharpe_annualized,
        "n_obs": len(best_returns),
        "skew": skew,
        "kurtosis_non_excess": kurt,
        "dsr": dsr,
        "dsr_gt_0.95": bool(dsr > DSR_BOUND),
    }


def cmd_collate(args: argparse.Namespace) -> None:
    cells_dir = Path(args.cells)
    cells: dict[str, dict[str, Any]] = {}
    for path in sorted(cells_dir.glob("*.json")):
        with open(path) as f:
            cell = json.load(f)
        key = f"{cell['variant']}_{cell['offset']}"
        cells[key] = cell

    dsr_a3 = _dsr_for_headline(cells, "A3", [0, 1, 2, 3])
    dsr_b0 = _dsr_for_headline(cells, "B0", [0, 1, 2, 3]) if "B0_0" in cells else None

    # RC gates
    a3_offsets = [cells[f"A3_{o}"] for o in (0, 1, 2, 3)]
    # RC-1 is scored in Phase 3, not recomputed here.
    rc2_sharpe_leg = all(c["sharpe_gt_0.5"] for c in a3_offsets)
    rc2_ci_leg = all(c["ci_excludes_zero"] for c in a3_offsets)
    rc2_dsr_leg = dsr_a3["dsr_gt_0.95"]
    rc2_fires = bool(rc2_sharpe_leg and rc2_ci_leg and rc2_dsr_leg)

    a0 = cells["A0_0"]
    a3_0 = cells["A3_0"]
    a3_returns = pl.read_parquet(a3_0["net_return_series_path"]).rename(
        {"trade_log_return_net": "refined_net"}
    )
    a0_returns = pl.read_parquet(a0["net_return_series_path"]).rename(
        {"trade_log_return_net": "baseline_net"}
    )
    diff_a = a3_returns.select("datetime", "refined_net").join(
        a0_returns.select("datetime", "baseline_net"), on="datetime", how="inner"
    )
    diff_a_vals = (
        (diff_a["refined_net"] - diff_a["baseline_net"]).drop_nulls().to_numpy()
    )
    diff_a_ci = research.block_bootstrap_ci(diff_a_vals)
    rc3_fires = bool(diff_a_ci[0] > 0)

    rc4_fires = all(c["beta_ok"] for c in a3_offsets)

    fund_a_fires = bool(rc2_sharpe_leg and rc2_dsr_leg)

    result: dict[str, Any] = {
        "n_trials_used": N_TRIALS,
        "cells": cells,
        "dsr_A3": dsr_a3,
        "gate_RC1_note": "RC-1 is scored in Phase 3 (phase_3_20_mechanism.json); not recomputed here",
        "gate_RC2": {
            "sharpe_leg": rc2_sharpe_leg,
            "ci_leg": rc2_ci_leg,
            "dsr_leg": rc2_dsr_leg,
            "fires": rc2_fires,
        },
        "gate_RC3": {
            "diff_bootstrap_ci_95": list(diff_a_ci),
            "point_estimate_favours_refined": bool(np.mean(diff_a_vals) > 0),
            "fires": rc3_fires,
        },
        "gate_RC4": {
            "fires": rc4_fires,
            "per_offset_beta_ok": [c["beta_ok"] for c in a3_offsets],
        },
        "gate_FUND_A": {"fires": fund_a_fires},
    }

    if dsr_b0 is not None:
        b0_offsets = [cells[f"B0_{o}"] for o in (0, 1, 2, 3)]
        xd2_sharpe_leg = all(c["sharpe_gt_0.5"] for c in b0_offsets)
        xd2_ci_leg = all(c["ci_excludes_zero"] for c in b0_offsets)
        xd2_dsr_leg = dsr_b0["dsr_gt_0.95"]
        xd2_fires = bool(xd2_sharpe_leg and xd2_ci_leg and xd2_dsr_leg)

        b0_0 = cells["B0_0"]
        b_single = cells["B_single_0"]
        b0_returns = pl.read_parquet(b0_0["net_return_series_path"]).rename(
            {"trade_log_return_net": "xvenue_net"}
        )
        b_single_returns = pl.read_parquet(b_single["net_return_series_path"]).rename(
            {"trade_log_return_net": "single_net"}
        )
        diff_b = b0_returns.select("datetime", "xvenue_net").join(
            b_single_returns.select("datetime", "single_net"),
            on="datetime",
            how="inner",
        )
        diff_b_vals = (
            (diff_b["xvenue_net"] - diff_b["single_net"]).drop_nulls().to_numpy()
        )
        diff_b_ci = research.block_bootstrap_ci(diff_b_vals)
        xd3_fires = bool(diff_b_ci[0] > 0)

        xd4_fires = all(c["beta_ok"] for c in b0_offsets)
        fund_b_fires = bool(xd2_sharpe_leg and xd2_dsr_leg)

        result["dsr_B0"] = dsr_b0
        result["gate_XD1_note"] = (
            "XD-1 is scored in Phase 3 (phase_3_20_mechanism.json); not recomputed here"
        )
        result["gate_XD2"] = {
            "sharpe_leg": xd2_sharpe_leg,
            "ci_leg": xd2_ci_leg,
            "dsr_leg": xd2_dsr_leg,
            "fires": xd2_fires,
        }
        result["gate_XD3"] = {
            "diff_bootstrap_ci_95": list(diff_b_ci),
            "point_estimate_favours_xvenue": bool(np.mean(diff_b_vals) > 0),
            "fires": xd3_fires,
        }
        result["gate_XD4"] = {
            "fires": xd4_fires,
            "per_offset_beta_ok": [c["beta_ok"] for c in b0_offsets],
        }
        result["gate_FUND_B"] = {"fires": fund_b_fires}
        mechanism_b_ran = True
    else:
        mechanism_b_ran = False
        result["mechanism_b_ran"] = False

    result["holdout_access"] = {
        "mechanism_a_unlocked": bool(rc2_fires and rc3_fires),
        "mechanism_b_unlocked": bool(
            mechanism_b_ran
            and result.get("gate_XD2", {}).get("fires")
            and result.get("gate_XD3", {}).get("fires")
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    print(f"Wrote {out_path}")
    print(f"RC-2={rc2_fires} RC-3={rc3_fires} RC-4={rc4_fires} FUND_A={fund_a_fires}")
    if mechanism_b_ran:
        print(
            f"XD-2={result['gate_XD2']['fires']} XD-3={result['gate_XD3']['fires']} "
            f"XD-4={result['gate_XD4']['fires']} FUND_B={result['gate_FUND_B']['fires']}"
        )
    print(f"holdout_access={result['holdout_access']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant")
    parser.add_argument("--offset", type=int, default=0)
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
