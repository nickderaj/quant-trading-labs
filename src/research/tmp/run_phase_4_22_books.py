"""Notebook 022, Phase 4/5: build the three pre-registered book
configurations (HL_ALWAYSON headline, HL_TIMED_FAST, HL_TIMED_SLOW),
the offset robustness grid, the Phase 5 ablations, and score gates
HD-2, HD-4, HD-5, HD-6, HD-7, FUND-HL and the JELLY event study.

12 book configurations total (phase_0_22_preregistration.json's
n_trials_itemisation) -- small enough to run inline, foreground, no
background runner (021's own precedent for a notebook an order of
magnitude cheaper than 020).

Usage: uv run python src/research/tmp/run_phase_4_22_books.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import scipy.stats as st

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "research" / "tmp"))

import basis_lib18 as bl18
import basis_lib22 as bl22

import research

TMP = REPO_ROOT / "src" / "research" / "tmp"
OUT_PATH = TMP / "phase_4_22_results.json"

ANNUALIZED_RATE = research.sharpe_to_annualized_rate(bl22.INTERVAL)
N_TRIALS = 12
BENCHMARK_BETA_BOUND = 0.10
ORIGIN_OFFSETS = [0, 1, 2, 3]

JELLY_WINDOW = (
    datetime(2025, 3, 24, tzinfo=UTC).replace(tzinfo=None),
    datetime(2025, 3, 29, tzinfo=UTC).replace(tzinfo=None),
)  # naive, matching the panel's own naive datetime column (bl22's convention)


def _basket_and_btc_benchmark() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Binance-only equal-weight crypto basket + BTC benchmark, restricted
    to this notebook's own (shorter) dev window -- built directly from
    bl18.load_basis_panel (frozen, import-only), mirroring 020's own
    _basket_and_btc_benchmark exactly.
    """
    panel, _manifest = bl18.load_basis_panel(
        start_date=bl22.DEV_START, end_date=bl22.DEV_END
    )
    featured = bl18.add_trade_features(panel)
    basket = research.equal_weight_basket_returns(
        featured, target_col="fwd_perp_return_1"
    ).rename({"trade_log_return": "trade_log_return_basket"})
    btc = featured.filter(pl.col("symbol") == "BTCUSDT").select(
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
    beta_basket = bl22.ols_beta(
        beta_frame["trade_log_return_net"].to_numpy(),
        beta_frame["trade_log_return_basket"].to_numpy(),
    )
    beta_btc = bl22.ols_beta(
        beta_frame["trade_log_return_net"].to_numpy(),
        beta_frame["btc_return"].to_numpy(),
    )
    return beta_basket, beta_btc


def _score_cell(
    variant: str,
    panel: pl.DataFrame,
    offset: int,
    basket: pl.DataFrame,
    btc: pl.DataFrame,
    *,
    round_turn_bp: float | None = None,
) -> dict[str, Any]:
    weights = bl22.build_hl_book_weights(panel, variant)
    tf = bl22.book_trade_frame(panel, weights, origin_offset=offset)
    costed = (
        bl22.apply_hlvenue_costs(tf)
        if round_turn_bp is None
        else bl22.apply_hlvenue_costs_at(tf, round_turn_bp)
    )
    metrics = bl22.book_metrics(costed, ANNUALIZED_RATE, variant)
    net_returns = costed["trade_log_return_net"].drop_nulls().to_numpy()
    ci_lo, ci_hi = research.block_bootstrap_ci(net_returns)
    beta_basket, beta_btc = _compute_beta(costed, basket, btc)
    beta_ok = bool(
        np.isfinite(beta_basket)
        and np.isfinite(beta_btc)
        and abs(beta_basket) < BENCHMARK_BETA_BOUND
        and abs(beta_btc) < BENCHMARK_BETA_BOUND
    )
    return {
        "variant": variant,
        "offset": offset,
        "round_turn_bp": round_turn_bp
        if round_turn_bp is not None
        else bl22.ROUND_TURN_BP_HL,
        "metrics": metrics,
        "net_ci_95": [ci_lo, ci_hi],
        "net_ci_excludes_zero": bool(ci_lo > 0),
        "beta_basket": beta_basket,
        "beta_btc": beta_btc,
        "beta_ok": beta_ok,
        "n_obs": len(net_returns),
        "weights": weights,
        "costed": costed,
    }


def _dsr_from_cell(cell: dict[str, Any]) -> dict[str, Any]:
    net_returns = cell["costed"]["trade_log_return_net"].drop_nulls().to_numpy()
    sharpe_per_period = cell["metrics"]["sharpe_net"] / ANNUALIZED_RATE
    skew = float(st.skew(net_returns, nan_policy="omit"))
    kurt = float(st.kurtosis(net_returns, fisher=False, nan_policy="omit"))
    dsr = research.deflated_sharpe_prob(
        sharpe_per_period,
        n_trials=N_TRIALS,
        n_obs=len(net_returns),
        skew=skew,
        kurtosis=kurt,
    )
    return {"deflated_sharpe_prob": dsr, "skew": skew, "kurtosis_non_excess": kurt}


def main() -> None:
    symbols = bl22.load_frozen_feed_screened_symbols()
    panel, _manifest = bl22.load_hlvenue_panel(symbols=symbols)
    panel = bl22.add_hlvenue_trade_features(panel)
    basket, btc = _basket_and_btc_benchmark()

    results: dict[str, Any] = {"n_trials": N_TRIALS, "cells": {}}

    # --- Phase 4: headline grid ---------------------------------------
    headline_offsets = {}
    for offset in ORIGIN_OFFSETS:
        cell = _score_cell("HL_ALWAYSON", panel, offset, basket, btc)
        headline_offsets[offset] = cell
        print(
            f"HL_ALWAYSON offset={offset}: sharpe_net={cell['metrics']['sharpe_net']:.3f} "
            f"ci={cell['net_ci_95']} beta_ok={cell['beta_ok']}"
        )

    fast_cell = _score_cell("HL_TIMED_FAST", panel, 0, basket, btc)
    slow_cell = _score_cell("HL_TIMED_SLOW", panel, 0, basket, btc)
    print(
        f"HL_TIMED_FAST offset=0: sharpe_net={fast_cell['metrics']['sharpe_net']:.3f}"
    )
    print(
        f"HL_TIMED_SLOW offset=0: sharpe_net={slow_cell['metrics']['sharpe_net']:.3f}"
    )

    headline0 = headline_offsets[0]
    dsr0 = _dsr_from_cell(headline0)

    # --- Phase 5 ablations (offset 0, HL_ALWAYSON unless noted) -------
    cost_0bp = _score_cell("HL_ALWAYSON", panel, 0, basket, btc, round_turn_bp=0.0)
    cost_19bp = _score_cell("HL_ALWAYSON", panel, 0, basket, btc, round_turn_bp=19.0)
    cost_40bp = _score_cell("HL_ALWAYSON", panel, 0, basket, btc, round_turn_bp=40.0)

    # top-2 contributing symbols, by gross weight*fwd_return contribution
    # on the headline (offset 0) book, computed once, not re-tuned.
    contrib = (
        headline0["weights"]
        .join(
            panel.select("datetime", "symbol", "fwd_hlvenue_paired_return_1"),
            on=["datetime", "symbol"],
            how="inner",
        )
        .with_columns(
            (pl.col("weight") * pl.col("fwd_hlvenue_paired_return_1")).alias("contrib")
        )
        .group_by("symbol")
        .agg(pl.col("contrib").sum().alias("total_contrib"))
        .sort("total_contrib", descending=True)
    )
    top2 = contrib.head(2)["symbol"].to_list()
    panel_ex_top2 = panel.filter(~pl.col("symbol").is_in(top2))
    ex_top2_cell = _score_cell("HL_ALWAYSON", panel_ex_top2, 0, basket, btc)

    btc_only_cell = None
    if "BTCUSDT" in panel["symbol"].unique().to_list():
        btc_only_cell = _score_cell(
            "HL_ALWAYSON", panel.filter(pl.col("symbol") == "BTCUSDT"), 0, basket, btc
        )
    eth_only_cell = None
    if "ETHUSDT" in panel["symbol"].unique().to_list():
        eth_only_cell = _score_cell(
            "HL_ALWAYSON", panel.filter(pl.col("symbol") == "ETHUSDT"), 0, basket, btc
        )

    # --- Gates ----------------------------------------------------------
    sharpes_by_offset = {
        o: headline_offsets[o]["metrics"]["sharpe_net"] for o in ORIGIN_OFFSETS
    }
    hd2_leg_sharpe = all(s > 0.5 for s in sharpes_by_offset.values())
    hd2_leg_ci = headline0["net_ci_excludes_zero"]
    hd2_leg_dsr = dsr0["deflated_sharpe_prob"] > 0.95
    hd2_fires = bool(hd2_leg_sharpe and hd2_leg_ci and hd2_leg_dsr)

    hd4_fires = bool(all(headline_offsets[o]["beta_ok"] for o in ORIGIN_OFFSETS))

    hd5_fires = bool(
        ex_top2_cell["metrics"]["sharpe_net"] > 0
        and ex_top2_cell["net_ci_excludes_zero"]
    )

    hd7_fires = bool(cost_40bp["metrics"]["sharpe_net"] > 0)

    fund_hl_fires = bool(
        sharpes_by_offset[0] > 0.5 and dsr0["deflated_sharpe_prob"] > 0.95
    )

    # --- JELLY event study (descriptive) --------------------------------
    jelly_window_frame = headline0["costed"].filter(
        (pl.col("datetime") >= JELLY_WINDOW[0])
        & (pl.col("datetime") <= JELLY_WINDOW[1])
    )
    all_net = headline0["costed"]["trade_log_return_net"].to_numpy()
    jelly_returns = jelly_window_frame["trade_log_return_net"].to_numpy()
    jelly_report = {
        "window": [str(JELLY_WINDOW[0]), str(JELLY_WINDOW[1])],
        "n_bars_in_window": len(jelly_returns),
        "cumulative_net_log_return_in_window": float(jelly_returns.sum())
        if len(jelly_returns)
        else None,
        "worst_bar_in_window": float(jelly_returns.min())
        if len(jelly_returns)
        else None,
        "worst_bar_in_window_percentile_rank_full_series": (
            float((all_net <= jelly_returns.min()).mean())
            if len(jelly_returns)
            else None
        ),
        "note": "JELLY itself is not Binance-listed and is not in this notebook's universe; this reports the headline book's own realized return through the platform-risk event window as a characterisation, not a JELLY-specific trade outcome.",
    }

    def _cell_summary(cell: dict[str, Any]) -> dict[str, Any]:
        d = {k: v for k, v in cell.items() if k not in ("weights", "costed")}
        return d

    results["cells"] = {
        "HL_ALWAYSON_offsets": {
            str(o): _cell_summary(headline_offsets[o]) for o in ORIGIN_OFFSETS
        },
        "HL_TIMED_FAST": _cell_summary(fast_cell),
        "HL_TIMED_SLOW": _cell_summary(slow_cell),
        "cost_0bp": _cell_summary(cost_0bp),
        "cost_19bp_bare_fee": _cell_summary(cost_19bp),
        "cost_40bp_stress": _cell_summary(cost_40bp),
        "exclude_top2": _cell_summary(ex_top2_cell),
        "btc_only": _cell_summary(btc_only_cell) if btc_only_cell else None,
        "eth_only": _cell_summary(eth_only_cell) if eth_only_cell else None,
    }
    results["top2_contributing_symbols"] = {
        "symbols": top2,
        "contributions": contrib.head(5).to_dicts(),
    }
    results["dsr_headline"] = dsr0
    results["gate_HD2"] = {
        "fires": hd2_fires,
        "leg_sharpe_gt_0.5_all_offsets": hd2_leg_sharpe,
        "leg_ci_excludes_zero": hd2_leg_ci,
        "leg_dsr_gt_0.95": hd2_leg_dsr,
        "sharpes_by_offset": sharpes_by_offset,
        "deflated_sharpe_prob": dsr0["deflated_sharpe_prob"],
    }
    results["gate_HD4"] = {
        "fires": hd4_fires,
        "per_offset_beta_ok": {
            o: headline_offsets[o]["beta_ok"] for o in ORIGIN_OFFSETS
        },
        "per_offset_beta_basket": {
            o: headline_offsets[o]["beta_basket"] for o in ORIGIN_OFFSETS
        },
        "per_offset_beta_btc": {
            o: headline_offsets[o]["beta_btc"] for o in ORIGIN_OFFSETS
        },
    }
    results["gate_HD5"] = {
        "fires": hd5_fires,
        "excluded_symbols": top2,
        "sharpe_net_ex_top2": ex_top2_cell["metrics"]["sharpe_net"],
        "ci_ex_top2": ex_top2_cell["net_ci_95"],
    }
    results["gate_HD6_falsification_check"] = {
        "sharpe_net_alwayson": sharpes_by_offset[0],
        "sharpe_net_timed_fast": fast_cell["metrics"]["sharpe_net"],
        "sharpe_net_timed_slow": slow_cell["metrics"]["sharpe_net"],
        "fast_needed_to_clear_0.5": bool(
            sharpes_by_offset[0] <= 0.5 and fast_cell["metrics"]["sharpe_net"] > 0.5
        ),
    }
    results["gate_HD7"] = {
        "fires": hd7_fires,
        "sharpe_net_at_40bp": cost_40bp["metrics"]["sharpe_net"],
    }
    results["gate_FUND_HL"] = {
        "fires": fund_hl_fires,
        "sharpe_net": sharpes_by_offset[0],
        "deflated_sharpe_prob": dsr0["deflated_sharpe_prob"],
    }
    results["jelly_event_study"] = jelly_report

    holdout_unlocked = bool(hd2_fires and hd4_fires and hd5_fires)
    results["holdout_access"] = {
        "unlocked": holdout_unlocked,
        "policy": "HD-2 AND HD-4 AND HD-5 all fire (phase_0_22_preregistration.json holdout_policy)",
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print()
    print(
        f"HD-2 fires={hd2_fires} HD-4 fires={hd4_fires} HD-5 fires={hd5_fires} "
        f"HD-7 fires={hd7_fires} FUND-HL fires={fund_hl_fires}"
    )
    print(f"holdout unlocked: {holdout_unlocked}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
