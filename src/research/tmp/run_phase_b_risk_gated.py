"""Phase B (checkpoint 2): gate Phase A's best turnover-qualifying variant
(rebalance throttling, k=6 - see src/results/007_alpha_generation.md's Phase A
section) on predicted tail risk from GARCH-NIG, notebook 6's own
best-certified density (Gate P fired at 12h: NIG/Johnson-SU/Hansen skew-t
all beat GARCH-t significantly on 5-6/6 symbols - src/results/006_distribution_zoo.md
Phase 3). The first use of this programme's risk findings as an alpha input
rather than a risk report.

Per section 4's rule, every comparison here is gated vs. IDENTICAL ungated
(the same throttle-k6 weights on the same frozen signal from Phase A), never
vs. the raw notebook-3 baseline - otherwise Phase A's own improvement would
get miscredited to Phase B.

No subagent fan-out here (only Phase A-by-band and Phase C/D-by-symbol are
named for fan-out): a full rolling GARCH-NIG refit for one symbol at 12h
takes ~3 seconds on this hardware (benchmarked directly), so 30 symbols
sequential is ~1.5 minutes - not the "compute-heavy" regime the 3-concurrent-
agent cap exists for.
"""

import json
import sys

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import dist_lib as L
import dist_lib6 as L6
import numpy as np
import polars as pl
from alpha_lib7 import (
    apply_book_scale,
    hysteresis_weights,
    throttle_weights,
    var_gate_standdown,
)
from backtest_configs import (
    GROSS_EXPOSURE,
    MAX_POSITION,
    ORIGIN_OFFSETS_DAYS,
    SLIPPAGE,
    SYMBOLS,
    TAKER_FEE,
    TOP_FRAC,
)
from densities import nig
from run_phase6_application import build_overlay_weight
from run_phase_a_turnover import (
    INTERVAL,
    build_signal_panel,
    fold_excess_returns,
)

import research

Q = 0.01
K_VALUES = [1.25, 1.5, 2.0]  # B1 stand-down thresholds, pre-declared
THROTTLE_K = 6  # Phase A's best turnover-qualifying, all-offset-positive variant


def build_var_forecast_panel() -> pl.DataFrame:
    """Per-symbol causal 1% conditional VaR path from GARCH-NIG at 12h,
    reusing dist_lib6.rolling_garch_forecast_zoo/zoo_quantile_forecast
    unchanged (both already built and tested in notebook 6)."""
    bpd = L6.BARS_PER_DAY[INTERVAL]
    refit_every = L6.MLE_REFIT_DAYS * bpd
    min_train = L6.MIN_TRAIN_DAYS * bpd

    frames = []
    for sym in SYMBOLS:
        try:
            df = L.build_asset_frame(sym, INTERVAL, end=research.HOLDOUT_START)
        except (ValueError, FileNotFoundError):
            print(f"  {sym}: no data, skipped", flush=True)
            continue
        if len(df) < min_train + bpd:
            print(f"  {sym}: too little history ({len(df)} bars), skipped", flush=True)
            continue
        ret = df["log_return"].fill_null(0.0).to_numpy()
        variance_fc, fits = L6.rolling_garch_forecast_zoo(
            ret, refit_every=refit_every, min_train=min_train,
            family_module=nig, max_train=L6.MLE_MAX_TRAIN,
        )
        if not fits:
            print(f"  {sym}: zero successful refits, skipped", flush=True)
            continue
        var_forecast = L6.zoo_quantile_forecast(variance_fc, fits, nig, Q)
        frames.append(
            pl.DataFrame(
                {"datetime": df["datetime"], "symbol": sym, "var_forecast": var_forecast}
            )
        )
        print(f"  {sym}: {len(fits)} refits", flush=True)

    return pl.concat(frames).drop_nulls()


def build_tilt_from_var(var_panel: pl.DataFrame) -> pl.DataFrame:
    """B2 per-symbol tilt: run_phase6_application.py's own build_overlay_weight,
    applied per symbol (reused unchanged, per section 1's 'reuse it, do not
    re-derive it')."""
    out = []
    for sym, g in var_panel.sort("datetime").group_by("symbol", maintain_order=True):
        g = g.sort("datetime")
        tilt = build_overlay_weight(g["var_forecast"].to_numpy())
        out.append(g.select("datetime", "symbol").with_columns(pl.Series("tilt", tilt)))
    return pl.concat(out).sort(["datetime", "symbol"])


def evaluate(name, weights, signal_panel, annualized_rate):
    trade_frame = research.portfolio_trade_frame(weights, signal_panel, target_col="fwd_return_1")
    metrics = research.portfolio_metrics(
        trade_frame, annualized_rate, taker_fee=TAKER_FEE, slippage=SLIPPAGE, label=name
    )
    costed = research.add_portfolio_costs(trade_frame, TAKER_FEE, SLIPPAGE)
    excess = fold_excess_returns(costed, signal_panel)
    ci_lo, ci_hi = research.bootstrap_ci(np.array(excess), n_boot=2000, seed=0)
    return {
        "name": name,
        "sharpe_net": metrics.get("sharpe_net"),
        "sharpe_gross": metrics.get("sharpe"),
        "max_drawdown_net": metrics.get("max_drawdown_net"),
        "mean_turnover_per_bar": metrics.get("mean_turnover_per_bar"),
        "bootstrap_ci_excess_return": [ci_lo, ci_hi],
    }


def main():
    annualized_rate = research.sharpe_to_annualized_rate(INTERVAL)

    print("=== building GARCH-NIG 1% VaR panel (all symbols, once) ===", flush=True)
    var_panel = build_var_forecast_panel()
    tilt_panel = build_tilt_from_var(var_panel)

    results: dict = {"interval": INTERVAL, "throttle_k": THROTTLE_K, "by_offset": {}}

    for offset in ORIGIN_OFFSETS_DAYS:
        print(f"=== offset={offset}d ===", flush=True)
        signal_panel = build_signal_panel(offset)

        # Identical ungated: Phase A's own throttle-k6 baseline weights.
        ungated_w = hysteresis_weights(
            signal_panel, "pred", band=0.0, size_col="vol_targeted_size",
            top_frac=TOP_FRAC, gross_exposure=GROSS_EXPOSURE, max_position_per_symbol=MAX_POSITION,
        )
        ungated_w = throttle_weights(ungated_w, k=THROTTLE_K)
        ungated = evaluate("ungated_throttle_k6", ungated_w, signal_panel, annualized_rate)

        variants = {"ungated_throttle_k6": ungated}

        # B1 stand-down variants
        for k in K_VALUES:
            book_scale = var_gate_standdown(var_panel, k=k)
            gated_w = apply_book_scale(ungated_w, book_scale)
            variants[f"B1_standdown_k{k}"] = evaluate(
                f"B1_standdown_k{k}", gated_w, signal_panel, annualized_rate
            )

        # B2 per-symbol tilt (applied to size BEFORE dollar-neutralizing,
        # then the identical band=0 + throttle-k6 mechanics as the ungated case)
        tilted_signal = signal_panel.join(tilt_panel, on=["datetime", "symbol"], how="left").with_columns(
            pl.col("tilt").fill_null(1.0)
        ).with_columns((pl.col("vol_targeted_size") * pl.col("tilt")).alias("tilted_size"))
        tilted_w = hysteresis_weights(
            tilted_signal, "pred", band=0.0, size_col="tilted_size",
            top_frac=TOP_FRAC, gross_exposure=GROSS_EXPOSURE, max_position_per_symbol=MAX_POSITION,
        )
        tilted_w = throttle_weights(tilted_w, k=THROTTLE_K)
        variants["B2_tilt"] = evaluate("B2_tilt", tilted_w, signal_panel, annualized_rate)

        results["by_offset"][str(offset)] = variants
        for name, v in variants.items():
            print(f"  {name}: net={v['sharpe_net']:.3f} dd={v['max_drawdown_net']:.3f}", flush=True)

    with open("src/research/tmp/phase_b_risk_gated_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwritten phase_b_risk_gated_results.json")


if __name__ == "__main__":
    main()
