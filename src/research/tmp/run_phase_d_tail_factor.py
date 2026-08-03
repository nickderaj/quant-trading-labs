"""Phase D (checkpoint 4): tail shape as its own cross-sectional ranking
factor, built directly from a rolling Hansen skew-t fit per symbol (nests
the symmetric Student-t exactly at lambda=0, notebook 6's own already-tested
`densities/hansen_skewt.py`).

Interval: 4h, the strongest of the two intervals where notebook 6's Gate P
fired for Hansen skew-t (6/6 symbols significantly beating GARCH-t -
src/results/006_distribution_zoo.md Phase 3) - the same "when the runbook
doesn't pin an interval, use the one the underlying density result is
strongest at" precedent run_phase6_application.py already set explicitly for
Phase 6's own substitution.

IC computed FIRST, before any portfolio is built (per section 4's own
instruction: "if the IC isn't significant, the portfolio result is noise and
should be reported as such rather than backtested into a spurious Sharpe").

No subagent fan-out here either - benchmarked at ~3-5s/symbol for a rolling
GARCH-(Hansen skew-t) two-stage fit at 4h (comparable to Phase B's GARCH-NIG
fit), i.e. ~2-3 minutes for 30 symbols sequential, not the compute-heavy
regime the cap is scoped for.
"""

import json
import sys

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import dist_lib as L
import dist_lib6 as L6
import numpy as np
import polars as pl
from alpha_lib7 import forward_fill_shape_path, zoo_es_forecast
from backtest_configs import (
    GROSS_EXPOSURE,
    MAX_POSITION,
    ORIGIN_OFFSETS_DAYS,
    SLIPPAGE,
    SYMBOLS,
    TAKER_FEE,
    TOP_FRAC,
)
from densities import hansen_skewt

import research

INTERVAL = "4h"
Q = 0.01
NW_LAG = 30  # Hansen skew-t refits every MLE_REFIT_DAYS=30 days - IC autocorrelates out to about that window


def build_tail_shape_panel() -> pl.DataFrame:
    """Per-symbol, per-bar causal (nu, lambda, predicted_ES) from a rolling
    GARCH-(Hansen skew-t) two-stage fit, reusing dist_lib6.rolling_garch_forecast_zoo
    unchanged."""
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
            print(f"  {sym}: too little history, skipped", flush=True)
            continue
        ret = df["log_return"].fill_null(0.0).to_numpy()
        variance_fc, fits = L6.rolling_garch_forecast_zoo(
            ret,
            refit_every=refit_every,
            min_train=min_train,
            family_module=hansen_skewt,
            max_train=L6.MLE_MAX_TRAIN,
        )
        if not fits:
            print(f"  {sym}: zero successful refits, skipped", flush=True)
            continue
        n = len(ret)
        nu_path = forward_fill_shape_path(fits, n, shape_idx=0)
        lam_path = forward_fill_shape_path(fits, n, shape_idx=1)
        es_path = zoo_es_forecast(variance_fc, fits, hansen_skewt, Q)
        frames.append(
            pl.DataFrame(
                {
                    "datetime": df["datetime"],
                    "symbol": sym,
                    "nu": nu_path,
                    "lam": lam_path,
                    "pred_es": es_path,
                    "fwd_return_1": df["log_return"].shift(-1),
                }
            )
        )
        print(
            f"  {sym}: {len(fits)} refits, median nu={np.nanmedian(nu_path):.2f}, "
            f"median lam={np.nanmedian(lam_path):.3f}",
            flush=True,
        )

    panel = pl.concat(frames).drop_nulls(["nu", "lam", "pred_es", "fwd_return_1"])
    return panel.with_columns(
        pl.col("lam").abs().alias("abs_lam"),
        (-pl.col("nu")).alias(
            "neg_nu"
        ),  # D2 wants LONG high-nu (thin), so rank on nu directly (not negated)
        (-pl.col("pred_es")).alias(
            "neg_pred_es"
        ),  # D3 wants LONG low predicted-ES magnitude -> rank on -|ES|
    ).with_columns(
        (-pl.col("abs_lam")).alias("neg_abs_lam")
    )  # D1 wants LONG low |lam| -> rank on -|lam|


FACTORS = {
    "D1_tail_quality": "neg_abs_lam",  # long low |lambda| (symmetric), short high |lambda|
    "D2_tail_premium": "nu",  # long high nu (thin tails), short low nu (fat tails)
    "D3_risk_ranking": "neg_pred_es",  # long low predicted-ES magnitude, short high
}


def fold_excess_returns(trade_frame_net, full_panel, splits) -> list[float]:
    """full_panel must be the SAME frame `splits` was computed on - test_idx
    are row positions into it, not into a later subset."""
    out = []
    for _tr, test_idx in splits:
        fold_panel = full_panel[test_idx]
        test_dates = fold_panel["datetime"].unique().to_list()
        strat_total = float(
            trade_frame_net.filter(pl.col("datetime").is_in(test_dates))[
                "trade_log_return_net"
            ].sum()
        )
        basket = research.equal_weight_basket_returns(fold_panel)
        basket_total = float(basket["trade_log_return"].sum())
        out.append(strat_total - basket_total)
    return out


def main():
    print(
        "=== building tail-shape panel (rolling GARCH-Hansen-skew-t, all symbols) ===",
        flush=True,
    )
    cache_path = "src/research/tmp/phase_d_tail_shape_panel_cache.parquet"
    import os

    if os.path.exists(cache_path):
        panel = pl.read_parquet(cache_path)
        print(f"loaded cached tail-shape panel from {cache_path}", flush=True)
    else:
        panel = build_tail_shape_panel()
        panel.write_parquet(cache_path)
    print(
        f"panel: {panel.height} rows, {panel['symbol'].n_unique()} symbols", flush=True
    )

    annualized_rate = research.sharpe_to_annualized_rate(INTERVAL)
    bpd = L6.BARS_PER_DAY[INTERVAL]
    train_bars = 365 * bpd
    test_bars = 91 * bpd

    results: dict = {
        "interval": INTERVAL,
        "n_symbols": panel["symbol"].n_unique(),
        "factors": {},
    }

    for factor_name, pred_col in FACTORS.items():
        print(f"--- {factor_name} (pred_col={pred_col}) ---", flush=True)
        ic_df = research.cross_sectional_ic(panel, pred_col, "fwd_return_1")
        ic_stats = research.cross_sectional_ic_stats(ic_df, nw_lag=NW_LAG)
        print(
            f"  IC: mean={ic_stats['mean_ic']:.4f} nw_t={ic_stats['nw_tstat']:.2f} "
            f"n_periods={ic_stats['n_periods']}",
            flush=True,
        )

        factor_result: dict = {"ic_stats": ic_stats, "by_offset": {}}
        ic_significant = abs(ic_stats["nw_tstat"]) > 2

        if not ic_significant:
            print(
                "  IC not significant (|t|<=2) - portfolio would be noise; reporting IC only, per section 4.",
                flush=True,
            )
            factor_result["portfolio_skipped_ic_not_significant"] = True
            results["factors"][factor_name] = factor_result
            continue

        for offset_days in ORIGIN_OFFSETS_DAYS:
            offset_bars = offset_days * bpd
            splits = research.panel_walk_forward_splits(
                panel, train_bars, test_bars, origin_offset=offset_bars
            )
            test_rows = (
                np.unique(np.concatenate([idx for _tr, idx in splits]))
                if splits
                else np.array([], dtype=int)
            )
            test_panel = panel[test_rows].sort(["datetime", "symbol"])

            weights = research.dollar_neutral_weights(
                test_panel,
                pred_col,
                top_frac=TOP_FRAC,
                gross_exposure=GROSS_EXPOSURE,
                max_position_per_symbol=MAX_POSITION,
            )
            trade_frame = research.portfolio_trade_frame(
                weights, test_panel, target_col="fwd_return_1"
            )
            metrics = research.portfolio_metrics(
                trade_frame,
                annualized_rate,
                taker_fee=TAKER_FEE,
                slippage=SLIPPAGE,
                label=factor_name,
            )
            costed = research.add_portfolio_costs(trade_frame, TAKER_FEE, SLIPPAGE)
            excess = fold_excess_returns(costed, panel, splits)
            ci_lo, ci_hi = research.bootstrap_ci(np.array(excess), n_boot=2000, seed=0)

            # Single-symbol robustness check (Gate C's "no single-X finding"
            # discipline, extended to symbols): a net-Sharpe-positive result
            # driven by one symbol's idiosyncratic history (e.g. a
            # once-in-the-sample exchange collapse) is not a cross-sectional
            # factor finding, even if it technically clears the gate's raw
            # numeric bar. Drop each symbol once and recompute net Sharpe;
            # flag if the sign flips.
            leg_composition = (
                weights.filter(pl.col("weight") != 0)
                .group_by("symbol")
                .agg(pl.len().alias("n_bars"))
                .sort("n_bars", descending=True)
            )
            top_symbol = leg_composition["symbol"][0] if len(leg_composition) else None
            sharpe_excl_top_symbol = None
            if top_symbol is not None:
                excl_panel = test_panel.filter(pl.col("symbol") != top_symbol)
                excl_weights = research.dollar_neutral_weights(
                    excl_panel,
                    pred_col,
                    top_frac=TOP_FRAC,
                    gross_exposure=GROSS_EXPOSURE,
                    max_position_per_symbol=MAX_POSITION,
                )
                excl_trade_frame = research.portfolio_trade_frame(
                    excl_weights, excl_panel, target_col="fwd_return_1"
                )
                excl_metrics = research.portfolio_metrics(
                    excl_trade_frame,
                    annualized_rate,
                    taker_fee=TAKER_FEE,
                    slippage=SLIPPAGE,
                    label=f"{factor_name}_excl_{top_symbol}",
                )
                sharpe_excl_top_symbol = excl_metrics.get("sharpe_net")

            factor_result["by_offset"][str(offset_days)] = {
                "sharpe_net": metrics.get("sharpe_net"),
                "sharpe_gross": metrics.get("sharpe"),
                "turnover_per_year": metrics.get("turnover_per_year"),
                "bootstrap_ci_excess_return": [ci_lo, ci_hi],
                "top_symbol_by_leg_bars": top_symbol,
                "sharpe_net_excl_top_symbol": sharpe_excl_top_symbol,
                "sign_flips_excl_top_symbol": (
                    (metrics.get("sharpe_net") > 0) != (sharpe_excl_top_symbol > 0)
                    if sharpe_excl_top_symbol is not None
                    and metrics.get("sharpe_net") is not None
                    else None
                ),
            }
            print(
                f"    offset={offset_days}: net={metrics.get('sharpe_net'):.3f} "
                f"gross={metrics.get('sharpe'):.3f} top_symbol={top_symbol} "
                f"net_excl_top={sharpe_excl_top_symbol:.3f} "
                f"sign_flips={factor_result['by_offset'][str(offset_days)]['sign_flips_excl_top_symbol']}",
                flush=True,
            )

        results["factors"][factor_name] = factor_result

    with open("src/research/tmp/phase_d_tail_factor_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwritten phase_d_tail_factor_results.json")


if __name__ == "__main__":
    main()
