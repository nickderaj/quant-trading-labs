"""Phase C (checkpoint 3): carry (funding rate) as a PRIMARY signal - a
transparent, single-feature cross-sectional ranking, never a fitted model,
per section 4's own "a positive result can't be a fitting artifact and a
null can't be blamed on an under-trained net" design.

Sign convention, decided BEFORE looking at any Phase C number (a pre-declared
correction, not a post-hoc tune): NEXT_PROMPT.md's own pseudocode passes
`pred_col="funding_rate_zscore_20"` unmodified into dollar_neutral_weights,
which would LONG the highest-funding (payer) symbols - the opposite of its
own stated intent ("short the payers, long the receivers"). Perpetual-futures
funding pays shorts when the rate is positive and longs when it is negative,
so the carry-consistent ranking is pred_col = -funding_rate (long the most
negative/receiver names, short the most positive/payer names). This also
matches notebook 3's own Phase 4 screening result
(src/results/003_cross_sectional_ic.md): raw funding_rate's cross-sectional IC
against forward return is NEGATIVE (-0.0095 at 4h) - i.e. ranking directly on
funding_rate points the wrong way, and the sign must be flipped for the
prediction to align positively with forward return. Both variants (raw
-funding_rate and z-scored -funding_rate_z20) are tested.

No subagent fan-out (only Phase A-by-band and Phase C/D-by-symbol are named
in NEXT_PROMPT.md for it): every symbol's funding history is already cached
locally as a parquet from notebook 3's own download
(src/research/cache/*-funding-*.parquet - verified present for all 30
symbols before this ran), so building the panel is I/O against local disk,
not the CPU-heavy regime the 3-concurrent-agent cap exists for.
"""

import json
import sys

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import numpy as np
import polars as pl
from alpha_lib7 import throttle_weights
from backtest_configs import (
    CACHE_DIR,
    DOWNLOAD_DIR,
    END,
    GROSS_EXPOSURE,
    MAX_POSITION,
    ORIGIN_OFFSETS_DAYS,
    SLIPPAGE,
    START,
    SYMBOLS,
    TAKER_FEE,
    TOP_FRAC,
)

import data
import features
import research

INTERVALS = ["4h", "12h", "1d"]
BARS_PER_DAY = {"4h": 6, "12h": 2, "1d": 1}
PRED_COLS = {"raw": "neg_funding_rate", "zscored": "neg_funding_rate_z20"}


def load_funding_by_symbol() -> dict[str, pl.DataFrame]:
    out = {}
    for sym in SYMBOLS:
        try:
            out[sym] = data.download_funding_rate_range(
                sym, START, END, cache_dir=CACHE_DIR
            )
        except ValueError:
            continue
    return out


def build_carry_panel(
    interval: str, funding_by_symbol: dict[str, pl.DataFrame]
) -> pl.DataFrame:
    panel = research.load_universe_panel(
        SYMBOLS,
        interval,
        START,
        END,
        min_cross_section=10,
        download_dir=DOWNLOAD_DIR,
        cache_dir=CACHE_DIR,
    )
    featured = features.build_feature_panel(panel, funding_by_symbol=funding_by_symbol)
    return featured.with_columns(
        (-pl.col("funding_rate")).alias("neg_funding_rate"),
        (-pl.col("funding_rate_z20")).alias("neg_funding_rate_z20"),
    )


def coverage_stats(panel: pl.DataFrame) -> dict:
    n_total = panel.height
    n_with_funding = panel.filter(pl.col("funding_rate").is_not_null()).height
    symbols_with_funding = (
        panel.filter(pl.col("funding_rate").is_not_null())["symbol"]
        .unique()
        .sort()
        .to_list()
    )
    return {
        "n_total_rows": n_total,
        "n_rows_with_funding": n_with_funding,
        "frac_rows_with_funding": n_with_funding / n_total if n_total else float("nan"),
        "n_symbols_with_funding": len(symbols_with_funding),
        "n_symbols_total": len(SYMBOLS),
        "symbols_with_funding": symbols_with_funding,
    }


def fold_excess_returns(
    trade_frame_net: pl.DataFrame, full_df: pl.DataFrame, splits
) -> list[float]:
    """full_df must be the SAME frame `splits` (panel_walk_forward_splits'
    output) was computed on - test_idx are row positions into it, not into
    whatever subset the caller later builds from it."""
    out = []
    for _train_idx, test_idx in splits:
        fold_panel = full_df[test_idx]
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


def evaluate(name, weights, panel, full_df, annualized_rate, splits):
    trade_frame = research.portfolio_trade_frame(
        weights, panel, target_col="fwd_return_1"
    )
    metrics = research.portfolio_metrics(
        trade_frame, annualized_rate, taker_fee=TAKER_FEE, slippage=SLIPPAGE, label=name
    )
    costed = research.add_portfolio_costs(trade_frame, TAKER_FEE, SLIPPAGE)
    excess = fold_excess_returns(costed, full_df, splits)
    ci_lo, ci_hi = research.bootstrap_ci(np.array(excess), n_boot=2000, seed=0)
    n_configs_trials = len(PRED_COLS) * len(INTERVALS) * len(ORIGIN_OFFSETS_DAYS)
    n_obs = len(trade_frame)
    sharpe_per_period = (
        metrics.get("sharpe_net", float("nan")) / annualized_rate
        if metrics.get("sharpe_net") is not None
        else float("nan")
    )
    dsr = (
        research.deflated_sharpe_prob(sharpe_per_period, n_configs_trials, n_obs)
        if np.isfinite(sharpe_per_period)
        else float("nan")
    )
    return {
        "name": name,
        "sharpe_net": metrics.get("sharpe_net"),
        "sharpe_gross": metrics.get("sharpe"),
        "mean_turnover_per_bar": metrics.get("mean_turnover_per_bar"),
        "turnover_per_year": metrics.get("turnover_per_year"),
        "bootstrap_ci_excess_return": [ci_lo, ci_hi],
        "deflated_sharpe_prob": dsr,
        "net_over_gross_sharpe_ratio": (
            metrics.get("sharpe_net") / metrics.get("sharpe")
            if metrics.get("sharpe") not in (None, 0)
            else None
        ),
    }


def main():
    print("=== loading funding rate (cached, all symbols) ===", flush=True)
    funding_by_symbol = load_funding_by_symbol()

    results: dict = {"intervals": {}}

    for interval in INTERVALS:
        print(f"=== interval={interval} ===", flush=True)
        panel = build_carry_panel(interval, funding_by_symbol)
        cov = coverage_stats(panel)
        print(
            f"  coverage: {cov['n_symbols_with_funding']}/{cov['n_symbols_total']} symbols, "
            f"{cov['frac_rows_with_funding']:.1%} of rows",
            flush=True,
        )

        annualized_rate = research.sharpe_to_annualized_rate(interval)
        bpd = BARS_PER_DAY[interval]
        train_bars = 365 * bpd
        test_bars = 91 * bpd

        interval_results: dict = {"coverage": cov, "by_pred": {}}

        for pred_kind, pred_col in PRED_COLS.items():
            by_offset = {}
            for offset_days in ORIGIN_OFFSETS_DAYS:
                needed = ["datetime", "symbol", "fwd_return_1", pred_col]
                df = panel.select(needed).drop_nulls().sort(["datetime", "symbol"])
                offset_bars = offset_days * bpd
                splits = research.panel_walk_forward_splits(
                    df, train_bars, test_bars, origin_offset=offset_bars
                )
                # Carry needs no per-fold model fit (transparent ranking on
                # the feature itself) - evaluate directly over every test
                # fold's rows, concatenated, exactly like the model-based
                # configs' OOS stitching.
                test_rows = (
                    np.concatenate([idx for _tr, idx in splits])
                    if splits
                    else np.array([], dtype=int)
                )
                test_panel = df[np.unique(test_rows)].sort(["datetime", "symbol"])

                weights = research.dollar_neutral_weights(
                    test_panel,
                    pred_col,
                    top_frac=TOP_FRAC,
                    gross_exposure=GROSS_EXPOSURE,
                    max_position_per_symbol=MAX_POSITION,
                )
                base = evaluate(
                    f"{pred_kind}_{interval}_offset{offset_days}",
                    weights,
                    test_panel,
                    df,
                    annualized_rate,
                    splits,
                )

                # Apply Phase A's own best turnover intervention (throttle
                # k=6) to the best carry variant, per section 4's own
                # instruction that carry should need it least.
                throttled = throttle_weights(weights, k=6)
                throttled_res = evaluate(
                    f"{pred_kind}_{interval}_offset{offset_days}_throttle_k6",
                    throttled,
                    test_panel,
                    df,
                    annualized_rate,
                    splits,
                )

                by_offset[str(offset_days)] = {
                    "base": base,
                    "throttled_k6": throttled_res,
                }
                print(
                    f"  {pred_kind} offset={offset_days}: net={base['sharpe_net']:.3f} "
                    f"turnover/yr={base['turnover_per_year']:.1f} "
                    f"throttled_net={throttled_res['sharpe_net']:.3f}",
                    flush=True,
                )

            interval_results["by_pred"][pred_kind] = by_offset

        results["intervals"][interval] = interval_results

    with open("src/research/tmp/phase_c_carry_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwritten phase_c_carry_results.json")


if __name__ == "__main__":
    main()
