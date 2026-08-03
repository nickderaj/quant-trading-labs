"""Task A inference correction for notebook 3 (src/results/003_cross_sectional_ic.md).

`backtest_configs.py`'s `run_config` never called `research.deflated_sharpe_prob`
directly - the 3.4%/0.0000005%/0.15% figures quoted in
`003_cross_sectional_ic.md`'s "Deflated Sharpe" section were computed ad hoc from each
config's offset-0 `stitched_metrics` (sharpe_net, un-annualized, and n_obs) using the
function's normal-distribution defaults (skew=0, kurtosis=3), matching every other
call site's mistake. The per-bar net return series that skew/kurtosis need was never
persisted (`backtest_results.json` strips `stitched_trade_frame` before serializing),
so it isn't recoverable from logged artifacts alone.

This script re-runs each of the 3 pre-declared configs at origin_offset=0 only
(the headline fold grid; the other three offsets in `config_log.jsonl` are
robustness checks, not inputs to the headline DSR/CI numbers) using the exact same
`run_config` machinery as `backtest_configs.py`, with no changes to the model,
features, splits, or cost model - the only new thing computed is the deflated Sharpe
(old normal-assumption vs new real skew/kurtosis) and the excess-return bootstrap CI
(old i.i.d. vs new block).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "src/research/tmp")

import numpy as np
import polars as pl
from backtest_configs import (
    CONFIGS,
    build_featured_panel,
    load_funding_by_symbol,
    train_predict_fold,
)
from scipy import stats

import research

TOP_FRAC = 0.2
GROSS_EXPOSURE = 1.0
MAX_POSITION = 0.25
TAKER_FEE = 0.0004
SLIPPAGE = 0.0001
BARS_PER_DAY = {"4h": 6, "12h": 2, "1d": 1}
TRAIN_TEST_DAYS = (365, 91)

TOTAL_TRIALS = 95  # true count from config_log.jsonl, per 003_cross_sectional_ic.md


def run_offset0(config: dict, funding_by_symbol: dict) -> dict:
    interval = config["interval"]
    feature_cols_raw = config["features"]
    feature_cols = [f"{c}_cs_z" for c in feature_cols_raw]
    target_col = "fwd_return_1_vol_norm"
    vol_col = "realized_vol_24"

    print(f"=== {config['id']} ({interval}) offset=0 ===", flush=True)
    featured = build_featured_panel(interval, funding_by_symbol)
    featured = featured.with_columns(
        research.vol_normalized_target(target_col="fwd_return_1", vol_col=vol_col)
    )
    needed = ["datetime", "symbol", "fwd_return_1", vol_col, *feature_cols, target_col]
    df = featured.select(needed).drop_nulls().sort(["datetime", "symbol"])
    df = df.filter(pl.col(vol_col) > 1e-12)

    vol_target = research._as_float(df[vol_col].median())
    annualized_rate = research.sharpe_to_annualized_rate(interval)
    bars_per_day = BARS_PER_DAY[interval]
    train_bars = TRAIN_TEST_DAYS[0] * bars_per_day
    test_bars = TRAIN_TEST_DAYS[1] * bars_per_day

    splits = research.panel_walk_forward_splits(
        df, train_bars, test_bars, origin_offset=0
    )
    fold_trade_frames = []
    fold_summaries = []
    for fold_id, (train_idx, test_idx) in enumerate(splits):
        train_df = df[train_idx]
        test_df = df[test_idx]
        preds, _, _desc = train_predict_fold(
            train_df, test_df, feature_cols, target_col
        )

        test_scored = test_df.with_columns(pl.Series("pred", preds)).with_columns(
            research.vol_targeted_size("pred", vol_col, vol_target)
        )
        weights = research.dollar_neutral_weights(
            test_scored,
            "pred",
            size_col="vol_targeted_size",
            top_frac=TOP_FRAC,
            gross_exposure=GROSS_EXPOSURE,
            max_position_per_symbol=MAX_POSITION,
        )
        trade_frame = research.portfolio_trade_frame(
            weights, test_scored, target_col="fwd_return_1"
        )
        basket = research.equal_weight_basket_returns(test_scored)

        fold_metrics = research.portfolio_metrics(
            trade_frame,
            annualized_rate,
            taker_fee=TAKER_FEE,
            slippage=SLIPPAGE,
            label=f"fold{fold_id}",
        )
        basket_total = float(basket["trade_log_return"].sum())
        strat_total_net = fold_metrics.get(
            "total_log_return_net", fold_metrics["total_log_return"]
        )
        fold_summaries.append(
            {
                "fold": fold_id,
                "excess_return_net": strat_total_net - basket_total,
            }
        )
        fold_trade_frames.append(
            trade_frame.with_columns(pl.lit(fold_id).alias("fold"))
        )

    stitched = pl.concat(fold_trade_frames, how="diagonal_relaxed").sort("datetime")
    stitched_metrics = research.portfolio_metrics(
        stitched, annualized_rate, taker_fee=TAKER_FEE, slippage=SLIPPAGE
    )

    net_return_col = (
        "trade_log_return_net"
        if "trade_log_return_net" in stitched.columns
        else "trade_log_return"
    )
    per_bar_net_returns = stitched[net_return_col].to_numpy()
    n_obs = len(per_bar_net_returns)
    sharpe_net_annualized = stitched_metrics.get(
        "sharpe_net", stitched_metrics.get("sharpe")
    )
    assert sharpe_net_annualized is not None
    per_period_sharpe = sharpe_net_annualized / annualized_rate

    skew = float(stats.skew(per_bar_net_returns))
    kurt = float(stats.kurtosis(per_bar_net_returns, fisher=False))

    dsr_normal = research.deflated_sharpe_prob(per_period_sharpe, TOTAL_TRIALS, n_obs)
    dsr_real = research.deflated_sharpe_prob(
        per_period_sharpe, TOTAL_TRIALS, n_obs, skew=skew, kurtosis=kurt
    )

    excess_returns = np.array([f["excess_return_net"] for f in fold_summaries])
    iid_ci = research.bootstrap_ci(excess_returns, n_boot=2000, seed=0)
    block_len = research._auto_block_length(excess_returns)
    block_ci = research.block_bootstrap_ci(excess_returns, n_boot=2000, seed=0)

    result = {
        "config_id": config["id"],
        "interval": interval,
        "n_obs": n_obs,
        "n_folds": len(fold_summaries),
        "sharpe_net_annualized": sharpe_net_annualized,
        "per_period_sharpe": per_period_sharpe,
        "skew": skew,
        "kurtosis_fisher_false": kurt,
        "dsr_normal_pct": dsr_normal * 100,
        "dsr_real_moments_pct": dsr_real * 100,
        "bootstrap_ci_iid": list(iid_ci),
        "bootstrap_ci_block": list(block_ci),
        "block_length": block_len,
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    # backtest_configs.py's train_predict_fold trains a fresh, unseeded nn.Linear
    # per fold, and never persists the per-bar stitched return series - so the
    # exact draw behind each config_log.jsonl headline number can't be reproduced,
    # only a fresh one. Seed for reproducibility of *this* script's own output,
    # matching the precedent in build_notebook.py (which also reruns cfg2_12h
    # fresh under a fixed seed rather than replaying the logged run verbatim).
    research.set_seed(123)
    funding_by_symbol = load_funding_by_symbol()
    out = {}
    for config in CONFIGS:
        out[config["id"]] = run_offset0(config, funding_by_symbol)
    Path("src/research/tmp/inference_correction_results.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    print("\nDone. Written to src/research/tmp/inference_correction_results.json")


if __name__ == "__main__":
    main()
