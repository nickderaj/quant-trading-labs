"""Phase 7: run cfg2_12h (the best Phase 6 config by headline net Sharpe),
completely unchanged, once, on the frozen 2025-07-01 -> 2026-07-01 holdout.

Loads the full 2021-07-01 -> 2026-07-01 panel (allow_holdout=True is only
ever passed here) and reuses panel_walk_forward_splits with the exact same
train_bars/test_bars/origin_offset=0 grid Phase 6 used, so the fold
boundaries are identical to Phase 6's for the pre-holdout period and simply
extend one more year. Only folds whose ENTIRE test window falls at or after
HOLDOUT_START are reported - the holdout is spent once, on exactly those
folds, with no retuning of anything.
"""

import json
import sys
from datetime import UTC, datetime

sys.path.insert(0, "src")

import numpy as np
import polars as pl
from backtest_configs import (
    BARS_PER_DAY,
    CACHE_DIR,
    DOWNLOAD_DIR,
    GROSS_EXPOSURE,
    MAX_POSITION,
    SLIPPAGE,
    START,
    SYMBOLS,
    TAKER_FEE,
    TOP_FRAC,
    TRAIN_TEST_DAYS,
    train_predict_fold,
)

import features
import research

HOLDOUT_END = datetime(2026, 7, 1, tzinfo=UTC)

CFG2_12H_FEATURES = [
    "mean_reversion_1",
    "mean_reversion_4",
    "mean_reversion_12",
    "realized_vol_8",
    "realized_vol_24",
    "realized_vol_96",
    "vol_of_vol_96",
]
INTERVAL = "12h"


def main():
    print(
        "Loading full panel including holdout (2021-07-01 -> 2026-07-01)...", flush=True
    )
    panel = research.load_universe_panel(
        SYMBOLS,
        INTERVAL,
        START,
        HOLDOUT_END,
        min_cross_section=10,
        download_dir=DOWNLOAD_DIR,
        cache_dir=CACHE_DIR,
        allow_holdout=True,
    )
    featured = features.build_feature_panel(panel)
    featured = featured.with_columns(
        research.vol_normalized_target(
            target_col="fwd_return_1", vol_col="realized_vol_24"
        )
    )
    feature_cols = [f"{c}_cs_z" for c in CFG2_12H_FEATURES]
    needed = [
        "datetime",
        "symbol",
        "fwd_return_1",
        "realized_vol_24",
        *feature_cols,
        "fwd_return_1_vol_norm",
    ]
    df = featured.select(needed).drop_nulls().sort(["datetime", "symbol"])
    n_before = len(df)
    df = df.filter(pl.col("realized_vol_24") > 1e-12)
    print(f"dropped {n_before - len(df)} zero-vol rows", flush=True)

    vol_target = research._as_float(
        df.filter(pl.col("datetime") < research.HOLDOUT_START.replace(tzinfo=None))[
            "realized_vol_24"
        ].median()
    )
    annualized_rate = research.sharpe_to_annualized_rate(INTERVAL)
    bars_per_day = BARS_PER_DAY[INTERVAL]
    train_bars = TRAIN_TEST_DAYS[0] * bars_per_day
    test_bars = TRAIN_TEST_DAYS[1] * bars_per_day

    splits = research.panel_walk_forward_splits(
        df, train_bars, test_bars, origin_offset=0
    )
    datetimes = df["datetime"].to_numpy()

    holdout_start_np = np.datetime64(research.HOLDOUT_START.replace(tzinfo=None))
    fold_trade_frames = []
    fold_summaries = []
    for fold_id, (train_idx, test_idx) in enumerate(splits):
        test_datetimes = datetimes[test_idx]
        if test_datetimes.min() < holdout_start_np:
            continue  # this fold's test window is (partly) pre-holdout, not a holdout fold

        train_df = df[train_idx]
        test_df = df[test_idx]
        preds, _, desc = train_predict_fold(
            train_df, test_df, feature_cols, "fwd_return_1_vol_norm"
        )

        test_scored = test_df.with_columns(pl.Series("pred", preds)).with_columns(
            research.vol_targeted_size("pred", "realized_vol_24", vol_target)
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
            label=f"holdout_fold{fold_id}",
        )
        basket_total = float(basket["trade_log_return"].sum())
        strat_total_net = fold_metrics.get(
            "total_log_return_net", fold_metrics["total_log_return"]
        )

        fold_summaries.append(
            {
                "fold": fold_id,
                "test_start": str(test_datetimes.min()),
                "test_end": str(test_datetimes.max()),
                "n_test_bars": len(test_idx),
                "sharpe_net": fold_metrics.get("sharpe_net"),
                "is_degenerate": desc["lm_is_degenerate"],
                "excess_return_net": strat_total_net - basket_total,
                "beats_basket": strat_total_net > basket_total,
            }
        )
        fold_trade_frames.append(
            trade_frame.with_columns(pl.lit(fold_id).alias("fold"))
        )
        print(
            f"holdout fold {fold_id}: {test_datetimes.min()} -> {test_datetimes.max()}, "
            f"sharpe_net={fold_metrics.get('sharpe_net'):.3f}, degenerate={desc['lm_is_degenerate']}",
            flush=True,
        )

    if not fold_trade_frames:
        print("No folds fell entirely within the holdout window - nothing to report.")
        return

    stitched = pl.concat(fold_trade_frames, how="diagonal_relaxed").sort("datetime")
    stitched = stitched.with_columns(
        pl.col("trade_log_return").cum_sum().alias("equity_curve")
    )
    holdout_metrics = research.portfolio_metrics(
        stitched, annualized_rate, taker_fee=TAKER_FEE, slippage=SLIPPAGE
    )

    basket_holdout = research.equal_weight_basket_returns(
        df.filter(pl.col("datetime").is_in(stitched["datetime"].implode()))
    )
    basket_metrics = research._series_metrics(
        basket_holdout["trade_log_return"], annualized_rate, "holdout_basket"
    )

    random_metrics = research.random_dollar_neutral_metrics(
        df.filter(pl.col("datetime").is_in(stitched["datetime"].implode())),
        annualized_rate,
        target_col="fwd_return_1",
        top_frac=TOP_FRAC,
        gross_exposure=GROSS_EXPOSURE,
        max_position_per_symbol=MAX_POSITION,
        taker_fee=TAKER_FEE,
        slippage=SLIPPAGE,
        no_seeds=200,
        seed=0,
    )

    degenerate_frac = float(np.mean([f["is_degenerate"] for f in fold_summaries]))
    win_rate = float(np.mean([f["beats_basket"] for f in fold_summaries]))
    excess_returns = [f["excess_return_net"] for f in fold_summaries]
    ci_lo, ci_hi = research.bootstrap_ci(np.array(excess_returns), n_boot=2000, seed=0)

    result = {
        "config_id": "cfg2_12h_holdout",
        "n_folds": len(fold_summaries),
        "vol_target": vol_target,
        "holdout_metrics": holdout_metrics,
        "basket_metrics": basket_metrics,
        "random_metrics_summary": {
            "mean_sharpe": research._as_float(random_metrics["sharpe"].mean()),
            "p90_sharpe": research._as_float(random_metrics["sharpe"].quantile(0.9)),
        },
        "degenerate_frac": degenerate_frac,
        "win_rate_vs_basket": win_rate,
        "bootstrap_ci_excess_return": [ci_lo, ci_hi],
        "fold_summaries": fold_summaries,
        "n_oos_bars": len(stitched),
    }

    with open("src/research/tmp/holdout_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    log_row = {
        "config_id": "cfg2_12h_holdout",
        "interval": INTERVAL,
        "features": CFG2_12H_FEATURES,
        "phase": "holdout",
        "n_folds": len(fold_summaries),
        "sharpe_net": holdout_metrics.get("sharpe_net"),
        "sharpe_gross": holdout_metrics.get("sharpe"),
        "degenerate_frac": degenerate_frac,
        "win_rate_vs_basket": win_rate,
    }
    with open("src/research/tmp/config_log.jsonl", "a") as f:
        f.write(json.dumps(log_row, default=str) + "\n")

    print("\n" + "=" * 60)
    print("HOLDOUT RESULT (2025-07-01 -> 2026-07-01), cfg2_12h, unchanged")
    print("=" * 60)
    print(f"n_folds={len(fold_summaries)}, n_oos_bars={len(stitched)}")
    print(
        f"sharpe_net={holdout_metrics.get('sharpe_net'):.3f} sharpe_gross={holdout_metrics.get('sharpe'):.3f}"
    )
    print(f"basket buy-hold sharpe={basket_metrics['sharpe']:.3f}")
    print(
        f"random baseline mean/p90 sharpe={result['random_metrics_summary']['mean_sharpe']:.3f}/{result['random_metrics_summary']['p90_sharpe']:.3f}"
    )
    print(f"degenerate_frac={degenerate_frac:.2f} win_rate_vs_basket={win_rate:.2f}")
    print(f"bootstrap 95% CI on excess return: [{ci_lo:.4f}, {ci_hi:.4f}]")


if __name__ == "__main__":
    main()
