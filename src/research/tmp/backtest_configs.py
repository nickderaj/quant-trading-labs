"""Phase 6 backtest: at most 3 pre-declared configs, run once each (plus
origin-shift robustness variants), net of costs. See
src/results/3_cross_sectional_ic.md's "Backtest configs (pre-declared
before running)" section for the declaration this implements.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, "src")

import numpy as np
import polars as pl
import torch
from torch import nn

import data
import features
import research

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "DOTUSDT",
    "AVAXUSDT",
    "MATICUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "ATOMUSDT",
    "UNIUSDT",
    "ETCUSDT",
    "XLMUSDT",
    "ALGOUSDT",
    "VETUSDT",
    "FILUSDT",
    "TRXUSDT",
    "EOSUSDT",
    "AAVEUSDT",
    "SANDUSDT",
    "MANAUSDT",
    "AXSUSDT",
    "THETAUSDT",
    "NEARUSDT",
    "FTMUSDT",
    "LUNAUSDT",
    "FTTUSDT",
]
START = datetime(2021, 7, 1, tzinfo=UTC)
END = research.HOLDOUT_START
CACHE_DIR = "src/research/cache"
DOWNLOAD_DIR = "src/research/tmp"
CONFIG_LOG_PATH = Path("src/research/tmp/config_log.jsonl")

TAKER_FEE = 0.0004
SLIPPAGE = 0.0001
TOP_FRAC = 0.2
GROSS_EXPOSURE = 1.0
MAX_POSITION = 0.25
NO_EPOCHS = 300
LR = 5e-4
ORIGIN_OFFSETS_DAYS = [0, 7, 14, 21]

BASE_FEATURES = [
    "mean_reversion_1",
    "mean_reversion_4",
    "mean_reversion_12",
    "realized_vol_8",
    "realized_vol_24",
    "realized_vol_96",
    "vol_of_vol_96",
]

CONFIGS = [
    {"id": "cfg1_4h", "interval": "4h", "features": BASE_FEATURES + ["funding_rate"]},
    {"id": "cfg2_12h", "interval": "12h", "features": list(BASE_FEATURES)},
    {"id": "cfg3_1d", "interval": "1d", "features": list(BASE_FEATURES)},
]

BARS_PER_DAY = {"4h": 6, "12h": 2, "1d": 1}
TRAIN_TEST_DAYS = (365, 91)  # ~1 year train, ~1 quarter test, rolling


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


def build_featured_panel(
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
    return features.build_feature_panel(panel, funding_by_symbol=funding_by_symbol)


def train_predict_fold(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Train a fresh pooled nn.Linear on train_df, predict test_df.
    Returns (test_predictions, scaled_test_x, bias) for the degenerate-bet check."""
    x_train = torch.tensor(train_df[feature_cols].to_numpy(), dtype=torch.float32)
    y_train = torch.tensor(
        train_df[target_col].to_numpy(), dtype=torch.float32
    ).reshape(-1, 1)
    x_test = torch.tensor(test_df[feature_cols].to_numpy(), dtype=torch.float32)

    mean, std = research._standardize_fit(x_train)
    x_train_scaled = research._standardize_apply(x_train, mean, std)
    x_test_scaled = research._standardize_apply(x_test, mean, std)

    model = nn.Linear(len(feature_cols), 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(NO_EPOCHS):
        y_hat = model(x_train_scaled)
        loss = loss_fn(y_hat, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_hat_test = model(x_test_scaled)

    weight = model.weight.detach().numpy().flatten()
    bias = float(model.bias.detach().numpy().item())
    desc = research.describe_linear_model(
        weight, bias, x_test_scaled.numpy(), feature_cols
    )
    return y_hat_test.numpy().flatten(), x_test_scaled.numpy(), desc


def run_config(config: dict, funding_by_symbol: dict[str, pl.DataFrame]) -> dict:
    interval = config["interval"]
    feature_cols_raw = config["features"]
    feature_cols = [f"{c}_cs_z" for c in feature_cols_raw]
    target_col = "fwd_return_1_vol_norm"
    vol_col = "realized_vol_24"

    print(f"=== {config['id']} ({interval}) ===", flush=True)
    featured = build_featured_panel(interval, funding_by_symbol)
    featured = featured.with_columns(
        research.vol_normalized_target(target_col="fwd_return_1", vol_col=vol_col)
    )
    needed = ["datetime", "symbol", "fwd_return_1", vol_col, *feature_cols, target_col]
    df = featured.select(needed).drop_nulls().sort(["datetime", "symbol"])
    # A handful of bars have exactly-zero realized vol (a frozen/pinned
    # price - e.g. LUNA's last few bars post-collapse, or a thinly-traded
    # symbol with no movement that day). Dividing by zero vol makes the
    # vol-normalized target +-inf, corrupting training; those bars carry no
    # usable vol-normalized signal anyway, so drop them rather than let inf
    # propagate silently.
    n_before = len(df)
    df = df.filter(pl.col(vol_col) > 1e-12)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(
            f"  dropped {n_dropped} zero-realized-vol rows ({n_dropped / n_before:.3%})",
            flush=True,
        )

    vol_target = research._as_float(df[vol_col].median())
    annualized_rate = research.sharpe_to_annualized_rate(interval)
    bars_per_day = BARS_PER_DAY[interval]
    train_bars = TRAIN_TEST_DAYS[0] * bars_per_day
    test_bars = TRAIN_TEST_DAYS[1] * bars_per_day

    origin_results: dict[int, dict[str, Any]] = {}
    for offset_days in ORIGIN_OFFSETS_DAYS:
        offset_bars = offset_days * bars_per_day
        splits = research.panel_walk_forward_splits(
            df, train_bars, test_bars, origin_offset=offset_bars
        )
        fold_trade_frames: list[pl.DataFrame] = []
        fold_summaries: list[dict[str, Any]] = []
        for fold_id, (train_idx, test_idx) in enumerate(splits):
            train_df = df[train_idx]
            test_df = df[test_idx]
            preds, _, desc = train_predict_fold(
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

        stitched = pl.concat(fold_trade_frames, how="diagonal_relaxed").sort("datetime")
        stitched = stitched.with_columns(
            pl.col("trade_log_return").cum_sum().alias("equity_curve")
        )
        stitched_metrics = research.portfolio_metrics(
            stitched, annualized_rate, taker_fee=TAKER_FEE, slippage=SLIPPAGE
        )

        degenerate_frac = (
            float(np.mean([f["is_degenerate"] for f in fold_summaries]))
            if fold_summaries
            else float("nan")
        )
        win_rate = (
            float(np.mean([f["beats_basket"] for f in fold_summaries]))
            if fold_summaries
            else float("nan")
        )

        origin_results[offset_days] = {
            "stitched_metrics": stitched_metrics,
            "fold_summaries": fold_summaries,
            "degenerate_frac": degenerate_frac,
            "win_rate_vs_basket": win_rate,
            "stitched_trade_frame": stitched,
        }

        log_row = {
            "config_id": config["id"],
            "interval": interval,
            "features": feature_cols_raw,
            "origin_offset_days": offset_days,
            "n_folds": len(fold_summaries),
            "sharpe_net": stitched_metrics.get("sharpe_net"),
            "sharpe_gross": stitched_metrics.get("sharpe"),
            "degenerate_frac": degenerate_frac,
            "win_rate_vs_basket": win_rate,
            "turnover_per_year": stitched_metrics.get("turnover_per_year"),
            "annual_fee_drag_pct": stitched_metrics.get("annual_fee_drag_pct"),
        }
        with open(CONFIG_LOG_PATH, "a") as f:
            f.write(json.dumps(log_row, default=str) + "\n")
        print(
            f"  offset={offset_days}d folds={len(fold_summaries)} sharpe_net={stitched_metrics.get('sharpe_net'):.3f} degenerate_frac={degenerate_frac:.2f} win_rate={win_rate:.2f}",
            flush=True,
        )

    # Headline result uses origin_offset=0
    headline = origin_results[0]
    stitched = headline["stitched_trade_frame"]

    oos_datetimes = stitched["datetime"].implode()
    basket_stitched = research.equal_weight_basket_returns(
        df.filter(pl.col("datetime").is_in(oos_datetimes))
    )
    basket_metrics = research._series_metrics(
        basket_stitched["trade_log_return"], annualized_rate, "basket_buy_hold"
    )
    always_short_metrics = research._series_metrics(
        -basket_stitched["trade_log_return"], annualized_rate, "always_short_basket"
    )

    random_metrics = research.random_dollar_neutral_metrics(
        df.filter(pl.col("datetime").is_in(oos_datetimes)),
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

    excess_returns = [f["excess_return_net"] for f in headline["fold_summaries"]]
    ci_lo, ci_hi = research.bootstrap_ci(np.array(excess_returns), n_boot=2000, seed=0)

    return {
        "config": config,
        "vol_target": vol_target,
        "origin_results": {
            k: {kk: vv for kk, vv in v.items() if kk != "stitched_trade_frame"}
            for k, v in origin_results.items()
        },
        "basket_metrics": basket_metrics,
        "always_short_metrics": always_short_metrics,
        "random_metrics_summary": {
            "mean_sharpe": research._as_float(random_metrics["sharpe"].mean()),
            "std_sharpe": research._as_float(random_metrics["sharpe"].std()),
            "p90_sharpe": research._as_float(random_metrics["sharpe"].quantile(0.9)),
        },
        "bootstrap_ci_excess_return": (ci_lo, ci_hi),
        "n_oos_bars_offset0": len(stitched),
    }


def main():
    funding_by_symbol = load_funding_by_symbol()
    results = {}
    for config in CONFIGS:
        results[config["id"]] = run_config(config, funding_by_symbol)

    with open("src/research/tmp/backtest_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    for cfg_id, r in results.items():
        print(f"\n{'=' * 60}\n{cfg_id}\n{'=' * 60}")
        h = r["origin_results"][0]["stitched_metrics"]
        print(
            f"headline sharpe_net={h.get('sharpe_net'):.3f} sharpe_gross={h.get('sharpe'):.3f}"
        )
        print(f"basket buy-hold sharpe={r['basket_metrics']['sharpe']:.3f}")
        print(
            f"random baseline mean sharpe={r['random_metrics_summary']['mean_sharpe']:.3f} p90={r['random_metrics_summary']['p90_sharpe']:.3f}"
        )
        print(f"bootstrap 95% CI on excess return: {r['bootstrap_ci_excess_return']}")
        print(f"win rate vs basket: {r['origin_results'][0]['win_rate_vs_basket']:.2f}")
        print(f"degenerate fold frac: {r['origin_results'][0]['degenerate_frac']:.2f}")


if __name__ == "__main__":
    main()
