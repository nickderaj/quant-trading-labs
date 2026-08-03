"""Phase 6 (GATED): risk-limit overlay on buy-and-hold BTC, evaluated on the
frozen holdout - the only place in this notebook the holdout is touched, run
once, unchanged, no retuning against the result, per notebook 3's own Phase 7
discipline (src/results/003_cross_sectional_ic.md).

Runs ONLY because Gate D fired (checkpoint 4: Gate P at 4h/12h, GARCH-NIG/
Johnson-SU/Hansen-skew-t all beat GARCH-t significantly on 5-6/6 symbols).

**Pre-declared spec (NEXT_RUN_PROMPT.md section 4 Phase 6), one flagged
adaptation.** The spec says "1-day 1% conditional VaR"; no gate fired at 1d
anywhere in this notebook (1d was the null interval throughout, exactly as
in notebook 5). The interval is substituted to **4h**, where Gate D fired
most robustly (all three winning zoo families at 6/6 symbols including BTC)
- a deliberate, stated adaptation to the gate that actually fired, not a
silent scope change. "Best-certified-density-conditional" (the spec's own
explicit alternative to "EVT-conditional") is **GARCH-NIG**, the
highest-log-score zoo family on BTC at 4h (checkpoint 4:
phase3_zoo_results.json, BTC 4h: NIG 3.278 vs Johnson-SU 3.2756 vs Hansen
3.269).

**Overlay logic** (adapted from "1-day" to the model's own native 4h
horizon): full exposure (weight=1) when the model's own 1% conditional VaR
forecast is within its own trailing 250-bar (~166 day) median magnitude;
scaled down proportionally (weight = trailing_median / |VaR_t|, clipped to
[0,1]) when predicted tail risk exceeds it.

**Benchmarks**: unmodified buy-and-hold, and the identical overlay driven by
GARCH-normal (the spec's own explicit second comparator). All three at
notebook 3's own fee/slippage constants
(`src/research/tmp/backtest_configs.py`: TAKER_FEE=0.0004, SLIPPAGE=0.0001).
"""

import sys

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import dist_lib as L
import dist_lib6 as L6
import numpy as np
import polars as pl
from backtest_configs import SLIPPAGE, TAKER_FEE
from densities import nig

import research

SYMBOL = "BTCUSDT"
INTERVAL = "4h"
Q = 0.01
TRAILING_MEDIAN_WINDOW = 250  # ~166 days at 4h, same order as MLE_MAX_TRAIN


def build_overlay_weight(var_forecast: np.ndarray) -> np.ndarray:
    """Full exposure when |VaR_t| is within its own trailing median; scaled
    down proportionally when it exceeds it. Trailing median uses a causal
    rolling window, shifted so bar t's weight never uses VaR_t itself in its
    own reference level."""
    abs_var = np.abs(var_forecast)
    s = pl.Series("abs_var", abs_var)
    trailing_median = (
        s.rolling_median(
            window_size=TRAILING_MEDIAN_WINDOW, min_samples=TRAILING_MEDIAN_WINDOW // 2
        )
        .shift(1)
        .to_numpy()
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        weight = np.where(abs_var > 0, trailing_median / abs_var, np.nan)
    weight = np.clip(weight, 0.0, 1.0)
    weight[~np.isfinite(weight)] = np.nan
    return weight


def backtest_overlay(
    ret: np.ndarray, weight: np.ndarray, holdout_mask: np.ndarray
) -> dict:
    valid = np.isfinite(weight) & holdout_mask
    w = np.where(np.isfinite(weight), weight, 0.0)
    trade_lr = w * ret
    turnover = np.abs(np.diff(w, prepend=0.0))
    cost_frac = TAKER_FEE + SLIPPAGE
    with np.errstate(invalid="ignore"):
        cost_lr = np.log(np.clip(1 - cost_frac * turnover, 1e-12, None))
    trade_lr_net = trade_lr + cost_lr

    gross = research._series_metrics(
        pl.Series(trade_lr[valid]),
        research.sharpe_to_annualized_rate(INTERVAL),
        "gross",
    )
    net = research._series_metrics(
        pl.Series(trade_lr_net[valid]),
        research.sharpe_to_annualized_rate(INTERVAL),
        "net",
    )
    return {
        "gross_sharpe": gross["sharpe"],
        "gross_total_log_return": gross["total_log_return"],
        "gross_max_drawdown": gross["max_drawdown"],
        "net_sharpe": net["sharpe"],
        "net_total_log_return": net["total_log_return"],
        "net_max_drawdown": net["max_drawdown"],
        "mean_turnover_per_bar": float(np.mean(turnover[valid])),
        "annualized_fee_drag": float(
            np.mean(cost_lr[valid]) * -1 * (365 * 6)
        ),  # 6 bars/day at 4h
        "n_bars": int(valid.sum()),
    }


def main():
    df = L.build_asset_frame(SYMBOL, INTERVAL, end=L.FULL_END)
    n = len(df)
    ret = df["log_return"].fill_null(0.0).to_numpy()
    dates = df["datetime"].to_numpy()
    holdout_mask = dates >= np.datetime64(research.HOLDOUT_START.replace(tzinfo=None))

    print(
        f"n_obs={n}, n_holdout_bars={holdout_mask.sum()}, "
        f"holdout starts {dates[holdout_mask][0] if holdout_mask.any() else 'N/A'}"
    )

    bpd = L6.BARS_PER_DAY[INTERVAL]
    mle_refit_every = L6.MLE_REFIT_DAYS * bpd
    min_train = L6.MIN_TRAIN_DAYS * bpd

    # GARCH-NIG: the best-certified-density-conditional model at this interval
    variance_fc_nig, fits_nig = L6.rolling_garch_forecast_zoo(
        ret,
        refit_every=mle_refit_every,
        min_train=min_train,
        family_module=nig,
        max_train=L6.MLE_MAX_TRAIN,
    )
    var_nig = L6.zoo_quantile_forecast(variance_fc_nig, fits_nig, nig, Q)

    # GARCH-normal: the spec's own explicit second comparator
    variance_fc_norm, _fits_norm = L.rolling_garch_forecast(
        ret,
        refit_every=mle_refit_every,
        min_train=min_train,
        innovation="normal",
        max_train=L6.MLE_MAX_TRAIN,
    )
    import scipy.stats as st

    sigma_norm = np.sqrt(np.where(variance_fc_norm > 0, variance_fc_norm, np.nan))
    var_norm = sigma_norm * st.norm.ppf(Q)

    weight_nig = build_overlay_weight(var_nig)
    weight_norm = build_overlay_weight(var_norm)

    result_nig = backtest_overlay(ret, weight_nig, holdout_mask)
    result_norm = backtest_overlay(ret, weight_norm, holdout_mask)

    bh_lr = ret[holdout_mask]
    bh = research._series_metrics(
        pl.Series(bh_lr), research.sharpe_to_annualized_rate(INTERVAL), "buy_and_hold"
    )

    def _exceedance_count(var_forecast):
        mask = holdout_mask & np.isfinite(var_forecast)
        hits = ret[mask] < var_forecast[mask]
        return int(hits.sum()), int(mask.sum())

    exc_nig, n_nig = _exceedance_count(var_nig)
    exc_norm, n_norm = _exceedance_count(var_norm)

    out = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "level": Q,
        "n_holdout_bars": int(holdout_mask.sum()),
        "buy_and_hold": {
            "sharpe": bh["sharpe"],
            "total_log_return": bh["total_log_return"],
            "compound_return": bh["compound_return"],
            "max_drawdown": bh["max_drawdown"],
        },
        "garch_nig_overlay": {
            **result_nig,
            "n_1pct_exceedances": exc_nig,
            "n_var_valid": n_nig,
            "expected_exceedances": round(0.01 * n_nig, 1),
        },
        "garch_normal_overlay": {
            **result_norm,
            "n_1pct_exceedances": exc_norm,
            "n_var_valid": n_norm,
            "expected_exceedances": round(0.01 * n_norm, 1),
        },
    }

    print("\n--- buy-and-hold (holdout) ---")
    print(out["buy_and_hold"])
    print("\n--- GARCH-NIG overlay (holdout) ---")
    print(out["garch_nig_overlay"])
    print("\n--- GARCH-normal overlay (holdout) ---")
    print(out["garch_normal_overlay"])

    import json

    with open("src/research/tmp/phase6_application_results.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("\nwritten phase6_application_results.json")


if __name__ == "__main__":
    main()
