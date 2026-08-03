"""Phase 1: the tail atlas (NEXT_PROMPT.md sec 4, Phase 1). Descriptive,
heavily plotted. Per product (16 commodities/ES) plus a BTCUSDT bridge series,
on Phase 0's clean continuous F1 log returns (log_return_ratioadj -- gap-free
across rolls; NOT log_return_backadj, whose additive offset drifts negative
over a long multi-roll history and poisons its log-returns):

- moments, Jarque-Bera, ann vol, max drawdown
- Hill tail index both tails (dist_lib5), alpha_left - alpha_right asymmetry
- ACF/Ljung-Box of |r| and r^2 (vol clustering)
- leverage vs inverse-leverage: corr(r_t, vol_{t+1}) with bootstrap CI
- Samuelson effect: vol by days-to-expiry bucket
- seasonality: month-of-year and day-of-week
- named-event annotation

Writes phase_1_results.json.
"""

import json
import sys
import time
from typing import Any

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import commod_lib8 as C
import dist_lib5 as L5
import numpy as np
import polars as pl
from scipy import stats as st

CURVE_DIR = "src/research/tmp/phase_0_curves"
OUT_PATH = "src/research/tmp/phase_1_results.json"
BTC_PATH = "src/research/cache/BTCUSDT-klines-1d-2021-07-01-2026-07-01.parquet"


def moments_block(ret: np.ndarray) -> dict:
    ret = ret[np.isfinite(ret)]
    jb_stat, jb_p = st.jarque_bera(ret)
    return {
        "n": len(ret),
        "mean": float(np.mean(ret)),
        "sd": float(np.std(ret)),
        "skew": float(st.skew(ret)),
        "excess_kurtosis": float(st.kurtosis(ret)),
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_p": float(jb_p),
        "ann_vol": float(np.std(ret) * np.sqrt(252)),
        "ann_mean": float(np.mean(ret) * 252),
    }


def max_drawdown(ret: np.ndarray) -> float:
    ret = ret[np.isfinite(ret)]
    equity = np.cumsum(ret)  # log-equity
    running_max = np.maximum.accumulate(equity)
    dd = equity - running_max
    return float(np.min(dd))


def hill_block(ret: np.ndarray) -> dict:
    out: dict[str, Any] = {}
    for tail in ["upper", "lower"]:
        path = L5.hill_alpha_path(ret, tail=tail, k_min=20)
        plateau = L5.find_hill_plateau(path["alpha"], path["k"])
        out[tail] = plateau
    alpha_left = (
        out["lower"].get("alpha_median", np.nan)
        if out["lower"].get("found")
        else np.nan
    )
    alpha_right = (
        out["upper"].get("alpha_median", np.nan)
        if out["upper"].get("found")
        else np.nan
    )
    out["alpha_left_minus_right"] = (
        float(alpha_left - alpha_right)
        if np.isfinite(alpha_left) and np.isfinite(alpha_right)
        else None
    )
    return out


def vol_clustering_block(ret: np.ndarray) -> dict:
    ret = ret[np.isfinite(ret)]
    abs_r, sq_r = np.abs(ret), ret**2
    return {
        "acf_abs_r": C.acf(abs_r, 10).tolist(),
        "acf_sq_r": C.acf(sq_r, 10).tolist(),
        "ljung_box_abs_r": C.ljung_box_test(abs_r, lags=10),
        "ljung_box_sq_r": C.ljung_box_test(sq_r, lags=10),
        "ljung_box_r": C.ljung_box_test(ret, lags=10),
    }


def leverage_block(ret: np.ndarray) -> dict:
    ret = ret[np.isfinite(ret)]
    vol_next = np.full(len(ret), np.nan)
    window = 5
    for i in range(window, len(ret)):
        vol_next[i - 1] = np.std(
            ret[i - window + 1 : i + 1]
        )  # forward-looking vol proxy at t+1
    return C.leverage_correlation(ret[:-1], vol_next[:-1])


def process_series(
    dates: list, ret: np.ndarray, dte: np.ndarray | None, product: str
) -> dict:
    block: dict[str, Any] = {}
    block["moments"] = moments_block(ret)
    block["max_drawdown"] = max_drawdown(ret)
    block["hill"] = hill_block(ret)
    block["vol_clustering"] = vol_clustering_block(ret)
    block["leverage"] = leverage_block(ret)
    if dte is not None:
        vol20 = pl.Series(ret).rolling_std(window_size=20).to_numpy()
        block["samuelson"] = C.samuelson_effect(vol20, dte)
    block["month_of_year"] = C.month_of_year_stats(dates, ret)
    block["day_of_week"] = C.day_of_week_stats(dates, ret)
    block["events"] = C.events_in_window(product, str(min(dates)), str(max(dates)))
    return block


def main():
    t0 = time.time()
    results: dict = {}

    for p in C.PRODUCTS:
        curve = pl.read_parquet(f"{CURVE_DIR}/{p}.parquet")
        sub = curve.select(["date", "log_return_ratioadj", "dte_f1"]).drop_nulls(
            subset=["log_return_ratioadj"]
        )
        sub = sub.filter(pl.col("log_return_ratioadj").is_finite())
        dates = sub["date"].to_list()
        ret = sub["log_return_ratioadj"].to_numpy()
        dte = sub["dte_f1"].to_numpy().astype(float)
        if len(ret) < 60:
            print(f"  {p}: too few observations ({len(ret)}), skipping")
            continue
        print(f"processing {p} ({len(ret)} obs)...")
        results[p] = process_series(dates, ret, dte, p)

    print("processing BTCUSDT bridge series...")
    btc = pl.read_parquet(BTC_PATH).sort("datetime")
    btc = btc.with_columns(
        (pl.col("close") / pl.col("close").shift(1)).log().alias("ret")
    )
    btc = btc.drop_nulls(subset=["ret"])
    btc_dates = btc["datetime"].dt.date().to_list()
    btc_ret = btc["ret"].to_numpy()
    results["BTCUSDT"] = process_series(btc_dates, btc_ret, None, "BTCUSDT")

    results["_config"] = {
        "products": C.PRODUCTS,
        "return_series_used": "log_return_ratioadj (gap-free continuous F1; NOT backadj -- additive back-adjustment goes negative over a long multi-roll history, which makes its log-returns nonsense; ratio-adjustment is multiplicative and stays well-behaved)",
        "bridge_series": "BTCUSDT 1d, 2021-07-01 to 2026-07-01",
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwritten {OUT_PATH} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
