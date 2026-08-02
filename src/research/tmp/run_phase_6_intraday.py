"""Phase 6: intraday appendix (NEXT_PROMPT.md sec 4, Phase 6). CL/BZ/HO/RB
1-minute bars, 2026-01-01 to 2026-07-19 only. Descriptive only, no gates, no
backtest, explicitly excluded from every conclusion (six months of data is
not enough for any distributional claim).

- intraday vol seasonality (mean |return| by minute-of-day, ET)
- EIA petroleum status announcement event study (Wed 10:30 ET, CL/HO/RB --
  not gas storage, since NG has no intraday file here per sec 1.6)
- realized-vol signature plot (RV at several sampling frequencies, the
  classic microstructure-noise diagnostic)

Writes phase_6_results.json.
"""

import json
import sys
import time

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import polars as pl

INTRADAY_DIR = "src/research/data/market/databento/intraday"
OUT_PATH = "src/research/tmp/phase_6_results.json"
PRODUCTS = ["CL", "BZ", "HO", "RB"]
EIA_PRODUCTS = ["CL", "HO", "RB"]  # petroleum status report subjects


def load_intraday(product: str) -> pl.DataFrame:
    df = pl.read_parquet(f"{INTRADAY_DIR}/{product}.parquet")
    df = df.sort("timestamp")
    # ET = UTC-5 (winter) / UTC-4 (summer); use polars tz conversion for DST correctness.
    df = df.with_columns(pl.col("timestamp").dt.convert_time_zone("America/New_York").alias("ts_et"))
    df = df.with_columns(
        # dt.hour()/dt.minute() return Int8 -- hour_et*60 overflows Int8's
        # +-127 range (600 for hour=10), silently wrapping and corrupting
        # every downstream minute-of-day computation. Cast to Int32 first.
        pl.col("ts_et").dt.hour().cast(pl.Int32).alias("hour_et"),
        pl.col("ts_et").dt.minute().cast(pl.Int32).alias("minute_et"),
        pl.col("ts_et").dt.weekday().alias("weekday"),  # 1=Mon..7=Sun
        pl.col("ts_et").dt.date().alias("date_et"),
    )
    df = df.with_columns((pl.col("close") / pl.col("close").shift(1)).log().alias("ret"))
    return df.filter(pl.col("ret").is_finite())


def vol_seasonality(df: pl.DataFrame) -> dict:
    agg = (
        df.with_columns((pl.col("hour_et") * 60 + pl.col("minute_et")).alias("minute_of_day"))
        .group_by("minute_of_day")
        .agg(pl.col("ret").abs().mean().alias("mean_abs_ret"), pl.len().alias("n"))
        .sort("minute_of_day")
    )
    return {"minute_of_day": agg["minute_of_day"].to_list(), "mean_abs_ret": agg["mean_abs_ret"].to_list(), "n": agg["n"].to_list()}


def eia_event_study(df: pl.DataFrame, window_min: int = 30) -> dict:
    """Wednesdays 10:30 ET petroleum status report. Compares mean |return| in
    the announcement window against the same time-of-day on non-Wednesdays,
    and against a same-day pre-announcement baseline."""
    ann_minute = 10 * 60 + 30
    df = df.with_columns((pl.col("hour_et") * 60 + pl.col("minute_et")).alias("minute_of_day"))

    in_window = df.filter(
        (pl.col("minute_of_day") >= ann_minute - window_min) & (pl.col("minute_of_day") <= ann_minute + window_min)
    )
    wed_window = in_window.filter(pl.col("weekday") == 3)
    other_window = in_window.filter(pl.col("weekday") != 3)

    wed_mean = float(wed_window["ret"].abs().mean()) if wed_window.height > 0 else None
    other_mean = float(other_window["ret"].abs().mean()) if other_window.height > 0 else None

    # minute-by-minute path around the announcement, Wednesdays only
    path = (
        wed_window.with_columns((pl.col("minute_of_day") - ann_minute).alias("minutes_from_announcement"))
        .group_by("minutes_from_announcement")
        .agg(pl.col("ret").abs().mean().alias("mean_abs_ret"))
        .sort("minutes_from_announcement")
    )
    return {
        "wednesday_window_mean_abs_ret": wed_mean,
        "other_days_window_mean_abs_ret": other_mean,
        "ratio": (wed_mean / other_mean) if (wed_mean and other_mean) else None,
        "n_wednesdays": wed_window["date_et"].n_unique(),
        "path_minutes_from_announcement": path["minutes_from_announcement"].to_list(),
        "path_mean_abs_ret": path["mean_abs_ret"].to_list(),
    }


def realized_vol_signature(df: pl.DataFrame, freqs_min: list[int] | None = None) -> dict:
    """RV at several sampling frequencies (the microstructure-noise
    signature plot): resample close prices to each frequency, compute daily
    RV = sum of squared log returns, report the mean daily RV per frequency.
    """
    if freqs_min is None:
        freqs_min = [1, 5, 15, 30, 60]
    out = {}
    for f in freqs_min:
        sampled = df.with_columns(((pl.col("hour_et") * 60 + pl.col("minute_et")) // f * f).alias("bucket"))
        bars = (
            sampled.group_by(["date_et", "bucket"])
            .agg(pl.col("close").last().alias("close"), pl.col("ts_et").last().alias("ts_et"))
            .sort(["date_et", "ts_et"])
        )
        bars = bars.with_columns((pl.col("close") / pl.col("close").shift(1)).log().over("date_et").alias("ret"))
        daily_rv = bars.group_by("date_et").agg((pl.col("ret") ** 2).sum().alias("rv"))
        out[str(f)] = {"mean_daily_rv": float(daily_rv["rv"].mean()), "n_days": daily_rv.height}
    return out


def main():
    t0 = time.time()
    results: dict = {}
    for p in PRODUCTS:
        print(f"processing {p}...", flush=True)
        df = load_intraday(p)
        entry = {
            "n_bars": df.height,
            "date_range": [str(df["date_et"].min()), str(df["date_et"].max())],
            "vol_seasonality": vol_seasonality(df),
            "realized_vol_signature": realized_vol_signature(df),
        }
        if p in EIA_PRODUCTS:
            entry["eia_petroleum_status_event_study"] = eia_event_study(df)
        results[p] = entry

    results["_scope_note"] = (
        "Six months of 1-minute data (2026-01-01 to 2026-07-19), four energy "
        "products only. Descriptive appendix, excluded from every gate and "
        "every conclusion elsewhere in this notebook."
    )
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwritten {OUT_PATH} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
