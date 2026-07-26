# Standard library
import os
import random
import re
from datetime import datetime, timedelta

# Third-party
import altair
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from tqdm import tqdm

# Aggregations applied per bucket to build OHLC bars from raw trades
OHLC_AGGS = [
    pl.col("price").first().alias("open"),
    pl.col("price").max().alias("high"),
    pl.col("price").min().alias("low"),
    pl.col("price").last().alias("close"),
    pl.col("qty").sum().alias("volume"),
]


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def sharpe_to_annualized_rate(interval: str,
                                trading_days_per_year: int = 365,
                                trading_hours_per_day: float = 24) -> float:
    match = re.match(r"(\d+)([dhms])", interval.lower())
    if not match:
        raise ValueError("Interval must match the format '1d', '2h', '15m', '30s'")
    
    value, unit = int(match.group(1)), match.group(2)
    
    if unit == 'd':
        periods = trading_days_per_year / value
    elif unit == 'h':
        periods = trading_days_per_year * (trading_hours_per_day / value)
    elif unit == 'm':
        periods = trading_days_per_year * (trading_hours_per_day * 60 / value)
    elif unit == 's':
        periods = trading_days_per_year * (trading_hours_per_day * 3600 / value)
    else:
        raise ValueError(f"Unsupported unit: {unit}")
    
    return np.sqrt(periods)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_ohlc_timeseries_range(
    sym: str,
    time_interval: str,
    start_date: datetime,
    end_date: datetime,
    data_path: str | None = None
) -> pl.DataFrame:
    if data_path is None:
        data_path = "./cache"

    if start_date > end_date:
        raise ValueError("start_date must be before or equal to end_date")

    ts_list = []
    total_days = (end_date - start_date).days + 1

    for i in tqdm(range(total_days), desc=f"Loading {sym}", unit="day"):
        current_date = start_date + timedelta(days=i)
        file_name = f"{sym}-trades-{current_date.strftime('%Y-%m-%d')}.parquet"
        file_path = os.path.join(data_path, file_name)

        if not os.path.exists(file_path):
            tqdm.write(f"[WARNING] Missing file: {file_name}")
            continue

        try:
            trades = pl.read_parquet(file_path)

            if "datetime" not in trades.columns:
                raise ValueError(f"Column 'datetime' not found in {file_name}")

            trades = trades.with_columns(pl.col("datetime").cast(pl.Datetime))

            ts = trades.group_by_dynamic("datetime", every=time_interval, offset="0m").agg(OHLC_AGGS)
            ts_list.append(ts)

        except Exception as e:  # noqa: BLE001 - one bad file shouldn't stop the whole range load
            tqdm.write(f"[ERROR] {file_name}: {e}")

    if not ts_list:
        raise ValueError(f"No trade data found for {sym} in range {start_date} to {end_date}")

    result = pl.concat(ts_list).sort("datetime").unique(subset=["datetime"])
    return result


def load_timeseries_range(
    sym: str,
    time_interval: str,
    start_date: datetime,
    end_date: datetime,
    agg_cols: pl.Expr | list[pl.Expr],
    data_path: str | None = None
) -> pl.DataFrame:
    # Default to the same cache dir the downloader writes to
    if data_path is None:
        data_path = "./cache"

    if start_date > end_date:
        raise ValueError("start_date must be before or equal to end_date")

    ts_list = []
    total_days = (end_date - start_date).days + 1

    # One cached parquet file per day, load and bucket each in turn
    for i in tqdm(range(total_days), desc=f"Loading {sym}", unit="day"):
        current_date = start_date + timedelta(days=i)
        file_name = f"{sym}-trades-{current_date.strftime('%Y-%m-%d')}.parquet"
        file_path = os.path.join(data_path, file_name)

        # Skip days that were never downloaded rather than failing the whole range
        if not os.path.exists(file_path):
            tqdm.write(f"[WARNING] Missing file: {file_name}")
            continue

        try:
            trades = pl.read_parquet(file_path)

            if "datetime" not in trades.columns:
                raise ValueError(f"Column 'datetime' not found in {file_name}")

            trades = trades.with_columns(pl.col("datetime").cast(pl.Datetime))

            # Bucket raw trades into time_interval windows using the caller's aggregations
            ts = trades.group_by_dynamic("datetime", every=time_interval, offset="0m").agg(agg_cols)
            ts_list.append(ts)

        except Exception as e:  # noqa: BLE001 - one bad file shouldn't stop the whole range load
            tqdm.write(f"[ERROR] {file_name}: {e}")

    if not ts_list:
        raise ValueError(f"No trade data found for {sym} in range {start_date} to {end_date}")

    # Combine all days and drop any duplicate timestamps from overlapping buckets
    result = pl.concat(ts_list).sort("datetime").unique(subset=["datetime"])
    return result


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_timeseries(
    ts: pl.DataFrame,
    sym: str,
    col: str,
    interval_size: str,
    mode: str = "static",
) -> altair.Chart | None:
    if mode == "static":
        plt.figure(figsize=(12, 6))
        plt.plot(ts["datetime"], ts[col], label=col)
        plt.title(f"{sym} {interval_size} Bars")
        plt.xlabel("time")
        plt.ylabel(col)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        return None

    if mode == "dynamic":
        return altair.Chart(ts).mark_line(tooltip=True).encode(
            x="datetime",
            y=col
        ).properties(
            width=800,
            height=400,
            title=f"{sym} {interval_size} {col}"
        ).configure_scale(zero=False).add_params(
            altair.selection_interval(bind="scales", encodings=["x"]),  # Only zoom x-axis
            altair.selection_interval(bind="scales", encodings=["y"]),  # Only zoom y-axis
        )

    raise ValueError(f"Unsupported mode: {mode!r}. Expected 'static' or 'dynamic'.")

def add_lags(df: pl.DataFrame, col: str, max_no_lags: int, forecast_step: int) -> pl.DataFrame:
    return df.with_columns([pl.col(col).shift(i * forecast_step).alias(f'{col}_lag_{i}') for i in range(1, max_no_lags + 1)])