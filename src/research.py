# Standard library
import os
import random
import re
from collections.abc import Sequence
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


def add_lags(df: pl.DataFrame, col: str, max_no_lags: int, forecast_step: int) -> pl.DataFrame:
    return df.with_columns([pl.col(col).shift(i * forecast_step).alias(f'{col}_lag_{i}') for i in range(1, max_no_lags + 1)])


# --------------------------------------------------------------------------
# Modeling
# --------------------------------------------------------------------------

def timeseries_train_test_split(
    df: pl.DataFrame,
    features: Sequence[str],
    target: str,
    test_size: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    df = df.drop_nulls()
    x = _to_tensor(df[features])
    y = _to_tensor(df[target]).reshape(-1, 1)
    x_train, x_test = _timeseries_split(x, test_size)
    y_train, y_test = _timeseries_split(y, test_size)
    return x_train, x_test, y_train, y_test


def _to_tensor(x: pl.DataFrame | pl.Series, dtype: torch.dtype | None = None) -> torch.Tensor:
    return torch.tensor(x.to_numpy(), dtype=torch.float32 if dtype is None else dtype)


def _timeseries_split(t: torch.Tensor, test_size: float = 0.25) -> tuple[torch.Tensor, torch.Tensor]:
    if not (0 < test_size < 1):
        raise ValueError(f"test_size must be between 0 and 1 (got {test_size})")

    split_idx = int(len(t) * (1 - test_size))
    return t[:split_idx], t[split_idx:]


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

def plot_distribution(data: pl.DataFrame, col: str, label: str | None = None, no_bins: int = 100) -> altair.Chart:
    return altair.Chart(data).mark_bar().encode(
        altair.X(f'{col}:Q', bin=altair.Bin(maxbins=no_bins)),
        y='count()'
    ).properties(
        width=600,
        height=400,
        title=f'Distribution of {label if label else col}'
    ).configure_scale(zero=False).add_params(
        altair.selection_interval(bind='scales')
)    

def plot_line(df, col_name):
    chart = df[col_name].plot.line()
    return chart.properties(
        width=800,
        height=400,
        title=col_name
)