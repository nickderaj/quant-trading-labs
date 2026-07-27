# Standard library
import itertools
import os
import random
import re
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Any

# Third-party
import altair
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from torch import nn
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


def sharpe_to_annualized_rate(
    interval: str, trading_days_per_year: int = 365, trading_hours_per_day: float = 24
) -> float:
    match = re.match(r"(\d+)([dhms])", interval.lower())
    if not match:
        raise ValueError("Interval must match the format '1d', '2h', '15m', '30s'")

    value, unit = int(match.group(1)), match.group(2)

    if unit == "d":
        periods = trading_days_per_year / value
    elif unit == "h":
        periods = trading_days_per_year * (trading_hours_per_day / value)
    elif unit == "m":
        periods = trading_days_per_year * (trading_hours_per_day * 60 / value)
    elif unit == "s":
        periods = trading_days_per_year * (trading_hours_per_day * 3600 / value)
    else:
        raise ValueError(f"Unsupported unit: {unit}")

    return np.sqrt(periods)


def log_return_col(col: str) -> str:
    return f"{col}_log_return"


def log_return(col: str, shift_size: int = 1) -> pl.Expr:
    return (
        (pl.col(col) / pl.col(col).shift(shift_size)).log().alias(log_return_col(col))
    )


def lag_cols(col: str, forecast_horizon: int, no_lags: int) -> list[pl.Expr]:
    return [
        pl.col(col).shift(forecast_horizon * i).alias(f"{col}_lag_{i}")
        for i in range(1, no_lags + 1)
    ]


def add_lags(
    df: pl.DataFrame, col: str, max_no_lags: int, forecast_step: int
) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.col(col).shift(i * forecast_step).alias(f"{col}_lag_{i}")
            for i in range(1, max_no_lags + 1)
        ]
    )


def lag_col_names(col: str, n: int) -> list[str]:
    return [f"{col}_lag_{i}" for i in range(1, n + 1)]


def auto_reg_corr_matrx(
    df: pl.DataFrame, target: str, max_no_lags: int
) -> pl.DataFrame:
    return df.drop_nulls().select([target] + lag_col_names(target, max_no_lags)).corr()


def add_log_return_features(
    df: pl.DataFrame,
    col: str,
    forecast_horizon: int,
    max_no_lags: int | None = None,
) -> pl.DataFrame:
    if max_no_lags is None:
        max_no_lags = 0
    df = df.with_columns(log_return(col, forecast_horizon))
    if max_no_lags > 0:
        df = add_lags(df, log_return_col(col), max_no_lags, forecast_horizon)
    return df


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


def _to_tensor(
    x: pl.DataFrame | pl.Series, dtype: torch.dtype | None = None
) -> torch.Tensor:
    return torch.tensor(x.to_numpy(), dtype=torch.float32 if dtype is None else dtype)


def _timeseries_split(
    t: torch.Tensor, test_size: float = 0.25
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (0 < test_size < 1):
        raise ValueError(f"test_size must be between 0 and 1 (got {test_size})")

    split_idx = int(len(t) * (1 - test_size))
    return t[:split_idx], t[split_idx:]


def _to_flat_array(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().numpy()
    return np.asarray(x).reshape(-1)


def _as_float(x: object) -> float:
    assert isinstance(x, (int, float)), f"Expected a numeric scalar, got {type(x)}"
    return float(x)


# --------------------------------------------------------------------------
# Trading evaluation
# --------------------------------------------------------------------------


def model_trade_results(
    y_true: torch.Tensor | np.ndarray, y_pred: torch.Tensor | np.ndarray
) -> pl.DataFrame:
    """Generate trade-level results from model predictions."""
    return (
        pl.DataFrame(
            {
                "y_pred": _to_flat_array(y_pred),
                "y_true": _to_flat_array(y_true),
            }
        )
        .with_columns(
            [
                (pl.col("y_pred").sign() == pl.col("y_true").sign()).alias("is_won"),
                pl.col("y_pred").sign().alias("position"),
            ]
        )
        .with_columns(
            [
                (pl.col("position") * pl.col("y_true")).alias("trade_log_return"),
            ]
        )
        .with_columns(
            [
                pl.col("trade_log_return").cum_sum().alias("equity_curve"),
            ]
        )
        .with_columns(
            [
                (pl.col("equity_curve") - pl.col("equity_curve").cum_max()).alias(
                    "drawdown_log_return"
                ),
            ]
        )
    )


def eval_model_performance(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    feature_names: Sequence[str],
    target_name: str,
    annualized_rate: float,
    log: bool = False,
) -> dict[str, Any]:
    """Calculate performance metrics for the trading model."""
    trade_results = model_trade_results(y_true, y_pred)

    win_rate = _as_float(trade_results["is_won"].mean())
    avg_win = _as_float(
        trade_results.filter(pl.col("is_won"))["trade_log_return"].mean()
    )
    avg_loss = _as_float(
        trade_results.filter(~pl.col("is_won"))["trade_log_return"].mean()
    )
    ev = win_rate * avg_win + (1 - win_rate) * avg_loss

    trade_log_return = trade_results["trade_log_return"]
    std = _as_float(trade_log_return.std())
    sharpe = _as_float(trade_log_return.mean()) / std if std else 0
    total_log_return = _as_float(trade_log_return.sum())

    metrics = {
        "features": ",".join(feature_names),
        "target": target_name,
        "no_trades": len(trade_results),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best_trade": _as_float(trade_log_return.max()),
        "worst_trade": _as_float(trade_log_return.min()),
        "ev": ev,
        "std": std,
        "total_log_return": total_log_return,
        "compound_return": float(np.exp(total_log_return) - 1),
        "max_drawdown": _as_float(trade_results["drawdown_log_return"].min()),
        "equity_trough": _as_float(trade_results["equity_curve"].min()),
        "equity_peak": _as_float(trade_results["equity_curve"].max()),
        "sharpe": float(sharpe * annualized_rate),
    }

    if log:
        print_model_performance(metrics)

    return metrics


def print_model_performance(metrics: dict[str, Any]) -> None:
    """Print eval_model_performance() output as human-readable lines."""
    print(f"Features:         {metrics['features']}")
    print(f"Target:           {metrics['target']}")
    print(f"No. Trades:       {metrics['no_trades']}")
    print(f"Win Rate:         {metrics['win_rate']:.2%}")
    print(f"Average Win:      {metrics['avg_win']:.4f}")
    print(f"Average Loss:     {metrics['avg_loss']:.4f}")
    print(f"Best Trade:       {metrics['best_trade']:.4f}")
    print(f"Worst Trade:      {metrics['worst_trade']:.4f}")
    print(f"Expected Value:   {metrics['ev']:.4f}")
    print(f"Std Dev:          {metrics['std']:.4f}")
    print(f"Total Log Return: {metrics['total_log_return']:.4f}")
    print(f"Compound Return:  {metrics['compound_return']:.2%}")
    print(f"Max Drawdown:     {metrics['max_drawdown']:.4f}")
    print(f"Equity Trough:    {metrics['equity_trough']:.4f}")
    print(f"Equity Peak:      {metrics['equity_peak']:.4f}")
    print(f"Sharpe Ratio:     {metrics['sharpe']:.4f}")
    if "weights" in metrics:
        print(f"Weights:          {metrics['weights']}")
    if "biases" in metrics:
        print(f"Biases:           {metrics['biases']}")


def get_linear_params(model: nn.Module) -> tuple[np.ndarray, float] | None:
    """Extract (weight, bias) from a model's `linear` submodule, if it has one."""
    linear = getattr(model, "linear", None)
    if not isinstance(linear, nn.Linear):
        return None

    weight = linear.weight.detach().cpu().numpy().flatten()
    bias = linear.bias.detach().cpu().numpy().item()
    return weight, bias


def benchmark_model_performance(
    df: pl.DataFrame,
    features: Sequence[str],
    target: str,
    model: nn.Module,
    annualized_rate: float,
    test_size: float = 0.25,
    loss: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    no_epochs: int = 2000,
    lr: float = 5e-4,
    log: bool = False,
) -> dict[str, Any]:
    """Train a model on df and return its eval_model_performance() metrics."""
    x_train, x_test, y_train, y_test = timeseries_train_test_split(
        df, features, target, test_size
    )

    if loss is None:
        loss = nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(no_epochs):
        y_hat_train = model(x_train)
        loss_value = loss(y_hat_train, y_train)

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_hat = model(x_test)

    metrics = eval_model_performance(y_test, y_hat, features, target, annualized_rate)

    linear_params = get_linear_params(model)
    if linear_params is not None:
        weight, bias = linear_params
        metrics["weights"] = str(weight)
        metrics["biases"] = str(bias)

    if log:
        print_model_performance(metrics)

    return metrics


def benchmark_linear_models(
    df: pl.DataFrame,
    target: str,
    feature_pool: Sequence[str],
    annualized_rate: float,
    model_factory: Callable[[int], nn.Module] | None = None,
    max_no_features: int = 1,
    no_epochs: int = 2000,
    lr: float = 5e-4,
    loss: nn.Module | None = None,
    test_size: float = 0.25,
) -> pl.DataFrame:
    """Benchmark model_factory over every combination of up to max_no_features features from feature_pool, sorted by sharpe.

    model_factory defaults to a plain nn.Linear(n, 1) if not given.

    no_epochs defaults to 2000 (was 200) because these features have near-zero
    autocorrelation: at 200 epochs / lr=5e-4 the linear weight barely moves off init,
    so sign(y_pred) is governed by the bias term alone, producing identical trades
    across different feature combos.
    """
    if model_factory is None:
        model_factory = lambda n: nn.Linear(n, 1)  # noqa: E731

    df = df.drop_nulls()

    feature_combos: list[tuple[str, ...]] = []
    for n in range(1, max_no_features + 1):
        feature_combos += list(itertools.combinations(feature_pool, n))

    benchmarks = [
        benchmark_model_performance(
            df,
            list(features),
            target,
            model_factory(len(features)),
            annualized_rate,
            test_size=test_size,
            loss=loss,
            no_epochs=no_epochs,
            lr=lr,
        )
        for features in feature_combos
    ]

    return pl.DataFrame(benchmarks).sort("sharpe", descending=True)


def batch_train_reg(
    model: nn.Module,
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    no_epochs: int,
    loss: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    lr: float | None = None,
    log: bool = False,
) -> torch.Tensor:
    """Train model on (x_train, y_train) full-batch, evaluate on (x_test, y_test), return test predictions."""
    if loss is None:
        loss = nn.L1Loss()
    if lr is None:
        lr = 2e-4
    if optimizer is None:
        # strong_wolfe line search is more stable than plain LBFGS for small regression models
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=1,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
        )

    if log:
        print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")
        print("Model architecture:")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.shape} ({param.numel()} params)")
        print("\nTraining model...")

    train_loss = 0.0
    log_tick_size = max(no_epochs // 10, 1)

    model.train()
    if isinstance(optimizer, torch.optim.LBFGS):
        for epoch in range(no_epochs):

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                loss_value = loss(model(x_train), y_train)
                loss_value.backward()
                return loss_value

            optimizer.step(closure)
            with torch.no_grad():
                train_loss = loss(model(x_train), y_train).item()
            if log and (epoch + 1) % log_tick_size == 0:
                print(f"Epoch [{epoch + 1}/{no_epochs}], Loss: {train_loss:.6f}")
    else:
        for epoch in range(no_epochs):
            optimizer.zero_grad()
            loss_value = loss(model(x_train), y_train)
            loss_value.backward()
            optimizer.step()
            train_loss = loss_value.item()
            if log and (epoch + 1) % log_tick_size == 0:
                print(f"Epoch [{epoch + 1}/{no_epochs}], Loss: {train_loss:.6f}")

    if log:
        print("\nLearned parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}:\n{param.data.numpy()}")

    model.eval()
    with torch.no_grad():
        y_hat = model(x_test)
        if log:
            test_loss = loss(y_hat, y_test)
            print(f"\nTest Loss: {test_loss.item():.6f}, Train Loss: {train_loss:.6f}")

    return y_hat


def learn_model_trades(
    df: pl.DataFrame,
    features: Sequence[str],
    target: str,
    model: nn.Module,
    test_size: float = 0.25,
    loss: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    no_epochs: int = 6000,
    lr: float | None = None,
    log: bool = False,
) -> pl.DataFrame:
    """Train model on df via batch_train_reg and return its model_trade_results()."""
    x_train, x_test, y_train, y_test = timeseries_train_test_split(
        df, features, target, test_size
    )
    y_hat = batch_train_reg(
        model,
        x_train,
        x_test,
        y_train,
        y_test,
        no_epochs,
        loss,
        optimizer,
        lr,
        log,
    )
    return model_trade_results(y_test, y_hat)


def learn_model_trade_pnl(
    df: pl.DataFrame,
    features: Sequence[str],
    target: str,
    model: nn.Module,
    test_size: float = 0.25,
    loss: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    no_epochs: int = 6000,
    lr: float | None = None,
    log: bool = False,
) -> pl.DataFrame:
    return learn_model_trades(
        df, features, target, model, test_size, loss, optimizer, no_epochs, lr, log
    )


def add_tx_fee(trades: pl.DataFrame, tx_fee: float, name: str):
    tx_fee_col = (
        pl.col("exit_trade_value") * tx_fee + pl.col("entry_trade_value") * tx_fee
    ).alias(f"tx_fee_{name}")
    return trades.with_columns(tx_fee_col)


def add_tx_fees(trades: pl.DataFrame, maker_fee: float, taker_fee: float):
    trades = add_tx_fee(trades, maker_fee, "maker")
    trades = add_tx_fee(trades, taker_fee, "taker")
    return trades


def add_tx_fees_log(trades: pl.DataFrame, maker_fee, taker_fee):
    return trades.with_columns(
        (pl.col("trade_log_return") + np.log(maker_fee)).alias(
            "trade_log_return_net_maker"
        ),
        (pl.col("trade_log_return") + np.log(taker_fee)).alias(
            "trade_log_return_net_taker"
        ),
    ).with_columns(
        pl.col("trade_log_return_net_maker").cum_sum().alias("equity_curve_net_maker"),
        pl.col("trade_log_return_net_taker").cum_sum().alias("equity_curve_net_taker"),
    )


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


def load_ohlc_timeseries_range(
    sym: str,
    time_interval: str,
    start_date: datetime,
    end_date: datetime,
    data_path: str | None = None,
) -> pl.DataFrame:
    if data_path is None:
        data_path = "./cache"

    if start_date > end_date:
        raise ValueError("start_date must be before or equal to end_date")

    # Aggregating raw ticks into bars is expensive (minutes per call), so cache
    # the result per symbol/interval/date-range alongside the raw tick cache.
    ohlc_cache_name = (
        f"{sym}-ohlc-{time_interval}-"
        f"{start_date.strftime('%Y-%m-%d')}-{end_date.strftime('%Y-%m-%d')}.parquet"
    )
    ohlc_cache_path = os.path.join(data_path, ohlc_cache_name)

    if os.path.exists(ohlc_cache_path):
        return pl.read_parquet(ohlc_cache_path)

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

            ts = trades.group_by_dynamic(
                "datetime", every=time_interval, offset="0m"
            ).agg(OHLC_AGGS)
            ts_list.append(ts)

        except Exception as e:  # noqa: BLE001 - one bad file shouldn't stop the whole range load
            tqdm.write(f"[ERROR] {file_name}: {e}")

    if not ts_list:
        raise ValueError(
            f"No trade data found for {sym} in range {start_date} to {end_date}"
        )

    result = pl.concat(ts_list).sort("datetime").unique(subset=["datetime"])
    result.write_parquet(ohlc_cache_path)
    return result


def load_timeseries_range(
    sym: str,
    time_interval: str,
    start_date: datetime,
    end_date: datetime,
    agg_cols: pl.Expr | list[pl.Expr],
    data_path: str | None = None,
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
            ts = trades.group_by_dynamic(
                "datetime", every=time_interval, offset="0m"
            ).agg(agg_cols)
            ts_list.append(ts)

        except Exception as e:  # noqa: BLE001 - one bad file shouldn't stop the whole range load
            tqdm.write(f"[ERROR] {file_name}: {e}")

    if not ts_list:
        raise ValueError(
            f"No trade data found for {sym} in range {start_date} to {end_date}"
        )

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
        return (
            altair.Chart(ts)
            .mark_line(tooltip=True)
            .encode(x="datetime", y=col)
            .properties(width=800, height=400, title=f"{sym} {interval_size} {col}")
            .configure_scale(zero=False)
            .add_params(
                altair.selection_interval(
                    bind="scales", encodings=["x"]
                ),  # Only zoom x-axis
                altair.selection_interval(
                    bind="scales", encodings=["y"]
                ),  # Only zoom y-axis
            )
        )

    raise ValueError(f"Unsupported mode: {mode!r}. Expected 'static' or 'dynamic'.")


def plot_distribution(
    data: pl.DataFrame, col: str, label: str | None = None, no_bins: int = 100
) -> altair.Chart:
    return (
        altair.Chart(data)
        .mark_bar()
        .encode(altair.X(f"{col}:Q", bin=altair.Bin(maxbins=no_bins)), y="count()")
        .properties(
            width=600, height=400, title=f"Distribution of {label if label else col}"
        )
        .configure_scale(zero=False)
        .add_params(altair.selection_interval(bind="scales"))
    )


def plot_line(df, col_name):
    chart = df[col_name].plot.line()
    return chart.properties(width=800, height=400, title=col_name)
