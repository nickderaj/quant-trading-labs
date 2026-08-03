# Standard library
import itertools
import os
import random
import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

# Third-party
import altair
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from scipy.stats import norm, rankdata
from torch import nn
from tqdm import tqdm

# First-party
import data

# Frozen holdout period (2025-07-01 -> 2026-07-01, see Phase 7). Nothing prior
# to Phase 7 should read past this date; load_universe_panel enforces it.
HOLDOUT_START = datetime(2025, 7, 1, tzinfo=UTC)

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
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    threshold: float = 0.0,
) -> pl.DataFrame:
    """Generate trade-level results from model predictions.

    threshold is a no-trade band on |y_pred|: predictions inside it are too
    small to trust over noise, so position is flat (0) rather than forced to
    sign(y_pred). is_won is null on flat bars so it doesn't distort win_rate.
    """
    in_market = pl.col("y_pred").abs() > threshold
    position = pl.when(in_market).then(pl.col("y_pred").sign()).otherwise(0.0)

    return (
        pl.DataFrame(
            {
                "y_pred": _to_flat_array(y_pred),
                "y_true": _to_flat_array(y_true),
            }
        )
        .with_columns(
            [
                pl.when(in_market)
                .then(pl.col("y_pred").sign() == pl.col("y_true").sign())
                .otherwise(None)
                .alias("is_won"),
                position.alias("position"),
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
    threshold: float = 0.0,
    log: bool = False,
) -> dict[str, Any]:
    """Calculate performance metrics for the trading model."""
    trade_results = model_trade_results(y_true, y_pred, threshold)
    traded = trade_results.filter(pl.col("position") != 0)

    won = traded.filter(pl.col("is_won"))
    lost = traded.filter(~pl.col("is_won"))

    win_rate = _as_float(traded["is_won"].mean()) if len(traded) else 0.0
    # A fold can easily have zero losing (or zero winning) trades, e.g. a
    # near-empty fold or a lopsided one; .mean() on an empty column is None,
    # not 0.0, so guard on length rather than trusting _as_float to coerce it.
    avg_win = _as_float(won["trade_log_return"].mean()) if len(won) else 0.0
    avg_loss = _as_float(lost["trade_log_return"].mean()) if len(lost) else 0.0
    ev = win_rate * avg_win + (1 - win_rate) * avg_loss

    trade_log_return = trade_results["trade_log_return"]
    std = _as_float(trade_log_return.std())
    sharpe = _as_float(trade_log_return.mean()) / std if std else 0
    total_log_return = _as_float(trade_log_return.sum())

    metrics = {
        "features": ",".join(feature_names),
        "target": target_name,
        "no_bars": len(trade_results),
        "no_trades": len(traded),
        "frac_time_in_market": len(traded) / len(trade_results)
        if len(trade_results)
        else 0.0,
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
    print(f"No. Bars:         {metrics['no_bars']}")
    print(f"No. Trades:       {metrics['no_trades']}")
    print(f"Time In Market:   {metrics['frac_time_in_market']:.2%}")
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
    threshold: float = 0.0,
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

    metrics = eval_model_performance(
        y_test, y_hat, features, target, annualized_rate, threshold
    )

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
    threshold: float = 0.0,
) -> pl.DataFrame:
    """Benchmark model_factory over every combination of up to max_no_features features from feature_pool, sorted by sharpe.

    model_factory defaults to a plain nn.Linear(n, 1) if not given.

    no_epochs defaults to 2000 (was 200) because these features have near-zero
    autocorrelation: at 200 epochs / lr=5e-4 the linear weight barely moves off init,
    so sign(y_pred) is governed by the bias term alone, producing identical trades
    across different feature combos.

    threshold sets a no-trade band on |y_pred| (see model_trade_results) so a
    model can sit flat instead of being forced into a bet every bar.
    """
    if model_factory is None:
        model_factory = lambda n: nn.Linear(n, 1)

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
            threshold=threshold,
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
    threshold: float = 0.0,
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
    return model_trade_results(y_test, y_hat, threshold)


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
    threshold: float = 0.0,
    log: bool = False,
) -> pl.DataFrame:
    return learn_model_trades(
        df,
        features,
        target,
        model,
        test_size,
        loss,
        optimizer,
        no_epochs,
        lr,
        threshold,
        log,
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


def add_trading_costs(
    trades: pl.DataFrame, taker_fee: float, slippage: float = 1e-4
) -> pl.DataFrame:
    """Charge turnover-based trading costs: taker_fee + slippage per unit of
    position turned over each bar (generalizes add_tx_fees_log with a
    slippage term, and is the version wired into walk_forward_run /
    stitched_metrics for Phase 0 onward).

    turnover_t = |position_t - position_{t-1}|, with position_{-1} treated as
    0 so the very first entry into a position is charged too. cost_t =
    turnover_t * (taker_fee + slippage) is a fraction of capital, expressed
    as a log-return drag via log(1 - cost_t) to match add_tx_fees_log's
    convention (additive on top of trade_log_return, so it composes under
    cum_sum).

    Adds: turnover, cost_log_return, trade_log_return_net, equity_curve_net,
    drawdown_log_return_net.
    """
    turnover = pl.col("position").diff().fill_null(pl.col("position")).abs()
    cost_frac = taker_fee + slippage
    return (
        trades.with_columns(turnover.alias("turnover"))
        .with_columns(
            (1 - cost_frac * pl.col("turnover")).log().alias("cost_log_return")
        )
        .with_columns(
            (pl.col("trade_log_return") + pl.col("cost_log_return")).alias(
                "trade_log_return_net"
            )
        )
        .with_columns(
            pl.col("trade_log_return_net").cum_sum().alias("equity_curve_net")
        )
        .with_columns(
            (pl.col("equity_curve_net") - pl.col("equity_curve_net").cum_max()).alias(
                "drawdown_log_return_net"
            )
        )
    )


def cost_summary(trades: pl.DataFrame, annualized_rate: float) -> dict[str, Any]:
    """Turnover and fee-drag summary from a trades frame with add_trading_costs applied.

    annualized_rate is sharpe_to_annualized_rate's sqrt(periods_per_year), so
    squaring it recovers periods_per_year for scaling per-bar quantities to
    an annual rate.
    """
    periods_per_year = annualized_rate**2
    mean_turnover = _as_float(trades["turnover"].mean()) if len(trades) else 0.0
    mean_cost = _as_float(trades["cost_log_return"].mean()) if len(trades) else 0.0
    annual_fee_drag_log = -mean_cost * periods_per_year
    return {
        "mean_turnover_per_bar": mean_turnover,
        "turnover_per_year": mean_turnover * periods_per_year,
        "annual_fee_drag_log": annual_fee_drag_log,
        "annual_fee_drag_pct": float(np.expm1(annual_fee_drag_log)),
    }


def add_tx_fees_log(trades: pl.DataFrame, maker_fee, taker_fee):
    """Charge fees only on the position actually turned over each bar.

    Flat -> +-1 turns over 1 unit (one side). -1 -> +1 turns over 2 units
    (close + open), i.e. a full round trip. A held position (no change)
    turns over 0 and pays nothing. fee is a fraction per unit turned over,
    so cost is log(1 - fee * turnover), not log(fee).
    """
    turnover = pl.col("position").diff().fill_null(pl.col("position")).abs()
    return (
        trades.with_columns(
            (1 - maker_fee * turnover).log().alias("tx_fee_log_maker"),
            (1 - taker_fee * turnover).log().alias("tx_fee_log_taker"),
        )
        .with_columns(
            (pl.col("trade_log_return") + pl.col("tx_fee_log_maker")).alias(
                "trade_log_return_net_maker"
            ),
            (pl.col("trade_log_return") + pl.col("tx_fee_log_taker")).alias(
                "trade_log_return_net_taker"
            ),
        )
        .with_columns(
            pl.col("trade_log_return_net_maker")
            .cum_sum()
            .alias("equity_curve_net_maker"),
            pl.col("trade_log_return_net_taker")
            .cum_sum()
            .alias("equity_curve_net_taker"),
        )
    )


# --------------------------------------------------------------------------
# Walk-forward validation
# --------------------------------------------------------------------------


def walk_forward_splits(
    n: int,
    train_bars: int,
    test_bars: int,
    step_bars: int | None = None,
    mode: str = "rolling",
    embargo_bars: int = 0,
    origin_offset: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate walk-forward (train_idx, test_idx) index pairs over a length-n series.

    mode="rolling": train window has fixed length train_bars and slides forward.
    mode="anchored": train window always starts at 0 and grows (expanding window).

    embargo_bars is a gap dropped between train end and test start, so a
    lagged/rolling feature computed at the start of a test fold can't reach
    back across the boundary into train data.

    origin_offset shifts where the first fold's train window begins, without
    changing train_bars/test_bars/step_bars. Re-running with different
    origin_offset values checks whether a result depends on where the
    walk-forward grid happens to be anchored in time.

    step_bars defaults to test_bars (non-overlapping test folds).
    """
    if step_bars is None:
        step_bars = test_bars

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    if mode == "rolling":
        train_start = origin_offset
        while True:
            train_end = train_start + train_bars
            test_start = train_end + embargo_bars
            test_end = test_start + test_bars
            if test_end > n:
                break
            splits.append(
                (np.arange(train_start, train_end), np.arange(test_start, test_end))
            )
            train_start += step_bars
    elif mode == "anchored":
        train_end = origin_offset + train_bars
        while True:
            test_start = train_end + embargo_bars
            test_end = test_start + test_bars
            if test_end > n:
                break
            splits.append((np.arange(0, train_end), np.arange(test_start, test_end)))
            train_end += step_bars
    else:
        raise ValueError(
            f"Unsupported mode: {mode!r}. Expected 'rolling' or 'anchored'."
        )

    return splits


def _standardize_fit(x_train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-feature mean/std from a train fold, for standardizing train and test alike.

    Fit on train only: using test-fold statistics would leak information from
    the future into a model that's supposed to be blind to it.
    """
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-12, torch.ones_like(std), std)
    return mean, std


def _standardize_apply(
    x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    return (x - mean) / std


def describe_linear_model(
    weight: np.ndarray,
    bias: float,
    x_test_scaled: np.ndarray,
    feature_names: Sequence[str],
) -> dict[str, Any]:
    """Tell a real fitted signal apart from a constant directional bet.

    linear_contrib = x @ weight varies bar to bar; bias is fixed. If |bias|
    exceeds the max magnitude linear_contrib reaches anywhere in the test
    fold, no observed feature value can flip sign(y_pred): the position is
    the same on every bar regardless of x. That's a constant long/short bet,
    not a fitted relationship, however good its Sharpe looks.

    weight/bias are expected in standardized-feature space (see
    _standardize_fit) so their scales are comparable to x_test_scaled and to
    each other.
    """
    linear_contrib = x_test_scaled @ weight
    max_abs_contrib = (
        float(np.abs(linear_contrib).max()) if len(linear_contrib) else 0.0
    )
    is_degenerate = abs(bias) > max_abs_contrib

    y_pred = linear_contrib + bias
    sign = np.sign(y_pred)
    frac_long = float((sign > 0).mean()) if len(sign) else 0.0
    frac_short = float((sign < 0).mean()) if len(sign) else 0.0
    frac_flat = float((sign == 0).mean()) if len(sign) else 0.0
    no_sign_flips = int((np.diff(sign) != 0).sum()) if len(sign) > 1 else 0

    if is_degenerate:
        direction = "SHORT" if bias < 0 else "LONG"
        verdict = (
            f"constant {direction} (|bias|={abs(bias):.4g} > "
            f"max|w.x|={max_abs_contrib:.4g}, {no_sign_flips} sign flips)"
        )
    else:
        verdict = (
            f"responsive (|bias|={abs(bias):.4g} <= "
            f"max|w.x|={max_abs_contrib:.4g}, {no_sign_flips} sign flips)"
        )

    return {
        "lm_weights": str(weight),
        "lm_bias": float(bias),
        "lm_max_abs_linear_contrib": max_abs_contrib,
        "lm_is_degenerate": is_degenerate,
        "lm_frac_long": frac_long,
        "lm_frac_short": frac_short,
        "lm_frac_flat": frac_flat,
        "lm_no_sign_flips": no_sign_flips,
        "lm_verdict": verdict,
    }


def walk_forward_run(
    df: pl.DataFrame,
    features: Sequence[str],
    target: str,
    model_factory: Callable[[int], nn.Module],
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    annualized_rate: float,
    loss: nn.Module | None = None,
    lr: float = 5e-4,
    no_epochs: int = 2000,
    threshold: float = 0.0,
    taker_fee: float | None = None,
    slippage: float = 1e-4,
) -> dict[str, pl.DataFrame]:
    """Run one walk-forward configuration end to end.

    Trains a fresh model_factory() instance on each fold's train window
    (features standardized on that fold's train data only, then applied to
    its test data), predicts the fold's test window, and stitches every
    fold's out-of-sample predictions into one continuous series so the whole
    run can be treated like a single long backtest.

    taker_fee, if given, charges add_trading_costs on both the per-fold and
    stitched trade frames, so every returned frame carries gross (columns
    from model_trade_results) and net (trade_log_return_net,
    equity_curve_net, drawdown_log_return_net, turnover) variants side by
    side. Leave None to skip cost accounting (gross only).

    Returns {"folds": per-fold metrics/weights/verdict, "stitched_trades":
    the concatenated OOS trade frame with equity/drawdown recomputed across
    fold boundaries}.
    """
    df = df.drop_nulls()
    if loss is None:
        loss = nn.MSELoss()

    x_all = _to_tensor(df[list(features)])
    y_all = _to_tensor(df[target]).reshape(-1, 1)
    datetimes = df["datetime"].to_numpy() if "datetime" in df.columns else None

    fold_records: list[dict[str, Any]] = []
    stitched: list[pl.DataFrame] = []

    for fold_id, (train_idx, test_idx) in enumerate(splits):
        x_train_raw, x_test_raw = x_all[train_idx], x_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        mean, std = _standardize_fit(x_train_raw)
        x_train = _standardize_apply(x_train_raw, mean, std)
        x_test = _standardize_apply(x_test_raw, mean, std)

        model = model_factory(len(features))
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
            y_hat_test = model(x_test)

        fold_metrics = eval_model_performance(
            y_test, y_hat_test, features, target, annualized_rate, threshold
        )
        fold_metrics["fold"] = fold_id
        fold_metrics["no_train_bars"] = len(train_idx)
        fold_metrics["test_start_idx"] = int(test_idx[0])
        fold_metrics["test_end_idx"] = int(test_idx[-1])
        if datetimes is not None:
            fold_metrics["test_start_date"] = str(datetimes[test_idx[0]])
            fold_metrics["test_end_date"] = str(datetimes[test_idx[-1]])

        linear_params = get_linear_params(model)
        if linear_params is not None:
            weight, bias = linear_params
            fold_metrics.update(
                describe_linear_model(weight, bias, x_test.numpy(), features)
            )

        fold_trades = model_trade_results(y_test, y_hat_test, threshold).with_columns(
            pl.lit(fold_id).alias("fold")
        )
        if datetimes is not None:
            fold_trades = fold_trades.with_columns(
                pl.Series("datetime", datetimes[test_idx])
            )

        if taker_fee is not None:
            fold_trades = add_trading_costs(fold_trades, taker_fee, slippage)
            net_fold_metrics = _series_metrics(
                fold_trades["trade_log_return_net"], annualized_rate, "fold_net"
            )
            fold_metrics["sharpe_net"] = net_fold_metrics["sharpe"]
            fold_metrics["total_log_return_net"] = net_fold_metrics["total_log_return"]
            fold_metrics["compound_return_net"] = net_fold_metrics["compound_return"]
            fold_metrics["max_drawdown_net"] = net_fold_metrics["max_drawdown"]
            fold_metrics.update(cost_summary(fold_trades, annualized_rate))

        fold_records.append(fold_metrics)
        stitched.append(fold_trades)

    stitched_df = (
        pl.concat(stitched, how="diagonal_relaxed") if stitched else pl.DataFrame()
    )
    if len(stitched_df):
        # model_trade_results computed equity/drawdown per fold in isolation;
        # redo both across the stitched series so they reflect one continuous
        # out-of-sample run rather than resetting to zero at each fold.
        stitched_df = stitched_df.with_columns(
            pl.col("trade_log_return").cum_sum().alias("equity_curve")
        ).with_columns(
            (pl.col("equity_curve") - pl.col("equity_curve").cum_max()).alias(
                "drawdown_log_return"
            )
        )
        if taker_fee is not None:
            # Recompute cost/net columns on the full stitched series (not
            # just concatenated per-fold ones) so equity_curve_net is one
            # continuous net-of-cost curve across fold boundaries, matching
            # how the gross equity_curve above is stitched.
            stitched_df = add_trading_costs(
                stitched_df.drop(
                    "turnover",
                    "cost_log_return",
                    "trade_log_return_net",
                    "equity_curve_net",
                    "drawdown_log_return_net",
                ),
                taker_fee,
                slippage,
            )

    return {
        "folds": pl.DataFrame(fold_records) if fold_records else pl.DataFrame(),
        "stitched_trades": stitched_df,
    }


def stitched_metrics(
    stitched_trades: pl.DataFrame, annualized_rate: float, label: str = "strategy"
) -> dict[str, Any]:
    """Summary metrics for a walk_forward_run's stitched OOS trade series.

    If stitched_trades carries add_trading_costs' net columns (i.e.
    walk_forward_run was called with taker_fee), also reports net sharpe/
    return/drawdown alongside the gross ones, plus turnover and annualized
    fee drag from cost_summary.
    """
    if len(stitched_trades) == 0:
        return {"label": label, "no_bars": 0}
    traded = stitched_trades.filter(pl.col("position") != 0)
    metrics = _series_metrics(
        stitched_trades["trade_log_return"], annualized_rate, label
    )
    metrics["no_trades"] = len(traded)
    metrics["frac_time_in_market"] = (
        len(traded) / len(stitched_trades) if len(stitched_trades) else 0.0
    )
    metrics["win_rate"] = _as_float(traded["is_won"].mean()) if len(traded) else 0.0

    if "trade_log_return_net" in stitched_trades.columns:
        net_metrics = _series_metrics(
            stitched_trades["trade_log_return_net"], annualized_rate, f"{label}_net"
        )
        metrics["sharpe_net"] = net_metrics["sharpe"]
        metrics["total_log_return_net"] = net_metrics["total_log_return"]
        metrics["compound_return_net"] = net_metrics["compound_return"]
        metrics["max_drawdown_net"] = net_metrics["max_drawdown"]
        metrics.update(cost_summary(stitched_trades, annualized_rate))

    return metrics


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


def _series_metrics(
    trade_log_return: pl.Series, annualized_rate: float, label: str
) -> dict[str, Any]:
    std = _as_float(trade_log_return.std()) if len(trade_log_return) else 0.0
    mean = _as_float(trade_log_return.mean()) if len(trade_log_return) else 0.0
    total = _as_float(trade_log_return.sum()) if len(trade_log_return) else 0.0
    cum = trade_log_return.cum_sum()
    dd = cum - cum.cum_max()
    return {
        "label": label,
        "no_bars": len(trade_log_return),
        "total_log_return": total,
        "compound_return": float(np.exp(total) - 1),
        "std": std,
        "sharpe": float((mean / std) * annualized_rate) if std else 0.0,
        "max_drawdown": _as_float(dd.min()) if len(dd) else 0.0,
    }


def buy_and_hold_returns(df: pl.DataFrame, price_col: str = "close") -> pl.DataFrame:
    """Buy-and-hold log-return series over df, one row per bar."""
    return (
        df.select("datetime", log_return(price_col).alias("trade_log_return"))
        .with_columns(pl.col("trade_log_return").fill_null(0.0))
        .with_columns(pl.col("trade_log_return").cum_sum().alias("equity_curve"))
        .with_columns(
            (pl.col("equity_curve") - pl.col("equity_curve").cum_max()).alias(
                "drawdown_log_return"
            )
        )
    )


def buy_and_hold_metrics(
    df: pl.DataFrame, annualized_rate: float, price_col: str = "close"
) -> dict[str, Any]:
    bh = buy_and_hold_returns(df, price_col)
    return _series_metrics(bh["trade_log_return"], annualized_rate, "buy_and_hold")


def constant_position_metrics(
    df: pl.DataFrame,
    position: float,
    annualized_rate: float,
    price_col: str = "close",
    label: str | None = None,
) -> dict[str, Any]:
    """Metrics for holding a fixed position the whole series: +1 always long,
    -1 always short, 0 always flat."""
    lr = df.select(log_return(price_col)).to_series().fill_null(0.0)
    trade_lr = lr * position
    return _series_metrics(
        trade_lr, annualized_rate, label or f"constant_{position:+.0f}"
    )


def random_position_metrics(
    df: pl.DataFrame,
    annualized_rate: float,
    no_seeds: int = 200,
    price_col: str = "close",
    seed: int = 0,
) -> pl.DataFrame:
    """Sharpe/return distribution from no_seeds random +-1 position series of
    the same length as df, as a null baseline for what luck alone can produce."""
    lr = df.select(log_return(price_col)).to_series().fill_null(0.0).to_numpy()
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(no_seeds):
        pos = rng.choice([-1.0, 1.0], size=len(lr))
        trade_lr = pos * lr
        std = trade_lr.std()
        sharpe = float((trade_lr.mean() / std) * annualized_rate) if std else 0.0
        rows.append(
            {
                "seed": s,
                "sharpe": sharpe,
                "total_log_return": float(trade_lr.sum()),
                "compound_return": float(np.exp(trade_lr.sum()) - 1),
            }
        )
    return pl.DataFrame(rows)


def equal_weight_basket_returns(
    panel: pl.DataFrame,
    target_col: str = "fwd_return_1",
    datetime_col: str = "datetime",
) -> pl.DataFrame:
    """Equal-weight, long-only basket return per bar: mean of target_col
    across every symbol present that bar.

    The buy-and-hold baseline for a cross-sectional book: since the
    strategy itself is dollar-neutral (see dollar_neutral_weights) and
    holds no net crypto beta, the fair passive comparison is "what if you
    just held the whole universe equally" rather than any single symbol.
    always-long/short/flat baselines fall out of this same series: long is
    this series itself, short is its negation, flat is a zero series -
    feed the returned column into constant_position_metrics-style handling
    by treating it as a synthetic single asset's returns.
    """
    return (
        panel.group_by(datetime_col)
        .agg(pl.col(target_col).mean().alias("trade_log_return"))
        .sort(datetime_col)
    )


def random_dollar_neutral_metrics(
    panel: pl.DataFrame,
    annualized_rate: float,
    datetime_col: str = "datetime",
    symbol_col: str = "symbol",
    target_col: str = "fwd_return_1",
    top_frac: float = 0.2,
    gross_exposure: float = 1.0,
    max_position_per_symbol: float = 0.25,
    taker_fee: float | None = None,
    slippage: float = 1e-4,
    no_seeds: int = 200,
    seed: int = 0,
) -> pl.DataFrame:
    """Sharpe/return distribution from no_seeds random-ranking dollar-neutral
    portfolios, as a null baseline for what a strategy with zero real skill
    but the exact same portfolio construction (same top_frac, gross
    exposure, per-symbol cap, and - if taker_fee is given - the same cost
    model) would produce by chance. Stronger than a simple +-1 coin-flip
    baseline since it goes through the identical dollar_neutral_weights /
    portfolio_trade_frame pipeline the real strategy uses, not a simplified
    stand-in for it.
    """
    rng = np.random.default_rng(seed)
    symbols_per_bar = panel.select(datetime_col, symbol_col).unique()
    rows = []
    for s in range(no_seeds):
        random_pred = symbols_per_bar.with_columns(
            pl.Series("__rand_pred__", rng.normal(size=len(symbols_per_bar)))
        )
        scored = panel.join(random_pred, on=[datetime_col, symbol_col])
        weights = dollar_neutral_weights(
            scored,
            "__rand_pred__",
            datetime_col=datetime_col,
            symbol_col=symbol_col,
            top_frac=top_frac,
            gross_exposure=gross_exposure,
            max_position_per_symbol=max_position_per_symbol,
        )
        trade_frame = portfolio_trade_frame(
            weights,
            panel,
            target_col=target_col,
            datetime_col=datetime_col,
            symbol_col=symbol_col,
        )
        metrics = portfolio_metrics(
            trade_frame,
            annualized_rate,
            taker_fee=taker_fee,
            slippage=slippage,
            label="random",
        )
        rows.append(
            {
                "seed": s,
                "sharpe": metrics.get("sharpe_net", metrics.get("sharpe")),
                "total_log_return": metrics.get(
                    "total_log_return_net", metrics.get("total_log_return")
                ),
                "compound_return": metrics.get(
                    "compound_return_net", metrics.get("compound_return")
                ),
            }
        )
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------
# IC (information coefficient) harness
# --------------------------------------------------------------------------


def newey_west_tstat(x: np.ndarray, lag: int) -> tuple[float, float]:
    """Newey-West (HAC, Bartlett kernel) t-stat for the mean of a possibly
    autocorrelated series.

    A feature built from a W-bar rolling window induces autocorrelation in
    any per-period statistic derived from it (like IC_t below) out to about
    W lags; the naive i.i.d. standard error of the mean understates
    uncertainty unless those lags are accounted for. Set lag to roughly the
    feature's own lookback window.

    Returns (mean, tstat). NaN input rows are dropped first.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")

    mean = float(x.mean())
    demeaned = x - mean
    variance = float(np.sum(demeaned**2) / n)
    for k in range(1, min(lag, n - 1) + 1):
        gamma_k = float(np.sum(demeaned[k:] * demeaned[:-k]) / n)
        weight = 1 - k / (lag + 1)  # Bartlett kernel
        variance += 2 * weight * gamma_k
    variance = max(variance, 0.0)

    se_mean = np.sqrt(variance / n)
    tstat = mean / se_mean if se_mean > 0 else float("nan")
    return mean, float(tstat)


def cross_sectional_ic(
    panel: pl.DataFrame,
    pred_col: str,
    target_col: str,
    datetime_col: str = "datetime",
    min_symbols: int = 5,
) -> pl.DataFrame:
    """Per-timestamp Spearman IC of pred_col vs target_col across symbols.

    The right IC for a dollar-neutral book: it only ever compares symbols at
    the same instant, so contemporaneous cross-symbol correlation (BTC and
    ETH moving together) is absorbed inside each IC_t rather than treated as
    independent information. Statistical safety instead comes from applying
    Newey-West (see newey_west_tstat) to the resulting IC_t series, since a
    slow feature makes IC_t itself autocorrelated across time.

    Returns one row per timestamp with at least min_symbols non-null
    (pred, target) pairs: [datetime_col, "ic", "n"].
    """
    valid = panel.select(datetime_col, pred_col, target_col).drop_nulls()
    rows: list[dict[str, Any]] = []
    for key, group in valid.group_by(datetime_col, maintain_order=True):
        if len(group) < min_symbols:
            continue
        x = group[pred_col].to_numpy()
        y = group[target_col].to_numpy()
        # A near-zero (not exactly zero) std here is a floating-point
        # summation artifact of a genuinely-constant cross-section (e.g. a
        # seasonality feature like dow_sin, identical for every symbol at a
        # given bar since it only depends on datetime) - exact equality
        # against 0.0 misses that and lets a spurious corrcoef NaN through.
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            continue
        rho = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
        rows.append({datetime_col: key[0], "ic": rho, "n": len(group)})

    return (
        pl.DataFrame(rows).sort(datetime_col)
        if rows
        else pl.DataFrame(
            schema={
                datetime_col: valid[datetime_col].dtype,
                "ic": pl.Float64,
                "n": pl.Int64,
            }
        )
    )


def cross_sectional_ic_stats(ic_df: pl.DataFrame, nw_lag: int) -> dict[str, Any]:
    """Summary stats for a cross_sectional_ic result: mean IC and its
    Newey-West t-stat over the IC_t time series."""
    if len(ic_df) == 0:
        return {"mean_ic": float("nan"), "nw_tstat": float("nan"), "n_periods": 0}
    mean, tstat = newey_west_tstat(ic_df["ic"].to_numpy(), nw_lag)
    return {"mean_ic": mean, "nw_tstat": tstat, "n_periods": len(ic_df)}


def panel_ic(
    panel: pl.DataFrame,
    pred_col: str,
    target_col: str,
    nw_lag: int,
    datetime_col: str = "datetime",
) -> dict[str, Any]:
    """Spearman IC stacked over the whole (symbol, bar) panel, with a
    Driscoll-Kraay-style standard error: cluster by timestamp (so BTC/ETH at
    the same bar aren't counted as independent draws) and then apply
    Newey-West across time (so a slow feature's autocorrelated signal
    doesn't inflate significance either). naive_tstat (assuming n_obs i.i.d.
    draws) is reported alongside for contrast - it is badly overstated,
    since the real information content is closer to n_timestamps than
    n_obs.

    Ranks are computed over the whole stacked panel and standardized so the
    mean of rank_x * rank_y equals the panel's Spearman correlation; the
    reported panel_ic is the equal-weighted-by-timestamp mean of that
    product (i.e. each time period counts once regardless of how many
    symbols were in the cross-section then), which is what the clustered SE
    is computed on, so point estimate and SE stay internally consistent.
    """
    df = panel.select(datetime_col, pred_col, target_col).drop_nulls()
    n_obs = len(df)
    if n_obs < 2:
        return {
            "panel_ic": float("nan"),
            "clustered_nw_tstat": float("nan"),
            "naive_tstat": float("nan"),
            "n_obs": n_obs,
            "n_timestamps": 0,
        }

    rank_x = rankdata(df[pred_col].to_numpy())
    rank_y = rankdata(df[target_col].to_numpy())
    rx = (rank_x - rank_x.mean()) / rank_x.std()
    ry = (rank_y - rank_y.mean()) / rank_y.std()
    products = rx * ry

    per_timestamp = (
        df.select(datetime_col)
        .with_columns(pl.Series("product", products))
        .group_by(datetime_col, maintain_order=True)
        .agg(pl.col("product").mean().alias("mean_product"))
        .sort(datetime_col)
    )

    panel_ic_value, clustered_tstat = newey_west_tstat(
        per_timestamp["mean_product"].to_numpy(), nw_lag
    )
    # Standard t-test for a correlation coefficient under i.i.d. sampling:
    # t = r * sqrt((n-2) / (1-r^2)), df = n-2. Deliberately treats every
    # (symbol, bar) row as independent, which is the assumption this
    # function exists to show is wrong - compare against clustered_tstat.
    pooled_ic = float(products.mean())
    naive_tstat = (
        pooled_ic * np.sqrt((n_obs - 2) / max(1 - pooled_ic**2, 1e-12))
        if n_obs > 2
        else float("nan")
    )

    return {
        "panel_ic": panel_ic_value,
        "clustered_nw_tstat": clustered_tstat,
        "naive_tstat": float(naive_tstat),
        "n_obs": n_obs,
        "n_timestamps": len(per_timestamp),
    }


def ic_stability(
    ic_df: pl.DataFrame, datetime_col: str = "datetime", rolling_window: int = 30
) -> dict[str, Any]:
    """Stability diagnostics for a cross_sectional_ic result: rolling mean
    IC, per-year mean IC, and the fraction of months with positive mean IC.

    Stability outranks magnitude: an IC of 0.03 that's positive in 55% of
    months is more trustworthy than an IC of 0.06 that came entirely from
    one quarter.
    """
    if len(ic_df) == 0:
        return {
            "rolling_ic": pl.DataFrame(),
            "per_year_ic": pl.DataFrame(),
            "frac_positive_months": float("nan"),
        }

    ic_df = ic_df.sort(datetime_col)
    rolling_ic = ic_df.with_columns(
        pl.col("ic").rolling_mean(rolling_window).alias("rolling_mean_ic")
    )
    per_year_ic = (
        ic_df.with_columns(pl.col(datetime_col).dt.year().alias("year"))
        .group_by("year")
        .agg(pl.col("ic").mean().alias("mean_ic"), pl.len().alias("n_periods"))
        .sort("year")
    )
    per_month = (
        ic_df.with_columns(pl.col(datetime_col).dt.truncate("1mo").alias("month"))
        .group_by("month")
        .agg(pl.col("ic").mean().alias("mean_ic"))
    )
    frac_positive_months = _as_float((per_month["mean_ic"] > 0).mean())

    return {
        "rolling_ic": rolling_ic,
        "per_year_ic": per_year_ic,
        "frac_positive_months": frac_positive_months,
    }


# --------------------------------------------------------------------------
# Multiple-testing correction
# --------------------------------------------------------------------------


def deflated_sharpe_prob(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probabilistic Sharpe Ratio deflated for how many configs were tried
    (Bailey & Lopez de Prado's Deflated Sharpe Ratio).

    Picking the best of many backtested configs inflates the winner's Sharpe
    even if every config was pure noise, because you're implicitly reporting
    max() over n_trials draws rather than one draw. This estimates
    P(true Sharpe > 0 | observed max Sharpe over n_trials attempts) by first
    estimating the Sharpe a noise process would be expected to produce as its
    best of n_trials tries, then asking how much the observed Sharpe clears
    that bar relative to its own estimation uncertainty.

    sharpe: observed (best) per-period Sharpe ratio (not annualized).
    n_trials: number of configurations tried before selecting this one.
    n_obs: number of return observations behind the Sharpe estimate.
    skew, kurtosis: of the per-period return distribution (kurtosis=3 is the
    normal/no-excess-kurtosis case).
    """
    if n_obs <= 1:
        return float("nan")

    sr_std = np.sqrt((1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe**2) / (n_obs - 1))
    if sr_std == 0:
        return 1.0

    euler_mascheroni = 0.5772156649
    if n_trials > 1:
        expected_max_sharpe = sr_std * (
            (1 - euler_mascheroni) * norm.ppf(1 - 1 / n_trials)
            + euler_mascheroni * norm.ppf(1 - 1 / (n_trials * np.e))
        )
    else:
        expected_max_sharpe = 0.0

    return float(norm.cdf((sharpe - expected_max_sharpe) / sr_std))


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of values.

    Resamples values with replacement n_boot times, takes the mean of each
    resample, and returns the [ (1-ci)/2, 1-(1-ci)/2 ] percentiles of that
    distribution - e.g. a fold-level excess-return series that includes
    zero in its 95% CI can't reject "no real edge" no matter how good the
    point estimate looks.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return float(lo), float(hi)


def _auto_block_length(values: np.ndarray, max_lag: int = 50) -> int:
    """Pick a block length from the series' own autocorrelation.

    Rule used (a simple, documented heuristic in the spirit of Politis-White,
    not their full plug-in estimator): compute the sample ACF up to
    `max_lag`, and take the smallest lag at which the ACF first drops inside
    its approximate 95% white-noise confidence band (+-1.96/sqrt(n)) and
    stays inside it for the following lag too (avoids stopping on a single
    lag that crosses zero by chance). That lag is treated as the point past
    which the series looks memoryless, and becomes the block length, floored
    at 1 (i.i.d. case) and capped at n // 2 so blocks can't exceed half the
    sample.
    """
    n = len(values)
    if n < 4:
        return 1
    x = values - values.mean()
    denom = np.sum(x**2)
    if denom == 0:
        return 1
    max_lag = min(max_lag, n - 2)
    band = 1.96 / np.sqrt(n)
    block_length = max_lag  # default: never dropped inside the band
    for lag in range(1, max_lag):
        acf_lag = np.sum(x[:-lag] * x[lag:]) / denom
        acf_next = (
            np.sum(x[: -(lag + 1)] * x[lag + 1 :]) / denom
            if lag + 1 <= max_lag
            else 0.0
        )
        if abs(acf_lag) < band and abs(acf_next) < band:
            block_length = lag
            break
    return int(np.clip(block_length, 1, max(1, n // 2)))


def _block_resample(
    values: np.ndarray,
    block_length: int | None,
    n_boot: int,
    seed: int,
    statistic,
) -> np.ndarray:
    """Shared block-resampling loop behind `block_bootstrap_ci` and
    `block_bootstrap_pvalue`: resample n_boot times (contiguous, wrapping
    blocks of `block_length`, chosen via `_auto_block_length` if None) and
    apply `statistic` to each resample, returning the array of n_boot
    statistic values. Factored out so both functions share one
    implementation of the resampling itself rather than drifting apart.
    """
    n = len(values)
    if block_length is None:
        block_length = _auto_block_length(values)
    block_length = max(1, min(block_length, n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_length))
    out = np.empty(n_boot)
    for i in range(n_boot):
        start_idx = rng.integers(0, n, size=n_blocks)
        sample = np.concatenate(
            [values[np.arange(s, s + block_length) % n] for s in start_idx]
        )[:n]
        out[i] = statistic(sample)
    return out


def block_bootstrap_ci(
    values: np.ndarray,
    block_length: int | None = None,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
    statistic=None,
) -> tuple[float, float]:
    """Stationary/moving-block bootstrap CI for a statistic of an
    autocorrelated series (the mean, by default).

    Fold-level and bar-level return series are not i.i.d. - adjacent folds
    overlap in market regime, adjacent bars are autocorrelated - so resampling
    single observations independently (`bootstrap_ci`) understates the true
    sampling variance and produces CIs that are too narrow. This resamples
    contiguous blocks of `block_length` consecutive observations (wrapping
    around the end of the series, i.e. the "circular"/stationary block
    bootstrap) with replacement until the resample has >= n original
    observations, then applies `statistic` (default: the mean), repeated
    n_boot times.

    block_length: if None, chosen automatically from the series' own
    autocorrelation via `_auto_block_length` (see its docstring for the
    rule). Pass an explicit value to override, e.g. block_length=1 to
    recover the i.i.d. bootstrap exactly.
    statistic: a callable array -> float, applied to each resample. Defaults
    to np.mean, preserving this function's original mean-only behavior for
    every existing caller. Pass e.g. `lambda x: hill_estimator(x, k=200)` to
    bootstrap a nonlinear statistic like a tail-index estimate instead.
    n_boot, ci, seed: as before.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    stat_fn = np.mean if statistic is None else statistic
    boot_stats = _block_resample(values, block_length, n_boot, seed, stat_fn)
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_stats, [alpha, 1 - alpha])
    return float(lo), float(hi)


def block_bootstrap_pvalue(
    values: np.ndarray,
    null_value: float = 0.0,
    block_length: int | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> float:
    """Two-sided block-bootstrap p-value for H0: mean(values) == null_value,
    valid for autocorrelated/heavy-tailed series where a normal-approximation
    t-test's own CLT may not hold (see
    docs/03-statistical-inference.md#central-limit-theorem-and-when-it-fails-under-heavy-tails).

    Standard shift-and-resample construction: recenter the data to have
    exactly `null_value` as its mean (subtract the observed mean, add back
    null_value), so resampling from it simulates the null hypothesis being
    exactly true, then see how extreme the *actual* observed mean is relative
    to that null resampling distribution. This does not require the mean's
    own sampling distribution to be normal, unlike a t-stat's p-value.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return float("nan")
    observed_mean = float(values.mean())
    shifted = values - observed_mean + null_value
    boot_means = _block_resample(shifted, block_length, n_boot, seed, np.mean)
    p_lo = float(np.mean(boot_means <= observed_mean))
    p_hi = float(np.mean(boot_means >= observed_mean))
    return float(min(1.0, 2.0 * min(p_lo, p_hi)))


# --------------------------------------------------------------------------
# Cross-sectional portfolio construction
# --------------------------------------------------------------------------


def panel_walk_forward_splits(
    panel: pl.DataFrame,
    train_bars: int,
    test_bars: int,
    step_bars: int | None = None,
    mode: str = "rolling",
    embargo_bars: int = 0,
    origin_offset: int = 0,
    datetime_col: str = "datetime",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """walk_forward_splits, but for a stacked multi-symbol panel.

    Folds are computed over unique timestamps (delegating to
    walk_forward_splits so the fold-boundary logic isn't duplicated), then
    each timestamp index is expanded to every row sharing that timestamp.
    This guarantees every symbol at a given bar stays on the same side of
    the train/test boundary - splitting by raw row position instead would
    let some symbols' bar-t rows leak into train while others land in test,
    even though they're the same instant.

    Returned indices are row positions into panel as passed in - the caller
    must index the same (unsorted) panel with them.
    """
    datetimes = panel[datetime_col].to_numpy()
    unique_times, inverse = np.unique(datetimes, return_inverse=True)

    time_splits = walk_forward_splits(
        len(unique_times),
        train_bars,
        test_bars,
        step_bars,
        mode,
        embargo_bars,
        origin_offset,
    )

    row_splits = []
    for train_time_idx, test_time_idx in time_splits:
        train_mask = np.isin(inverse, train_time_idx)
        test_mask = np.isin(inverse, test_time_idx)
        row_splits.append((np.flatnonzero(train_mask), np.flatnonzero(test_mask)))
    return row_splits


def vol_normalized_target(target_col: str, vol_col: str) -> pl.Expr:
    """target_col / vol_col as the regression target instead of target_col
    directly, so training loss stops being dominated by high-vol bars/eras
    (e.g. 2022) at the expense of fitting low-vol ones. vol_col should be a
    causal, already-computed realized vol column (see features.realized_vol)
    known as of the same bar as the prediction.
    """
    return (pl.col(target_col) / pl.col(vol_col)).alias(f"{target_col}_vol_norm")


def vol_targeted_size(pred_col: str, vol_col: str, vol_target: float) -> pl.Expr:
    """clip(pred, -1, 1) * (vol_target / vol_t): continuous position size
    instead of sign(pred).

    Since the model is trained on vol_normalized_target, pred is already in
    "predicted return per unit of that bar's vol" units, so clipping to
    [-1, 1] is a natural cap (don't bet bigger than a 1-sigma-equivalent
    move) rather than an arbitrary threshold. Scaling by vol_target / vol_t
    then converts that into an actual position size that keeps realized
    portfolio vol roughly constant across symbols and regimes - a
    risk-management effect, not a parameter to tune per backtest.
    """
    return (pl.col(pred_col).clip(-1, 1) * (vol_target / pl.col(vol_col))).alias(
        "vol_targeted_size"
    )


def dollar_neutral_weights(
    panel: pl.DataFrame,
    pred_col: str,
    datetime_col: str = "datetime",
    symbol_col: str = "symbol",
    top_frac: float = 0.2,
    size_col: str | None = None,
    gross_exposure: float = 1.0,
    max_position_per_symbol: float = 0.25,
) -> pl.DataFrame:
    """Per-bar, dollar-neutral long/short portfolio weights.

    Each bar: rank symbols by pred_col, take the top top_frac as the long
    leg and the bottom top_frac as the short leg (everything else gets
    weight 0). This strips whatever's common to the whole cross-section
    that bar (crypto beta) since both legs only ever bet on relative
    ranking, never on direction. Within a leg, weight is proportional to
    |size_col| if given (e.g. vol_targeted_size - a more confident/higher
    vol-adjusted prediction gets more capital) or equal if size_col is None.

    Long leg weights sum to +gross_exposure / 2, short leg to
    -gross_exposure / 2 (net = 0, gross = sum(abs(weight)) = gross_exposure
    before any capping). max_position_per_symbol then clips each symbol's
    weight - this can only shrink gross exposure further, never breach the
    target, so no separate total-gross-cap step is needed on top of it.

    Returns one row per (datetime, symbol) with a "weight" column (0 for
    symbols in neither leg that bar).
    """
    cols = [datetime_col, symbol_col, pred_col] + ([size_col] if size_col else [])
    df = panel.select(cols).drop_nulls()

    rows: list[dict[str, Any]] = []
    for key, group in df.group_by(datetime_col, maintain_order=True):
        n = len(group)
        k = max(1, int(np.floor(n * top_frac)))
        preds = group[pred_col].to_numpy()
        symbols = group[symbol_col].to_list()
        size = np.abs(group[size_col].to_numpy()) if size_col else np.ones(n)
        order = np.argsort(preds)
        short_idx, long_idx = order[:k], order[-k:]

        weight = np.zeros(n)
        long_size, short_size = size[long_idx], size[short_idx]
        if long_size.sum() > 0:
            weight[long_idx] = (gross_exposure / 2) * long_size / long_size.sum()
        if short_size.sum() > 0:
            weight[short_idx] = -(gross_exposure / 2) * short_size / short_size.sum()
        weight = np.clip(weight, -max_position_per_symbol, max_position_per_symbol)

        for sym, w in zip(symbols, weight, strict=True):
            rows.append({datetime_col: key[0], symbol_col: sym, "weight": float(w)})

    return (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(
            schema={
                datetime_col: df[datetime_col].dtype,
                symbol_col: pl.Utf8,
                "weight": pl.Float64,
            }
        )
    )


def portfolio_turnover(
    weights: pl.DataFrame, datetime_col: str = "datetime", symbol_col: str = "symbol"
) -> pl.DataFrame:
    """Per-bar total turnover of a multi-symbol weights panel: sum over
    symbols of |weight_t - weight_{t-1}| (each symbol's own weight history;
    a symbol's first appearance is charged its full entry weight, matching
    add_trading_costs' treatment of position_{-1} = 0).
    """
    w = weights.sort([symbol_col, datetime_col])
    w = w.with_columns(
        pl.col("weight")
        .diff()
        .fill_null(pl.col("weight"))
        .abs()
        .over(symbol_col)
        .alias("symbol_turnover")
    )
    return (
        w.group_by(datetime_col)
        .agg(pl.col("symbol_turnover").sum().alias("turnover"))
        .sort(datetime_col)
    )


def portfolio_trade_frame(
    weights: pl.DataFrame,
    returns: pl.DataFrame,
    target_col: str = "fwd_return_1",
    datetime_col: str = "datetime",
    symbol_col: str = "symbol",
) -> pl.DataFrame:
    """Combine per-symbol weights with forward returns into one row per bar:
    "trade_log_return" (the portfolio's gross bar return, sum_i weight_i *
    return_i) and "turnover" (see portfolio_turnover). Shaped to match what
    add_portfolio_costs / _series_metrics / cost_summary expect, the same
    summary functions the single-asset walk_forward_run path uses.
    """
    joined = weights.join(
        returns.select(datetime_col, symbol_col, target_col),
        on=[datetime_col, symbol_col],
        how="inner",
    )
    port_return = (
        joined.with_columns(
            (pl.col("weight") * pl.col(target_col)).alias("weighted_return")
        )
        .group_by(datetime_col)
        .agg(pl.col("weighted_return").sum().alias("trade_log_return"))
    )
    turnover = portfolio_turnover(weights, datetime_col, symbol_col)
    return port_return.join(turnover, on=datetime_col).sort(datetime_col)


def add_portfolio_costs(
    trade_frame: pl.DataFrame, taker_fee: float, slippage: float = 1e-4
) -> pl.DataFrame:
    """add_trading_costs' cost math, applied to a portfolio_trade_frame
    result where "turnover" is already the summed per-symbol turnover for
    that bar (see portfolio_turnover) rather than a diff of one position
    column - the rest of the accounting (cost_log_return, _net variants) is
    identical.
    """
    cost_frac = taker_fee + slippage
    return (
        trade_frame.with_columns(
            (1 - cost_frac * pl.col("turnover")).log().alias("cost_log_return")
        )
        .with_columns(
            (pl.col("trade_log_return") + pl.col("cost_log_return")).alias(
                "trade_log_return_net"
            )
        )
        .with_columns(
            pl.col("trade_log_return_net").cum_sum().alias("equity_curve_net")
        )
        .with_columns(
            (pl.col("equity_curve_net") - pl.col("equity_curve_net").cum_max()).alias(
                "drawdown_log_return_net"
            )
        )
    )


def portfolio_metrics(
    trade_frame: pl.DataFrame,
    annualized_rate: float,
    taker_fee: float | None = None,
    slippage: float = 1e-4,
    label: str = "portfolio",
) -> dict[str, Any]:
    """Summary metrics for a portfolio_trade_frame result - the multi-symbol
    analogue of stitched_metrics. Reports gross always, and net/cost fields
    (via add_portfolio_costs + cost_summary) whenever taker_fee is given.
    """
    if len(trade_frame) == 0:
        return {"label": label, "no_bars": 0}

    metrics = _series_metrics(trade_frame["trade_log_return"], annualized_rate, label)

    if taker_fee is not None:
        costed = add_portfolio_costs(trade_frame, taker_fee, slippage)
        net_metrics = _series_metrics(
            costed["trade_log_return_net"], annualized_rate, f"{label}_net"
        )
        metrics["sharpe_net"] = net_metrics["sharpe"]
        metrics["total_log_return_net"] = net_metrics["total_log_return"]
        metrics["compound_return_net"] = net_metrics["compound_return"]
        metrics["max_drawdown_net"] = net_metrics["max_drawdown"]
        metrics.update(cost_summary(costed, annualized_rate))

    return metrics


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


def load_universe_panel(
    symbols: Sequence[str],
    interval: str,
    start_date: datetime,
    end_date: datetime,
    min_cross_section: int = 10,
    download_dir: str = "tmp",
    cache_dir: str = "cache",
    allow_holdout: bool = False,
) -> pl.DataFrame:
    """Load a ragged cross-sectional panel of OHLCV klines for symbols over [start_date, end_date].

    HOLDOUT_START (2025-07-01) is frozen: any end_date reaching into it raises
    unless allow_holdout=True, so the boundary is enforced here structurally
    rather than left to notebook discipline. Only the Phase 7 holdout run
    should ever pass allow_holdout=True.

    Symbols not yet listed at start_date, or delisted before end_date, simply
    contribute fewer rows (data.download_klines_range skips months with no
    archive file rather than raising) - no forward/back-fill is applied, so
    the panel is ragged by construction and each symbol's first/last valid
    bar reflects its real listing/delisting history. A symbol with zero data
    anywhere in the range is dropped with a warning rather than failing the
    whole load.

    A bar (timestamp) is kept only if at least min_cross_section symbols have
    data there, so no cross-sectional rank/z-score is ever computed from a
    near-empty cross-section early in a symbol's history.

    Adds a "symbol" column; result is sorted by (datetime, symbol).
    """
    if end_date > HOLDOUT_START and not allow_holdout:
        raise ValueError(
            f"end_date {end_date} reaches into the frozen holdout period "
            f"(>= {HOLDOUT_START:%Y-%m-%d}). Pass allow_holdout=True only for "
            "the Phase 7 holdout run."
        )

    frames = []
    for sym in symbols:
        try:
            df = data.download_klines_range(
                sym, interval, start_date, end_date, download_dir, cache_dir
            )
        except ValueError as e:
            tqdm.write(f"[WARNING] {sym}: no data in range, skipped ({e})")
            continue
        frames.append(df.with_columns(pl.lit(sym).alias("symbol")))

    if not frames:
        raise ValueError(f"No data found for any symbol in {start_date} to {end_date}")

    panel = pl.concat(frames, how="diagonal_relaxed").sort(["datetime", "symbol"])

    cross_section_width = panel.group_by("datetime").agg(pl.len().alias("n_symbols"))
    valid_datetimes = cross_section_width.filter(
        pl.col("n_symbols") >= min_cross_section
    )["datetime"]
    panel = panel.filter(pl.col("datetime").is_in(valid_datetimes))

    return panel


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
