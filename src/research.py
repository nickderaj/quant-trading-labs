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
from scipy.stats import norm
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
