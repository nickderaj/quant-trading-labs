"""Technical/statistical primitives used by the regime dimension modules.

Ported subset of ``ultron_finance.indicators``, ``ultron_finance.stats``, and
``ultron_finance.futures.carry`` (see
``../ultron/libs/finance/src/ultron_finance/{indicators,stats}.py`` and
``.../futures/carry.py``) -- only the functions actually reached by
``regime/dimensions/*.py`` (NEXT_PROMPT.md Sec2.3) are ported here, combined
into one module since none of them depend on the rest of those source files.

No repo-local equivalent exists for any of these: `src/features.py` and
`src/research.py` operate on polars expressions over this repo's 6h crypto
bar schema, not general-purpose pandas Series/DataFrame transforms, and
`src/research/tmp/spread_lib10.py` / `commod_lib8.py`'s regime helpers are
different constructions (deadband/persistence state machines, tercile
macro regimes) rather than these indicator primitives -- wiring the ported
dimensions to them would mean reimplementing this module inside them, not
reuse. So this is a straight port, not a delegation.

The source's `ultron_logging.get_logger`-based short-history warnings are
replaced with stdlib `logging` -- that dependency is not vendored in.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

STORAGE_LOW = 0.30
STORAGE_HIGH = 0.60
STORAGE_MID = (STORAGE_LOW + STORAGE_HIGH) / 2
FINANCING_RATE_APPROX = 0.05


def _warn_if_short(series_len: int, window: int, name: str) -> None:
    if series_len < window:
        _logger.warning(
            "insufficient history for indicator: indicator=%s rows=%d window=%d",
            name,
            series_len,
            window,
        )


# --------------------------------------------------------------------------- #
# from ultron_finance.indicators
# --------------------------------------------------------------------------- #
def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average over a trailing `window` of bars."""
    _warn_if_short(len(series), window, "sma")
    return series.rolling(window=window, min_periods=window).mean()


def log_returns(close: pd.Series, periods: int = 1) -> pd.Series:
    """Log price returns over ``periods`` bars."""
    with np.errstate(invalid="ignore"):
        return pd.Series(np.log(close / close.shift(periods)), index=close.index)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range, Wilder-smoothed. `df` needs `high`, `low`, `close`."""
    _warn_if_short(len(df), window, "atr")
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    high_low = high - low
    high_prev_close = (high - prev_close).abs()
    low_prev_close = (low - prev_close).abs()

    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    true_range.iloc[0] = high_low.iloc[0]

    wilder = true_range.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    return wilder


def dmi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Wilder directional movement index with plus/minus DI and ADX."""
    _warn_if_short(len(df), window, "dmi")
    high, low, close = df["high"], df["low"], df["close"]
    up_move, down_move = high.diff(), -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    prev_close = close.shift()
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1
    )
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    smooth_tr = tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    plus_di = (
        100 * plus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / smooth_tr
    )
    minus_di = (
        100 * minus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / smooth_tr
    )
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_value = dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx_value})


def efficiency_ratio(close: pd.Series, window: int = 20) -> pd.Series:
    """Kaufman efficiency ratio, bounded between zero and one."""
    _warn_if_short(len(close), window, "efficiency_ratio")
    change = close.diff(window).abs()
    volatility = close.diff().abs().rolling(window, min_periods=window).sum()
    return (change / volatility.replace(0, np.nan)).clip(0, 1)


def realized_vol(close: pd.Series, window: int = 20, periods_per_year: int = 252) -> pd.Series:
    """Annualized rolling standard deviation of log returns."""
    _warn_if_short(len(close), window, "realized_vol")
    return log_returns(close).rolling(window, min_periods=window).std() * float(
        np.sqrt(periods_per_year)
    )


def vol_of_vol(close: pd.Series, vol_window: int = 20, vov_window: int = 60) -> pd.Series:
    """Rolling volatility of annualized realized volatility."""
    _warn_if_short(len(close), vol_window + vov_window, "vol_of_vol")
    return realized_vol(close, vol_window).rolling(vov_window, min_periods=vov_window).std()


def rolling_autocorr(returns: pd.Series, lag: int = 1, window: int = 60) -> pd.Series:
    """Rolling return autocorrelation at ``lag``."""
    _warn_if_short(len(returns), window, "rolling_autocorr")
    return returns.rolling(window, min_periods=window).corr(returns.shift(lag))


def variance_ratio(close: pd.Series, q: int = 5, window: int = 252) -> pd.Series:
    """Rolling Lo-MacKinlay variance ratio; above one indicates persistence."""
    _warn_if_short(len(close), window, "variance_ratio")
    returns = log_returns(close)
    one_period_var = returns.rolling(window, min_periods=window).var()
    q_period_var = returns.rolling(q).sum().rolling(window, min_periods=window).var()
    return q_period_var / (q * one_period_var.replace(0, np.nan))


def ma_slope(series: pd.Series, window: int = 100, slope_window: int = 20) -> pd.Series:
    """Percentage change per bar of a trailing simple moving average."""
    average = sma(series, window)
    return (average - average.shift(slope_window)) / (slope_window * average)


def ma_separation(close: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """Relative separation of fast and slow simple moving averages."""
    slow_average = sma(close, slow)
    return (sma(close, fast) - slow_average) / slow_average


def bollinger_width(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """Bollinger band width as a fraction of the moving average."""
    average = sma(close, window)
    std = close.rolling(window, min_periods=window).std()
    return 2 * num_std * std / average


def rolling_percentile_rank(s: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    """Percentile rank of each observation in its inclusive trailing window."""
    _warn_if_short(len(s), min_periods, "rolling_percentile_rank")

    def rank(values: np.ndarray[Any, np.dtype[np.float64]]) -> float:
        current = values[-1]
        valid = values[~np.isnan(values)]
        if np.isnan(current) or not len(valid):
            return float("nan")
        return float((valid <= current).sum() / len(valid))

    return s.rolling(window, min_periods=min_periods).apply(rank, raw=True)


# --------------------------------------------------------------------------- #
# from ultron_finance.stats
# --------------------------------------------------------------------------- #
def compute_half_life(s: pd.Series) -> float:
    """Single-pass AR(1) half-life of mean reversion. Returns NaN if b >= 0."""
    arr = s.dropna().to_numpy()
    if len(arr) < 30:
        return float("nan")
    d_s = np.diff(arr)
    s_lag = arr[:-1]
    mask = np.isfinite(d_s) & np.isfinite(s_lag)
    if mask.sum() < 20:
        return float("nan")
    coeffs = np.polyfit(s_lag[mask], d_s[mask], 1)
    b = coeffs[0]
    return float(-np.log(2) / b) if b < 0 else float("nan")


def compute_zscore(spread: pd.Series, lookback: int) -> pd.Series:
    """Rolling z-score: (spread - rolling_mean) / rolling_std, shift(1) for no look-ahead."""
    mu = spread.rolling(lookback, min_periods=lookback // 2).mean()
    sigma = spread.rolling(lookback, min_periods=lookback // 2).std()
    z = (spread - mu) / sigma.replace(0, np.nan)
    return z.shift(1)


# --------------------------------------------------------------------------- #
# from ultron_finance.futures.carry
# --------------------------------------------------------------------------- #
def compute_carry_fv(
    leg2_price: pd.Series,
    storage_per_month: float = STORAGE_MID,
    financing_rate: float = FINANCING_RATE_APPROX,
) -> pd.Series:
    return -(storage_per_month + financing_rate * leg2_price / 12)


def compute_excess_spread(spread_value: pd.Series, carry_fv: pd.Series) -> pd.Series:
    return (spread_value - carry_fv).rename("excess_spread")


def roll_yield(front: pd.Series, deferred: pd.Series) -> pd.Series:
    return (front - deferred) / deferred


def annualized_roll_yield(
    front: pd.Series, deferred: pd.Series, days_between: pd.Series | float
) -> pd.Series:
    return ((front / deferred) - 1) * (365.0 / days_between)


def curve_slope(curve: pd.DataFrame, near: str = "close_f1", far: str = "close_f12") -> pd.Series:
    if near not in curve:
        raise KeyError(f"Curve has no {near!r} column")
    candidates = [str(column) for column in curve.columns if str(column).startswith("close_f")]
    far_column = (
        far
        if far in curve
        else max(candidates, key=lambda column: int(column.removeprefix("close_f")))
    )
    return (curve[far_column] - curve[near]) / curve[near]


def vol_scaled_carry(ann_roll_yield: pd.Series, realized_vol: pd.Series) -> pd.Series:
    return ann_roll_yield / realized_vol.replace(0, float("nan"))
