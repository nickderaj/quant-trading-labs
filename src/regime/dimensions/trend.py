"""Trend regime indicators.

Ported verbatim from ``ultron_finance.regime.dimensions.trend``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime.engine import RegimeInputs
from regime.indicators import (
    dmi,
    efficiency_ratio,
    log_returns,
    ma_separation,
    ma_slope,
    sma,
    variance_ratio,
)
from regime.registry import register


def _close(inputs: RegimeInputs) -> pd.Series:
    return inputs.ohlcv["close"]


@register("trend.price_vs_ma", {"ohlcv"}, window=200)
def price_vs_ma(inputs: RegimeInputs, window: int = 200) -> pd.Series:
    close = _close(inputs)
    return (close - sma(close, window)) / sma(close, window)


@register("trend.ma_slope", {"ohlcv"}, window=100, slope_window=20)
def trend_ma_slope(inputs: RegimeInputs, window: int = 100, slope_window: int = 20) -> pd.Series:
    return ma_slope(_close(inputs), window, slope_window)


@register("trend.nday_log_return", {"ohlcv"}, n=60)
def nday_log_return(inputs: RegimeInputs, n: int = 60) -> pd.Series:
    return log_returns(_close(inputs), n)


@register("trend.adx", {"ohlcv"}, window=14)
def trend_adx(inputs: RegimeInputs, window: int = 14) -> pd.Series:
    values = dmi(inputs.ohlcv, window)
    return pd.Series(
        values["adx"] * np.sign(values["plus_di"] - values["minus_di"]), index=inputs.ohlcv.index
    )


@register("trend.efficiency_ratio", {"ohlcv"}, window=20)
def trend_efficiency(inputs: RegimeInputs, window: int = 20) -> pd.Series:
    return pd.Series(
        efficiency_ratio(_close(inputs), window) * np.sign(_close(inputs).diff(window)),
        index=inputs.ohlcv.index,
    )


@register("trend.ma_separation", {"ohlcv"}, fast=50, slow=200)
def trend_separation(inputs: RegimeInputs, fast: int = 50, slow: int = 200) -> pd.Series:
    return ma_separation(_close(inputs), fast, slow)


@register("trend.variance_ratio", {"ohlcv"}, q=5, window=252)
def trend_variance_ratio(inputs: RegimeInputs, q: int = 5, window: int = 252) -> pd.Series:
    return variance_ratio(_close(inputs), q, window) - 1
