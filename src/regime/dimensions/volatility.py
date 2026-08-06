"""Volatility regime indicators.

Ported verbatim from ``ultron_finance.regime.dimensions.volatility``.
"""

from __future__ import annotations

import pandas as pd

from regime.engine import RegimeInputs
from regime.indicators import (
    atr,
    bollinger_width,
    realized_vol,
    rolling_percentile_rank,
    vol_of_vol,
)
from regime.registry import register


@register("vol.atr_percentile", {"ohlcv"}, window=14, pct_window=252)
def atr_percentile(
    inputs: RegimeInputs, window: int = 14, pct_window: int = 252
) -> pd.Series:
    raw = atr(inputs.ohlcv, window) / inputs.ohlcv["close"]
    return 2 * rolling_percentile_rank(raw, pct_window) - 1


@register("vol.realized_vol_percentile", {"ohlcv"}, window=20, pct_window=252)
def realized_vol_percentile(
    inputs: RegimeInputs, window: int = 20, pct_window: int = 252
) -> pd.Series:
    return (
        2
        * rolling_percentile_rank(
            realized_vol(inputs.ohlcv["close"], window), pct_window
        )
        - 1
    )


@register("vol.vol_of_vol", {"ohlcv"}, vol_window=20, vov_window=60)
def volatility_of_volatility(
    inputs: RegimeInputs, vol_window: int = 20, vov_window: int = 60
) -> pd.Series:
    return vol_of_vol(inputs.ohlcv["close"], vol_window, vov_window)


@register("vol.bollinger_width_percentile", {"ohlcv"}, window=20, pct_window=252)
def bollinger_width_percentile(
    inputs: RegimeInputs, window: int = 20, pct_window: int = 252
) -> pd.Series:
    return (
        2
        * rolling_percentile_rank(
            bollinger_width(inputs.ohlcv["close"], window), pct_window
        )
        - 1
    )


@register("vol.vix_level", {"ohlcv", "macro"}, column="VIXCLS", pct_window=252)
def vix_level(
    inputs: RegimeInputs, column: str = "VIXCLS", pct_window: int = 252
) -> pd.Series:
    if inputs.macro is None or column not in inputs.macro:
        return pd.Series(index=inputs.ohlcv.index, dtype=float)
    return (
        2
        * rolling_percentile_rank(
            inputs.macro[column].reindex(inputs.ohlcv.index).ffill(), pct_window
        )
        - 1
    )
