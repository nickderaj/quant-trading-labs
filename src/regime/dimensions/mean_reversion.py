"""Mean-reversion regime indicators.

Ported verbatim from ``ultron_finance.regime.dimensions.mean_reversion``.
"""

from __future__ import annotations

import pandas as pd

from regime.engine import RegimeInputs
from regime.indicators import (
    compute_half_life,
    compute_zscore,
    log_returns,
    rolling_autocorr,
    variance_ratio,
)
from regime.registry import register


@register("mr.autocorr", {"ohlcv"}, lag=1, window=60)
def mr_autocorr(inputs: RegimeInputs, lag: int = 1, window: int = 60) -> pd.Series:
    return -rolling_autocorr(log_returns(inputs.ohlcv["close"]), lag, window)


@register("mr.variance_ratio", {"ohlcv"}, q=5, window=252)
def mr_variance_ratio(inputs: RegimeInputs, q: int = 5, window: int = 252) -> pd.Series:
    return 1 - variance_ratio(inputs.ohlcv["close"], q, window)


@register("mr.half_life", {"ohlcv"}, window=252)
def half_life(inputs: RegimeInputs, window: int = 252) -> pd.Series:
    close = inputs.ohlcv["close"]
    return close.rolling(window, min_periods=window).apply(
        lambda values: compute_half_life(pd.Series(values)), raw=True
    )


@register("mr.zscore_extremity", {"ohlcv"}, window=60)
def zscore_extremity(inputs: RegimeInputs, window: int = 60) -> pd.Series:
    return compute_zscore(inputs.ohlcv["close"], window).abs()
