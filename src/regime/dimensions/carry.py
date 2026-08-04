"""Futures carry regime indicators.

Ported verbatim from ``ultron_finance.regime.dimensions.carry``.
"""

from __future__ import annotations

import pandas as pd

from regime.engine import RegimeInputs
from regime.indicators import annualized_roll_yield, realized_vol, vol_scaled_carry
from regime.registry import register


def _curve(inputs: RegimeInputs) -> pd.DataFrame:
    if inputs.curve is None:
        raise ValueError("curve input is required")
    return inputs.curve


def _annualized(inputs: RegimeInputs) -> pd.Series:
    curve = _curve(inputs)
    days = (
        (curve["dte_f2"] - curve["dte_f1"]).replace(0, float("nan"))
        if {"dte_f1", "dte_f2"}.issubset(curve.columns)
        else 30.0
    )
    return annualized_roll_yield(curve["close_f1"], curve["close_f2"], days)


@register("carry.ann_roll_yield", {"curve"})
def carry_annualized_roll_yield(inputs: RegimeInputs) -> pd.Series:
    return _annualized(inputs)


@register("carry.vol_scaled", {"curve"}, vol_window=20)
def carry_vol_scaled(inputs: RegimeInputs, vol_window: int = 20) -> pd.Series:
    return vol_scaled_carry(_annualized(inputs), realized_vol(inputs.ohlcv["close"], vol_window))
