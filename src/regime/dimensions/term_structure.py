"""Term-structure regime indicators for futures curves.

Ported from ``ultron_finance.regime.dimensions.term_structure`` with one
deliberate deviation: ``term_structure_slope`` calls ``curve_slope`` with
``far="close_f3"`` explicitly rather than the source's default
``far="close_f12"``. This repo's curve parquets
(``src/research/data/market/research/{cl,gc,hg,ng,si}_curve.parquet``) only
carry `close_f1..close_f3`, so `curve_slope`'s own fallback (pick the
furthest available `close_f*` column when the requested one is absent)
would already resolve to `close_f3` implicitly -- but that fallback is a
silent, data-shape-dependent behaviour, not a documented contract, so it is
pinned explicitly here instead of relied on. See
NEXT_PROMPT.md Sec3.4 landmine #1 and
`src/results/014_market_regime_engine_and_accuracy.md`: this dimension is
therefore scored on a 3-month curve, not production's 12-month curve, and
term_structure/carry numbers do not transfer one-for-one to production.
"""

from __future__ import annotations

import pandas as pd

from regime.engine import RegimeInputs
from regime.indicators import (
    annualized_roll_yield,
    compute_carry_fv,
    compute_excess_spread,
    curve_slope,
)
from regime.registry import register
from regime.scoring import rolling_zscore


def _curve(inputs: RegimeInputs) -> pd.DataFrame:
    if inputs.curve is None:
        raise ValueError("curve input is required")
    return inputs.curve


def _days_between(curve: pd.DataFrame) -> pd.Series:
    if "dte_f1" in curve and "dte_f2" in curve:
        return (curve["dte_f2"] - curve["dte_f1"]).replace(0, float("nan"))
    return pd.Series(30.0, index=curve.index)


@register("ts.curve_slope", {"curve"})
def term_structure_slope(inputs: RegimeInputs) -> pd.Series:
    """Sign-flipped slope: positive values indicate backwardation.

    ``far="close_f3"`` -- see module docstring; this repo has no f12 leg.
    """
    return -curve_slope(_curve(inputs), far="close_f3")


@register("ts.calendar_spread_z", {"curve"}, window=252)
def calendar_spread_z(inputs: RegimeInputs, window: int = 252) -> pd.Series:
    curve = _curve(inputs)
    return rolling_zscore(curve["close_f1"] - curve["close_f2"], window)


@register("ts.ann_roll_yield", {"curve"})
def term_structure_roll_yield(inputs: RegimeInputs) -> pd.Series:
    curve = _curve(inputs)
    return annualized_roll_yield(
        curve["close_f1"], curve["close_f2"], _days_between(curve)
    )


@register("ts.excess_spread", {"curve"})
def excess_spread(inputs: RegimeInputs) -> pd.Series:
    curve = _curve(inputs)
    spread = curve["close_f1"] - curve["close_f2"]
    return compute_excess_spread(spread, compute_carry_fv(curve["close_f2"]))
