"""Macro and positioning regime indicators.

Ported verbatim from ``ultron_finance.regime.dimensions.macro``, including
the source bug in ``cot_noncomm``: it is registered with ``requires={"macro"}``
but its body reads ``inputs.cot``. Effect: ``engine.detect`` gates this
indicator on ``inputs.macro is not None`` (never on ``inputs.cot``), so
passing ``cot=None`` does not skip the ``risk`` dimension -- the column comes
back all-NaN and ``combine`` drops it via the coverage rule, leaving `risk`
quietly reweighted onto its other five indicators. Preserved bug-for-bug per
NEXT_PROMPT.md Sec3.3 ("this notebook's job is to score what production
actually runs"); the fix belongs in the source repo, not here.
"""

from __future__ import annotations

import pandas as pd

from regime.engine import RegimeInputs
from regime.registry import register


def _frame_column(inputs: RegimeInputs, frame: str, column: str) -> pd.Series:
    source = inputs.macro if frame == "macro" else inputs.cot
    if source is None or column not in source:
        return pd.Series(index=inputs.ohlcv.index, dtype=float)
    return source[column].reindex(inputs.ohlcv.index)


@register("macro.vix", {"macro"}, column="VIXCLS")
def vix(inputs: RegimeInputs, column: str = "VIXCLS") -> pd.Series:
    return _frame_column(inputs, "macro", column)


@register("macro.yield_curve", {"macro"}, column="T10Y2Y")
def yield_curve(inputs: RegimeInputs, column: str = "T10Y2Y") -> pd.Series:
    return _frame_column(inputs, "macro", column)


@register("macro.yield_curve_3m10y", {"macro"}, column="T10Y3M")
def yield_curve_3m10y(inputs: RegimeInputs, column: str = "T10Y3M") -> pd.Series:
    return _frame_column(inputs, "macro", column)


@register("macro.hy_oas", {"macro"}, column="BAMLH0A0HYM2")
def hy_oas(inputs: RegimeInputs, column: str = "BAMLH0A0HYM2") -> pd.Series:
    return _frame_column(inputs, "macro", column)


@register("macro.hy_oas_delta", {"macro"}, column="BAMLH0A0HYM2", periods=63)
def hy_oas_delta(
    inputs: RegimeInputs, column: str = "BAMLH0A0HYM2", periods: int = 63
) -> pd.Series:
    return _frame_column(inputs, "macro", column).diff(periods)


@register("macro.ig_oas", {"macro"}, column="BAMLC0A0CM")
def ig_oas(inputs: RegimeInputs, column: str = "BAMLC0A0CM") -> pd.Series:
    return _frame_column(inputs, "macro", column)


@register("macro.fed_funds_delta", {"macro"}, column="DFF", periods=126)
def fed_funds_delta(
    inputs: RegimeInputs, column: str = "DFF", periods: int = 126
) -> pd.Series:
    return _frame_column(inputs, "macro", column).diff(periods)


@register("macro.cot_noncomm", {"macro"}, column="noncomm_net_pct_oi")
def cot_noncomm(inputs: RegimeInputs, column: str = "noncomm_net_pct_oi") -> pd.Series:
    return _frame_column(inputs, "cot", column)
