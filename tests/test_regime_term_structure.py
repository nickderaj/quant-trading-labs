from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regime import RegimeEngine, RegimeInputs


def _inputs(front: float, deferred: float, rows: int = 300) -> RegimeInputs:
    index = pd.date_range("2020-01-01", periods=rows, freq="B")
    close = pd.Series(100.0 + np.arange(rows) * 0.01, index=index)
    ohlcv = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close})
    curve = pd.DataFrame(
        {
            "close_f1": front,
            "close_f2": deferred,
            "dte_f1": 30.0,
            "dte_f2": 60.0,
        },
        index=index,
    )
    return RegimeInputs(ohlcv=ohlcv, curve=curve)


def test_commodity_curve_labels_contango_and_negative_carry() -> None:
    result = RegimeEngine.from_default("commodity_default").detect(_inputs(90.0, 100.0))

    assert result.labels["term_structure"].dropna().iloc[-1] == "contango"
    assert result.labels["carry"].dropna().iloc[-1] == "negative"


def test_commodity_curve_labels_backwardation_and_positive_carry() -> None:
    result = RegimeEngine.from_default("commodity_default").detect(_inputs(110.0, 100.0))

    assert result.labels["term_structure"].dropna().iloc[-1] == "backwardation"
    assert result.labels["carry"].dropna().iloc[-1] == "positive"
