from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regime import RegimeEngine, RegimeInputs
from regime.aggregate import aggregate_scores, basket_labels


def _inputs(stress: bool, rows: int = 700) -> RegimeInputs:
    index = pd.date_range("2018-01-01", periods=rows, freq="B")
    cycle = np.sin(np.arange(rows) / 19)
    vix = 20.0 + cycle
    hy = 4.0 + cycle * 0.15
    ig = 1.5 + cycle * 0.08
    curve = 0.5 + cycle * 0.12
    curve_3m = 0.6 + cycle * 0.12
    fed = 2.0 + np.arange(rows) * 0.001
    cot = cycle * 0.1
    if stress:
        vix[-100:] = 65.0
        hy[-100:] = 12.0
        ig[-100:] = 5.0
        curve[-100:] = -2.0
        curve_3m[-100:] = -2.0
        fed[-100:] = fed[-101] + np.arange(1, 101) * 0.03
        cot[-100:] = -0.5
    else:
        vix[-100:] = 8.0
        hy[-100:] = 1.0
        ig[-100:] = 0.3
        curve[-100:] = 3.0
        curve_3m[-100:] = 3.0
        fed[-100:] = fed[-101] - np.arange(1, 101) * 0.03
        cot[-100:] = 0.5
    close = pd.Series(100.0 + np.arange(rows) * 0.1, index=index)
    ohlcv = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close}
    )
    macro = pd.DataFrame(
        {
            "VIXCLS": vix,
            "BAMLH0A0HYM2": hy,
            "BAMLC0A0CM": ig,
            "T10Y2Y": curve,
            "T10Y3M": curve_3m,
            "DFF": fed,
        },
        index=index,
    )
    cot_frame = pd.DataFrame({"noncomm_net_pct_oi": cot}, index=index)
    return RegimeInputs(ohlcv=ohlcv, macro=macro, cot=cot_frame)


def test_macro_stress_and_benign_fixtures_have_expected_risk_labels() -> None:
    engine = RegimeEngine.from_default("macro_default")

    assert engine.detect(_inputs(True)).labels["risk"].dropna().iloc[-1] == "risk_off"
    assert engine.detect(_inputs(False)).labels["risk"].dropna().iloc[-1] == "risk_on"


def test_macro_engine_accepts_missing_optional_cot() -> None:
    inputs = _inputs(True)
    result = RegimeEngine.from_default("macro_default").detect(
        RegimeInputs(ohlcv=inputs.ohlcv, macro=inputs.macro)
    )

    assert {"risk", "yield_curve", "credit"} <= set(result.scores)
    assert result.labels["risk"].dropna().iloc[-1] == "risk_off"


def test_basket_aggregation_weights_median_coverage_and_labels() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="B")
    per_symbol = {
        "CL": pd.DataFrame({"risk": [-1.0, 1.0]}, index=index),
        "BZ": pd.DataFrame({"risk": [1.0, np.nan]}, index=index),
        "HO": pd.DataFrame({"risk": [1.0, 1.0]}, index=index),
    }
    weighted = aggregate_scores(per_symbol, {"CL": 2.0, "BZ": 1.0, "HO": 1.0})
    median = aggregate_scores(per_symbol, method="median", min_coverage=0.75)

    assert weighted["risk"].tolist() == [0.0, 1.0]
    assert median["risk"].iloc[0] == 1.0
    assert pd.isna(median["risk"].iloc[1])
    labels = basket_labels(weighted, RegimeEngine.from_default("macro_default").config)
    assert labels["risk"].iloc[0] == "neutral"
