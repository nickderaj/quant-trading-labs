from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regime import RegimeEngine, RegimeInputs


def _ohlcv(rows: int = 600) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=rows, freq="B")
    close = pd.Series(np.exp(np.linspace(0, 1, rows)), index=index)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close})


def test_default_engine_detects_and_is_no_lookahead() -> None:
    engine = RegimeEngine.from_default("commodity_default")
    ohlcv = _ohlcv()
    full = engine.detect(RegimeInputs(ohlcv))
    truncated = engine.detect(RegimeInputs(ohlcv.iloc[:595]))

    assert {"trend", "volatility", "mean_reversion"} <= set(full.scores)
    pd.testing.assert_frame_equal(full.scores.iloc[:595], truncated.scores, check_exact=False)
    pd.testing.assert_frame_equal(full.labels.iloc[:595], truncated.labels)


def test_geometric_uptrend_labels_bull() -> None:
    index = pd.date_range("2020-01-01", periods=600, freq="B")
    close = pd.Series(50.0 * np.exp(np.linspace(0, 0.9, 600)), index=index)
    ohlcv = pd.DataFrame(
        {"open": close, "high": close * 1.005, "low": close * 0.995, "close": close}
    )
    result = RegimeEngine.from_default("commodity_default").detect(RegimeInputs(ohlcv))

    assert result.labels["trend"].iloc[-1] == "bull"


def test_sine_wave_labels_mean_reverting() -> None:
    index = pd.date_range("2020-01-01", periods=1200, freq="B")
    close = pd.Series(100.0 + 5.0 * np.sin(2 * np.pi * np.arange(1200) / 10.0), index=index)
    ohlcv = pd.DataFrame(
        {"open": close, "high": close * 1.005, "low": close * 0.995, "close": close}
    )
    result = RegimeEngine.from_default("commodity_default").detect(RegimeInputs(ohlcv))

    assert result.labels["mean_reversion"].iloc[-1] == "mean_reverting"
    assert result.labels["trend"].iloc[-1] == "sideways"


def test_vol_spike_labels_high_or_extreme() -> None:
    rng = np.random.default_rng(7)
    index = pd.date_range("2020-01-01", periods=600, freq="B")
    calm = rng.normal(0, 0.003, 550)
    spike = rng.normal(0, 0.09, 50)
    returns = np.concatenate([calm, spike])
    close = pd.Series(100.0 * np.exp(np.cumsum(returns)), index=index)
    ohlcv = pd.DataFrame({"open": close, "high": close * 1.02, "low": close * 0.98, "close": close})
    result = RegimeEngine.from_default("commodity_default").detect(RegimeInputs(ohlcv))

    assert result.labels["volatility"].iloc[-1] in {"high", "extreme"}


def test_missing_macro_input_skips_dimension_and_omits_column(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ohlcv = _ohlcv()
    with caplog.at_level("WARNING"):
        result = RegimeEngine.from_default("equity_default").detect(RegimeInputs(ohlcv))

    assert "volatility" not in result.scores.columns
    assert "volatility" not in result.labels.columns
    assert {"trend", "mean_reversion"} <= set(result.scores)
    assert any("skipped" in record.message for record in caplog.records)
