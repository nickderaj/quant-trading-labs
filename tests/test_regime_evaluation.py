from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regime import RegimeEngine, RegimeInputs
from regime.evaluation import evaluate, label_stability, no_lookahead_check


def _inputs(rows: int = 600) -> RegimeInputs:
    index = pd.date_range("2020-01-01", periods=rows, freq="B")
    close = pd.Series(100.0 * np.exp(np.arange(rows) / 1_000), index=index)
    return RegimeInputs(pd.DataFrame({"open": close, "high": close, "low": close, "close": close}))


def test_evaluation_and_walk_forward_smoke() -> None:
    engine = RegimeEngine.from_default("commodity_default")
    inputs = _inputs()
    summary = evaluate(engine, inputs)

    assert no_lookahead_check(engine, inputs)
    assert summary["config_hash"] == engine.config.config_hash()
    assert {"trend", "volatility", "mean_reversion"} <= set(summary["dimensions"])
    assert set(summary["dimensions"]["trend"]) == {
        "stability",
        "transitions",
        "durations",
        "time_in_regime",
    }


def test_label_stability_handles_missing_values() -> None:
    labels = pd.Series([pd.NA, "bull", "bull", "bear", pd.NA], dtype="string")
    assert label_stability(labels) == {
        "flip_rate": 0.5,
        "avg_duration": 1.5,
        "pct_time_labeled": 0.6,
    }
