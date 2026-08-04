"""Walk-forward checks and descriptive evaluation for regime engines.

Ported verbatim from ``ultron_finance.regime.evaluation``
(``../ultron/libs/finance/src/ultron_finance/regime/evaluation.py``).
``no_lookahead_check`` is the load-bearing structural correctness gate: it
re-runs detection on truncated history and asserts the retained rows are
bit-identical, proving nothing in the pipeline peeks past the current bar.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
from pandas.testing import assert_frame_equal

from regime.engine import RegimeEngine, RegimeInputs
from regime.transitions import regime_durations, time_in_regime, transition_matrix


def no_lookahead_check(
    engine: RegimeEngine,
    inputs: RegimeInputs,
    truncations: Sequence[int] = (1, 5, 21),
    atol: float = 1e-9,
) -> bool:
    """Return whether every truncated detection agrees with the full history."""
    full = engine.detect(inputs)
    for truncation in truncations:
        if truncation < 1 or truncation >= len(inputs.ohlcv):
            raise ValueError("truncations must be positive and shorter than the input")
        shortened = RegimeInputs(
            ohlcv=inputs.ohlcv.iloc[:-truncation],
            curve=None if inputs.curve is None else inputs.curve.iloc[:-truncation],
            macro=None if inputs.macro is None else inputs.macro.iloc[:-truncation],
            cot=None if inputs.cot is None else inputs.cot.iloc[:-truncation],
        )
        candidate = engine.detect(shortened)
        try:
            assert_frame_equal(full.scores.iloc[:-truncation], candidate.scores, atol=atol)
            assert_frame_equal(full.labels.iloc[:-truncation], candidate.labels)
        except AssertionError:
            return False
    return True


def label_stability(labels: pd.Series) -> dict[str, float]:
    """Measure flip rate, average spell duration, and labelled coverage."""
    valid = labels.dropna().astype("string")
    if valid.empty:
        return {"flip_rate": 0.0, "avg_duration": 0.0, "pct_time_labeled": 0.0}
    changes = valid.ne(valid.shift()).fillna(True).astype("int64")
    flips = int(changes.sum() - 1)
    spells = flips + 1
    return {
        "flip_rate": flips / max(len(valid) - 1, 1),
        "avg_duration": len(valid) / spells,
        "pct_time_labeled": len(valid) / len(labels),
    }


def evaluate(engine: RegimeEngine, inputs: RegimeInputs) -> dict[str, Any]:
    """Bundle per-dimension persistence diagnostics with the config identity."""
    result = engine.detect(inputs)
    dimensions: dict[str, dict[str, Any]] = {}
    for dimension in result.labels:
        labels = result.labels[dimension]
        dimensions[str(dimension)] = {
            "stability": label_stability(labels),
            "transitions": transition_matrix(labels),
            "durations": regime_durations(labels),
            "time_in_regime": time_in_regime(labels),
        }
    return {"config_hash": engine.config.config_hash(), "dimensions": dimensions}
