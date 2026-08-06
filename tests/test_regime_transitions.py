from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regime.transitions import (
    expected_remaining_duration,
    predict_next,
    regime_durations,
    time_in_regime,
    transition_matrix,
)


def test_hand_computable_transitions_and_predictions() -> None:
    labels = pd.Series(["bull", "bull", "bear", "bear", "bull"], dtype="string")
    counts = transition_matrix(labels, normalize=False)
    matrix = transition_matrix(labels)

    assert counts.to_dict() == {
        "bear": {"bear": 1, "bull": 1},
        "bull": {"bear": 1, "bull": 1},
    }
    assert matrix.loc["bull", "bear"] == 0.5
    assert predict_next("bull", matrix, 2).to_dict() == {"bear": 0.5, "bull": 0.5}
    assert expected_remaining_duration(matrix).to_dict() == {"bear": 2.0, "bull": 2.0}


def test_durations_time_and_absorbing_state() -> None:
    labels = pd.Series(["bull", "bull", "bear", "bear", "bull"], dtype="string")
    durations = regime_durations(labels)

    assert durations.loc["bull"].to_dict() == {
        "n_spells": 2.0,
        "mean": 1.5,
        "median": 1.5,
        "max": 2.0,
    }
    assert time_in_regime(labels).to_dict() == {"bear": 0.4, "bull": 0.6}
    absorbing = pd.DataFrame([[1.0]], index=["flat"], columns=["flat"])
    assert math.isinf(expected_remaining_duration(absorbing).iloc[0])
