from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regime.config import LabelBand
from regime.scoring import (
    combine,
    label_with_hysteresis,
    linear_score,
    percentile_to_score,
    squash_z,
)


def test_combine_renormalizes_available_weights() -> None:
    scores = pd.DataFrame({"a": [1.0, 1.0], "b": [float("nan"), -1.0]})
    actual = combine(scores, {"a": 1.0, "b": 1.0})
    assert actual.tolist() == [1.0, 0.0]


def test_hysteresis_honours_dwell_and_nan() -> None:
    score = pd.Series([-0.5, 0.4, 0.4, float("nan"), 0.4, 0.4])
    labels = label_with_hysteresis(
        score,
        [LabelBand(label="bear", upper=0), LabelBand(label="bull", lower=0)],
        min_dwell=2,
    )
    assert labels.tolist() == ["bear", "bear", "bull", pd.NA, "bull", "bull"]
    assert linear_score(pd.Series([-10.0, 0.0, 10.0]), 0, 1).tolist() == [-1.0, 0.0, 1.0]


def test_hysteresis_honours_margin_and_initializes_first_bar() -> None:
    bands = [LabelBand(label="bear", upper=0), LabelBand(label="bull", lower=0)]
    labels = label_with_hysteresis(pd.Series([0.5, -0.05, -0.15, -0.15]), bands, 0.1, 2)
    assert labels.tolist() == ["bull", "bull", "bull", "bear"]


def test_score_transforms_are_bounded() -> None:
    assert squash_z(pd.Series([-100.0, 0.0, 100.0])).between(-1, 1).all()
    assert percentile_to_score(pd.Series([0.0, 0.5, 1.0])).tolist() == [-1.0, 0.0, 1.0]
