from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regime.forecast_eval import (
    purged_embargoed_walk_forward_splits,
    walk_forward_splits,
)


def _jump_frame(
    n: int, horizon: int, jump_pos: int, jump_size: float = 500.0
) -> pd.Series:
    """A driftless random walk with one large deterministic jump inserted at
    ``jump_pos``. sign(forward h-day return) is ~50/50 everywhere except in
    the +-horizon neighbourhood of the jump, where every overlapping label
    window is dominated by the same single event -- exactly the "last h
    training rows have labels that resolve inside the test window" failure
    mode: the jump sits just past the train/test boundary, so both train's
    tail labels and test's head labels are scoring the *same* event, not
    independent draws."""
    rng = np.random.default_rng(0)
    steps = rng.standard_normal(n)
    steps[jump_pos] += jump_size
    return pd.Series(
        np.cumsum(steps), index=pd.date_range("2000-01-01", periods=n, freq="B")
    )


def _majority_vote_model(train_target: pd.Series, tail: int) -> str:
    """The naive "model": whatever a fold-blind majority-class classifier
    would learn from the last `tail` training labels -- no real features,
    just what a memorising/overfit model picks up from recent training
    labels when they happen to share a dominant driving event with the
    test fold's own labels."""
    recent = train_target.dropna().iloc[-tail:]
    if recent.empty:
        return "0"
    return cast(str, recent.mode().iloc[0])


def test_purge_and_embargo_remove_the_shared_event_contamination() -> None:
    horizon = 63
    n = 1200
    min_train, test_size, step = 500, 200, 200
    train_end = min_train - 1
    # Place the jump halfway through the horizon window straddling the
    # train/test boundary, so it sits inside the forward-return window of
    # both the last ~h/2 training rows and the first ~h/2 test rows --
    # exactly the shared-event contamination purge/embargo removes.
    jump_pos = train_end + horizon // 2
    value = _jump_frame(n, horizon, jump_pos)
    with np.errstate(invalid="ignore"):
        fwd = value.shift(-horizon) - value
    target = fwd.apply(
        lambda x: "up" if x > 0 else ("down" if x < 0 else "flat")
    ).astype("string")
    target[fwd.isna()] = pd.NA  # type: ignore[call-overload]

    value_index = pd.DatetimeIndex(value.index)
    raw_folds = walk_forward_splits(value_index, min_train, test_size, step)
    clean_folds = purged_embargoed_walk_forward_splits(
        value_index, min_train, test_size, horizon, step
    )
    assert raw_folds and clean_folds

    train_idx, raw_test_idx = raw_folds[0]
    _, clean_test_idx = clean_folds[0]
    naive_pred = _majority_vote_model(target.loc[train_idx], tail=2 * horizon)

    def _accuracy(test_idx: pd.DatetimeIndex) -> float:
        t = target.loc[test_idx].dropna()
        if t.empty:
            return float("nan")
        return float((t.to_numpy() == naive_pred).mean())

    # The contamination is concentrated in the first ~h/2 rows of the
    # unpurged test fold (the rows whose forward window still reaches back
    # to the shared jump) -- exactly the rows the embargo removes -- so
    # that is where the leak's effect is measured, not diluted across the
    # whole (mostly-unaffected) fold.
    contaminated = jump_pos - train_end
    raw_accuracy = _accuracy(raw_test_idx[:contaminated])
    clean_accuracy = _accuracy(clean_test_idx)

    # Unpurged: the test fold's head still contains the rows whose forward
    # window covers the same jump that dominated train's tail labels, so a
    # model that learned nothing but "recent training labels lean 'up'"
    # scores well above chance on them by construction, not genuine skill.
    assert raw_accuracy > 0.9
    # Purged + embargoed: those contaminated rows are gone from both sides
    # of the boundary, so the same naive model is back to chance on what's
    # left (a driftless random walk away from the single jump).
    assert 0.35 <= clean_accuracy <= 0.65


def test_purged_splits_drop_final_horizon_train_rows() -> None:
    idx = pd.date_range("2000-01-01", periods=800, freq="B")
    raw = walk_forward_splits(idx, min_train=300, test_size=100, step=100)
    clean = purged_embargoed_walk_forward_splits(
        idx, min_train=300, test_size=100, horizon=21, step=100
    )
    assert len(clean) == len(raw)
    for (raw_train, raw_test), (clean_train, clean_test) in zip(
        raw, clean, strict=True
    ):
        assert len(clean_train) == len(raw_train) - 21
        assert len(clean_test) == len(raw_test) - 21
        assert clean_train.equals(raw_train[:-21])
        assert clean_test.equals(raw_test[21:])


def test_purged_splits_reject_nonpositive_horizon() -> None:
    idx = pd.date_range("2000-01-01", periods=100, freq="B")
    with pytest.raises(ValueError):
        purged_embargoed_walk_forward_splits(idx, min_train=50, test_size=20, horizon=0)
