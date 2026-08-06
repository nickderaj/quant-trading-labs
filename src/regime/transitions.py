"""Transition and persistence statistics for regime labels.

Ported verbatim from ``ultron_finance.regime.transitions``
(``../ultron/libs/finance/src/ultron_finance/regime/transitions.py``).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd


def _valid_labels(labels: pd.Series) -> pd.Series:
    return labels.dropna().astype("string")


def transition_matrix(labels: pd.Series, normalize: bool = True) -> pd.DataFrame:
    """Return label-to-label transition counts or a row-stochastic matrix."""
    valid = _valid_labels(labels)
    states = pd.Index(sorted(valid.unique().tolist()), dtype="string")
    matrix = pd.DataFrame(0.0, index=states, columns=states)
    if len(valid) < 2:
        return matrix
    previous = valid.iloc[:-1].to_numpy()
    current = valid.iloc[1:].to_numpy()
    for before, after in zip(previous, current, strict=True):
        matrix.loc[before, after] += 1
    if not normalize:
        return matrix.astype(int)
    return matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def regime_durations(labels: pd.Series) -> pd.DataFrame:
    """Summarise contiguous non-missing label spells by state."""
    valid = _valid_labels(labels)
    if valid.empty:
        return pd.DataFrame(columns=["n_spells", "mean", "median", "max"])
    spell_id = valid.ne(valid.shift()).fillna(True).astype("int64").cumsum()
    spells = valid.groupby(spell_id).agg(label="first", bars="size")
    grouped = spells.groupby("label")["bars"]
    return cast(
        pd.DataFrame,
        grouped.agg(n_spells="count", mean="mean", median="median", max="max"),
    )


def time_in_regime(labels: pd.Series) -> pd.Series:
    """Return each label's fraction of all non-missing labelled bars."""
    valid = _valid_labels(labels)
    return valid.value_counts(normalize=True, sort=False).sort_index()


def expected_remaining_duration(matrix: pd.DataFrame) -> pd.Series:
    """Estimate expected remaining spell duration from diagonal persistence."""
    if not matrix.index.equals(matrix.columns):
        raise ValueError("matrix must have identical index and columns")
    diagonal = pd.Series(np.diag(matrix.to_numpy()), index=matrix.index, dtype=float)
    return 1.0 / (1.0 - diagonal)


def predict_next(
    current_label: str, matrix: pd.DataFrame, horizon: int = 1
) -> pd.Series:
    """Return the baseline probability distribution after ``horizon`` transitions."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if current_label not in matrix.index or not matrix.index.equals(matrix.columns):
        raise ValueError("current_label and matrix states must match")
    powered = np.linalg.matrix_power(matrix.to_numpy(dtype=float), horizon)
    return pd.Series(powered[matrix.index.get_loc(current_label)], index=matrix.columns)
