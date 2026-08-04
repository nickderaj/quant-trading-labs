"""Walk-forward-safe score transformations and stateful labelling.

Ported verbatim from ``ultron_finance.regime.scoring``
(``../ultron/libs/finance/src/ultron_finance/regime/scoring.py``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from regime.config import LabelBand


def rolling_zscore(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    rolling = s.rolling(window, min_periods=min_periods or window)
    return (s - rolling.mean()) / rolling.std().replace(0, np.nan)


def rolling_percentile(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    required = min_periods or window

    def rank(values: np.ndarray[Any, np.dtype[np.float64]]) -> float:
        valid = values[~np.isnan(values)]
        return (
            float((valid <= values[-1]).sum() / len(valid))
            if len(valid) and not np.isnan(values[-1])
            else np.nan
        )

    return s.rolling(window, min_periods=required).apply(rank, raw=True)


def squash_z(z: pd.Series, scale: float = 2.0) -> pd.Series:
    return pd.Series(np.tanh(z / scale), index=z.index, name=z.name)


def percentile_to_score(p: pd.Series) -> pd.Series:
    return 2 * p - 1


def linear_score(raw: pd.Series, center: float, half_range: float) -> pd.Series:
    return ((raw - center) / half_range).clip(-1, 1)


def combine(
    scores: pd.DataFrame, weights: Mapping[str, float], min_coverage: float = 0.5
) -> pd.Series:
    columns = [str(column) for column in scores if str(column) in weights]
    weight_series = pd.Series({column: weights[column] for column in columns}, dtype=float)
    available = scores[columns].notna().mul(weight_series, axis=1).sum(axis=1)
    total = weight_series.sum()
    combined = scores[columns].mul(weight_series, axis=1).sum(axis=1) / available.replace(0, np.nan)
    return combined.where(available >= min_coverage * total)


def smooth(score: pd.Series, span: int) -> pd.Series:
    return score.ewm(span=span, adjust=False).mean()


def _instantaneous(value: float, bands: Sequence[LabelBand]) -> str:
    for band in bands:
        lower = -float("inf") if band.lower is None else band.lower
        upper = float("inf") if band.upper is None else band.upper
        if lower <= value < upper:
            return band.label
    raise ValueError(f"Score {value} is outside configured label bands")


def _inside_expanded(value: float, label: str, bands: Sequence[LabelBand], margin: float) -> bool:
    band = next(item for item in bands if item.label == label)
    lower = -float("inf") if band.lower is None else band.lower - margin
    upper = float("inf") if band.upper is None else band.upper + margin
    return lower <= value < upper


def label_with_hysteresis(
    score: pd.Series,
    bands: Sequence[LabelBand],
    hysteresis_margin: float = 0.1,
    min_dwell: int = 5,
) -> pd.Series:
    """Label scores without flapping, using only current and prior observations."""
    result = pd.Series(pd.NA, index=score.index, dtype="string")
    current: str | None = None
    pending: str | None = None
    dwell = 0
    for position, value in enumerate(score):
        if pd.isna(value):
            continue
        instantaneous = _instantaneous(float(value), bands)
        if current is None:
            current = instantaneous
        elif instantaneous == current or _inside_expanded(
            float(value), current, bands, hysteresis_margin
        ):
            pending, dwell = None, 0
        elif instantaneous == pending:
            dwell += 1
        else:
            pending, dwell = instantaneous, 1
        if pending is not None and dwell >= min_dwell:
            current, pending, dwell = pending, None, 0
        result.iloc[position] = current
    return result
