"""Point-in-time regime forecasting from re-weighted per-indicator scores.

Ported verbatim from ``ultron_finance.regime.prediction``
(``../ultron/libs/finance/src/ultron_finance/regime/prediction.py``).

Given the engine's per-indicator *scaled* scores at day t (data <= t), a candidate
weight vector produces a leading composite score ``s_t`` in [-1, 1]. Banding ``s_t``
through the dimension's thresholds yields a predicted label for t+h. Optionally the
one-hot prediction is blended with a walk-forward Markov prior. Baselines
(persistence, Markov, class prior) live here too so callers score everything the
same way.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from regime.aggregate import aggregate_scores
from regime.config import DimensionConfig
from regime.engine import RegimeResult
from regime.scoring import (
    _instantaneous,
    combine,
    label_with_hysteresis,
    smooth,
)

_PERIOD_UNITS = {"d": 1, "w": 5, "m": 21, "q": 63, "y": 252}
_PERIOD_RE = re.compile(r"^(\d+)([dwmqy])$")


def horizon_to_days(horizon: int | str) -> int:
    """Normalise a horizon to trading days. Accepts an int, or a period string
    like ``"5d"``, ``"1w"``, ``"1m"``, ``"6m"``, ``"1y"``, ``"10y"``
    (month=21, quarter=63, year=252 trading days)."""
    if isinstance(horizon, int):
        if horizon < 1:
            raise ValueError("horizon must be positive")
        return horizon
    match = _PERIOD_RE.match(horizon.strip().lower())
    if not match:
        raise ValueError(f"Unrecognised horizon: {horizon!r} (use e.g. '1m', '1y')")
    return int(match.group(1)) * _PERIOD_UNITS[match.group(2)]


class ForecastConfig(BaseModel):
    dimension: str
    horizon: int | str
    weights: dict[str, float]
    markov_alpha: float = Field(default=0.0, ge=0, le=1)
    markov_min_history: int = Field(default=252, gt=0)
    use_hysteresis: bool = True
    smoothing_span: int | None = Field(default=None, gt=0)

    @property
    def horizon_days(self) -> int:
        return horizon_to_days(self.horizon)


@dataclass(frozen=True)
class ForecastResult:
    score: pd.Series  # leading composite s_t in [-1, 1], indexed at t
    labels: pd.Series  # predicted label for t+h, indexed at t
    probs: pd.DataFrame  # per-class probabilities, rows sum to 1, indexed at t
    horizon: int
    dimension: str


# --------------------------------------------------------------------------- #
# Adapters: recover scaled per-indicator frames
# --------------------------------------------------------------------------- #
def _dimension_config(result: RegimeResult, dimension: str) -> DimensionConfig:
    for dim in result.config.dimensions:
        if dim.key == dimension:
            return dim
    raise KeyError(f"dimension {dimension!r} not in config")


def scaled_indicator_frame(result: RegimeResult, dimension: str) -> pd.DataFrame:
    """Recover per-indicator scaled scores (direction applied, weight removed)
    from ``result.contributions``. Columns are indicator names (e.g. 'macro.vix')
    so they match the weight keys ``combine`` expects."""
    if result.config.shift != 0:
        raise ValueError(
            "prediction requires shift == 0; contributions are not shifted "
            "(engine.py) so a non-zero shift would misalign features and labels"
        )
    dim = _dimension_config(result, dimension)
    frame: dict[str, pd.Series] = {}
    for ind in dim.indicators:
        col = f"{dimension}.{ind.name}"
        if col in result.contributions:
            frame[ind.name] = result.contributions[col] / ind.weight
    return pd.DataFrame(frame, index=result.contributions.index)


def basket_scaled_frame(
    results: Mapping[str, RegimeResult],
    dimension: str,
    symbol_weights: Mapping[str, float] | None = None,
    min_coverage: float = 0.5,
) -> pd.DataFrame:
    """Aggregate per-symbol scaled-indicator frames into one basket-level frame,
    reusing ``aggregate_scores`` (NaN-aware weighted cross-symbol mean)."""
    per_symbol = {
        symbol: scaled_indicator_frame(result, dimension)
        for symbol, result in results.items()
    }
    return aggregate_scores(per_symbol, symbol_weights, "mean", min_coverage)


def base_weights(result: RegimeResult, dimension: str) -> dict[str, float]:
    """The config's own indicator weights for a dimension."""
    dim = _dimension_config(result, dimension)
    return {ind.name: ind.weight for ind in dim.indicators}


# --------------------------------------------------------------------------- #
# Leading-score forecaster
# --------------------------------------------------------------------------- #
def forecast_scores(
    scaled: pd.DataFrame,
    dim_config: DimensionConfig,
    weights: Mapping[str, float],
    smoothing_span: int | None = None,
) -> pd.Series:
    """combine(weighted, min_coverage) then EMA smooth. With the config's own
    weights and span this reproduces ``result.scores[dimension]`` exactly."""
    combined = combine(scaled, weights, dim_config.min_coverage)
    return smooth(combined, smoothing_span or dim_config.smoothing_span)


def _band_labels(dim_config: DimensionConfig) -> list[str]:
    return [band.label for band in dim_config.bands]


def forecast_labels(
    score: pd.Series, dim_config: DimensionConfig, use_hysteresis: bool = True
) -> pd.Series:
    if use_hysteresis:
        return label_with_hysteresis(
            score, dim_config.bands, dim_config.hysteresis_margin, dim_config.min_dwell
        )
    labels = pd.Series(pd.NA, index=score.index, dtype="string")
    for pos, value in enumerate(score):
        if pd.notna(value):
            labels.iloc[pos] = _instantaneous(float(value), dim_config.bands)
    return labels


def _one_hot(labels: pd.Series, states: Sequence[str]) -> pd.DataFrame:
    onehot = pd.DataFrame(0.0, index=labels.index, columns=list(states))
    for pos, label in enumerate(labels):
        if pd.notna(label) and label in onehot.columns:
            col_pos = cast(int, onehot.columns.get_loc(str(label)))
            onehot.iloc[pos, col_pos] = 1.0
    mask = labels.notna().to_numpy()
    onehot.loc[~mask] = np.nan
    return onehot


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def persistence_forecast(labels: pd.Series, horizon: int) -> pd.Series:
    """Predict that today's label persists to t+h (indexed at t)."""
    return labels.astype("string").copy()


def markov_forecast(
    labels: pd.Series,
    horizon: int,
    states: Sequence[str],
    min_history: int = 252,
) -> pd.DataFrame:
    """Walk-forward Markov forecast: at each t, a row-stochastic transition matrix
    is estimated from transitions observed <= t (incremental counts), raised to
    ``horizon``, and the row for label_t is returned. Columns are the fixed
    ``states`` (band labels) so no future vocabulary leaks in. Rows before
    ``min_history`` valid observations are NaN."""
    states = list(states)
    idx = {state: i for i, state in enumerate(states)}
    counts = np.zeros((len(states), len(states)), dtype=float)
    out = pd.DataFrame(np.nan, index=labels.index, columns=states)
    seen = 0
    prev: str | None = None
    for pos, raw in enumerate(labels):
        if pd.isna(raw):
            prev = None
            continue
        label = str(raw)
        if prev is not None and prev in idx and label in idx:
            counts[idx[prev], idx[label]] += 1.0
        seen += 1
        if seen >= min_history and label in idx:
            row_sums = counts.sum(axis=1, keepdims=True)
            probs = np.divide(
                counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0
            )
            powered = np.linalg.matrix_power(probs, horizon)
            out.iloc[pos] = powered[idx[label]]
        prev = label
    return out


def prior_forecast(
    labels: pd.Series, states: Sequence[str], min_history: int = 252
) -> pd.DataFrame:
    """Expanding class-frequency forecast: at each t, P(class) = frequency over
    labels observed <= t. Matters at long horizons where the Markov power
    converges to the stationary distribution."""
    states = list(states)
    out = pd.DataFrame(np.nan, index=labels.index, columns=states)
    counts = dict.fromkeys(states, 0)
    seen = 0
    for pos, raw in enumerate(labels):
        if pd.isna(raw):
            continue
        label = str(raw)
        if label in counts:
            counts[label] += 1
        seen += 1
        if seen >= min_history:
            total = sum(counts.values())
            if total:
                out.iloc[pos] = [counts[s] / total for s in states]
    return out


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def forecast(
    scaled: pd.DataFrame,
    dim_config: DimensionConfig,
    config: ForecastConfig,
    history_labels: pd.Series | None = None,
) -> ForecastResult:
    """Leading-score forecast, optionally blended with a Markov prior.

    ``probs = (1 - alpha) * one_hot(predicted label) + alpha * markov_row``.
    ``alpha == 0`` is the pure leading-score model; ``alpha == 1`` is exactly the
    existing Markov baseline. ``history_labels`` (the dimension's realized labels)
    is required when ``alpha > 0``."""
    h = config.horizon_days
    states = _band_labels(dim_config)
    score = forecast_scores(scaled, dim_config, config.weights, config.smoothing_span)
    pred = forecast_labels(score, dim_config, config.use_hysteresis)
    probs = _one_hot(pred, states)
    if config.markov_alpha > 0:
        if history_labels is None:
            raise ValueError("history_labels is required when markov_alpha > 0")
        markov = markov_forecast(history_labels, h, states, config.markov_min_history)
        markov = markov.reindex(index=probs.index)
        probs = (1 - config.markov_alpha) * probs + config.markov_alpha * markov
    return ForecastResult(
        score=score, labels=pred, probs=probs, horizon=h, dimension=config.dimension
    )


def forecast_from_result(
    result: RegimeResult, config: ForecastConfig
) -> ForecastResult:
    """Convenience wrapper for the single-symbol case."""
    dim = _dimension_config(result, config.dimension)
    scaled = scaled_indicator_frame(result, config.dimension)
    return forecast(scaled, dim, config, history_labels=result.labels[config.dimension])


__all__ = [
    "ForecastConfig",
    "ForecastResult",
    "base_weights",
    "basket_scaled_frame",
    "forecast",
    "forecast_from_result",
    "forecast_labels",
    "forecast_scores",
    "horizon_to_days",
    "markov_forecast",
    "persistence_forecast",
    "prior_forecast",
    "scaled_indicator_frame",
]
