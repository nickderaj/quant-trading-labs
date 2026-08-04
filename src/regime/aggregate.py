"""Aggregate per-symbol regime scores into a basket-level regime.

Ported verbatim from ``ultron_finance.regime.aggregate``
(``../ultron/libs/finance/src/ultron_finance/regime/aggregate.py``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import pandas as pd

from regime.config import RegimeConfig
from regime.scoring import label_with_hysteresis


def aggregate_scores(
    per_symbol: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float] | None = None,
    method: Literal["mean", "median"] = "mean",
    min_coverage: float = 0.5,
) -> pd.DataFrame:
    """Aggregate aligned per-symbol score frames, respecting availability coverage."""
    if not per_symbol:
        return pd.DataFrame()
    if method not in {"mean", "median"}:
        raise ValueError("method must be 'mean' or 'median'")
    if not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage must be between zero and one")
    unknown = set(weights or ()).difference(per_symbol)
    if unknown:
        raise ValueError(f"weights contains unknown symbols: {sorted(unknown)}")
    symbol_weights = pd.Series({key: (weights or {}).get(key, 1.0) for key in per_symbol})
    if (symbol_weights <= 0).any():
        raise ValueError("weights must be positive")
    dimensions = sorted({str(column) for frame in per_symbol.values() for column in frame})
    result: dict[str, pd.Series] = {}
    for dimension in dimensions:
        values = pd.concat(
            {
                symbol: frame[dimension]
                for symbol, frame in per_symbol.items()
                if dimension in frame
            },
            axis=1,
            sort=True,
        )
        if method == "median":
            score = values.median(axis=1)
            coverage = values.notna().mean(axis=1)
        else:
            available_weights = (
                values.notna().mul(symbol_weights.reindex(values.columns), axis=1).sum(axis=1)
            )
            score = values.mul(symbol_weights.reindex(values.columns), axis=1).sum(axis=1)
            score = score / available_weights.replace(0, float("nan"))
            coverage = available_weights / symbol_weights.reindex(values.columns).sum()
        result[dimension] = score.where(coverage >= min_coverage)
    return pd.DataFrame(result)


def basket_labels(agg_scores: pd.DataFrame, config: RegimeConfig) -> pd.DataFrame:
    """Apply configured bands and hysteresis to aggregate scores."""
    labels: dict[str, pd.Series] = {}
    for dimension in config.dimensions:
        if dimension.enabled and dimension.key in agg_scores:
            labels[dimension.key] = label_with_hysteresis(
                agg_scores[dimension.key],
                dimension.bands,
                dimension.hysteresis_margin,
                dimension.min_dwell,
            )
    return pd.DataFrame(labels, index=agg_scores.index).astype("string")
