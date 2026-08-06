"""Config-driven current market-regime detection.

Ported verbatim from ``ultron_finance.regime.engine``
(``../ultron/libs/finance/src/ultron_finance/regime/engine.py``), except the
source's ``ultron_logging.get_logger`` is replaced with the stdlib
``logging`` module -- that dependency is not vendored into this repo.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import pandas as pd

from regime.config import RegimeConfig
from regime.registry import get
from regime.scoring import (
    combine,
    label_with_hysteresis,
    linear_score,
    percentile_to_score,
    rolling_percentile,
    rolling_zscore,
    smooth,
    squash_z,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegimeInputs:
    ohlcv: pd.DataFrame
    curve: pd.DataFrame | None = None
    macro: pd.DataFrame | None = None
    cot: pd.DataFrame | None = None


@dataclass(frozen=True)
class RegimeResult:
    scores: pd.DataFrame
    labels: pd.DataFrame
    indicators: pd.DataFrame
    contributions: pd.DataFrame
    config: RegimeConfig

    def composite_label(
        self, dims: Sequence[str] | None = None, sep: str = "|"
    ) -> pd.Series:
        selected = list(dims) if dims is not None else list(self.labels.columns)
        return self.labels[selected].astype("string").agg(sep.join, axis=1)

    def latest(self) -> dict[str, tuple[str, float]]:
        latest: dict[str, tuple[str, float]] = {}
        for dimension in map(str, self.scores.columns):
            score = self.scores[dimension].dropna()
            labels = self.labels[dimension].dropna()
            if not score.empty and not labels.empty:
                latest[dimension] = (str(labels.iloc[-1]), float(score.iloc[-1]))
        return latest


class RegimeEngine:
    def __init__(self, config: RegimeConfig) -> None:
        self.config = config

    @classmethod
    def from_yaml(cls, path: Path) -> RegimeEngine:
        return cls(RegimeConfig.from_yaml(path))

    @classmethod
    def from_default(cls, name: str) -> RegimeEngine:
        resource = files("regime").joinpath("configs", f"{name}.yaml")
        with resource.open("r") as handle:
            import yaml

            return cls(RegimeConfig.model_validate(yaml.safe_load(handle)))

    def detect(self, inputs: RegimeInputs) -> RegimeResult:
        index = inputs.ohlcv.index
        raw_frames: list[pd.Series] = []
        contribution_frames: list[pd.Series] = []
        scores: dict[str, pd.Series] = {}
        labels: dict[str, pd.Series] = {}
        for dimension in self.config.dimensions:
            if not dimension.enabled:
                continue
            values: dict[str, pd.Series] = {}
            skipped = False
            for indicator_config in dimension.indicators:
                indicator, meta = get(indicator_config.name)
                if any(
                    getattr(inputs, requirement) is None
                    for requirement in meta.requires
                ):
                    logger.warning(
                        "regime dimension skipped because input is absent: "
                        "dimension=%s indicator=%s",
                        dimension.key,
                        meta.key,
                    )
                    skipped = True
                    break
                params = dict(meta.default_params) | indicator_config.params
                raw = indicator(inputs, **params).reindex(index)
                raw.name = f"{dimension.key}.{indicator_config.name}"
                raw_frames.append(raw)
                scaling = indicator_config.scaling
                if scaling.method == "zscore":
                    scaled = squash_z(
                        rolling_zscore(raw, scaling.window, scaling.min_periods),
                        scaling.scale,
                    )
                elif scaling.method == "percentile":
                    scaled = percentile_to_score(
                        rolling_percentile(raw, scaling.window, scaling.min_periods)
                    )
                elif scaling.method == "linear":
                    scaled = linear_score(raw, scaling.center, scaling.half_range)
                else:
                    scaled = raw.clip(-1, 1)
                values[indicator_config.name] = scaled * indicator_config.direction
            if skipped:
                continue
            scaled_frame = pd.DataFrame(values, index=index)
            weights = {item.name: item.weight for item in dimension.indicators}
            score = smooth(
                combine(scaled_frame, weights, dimension.min_coverage),
                dimension.smoothing_span,
            )
            scores[dimension.key] = score
            labels[dimension.key] = label_with_hysteresis(
                score, dimension.bands, dimension.hysteresis_margin, dimension.min_dwell
            )
            contribution_frames.extend(
                (scaled_frame[str(name)] * weights[str(name)]).rename(
                    f"{dimension.key}.{name}"
                )
                for name in scaled_frame.columns
            )
        score_frame = pd.DataFrame(scores, index=index)
        label_frame = pd.DataFrame(labels, index=index).astype("string")
        if self.config.shift:
            score_frame = score_frame.shift(self.config.shift)
            label_frame = label_frame.shift(self.config.shift).astype("string")
        return RegimeResult(
            scores=score_frame,
            labels=label_frame,
            indicators=pd.concat(raw_frames, axis=1)
            if raw_frames
            else pd.DataFrame(index=index),
            contributions=pd.concat(contribution_frames, axis=1)
            if contribution_frames
            else pd.DataFrame(index=index),
            config=self.config,
        )
