"""Validated, reproducible configuration for regime detection.

Ported verbatim from ``ultron_finance.regime.config``
(``../ultron/libs/finance/src/ultron_finance/regime/config.py``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ScalingConfig(BaseModel):
    method: Literal["zscore", "percentile", "linear", "raw"] = "zscore"
    window: int = Field(default=252, gt=0)
    min_periods: int | None = Field(default=None, gt=0)
    scale: float = Field(default=2.0, gt=0)
    center: float = 0.0
    half_range: float = Field(default=1.0, gt=0)


class IndicatorConfig(BaseModel):
    name: str
    weight: float = Field(default=1.0, gt=0)
    direction: Literal[1, -1] = 1
    params: dict[str, Any] = Field(default_factory=dict)
    scaling: ScalingConfig = Field(default_factory=ScalingConfig)


class LabelBand(BaseModel):
    label: str
    lower: float | None = None
    upper: float | None = None


class DimensionConfig(BaseModel):
    key: str
    enabled: bool = True
    indicators: list[IndicatorConfig] = Field(default_factory=list)
    smoothing_span: int = Field(default=5, gt=0)
    min_coverage: float = Field(default=0.5, ge=0, le=1)
    bands: list[LabelBand] = Field(default_factory=list)
    hysteresis_margin: float = Field(default=0.1, ge=0)
    min_dwell: int = Field(default=5, gt=0)

    @model_validator(mode="after")
    def validate_definition(self) -> DimensionConfig:
        if not self.enabled:
            return self
        if not self.indicators:
            raise ValueError("enabled dimensions require at least one indicator")
        if not self.bands:
            raise ValueError("enabled dimensions require label bands")
        lower = -float("inf")
        for band in self.bands:
            actual_lower = -float("inf") if band.lower is None else band.lower
            actual_upper = float("inf") if band.upper is None else band.upper
            if actual_lower != lower or actual_upper <= actual_lower:
                raise ValueError("bands must be ordered, contiguous, and non-empty")
            lower = actual_upper
        if lower != float("inf"):
            raise ValueError("bands must exhaustively cover [-1, 1]")
        if self.bands[0].lower is not None or self.bands[-1].upper is not None:
            raise ValueError("first/last bands must be unbounded")
        return self


class RegimeConfig(BaseModel):
    version: str
    name: str
    asset_class: str
    shift: int = Field(default=0, ge=0)
    dimensions: list[DimensionConfig]

    @classmethod
    def from_yaml(cls, path: Path) -> RegimeConfig:
        with path.open() as handle:
            payload = yaml.safe_load(handle)
        return cls.model_validate(payload)

    def config_hash(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(encoded.encode()).hexdigest()
