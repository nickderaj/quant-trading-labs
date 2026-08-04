"""Daily multi-dimensional market-regime detection engine.

Ported verbatim (module-for-module, function-for-function) from the
production engine at ``../ultron/libs/finance/src/ultron_finance/regime/``
and its report layer at
``../ultron/apps/trading-labs/common/regime_report/``. See
``src/results/014_market_regime_engine_and_accuracy.md`` for the port-fidelity
check and every disclosed deviation (COT gap, curve_slope f3 substitution,
etc). This package is durable infrastructure for future notebooks, not
notebook-014 scratch -- do not move it into ``src/research/tmp/``.

NOTE: the public surface grows across the port's stages (engine core first,
then the loaders/builder/charts adapter layer) -- see
``src/results/014_market_regime_engine_and_accuracy.md`` for status.
"""

from __future__ import annotations

from regime import dimensions as dimensions
from regime.aggregate import aggregate_scores, basket_labels
from regime.builder import RegimeReport, SectorRegime, build_regime_report
from regime.charts import render_regime_charts, ribbon_figure, to_png
from regime.config import (
    DimensionConfig,
    IndicatorConfig,
    LabelBand,
    RegimeConfig,
    ScalingConfig,
)
from regime.engine import RegimeEngine, RegimeInputs, RegimeResult
from regime.evaluation import evaluate, label_stability, no_lookahead_check
from regime.transitions import (
    expected_remaining_duration,
    predict_next,
    regime_durations,
    time_in_regime,
    transition_matrix,
)
from regime.universe import RegimeUniverse, load_regime_universe

__all__ = [
    "DimensionConfig",
    "IndicatorConfig",
    "LabelBand",
    "RegimeConfig",
    "RegimeEngine",
    "RegimeInputs",
    "RegimeReport",
    "RegimeResult",
    "RegimeUniverse",
    "ScalingConfig",
    "SectorRegime",
    "aggregate_scores",
    "basket_labels",
    "build_regime_report",
    "dimensions",
    "evaluate",
    "expected_remaining_duration",
    "label_stability",
    "load_regime_universe",
    "no_lookahead_check",
    "predict_next",
    "regime_durations",
    "render_regime_charts",
    "ribbon_figure",
    "time_in_regime",
    "to_png",
    "transition_matrix",
]
