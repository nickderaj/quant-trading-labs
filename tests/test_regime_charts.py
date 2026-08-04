from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regime.builder import RegimeReport, SectorRegime
from regime.charts import EpisodeSpan, render_regime_charts, ribbon_figure, to_png

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _history(dimensions: list[str], days: int = 40) -> pd.DataFrame:
    index = pd.date_range("2026-05-01", periods=days, freq="B")
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {dimension: np.clip(rng.normal(0, 0.4, days), -1, 1) for dimension in dimensions},
        index=index,
    )


def _labels(history: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            dimension: pd.array(
                ["bear" if v < -0.15 else "bull" if v > 0.15 else "neutral" for v in history[dimension]],
                dtype="string",
            )
            for dimension in history
        },
        index=history.index,
    )


def _sector(name: str, kind: str, latest: dict[str, tuple[str, float]]) -> SectorRegime:
    history = _history(list(latest))
    labels = _labels(history)
    return SectorRegime(name, kind, latest, labels, history, list(latest), [])


def make_report() -> RegimeReport:
    return RegimeReport(
        date(2026, 7, 16),
        [
            _sector(
                "Macro",
                "macro",
                {
                    "risk": ("neutral", 0.12),
                    "yield_curve": ("flat", -0.05),
                    "credit": ("tight", 0.45),
                },
            ),
            _sector(
                "Commodities",
                "basket",
                {
                    "trend": ("sideways", 0.08),
                    "volatility": ("normal", -0.21),
                    "mean_reversion": ("neutral", 0.02),
                },
            ),
            _sector(
                "FX",
                "basket",
                {
                    "trend": ("bear", -0.41),
                    "volatility": ("normal", -0.10),
                    "mean_reversion": ("trending", -0.34),
                },
            ),
            _sector(
                "oil products",
                "basket",
                {
                    "trend": ("sideways", -0.02),
                    "volatility": ("high", 0.41),
                    "mean_reversion": ("neutral", 0.05),
                },
            ),
        ],
        [],
    )


def test_render_regime_charts_returns_snapshot_and_history_pngs() -> None:
    charts = render_regime_charts(make_report())

    assert len(charts) == 2
    for image, caption in charts:
        assert image.startswith(_PNG_MAGIC)
        assert caption
    assert "snapshot" in charts[0][1]
    assert "6 months" in charts[1][1]


def test_render_regime_charts_without_history_only_renders_snapshot() -> None:
    sector = SectorRegime(
        "FX", "basket", {"trend": ("bull", 0.6)}, pd.DataFrame(), pd.DataFrame(), ["6E=F"], []
    )
    charts = render_regime_charts(RegimeReport(date(2026, 7, 16), [sector], []))

    assert len(charts) == 1
    assert charts[0][0].startswith(_PNG_MAGIC)


def test_render_regime_charts_raises_on_empty_report() -> None:
    with pytest.raises(ValueError):
        render_regime_charts(RegimeReport(date(2026, 7, 16), [], []))


def test_ribbon_figure_renders_with_episode_overlay() -> None:
    fig = ribbon_figure(
        make_report(), episodes=[EpisodeSpan("2026-05-15", "2026-05-25", "test episode")]
    )
    png = to_png(fig)

    assert png.startswith(_PNG_MAGIC)


def test_ribbon_figure_raises_when_no_rows_match() -> None:
    with pytest.raises(ValueError):
        ribbon_figure(make_report(), sectors=["does-not-exist"])
