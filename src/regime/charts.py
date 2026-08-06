"""Render regime-panel charts.

Ported and adapted from
``ultron.apps.trading-labs.common.regime_report.charts``
(``../ultron/apps/trading-labs/common/regime_report/charts.py``). Three
changes from the source, per NEXT_PROMPT.md Sec5:

1. Each renderer is split into a ``*_figure(...) -> Figure`` function plus
   the shared ``to_png(fig) -> bytes`` helper, so notebooks can display
   figures inline; the source's unconditional ``matplotlib.use("Agg")`` at
   import time is dropped -- set the backend at the call site if needed.
2. The palette is unchanged: one categorical colour slot per dimension held
   constant across every chart, the blue<->red diverging pair with a
   neutral grey midpoint for polarity, light-surface chrome.
3. New: ``ribbon_figure`` -- a full-history label ribbon (one horizontal
   band per sector x dimension, coloured by label, spanning the whole
   sample) with optional ground-truth episode spans overlaid. The source
   has no equivalent (its report only ever shows "today"); this is the
   chart Phase 3's accuracy write-up depends on.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import NamedTuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from regime.builder import RegimeReport, SectorRegime

__all__ = [
    "EpisodeSpan",
    "history_figure",
    "render_regime_charts",
    "ribbon_figure",
    "snapshot_figure",
    "to_png",
]

_HISTORY_DAYS = 126  # ~6 trading months

_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"

# Diverging pair (blue <-> red poles, neutral gray midpoint): -1 renders red
# (bear / risk-off / wide), +1 renders blue (bull / risk-on / tight).
_DIVERGING = LinearSegmentedColormap.from_list(
    "regime", ["#e34948", "#f0efec", "#2a78d6"]
)
_NORM = Normalize(vmin=-1.0, vmax=1.0)

# Each dimension owns one categorical slot, identical in every chart, so a
# color always means the same dimension (color follows the entity, not rank).
_DIMENSION_COLORS = {
    "risk": "#2a78d6",  # blue (slot 1)
    "yield_curve": "#008300",  # green (slot 2)
    "credit": "#e87ba4",  # magenta (slot 3)
    "trend": "#eda100",  # yellow (slot 4)
    "volatility": "#1baf7a",  # aqua (slot 5)
    "mean_reversion": "#eb6834",  # orange (slot 6)
    "term_structure": "#4a3aa7",  # violet (slot 7)
    "carry": "#e34948",  # red (slot 8)
}
_DIMENSION_ORDER = tuple(_DIMENSION_COLORS)
_TOP_SECTORS = ("Macro", "Commodities", "FX")


class EpisodeSpan(NamedTuple):
    """A ground-truth regime episode to overlay on the label ribbon."""

    start: str
    end: str
    label: str


def _ordered_dimensions(sectors: list[SectorRegime]) -> list[str]:
    present = {dimension for sector in sectors for dimension in sector.latest}
    ordered = [dimension for dimension in _DIMENSION_ORDER if dimension in present]
    return ordered + sorted(present.difference(ordered))


def _tidy(text: str) -> str:
    return text.replace("_", " ")


def to_png(fig: Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=180,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
    )
    plt.close(fig)
    return buffer.getvalue()


def _heatmap_panel(
    axis: Axes, sectors: list[SectorRegime], show_column_labels: bool
) -> None:
    dimensions = _ordered_dimensions(sectors)
    axis.set_facecolor(_SURFACE)
    axis.set_xlim(0, len(dimensions))
    axis.set_ylim(0, len(sectors))
    axis.invert_yaxis()
    axis.set_xticks([])
    axis.set_yticks([position + 0.5 for position in range(len(sectors))])
    axis.set_yticklabels(
        [sector.name for sector in sectors], fontsize=9, color=_INK_SECONDARY
    )
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    if show_column_labels:
        for column, dimension in enumerate(dimensions):
            axis.text(
                column + 0.5,
                -0.15,
                _tidy(dimension),
                ha="center",
                va="bottom",
                fontsize=9,
                color=_INK_MUTED,
            )
    for row, sector in enumerate(sectors):
        for column, dimension in enumerate(dimensions):
            entry = sector.latest.get(dimension)
            if entry is None:
                continue
            label, score = entry
            # 2px-equivalent surface-colored edge keeps a visible gap
            # between fills, per the mark spec.
            axis.add_patch(
                Rectangle(
                    (column, row),
                    1,
                    1,
                    facecolor=_DIVERGING(_NORM(score)),
                    edgecolor=_SURFACE,
                    linewidth=2,
                )
            )
            ink = "#ffffff" if abs(score) > 0.75 else _INK_PRIMARY
            axis.text(
                column + 0.5,
                row + 0.38,
                f"{score:+.2f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=ink,
            )
            axis.text(
                column + 0.5,
                row + 0.72,
                _tidy(label),
                ha="center",
                va="center",
                fontsize=8,
                color=ink if ink == "#ffffff" else _INK_SECONDARY,
            )


def snapshot_figure(report: RegimeReport) -> Figure:
    macro = [sector for sector in report.sectors if sector.kind == "macro"]
    baskets = [sector for sector in report.sectors if sector.kind == "basket"]
    panels = [group for group in (macro, baskets) if group]
    if not panels:
        raise ValueError("regime report has no sectors to draw")
    row_counts = [len(group) for group in panels]
    fig_height = 1.1 + 0.62 * sum(row_counts) + 0.5 * len(panels)
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(7.2, fig_height),
        height_ratios=[count + 0.45 for count in row_counts],
        facecolor=_SURFACE,
    )
    for axis, group in zip(
        [axes] if len(panels) == 1 else list(axes), panels, strict=True
    ):
        _heatmap_panel(axis, group, show_column_labels=True)
    as_of = report.as_of.isoformat() if report.as_of is not None else "latest"
    fig.suptitle(
        f"Market regime — {as_of}",
        fontsize=13,
        fontweight="bold",
        color=_INK_PRIMARY,
        x=0.02,
        ha="left",
    )
    colorbar = fig.colorbar(
        ScalarMappable(norm=_NORM, cmap=_DIVERGING),
        ax=fig.axes,
        orientation="horizontal",
        fraction=0.03,
        pad=0.04,
        ticks=(-1, 0, 1),
    )
    colorbar.ax.set_xticklabels(
        ["−1  bear / risk-off / low vol", "0", "+1  bull / risk-on / high vol"],
        fontsize=8,
        color=_INK_SECONDARY,
    )
    # matplotlib annotates `Colorbar.outline` imprecisely; it is a Spine here.
    colorbar.outline.set_visible(False)  # type: ignore[operator]
    return fig


def _history_panel(axis: Axes, sector: SectorRegime) -> None:
    axis.set_facecolor(_SURFACE)
    history = sector.score_history.tail(_HISTORY_DAYS)
    for dimension in _ordered_dimensions([sector]):
        if dimension not in history:
            continue
        series = history[dimension].dropna()
        axis.plot(
            series.index,
            series,
            color=_DIMENSION_COLORS.get(dimension, _INK_MUTED),
            linewidth=1.8,
            label=_tidy(dimension),
        )
    axis.set_ylim(-1.05, 1.05)
    axis.set_yticks((-1, 0, 1))
    axis.axhline(0, color=_BASELINE, linewidth=1)
    axis.grid(axis="y", color=_GRIDLINE, linewidth=0.8)
    axis.tick_params(colors=_INK_MUTED, labelsize=8, length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(
        sector.name,
        loc="left",
        fontsize=10,
        fontweight="bold",
        color=_INK_PRIMARY,
        pad=4,
    )
    # Anchored above the plot area (sharing the title row) so the legend can
    # never sit on top of the lines.
    axis.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncols=4,
        frameon=False,
        fontsize=8,
        labelcolor=_INK_SECONDARY,
        handlelength=1.4,
        borderaxespad=0.2,
    )


def history_figure(report: RegimeReport) -> Figure | None:
    sectors = [
        sector
        for name in _TOP_SECTORS
        for sector in report.sectors
        if sector.name == name and not sector.score_history.empty
    ]
    if not sectors:
        return None
    fig, axes = plt.subplots(
        len(sectors),
        1,
        figsize=(7.2, 1.9 * len(sectors) + 0.6),
        sharex=True,
        facecolor=_SURFACE,
    )
    for axis, sector in zip(
        [axes] if len(sectors) == 1 else list(axes), sectors, strict=True
    ):
        _history_panel(axis, sector)
    fig.suptitle(
        "Regime scores — last 6 months",
        fontsize=13,
        fontweight="bold",
        color=_INK_PRIMARY,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _row_label(sector: SectorRegime, dimension: str) -> str:
    return f"{sector.name} · {_tidy(dimension)}"


def _label_color(
    sector: SectorRegime, dimension: str, label: object
) -> tuple[float, float, float, float]:
    """Colour a categorical label by the mean score observed while it was
    active, so the ribbon uses the same diverging scale as every other
    chart without needing the dimension's band config."""
    scores = sector.score_history[dimension]
    labels = sector.label_history[dimension]
    mask = labels == label
    mean_score = scores.where(mask).mean()
    if pd.isna(mean_score):
        return (0.7, 0.7, 0.7, 1.0)
    return _DIVERGING(_NORM(float(mean_score)))


def ribbon_figure(
    report: RegimeReport,
    episodes: Sequence[EpisodeSpan] = (),
    sectors: Sequence[str] | None = None,
) -> Figure:
    """Full-history label ribbon: one horizontal band per sector x dimension,
    coloured by which label was active on each day, spanning the whole
    sample. ``episodes`` (hand-labelled ground-truth spans) are drawn as
    shaded vertical bands behind the ribbon so mislabelled stretches are
    visible at a glance."""
    rows: list[tuple[SectorRegime, str]] = []
    for sector in report.sectors:
        if sectors is not None and sector.name not in sectors:
            continue
        for dimension in _ordered_dimensions([sector]):
            if dimension in sector.label_history:
                rows.append((sector, dimension))
    if not rows:
        raise ValueError("no sector/dimension rows to draw")

    fig_height = 0.9 + 0.34 * len(rows)
    fig, axis = plt.subplots(1, 1, figsize=(11.0, fig_height), facecolor=_SURFACE)
    axis.set_facecolor(_SURFACE)

    all_index = rows[0][0].label_history.index
    for sector, dimension in rows[1:]:
        all_index = all_index.union(sector.label_history.index)
    x_start, x_end = all_index.min(), all_index.max()

    for span in episodes:
        # matplotlib's Axes stubs type x/y bounds as float; datetime x-values
        # are accepted at runtime via the axis's unit converter.
        axis.axvspan(
            pd.Timestamp(span.start),  # type: ignore[arg-type]
            pd.Timestamp(span.end),  # type: ignore[arg-type]
            color=_INK_MUTED,
            alpha=0.12,
            lw=0,
        )

    for row, (sector, dimension) in enumerate(rows):
        labels = sector.label_history[dimension].dropna()
        if labels.empty:
            continue
        # Collapse into contiguous spells so each is drawn as one patch.
        change = labels.ne(labels.shift()).fillna(True).astype("int64").cumsum()
        for _, spell in labels.groupby(change):
            start, end = spell.index[0], spell.index[-1]
            color = _label_color(sector, dimension, spell.iloc[0])
            axis.add_patch(
                Rectangle(
                    (start, row),
                    end - start,
                    1,
                    facecolor=color,
                    edgecolor="none",
                )
            )

    axis.set_xlim(x_start, x_end)
    axis.set_ylim(0, len(rows))
    axis.invert_yaxis()
    axis.set_yticks([r + 0.5 for r in range(len(rows))])
    axis.set_yticklabels(
        [_row_label(sector, dimension) for sector, dimension in rows],
        fontsize=8,
        color=_INK_SECONDARY,
    )
    axis.tick_params(length=0, labelsize=8)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for span in episodes:
        axis.text(
            pd.Timestamp(span.start),  # type: ignore[arg-type]
            -0.3,
            span.label,
            fontsize=7,
            color=_INK_MUTED,
            rotation=90,
            va="bottom",
            ha="left",
        )
    fig.suptitle(
        "Regime labels, full history",
        fontsize=13,
        fontweight="bold",
        color=_INK_PRIMARY,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def render_regime_charts(report: RegimeReport) -> list[tuple[bytes, str]]:
    """Render the report's snapshot + history charts as `(png_bytes, caption)`
    pairs -- kept for parity with the source's public entry point."""
    as_of = report.as_of.isoformat() if report.as_of is not None else "latest"
    charts = [
        (
            to_png(snapshot_figure(report)),
            f"Regime snapshot {as_of} — score −1 (red) … +1 (blue)",
        )
    ]
    history = history_figure(report)
    if history is not None:
        charts.append(
            (to_png(history), "Macro / Commodities / FX regime scores, last 6 months")
        )
    return charts
