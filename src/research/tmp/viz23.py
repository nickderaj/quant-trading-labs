"""Shared matplotlib styling for notebook 023's charts.

Palette: this repo's validated default categorical palette (dataviz skill,
references/palette.md), fixed slot order, never cycled/reassigned per chart.
Surface/text/gridlines match the skill's light-mode tokens.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3de"

# Fixed categorical order -- never reassigned or cycled per chart.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"

# Gabillon/benchmark identity, held fixed across every chart in the notebook.
SERIES_COLOR = {
    "gabillon26": BLUE,
    "flat": ORANGE,
    "prevcurve": VIOLET,
    "spline": AQUA,
    "ns": YELLOW,
    "pca": MAGENTA,
}
SERIES_LABEL = {
    "gabillon26": "Gabillon (26)",
    "flat": "Flat forward",
    "prevcurve": "Previous curve",
    "spline": "Cubic spline",
    "ns": "Nelson-Siegel",
    "pca": "PCA (2-factor)",
}

# Diverging pair for the paired-gap chart: blue<->red, neutral gray midpoint.
DIVERGE_NEG = BLUE  # Gabillon better that day
DIVERGE_POS = RED  # Gabillon worse that day
NEUTRAL = "#f0efec"


def style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(
            title,
            color=TEXT_PRIMARY,
            fontsize=12,
            fontweight="bold",
            loc="left",
            pad=10,
        )
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9)


def new_fig(figsize=(8, 4.5)):
    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    fig.set_facecolor(SURFACE)
    return fig, ax


def legend(ax, **kwargs) -> None:
    leg = ax.legend(
        frameon=False,
        labelcolor=TEXT_PRIMARY,
        fontsize=9,
        **kwargs,
    )
    if leg:
        leg.set_zorder(10)


def clip_for_display(arr, cap):
    """Cap extrapolation blow-ups (e.g. cubic spline outside its fit range)
    at `cap` for display only -- the true values (including inf) are what's
    scored and reported in the write-up; this only prevents one degenerate
    day from crushing the y-axis of a distribution chart."""
    a = np.asarray(arr, dtype=float)
    return np.clip(a, None, cap)
