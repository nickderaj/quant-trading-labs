"""Shared matplotlib styling for notebook 024's charts.

Palette: this repo's validated default categorical palette (dataviz skill,
references/palette.md), fixed slot order, never cycled/reassigned per chart.
Surface/text/gridlines match the skill's light-mode tokens. Copied from
viz23.py; SERIES_COLOR/SERIES_LABEL replaced with SECTOR_COLOR/PRODUCT_COLOR.
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
GRAY = "#9a988f"

SECTOR_COLOR = {"energy": ORANGE, "metals": YELLOW, "ags": AQUA, "control": VIOLET}

# Per-product colours, assigned once in PRODUCTS order, never reassigned per
# chart -- the same product is the same colour in every figure in the notebook.
_PRODUCT_PALETTE = [
    BLUE,
    ORANGE,
    AQUA,
    YELLOW,
    MAGENTA,
    GREEN,
    VIOLET,
    RED,
    "#7a5230",
    "#2f8f8f",
    "#b05fd1",
    "#5a7a2a",
    "#c94f8a",
    "#3d5c9c",
    "#a86b1f",
    "#4a9d5f",
]


def build_product_color(products: list[str]) -> dict[str, str]:
    return {
        p: _PRODUCT_PALETTE[i % len(_PRODUCT_PALETTE)] for i, p in enumerate(products)
    }


DIVERGE_NEG = BLUE
DIVERGE_POS = RED
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
    a = np.asarray(arr, dtype=float)
    return np.clip(a, None, cap)


def shade_crisis(
    ax, start: str, end: str, label: str | None = None, color: str = RED
) -> None:
    ax.axvspan(
        np.datetime64(start), np.datetime64(end), color=color, alpha=0.08, zorder=0
    )
    if label:
        ax.text(
            np.datetime64(start),
            ax.get_ylim()[1] * 0.97,
            label,
            color=TEXT_SECONDARY,
            fontsize=7.5,
            va="top",
            rotation=0,
        )
