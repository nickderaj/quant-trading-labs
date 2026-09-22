"""Shared matplotlib styling for notebook 025's charts.

Palette: this repo's validated default categorical palette (dataviz skill,
references/palette.md), fixed slot order, never cycled/reassigned per chart.
Surface/text/gridlines match the skill's light-mode tokens. Copied from
viz24.py; extended with the instrument/structure colour maps this notebook
needs (024's SECTOR_COLOR/PRODUCT_COLOR are not meaningful here).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

DAYS_PER_YEAR = 365.25

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

# The premium caption every chart showing a model price must carry.
MODEL_PRICE_CAPTION = "model price, from realised vol -- not a market quote"

# Structure colours -- assigned once, the same structure is the same colour
# in every figure in the notebook (payoff diagrams: hedged line solid, the
# unhedged line always dashed grey, the strike/barrier a thin vertical rule).
STRUCTURE_COLOR: dict[str, str] = {
    "unhedged": GRAY,
    "futures": VIOLET,
    "vanilla": ORANGE,
    "collar": AQUA,
    "barrier": BLUE,
    "asian": YELLOW,
    "spread": MAGENTA,
    "quanto": GREEN,
    "note": RED,
}

# Pricing-method colours -- consistent across Part C's cross-validation and
# convergence figures.
METHOD_COLOR: dict[str, str] = {
    "black76": ORANGE,
    "bachelier": BLUE,
    "displaced": AQUA,
    "binomial": VIOLET,
    "pde": YELLOW,
    "mc": MAGENTA,
    "lsm": RED,
    "markov_functional": GREEN,
    "bootstrap": GRAY,
}

_INSTRUMENT_PALETTE = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]


def build_instrument_color(instruments: list[str]) -> dict[str, str]:
    return {
        p: _INSTRUMENT_PALETTE[i % len(_INSTRUMENT_PALETTE)]
        for i, p in enumerate(instruments)
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


def days_to_years(days) -> np.ndarray:
    """Convert a days-to-expiry array/scalar to years, for the x-axis."""
    return np.asarray(days, dtype=float) / DAYS_PER_YEAR


def style_years_axis(ax, axis: str = "x") -> None:
    """On a log-scaled years-to-expiry axis, show plain numbers (0.5, 1, 5, 10)
    instead of matplotlib's default power-of-ten tick labels (10^0, 10^1)."""
    which = ax.xaxis if axis == "x" else ax.yaxis
    ticks = [t for t in (0.1, 0.25, 0.5, 1, 2, 3, 5, 7, 10) if t > 0]
    which.set_major_locator(mticker.FixedLocator(ticks))
    which.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    which.set_minor_formatter(mticker.NullFormatter())


def style_log_axis_plain(ax, axis: str = "y") -> None:
    """On ANY log-scaled axis whose values are not years (volatility, return
    density, premium, ...), replace matplotlib's default power-of-ten +
    2x/5x-multiple tick labels with plain decimal numbers. Call AFTER
    set_xscale/set_yscale("log")."""
    which = ax.xaxis if axis == "x" else ax.yaxis
    which.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    which.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    which.set_minor_formatter(mticker.NullFormatter())


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


def mark_barrier(ax, level: float, label: str = "barrier", color: str = RED) -> None:
    """A thin vertical/horizontal rule marking a strike or barrier level, with
    a labelled annotation -- used on every payoff/premium chart in Part A."""
    ax.axhline(level, color=color, linewidth=1.0, linestyle="--", zorder=1)
    ax.text(
        ax.get_xlim()[1],
        level,
        f"  {label}",
        color=color,
        fontsize=8,
        va="center",
        ha="left",
    )


def price_caption(ax) -> None:
    """Stamp the mandatory honesty caption on every chart showing a premium."""
    ax.text(
        0.0,
        -0.18,
        MODEL_PRICE_CAPTION,
        transform=ax.transAxes,
        color=TEXT_SECONDARY,
        fontsize=7.5,
        style="italic",
    )
