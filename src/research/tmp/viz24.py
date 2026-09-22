"""Shared matplotlib styling for notebook 024's charts.

Palette: this repo's validated default categorical palette (dataviz skill,
references/palette.md), fixed slot order, never cycled/reassigned per chart.
Surface/text/gridlines match the skill's light-mode tokens. Copied from
viz23.py; SERIES_COLOR/SERIES_LABEL replaced with SECTOR_COLOR/PRODUCT_COLOR.
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

# energy/metals were both ORANGE/YELLOW originally -- too close in hue and
# lightness to tell apart at a glance (reported as illegible). BLUE gives
# metals a hue clearly separated from energy's orange and ags' teal-green.
SECTOR_COLOR = {"energy": ORANGE, "metals": BLUE, "ags": AQUA, "control": VIOLET}

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


def days_to_years(days) -> np.ndarray:
    """Convert a days-to-expiry array/scalar to years, for the x-axis."""
    return np.asarray(days, dtype=float) / DAYS_PER_YEAR


def style_years_axis(ax, axis: str = "x") -> None:
    """On a log-scaled years-to-expiry axis, show plain numbers (0.5, 1, 5, 10)
    instead of matplotlib's default power-of-ten tick labels (10^0, 10^1) --
    reported as hard to read at a glance for a non-technical audience."""
    which = ax.xaxis if axis == "x" else ax.yaxis
    ticks = [t for t in (0.1, 0.25, 0.5, 1, 2, 3, 5, 7, 10) if t > 0]
    which.set_major_locator(mticker.FixedLocator(ticks))
    which.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    which.set_minor_formatter(mticker.NullFormatter())


def style_log_axis_plain(ax, axis: str = "y") -> None:
    """On ANY log-scaled axis whose values are not years (volatility, return
    density, kurtosis, ...), replace matplotlib's default power-of-ten +
    2x/5x-multiple tick labels (10^-1, 2x10^-1, ...) with plain decimal
    numbers (0.1, 0.2, ...). Forces major ticks at 1/2/5 x each decade so a
    range spanning less than one decade (common here) still gets more than a
    single labelled gridline, instead of relying on matplotlib's default
    "auto" choice of what counts as major vs. minor. Call AFTER
    set_xscale/set_yscale("log"); safe to call on both axes of a loglog plot.
    """
    which = ax.xaxis if axis == "x" else ax.yaxis
    which.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    which.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    which.set_minor_formatter(mticker.NullFormatter())


def min_contracts_filter(buckets: list[dict], min_contracts: int = 5) -> list[dict]:
    """Drop bucket records backed by fewer than `min_contracts` distinct
    contracts -- a bucket with n_contracts=1 or 2 is one contract's history,
    not a maturity-effect estimate, and previously showed up as a spurious
    volatility spike at the far-dated end of several products' profiles."""
    return [b for b in buckets if b.get("n_contracts", 0) >= min_contracts]


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
