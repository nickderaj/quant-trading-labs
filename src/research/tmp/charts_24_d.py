"""Section D: realised smile/skew exploration for notebook 024.

This module plots realised (not implied) moments: skewness, kurtosis, volatility
smile (by moneyness), and return distributions across the 16 futures markets.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib24
import matplotlib.pyplot as plt
import numpy as np
import viz24 as vz


def fig_D1(data: dict) -> tuple:
    """Realised skewness by maturity bucket (DTE), all 16 products coloured by sector.

    DTE_mid on x-axis, skew on y-axis. Sector-coloured lines. Zero line marked.
    Sector legend (proxy handles, not all 16 products).
    """
    smile_skew = data["smile_skew"]
    moments = smile_skew["moments"]

    fig, ax = vz.new_fig(figsize=(10, 5.5))

    # Sector-level grouping for a compact legend
    sector_handles: dict[str, object] = {}
    sector_lines: dict[str, list] = {}  # sector -> list of artists

    for product in lib24.PRODUCTS:
        if product not in moments:
            continue
        sector = lib24.SECTOR[product]
        color = vz.SECTOR_COLOR[sector]

        data_points = moments[product]
        if not data_points:
            continue

        dte_mid = np.array([d["dte_mid"] for d in data_points])
        skew = np.array([d["skew"] for d in data_points])

        (line,) = ax.plot(dte_mid, skew, color=color, linewidth=1.2, alpha=0.8)

        if sector not in sector_lines:
            sector_lines[sector] = []
        sector_lines[sector].append(line)

    # Zero line
    ax.axhline(0, color=vz.GRAY, linestyle="--", linewidth=0.8, zorder=0)

    # Compact legend: one entry per sector using a proxy handle
    sector_order = ["energy", "metals", "ags", "control"]
    for sector in sector_order:
        if sector_lines.get(sector):
            # Use the first line of each sector as the legend proxy
            sector_handles[sector] = sector_lines[sector][0]

    ax.legend(
        [sector_handles[s] for s in sector_order if s in sector_handles],
        [s.capitalize() for s in sector_order if s in sector_handles],
        loc="upper left",
    )

    vz.style_ax(
        ax,
        title="Realised Skewness by Maturity",
        xlabel="Days to Expiry",
        ylabel="Skewness",
    )
    fig.tight_layout()
    return fig, ax


def fig_D2(data: dict) -> tuple:
    """Realised excess kurtosis by maturity bucket (DTE), all 16 products.

    Log-y axis (kurtosis can be very large and positive).
    Same sector colouring as D1.
    """
    smile_skew = data["smile_skew"]
    moments = smile_skew["moments"]

    fig, ax = vz.new_fig(figsize=(10, 5.5))

    sector_lines: dict[str, list] = {}  # sector -> list of artists

    for product in lib24.PRODUCTS:
        if product not in moments:
            continue
        sector = lib24.SECTOR[product]
        color = vz.SECTOR_COLOR[sector]

        data_points = moments[product]
        if not data_points:
            continue

        dte_mid = np.array([d["dte_mid"] for d in data_points])
        kurtosis = np.array([d["excess_kurtosis"] for d in data_points])

        # Floor negative or zero kurtosis at a small epsilon for log-y display
        kurtosis = np.where(kurtosis > 0.01, kurtosis, 0.01)

        (line,) = ax.plot(dte_mid, kurtosis, color=color, linewidth=1.2, alpha=0.8)

        if sector not in sector_lines:
            sector_lines[sector] = []
        sector_lines[sector].append(line)

    ax.set_yscale("log")

    # Compact legend: one entry per sector
    sector_order = ["energy", "metals", "ags", "control"]
    sector_handles = {}
    for sector in sector_order:
        if sector_lines.get(sector):
            sector_handles[sector] = sector_lines[sector][0]

    ax.legend(
        [sector_handles[s] for s in sector_order if s in sector_handles],
        [s.capitalize() for s in sector_order if s in sector_handles],
        loc="upper left",
    )

    vz.style_ax(
        ax,
        title="Realised Excess Kurtosis by Maturity",
        xlabel="Days to Expiry",
        ylabel="Excess Kurtosis (log scale)",
    )
    fig.tight_layout()
    return fig, ax


def fig_D3(data: dict) -> tuple:
    """Realised smile: CL and NG side-by-side.

    Each panel: pooled smile (BLUE, thick) + maturity bins (ORANGE, AQUA, VIOLET, thin).
    Title must contain "realised" to clarify this is not implied vol.
    """
    smile_skew = data["smile_skew"]
    smile = smile_skew["smile"]

    fig, (ax_cl, ax_ng) = plt.subplots(1, 2, figsize=(12, 5), dpi=110)
    fig.set_facecolor(vz.SURFACE)

    products = ["CL", "NG"]
    axes = [ax_cl, ax_ng]

    for ax, product in zip(axes, products):
        if product not in smile:
            continue

        product_smile = smile[product]

        # Pooled smile
        pooled = product_smile["pooled"]
        if pooled:
            moneyness = np.array([d["moneyness_mid"] for d in pooled])
            vol = np.array([d["vol"] for d in pooled])
            ax.plot(
                moneyness,
                vol,
                color=vz.BLUE,
                linewidth=2.0,
                label="Pooled",
                zorder=2,
            )

        # By maturity bins
        by_maturity = product_smile.get("by_maturity_bin", {})
        bin_colors = {"0_90": vz.ORANGE, "90_365": vz.AQUA, "365_plus": vz.VIOLET}
        bin_labels = {"0_90": "0–90d", "90_365": "90–365d", "365_plus": "365+d"}

        for bin_key in ["0_90", "90_365", "365_plus"]:
            if bin_key not in by_maturity:
                continue
            bin_data = by_maturity[bin_key]
            if not bin_data:
                continue
            moneyness = np.array([d["moneyness_mid"] for d in bin_data])
            vol = np.array([d["vol"] for d in bin_data])
            ax.plot(
                moneyness,
                vol,
                color=bin_colors[bin_key],
                linewidth=1.0,
                label=bin_labels[bin_key],
                zorder=1,
            )

        vz.style_ax(
            ax,
            title=f"{product} — Realised Vol vs. Log-Moneyness",
            xlabel="Log-Moneyness",
            ylabel="Realised Volatility",
        )
        vz.legend(ax, loc="best")

    fig.tight_layout()
    return fig, (ax_cl, ax_ng)


def fig_D4(data: dict) -> tuple:
    """Vol-vs-DTE profiles split by curve state (contango vs backwardation).

    CL and NG side-by-side, log-x axis.
    """
    smile_skew = data["smile_skew"]
    skew = smile_skew["skew"]

    fig, (ax_cl, ax_ng) = plt.subplots(1, 2, figsize=(12, 5), dpi=110)
    fig.set_facecolor(vz.SURFACE)

    products = ["CL", "NG"]
    axes = [ax_cl, ax_ng]

    for ax, product in zip(axes, products):
        if product not in skew:
            continue

        product_skew = skew[product]

        # Contango
        if "contango" in product_skew:
            contango_data = product_skew["contango"]
            if contango_data:
                dte = np.array([d["dte_mid"] for d in contango_data])
                vol = np.array([d["vol"] for d in contango_data])
                ax.plot(
                    dte,
                    vol,
                    color=vz.BLUE,
                    linewidth=1.5,
                    label="Contango",
                    zorder=2,
                )

        # Backwardation
        if "backwardation" in product_skew:
            backwardation_data = product_skew["backwardation"]
            if backwardation_data:
                dte = np.array([d["dte_mid"] for d in backwardation_data])
                vol = np.array([d["vol"] for d in backwardation_data])
                ax.plot(
                    dte,
                    vol,
                    color=vz.RED,
                    linewidth=1.5,
                    label="Backwardation",
                    zorder=1,
                )

        ax.set_xscale("log")
        vz.style_ax(
            ax,
            title=f"{product} — Vol vs. DTE by Curve State",
            xlabel="Days to Expiry (log scale)",
            ylabel="Realised Volatility",
        )
        vz.legend(ax, loc="best")

    fig.tight_layout()
    return fig, (ax_cl, ax_ng)


def fig_D5(data: dict) -> tuple:
    """Return density histograms (front vs deferred) for CL and NG.

    Side-by-side panels, log-y axis, density=True, alpha=0.55.
    """
    smile_skew = data["smile_skew"]
    densities = smile_skew["densities"]

    fig, (ax_cl, ax_ng) = plt.subplots(1, 2, figsize=(12, 5), dpi=110)
    fig.set_facecolor(vz.SURFACE)

    products = ["CL", "NG"]
    axes = [ax_cl, ax_ng]

    for ax, product in zip(axes, products):
        if product not in densities:
            continue

        product_density = densities[product]

        # Front (0–60d)
        if "front" in product_density:
            front = np.asarray(product_density["front"], dtype=float)
            front = front[np.isfinite(front)]
            if len(front) > 0:
                ax.hist(
                    front,
                    bins=60,
                    density=True,
                    alpha=0.55,
                    color=vz.BLUE,
                    label="Front (0–60d)",
                    zorder=2,
                )

        # Deferred (365–730d)
        if "deferred" in product_density:
            deferred = np.asarray(product_density["deferred"], dtype=float)
            deferred = deferred[np.isfinite(deferred)]
            if len(deferred) > 0:
                ax.hist(
                    deferred,
                    bins=60,
                    density=True,
                    alpha=0.55,
                    color=vz.RED,
                    label="Deferred (365–730d)",
                    zorder=1,
                )

        ax.set_yscale("log")
        vz.style_ax(
            ax,
            title=f"{product} — Return Density Distribution",
            xlabel="Log-Return",
            ylabel="Density (log scale)",
        )
        vz.legend(ax, loc="best")

    fig.tight_layout()
    return fig, (ax_cl, ax_ng)
