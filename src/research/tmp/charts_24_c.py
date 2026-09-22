"""Section C figures: Seasonality (C1-C4) for notebook 024.

Plots seasonal patterns across energy (NG), grains, and control (GC).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib24
import matplotlib.pyplot as plt
import numpy as np
import viz24 as vz


def fig_C1(data: dict):
    """NG winter vs summer vol-vs-dte curve."""
    seasonality = data["seasonality"]

    fig, ax = vz.new_fig(figsize=(9, 5.5))

    # Winter
    winter_data = seasonality["ng_seasonality"]["winter"]
    if winter_data:
        winter_dte = [b["dte_mid"] for b in winter_data]
        winter_vol = [b["vol"] for b in winter_data]
        ax.plot(
            winter_dte,
            winter_vol,
            marker="o",
            label="Winter",
            color=vz.ORANGE,
            linewidth=1.5,
            markersize=4,
        )

    # Summer
    summer_data = seasonality["ng_seasonality"]["summer"]
    if summer_data:
        summer_dte = [b["dte_mid"] for b in summer_data]
        summer_vol = [b["vol"] for b in summer_data]
        ax.plot(
            summer_dte,
            summer_vol,
            marker="o",
            label="Summer",
            color=vz.BLUE,
            linewidth=1.5,
            markersize=4,
        )

    ax.set_xscale("log")
    vz.style_ax(
        ax, title="NG: Winter vs Summer volatility", xlabel="DTE (log)", ylabel="Vol"
    )
    vz.legend(ax, loc="best")

    return fig, ax


def fig_C2(data: dict):
    """2x2 grain panels: vol by calendar month."""
    seasonality = data["seasonality"]
    grains = ["ZC", "ZS", "ZW", "KE"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.set_facecolor(vz.SURFACE)
    axes_flat = axes.flatten()

    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    for idx, grain in enumerate(grains):
        ax = axes_flat[idx]
        grain_data = seasonality["grains_seasonality"].get(grain, {})

        months = []
        vols = []
        for month_num in range(1, 13):
            vol = grain_data.get(str(month_num))
            if vol is not None:
                months.append(month_num - 1)
                vols.append(vol)

        if months:
            ax.bar(months, vols, color=vz.SECTOR_COLOR.get("ags", vz.AQUA), width=0.7)
            ax.set_xticks(range(12))
            ax.set_xticklabels(month_names, rotation=45, fontsize=8)

        vz.style_ax(
            ax,
            title=f"{grain}: {lib24.FULL_NAME.get(grain, grain)}",
            ylabel="Vol",
        )

    return fig, axes_flat


def fig_C3(data: dict):
    """Heatmap: month x maturity for NG and CL side by side."""
    seasonality = data["seasonality"]
    heatmap_data = seasonality["heatmap_matrix"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.set_facecolor(vz.SURFACE)

    month_labels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    maturity_labels = ["Front (0-90d)", "Mid (90-365d)", "Back (365+d)"]

    products_to_plot = [("NG", vz.BLUE), ("CL", vz.ORANGE)]

    for (product, color_hint), ax in zip(products_to_plot, axes):
        if product not in heatmap_data:
            ax.text(0.5, 0.5, f"No data for {product}", ha="center", va="center")
            continue

        product_dict = heatmap_data[product]

        # Build 12x3 matrix
        matrix = np.zeros((12, 3))
        matrix[:] = np.nan

        for key, val in product_dict.items():
            parts = key.split("_")
            if len(parts) >= 2:
                month_str = parts[0]
                try:
                    month_num = int(month_str) - 1
                except (ValueError, IndexError):
                    continue

                # Determine maturity bucket index
                if "front" in key:
                    bucket_idx = 0
                elif "mid" in key:
                    bucket_idx = 1
                elif "back" in key:
                    bucket_idx = 2
                else:
                    continue

                if 0 <= month_num < 12 and 0 <= bucket_idx < 3:
                    matrix[month_num, bucket_idx] = val

        # Plot heatmap
        im = ax.imshow(matrix, aspect="auto", cmap="Blues", origin="upper")

        # Labels
        ax.set_xticks(range(3))
        ax.set_xticklabels(maturity_labels, fontsize=9)
        ax.set_yticks(range(12))
        ax.set_yticklabels(month_labels, fontsize=9)
        ax.set_title(
            f"{product}: Seasonality heatmap (month x maturity)",
            fontsize=11,
            fontweight="bold",
            color=vz.TEXT_PRIMARY,
        )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Vol", fontsize=9, color=vz.TEXT_SECONDARY)

        ax.set_facecolor(vz.SURFACE)

    return fig, axes


def fig_C4(data: dict):
    """GC (control) winter vs summer vol-vs-dte: should show no seasonality."""
    seasonality = data["seasonality"]

    fig, ax = vz.new_fig(figsize=(9, 5.5))

    # Winter
    winter_data = seasonality["gc_control"]["winter"]
    if winter_data:
        winter_dte = [b["dte_mid"] for b in winter_data]
        winter_vol = [b["vol"] for b in winter_data]
        ax.plot(
            winter_dte,
            winter_vol,
            marker="o",
            label="Winter",
            color=vz.GRAY,
            linewidth=1.5,
            markersize=4,
        )

    # Summer
    summer_data = seasonality["gc_control"]["summer"]
    if summer_data:
        summer_dte = [b["dte_mid"] for b in summer_data]
        summer_vol = [b["vol"] for b in summer_data]
        ax.plot(
            summer_dte,
            summer_vol,
            marker="o",
            label="Summer",
            color=vz.YELLOW,
            linewidth=1.5,
            markersize=4,
        )

    ax.set_xscale("log")
    vz.style_ax(
        ax,
        title="GC (Gold): Winter vs Summer volatility - control",
        xlabel="DTE (log)",
        ylabel="Vol",
    )
    vz.legend(ax, loc="best")

    return fig, ax
