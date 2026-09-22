"""Section B figures: Time dimension (B1-B8) for notebook 024.

Plots rolling volatility, term structure, and correlation trends over 16 years.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib24
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import viz24 as vz


def fig_B1(data: dict):
    """CL: front (0-60d) vs deferred (365-730d) vol, with crisis shading."""
    # Load rolling parquet
    rolling_path = Path(__file__).resolve().parent / "phase_4_24_rolling_CL.parquet"
    df = pd.read_parquet(rolling_path)
    df["date"] = pd.to_datetime(df["date"])

    fig, ax = vz.new_fig(figsize=(9, 5))

    ax.plot(
        df["date"],
        df["vol_0_60"],
        label="Front (0-60d)",
        color=vz.BLUE,
        linewidth=1.5,
    )
    ax.plot(
        df["date"],
        df["vol_365_730"],
        label="Deferred (365-730d)",
        color=vz.RED,
        linewidth=1.5,
    )

    # Shade all crisis windows
    for window_name, (start, end) in lib24.CRISIS_WINDOWS.items():
        label = window_name.replace("_", " ").title()
        color = (
            vz.RED
            if "collapse" in window_name
            or "covid" in window_name
            or "ukraine" in window_name
            else vz.GRAY
        )
        vz.shade_crisis(ax, start, end, label=label, color=color)

    vz.style_ax(
        ax, title="CL: Volatility term structure over time", ylabel="Annualized vol"
    )
    vz.legend(ax, loc="upper left")

    return fig, ax


def fig_B2(data: dict):
    """CL: ratio of front/deferred vol, with y=1 reference line."""
    # Load rolling parquet
    rolling_path = Path(__file__).resolve().parent / "phase_4_24_rolling_CL.parquet"
    df = pd.read_parquet(rolling_path)
    df["date"] = pd.to_datetime(df["date"])

    fig, ax = vz.new_fig(figsize=(9, 5))

    ax.plot(
        df["date"],
        df["ratio_front_deferred"],
        label="Front/Deferred ratio",
        color=vz.BLUE,
        linewidth=1.5,
    )
    ax.axhline(1.0, color=vz.GRAY, linestyle="--", linewidth=1.5, label="y=1.0")

    # Shade crisis windows
    for window_name, (start, end) in lib24.CRISIS_WINDOWS.items():
        label = window_name.replace("_", " ").title()
        color = (
            vz.RED
            if "collapse" in window_name
            or "covid" in window_name
            or "ukraine" in window_name
            else vz.GRAY
        )
        vz.shade_crisis(ax, start, end, label=label, color=color)

    vz.style_ax(ax, title="CL: Front/Deferred volatility ratio", ylabel="Ratio")
    vz.legend(ax, loc="upper left")

    return fig, ax


def fig_B3(data: dict):
    """NG: two-panel (front vol, deferred vol, ratio) as 3 rows."""
    # Load rolling parquet
    rolling_path = Path(__file__).resolve().parent / "phase_4_24_rolling_NG.parquet"
    df = pd.read_parquet(rolling_path)
    df["date"] = pd.to_datetime(df["date"])

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    fig.set_facecolor(vz.SURFACE)

    # Row 0: Front and deferred vol
    axes[0].plot(
        df["date"],
        df["vol_0_60"],
        label="Front (0-60d)",
        color=vz.BLUE,
        linewidth=1.5,
    )
    axes[0].plot(
        df["date"],
        df["vol_365_730"],
        label="Deferred (365-730d)",
        color=vz.RED,
        linewidth=1.5,
    )
    vz.style_ax(
        axes[0], title="NG: Front and deferred volatility", ylabel="Annualized vol"
    )
    vz.legend(axes[0], loc="upper left")

    # Row 1: Ratio
    axes[1].plot(
        df["date"],
        df["ratio_front_deferred"],
        label="Front/Deferred ratio",
        color=vz.BLUE,
        linewidth=1.5,
    )
    axes[1].axhline(1.0, color=vz.GRAY, linestyle="--", linewidth=1.5, label="y=1.0")
    vz.style_ax(axes[1], title="NG: Front/Deferred volatility ratio", ylabel="Ratio")
    vz.legend(axes[1], loc="upper left")

    # Row 2: All maturity buckets
    axes[2].plot(
        df["date"], df["vol_0_60"], label="0-60d", color=vz.BLUE, linewidth=1.5
    )
    axes[2].plot(
        df["date"], df["vol_60_180"], label="60-180d", color=vz.AQUA, linewidth=1.5
    )
    axes[2].plot(
        df["date"], df["vol_180_365"], label="180-365d", color=vz.YELLOW, linewidth=1.5
    )
    axes[2].plot(
        df["date"], df["vol_365_730"], label="365-730d", color=vz.RED, linewidth=1.5
    )
    axes[2].plot(
        df["date"],
        df["vol_730_3650"],
        label="730-3650d",
        color=vz.MAGENTA,
        linewidth=1.5,
    )
    vz.style_ax(
        axes[2],
        title="NG: All maturity buckets",
        ylabel="Annualized vol",
        xlabel="Date",
    )
    vz.legend(axes[2], loc="upper left")

    # Shade crisis on all rows
    for window_name, (start, end) in lib24.CRISIS_WINDOWS.items():
        color = (
            vz.RED
            if "collapse" in window_name
            or "covid" in window_name
            or "ukraine" in window_name
            else vz.GRAY
        )
        for ax in axes:
            vz.shade_crisis(ax, start, end, label=None, color=color)

    return fig, axes


def fig_B4(data: dict):
    """6 products' front/deferred ratio: CL, NG, GC, ZC, ZW, ES."""
    products = ["CL", "NG", "GC", "ZC", "ZW", "ES"]
    product_colors = vz.build_product_color(products)

    fig, ax = vz.new_fig(figsize=(10, 5.5))

    for product in products:
        rolling_path = (
            Path(__file__).resolve().parent / f"phase_4_24_rolling_{product}.parquet"
        )
        if not rolling_path.exists():
            continue

        df = pd.read_parquet(rolling_path)
        df["date"] = pd.to_datetime(df["date"])

        ax.plot(
            df["date"],
            df["ratio_front_deferred"],
            label=product,
            color=product_colors[product],
            linewidth=1.5,
        )

    ax.axhline(1.0, color=vz.GRAY, linestyle="--", linewidth=1, alpha=0.5)

    vz.style_ax(ax, title="Front/Deferred ratio: six major products", ylabel="Ratio")
    vz.legend(ax, loc="upper left", ncol=2)

    return fig, ax


def fig_B5(data: dict):
    """CL term structure spaghetti: specific dates' annual vol_by_bucket curves."""
    timeseries = data["timeseries"]

    # Dates to plot: crisis and calm
    dates_to_plot = [
        ("2014-12-15", "oil_collapse_2014_15"),
        ("2020-03-15", "covid_2020"),
        ("2022-03-15", "ukraine_2022"),
        ("2017-06-15", "calm_2017"),
        ("2018-06-15", "calm_2017"),
    ]

    fig, ax = vz.new_fig(figsize=(9, 5.5))

    for date_str, crisis_name in dates_to_plot:
        date_obj = pd.Timestamp(date_str)
        year_str = str(date_obj.year)

        # Get annual vol_by_bucket for that year from CL
        annual_data = timeseries["correlation_and_annual"]["CL"]["annual_vol_by_bucket"]
        if year_str not in annual_data:
            continue

        bucket_list = annual_data[year_str]
        if not bucket_list:
            continue

        dte_mids = [b["dte_mid"] for b in bucket_list]
        vols = [b["vol"] for b in bucket_list]

        # Color: crisis years in red/orange, calm in gray
        is_calm = "calm" in crisis_name
        color = vz.GRAY if is_calm else vz.RED
        ax.loglog(
            dte_mids,
            vols,
            marker="o",
            label=f"{date_str} ({crisis_name})",
            color=color,
            linewidth=1.5,
            markersize=3,
        )

    vz.style_ax(
        ax,
        title="CL: Term structure by date (annual aggregates)",
        xlabel="DTE (log)",
        ylabel="Vol (log)",
    )
    vz.legend(ax, loc="best")

    return fig, ax


def fig_B6(data: dict):
    """Vol-of-vol by maturity bucket: sector-aggregated bars."""
    timeseries = data["timeseries"]

    # Compute sector-level vol-of-vol
    buckets = ["0_60", "60_180", "180_365", "365_730", "730_3650"]
    bucket_labels = ["0-60d", "60-180d", "180-365d", "365-730d", "730-3650d"]
    sectors = sorted(lib24.SECTOR.values())

    # Build a 5 x len(sectors) matrix
    vol_of_vol_matrix = np.zeros((len(buckets), len(sectors)))

    for b_idx, bucket in enumerate(buckets):
        for s_idx, sector in enumerate(sectors):
            values = []
            for product in lib24.PRODUCTS:
                if lib24.SECTOR[product] != sector:
                    continue
                if product not in timeseries["products"]:
                    continue
                vol_of_vol = timeseries["products"][product][
                    "vol_of_vol_by_bucket"
                ].get(bucket)
                if vol_of_vol is not None:
                    values.append(vol_of_vol)
            if values:
                vol_of_vol_matrix[b_idx, s_idx] = np.mean(values)
            else:
                vol_of_vol_matrix[b_idx, s_idx] = np.nan

    # Plot grouped bar chart
    fig, ax = vz.new_fig(figsize=(10, 5.5))

    x_pos = np.arange(len(buckets))
    bar_width = 0.2

    for s_idx, sector in enumerate(sectors):
        offset = (s_idx - len(sectors) / 2 + 0.5) * bar_width
        ax.bar(
            x_pos + offset,
            vol_of_vol_matrix[:, s_idx],
            bar_width,
            label=sector,
            color=vz.SECTOR_COLOR[sector],
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(bucket_labels)
    vz.style_ax(
        ax, title="Vol-of-vol by maturity bucket (sector means)", ylabel="Vol-of-vol"
    )
    vz.legend(ax)

    return fig, ax


def fig_B7(data: dict):
    """CL and NG rolling 126-day correlation."""
    fig, ax = vz.new_fig(figsize=(10, 5.5))

    # CL correlation
    corr_path_cl = Path(__file__).resolve().parent / "phase_4_24_corr_CL.parquet"
    if corr_path_cl.exists():
        df_cl = pd.read_parquet(corr_path_cl)
        df_cl["date"] = pd.to_datetime(df_cl["date"])
        ax.plot(
            df_cl["date"],
            df_cl["rolling_corr_126d"],
            label="CL",
            color=vz.ORANGE,
            linewidth=1.5,
        )

    # NG correlation
    corr_path_ng = Path(__file__).resolve().parent / "phase_4_24_corr_NG.parquet"
    if corr_path_ng.exists():
        df_ng = pd.read_parquet(corr_path_ng)
        df_ng["date"] = pd.to_datetime(df_ng["date"])
        ax.plot(
            df_ng["date"],
            df_ng["rolling_corr_126d"],
            label="NG",
            color=vz.BLUE,
            linewidth=1.5,
        )

    ax.axhline(0.0, color=vz.GRAY, linestyle="--", linewidth=1, alpha=0.5)

    vz.style_ax(
        ax, title="Rolling 126-day correlation: CL and NG", ylabel="Correlation"
    )
    vz.legend(ax, loc="upper left")

    return fig, ax


def fig_B8(data: dict):
    """CL annual small multiples: one panel per year 2010-2026."""
    timeseries = data["timeseries"]

    annual_data = timeseries["correlation_and_annual"]["CL"]["annual_vol_by_bucket"]
    years = sorted([int(y) for y in annual_data])

    # Create grid
    ncols = 5
    nrows = (len(years) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 10), sharex=False, sharey=True)
    fig.set_facecolor(vz.SURFACE)
    axes_flat = axes.flatten()

    for i, year in enumerate(years):
        ax = axes_flat[i]
        bucket_list = annual_data[str(year)]

        if bucket_list:
            dte_mids = [b["dte_mid"] for b in bucket_list]
            vols = [b["vol"] for b in bucket_list]
            ax.loglog(
                dte_mids, vols, marker="o", color=vz.BLUE, linewidth=1, markersize=2
            )

        ax.set_title(str(year), fontsize=9, fontweight="bold", color=vz.TEXT_PRIMARY)
        ax.tick_params(labelsize=7)
        ax.grid(True, which="both", color=vz.GRID, linewidth=0.5, alpha=0.5)
        ax.set_facecolor(vz.SURFACE)

    # Hide unused subplots
    for i in range(len(years), len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(
        "CL: Annual term structure by year",
        fontsize=12,
        fontweight="bold",
        color=vz.TEXT_PRIMARY,
        y=0.98,
    )
    fig.text(0.5, 0.01, "DTE (log)", ha="center", fontsize=9, color=vz.TEXT_SECONDARY)
    fig.text(
        0.01,
        0.5,
        "Vol (log)",
        va="center",
        rotation="vertical",
        fontsize=9,
        color=vz.TEXT_SECONDARY,
    )

    return fig, axes_flat
