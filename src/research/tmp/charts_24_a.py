"""Chart module for notebook 024: Samuelson effect across 16 futures markets.

Seven figures (A1-A7) visualizing volatility term structure and maturity effects.
Data sources: phase_2_24_profiles.json (vol buckets) and phase_3_24_slopes.json (slopes).

Every bucket list pulled straight from phase_2_24_profiles.json is passed through
vz.min_contracts_filter(..., min_contracts=5) before plotting: a bucket backed by
fewer than 5 distinct contracts is one or two contracts' history, not a maturity
estimate, and several products' longest bucket is exactly that (see NEXT_PROMPT.md
follow-up: the "spike" originally visible at the far-dated end of several profiles).
Every log-scaled days-to-expiry axis is shown in years (vz.style_years_axis), not
raw days -- matplotlib's default power-of-ten day labels were reported illegible.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib24
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import viz24 as vz

MIN_CONTRACTS = 5


def _clean_buckets(product_buckets: list[dict]) -> list[dict]:
    return vz.min_contracts_filter(product_buckets, min_contracts=MIN_CONTRACTS)


def fig_A1(data: dict) -> tuple:
    """CL hero chart: years-to-expiry vs vol with power-law fit and n_obs-weighted band.

    data = {"profiles": {...}, "slopes": {...}}
    """
    fig, ax = vz.new_fig(figsize=(8, 5))

    buckets = _clean_buckets(data["profiles"]["products"]["CL"]["mad"]["gap1_vol0"])

    years = vz.days_to_years([b["dte_mid"] for b in buckets])
    vol = np.array([b["vol"] for b in buckets])
    n_obs = np.array([b["n_obs"] for b in buckets])

    sizes = np.clip(n_obs / n_obs.max() * 100, 20, 150)
    ax.scatter(
        years, vol, s=sizes, alpha=0.6, color=vz.BLUE, zorder=3, label="Observations"
    )

    log_x = np.log(years)
    log_y = np.log(vol)
    coef = np.polyfit(log_x, log_y, 1)
    slope, intercept = coef[0], coef[1]

    x_smooth = np.logspace(np.log10(years.min()), np.log10(years.max()), 200)
    y_smooth = np.exp(intercept) * (x_smooth**slope)
    ax.plot(
        x_smooth,
        y_smooth,
        color=vz.BLUE,
        linewidth=2,
        label=f"Power-law fit (slope={slope:.3f})",
        zorder=2,
    )

    rel_err = 1 / np.sqrt(n_obs)
    vol_lo = vol * (1 - rel_err)
    vol_hi = vol * (1 + rel_err)
    ax.fill_between(
        years,
        vol_lo,
        vol_hi,
        alpha=0.15,
        color=vz.BLUE,
        label="±1/√n_obs (illustrative)",
    )

    ax.set_xscale("log")
    vz.style_years_axis(ax)
    vz.style_ax(
        ax,
        title="CL: Volatility term structure",
        xlabel="Years to expiry",
        ylabel="Annualised vol",
    )
    vz.legend(ax, loc="upper right")
    return fig, ax


def fig_A2(data: dict) -> tuple:
    """CL: three estimators (mad, winsor, std) overlaid on log-log axes.

    data = {"profiles": {...}, "slopes": {...}}
    """
    fig, ax = vz.new_fig(figsize=(8, 5))

    product_data = data["profiles"]["products"]["CL"]

    for estimator, color, label in [
        ("mad", vz.BLUE, "MAD (robust)"),
        ("winsor", vz.ORANGE, "Winsorized (robust)"),
        ("std", vz.RED, "Std dev (unstable)"),
    ]:
        buckets = _clean_buckets(product_data[estimator]["gap1_vol0"])
        years = vz.days_to_years([b["dte_mid"] for b in buckets])
        vol = np.array([b["vol"] for b in buckets])
        ax.plot(
            years, vol, marker="o", linewidth=1.5, color=color, label=label, zorder=2
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    vz.style_years_axis(ax)
    vz.style_log_axis_plain(ax, axis="y")
    vz.style_ax(
        ax,
        title="CL: Estimator comparison (gap1, vol0)",
        xlabel="Years to expiry (log)",
        ylabel="Annualised vol (log)",
    )
    vz.legend(ax, loc="upper right")
    return fig, ax


def fig_A3(data: dict) -> tuple:
    """4x4 small multiples: all 16 products, sector-colored spines.

    data = {"profiles": {...}, "slopes": {...}}
    """
    fig, axes = plt.subplots(4, 4, figsize=(14, 12), dpi=110)
    fig.set_facecolor(vz.SURFACE)
    axes = axes.flatten()

    for idx, product in enumerate(lib24.PRODUCTS):
        ax = axes[idx]

        ax.set_facecolor(vz.SURFACE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "right", "top", "bottom"):
            color = vz.SECTOR_COLOR[lib24.SECTOR[product]]
            ax.spines[spine].set_color(color)
            ax.spines[spine].set_linewidth(1.5)
        ax.tick_params(colors=vz.TEXT_SECONDARY, labelsize=8)
        ax.grid(axis="y", color=vz.GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

        buckets = _clean_buckets(
            data["profiles"]["products"][product]["mad"]["gap1_vol0"]
        )
        if buckets:
            years = vz.days_to_years([b["dte_mid"] for b in buckets])
            vol = np.array([b["vol"] for b in buckets])
            ax.plot(
                years,
                vol,
                marker="o",
                linewidth=1.5,
                color=vz.SECTOR_COLOR[lib24.SECTOR[product]],
                zorder=2,
            )
            ax.set_xscale("log")
            vz.style_years_axis(ax)

        title = f"{product} — {lib24.FULL_NAME[product]}"
        ax.set_title(
            title,
            color=vz.TEXT_PRIMARY,
            fontsize=9,
            fontweight="bold",
            loc="left",
            pad=6,
        )
        ax.set_xlabel("", fontsize=8)
        ax.set_ylabel("", fontsize=8)

    fig.text(
        0.5, 0.02, "Years to expiry", ha="center", fontsize=10, color=vz.TEXT_SECONDARY
    )
    fig.text(
        0.02,
        0.5,
        "Annualised vol",
        va="center",
        rotation="vertical",
        fontsize=10,
        color=vz.TEXT_SECONDARY,
    )

    plt.tight_layout(rect=(0.04, 0.04, 1, 1))
    return fig, axes


def fig_A4(data: dict) -> tuple:
    """All 16 products overlaid, each normalized to its 365-day bucket = 1.0, colored by sector.

    data = {"profiles": {...}, "slopes": {...}}
    """
    fig, ax = vz.new_fig(figsize=(10, 5))

    product_color = vz.build_product_color(lib24.PRODUCTS)

    for product in lib24.PRODUCTS:
        buckets = _clean_buckets(
            data["profiles"]["products"][product]["mad"]["gap1_vol0"]
        )
        if not buckets:
            continue

        years = vz.days_to_years([b["dte_mid"] for b in buckets])
        vol = np.array([b["vol"] for b in buckets])

        ref_vol = None
        for b in buckets:
            if b["dte_lo"] == 365.0 or (b["dte_lo"] < 365.0 <= b["dte_hi"]):
                ref_vol = b["vol"]
                break
        if ref_vol is None or ref_vol == 0:
            ref_vol = vol[len(vol) // 2]

        vol_normalized = vol / ref_vol
        color = product_color[product]

        ax.plot(
            years,
            vol_normalized,
            marker="o",
            linewidth=1.5,
            color=color,
            label=product,
            zorder=2,
        )

    ax.axhline(y=1.0, color=vz.GRID, linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    vz.style_years_axis(ax)

    vz.style_ax(
        ax,
        title="Volatility term structure: all 16 products (normalized)",
        xlabel="Years to expiry",
        ylabel="Vol / 365-day bucket",
    )

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color=vz.SECTOR_COLOR[s], lw=2, label=s)
        for s in ["energy", "metals", "ags", "control"]
    ]
    vz.legend(ax, handles=legend_elements, loc="upper right")

    return fig, ax


def fig_A5(data: dict) -> tuple:
    """3 side-by-side sector panels (energy, metals, ags) with ES dashed gray reference.

    data = {"profiles": {...}, "slopes": {...}}
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=110)
    fig.set_facecolor(vz.SURFACE)

    product_color = vz.build_product_color(lib24.PRODUCTS)

    es_buckets = _clean_buckets(data["profiles"]["products"]["ES"]["mad"]["gap1_vol0"])
    es_years = vz.days_to_years([b["dte_mid"] for b in es_buckets])
    es_vol = np.array([b["vol"] for b in es_buckets])

    for ax_idx, sector in enumerate(["energy", "metals", "ags"]):
        ax = axes[ax_idx]

        ax.set_facecolor(vz.SURFACE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(vz.GRID)
        ax.tick_params(colors=vz.TEXT_SECONDARY, labelsize=9)
        ax.grid(axis="y", color=vz.GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        ax.plot(
            es_years,
            es_vol,
            linestyle="--",
            linewidth=1.5,
            color=vz.GRAY,
            label="ES (control)",
            zorder=1,
        )

        sector_products = [p for p in lib24.PRODUCTS if lib24.SECTOR[p] == sector]
        for product in sector_products:
            buckets = _clean_buckets(
                data["profiles"]["products"][product]["mad"]["gap1_vol0"]
            )
            if buckets:
                years = vz.days_to_years([b["dte_mid"] for b in buckets])
                vol = np.array([b["vol"] for b in buckets])
                color = product_color[product]
                ax.plot(
                    years,
                    vol,
                    marker="o",
                    linewidth=1.5,
                    color=color,
                    label=product,
                    zorder=2,
                )

        ax.set_xscale("log")
        vz.style_years_axis(ax)
        vz.style_ax(
            ax,
            title=sector.capitalize(),
            xlabel="Years to expiry",
            ylabel="Annualised vol",
        )
        vz.legend(ax, loc="best")

    return fig, axes


def fig_A6(data: dict) -> tuple:
    """Ranked horizontal bar chart from ranking_primary, colored by sector, with error whiskers.

    data = {"profiles": {...}, "slopes": {...}}
    """
    fig, ax = vz.new_fig(figsize=(8, 6))

    ranking = data["slopes"]["ranking_primary"]

    products = [r["product"] for r in ranking]
    slopes = np.array([r["slope"] for r in ranking])
    ci_lo = np.array([r["ci_lo"] for r in ranking])
    ci_hi = np.array([r["ci_hi"] for r in ranking])

    colors = [vz.SECTOR_COLOR[lib24.SECTOR[p]] for p in products]

    y_pos = np.arange(len(products))

    ax.barh(y_pos, slopes, color=colors, alpha=0.7, zorder=2)
    errors = [slopes - ci_lo, ci_hi - slopes]
    ax.errorbar(
        slopes,
        y_pos,
        xerr=errors,
        fmt="none",
        ecolor="black",
        elinewidth=1,
        capsize=3,
        zorder=3,
    )

    ax.axvline(x=0, color=vz.GRID, linewidth=0.8, linestyle="-", zorder=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(products, fontsize=9)
    ax.invert_yaxis()

    vz.style_ax(
        ax,
        title="Samuelson effect: ranked slopes",
        xlabel="Slope (log vol vs log days-to-expiry)",
        ylabel="Product",
    )

    return fig, ax


def fig_A7(data: dict) -> tuple:
    """Heatmap: products (rows) x maturity buckets (columns), normalized to 365-day bucket.

    Cells backed by fewer than MIN_CONTRACTS distinct contracts are masked
    (shown blank) rather than plotted as if they were reliable estimates.

    data = {"profiles": {...}, "slopes": {...}}
    """
    dte_mids_all = set()
    for product in lib24.PRODUCTS:
        buckets = data["profiles"]["products"][product]["mad"]["gap1_vol0"]
        for b in buckets:
            dte_mids_all.add(b["dte_mid"])

    dte_mids_sorted = sorted(dte_mids_all)
    dte_to_col = {dte: i for i, dte in enumerate(dte_mids_sorted)}

    n_products = len(lib24.PRODUCTS)
    n_dte = len(dte_mids_sorted)
    matrix = np.full((n_products, n_dte), np.nan)

    for p_idx, product in enumerate(lib24.PRODUCTS):
        buckets = data["profiles"]["products"][product]["mad"]["gap1_vol0"]

        clean = _clean_buckets(buckets)
        ref_vol = None
        for b in clean:
            if b["dte_lo"] == 365.0 or (b["dte_lo"] < 365.0 <= b["dte_hi"]):
                ref_vol = b["vol"]
                break
        if ref_vol is None or ref_vol == 0:
            vols = [b["vol"] for b in clean if b["vol"] > 0]
            ref_vol = np.median(vols) if vols else 1.0

        for b in clean:
            col = dte_to_col[b["dte_mid"]]
            matrix[p_idx, col] = b["vol"] / ref_vol

    fig, ax = vz.new_fig(figsize=(12, 6))

    norm = matplotlib.colors.TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=2.0)
    cmap = plt.cm.RdBu_r
    cmap.set_bad(vz.GRID)

    im = ax.imshow(
        matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest", zorder=2
    )

    ax.set_xticks(np.arange(n_dte))
    ax.set_yticks(np.arange(n_products))
    year_labels = [f"{d / vz.DAYS_PER_YEAR:.1f}y" for d in dte_mids_sorted]
    ax.set_xticklabels(year_labels, fontsize=8, rotation=45)
    ax.set_yticklabels(lib24.PRODUCTS, fontsize=9)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(vz.GRID)

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Vol / 365-day bucket", color=vz.TEXT_SECONDARY, fontsize=9)
    cbar.ax.tick_params(labelsize=8, colors=vz.TEXT_SECONDARY)

    ax.set_xlabel("Years to expiry", color=vz.TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel("Product", color=vz.TEXT_SECONDARY, fontsize=10)
    ax.set_title(
        "Volatility heatmap: term structure across all products "
        "(blank = fewer than 5 contracts backing that cell)",
        color=vz.TEXT_PRIMARY,
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=10,
    )

    return fig, ax
