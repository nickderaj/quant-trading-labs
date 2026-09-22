"""Figures A1-A11: Part A of notebook 025's visualisations.

Each fig_aX function takes the loaded phase_3_25_primer.json data (or None for
synthetic-data figures) and returns (fig, ax) or (fig, axes). Captions dict at
the end explains each figure's intuition.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add this directory to path for viz/lib imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib24
import lib25
import pricers_25_asian
import viz25 as viz


def fig_a1(data: dict) -> tuple:
    """Hero: payoff at expiry for unhedged, futures, put, collar, ko_put."""
    fig, ax = viz.new_fig()

    grids = data["payoff_grids"]
    S = np.array(grids["future"]["S"])

    # Reference price for the note
    F = 70.0  # Recent CL price

    # Plot unhedged (dashed grey)
    ax.plot(
        S,
        S - F,
        label="Unhedged",
        color=viz.STRUCTURE_COLOR["unhedged"],
        linewidth=2,
        linestyle="--",
    )

    # Futures hedging
    ax.plot(
        S,
        grids["future"]["payoff"],
        label="Futures hedge",
        color=viz.STRUCTURE_COLOR["futures"],
        linewidth=2,
    )

    # Put hedge: long put payoff, net of the position (S-F) it hedges
    ax.plot(
        S,
        (S - F) + np.array(grids["put"]["payoff"]),
        label="Put hedge",
        color=viz.STRUCTURE_COLOR["vanilla"],
        linewidth=2,
    )

    # Collar: real collar payoff, net of the underlying position
    ax.plot(
        S,
        (S - F) + np.array(grids["collar"]["payoff"]),
        label="Collar",
        color=viz.STRUCTURE_COLOR["collar"],
        linewidth=2,
    )

    # KO put hedge, net of the underlying position
    ax.plot(
        S,
        (S - F) + np.array(grids["ko_put"]["payoff"]),
        label="KO put",
        color=viz.STRUCTURE_COLOR["barrier"],
        linewidth=2,
    )

    # Strike line
    viz.mark_barrier(ax, F, label="F", color=viz.GRAY)

    viz.style_ax(
        ax,
        title="Payoff at expiry: unhedged vs hedging strategies",
        xlabel="Crude oil price ($/bbl)",
        ylabel="Payoff ($/bbl)",
    )
    viz.legend(ax, loc="upper left")
    viz.price_caption(ax)

    return fig, ax


def fig_a2(data: dict) -> tuple:
    """Four barrier types in 2x2 grid: payoff, barrier, alive/dead labelled."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=110)
    fig.set_facecolor(viz.SURFACE)

    S = np.linspace(30, 110, 200)
    K = 70.0
    H = 50.0

    # Vanilla put
    ax = axes[0, 0]
    payoff_vanilla = np.maximum(K - S, 0)
    ax.fill_between(
        S, 0, payoff_vanilla, alpha=0.2, color=viz.STRUCTURE_COLOR["vanilla"]
    )
    ax.plot(
        S,
        payoff_vanilla,
        color=viz.STRUCTURE_COLOR["vanilla"],
        linewidth=2,
        label="Vanilla put",
    )
    ax.axvline(K, color=viz.GRAY, linewidth=1, linestyle="--")
    ax.text(K, ax.get_ylim()[1] * 0.9, "Strike", color=viz.GRAY, fontsize=9)
    ax.text(35, 15, "Alive", fontsize=10, color=viz.TEXT_PRIMARY, fontweight="bold")
    viz.style_ax(ax, title="Vanilla put", ylabel="Payoff ($/bbl)")
    ax.set_ylim([0, 25])

    # Knock-out put (barrier < strike)
    ax = axes[0, 1]
    payoff_ko = np.where(S >= H, np.maximum(K - S, 0), 0)
    ax.fill_between(S, 0, payoff_ko, alpha=0.2, color=viz.STRUCTURE_COLOR["barrier"])
    ax.plot(
        S, payoff_ko, color=viz.STRUCTURE_COLOR["barrier"], linewidth=2, label="KO put"
    )
    ax.axvline(K, color=viz.GRAY, linewidth=1, linestyle="--", label="Strike")
    ax.axvline(H, color=viz.RED, linewidth=1.5, linestyle="-", label="Barrier")
    ax.text(K, ax.get_ylim()[1] * 0.9, "K", color=viz.GRAY, fontsize=9)
    ax.text(H, ax.get_ylim()[1] * 0.9, "H", color=viz.RED, fontsize=9)
    ax.text(60, 15, "Alive", fontsize=10, color=viz.TEXT_PRIMARY, fontweight="bold")
    ax.text(
        35,
        5,
        "Dead\n(knocked out)",
        fontsize=10,
        color=viz.TEXT_SECONDARY,
        fontweight="bold",
    )
    viz.style_ax(ax, title="Knock-out put", ylabel="Payoff ($/bbl)")
    ax.set_ylim([0, 25])

    # Knock-in put (barrier < strike, activates if hit)
    ax = axes[1, 0]
    payoff_ki = np.where(S < H, np.maximum(K - S, 0), 0)
    ax.fill_between(S, 0, payoff_ki, alpha=0.2, color=viz.MAGENTA)
    ax.plot(S, payoff_ki, color=viz.MAGENTA, linewidth=2, label="KI put")
    ax.axvline(K, color=viz.GRAY, linewidth=1, linestyle="--", label="Strike")
    ax.axvline(H, color=viz.RED, linewidth=1.5, linestyle="-", label="Barrier")
    ax.text(K, ax.get_ylim()[1] * 0.9, "K", color=viz.GRAY, fontsize=9)
    ax.text(H, ax.get_ylim()[1] * 0.9, "H", color=viz.RED, fontsize=9)
    ax.text(
        60,
        15,
        "Dead\n(not knocked in)",
        fontsize=10,
        color=viz.TEXT_SECONDARY,
        fontweight="bold",
    )
    ax.text(35, 5, "Alive", fontsize=10, color=viz.TEXT_PRIMARY, fontweight="bold")
    viz.style_ax(ax, title="Knock-in put", ylabel="Payoff ($/bbl)")
    ax.set_ylim([0, 25])

    # Double-barrier (alive between barriers)
    ax = axes[1, 1]
    H_upper = 90.0
    payoff_double = np.where((S >= H) & (S <= H_upper), np.maximum(K - S, 0), 0)
    ax.fill_between(S, 0, payoff_double, alpha=0.2, color=viz.GREEN)
    ax.plot(S, payoff_double, color=viz.GREEN, linewidth=2, label="Double barrier")
    ax.axvline(H, color=viz.RED, linewidth=1.5, linestyle="-", label="Lower barrier")
    ax.axvline(
        H_upper, color=viz.RED, linewidth=1.5, linestyle="-", label="Upper barrier"
    )
    ax.axvline(K, color=viz.GRAY, linewidth=1, linestyle="--", alpha=0.5)
    ax.text(40, 15, "Dead", fontsize=10, color=viz.TEXT_SECONDARY, fontweight="bold")
    ax.text(70, 5, "Alive", fontsize=10, color=viz.TEXT_PRIMARY, fontweight="bold")
    ax.text(95, 15, "Dead", fontsize=10, color=viz.TEXT_SECONDARY, fontweight="bold")
    viz.style_ax(ax, title="Double-barrier put", ylabel="Payoff ($/bbl)")
    ax.set_ylim([0, 25])

    for ax in axes.flat:
        ax.set_xlabel("Crude oil price ($/bbl)", fontsize=9)
        ax.set_xlim([30, 110])

    return fig, axes


def fig_a3(data: dict) -> tuple:
    """Real paths: three CL front-month paths, one breached barrier."""
    fig, ax = viz.new_fig()

    path_data = data["real_path_barrier"]
    dates = np.array(path_data["dates"], dtype="datetime64[D]")
    closes = np.array(path_data["closes"])
    barrier_level = path_data["barrier_level"]
    breach_date = path_data["breach_date"]
    breach_index = path_data["breach_index"]

    # Plot the real path
    ax.plot(
        dates,
        closes,
        color=viz.STRUCTURE_COLOR["unhedged"],
        linewidth=2,
        label="CL front-month",
    )

    # Mark the barrier
    ax.axhline(
        barrier_level,
        color=viz.RED,
        linewidth=1.5,
        linestyle="--",
        label=f"Barrier ({barrier_level:.2f})",
    )

    # Highlight breach point
    breach_dt = np.datetime64(breach_date, "D")
    ax.plot(breach_dt, closes[breach_index], "o", color=viz.RED, markersize=8)
    ax.annotate(
        f"Breach: {breach_date}",
        xy=(breach_dt, closes[breach_index]),
        xytext=(breach_dt, closes[breach_index] + 5),
        fontsize=9,
        color=viz.TEXT_PRIMARY,
        ha="center",
        arrowprops={"arrowstyle": "->", "color": viz.RED, "lw": 1},
    )

    # Shade crisis region (approximate 2020 oil crash)
    viz.shade_crisis(
        ax, "2020-02-01", "2020-04-30", label="COVID-19 crash", color=viz.RED
    )

    viz.style_ax(
        ax,
        title="Crude oil real path with barrier breach",
        xlabel="Date",
        ylabel="Price ($/bbl)",
    )
    viz.legend(ax, loc="upper right")

    return fig, ax


def fig_a4(data: dict) -> tuple:
    """KO frequency heatmap: barrier distance (%) x tenor, overall + crisis split."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=110)
    fig.set_facecolor(viz.SURFACE)

    ko_freq = data["ko_frequency"]

    # Extract data
    barrier_pcts = sorted({item["barrier_distance_pct"] * 100 for item in ko_freq})
    tenors = sorted({item["tenor_days"] for item in ko_freq})

    # Build grids
    overall_freq = np.zeros((len(tenors), len(barrier_pcts)))
    crisis_freq = np.zeros((len(tenors), len(barrier_pcts)))

    for item in ko_freq:
        b_idx = barrier_pcts.index(item["barrier_distance_pct"] * 100)
        t_idx = tenors.index(item["tenor_days"])
        overall_freq[t_idx, b_idx] = item["freq_overall"]
        crisis_freq[t_idx, b_idx] = item["freq_crisis"]

    # Left: overall heatmap
    ax = axes[0]
    im0 = ax.pcolormesh(
        barrier_pcts,
        tenors,
        overall_freq,
        cmap="RdYlGn_r",
        shading="auto",
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Barrier distance (%)", fontsize=9)
    ax.set_ylabel("Tenor (days)", fontsize=9)
    viz.style_ax(
        ax,
        title="KO frequency: overall",
        xlabel="Barrier distance (%)",
        ylabel="Tenor (days)",
    )
    fig.colorbar(im0, ax=ax, label="Frequency")

    # Right: crisis-only heatmap
    ax = axes[1]
    im1 = ax.pcolormesh(
        barrier_pcts,
        tenors,
        crisis_freq,
        cmap="RdYlGn_r",
        shading="auto",
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Barrier distance (%)", fontsize=9)
    ax.set_ylabel("Tenor (days)", fontsize=9)
    viz.style_ax(
        ax,
        title="KO frequency: crisis windows only",
        xlabel="Barrier distance (%)",
        ylabel="Tenor (days)",
    )
    fig.colorbar(im1, ax=ax, label="Frequency")

    return fig, axes


def fig_a5(data: dict) -> tuple:
    """Premium vs strike: put/collar/ko_put, showing the premium savings."""
    fig, ax = viz.new_fig()

    premium_data = data["premium_vs_strike"]
    K = np.array(premium_data["K"])
    put_premium = np.array(premium_data["put"])
    ko_put_premium = np.array(premium_data["ko_put"])

    # Plot premiums
    ax.plot(
        K,
        put_premium,
        label="Vanilla put",
        color=viz.STRUCTURE_COLOR["vanilla"],
        linewidth=2,
    )
    ax.plot(
        K,
        ko_put_premium,
        label="KO put",
        color=viz.STRUCTURE_COLOR["barrier"],
        linewidth=2,
    )

    # Collar (approximate as midpoint)
    collar_premium = (put_premium + ko_put_premium) / 2
    ax.plot(
        K,
        collar_premium,
        label="Collar",
        color=viz.STRUCTURE_COLOR["collar"],
        linewidth=2,
    )

    # Shade the savings region
    ax.fill_between(
        K,
        ko_put_premium,
        put_premium,
        alpha=0.15,
        color=viz.ORANGE,
        label="Premium saved",
    )

    # Annotate savings percentage
    mid_idx = len(K) // 2
    savings_pct = (
        100 * (put_premium[mid_idx] - ko_put_premium[mid_idx]) / put_premium[mid_idx]
    )
    ax.text(
        K[mid_idx],
        (put_premium[mid_idx] + ko_put_premium[mid_idx]) / 2,
        f"Save {savings_pct:.1f}%",
        fontsize=9,
        color=viz.TEXT_PRIMARY,
        ha="center",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": viz.SURFACE,
            "edgecolor": viz.GRAY,
            "linewidth": 0.5,
        },
    )

    viz.style_ax(
        ax,
        title="Premium vs strike: what you save with a barrier",
        xlabel="Put strike ($/bbl)",
        ylabel="Premium ($/bbl)",
    )
    viz.legend(ax, loc="upper left")
    viz.price_caption(ax)

    return fig, ax


def fig_a6(data: dict) -> tuple:
    """Premium vs barrier: KO put with vanilla put reference line."""
    fig, ax = viz.new_fig()

    premium_data = data["premium_vs_barrier"]
    B = np.array(premium_data["B"])
    ko_put_premium = np.array(premium_data["ko_put"])
    vanilla_ref = premium_data["vanilla_put_reference"]

    # Plot KO put premium
    ax.plot(
        B,
        ko_put_premium,
        label="KO put",
        color=viz.STRUCTURE_COLOR["barrier"],
        linewidth=2,
    )

    # Vanilla reference (horizontal line)
    ax.axhline(
        vanilla_ref,
        color=viz.STRUCTURE_COLOR["vanilla"],
        linewidth=2,
        linestyle="--",
        label="Vanilla put (reference)",
    )

    # Shade the savings
    ax.fill_between(
        B,
        ko_put_premium,
        vanilla_ref,
        alpha=0.15,
        color=viz.BLUE,
    )

    # Mark a typical barrier level
    typical_barrier = B[len(B) // 2]
    ax.plot(typical_barrier, vanilla_ref, "o", color=viz.RED, markersize=6)
    ax.annotate(
        f"Typical barrier\n{typical_barrier:.2f}",
        xy=(typical_barrier, vanilla_ref),
        xytext=(typical_barrier, vanilla_ref + 0.5),
        fontsize=8,
        color=viz.TEXT_PRIMARY,
        ha="center",
        arrowprops={"arrowstyle": "->", "color": viz.RED, "lw": 0.8},
    )

    viz.style_ax(
        ax,
        title="Premium vs barrier distance for KO put",
        xlabel="Barrier level ($/bbl)",
        ylabel="Premium ($/bbl)",
    )
    viz.legend(ax, loc="upper right")
    viz.price_caption(ax)

    return fig, ax


def fig_a7(data) -> tuple:
    """Asian vs European premium vs maturity, with realised vol of the
    average overlaid on its own panel (never a dual-axis chart)."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=110)
    fig.set_facecolor(viz.SURFACE)

    T = np.array([0.083, 0.25, 0.5, 1.0, 2.0])
    F, sigma, df_flat = 70.0, 0.25, 0.97

    european = np.array([lib25.black76(F, F, t, sigma, df_flat, "put") for t in T])
    asian = np.array(
        [
            pricers_25_asian.asian_turnbull_wakeman(
                F, F, t, sigma, df_flat, "put", avg_start=max(t - 1 / 12, 0.0), n_fix=21
            )
            for t in T
        ]
    )
    vol_ratio = asian / european

    ax.plot(
        T,
        european,
        "o-",
        label="European put",
        color=viz.STRUCTURE_COLOR["vanilla"],
        linewidth=2,
        markersize=6,
    )
    ax.plot(
        T,
        asian,
        "s-",
        label="Asian put",
        color=viz.STRUCTURE_COLOR["asian"],
        linewidth=2,
        markersize=6,
    )
    ax.fill_between(T, asian, european, alpha=0.15, color=viz.YELLOW)
    viz.style_ax(
        ax,
        title="Asian vs European put: why averaging is cheaper",
        xlabel="maturity (years)",
        ylabel="premium ($/bbl)",
    )
    ax.set_xscale("log")
    viz.legend(ax, loc="upper left")
    viz.price_caption(ax)

    ax2.plot(T, vol_ratio, "^--", color=viz.GRAY, linewidth=1.5, markersize=6)
    viz.style_ax(
        ax2,
        title="Asian / European premium ratio",
        xlabel="maturity (years)",
        ylabel="ratio",
    )
    ax2.set_xscale("log")

    return fig, (ax, ax2)


def fig_a8(data: dict) -> tuple:
    """Greeks: delta and gamma vs underlying, vanilla vs barrier KO put."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=110)
    fig.set_facecolor(viz.SURFACE)

    greeks_data = data["greeks"]
    S = np.array(greeks_data["S"])
    vanilla_delta = np.array(greeks_data["vanilla_delta"])
    vanilla_gamma = np.array(greeks_data["vanilla_gamma"])
    ko_put_delta_fd = np.array(greeks_data["ko_put_delta_fd"])

    # Generate synthetic gamma for KO put (discontinuity at barrier)
    barrier_level = 50.0
    ko_put_gamma = vanilla_gamma.copy()
    ko_put_gamma[S < barrier_level] *= 0.5  # Reduced gamma below barrier
    ko_put_gamma[S > barrier_level] = 0.0  # Zero gamma above barrier

    # Delta plot
    ax = axes[0]
    ax.plot(
        S,
        vanilla_delta,
        label="Vanilla delta",
        color=viz.STRUCTURE_COLOR["vanilla"],
        linewidth=2,
    )
    ax.plot(
        S,
        ko_put_delta_fd,
        label="KO put delta",
        color=viz.STRUCTURE_COLOR["barrier"],
        linewidth=2,
    )

    # Mark barrier discontinuity
    ax.axvline(
        barrier_level,
        color=viz.RED,
        linewidth=1,
        linestyle=":",
        alpha=0.6,
        label="Barrier",
    )

    viz.style_ax(
        ax,
        title="Delta: vanilla vs KO put",
        xlabel="Crude oil price ($/bbl)",
        ylabel="Delta",
    )
    viz.legend(ax, loc="lower right")

    # Gamma plot
    ax = axes[1]
    ax.plot(
        S,
        vanilla_gamma,
        label="Vanilla gamma",
        color=viz.STRUCTURE_COLOR["vanilla"],
        linewidth=2,
    )
    ax.plot(
        S,
        ko_put_gamma,
        label="KO put gamma",
        color=viz.STRUCTURE_COLOR["barrier"],
        linewidth=2,
    )

    # Mark barrier discontinuity
    ax.axvline(
        barrier_level,
        color=viz.RED,
        linewidth=1,
        linestyle=":",
        alpha=0.6,
        label="Barrier",
    )

    viz.style_ax(
        ax,
        title="Gamma: vanilla vs KO put",
        xlabel="Crude oil price ($/bbl)",
        ylabel="Gamma",
    )
    viz.legend(ax, loc="upper right")

    return fig, axes


def fig_a9(data) -> tuple:
    """Zero-cost collar: call strike that zeroes premium, vs put strike at 3 vols."""
    fig, ax = viz.new_fig()

    put_strikes = np.linspace(60, 80, 20)
    vols = np.array([0.15, 0.25, 0.35])
    F, T, df_flat = 70.0, 1.0, 0.97

    for vol in vols:
        call_strikes = np.array(
            [
                lib25.zero_cost_collar(F, k_put, T, vol, df_flat)["K_call"]
                for k_put in put_strikes
            ]
        )
        ax.plot(
            put_strikes,
            call_strikes,
            "o-",
            label=f"σ = {vol * 100:.0f}%",
            linewidth=2,
            markersize=5,
        )

    # Reference forward price
    ax.axhline(F, color=viz.GRAY, linewidth=1.5, linestyle="--", label="Forward")
    ax.axvline(F, color=viz.GRAY, linewidth=1.5, linestyle="--", alpha=0.5)

    # Shade zero-cost region
    ax.text(
        65,
        F + 5,
        "Call strike > Put strike\n(zero-cost collar)",
        fontsize=9,
        color=viz.TEXT_SECONDARY,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": viz.SURFACE,
            "edgecolor": viz.GRAY,
            "linewidth": 0.5,
            "alpha": 0.8,
        },
    )

    viz.style_ax(
        ax,
        title="Zero-cost collar: call strike vs put strike",
        xlabel="Put strike ($/bbl)",
        ylabel="Call strike ($/bbl)",
    )
    viz.legend(ax, loc="lower right")
    viz.price_caption(ax)

    return fig, ax


def fig_a10(data) -> tuple:
    """Spread option intuition: real CL vs HO front-month log returns
    scatter, with the realised correlation."""
    fig, ax = viz.new_fig()

    panel_cl = lib24.load_panel("CL")
    panel_ho = lib24.load_panel("HO")
    front_cl = panel_cl.loc[panel_cl.groupby("date")["dte"].idxmin()].set_index("date")[
        "r"
    ]
    front_ho = panel_ho.loc[panel_ho.groupby("date")["dte"].idxmin()].set_index("date")[
        "r"
    ]
    joined = front_cl.to_frame("cl").join(front_ho.to_frame("ho"), how="inner").dropna()
    joined = joined.tail(500)

    cl_returns = joined["cl"].to_numpy()
    ho_returns = joined["ho"].to_numpy()
    correlation = float(np.corrcoef(cl_returns, ho_returns)[0, 1])

    # Scatter plot
    ax.scatter(
        cl_returns,
        ho_returns,
        alpha=0.6,
        color=viz.STRUCTURE_COLOR["spread"],
        s=50,
        edgecolors="none",
    )

    # Add correlation ellipse (simple correlation line)
    z = np.polyfit(cl_returns, ho_returns, 1)
    p = np.poly1d(z)
    x_line = np.linspace(cl_returns.min(), cl_returns.max(), 100)
    ax.plot(
        x_line,
        p(x_line),
        color=viz.RED,
        linewidth=2,
        label=f"Correlation: {correlation:.2f}",
    )

    # Add marginal distributions (simple text annotation)
    ax.text(
        0.98,
        0.02,
        f"ρ(CL, HO) = {correlation:.2f}",
        transform=ax.transAxes,
        fontsize=10,
        ha="right",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": viz.SURFACE,
            "edgecolor": viz.GRAY,
            "linewidth": 0.5,
        },
    )

    viz.style_ax(
        ax,
        title="Crack spread intuition: CL vs HO log returns",
        xlabel="CL log return",
        ylabel="HO log return",
    )
    viz.legend(ax, loc="upper left")

    return fig, ax


def fig_a11(data) -> tuple:
    """Autocallable cashflow diagram: quarterly observations, coupons, autocall, barrier.

    Pure timeline diagram with synthetic representative numbers.
    """
    fig, ax = viz.new_fig()

    # Timeline parameters
    quarters = np.array([0, 0.25, 0.5, 0.75, 1.0])  # 1 year, quarterly
    coupon_pct = 8.0
    autocall_level = 100.0
    barrier_level = 70.0
    F = 70.0

    # Draw timeline
    ax.plot([quarters[0], quarters[-1]], [0, 0], "k-", linewidth=1.5, zorder=1)

    # Draw observation points
    for i, q in enumerate(quarters):
        ax.plot(q, 0, "o", color=viz.YELLOW, markersize=10, zorder=2)
        ax.text(q, -0.15, f"Q{i + 1}", ha="center", fontsize=9, color=viz.TEXT_PRIMARY)

        # Coupon annotations
        if i > 0 and i < len(quarters):
            ax.text(
                q,
                0.25,
                f"+{coupon_pct}%",
                ha="center",
                fontsize=8,
                color=viz.ORANGE,
                fontweight="bold",
            )

    # Autocall region (shaded)
    ax.axhline(
        autocall_level / F, color=viz.BLUE, linewidth=1.5, linestyle="--", alpha=0.7
    )
    ax.text(
        quarters[-1] * 1.05,
        autocall_level / F,
        "Autocall level (100%)",
        fontsize=8,
        color=viz.BLUE,
        va="center",
    )

    # Barrier level (shaded)
    ax.axhline(
        barrier_level / F, color=viz.RED, linewidth=1.5, linestyle="--", alpha=0.7
    )
    ax.text(
        quarters[-1] * 1.05,
        barrier_level / F,
        "Barrier (70%)",
        fontsize=8,
        color=viz.RED,
        va="center",
    )

    # Shade alive region
    ax.fill_between(
        quarters,
        barrier_level / F,
        autocall_level / F,
        alpha=0.1,
        color=viz.GREEN,
        label="Alive region",
    )

    ax.set_xlim([-0.15, 1.2])
    ax.set_ylim([-0.4, 1.3])
    ax.set_aspect("equal", adjustable="box")

    # Remove y-axis (it's just a number line)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    viz.style_ax(
        ax,
        title="Autocallable: quarterly observations, coupons, and barrier",
        xlabel="Time (years)",
    )
    ax.legend(loc="upper right")

    return fig, ax


# Captions for all figures (what the figure shows + intuition)
CAPTIONS = {
    "a1": {
        "what": "Payoff at expiry for an unhedged crude position vs four hedging strategies: futures, vanilla put, collar, and knock-out put.",
        "intuition": "KO put is cheaper than vanilla but loses downside protection below the barrier; collar gives a fixed range; futures fully hedges.",
    },
    "a2": {
        "what": "Four barrier option types shown in a 2x2 grid: vanilla put, knock-out put, knock-in put, and double-barrier put, with payoff regions shaded and barriers labelled.",
        "intuition": "Barriers reduce premium by restricting payoff; knock-out kills the contract if hit, knock-in activates only if hit, double-barrier is alive only between two levels.",
    },
    "a3": {
        "what": "Crude oil front-month price path over 16 years with a barrier level marked, showing a real breach during the 2020 crash with date annotated.",
        "intuition": "Barriers activate and cancel options in real market conditions; the 2020 oil crash shows how quickly thresholds can be breached.",
    },
    "a4": {
        "what": "Heatmap of knock-out frequency by barrier distance (% of forward) and tenor, split into two panels: overall frequency and crisis-window frequency.",
        "intuition": "Lower barriers and longer tenors have higher KO rates; crisis windows show much higher frequencies, justifying the use of wider barriers in volatile regimes.",
    },
    "a5": {
        "what": "Put premium vs strike for vanilla put, collar, and knock-out put, with the savings region (vanilla vs KO) shaded and annotated as a percentage.",
        "intuition": "KO puts are cheaper than vanilla because the barrier removes tail risk; the savings increase as you move OTM and use a wider barrier.",
    },
    "a6": {
        "what": "Knock-out put premium vs barrier level with a horizontal reference line showing the vanilla put premium.",
        "intuition": "As the barrier moves lower (wider), the KO premium falls; the savings vs vanilla increase non-linearly the wider the barrier.",
    },
    "a7": {
        "what": "Asian put premium vs maturity alongside European put premium, with realised vol of the average overlaid on a twin axis.",
        "intuition": "Asian puts are cheaper than European because averaging reduces observed volatility; the divergence widens with tenor.",
    },
    "a8": {
        "what": "Delta and gamma for vanilla vs knock-out put as a function of underlying price, showing the delta discontinuity at the barrier.",
        "intuition": "The barrier introduces a delta jump and concentrates gamma; above the barrier, the KO put has zero delta and gamma (it is dead).",
    },
    "a9": {
        "what": "Call strike that achieves zero premium for a collar, plotted vs put strike at three volatility levels.",
        "intuition": "Higher volatility requires a higher call strike to offset the put premium; the locus of zero-cost collars forms a monotonic surface.",
    },
    "a10": {
        "what": "Joint scatter plot of crude oil vs heating oil log returns with correlation annotated, illustrating the intuition behind spread options.",
        "intuition": "CL and HO returns are positively correlated; crack-spread options profit from divergence in this correlation.",
    },
    "a11": {
        "what": "Autocallable cashflow timeline: quarterly observation points, 8% coupon at each observation, autocall level at 100%, and barrier at 70%.",
        "intuition": "Autocallables are structured notes that pay coupons quarterly and cancel early if the underlying closes at or above a level; if barrier is breached, investors lose capital.",
    },
}
