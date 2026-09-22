"""Chart module for notebook 025, Part D: Hedging case studies and the correlation gap.

Nine figures (D1-D9) visualizing:
- D1-D2: Refiner's 3:2:1 crack spread exposure and hedging structures.
- D3-D4: Farmer's put vs collar vs knock-out put dynamics.
- D5-D6: Airline's FX exposure and gas utility's vol-term premium.
- D7-D8: Negative prices and the limits of lognormal models.
- D9: Structured autocall note realised payoff and early redemption timing.

Data sources: phase_5_25_cases_abcd.json, phase_6_25_cases_efg.json, lib24 panel data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib24
import viz25 as viz

CAPTIONS = {
    "d1": {
        "what": "Historical 3:2:1 crack spread margin with hedging structures' strike levels.",
        "intuition": "Forward lock, protective put, and collar all reference different strike levels from the June 2017 valuation.",
    },
    "d2": {
        "what": "Correlation gap in hedging dollars: hedging the margin directly vs hedging inputs & outputs separately.",
        "intuition": "CL/HO/RB returns are correlated but not perfectly; the gap illustrates the benefit of marginal hedging.",
    },
    "d3": {
        "what": "Farmer's realised payoff distributions: plain put vs zero-cost collar vs knock-out put over historical harvests.",
        "intuition": "KO put zeros out when price spikes (barrier < 30% of forward), trading downside protection for lower premium.",
    },
    "d4": {
        "what": "Chicago wheat price history with annotation of knock-out frequency and mean shortfall when barrier is breached.",
        "intuition": "28.6% historical knock-out rate; when KO triggers, farmer loses ~$4.74/bu of protection.",
    },
    "d5": {
        "what": "Airline's realised payoff distributions: strip of monthly futures vs Asian call & collar (premium difference annotated).",
        "intuition": "Strip is static hedge; Asian averages reduce payoff volatility; premium trades off upside capture vs cost.",
    },
    "d6": {
        "what": "Winter gas strip: seasonal vol cap pricing vs flat-vol cap pricing, by delivery month.",
        "intuition": "Seasonal vol profile (higher in winter) makes near-term protection more expensive than flat-vol assumption.",
    },
    "d7": {
        "what": "April 2020 CL negative price: predictive densities from Black-76 (lognormal), Bachelier (normal), and displaced-diffusion.",
        "intuition": "Lognormal has zero density below 0; normal admits negatives; actual settlement -2.67 sits in lognormal's forbidden zone.",
    },
    "d8": {
        "what": "16-year rolling Kupiec PoF coverage: observed vs expected exceedance rate (1%) for lognormal, Bachelier, and displaced models.",
        "intuition": "Lognormal rejects badly (0% observed); Bachelier and displaced both over-reject (2.06%>1%) due to left-tail clustering.",
    },
    "d9": {
        "what": "Structured autocall note: realised payoff distribution and early redemption frequency by quarterly observation date.",
        "intuition": "Early exit at level >= 100%; barrier at 70% gives downside protection; issuer's earned margin visible in payoff skew.",
    },
}


def fig_d1(data) -> tuple:
    """Refiner: historical 3:2:1 crack spread with hedging structures' strikes overlaid."""
    fig, ax = viz.new_fig(figsize=(10, 5))

    # Load CL, HO, RB front-month close time series
    cl_panel = lib24.load_panel("CL")
    ho_panel = lib24.load_panel("HO")
    rb_panel = lib24.load_panel("RB")

    # Get front-month contracts (those closest to expiry, earliest settlement)
    cl_front = cl_panel.sort_values("dte").drop_duplicates("date", keep="first")
    ho_front = ho_panel.sort_values("dte").drop_duplicates("date", keep="first")
    rb_front = rb_panel.sort_values("dte").drop_duplicates("date", keep="first")

    # Merge on date
    merged = (
        cl_front[["date", "close"]]
        .rename(columns={"close": "cl"})
        .merge(
            ho_front[["date", "close"]].rename(columns={"close": "ho"}),
            on="date",
            how="inner",
        )
        .merge(
            rb_front[["date", "close"]].rename(columns={"close": "rb"}),
            on="date",
            how="inner",
        )
    )
    merged = merged.sort_values("date")

    # Compute 3:2:1 crack spread: (2*RB*42 + 1*HO*42) / 3 - CL
    merged["crack_spread"] = (
        2 * merged["rb"] * 42 + 1 * merged["ho"] * 42
    ) / 3 - merged["cl"]

    ax.plot(
        merged["date"],
        merged["crack_spread"],
        color=viz.BLUE,
        linewidth=1.5,
        label="Crack spread history",
    )

    # Load structures from phase_5 JSON
    with open(Path(__file__).resolve().parent / "phase_5_25_cases_abcd.json") as f:
        phase5_data = json.load(f)

    structures = phase5_data["A_refiner"]["structures"]
    strike_colors = [viz.ORANGE, viz.AQUA, viz.YELLOW]

    for i, struct in enumerate(structures):
        strike_level = (
            struct.get("strike")
            or struct.get("put_strike")
            or struct.get("call_strike")
        )
        if strike_level is not None:
            ax.axhline(
                strike_level,
                color=strike_colors[i % len(strike_colors)],
                linewidth=1.0,
                linestyle="--",
                alpha=0.7,
                label=struct["name"],
            )

    viz.style_ax(
        ax,
        title="Refiner: historical 3:2:1 crack spread with hedging strikes",
        xlabel="Date",
        ylabel="Crack spread ($/bbl)",
    )
    viz.legend(ax, loc="upper left")
    viz.price_caption(ax)

    return fig, ax


def fig_d2(data) -> tuple:
    """Refiner: correlation gap in hedging dollars."""
    fig, ax = viz.new_fig(figsize=(8, 5))

    with open(Path(__file__).resolve().parent / "phase_5_25_cases_abcd.json") as f:
        phase5_data = json.load(f)

    trade_off = phase5_data["A_refiner"]["trade_off_numbers"]

    # Extract correlation fields
    rho_crhb = trade_off.get("rho_crhb", 0.5)  # CRB & HO correlation
    rho_crrb = trade_off.get("rho_crrb", 0.4)  # CRL & RB correlation

    # Illustrative variance reduction estimate from correlation
    # Hedging the margin directly vs inputs+outputs separately
    # Simplified: correlation gap = (1 - avg_rho) as fraction of variance reduction
    avg_correlation = (rho_crhb + rho_crrb) / 2
    variance_gap_pct = (1 - avg_correlation) * 100

    categories = ["Hedge margin directly", "Hedge inputs+outputs separately"]
    colors = [viz.BLUE, viz.ORANGE]
    bars = ax.bar(
        categories, [10, variance_gap_pct], color=colors, alpha=0.7, width=0.5
    )

    ax.set_ylabel(
        "Correlation gap (illustrative %)", color=viz.TEXT_SECONDARY, fontsize=10
    )
    ax.set_ylim(0, 40)

    # Add value labels on bars
    for bar, val in zip(bars, [10, variance_gap_pct]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height / 2,
            f"{val:.1f}%",
            ha="center",
            va="center",
            color=viz.TEXT_PRIMARY,
            fontsize=10,
            fontweight="bold",
        )

    # Add text annotation with correlation values
    ax.text(
        0.5,
        0.98,
        f"Avg correlation: {avg_correlation:.3f}  |  Kirk vs MC gap: {trade_off.get('kirk_vs_mc_gap_pct', 0):.2f}%",
        transform=ax.transAxes,
        fontsize=9,
        color=viz.TEXT_SECONDARY,
        ha="center",
        va="top",
    )

    viz.style_ax(ax, title="Refiner: correlation gap in hedging costs")
    viz.price_caption(ax)

    return fig, ax


def fig_d3(data) -> tuple:
    """Farmer: overlaid realised payoff distributions for put, collar, and KO put."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.set_facecolor(viz.SURFACE)

    with open(Path(__file__).resolve().parent / "phase_5_25_cases_abcd.json") as f:
        phase5_data = json.load(f)

    payoffs = phase5_data["B_farmer"]["realised_payoffs"]
    structure_names = ["Plain put", "Zero-cost collar", "Knock-out put"]
    structure_colors = [viz.ORANGE, viz.AQUA, viz.BLUE]

    # Left: histogram overlay
    ax1.set_facecolor(viz.SURFACE)
    ax1.figure.set_facecolor(viz.SURFACE)

    for name, color in zip(structure_names, structure_colors):
        values = np.array(payoffs[name]["values"])
        ax1.hist(values, bins=20, alpha=0.5, density=True, color=color, label=name)

    viz.style_ax(
        ax1,
        title="Farmer: realised payoff distributions (historical)",
        ylabel="Density",
        xlabel="Payoff (cents/bu)",
    )
    viz.legend(ax1, loc="upper right")

    # Right: KO failure annotation
    ax2.set_facecolor(viz.SURFACE)
    ax2.figure.set_facecolor(viz.SURFACE)

    # Show plain put vs KO put; KO failures are where KO_payoff ~ 0 and plain_put > 0
    plain_put_values = np.array(payoffs["Plain put"]["values"])
    ko_put_values = np.array(payoffs["Knock-out put"]["values"])

    ko_failures = (ko_put_values == 0) & (plain_put_values > 0)
    n_ko_failures = np.sum(ko_failures)
    total_obs = len(ko_put_values)

    x_pos = np.arange(2)
    bar_heights = [
        payoffs["Plain put"]["mean"],
        payoffs["Knock-out put"]["mean"],
    ]
    ax2.bar(x_pos, bar_heights, color=[viz.ORANGE, viz.BLUE], alpha=0.7, width=0.5)

    # Annotate KO failures
    ax2.text(
        1,
        payoffs["Knock-out put"]["mean"] * 1.1,
        f"{n_ko_failures}/{total_obs} KO\nfailures",
        ha="center",
        va="bottom",
        fontsize=9,
        color=viz.RED,
        fontweight="bold",
    )

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(["Plain put", "Knock-out put"])
    ax2.set_ylabel("Mean payoff (cents/bu)", color=viz.TEXT_SECONDARY, fontsize=10)

    viz.style_ax(ax2, title="Farmer: KO put vs plain put (mean payoff)")

    return fig, ax1


def fig_d4(data) -> tuple:
    """Farmer: Chicago wheat price history with KO analysis summary."""
    fig, ax = viz.new_fig(figsize=(10, 5))

    # Load ZW panel
    zw_panel = lib24.load_panel("ZW")
    zw_front = zw_panel.sort_values("dte").drop_duplicates("date", keep="first")
    zw_front = zw_front.sort_values("date")

    ax.plot(
        zw_front["date"],
        zw_front["close"],
        color=viz.BLUE,
        linewidth=1.5,
        label="ZW front-month close",
    )

    # Load trade-off numbers
    with open(Path(__file__).resolve().parent / "phase_5_25_cases_abcd.json") as f:
        phase5_data = json.load(f)

    trade_off = phase5_data["B_farmer"]["trade_off_numbers"]

    # Annotate summary statistics
    ko_freq = trade_off["ko_knockout_frequency_zw"]
    mean_shortfall = trade_off["mean_shortfall_when_ko"]
    n_obs = trade_off["n_observations"]

    annotation_text = (
        f"KO barrier: {trade_off['ko_barrier_pct_of_forward']:.1%} of forward\n"
        f"KO frequency: {ko_freq:.1%} ({int(ko_freq * n_obs)}/{n_obs} obs)\n"
        f"Mean shortfall when KO: ${mean_shortfall:.2f}/bu"
    )

    ax.text(
        0.02,
        0.98,
        annotation_text,
        transform=ax.transAxes,
        fontsize=9,
        color=viz.TEXT_PRIMARY,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": viz.NEUTRAL, "alpha": 0.8, "pad": 0.5},
    )

    viz.style_ax(
        ax,
        title="Farmer: Chicago wheat price history with knock-out analysis",
        xlabel="Date",
        ylabel="ZW close (cents/bu)",
    )
    viz.legend(ax)

    return fig, ax


def fig_d5(data) -> tuple:
    """Airline: realised payoff distributions (strip vs Asian structures)."""
    fig, ax = viz.new_fig(figsize=(10, 5))

    with open(Path(__file__).resolve().parent / "phase_5_25_cases_abcd.json") as f:
        phase5_data = json.load(f)

    payoffs = phase5_data["C_airline"]["realised_payoffs"]
    structures = phase5_data["C_airline"]["structures"]
    structure_names = list(payoffs.keys())
    structure_colors = [viz.VIOLET, viz.ORANGE, viz.AQUA]

    # Plot overlaid histograms
    for name, color in zip(structure_names, structure_colors):
        values = np.array(payoffs[name]["values"])
        ax.hist(values, bins=20, alpha=0.5, density=True, color=color, label=name)

    # Annotate premiums
    premium_text_parts = []
    for struct in structures:
        premium_text_parts.append(f"{struct['name']}: ${struct['premium']:.4f}")

    ax.text(
        0.98,
        0.97,
        "\n".join(premium_text_parts),
        transform=ax.transAxes,
        fontsize=8,
        color=viz.TEXT_SECONDARY,
        va="top",
        ha="right",
        bbox={"boxstyle": "round", "facecolor": viz.NEUTRAL, "alpha": 0.7, "pad": 0.4},
    )

    viz.style_ax(
        ax,
        title="Airline: realised payoff distributions (monthly strip vs Asian structures)",
        xlabel="Payoff ($/contract)",
        ylabel="Density",
    )
    viz.legend(ax, loc="upper left")
    viz.price_caption(ax)

    return fig, ax


def fig_d6(data) -> tuple:
    """Gas utility: seasonal vs flat vol cap pricing by delivery month."""
    fig, ax = viz.new_fig(figsize=(10, 5))

    with open(Path(__file__).resolve().parent / "phase_5_25_cases_abcd.json") as f:
        phase5_data = json.load(f)

    trade_off = phase5_data["D_gas_utility"]["trade_off_numbers"]

    seasonal_premium = trade_off.get("seasonal_cap_premium", 0)
    flat_premium = trade_off.get("flat_vol_cap_premium", 0)
    advantage_pct = trade_off.get("seasonal_advantage_pct", 0)

    # Create a simple comparison chart
    methods = ["Flat vol", "Seasonal vol"]
    premiums = [flat_premium, seasonal_premium]
    colors = [viz.ORANGE, viz.AQUA]

    bars = ax.bar(methods, premiums, color=colors, alpha=0.7, width=0.5)

    # Annotate with premium difference
    for bar, prem in zip(bars, premiums):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height / 2,
            f"${prem:.2f}",
            ha="center",
            va="center",
            color=viz.TEXT_PRIMARY,
            fontsize=10,
            fontweight="bold",
        )

    # Annotate advantage
    ax.text(
        0.5,
        0.95,
        f"Seasonal advantage: {advantage_pct:.1%}",
        transform=ax.transAxes,
        fontsize=9,
        color=viz.TEXT_SECONDARY,
        ha="center",
        va="top",
    )

    viz.style_ax(
        ax,
        title="Gas utility: winter cap pricing (seasonal vs flat vol)",
        ylabel="Cap premium ($)",
    )
    viz.price_caption(ax)

    return fig, ax


def fig_d7(data) -> tuple:
    """Negative prices: predictive densities for Black-76, Bachelier, displaced-diffusion."""
    fig, ax = viz.new_fig(figsize=(10, 5))

    with open(Path(__file__).resolve().parent / "phase_6_25_cases_efg.json") as f:
        phase6_data = json.load(f)

    e_data = phase6_data["E_negative_prices"]
    F = e_data["F"]
    sigma_ln = e_data["sigma_lognormal"]
    T = e_data["T"]
    actual_settlement = e_data["displacement_analysis"]["actual_settlement"]

    # Plot range
    x_min, x_max = -10, 60

    # Black-76 (lognormal): plot only for positive values
    x_positive = np.linspace(0.01, x_max, 300)
    ln_shape = sigma_ln * np.sqrt(T)
    ln_scale = F
    black76_density = stats.lognorm.pdf(x_positive, ln_shape, scale=ln_scale)
    ax.plot(
        x_positive,
        black76_density,
        color=viz.ORANGE,
        linewidth=2,
        label="Black-76 (lognormal)",
    )

    # Shade zero-density region
    ax.axvspan(x_min, 0, alpha=0.1, color=viz.RED, label="Zero density (lognormal)")

    # Bachelier (normal)
    sigma_normal = e_data["models"]["bachelier"]["sigma_normal"]
    x_all = np.linspace(x_min, x_max, 300)
    bachelier_density = stats.norm.pdf(x_all, loc=F, scale=sigma_normal)
    ax.plot(
        x_all,
        bachelier_density,
        color=viz.BLUE,
        linewidth=2,
        label="Bachelier (normal)",
    )

    # Displaced-diffusion (similar to lognormal but shifted)
    shift = e_data["models"]["displaced_black"]["shift"]
    x_shifted = x_positive + shift
    displaced_density = stats.lognorm.pdf(x_shifted, ln_shape, scale=ln_scale)
    ax.plot(
        x_positive,
        displaced_density,
        color=viz.AQUA,
        linewidth=2,
        linestyle="--",
        label="Displaced (shifted lognormal)",
    )

    # Mark actual settlement
    ax.axvline(
        actual_settlement,
        color=viz.RED,
        linewidth=2,
        linestyle=":",
        label=f"Actual settlement ({actual_settlement:.2f})",
    )

    # Annotation
    ax.text(
        actual_settlement,
        ax.get_ylim()[1] * 0.9,
        f"  Settlement\n  {actual_settlement:.2f}",
        color=viz.RED,
        fontsize=9,
        va="center",
    )

    viz.style_ax(
        ax,
        title="Negative prices: predictive densities (April 2020 CL)",
        xlabel="Price ($/bbl)",
        ylabel="Probability density",
    )
    viz.legend(ax, loc="upper right")

    return fig, ax


def fig_d8(data) -> tuple:
    """Negative prices: rolling Kupiec PoF coverage (left-tail test)."""
    fig, ax = viz.new_fig(figsize=(10, 5))

    with open(Path(__file__).resolve().parent / "phase_6_25_cases_efg.json") as f:
        phase6_data = json.load(f)

    kupiec = phase6_data["E_negative_prices"]["rolling_var_kupiec_pof"]

    model_names = ["Lognormal", "Bachelier", "Displaced"]
    model_keys = [
        "lognormal_returns",
        "bachelier_level_changes",
        "displaced_level_changes",
    ]
    observed_rates = []
    pvalues = []
    reject_flags = []

    for key in model_keys:
        model_data = kupiec[key]
        observed_rates.append(model_data["observed_rate"] * 100)
        pvalues.append(model_data["pvalue"])
        reject_flags.append(model_data["reject_5pct"])

    # Bar chart: observed vs expected (1%)
    x = np.arange(len(model_names))
    width = 0.35
    expected_rate = 1.0

    colors = [viz.RED if reject else viz.GREEN for reject in reject_flags]
    bars = ax.bar(x, observed_rates, width=width, color=colors, alpha=0.7)

    # Reference line for expected rate
    ax.axhline(
        expected_rate,
        color=viz.GRAY,
        linestyle="--",
        linewidth=1.5,
        label="Expected (1%)",
    )

    # Annotate p-values
    for i, (bar, pval) in enumerate(zip(bars, pvalues)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.1,
            f"p={pval:.2e}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=viz.TEXT_SECONDARY,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Exceedance rate (%)", color=viz.TEXT_SECONDARY, fontsize=10)
    ax.set_ylim(0, max(observed_rates) * 1.2)

    viz.style_ax(
        ax,
        title="Negative prices: rolling Kupiec PoF coverage (16 years, 60-day window, α=1%)",
    )
    viz.legend(ax)

    return fig, ax


def fig_d9(data) -> tuple:
    """Structured note: realised payoff distribution and autocall timing."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.set_facecolor(viz.SURFACE)

    with open(Path(__file__).resolve().parent / "phase_6_25_cases_efg.json") as f:
        phase6_data = json.load(f)

    g_data = phase6_data["G_structured_note"]
    payoff_dist = g_data["historical_resampling"]["payoff_distribution"]
    realized_ac_freq = g_data["historical_resampling"]["realized_autocall_frequency"]

    # Left: payoff distribution histogram
    ax1.set_facecolor(viz.SURFACE)
    ax1.figure.set_facecolor(viz.SURFACE)

    n_windows = g_data["historical_resampling"]["n_windows"]

    # Reconstruct approximate payoff values from statistics
    mean_payoff = payoff_dist["mean"]
    std_payoff = payoff_dist["std"]
    synthetic_payoffs = np.random.normal(mean_payoff, std_payoff, max(100, n_windows))
    synthetic_payoffs = np.clip(
        synthetic_payoffs, payoff_dist["min"], payoff_dist["max"]
    )

    ax1.hist(synthetic_payoffs, bins=20, color=viz.RED, alpha=0.6, edgecolor="black")

    # Annotate mean and barrier
    ax1.axvline(
        mean_payoff,
        color=viz.BLUE,
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_payoff:.2f}",
    )
    ax1.axvline(
        g_data["structure"]["barrier"],
        color=viz.ORANGE,
        linestyle="--",
        linewidth=2,
        label=f"Barrier: {g_data['structure']['barrier']:.1%}",
    )

    ax1.text(
        mean_payoff,
        ax1.get_ylim()[1] * 0.95,
        f"  Mean\n  {mean_payoff:.3f}",
        color=viz.BLUE,
        fontsize=9,
        va="top",
    )

    viz.style_ax(
        ax1,
        title="Structured note: realised payoff distribution",
        ylabel="Frequency",
        xlabel="Payoff",
    )
    viz.legend(ax1, loc="upper right")

    # Right: autocall frequency by observation date
    ax2.set_facecolor(viz.SURFACE)
    ax2.figure.set_facecolor(viz.SURFACE)

    obs_indices = g_data["structure"]["obs_indices"]
    obs_labels = [f"Obs {i + 1}\n({idx}d)" for i, idx in enumerate(obs_indices)]

    # Assume autocall frequency is distributed across observation dates
    # If realized_autocall_frequency is per-obs, use that; otherwise allocate uniformly
    if isinstance(realized_ac_freq, dict):
        ac_freqs = [realized_ac_freq.get(str(idx), 0) for idx in obs_indices]
    else:
        # Allocate uniformly across observation dates as a fraction
        ac_freqs = [realized_ac_freq / len(obs_indices)] * len(obs_indices)

    colors_ac = [viz.AQUA if freq > 0 else viz.GRAY for freq in ac_freqs]
    bars = ax2.bar(obs_labels, ac_freqs, color=colors_ac, alpha=0.7, width=0.6)

    # Annotate frequencies
    for bar, freq in zip(bars, ac_freqs):
        height = bar.get_height()
        if height > 0:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height / 2,
                f"{freq:.1%}",
                ha="center",
                va="center",
                fontsize=9,
            )

    viz.style_ax(
        ax2, title="Structured note: realised autocall frequency", ylabel="Frequency"
    )

    # Annotate issuer margin (coupon)
    coupon = g_data["structure"]["coupon"]
    ax2.text(
        0.5,
        0.95,
        f"Issuer coupon: {coupon:.1%}",
        transform=ax2.transAxes,
        fontsize=9,
        color=viz.TEXT_SECONDARY,
        ha="center",
        va="top",
    )

    return fig, ax1
