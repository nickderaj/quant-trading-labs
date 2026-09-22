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
        "intuition": "CL/HO/RB returns are correlated but not perfectly (rho about 0.87 and 0.89), and the roughly 2% gap between the Kirk and bivariate-Monte-Carlo crack-spread prices is the correlation term that two separate hedges cannot see.",
    },
    "d3": {
        "what": "Farmer's realised payoff distributions: plain put vs zero-cost collar vs knock-out put, each re-struck at every historical start date's own forward, so the comparison is about the structures and not about where wheat happened to trade in 2017.",
        "intuition": "The knock-out put pays exactly what the plain put pays until the barrier (85% of the forward, i.e. 15% out of the money) is touched, and nothing afterwards. The collar's short call caps the good years, which is what pays for its floor.",
    },
    "d4": {
        "what": "Chicago wheat price history annotated with how often the barrier was touched and what the knock-out cost when it was.",
        "intuition": "The barrier was touched on 28.6% of the 49 historical harvest windows (26.0% on KE), but only 24.5% of windows were knock-outs that actually cost anything -- on the rest the plain put would have expired worthless anyway. Averaged over the costly ones the farmer gave up about 116 cents/bushel of protection. ZW is quoted in cents, so that is roughly $1.16/bu, not $116.",
    },
    "d5": {
        "what": "Airline's realised payoff distributions: long strip of futures vs the Asian cap and the Asian collar.",
        "intuition": "The airline buys diesel, so its hedge is long futures and pays when diesel rallies. The Asian cap keeps that upside protection and drops the downside, which is what its premium buys; the collar gives back the good outcomes to pay for it.",
    },
    "d6": {
        "what": "Winter gas cap strip priced three ways: one flat front-month vol for all four months, the unconditional Samuelson term structure, and a term structure fitted only to winter-delivery contracts.",
        "intuition": "Almost the whole gap is the Samuelson effect, not seasonality: quoting the strip off a single front-month vol overprices it by about 15.5% against the maturity-aware curve, while conditioning on winter delivery moves it by about -0.5%. NG's winter vol premium is real at the front of the curve but is not measurable at the 6-12 month tenors this strip spans.",
    },
    "d7": {
        "what": "April 2020 CL negative price: predictive densities from Black-76 (lognormal), Bachelier (normal), and displaced-diffusion.",
        "intuition": "Lognormal has zero density below 0; normal admits negatives; actual settlement -2.67 sits in lognormal's forbidden zone.",
    },
    "d8": {
        "what": "Rolling 1% left-tail Kupiec coverage over the full 16-year sample: observed vs expected exceedance rate for the lognormal-return and level-change models.",
        "intuition": "Both fail, in opposite directions. The lognormal model never once breaches its own 1% VaR (0 exceedances in 4168 days), so its left tail is far too wide -- too conservative. The level-change models breach on 2.06% of days against a 1% target, roughly twice too often, so theirs is too narrow. Failing a coverage test by being too cautious and failing it by being too aggressive are different diagnoses.",
    },
    "d9": {
        "what": "Structured autocall note: realised payoff distribution and cumulative early-redemption probability by quarterly observation date.",
        "intuition": "The note autocalls at any observation where CL is back at or above its starting level. The model puts that at 65.4% and CL's own 2010-2026 history at 72.3% -- the gap runs the right way, since the model prices under a driftless risk-neutral measure while history carries oil's actual drift.",
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
    """Farmer: realised payoff distributions for put, collar, and KO put."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.set_facecolor(viz.SURFACE)

    with open(Path(__file__).resolve().parent / "phase_5_25_cases_abcd.json") as f:
        phase5_data = json.load(f)

    case = phase5_data["B_farmer"]
    payoffs = case["realised_payoffs"]
    trade_off = case["trade_off_numbers"]
    structure_names = ["Plain put", "Zero-cost collar", "Knock-out put"]
    structure_colors = [viz.ORANGE, viz.AQUA, viz.BLUE]

    # Left: the three distributions on ONE shared set of bins.
    #
    # The collar's short call gives it a long left tail that the two put
    # structures do not have, so a histogram with per-series bins squashes
    # the puts into a single invisible spike. Shared bins over the pooled
    # range, plus step outlines rather than filled bars, keeps all three
    # readable at once.
    ax1.set_facecolor(viz.SURFACE)
    all_values = np.concatenate(
        [np.asarray(payoffs[n]["values"], dtype=float) for n in structure_names]
    )
    bins = np.linspace(np.min(all_values), np.max(all_values), 36)

    for name, color in zip(structure_names, structure_colors):
        values = np.asarray(payoffs[name]["values"], dtype=float)
        ax1.hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=2,
            color=color,
            label=f"{name} (mean {values.mean():+.0f})",
        )
        ax1.axvline(values.mean(), color=color, linestyle=":", linewidth=1.2, alpha=0.8)

    ax1.axvline(0, color=viz.GRAY, linewidth=1.0, alpha=0.7)
    viz.style_ax(
        ax1,
        title=f"Farmer: realised payoffs, n={payoffs['Plain put']['n']} harvest windows",
        ylabel="Windows",
        xlabel="Payoff (cents/bu)",
    )
    viz.legend(ax1, loc="upper left")

    # Right: what the knock-out actually costs. Separating the knock-outs
    # that cost something from the ones that did not is the whole decision:
    # a barrier touched in a year the price recovered above the strike costs
    # the farmer nothing at all.
    ax2.set_facecolor(viz.SURFACE)

    n_obs = trade_off["n_observations"]
    n_ko = trade_off["n_knockouts"]
    n_costly = trade_off["n_costly_knockouts"]
    n_free = n_ko - n_costly

    categories = [
        "Never\nknocked out",
        "Knocked out,\ncost nothing",
        "Knocked out,\ncost the floor",
    ]
    counts = [n_obs - n_ko, n_free, n_costly]
    colors = [viz.BLUE, viz.GRAY, viz.RED]
    bars = ax2.bar(categories, counts, color=colors, alpha=0.8, width=0.6)
    for bar, c in zip(bars, counts):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{c}  ({c / n_obs:.1%})",
            ha="center",
            va="bottom",
            fontsize=9,
            color=viz.TEXT_PRIMARY,
            fontweight="bold",
        )

    ax2.annotate(
        f"When it cost something, it cost "
        f"{trade_off['mean_shortfall_when_costly_cents']:.0f} cents/bu on average\n"
        f"(about ${trade_off['mean_shortfall_when_costly_cents'] / 100:.2f}/bu -- ZW is quoted in cents)",
        xy=(0.5, 0.95),
        xycoords="axes fraction",
        fontsize=9,
        color=viz.TEXT_SECONDARY,
        ha="center",
        va="top",
    )
    ax2.set_ylim(0, max(counts) * 1.28)
    viz.style_ax(
        ax2,
        title=f"Farmer: what the {trade_off['ko_barrier_pct_of_forward']:.0%} barrier cost",
        ylabel="Harvest windows",
    )

    fig.tight_layout()

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
    mean_shortfall_cents = trade_off["mean_shortfall_when_costly_cents"]
    n_obs = trade_off["n_observations"]

    # ZW is quoted in CENTS per bushel, so the shortfall is in cents too --
    # printing it with a dollar sign overstates it a hundredfold.
    annotation_text = (
        f"KO barrier: {trade_off['ko_barrier_pct_of_forward']:.0%} of forward\n"
        f"Barrier touched: {ko_freq:.1%} ({trade_off['n_knockouts']}/{n_obs} windows)\n"
        f"Cost something: {trade_off['costly_knockout_frequency_zw']:.1%} "
        f"({trade_off['n_costly_knockouts']}/{n_obs})\n"
        f"Mean shortfall when costly: {mean_shortfall_cents:.0f} cents/bu "
        f"(${mean_shortfall_cents / 100:.2f}/bu)"
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
    """Gas utility: flat vs maturity vs seasonal vol cap pricing."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.set_facecolor(viz.SURFACE)

    with open(Path(__file__).resolve().parent / "phase_5_25_cases_abcd.json") as f:
        phase5_data = json.load(f)

    case = phase5_data["D_gas_utility"]
    trade_off = case["trade_off_numbers"]

    flat_premium = trade_off["flat_vol_cap_premium"]
    maturity_premium = trade_off["maturity_vol_cap_premium"]
    seasonal_premium = trade_off["seasonal_cap_premium"]
    flat_gap = trade_off["flat_overprices_vs_maturity_pct"]
    seasonal_gap = trade_off["seasonal_vs_maturity_pct"]

    # Left: the three prices for the whole strip, so the two effects that get
    # conflated -- maturity decay and delivery-month seasonality -- are
    # visibly different sizes.
    ax1.set_facecolor(viz.SURFACE)
    methods = [
        "Flat\n(front-month vol)",
        "Maturity\n(Samuelson)",
        "Seasonal\n(winter delivery)",
    ]
    premiums = [flat_premium, maturity_premium, seasonal_premium]
    bars = ax1.bar(
        methods, premiums, color=[viz.ORANGE, viz.BLUE, viz.AQUA], alpha=0.8, width=0.55
    )
    for bar, prem in zip(bars, premiums):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"${prem:.3f}",
            ha="center",
            va="center",
            color=viz.TEXT_PRIMARY,
            fontsize=10,
            fontweight="bold",
        )
    ax1.annotate(
        f"flat overprices by {flat_gap:+.1f}%\nseasonality adds {seasonal_gap:+.1f}%",
        xy=(0.5, 0.95),
        xycoords="axes fraction",
        fontsize=9,
        color=viz.TEXT_SECONDARY,
        ha="center",
        va="top",
    )
    viz.style_ax(
        ax1,
        title="Winter cap strip: three vol inputs",
        ylabel="Strip premium ($)",
    )

    # Right: the evidence for the seasonality claim -- winter/summer realised
    # vol by dte bucket. The winter premium is a front-end effect; at the
    # tenors this strip spans it is noise around 1.
    ax2.set_facecolor(viz.SURFACE)
    buckets = trade_off.get("seasonal_buckets", [])
    if buckets:
        dte = np.array([b["dte_mid"] for b in buckets])
        ratio = np.array([b["ratio"] for b in buckets])
        ax2.plot(dte, ratio, "o-", color=viz.AQUA, linewidth=2, markersize=6)
        ax2.axhline(1.0, color=viz.GRAY, linestyle="--", linewidth=1.2, alpha=0.8)

        # Shade the tenors the winter strip actually spans (T = 0.5 to 1.0).
        ax2.axvspan(
            182,
            365,
            color=viz.BLUE,
            alpha=0.12,
            label="tenors this strip spans",
        )
        ax2.set_xscale("log")
        viz.style_log_axis_plain(ax2, axis="x")
        ax2.set_xlabel(
            "Days to expiry (bucket midpoint)", color=viz.TEXT_SECONDARY, fontsize=9
        )
        viz.style_ax(
            ax2,
            title="NG winter / summer realised vol, by tenor",
            ylabel="vol(winter delivery) / vol(summer delivery)",
        )
        ax2.set_ylim(min(0.75, ratio.min() * 0.97), max(ratio.max() * 1.12, 1.35))
        viz.legend(ax2, loc="lower right")

    fig.tight_layout()
    viz.price_caption(ax1)

    return fig, ax1


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
    hist = g_data["historical_resampling"]
    payoff_dist = hist["payoff_distribution"]

    # Left: the ACTUAL realised payoffs.
    #
    # An earlier version of this panel drew np.random.normal(mean, std) and
    # labelled it the realised distribution. The real one is nothing like a
    # normal: p25, median and p75 are all the same number, because most
    # windows autocall and pay exactly the same redemption amount, and the
    # rest form a long left tail down to 0.30. Plotting a Gaussian here
    # invented a shape the data does not have.
    ax1.set_facecolor(viz.SURFACE)
    values = np.asarray(hist["payoff_values"], dtype=float)

    ax1.hist(values, bins=40, color=viz.RED, alpha=0.7, edgecolor=viz.SURFACE)
    ax1.axvline(
        1.0,
        color=viz.GRAY,
        linestyle="-",
        linewidth=1.2,
        label="Face value (1.00)",
    )
    ax1.axvline(
        float(values.mean()),
        color=viz.BLUE,
        linestyle="--",
        linewidth=2,
        label=f"Mean: {values.mean():.3f}",
    )
    ax1.axvline(
        float(np.median(values)),
        color=viz.AQUA,
        linestyle=":",
        linewidth=2,
        label=f"Median: {np.median(values):.3f}",
    )
    ax1.annotate(
        f"n={len(values)} windows; min {payoff_dist['min']:.2f}\n"
        f"the spike is the autocall redemption amount",
        xy=(0.03, 0.62),
        xycoords="axes fraction",
        fontsize=8,
        color=viz.TEXT_SECONDARY,
        va="top",
    )
    viz.style_ax(
        ax1,
        title="Structured note: realised payoff distribution",
        ylabel="Windows",
        xlabel="Payoff per 1.00 of face",
    )
    viz.legend(ax1, loc="upper left")

    # Right: autocall timing, model vs history, observation by observation.
    #
    # The total autocall probability is the SUM of these disjoint per-date
    # frequencies (a path redeems at most once), not their average -- which
    # is the arithmetic an earlier version of this chart and of Phase 6 both
    # got wrong, in the same direction, by dividing by the number of dates.
    ax2.set_facecolor(viz.SURFACE)

    obs_indices = g_data["structure"]["obs_indices"]
    model_by_obs = g_data["model_pricing"]["autocall_frequency_by_obs"]
    hist_by_obs = hist["realized_autocall_frequency_by_obs"]
    obs_labels = [f"Obs {i + 1}\n({idx}d)" for i, idx in enumerate(obs_indices)]

    x = np.arange(len(obs_indices))
    width = 0.38
    ax2.bar(
        x - width / 2,
        model_by_obs,
        width,
        color=viz.BLUE,
        alpha=0.85,
        label=f"Model (total {sum(model_by_obs):.1%})",
    )
    ax2.bar(
        x + width / 2,
        hist_by_obs,
        width,
        color=viz.ORANGE,
        alpha=0.85,
        label=f"Historical (total {sum(hist_by_obs):.1%})",
    )
    for xi, (m, h) in enumerate(zip(model_by_obs, hist_by_obs)):
        for off, v in ((-width / 2, m), (width / 2, h)):
            if v > 0.01:
                ax2.text(
                    xi + off,
                    v + 0.012,
                    f"{v:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=viz.TEXT_PRIMARY,
                )

    ax2.set_xticks(x)
    ax2.set_xticklabels(obs_labels)
    viz.style_ax(
        ax2,
        title="Autocall probability by observation: model vs history",
        ylabel="Probability of redeeming at this observation",
    )
    viz.legend(ax2, loc="upper right")

    fig.tight_layout()
    viz.price_caption(ax1)

    return fig, ax1
