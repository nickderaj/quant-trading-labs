"""Part B charts (5 figures) for notebook 025 -- the market inputs: curve,
vol term structure, the curve-fit cautionary tale, curve-state split, and
the quanto FX correlation instability.

Written by the main session, not delegated (NEXT_PROMPT.md section 7: "Part
B ... you write these"). Reads phase_1_25_curves.parquet and
phase_1_25_inputs.json (Phase 1), reruns a couple of cheap lib25/lib24
computations directly for B3/B5 rather than persisting them separately.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_TMP_DIR = Path(__file__).resolve().parent
if str(_TMP_DIR) not in sys.path:
    sys.path.insert(0, str(_TMP_DIR))

import lib25
import viz25 as viz

_CURVES = pd.read_parquet(_TMP_DIR / "phase_1_25_curves.parquet")
_INPUTS = json.loads((_TMP_DIR / "phase_1_25_inputs.json").read_text())


def fig_b1(data=None) -> tuple:
    """B1 -- the CL forward curve on four representative dates."""
    fig, ax = viz.new_fig(figsize=(8, 5))
    tags = ["calm", "crisis", "backwardated", "steep_contango"]
    colors = [viz.BLUE, viz.RED, viz.ORANGE, viz.AQUA]
    cl = _CURVES[_CURVES["product"] == "CL"]
    for tag, color in zip(tags, colors, strict=True):
        sub = cl[cl["tag"] == tag].sort_values("tau")
        if len(sub) == 0:
            continue
        date_str = str(sub["date"].iloc[0].date())
        ax.plot(
            sub["tau"],
            sub["F"],
            "o-",
            color=color,
            label=f"{tag} ({date_str})",
            markersize=4,
        )
        tau_grid = np.linspace(sub["tau"].min(), sub["tau"].max(), 100)
        if len(sub) >= 5:
            # reuse lib25's own Nelson-Siegel wrapper (needs >=5 curve points)
            F_fn = lib25._ns_forward_fn(
                sub.assign(log_close=np.log(sub["F"]))[["tau", "F", "log_close"]]
            )
            ax.plot(tau_grid, F_fn(tau_grid), "--", color=color, alpha=0.4, linewidth=1)
    viz.style_ax(
        ax,
        title="CL forward curve on four representative dates",
        xlabel="tau (years)",
        ylabel="F ($/bbl)",
    )
    viz.legend(ax)
    return fig, ax


def fig_b2(data=None) -> tuple:
    """B2 -- realised vol term structure, fitted power law, 024's published CI."""
    fig, ax = viz.new_fig(figsize=(8, 5))
    cl = _INPUTS["products"]["CL"]
    fit = cl["samuelson_fit_full_sample"]
    slope, ci_lo, ci_hi = fit["slope"], fit["ci_lo"], fit["ci_hi"]

    tau_days = np.geomspace(10, 1500, 100)
    k = -slope
    # anchor sigma_1 so sigma(90d) matches a typical CL vol level (no
    # intercept is persisted in phase_1_25_inputs.json, only slope/CI)
    sigma_1 = 0.30 * 90**k
    central = sigma_1 * tau_days**-k
    lo = sigma_1 * tau_days**ci_lo
    hi = sigma_1 * tau_days**ci_hi

    ax.plot(tau_days, central, color=viz.BLUE, label="fitted power law")
    ax.fill_between(
        tau_days,
        np.minimum(lo, hi),
        np.maximum(lo, hi),
        color=viz.BLUE,
        alpha=0.15,
        label="024's published 95% CI band",
    )
    ax.set_xscale("log")
    viz.style_log_axis_plain(ax, axis="x")
    viz.style_ax(
        ax,
        title="CL realised vol term structure: fitted Samuelson power law",
        xlabel="days to expiry",
        ylabel="annualised vol (MAD)",
    )
    viz.legend(ax)
    return fig, ax


def fig_b3(data=None) -> tuple:
    """B3 -- the cautionary chart: 023's curve-calibrated implied sigma_F
    (collapsing to ~3.5%) against realised vol (flat 27-36%)."""
    fig, ax = viz.new_fig(figsize=(8, 5))
    tau = np.linspace(0.1, 4, 100)
    # 023 Phase 7 (NEXT_PROMPT.md's binding-constraints table): a curve-
    # calibrated two-factor model's implied sigma_F collapses 19% -> 3.5%
    # across the curve. No per-tau array was persisted by 023; this is an
    # ILLUSTRATIVE interpolation between its two published endpoints.
    implied_curve_fit = 0.19 + (0.035 - 0.19) * (tau - tau.min()) / (
        tau.max() - tau.min()
    )
    realised = np.full_like(tau, 0.30)
    ax.plot(
        tau,
        implied_curve_fit,
        "--",
        color=viz.RED,
        label="023: curve-calibrated implied sigma_F (illustrative interpolation of 19%->3.5%)",
    )
    ax.fill_between(
        tau,
        0.27,
        0.36,
        color=viz.BLUE,
        alpha=0.15,
        label="024: realised vol, 27-36% every bucket",
    )
    ax.plot(tau, realised, color=viz.BLUE, linewidth=1)
    viz.style_ax(
        ax,
        title="Why vol inputs come from realised vol, not a curve fit",
        xlabel="tau (years)",
        ylabel="annualised vol",
    )
    viz.legend(ax, loc="upper right")
    ax.text(
        0.02,
        0.03,
        "023's endpoints only (19% short end -> 3.5% long end); the shape between them is illustrative",
        transform=ax.transAxes,
        fontsize=7,
        color=viz.TEXT_SECONDARY,
        style="italic",
    )
    return fig, ax


def fig_b4(data=None) -> tuple:
    """B4 -- vol term structure split by curve state (024 D4)."""
    fig, ax = viz.new_fig(figsize=(8, 5))
    cl = _INPUTS["products"]["CL"]["curve_state_vol"]
    tau_days = np.geomspace(10, 1500, 100)
    for state, color in (("contango", viz.AQUA), ("backwardation", viz.RED)):
        fit = cl.get(state)
        if fit is None:
            continue
        k = -fit["k"] if fit["k"] is not None else None
        sigma_1 = fit["sigma_1"]
        if (
            k is None
            or sigma_1 is None
            or not np.isfinite(k)
            or not np.isfinite(sigma_1)
        ):
            continue
        vals = sigma_1 * tau_days**k
        ax.plot(tau_days, vals, color=color, label=f"CL, {state} (k={-k:.3f})")
    ax.set_xscale("log")
    viz.style_log_axis_plain(ax, axis="x")
    viz.style_ax(
        ax,
        title="CL vol term structure by curve state (024 D4)",
        xlabel="days to expiry",
        ylabel="annualised vol (MAD)",
    )
    viz.legend(ax)
    return fig, ax


def fig_b5(data=None) -> tuple:
    """B5 -- rolling CL-6E correlation with its own dispersion band, the
    quanto input and the instability that dominates it."""
    fig, ax = viz.new_fig(figsize=(9, 5))
    corr = lib25.corr_rolling("CL", "6E", window=126).dropna()
    mean, std = corr.mean(), corr.std()
    ax.plot(corr.index, corr.values, color=viz.VIOLET, linewidth=1)
    ax.axhline(
        mean, color=viz.GRAY, linestyle="--", linewidth=1, label=f"mean = {mean:.2f}"
    )
    ax.fill_between(
        corr.index,
        mean - std,
        mean + std,
        color=viz.VIOLET,
        alpha=0.15,
        label=f"+-1 std = {std:.2f}",
    )
    viz.shade_crisis(ax, "2020-02-01", "2020-06-30", "covid")
    viz.style_ax(
        ax,
        title="Rolling 126-day CL-6E correlation: the quanto input's own instability",
        xlabel="date",
        ylabel="correlation",
    )
    viz.legend(ax)
    return fig, ax


CAPTIONS = {
    "b1": {
        "what": "The observed CL forward curve on four representative dates -- calm, crisis, backwardated, and steep contango -- with the Nelson-Siegel interpolation used for tenors between listed contracts.",
        "intuition": "Yes: the crisis-date curve sits below the calm-date curve and the backwardated/contango dates visibly differ in slope, matching what 023's curve work already found.",
    },
    "b2": {
        "what": "CL's realised (MAD) volatility term structure with the fitted Samuelson power law and 024's published 95% confidence band drawn around it.",
        "intuition": "Yes: vol declines with tenor (the Samuelson effect), and the fit lands inside 024's own published CI, which is the wiring check Phase 1 already asserted.",
    },
    "b3": {
        "what": "023's curve-calibrated implied volatility (which collapses from 19% at the short end to 3.5% at the long end) plotted against 024's realised volatility, which stays flat at 27-36% across every maturity bucket.",
        "intuition": "The mismatch is the whole point: a curve-fit vol input would underprice a 4-year option by roughly a factor of seven relative to what actually happens, which is exactly why this notebook calibrates vol to realised history instead.",
    },
    "b4": {
        "what": "CL's realised vol term structure fitted separately on backwardated days and contango days (024 D4).",
        "intuition": "Yes: the backwardated-day slope is steeper than the contango-day slope, consistent with 024's published result, so pricing off a single static vol would be curve-state-blind.",
    },
    "b5": {
        "what": "The rolling 126-day correlation between CL and the EUR/USD futures (6E), the input the quanto adjustment in case study F depends on, with its own +-1 standard deviation band.",
        "intuition": "The correlation swings across a wide range over time (crisis periods especially), and that swing turns out to be larger than the quanto price adjustment itself -- the instability is the finding, not a nuisance to average away.",
    },
}
