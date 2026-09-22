"""Part E charts (2 figures) for notebook 025 -- honest limits: what this
notebook cannot see (the variance risk premium) and what a real desk adds
on top of a model price (bid-offer, credit, funding, margin).

Written by the main session, not delegated (NEXT_PROMPT.md section 7: "Part
E ... you write these"). Both figures are explicitly illustrative -- there
are no option quotes in this repo to measure either gap directly, and each
figure says so in its own caption.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_TMP_DIR = Path(__file__).resolve().parent
if str(_TMP_DIR) not in sys.path:
    sys.path.insert(0, str(_TMP_DIR))

import lib25
import viz25 as viz


def fig_e1(data=None) -> tuple:
    """E1 -- realised vs a plausible implied level: the variance risk
    premium this notebook cannot see, and the resulting understatement in
    dollar premium terms."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=110)
    fig.set_facecolor(viz.SURFACE)

    tau_days = np.geomspace(10, 1500, 100)
    realised = 0.30 * (tau_days / 90) ** -0.174
    # illustrative variance risk premium: options trade above realised vol,
    # typically by a roughly constant multiplicative markup across tenors --
    # no vanilla quotes exist in this repo to measure the true premium, so
    # this 15% markup is a plausible, clearly-labelled illustration only.
    plausible_implied = realised * 1.15

    ax1.plot(
        tau_days,
        realised,
        color=viz.BLUE,
        label="realised vol (what this notebook uses)",
    )
    ax1.plot(
        tau_days,
        plausible_implied,
        "--",
        color=viz.RED,
        label="plausible implied vol (illustrative, +15%)",
    )
    ax1.set_xscale("log")
    viz.style_log_axis_plain(ax1, axis="x")
    viz.style_ax(
        ax1,
        title="The variance risk premium this notebook cannot see",
        xlabel="days to expiry",
        ylabel="annualised vol",
    )
    viz.legend(ax1)

    F, K, T, df = 70.0, 70.0, 1.0, 0.97
    sigma_r = 0.30
    sigma_i = 0.30 * 1.15
    price_realised = lib25.black76(F, K, T, sigma_r, df, "call")
    price_implied = lib25.black76(F, K, T, sigma_i, df, "call")
    understatement_pct = (price_implied - price_realised) / price_implied * 100

    ax2.bar(
        ["priced off\nrealised vol", "priced off\nplausible implied"],
        [price_realised, price_implied],
        color=[viz.BLUE, viz.RED],
    )
    ax2.text(
        0.5,
        max(price_realised, price_implied) * 1.05,
        f"-{understatement_pct:.0f}%",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color=viz.TEXT_PRIMARY,
    )
    viz.style_ax(
        ax2,
        title="ATM 1y call, illustrative",
        ylabel="premium ($/bbl)",
    )
    viz.price_caption(ax2)
    fig.suptitle(
        "Every premium in this notebook is probably an underestimate of a dealer's quote",
        fontsize=10,
        color=viz.TEXT_SECONDARY,
        x=0.02,
        ha="left",
    )
    return fig, (ax1, ax2)


def fig_e2(data=None) -> tuple:
    """E2 -- what a desk adds: a waterfall from model price to the quote a
    client actually gets. Illustrative, and labelled as illustrative."""
    fig, ax = viz.new_fig(figsize=(9, 5))

    labels = [
        "model\nprice",
        "+ bid-offer",
        "+ credit/\nfunding",
        "+ margin",
        "client\nquote",
    ]
    steps = [0, 0.08, 0.05, 0.07, 0]  # illustrative fractional add-ons
    base = 8.99  # a representative ATM premium from this notebook, $/bbl

    values = [base]
    for s in steps[1:-1]:
        values.append(values[-1] * (1 + s))
    values.append(values[-1])

    colors = [viz.BLUE, viz.ORANGE, viz.ORANGE, viz.ORANGE, viz.RED]

    x = np.arange(len(labels))
    # simple waterfall: bar from 0 to each cumulative value's top, coloured
    # by stage, with a connecting step line
    ax.bar(x[0], values[0], color=colors[0], width=0.6)
    for i in range(1, len(labels) - 1):
        ax.bar(
            x[i],
            values[i] - values[i - 1],
            bottom=values[i - 1],
            color=colors[i],
            width=0.6,
        )
    ax.bar(x[-1], values[-1], color=colors[-1], width=0.6)

    for i, v in enumerate(values):
        ax.text(
            i, v + 0.15, f"${v:.2f}", ha="center", fontsize=9, color=viz.TEXT_PRIMARY
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    viz.style_ax(
        ax,
        title="What a desk adds on top of a model price (illustrative)",
        ylabel="premium ($/bbl)",
    )
    ax.text(
        0.02,
        0.95,
        "Illustrative waterfall -- no bid-offer, credit, funding or margin data exists in this repo; "
        "the stage sizes are representative, not measured.",
        transform=ax.transAxes,
        fontsize=7.5,
        color=viz.TEXT_SECONDARY,
        style="italic",
        va="top",
    )
    return fig, ax


CAPTIONS = {
    "e1": {
        "what": "This notebook's realised-vol input against a plausible implied-vol level (illustrative, since no option quotes exist in this repo to measure the true variance risk premium), and the resulting understatement in an ATM 1-year call's premium.",
        "intuition": "Options generally trade above realised volatility, so this notebook's model prices are probably conservative -- an honest limit stated once, clearly, rather than left implicit.",
    },
    "e2": {
        "what": "An illustrative waterfall from this notebook's model price to a representative client quote, naming the stages (bid-offer, credit/funding, margin) a real desk adds that this notebook does not model anywhere.",
        "intuition": "None of the stage sizes are measured from data in this repo -- the figure exists to name what is missing, not to quantify it precisely, and it says so on its own face.",
    },
}
