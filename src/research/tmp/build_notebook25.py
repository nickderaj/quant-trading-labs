"""Notebook 025 builder -- a narrative over the phase JSONs and chart
modules. Loads results and renders them; does not re-run any pricing.

Usage (from repo root): uv run python src/research/tmp/build_notebook25.py
"""

from __future__ import annotations

import json as _json
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TMP = REPO_ROOT / "src" / "research" / "tmp"
OUT_PATH = (
    REPO_ROOT / "src" / "research" / "025_pricing_bespoke_commodity_derivatives.ipynb"
)

sys.path.insert(0, str(TMP))
import charts_25_a as ca
import charts_25_b as cb
import charts_25_c as cc
import charts_25_d as cd
import charts_25_e as ce

phase5 = _json.loads((TMP / "phase_5_25_cases_abcd.json").read_text())
phase6 = _json.loads((TMP / "phase_6_25_cases_efg.json").read_text())


def cid() -> str:
    return uuid.uuid4().hex[:8]


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cid(),
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cid(),
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def explain(module, fig_id: str) -> dict:
    entry = module.CAPTIONS[fig_id]
    return md(
        f"**What this shows:** {entry['what']}\n\n**Does it match intuition?** {entry['intuition']}"
    )


cells: list[dict] = []

# --------------------------------------------------------------------------- #
# Title cell -- the four honesty caveats, up front and unmissable
# --------------------------------------------------------------------------- #

cells.append(
    md("""\
# Notebook 025 -- Pricing bespoke commodity derivatives, and a lesson

**The question this notebook answers:** if someone handed you a bespoke commodity
derivative tomorrow and asked "what is this worth, and how should our client hedge?",
what would you actually do?

It is a **teaching notebook**, not a "model A beats model B" study, structured in five
parts: **A** the primer (what each instrument *is*), **B** the inputs (where the
numbers come from), **C** the methods (how to actually price each instrument, eight
ways, cross-checked against each other), **D** seven worked hedging case studies, and
**E** honest limits.

### Four things to hold in mind on every page

1. **There are no option prices in this repo. None.** Every premium shown below is a
   *model* price computed from *realised* volatility, never a market quote.
2. **Realised vol is a biased proxy for implied vol.** Options generally trade *above*
   realised volatility (the variance risk premium), so every premium here is probably
   an **underestimate** of what a dealer would actually charge.
3. **No bid-offer, credit, funding, or margin cost is modelled anywhere here.** The gap
   between a model price and a client quote is illustrated once, in Part E, as a
   labelled waterfall -- never elsewhere.
4. **No Sharpe ratio. No returns backtest. No trading rule.** Where a case study shows
   "what would have happened," it is a *realised payoff computation* -- the payoff of a
   defined structure against recorded settlement prices -- never a strategy backtest.

None of the pricing mathematics is new: Black-76 is 1976, Reiner-Rubinstein 1991,
Longstaff-Schwartz 2001. The contribution is the *teaching*: one consistent harness,
real commodity data, cross-validated pricers, and worked client problems.

Full spec: `NEXT_PROMPT.md`. Full narrative and numbers:
`src/results/025_pricing_bespoke_commodity_derivatives.md`.
""")
)

cells.append(
    code("""\
import json

TMP = "tmp"


def load(name):
    with open(f"{TMP}/{name}") as f:
        return json.load(f)


prereg = load("phase_0_25_preregistration.json")
for line in prereg["honesty_list"]:
    print("-", line)
""")
)

cells.append(
    md(
        "## Phase 0 -- pre-registration\n\n"
        "A pedagogical study: no holdout spent, no trading gate. The Phase 2 "
        "cross-validation tolerances and the seven case studies' candidate structures "
        "were fixed here, before any price was computed."
    )
)
cells.append(code('prereg["case_studies_fixed_in_advance"]'))

cells.append(
    md(
        "## Phase 1 -- market inputs\n\n"
        "The forward curve (Nelson-Siegel interpolation, never Gabillon -- 023's "
        "gate), and the realised vol term structure, checked against 024's published "
        "Samuelson slopes before anything downstream runs."
    )
)
cells.append(
    code("""\
phase1 = load("phase_1_25_inputs.json")
for p, d in phase1["products"].items():
    fit = d["samuelson_fit_full_sample"]
    print(f"{p}: slope={fit['slope']:.3f}  within widened band of 024 = {d['within_widened_band_of_024']}")
""")
)

cells.append(
    code("""\
import sys

sys.path.insert(0, "tmp")
import matplotlib.pyplot as plt
import viz25 as vz

plt.rcParams["figure.facecolor"] = vz.SURFACE
plt.rcParams["savefig.facecolor"] = vz.SURFACE

# This notebook executes with cwd = src/research/ (its own directory), but
# lib24/lib25's data-directory constants are repo-root-relative strings --
# point them at the repo root explicitly so every chart module's direct
# lib24.load_panel / lib25.fx_series / lib25.discount_curve calls resolve.
import pathlib

import lib24
import lib25

_repo_root = pathlib.Path.cwd().parent.parent
lib24.DATA_DIR = str(_repo_root / "src/research/data/market/databento")
lib25.DATA_DIR = str(_repo_root / "src/research/data/market/databento")
lib25.FRED_DIR = str(_repo_root / "src/research/data/market/fred")
lib25.FX_DIR = str(_repo_root / "src/research/data/market/yfinance/daily")

phase2 = load("phase_2_25_crossval.json")
phase3 = load("phase_3_25_primer.json")
phase4 = load("phase_4_25_methods.json")
phase5 = load("phase_5_25_cases_abcd.json")
phase6 = load("phase_6_25_cases_efg.json")
""")
)

# --------------------------------------------------------------------------- #
# Part A -- the primer (11 figures)
# --------------------------------------------------------------------------- #

cells.append(
    md(
        "## Part A -- the primer\n\n"
        "What each instrument *is*, taught through pictures: payoff diagrams, real "
        "price paths with barriers drawn on them, premium-vs-strike curves."
    )
)
for i in range(1, 12):
    fig_id = f"a{i}"
    cells.append(md(f"### A{i}"))
    cells.append(
        code(f"""\
import charts_25_a as ca

fig, ax = ca.fig_{fig_id}(phase3)
plt.show()
""")
    )
    cells.append(explain(ca, fig_id))

# --------------------------------------------------------------------------- #
# Part B -- the inputs (5 figures)
# --------------------------------------------------------------------------- #

cells.append(
    md(
        "## Part B -- the inputs\n\n"
        "Where the numbers come from: the forward curve, the volatility term "
        "structure, rates, FX. Grounded in this repo's own 023/024 results."
    )
)
for i in range(1, 6):
    fig_id = f"b{i}"
    cells.append(md(f"### B{i}"))
    cells.append(
        code(f"""\
import charts_25_b as cb

fig, ax = cb.fig_{fig_id}()
plt.show()
""")
    )
    cells.append(explain(cb, fig_id))

# --------------------------------------------------------------------------- #
# Part C -- the methods (8 figures)
# --------------------------------------------------------------------------- #

cells.append(
    md(
        "## Part C -- the methods\n\n"
        "*How you actually price it* -- eight methods, from closed form to Monte "
        "Carlo to Markov functional, each cross-checked against the others. "
        "**This is the only pass/fail phase in the notebook** -- the proof the "
        "pricers are sound."
    )
)
cells.append(
    code("""\
for c in phase2["matrix"]:
    status = "PASS" if c["passed"] else "FAIL"
    print(f"{status}  {c['name']}  ({c['tolerance']})")
print()
print("ALL PASSED:", phase2["all_passed"])
""")
)
for i in range(1, 9):
    fig_id = f"c{i}"
    cells.append(md(f"### C{i}"))
    cells.append(
        code(f"""\
import charts_25_c as cc

fig, ax = cc.fig_{fig_id}(None)
plt.show()
""")
    )
    cells.append(explain(cc, fig_id))

# --------------------------------------------------------------------------- #
# Part D -- the case studies (9 figures)
# --------------------------------------------------------------------------- #

cells.append(
    md(
        "## Part D -- the case studies\n\n"
        "Seven real hedging problems, each worked end to end through the same "
        "five-step template: **1.** the exposure, **2.** candidate structures, "
        "**3.** the prices (with the model-price caveat), **4.** what actually "
        "happened over history (a realised-payoff distribution, not a backtest), "
        "**5.** the trade-off.\n\n"
        "One thing to read carefully in step 4: every realised payoff below is "
        "the payoff of the **hedge leg on its own**, not of the hedged position. "
        "A collar that shows a large negative mean has not lost the client "
        "money -- its short option is giving back gains on a physical "
        "position that is not drawn here, which is the whole point of a "
        "hedge. Compare the structures against each other, not against zero."
    )
)

# All seven case studies run through the same five-step template the Part D
# header promises. Steps 4 and 5 are written per case, because "what actually
# happened" and "the trade-off" are the two steps that cannot be generated
# from the structure list alone -- and an earlier version of this notebook
# left E, F and G as a placeholder line pointing at the write-up instead.
_CASE_NARRATIVE = {
    "A_refiner": {
        "letter": "A",
        "title": "the refiner",
        "history": (
            "Over {n} historical windows the crack spread moved against the "
            "refiner about as often as for it: selling it forward returned "
            "{fwd_mean:+.2f} $/bbl on average, the put {put_mean:+.2f}, the "
            "collar {col_mean:+.2f}. Every structure is re-struck at each "
            "window's own crack level, so this compares structures, not "
            "vintages."
        ),
        "tradeoff": (
            "Hedging the margin directly and hedging inputs and outputs "
            "separately are not the same trade. Kirk and a bivariate Monte "
            "Carlo agree on the crack put to within {gap:.1f}%, and that "
            "agreement is the point: the spread option prices the "
            "correlation between crude and products (rho about {rho_ho:.2f} "
            "for CL-HO, {rho_rb:.2f} for CL-RB), which two separate hedges "
            "cannot see at any price."
        ),
    },
    "B_farmer": {
        "letter": "B",
        "title": "the wheat farmer",
        "history": (
            "Over {n} historical ~4-month harvest windows the barrier was "
            "touched {ko:.1%} of the time on ZW ({ko_ke:.1%} on KE). But "
            "only {costly:.1%} of windows were knock-outs that cost anything "
            "-- on the other {n_free} the plain put would have expired "
            "worthless regardless. Averaged over the costly ones the farmer "
            "gave up {shortfall:.0f} cents/bushel of protection, or about "
            "${shortfall_usd:.2f}/bu. ZW is quoted in cents."
        ),
        "tradeoff": (
            "The knock-out put costs {ko_prem:.0f} cents against the plain "
            "put's {plain_prem:.0f}, a saving of {saving:.0f}%. The question "
            "is not whether {ko:.1%} is a large number in the abstract, but "
            "whether saving {saving:.0f}% of the premium is worth roughly a "
            "one-in-four chance of losing the floor in exactly the year the "
            "price is collapsing -- which is the year the farmer bought it "
            "for. The zero-cost collar avoids the premium entirely by "
            "selling the upside above its call strike."
        ),
    },
    "C_airline": {
        "letter": "C",
        "title": "the airline / diesel buyer",
        "history": (
            "Over {n} historical windows the long futures strip returned "
            "{fwd_mean:+.3f} $/gal on average and the Asian cap "
            "{cap_mean:+.3f}. The airline buys diesel, so its hedge is long "
            "the commodity: it pays when diesel rallies, which is what "
            "offsets the higher physical bill."
        ),
        "tradeoff": (
            "Averaging is what makes this cheap. The realised vol of the "
            "monthly average is {ratio:.0%} of the vol of a single "
            "settlement date, and the Asian call prices {asian_vs_euro_abs:.0f}% "
            "below the equivalent European as a result. The textbook "
            "sigma/sqrt(3) shortcut misprices it by {pitfall:+.1f}% against "
            "the schedule-correct calculation -- small here, but it is a "
            "shortcut with no reason to be small in general."
        ),
    },
    "D_gas_utility": {
        "letter": "D",
        "title": "the gas utility",
        "history": (
            "Over {n} historical winter windows the long strip returned "
            "{fwd_mean:+.3f} $/MMBtu on average and the cap {cap_mean:+.3f}. "
            "The {ko_pct:.0%} knock-out barrier was touched in {ko_freq:.0%} "
            "of those windows, monitored along the path rather than tested "
            "once at expiry."
        ),
        "tradeoff": (
            "Quoting the whole winter strip off a single front-month vol "
            "overprices it by {flat_gap:.1f}% against a maturity-aware "
            "curve. Conditioning that curve on winter *delivery* -- genuine "
            "seasonality, the thing this case study is nominally about -- "
            "moves it by {sea_gap:+.1f}%. NG's winter vol premium is real at "
            "the front of the curve but is not measurable at the 6-12 month "
            "tenors this strip spans, so the Samuelson effect is doing "
            "essentially all the work. The knock-out saves "
            "{ko_saving:.1f}% of the premium at a barrier that can actually "
            "be touched."
        ),
    },
}


def _fmt(template: str, **kw) -> str:
    return template.format(**kw)


for _key, _spec in _CASE_NARRATIVE.items():
    case = phase5[_key]
    letter = _spec["letter"]
    payoffs = case["realised_payoffs"]
    t = case["trade_off_numbers"]
    lines = [
        f"### Case {letter} -- {_spec['title']}\n",
        f"**1. The exposure.** {case['exposure'][0].upper() + case['exposure'][1:]}.\n",
        (
            "**2. Candidate structures, and 3. what they cost.** Model prices "
            "from realised vol, never market quotes:\n"
        ),
    ]
    for st in case["structures"]:
        name = st.get("name", "structure")
        prem = st.get("premium")
        prem_str = f"{prem:.3f}" if isinstance(prem, int | float) else "n/a"
        lines.append(
            f"- **{name}** ({st.get('description', '')}): premium = {prem_str}"
        )

    names = list(payoffs)
    if _key == "A_refiner":
        hist = _fmt(
            _spec["history"],
            n=payoffs[names[0]]["n"],
            fwd_mean=payoffs[names[0]]["mean"],
            put_mean=payoffs[names[1]]["mean"],
            col_mean=payoffs[names[2]]["mean"],
        )
        trade = _fmt(
            _spec["tradeoff"],
            gap=t["kirk_vs_mc_gap_pct"],
            rho_ho=t["rho_crhb"],
            rho_rb=t["rho_crrb"],
        )
    elif _key == "B_farmer":
        plain_prem = case["structures"][0]["premium"]
        ko_prem = case["structures"][2]["premium"]
        hist = _fmt(
            _spec["history"],
            n=t["n_observations"],
            ko=t["ko_knockout_frequency_zw"],
            ko_ke=t["ko_knockout_frequency_ke"],
            costly=t["costly_knockout_frequency_zw"],
            n_free=t["n_knockouts"] - t["n_costly_knockouts"],
            shortfall=t["mean_shortfall_when_costly_cents"],
            shortfall_usd=t["mean_shortfall_when_costly_cents"] / 100,
        )
        trade = _fmt(
            _spec["tradeoff"],
            ko_prem=ko_prem,
            plain_prem=plain_prem,
            saving=100 * (1 - ko_prem / plain_prem),
            ko=t["ko_knockout_frequency_zw"],
        )
    elif _key == "C_airline":
        hist = _fmt(
            _spec["history"],
            n=payoffs[names[0]]["n"],
            fwd_mean=payoffs[names[0]]["mean"],
            cap_mean=payoffs[names[1]]["mean"],
        )
        trade = _fmt(
            _spec["tradeoff"],
            ratio=t["vol_reduction_ratio"],
            asian_vs_euro_abs=abs(t["asian_vs_european_premium_pct"]),
            pitfall=t["wrong_vs_correct_pitfall_pct"],
        )
    else:
        hist = _fmt(
            _spec["history"],
            n=payoffs[names[0]]["n"],
            fwd_mean=payoffs[names[0]]["mean"],
            cap_mean=payoffs[names[1]]["mean"],
            ko_pct=t["ko_headline_barrier_pct"],
            ko_freq=t["ko_realised_hit_frequency"],
        )
        trade = _fmt(
            _spec["tradeoff"],
            flat_gap=t["flat_overprices_vs_maturity_pct"],
            sea_gap=t["seasonal_vs_maturity_pct"],
            ko_saving=t["ko_savings_pct"],
        )

    lines.append(f"\n**4. What actually happened.** {hist}\n")
    lines.append(f"**5. The trade-off.** {trade}")
    cells.append(md("\n".join(lines)))


_E = phase6["E_negative_prices"]
_F = phase6["F_european_buyer"]
_G = phase6["G_structured_note"]
_kup = _E["rolling_var_kupiec_pof"]

cells.append(
    md(
        "### Case E -- the producer, and negative prices\n\n"
        f"**1. The exposure.** A ${_E['strike']:.0f} put on {_E['ticker']}, priced as "
        f"of {_E['valuation_date']}, two weeks before the contract settled at "
        f"{_E['displacement_analysis']['actual_settlement']}.\n\n"
        "**2-3. The structures and their prices.** The same put under three "
        "measures:\n\n"
        f"- **Black-76 (lognormal)**: {_E['models']['black76']['put_price']:.3g}\n"
        f"- **Bachelier (normal)**: {_E['models']['bachelier']['put_price']:.3g}\n"
        f"- **Displaced diffusion** (shift {_E['models']['displaced_black']['shift']:.0f}): "
        f"{_E['models']['displaced_black']['put_price']:.3g}\n\n"
        "**4. What actually happened.** The contract settled at "
        f"{_E['displacement_analysis']['actual_settlement']}. Black-76 assigns that "
        "outcome a log density of `-inf` -- not a small probability, *zero* "
        "probability, since a lognormal has no support below zero. Bachelier "
        f"scores {_E['models']['bachelier']['log_density_at_negative_2_67']:.1f} and "
        "the displaced model "
        f"{_E['models']['displaced_black']['log_density_at_negative_2_67']:.1f}: finite, "
        "computable, survivable. Systematically, over the full 16-year sample, "
        "rolling 1% left-tail Kupiec coverage rejects **both** families, in "
        "opposite directions: the lognormal-return model never breaches its own "
        f"1% VaR at all ({_kup['lognormal_returns']['n_exceed']} exceedances in "
        f"{_kup['lognormal_returns']['n']} days, p={_kup['lognormal_returns']['pvalue']:.1e}), "
        "so its left tail is far too *wide*; the level-change model breaches on "
        f"{_kup['bachelier_level_changes']['observed_rate']:.2%} of days against a 1% "
        f"target (p={_kup['bachelier_level_changes']['pvalue']:.1e}), roughly twice "
        "too often, so its tail is too *narrow*.\n\n"
        "**5. The trade-off.** Failing a coverage test by being too cautious and "
        "failing it by being too aggressive are different diagnoses with "
        "different fixes. Neither model is calibrated, but only one of them "
        "declared the thing that actually happened impossible."
    )
)

_q = _F["rho_sensitivity_quanto"]
cells.append(
    md(
        "### Case F -- the European buyer\n\n"
        "**1. The exposure.** Long CL, reporting in EUR: two random variables, "
        "the oil price and the exchange rate, and a choice about which one to "
        "fix.\n\n"
        "**2-3. The structures and their prices.**\n\n"
        f"- **Unadjusted vanilla call**: {_F['structures']['quanto_call']['unadjusted_vanilla_call']:.3f}\n"
        f"- **Quanto (fixed-FX) call**: {_F['structures']['quanto_call']['price']:.3f} "
        f"(adjustment {_F['structures']['quanto_call']['quanto_adjustment']:+.3f})\n"
        f"- **Composite (floating-FX) call**: {_F['structures']['composite_call']['price']:.3f}\n\n"
        "**4. What actually happened.** The CL-6E correlation this adjustment "
        f"depends on has a rolling mean of {_F['fx_rolling_corr_with_CL']['mean']:.2f} "
        f"and a standard deviation of {_F['fx_rolling_corr_with_CL']['std']:.2f} over "
        "the sample -- it is not a constant, and over a 126-day window it ranges "
        f"from {_q['rho_lo']:.2f} to {_q['rho_hi']:.2f}.\n\n"
        "**5. The trade-off.** Repricing the quanto across that +-1 sigma range "
        f"moves it by {_q['price_range']:.3f}, about "
        f"{_q['price_range'] / abs(_F['structures']['quanto_call']['quanto_adjustment']):.0f} "
        "times the size of the adjustment itself. The correction is real, but "
        "the input it depends on is less stable than the correction is large -- "
        "which is worth knowing before quoting the quanto as the precise answer."
    )
)

_hist = _G["historical_resampling"]
_mp = _G["model_pricing"]
cells.append(
    md(
        "### Case G -- the structured note (capstone)\n\n"
        "**1. The exposure.** A one-year oil-linked autocallable reverse "
        f"convertible on CL: {_G['structure']['coupon']:.0%} coupon, "
        f"{_G['structure']['autocall_level']:.0%} autocall trigger, "
        f"{_G['structure']['barrier']:.0%} principal barrier, observed quarterly.\n\n"
        "**2-3. The structure and its price.** Decomposed into legs and priced "
        f"by Monte Carlo ({_mp['note_fair_value']:.4f} per 1.00 of face, standard "
        f"error {_mp['price_se']:.4f}), implying an issuer margin of "
        f"{_mp['implied_issuer_margin']:.2%}.\n\n"
        "**4. What actually happened.** Over "
        f"{_hist['n_windows']} historical one-year windows the note autocalled "
        f"{_hist['realized_autocall_frequency']:.1%} of the time, against the "
        f"model's {_hist['model_expected_autocall_frequency']:.1%}. Observation by "
        "observation the two line up closely (model "
        + ", ".join(f"{x:.1%}" for x in _mp["autocall_frequency_by_obs"])
        + "; historical "
        + ", ".join(f"{x:.1%}" for x in _hist["realized_autocall_frequency_by_obs"])
        + "), with the whole gap at the first observation -- as expected, since "
        "the model prices under a driftless risk-neutral measure while CL's "
        "2010-2026 history carries its actual drift.\n\n"
        "**5. The trade-off.** The modelled issuer margin "
        f"({_mp['implied_issuer_margin']:.2%}) is smaller than the Monte Carlo "
        f"standard error around it (+-{_mp['price_se']:.4f}), so this notebook "
        "**cannot** distinguish a real margin from simulation noise here. That "
        "is the honest reading: the decomposition works, the path count does "
        "not support a claim about issuer economics."
    )
)

for i in range(1, 10):
    fig_id = f"d{i}"
    cells.append(md(f"#### D{i}"))
    cells.append(
        code(f"""\
import charts_25_d as cd

fig, ax = cd.fig_{fig_id}(None)
plt.show()
""")
    )
    cells.append(explain(cd, fig_id))

# --------------------------------------------------------------------------- #
# Part E -- limits (2 figures)
# --------------------------------------------------------------------------- #

cells.append(
    md(
        "## Part E -- what a real desk adds that this notebook does not\n\n"
        "Honest limits: the variance risk premium this notebook cannot see, and "
        "everything between a model price and a client quote. Also explicitly out "
        "of scope: swing/optionality on delivery volume (needs a multi-exercise "
        "stochastic-control solver and would double this notebook)."
    )
)
for i in range(1, 3):
    fig_id = f"e{i}"
    cells.append(md(f"### E{i}"))
    cells.append(
        code(f"""\
import charts_25_e as ce

fig, ax = ce.fig_{fig_id}()
plt.show()
""")
    )
    cells.append(explain(ce, fig_id))

# --------------------------------------------------------------------------- #

cells.append(
    md(
        "## Bottom line\n\n"
        "See `src/results/025_pricing_bespoke_commodity_derivatives.md` for the full "
        "write-up, the method-vs-method agreement table, and what to test next."
    )
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT_PATH.write_text(_json.dumps(nb, indent=1))
print(f"wrote {OUT_PATH}")
