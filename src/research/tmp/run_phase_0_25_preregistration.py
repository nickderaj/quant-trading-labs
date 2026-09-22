"""Phase 0 -- pre-registration for notebook 025 (pricing bespoke commodity
derivatives). Written before any price is computed.

This is a TEACHING notebook, not a "does this model beat that model" study.
There are no option prices anywhere in this repo, so every premium the
notebook shows is a *model* price computed from realised volatility, never a
market quote. See NEXT_PROMPT.md section 0 for the full argument.
"""

import json
from pathlib import Path

TMP = Path(__file__).resolve().parent

HONESTY_LIST = [
    (
        "No market option prices exist anywhere in this repo. Every premium shown "
        "is a model price computed from realised volatility, never a market quote."
    ),
    (
        "Realised volatility is a downward-biased proxy for implied volatility "
        "(the variance risk premium). Options generally trade above realised vol, "
        "so every premium in this notebook is probably an underestimate of what a "
        "dealer would actually charge."
    ),
    (
        "No bid-offer, credit, funding, or margin cost is modelled anywhere in "
        "this notebook. The gap between a model price and a client quote is "
        "illustrated once, in Part E, as a labelled waterfall -- never elsewhere."
    ),
    (
        "Realised-payoff figures are arithmetic, not backtests: the payoff of a "
        "defined structure against recorded settlement prices, over the full "
        "sample. There is no trading rule, no Sharpe ratio, and no P&L claim "
        "anywhere in this notebook. If a script computes a Sharpe ratio, that is "
        "a bug, not a result."
    ),
]

PREREG = {
    "written": "2026-09-22",
    "study_type": "pedagogical",
    "framing": (
        "A teaching notebook answering: if someone handed you a bespoke "
        "commodity derivative tomorrow and asked what it is worth and how to "
        "hedge it, what would you actually do? Not a claim that any model "
        "'beats' another, or that these are tradeable prices."
    ),
    "no_holdout_spent": True,
    "no_trading_gate": True,
    "no_sharpe_no_backtest": True,
    "honesty_list": HONESTY_LIST,
    "binding_constraints_from_023_024": {
        "no_gabillon_curve": (
            "023 Phase 9: Gabillon lost the held-out-maturity gate to "
            "Nelson-Siegel by +24.9 $/bbl, CI [+22.0, +28.0]. The forward "
            "curve here uses the observed curve directly plus Nelson-Siegel "
            "interpolation (lib23.bench_nelson_siegel), never Gabillon."
        ),
        "vol_from_realised_not_curve_fit": (
            "023 Phase 7: a curve-calibrated two-factor model's implied "
            "sigma_F collapses 19% -> 3.5% across the curve, while realised "
            "vol is 27-36% in every maturity bucket. Vol inputs come from "
            "realised vol (lib24.mad_vol) exclusively -- never from a curve "
            "fit. Shown as a cautionary teaching figure (B3)."
        ),
        "samuelson_slope_ci": (
            "024 Phase 3: CL Samuelson slope -0.174 [-0.197, -0.150]; NG "
            "-0.284; power law in dte. Phase 1 must print the fitted k "
            "against these published values and assert it lands inside a "
            "wide tolerance band -- a miss is a wiring bug, not a discovery."
        ),
        "mad_vol_not_std": (
            "024 section 2: plain std claims WTI 4-year vol is 230%; MAD says "
            "15%. lib24.mad_vol and lib24.load_panel's hygiene are reused "
            "verbatim, never reimplemented."
        ),
        "fat_tails_everywhere": (
            "024 D1/D2: every product fat-tailed, front end worst; energy "
            "front-end negative skew. Motivates the Bachelier/displaced-"
            "diffusion pricers and the historical bootstrap in Part C."
        ),
        "vol_depends_on_curve_state": (
            "024 D4: CL backwardation slope -0.218 vs contango -0.149. Shown "
            "as a sensitivity (B4), never silently averaged over."
        ),
        "negative_price_event": {
            "ticker": "CL202005",
            "date": "2020-04-20",
            "close": -2.67,
            "volume": 102083,
            "note": (
                "Genuine history. Black-76 assigns it probability zero -- the "
                "lognormal density has no support below 0. Case study E."
            ),
        },
    },
    "phase_2_crossval_tolerances": {
        "european_call_put": {
            "methods": ["black76", "binomial_n_large", "pde_crank_nicolson", "mc"],
            "tolerance": "mc within 3 SE; tree and PDE within 0.1% of closed form",
        },
        "european_negative_capable": {
            "methods": ["bachelier", "mc_arithmetic"],
            "tolerance": "3 standard errors",
        },
        "displaced_diffusion_limits": {
            "methods": [
                "displaced_black -> black76 (shift->0)",
                "displaced_black -> bachelier (shift->large)",
            ],
            "tolerance": "0.1%",
        },
        "barrier_all_four_types": {
            "methods": [
                "reiner_rubinstein_analytic",
                "pde_barrier_on_grid_node",
                "mc_bgk_continuity",
            ],
            "tolerance": "0.5% analytic vs PDE; MC within 3 SE",
        },
        "barrier_in_out_parity": {
            "methods": ["knock_in + knock_out == vanilla"],
            "tolerance": "1e-10",
        },
        "geometric_asian": {
            "methods": ["closed_form", "mc"],
            "tolerance": "3 SE",
        },
        "arithmetic_asian": {
            "methods": ["turnbull_wakeman", "mc_geometric_control_variate"],
            "tolerance": "1%",
        },
        "asian_ordering": {
            "methods": ["arithmetic >= geometric", "asian <= european same strike"],
            "tolerance": "strict",
        },
        "american": {
            "methods": ["binomial", "pde", "longstaff_schwartz_with_dual_upper_bound"],
            "tolerance": "LSM inside [binomial - 3 SE, dual upper bound]",
        },
        "american_ordering": {
            "methods": ["american >= european >= intrinsic"],
            "tolerance": "strict",
        },
        "spread_option": {
            "methods": ["margrabe_at_k0", "kirk", "bivariate_mc"],
            "tolerance": "Margrabe vs MC 3 SE; Kirk allowed to deviate near zero spread/high correlation -- recorded, not a failure",
        },
        "quanto": {
            "methods": ["analytic_drift_adjustment", "mc_correlated_fx"],
            "tolerance": "3 SE",
        },
        "put_call_parity": {
            "methods": ["every European pricer"],
            "tolerance": "1e-8",
        },
    },
    "case_studies_fixed_in_advance": {
        "A_refiner": {
            "exposure": "long crude, short products -- the 3:2:1 crack spread margin, not the oil price",
            "structures": [
                "sell the crack forward as a swap",
                "buy a crack put (spread option, priced with Kirk and bivariate MC)",
                "a crack collar",
            ],
        },
        "B_wheat_farmer": {
            "exposure": "short the harvest price on ZW (cross-check KE); wants a floor, balks at the premium",
            "structures": [
                "a plain put",
                "a zero-cost collar (solved call strike)",
                "a knock-out put",
            ],
        },
        "C_airline_diesel_buyer": {
            "exposure": "buys physical diesel at a monthly average price on HO",
            "structures": [
                "a strip of futures",
                "an Asian (average-price) call",
                "an Asian collar",
            ],
        },
        "D_gas_utility": {
            "exposure": "winter NG demand, seasonal vol (024 Phase 5)",
            "structures": [
                "a winter strip of futures",
                "a strip of caps (calls)",
                "a cheaper cap with a knock-out on a warm-winter price collapse",
            ],
        },
        "E_producer_negative_prices": {
            "exposure": "a $10 put on CL202005 priced as of early April 2020",
            "structures": ["black76", "bachelier", "displaced_diffusion"],
        },
        "F_european_buyer": {
            "exposure": "long CL exposure, reports in EUR",
            "structures": [
                "unhedged FX",
                "a separate FX forward",
                "a quanto (fixed-FX) call",
                "a composite (floating-FX) call",
            ],
        },
        "G_structured_note": {
            "exposure": "a one-year oil-linked autocallable reverse convertible on CL",
            "structures": [
                "decomposed into legs, priced and summed against the note's nominal"
            ],
        },
    },
    "explicitly_out_of_scope": [
        (
            "Swing/optionality on delivery volume (needs a multi-exercise "
            "stochastic-control solver; would double the notebook)."
        ),
    ],
}

if __name__ == "__main__":
    out = TMP / "phase_0_25_preregistration.json"
    out.write_text(json.dumps(PREREG, indent=2, default=str))
    print(f"wrote {out}")
