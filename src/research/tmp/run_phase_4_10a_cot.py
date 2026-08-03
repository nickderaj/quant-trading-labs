"""10a Phase 4: inventory/positioning context, CL only (NEXT_PROMPT.md sec 5
Phase 4, "optional, if cheap"). Descriptive only -- no Sharpe, no gate.

This repo's cache holds exactly one CFTC COT series (`067651` = light sweet
crude, NYMEX -- docs/09-market-data-and-microstructure.md's own
"Hedging pressure and the COT report" pitfall), so this is a single-product
check, never extrapolated into a panel claim: does CL's net non-commercial
positioning corroborate its own term-structure regime label (inventory
theory is the mechanism behind both)?

Writes phase_4_10a_results.json.
"""

import json
import sys

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import commod_lib8 as C
import numpy as np
import polars as pl
import spread_lib10 as S

COT_PATH = "src/research/data/market/cot/067651.parquet"
CURVE_PATH = "src/research/tmp/phase_0_curves/CL.parquet"
OUT_PATH = "src/research/tmp/phase_4_10a_results.json"
DEV_START = "2010-06-06"
DEV_END = "2024-12-31"


def main():
    cot_raw = pl.read_parquet(COT_PATH)
    cot = S.cot_net_noncomm_fraction(cot_raw)
    cot = cot.filter(
        (pl.col("public_date") >= pl.lit(DEV_START).str.to_date())
        & (pl.col("public_date") <= pl.lit(DEV_END).str.to_date())
    ).sort("public_date")

    curve = pl.read_parquet(CURVE_PATH)
    curve = curve.filter(
        (pl.col("date") >= pl.lit(DEV_START).str.to_date())
        & (pl.col("date") <= pl.lit(DEV_END).str.to_date())
    ).sort("date")
    ts = C.term_structure_state(
        curve.select(["date", "close_f1", "dte_f1", "close_f2", "dte_f2"])
    )
    curve = curve.join(ts, on="date", how="left")

    # as-of join: each daily curve row picks up the most recent COT report
    # already public on or before that date (no lookahead -- public_date is
    # already lagged past the CFTC's own Friday release, sec `cot_net_noncomm_fraction`).
    joined = curve.sort("date").join_asof(
        cot.sort("public_date"),
        left_on="date",
        right_on="public_date",
        strategy="backward",
    )
    joined = joined.drop_nulls(subset=["net_noncomm_frac", "term_structure_state"])

    by_regime: dict = {}
    for state in ["backwardation", "contango"]:
        sel = joined.filter(pl.col("term_structure_state") == state)[
            "net_noncomm_frac"
        ].to_numpy()
        by_regime[state] = {
            "n": len(sel),
            "mean_net_noncomm_frac": float(np.mean(sel)) if len(sel) >= 20 else None,
            "std": float(np.std(sel)) if len(sel) >= 20 else None,
        }

    backward_vals = joined.filter(pl.col("term_structure_state") == "backwardation")[
        "net_noncomm_frac"
    ].to_numpy()
    contango_vals = joined.filter(pl.col("term_structure_state") == "contango")[
        "net_noncomm_frac"
    ].to_numpy()
    from scipy import stats as st

    t_stat, p_value = st.ttest_ind(backward_vals, contango_vals, equal_var=False)

    corr = float(
        np.corrcoef(
            joined["roll_slope_annualized"].to_numpy(),
            joined["net_noncomm_frac"].to_numpy(),
        )[0, 1]
    )

    inventory_theory_corroborated = bool(
        by_regime["backwardation"]["mean_net_noncomm_frac"] is not None
        and by_regime["contango"]["mean_net_noncomm_frac"] is not None
        and by_regime["backwardation"]["mean_net_noncomm_frac"]
        > by_regime["contango"]["mean_net_noncomm_frac"]
        and p_value < 0.05
    )

    results = {
        "product": "CL",
        "cftc_code": "067651",
        "coverage_note": (
            "This repo's data/market/cot/ directory holds exactly one CFTC series "
            "(CL light sweet crude, NYMEX). A cross-product positioning panel is not "
            "possible with this repo's own data -- CL-only, never extrapolated, per "
            "docs/09's own documented pitfall."
        ),
        "n_days_joined": joined.height,
        "net_noncomm_frac_by_regime": by_regime,
        "welch_ttest_backwardation_vs_contango": {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
        },
        "corr_roll_slope_vs_net_noncomm_frac": corr,
        "inventory_theory_corroborated": inventory_theory_corroborated,
        "interpretation": (
            "Net speculative (non-commercial) length is predicted, under Keynes' normal-"
            "backwardation / inventory theory, to run higher when the market is "
            "backwardated (low inventory, hedgers paying speculators to hold long risk) "
            "than when contangoed. A positive corr(roll_slope, net_noncomm_frac) would be "
            "the OPPOSITE of this prediction (recall: roll_slope < 0 IS backwardation), "
            "so the theory-consistent sign is corr < 0."
        ),
        "_note": "descriptive only -- no strategy verdicts, no Sharpe, no gate (NEXT_PROMPT.md sec 1 rule 1). Dev window only.",
    }
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"written {OUT_PATH}")
    print(
        f"backwardation mean_net_noncomm_frac={by_regime['backwardation']['mean_net_noncomm_frac']:.4f} "
        f"contango={by_regime['contango']['mean_net_noncomm_frac']:.4f} p={p_value:.4g} corr={corr:.3f}"
    )


if __name__ == "__main__":
    main()
