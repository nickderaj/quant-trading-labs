"""10a Phase 3: regime-conditional structure -- the heart of 10a
(NEXT_PROMPT.md sec 5 Phase 3). Descriptive only -- no Sharpe, no cost model,
no gate verdict (sec 1 rule 1). Inter-commodity spreads only, per sec 4.2
("the regime hypothesis is meaningful here... gating a calendar spread on
contango/backwardation is close to conditioning the signal on its own sign").
Calendar spreads get one explicitly-labelled circularity diagnostic instead
(never used as evidence for the regime hypothesis, per sec 4.2's requirement).

**Primary regime-leg declaration (sec 4.1, made here, in advance, applied
identically to every spread -- not cherry-picked per spread):** the regime
that conditions spread `X` is `leg1`'s own term structure, i.e. the first leg
listed in the spread's own `leg_roles` metadata (for brent_wti: BZ). This is
a fixed, mechanical rule (leg ordering is a property of the pre-built
dataset, not chosen after looking at which leg "worked"), not a per-spread
optimisation. A `both_legs_agree` variant is computed ONLY for brent_wti
(sec 4.1's own named operator case) as an additional, explicitly-labelled
robustness configuration -- not run for every spread, to keep the
configuration count bounded and because sec 4.1 names brent_wti specifically
for this check.

Three regime definitions (sec 4.1, capped at 3, every one entering the DSR
count once a gate is built on it): (i) raw sign (`term_structure_state`),
(ii) sign with a deadband, (iii) a persistence requirement. This phase
originally declared (i) raw sign primary, decided before this phase's own
descriptive numbers were computed so that no later-looking-favourable
definition could retroactively become "primary" (the exact selection-after-
looking trap sec 4.1 warns against). All three definitions are still
computed and reported below regardless, so nothing here was data-snooped --
but Phase 5's pre-registration (run_phase_5_10a_preregistration.py)
**supersedes** this module's own PRIMARY_REGIME_DEFINITION constant for
Gate SPR's actual headline: raw sign is defined on ~100% of trading days (an
exactly-zero annualised slope is vanishingly rare), so gating on it barely
differs from the unconditional book and cannot test the operator's
"definite state, not regime-blind" claim. Phase 5 promotes (ii) deadband to
primary instead, on that structural (pre-registration-time, not
result-driven) basis -- see Phase 5's own REGIME_DEFINITIONS rationale.

Writes phase_3_10a_results.json.
"""

import json
import sys

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import commod_lib8 as C
import numpy as np
import polars as pl
import spread_lib10 as S

SPREAD_DIR = "src/research/data/market/spreads"
CURVE_DIR = "src/research/tmp/phase_0_curves"
TAXONOMY_PATH = "src/research/tmp/phase_2_10a_results.json"
OUT_PATH = "src/research/tmp/phase_3_10a_results.json"
DEV_END = "2024-12-31"

PRIMARY_REGIME_DEFINITION = "raw_sign"  # declared in advance -- see module docstring


def load_regime_frame(product: str) -> pl.DataFrame:
    curve = pl.read_parquet(f"{CURVE_DIR}/{product}.parquet")
    curve = curve.filter(pl.col("date") <= pl.lit(DEV_END).str.to_date()).sort("date")
    ts = C.term_structure_state(curve.select(["date", "close_f1", "dte_f1", "close_f2", "dte_f2"]))
    ts = ts.with_columns(S.regime_deadband(ts["roll_slope_annualized"]).alias("state_deadband"))
    ts = ts.with_columns(S.regime_persistent(ts["term_structure_state"]).alias("state_persistent"))
    return ts


def analyze_spread(name: str, taxonomy: dict) -> dict:
    df = pl.read_parquet(f"{SPREAD_DIR}/{name}.parquet")
    df = df.filter(pl.col("date") <= pl.lit(DEV_END).str.to_date()).sort("date")
    clean = df.filter(~pl.col("roll_window_flag"))

    leg1_product = taxonomy["leg_products"][0]
    regime = load_regime_frame(leg1_product)
    merged = clean.join(regime, on="date", how="left")

    leg1 = merged["leg1_price"].to_numpy()
    leg2 = merged["leg2_price"].to_numpy()
    r1 = np.diff(np.log(leg1), prepend=np.nan)
    r2 = np.diff(np.log(leg2), prepend=np.nan)
    spread_ret = np.diff(merged["value"].to_numpy(), prepend=np.nan)

    out: dict = {"primary_regime_leg": leg1_product}

    for def_name, col in [
        ("raw_sign", "term_structure_state"),
        ("deadband", "state_deadband"),
        ("persistent", "state_persistent"),
    ]:
        labels = merged[col].to_list()
        ar1_by_regime = S.regime_conditional_ar1(merged["value"].to_numpy(), labels)
        vol_by_regime = S.regime_conditional_vol(spread_ret, labels)

        leg_corr_by_regime: dict = {}
        labels_arr = np.asarray(labels, dtype=object)
        for state in sorted({s for s in labels if s is not None}):
            mask = (labels_arr == state) & np.isfinite(r1) & np.isfinite(r2)
            n = int(mask.sum())
            leg_corr_by_regime[state] = {
                "n": n,
                "corr": float(np.corrcoef(r1[mask], r2[mask])[0, 1]) if n >= 30 else None,
            }

        out[def_name] = {
            "ar1_by_regime": ar1_by_regime,
            "vol_by_regime": vol_by_regime,
            "leg_corr_by_regime": leg_corr_by_regime,
        }

    return out


def main():
    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)["per_spread"]
    inter_commodity = [name for name, v in taxonomy.items() if v["taxonomy"] == "inter_commodity"]
    calendar = [name for name, v in taxonomy.items() if v["taxonomy"] == "calendar"]

    per_spread: dict = {}
    for name in inter_commodity:
        per_spread[name] = analyze_spread(name, taxonomy[name])

    # brent_wti "both legs agree" robustness variant (sec 4.1's named case).
    bz_regime = load_regime_frame("BZ").rename(
        {c: f"bz_{c}" for c in ["roll_slope_annualized", "term_structure_state", "state_deadband", "state_persistent"]}
    )
    cl_regime = load_regime_frame("CL").rename(
        {c: f"cl_{c}" for c in ["roll_slope_annualized", "term_structure_state", "state_deadband", "state_persistent"]}
    )
    bw = pl.read_parquet(f"{SPREAD_DIR}/brent_wti.parquet")
    bw = bw.filter(pl.col("date") <= pl.lit(DEV_END).str.to_date()).sort("date")
    bw_clean = bw.filter(~pl.col("roll_window_flag"))
    merged = bw_clean.join(bz_regime, on="date", how="left").join(cl_regime, on="date", how="left")
    both_agree = (
        pl.when(pl.col("bz_term_structure_state") == pl.col("cl_term_structure_state"))
        .then(pl.col("bz_term_structure_state"))
        .otherwise(pl.lit("disagree"))
    )
    merged = merged.with_columns(both_agree.alias("state_both_agree"))
    ar1_both_agree = S.regime_conditional_ar1(merged["value"].to_numpy(), merged["state_both_agree"].to_list())
    n_agree = int((merged["state_both_agree"] != "disagree").sum())
    brent_wti_both_agree = {
        "n_days_total": merged.height,
        "n_days_agree": n_agree,
        "frac_days_agree": n_agree / merged.height if merged.height else None,
        "ar1_by_state": ar1_both_agree,
    }

    # Calendar spreads: circularity diagnostic only (sec 4.2), never used as
    # regime-hypothesis evidence. Same machinery, explicitly labelled.
    calendar_diagnostic: dict = {}
    for name in calendar:
        calendar_diagnostic[name] = analyze_spread(name, taxonomy[name])

    # Cross-spread summary for the plain-English regime-hypothesis verdict:
    # for each inter-commodity spread, does the primary (raw_sign) regime
    # definition show mean reversion (|t|>2, beta<0) STRONGER in one state
    # than the other (comparing |beta| between contango and backwardation)?
    cross_spread_summary = []
    for name in inter_commodity:
        raw = per_spread[name]["raw_sign"]["ar1_by_regime"]
        cb = raw.get("contango", {}).get("fit")
        bw_fit = raw.get("backwardation", {}).get("fit")
        pooled = raw.get("_pooled", {}).get("fit")
        stronger_in = None
        if cb and bw_fit and cb.get("beta") is not None and bw_fit.get("beta") is not None:
            stronger_in = "backwardation" if abs(bw_fit["beta"]) > abs(cb["beta"]) else "contango"
        cross_spread_summary.append({
            "spread": name,
            "contango_beta": cb.get("beta") if cb else None,
            "contango_mean_reverting": cb.get("mean_reverting") if cb else None,
            "backwardation_beta": bw_fit.get("beta") if bw_fit else None,
            "backwardation_mean_reverting": bw_fit.get("mean_reverting") if bw_fit else None,
            "pooled_mean_reverting": pooled.get("mean_reverting") if pooled else None,
            "mean_reversion_stronger_in": stronger_in,
        })
    n_stronger_backwardation = sum(1 for r in cross_spread_summary if r["mean_reversion_stronger_in"] == "backwardation")

    results = {
        "primary_regime_definition": PRIMARY_REGIME_DEFINITION,
        "per_spread_inter_commodity": per_spread,
        "brent_wti_both_legs_agree_variant": brent_wti_both_agree,
        "calendar_spread_circularity_diagnostic": calendar_diagnostic,
        "cross_spread_summary": cross_spread_summary,
        "n_inter_commodity_spreads_stronger_reversion_in_backwardation": n_stronger_backwardation,
        "n_inter_commodity_spreads_total": len(inter_commodity),
        "_note": (
            "descriptive only -- no strategy verdicts, no Sharpe, no gate "
            "(NEXT_PROMPT.md sec 1 rule 1). Calendar-spread block is a labelled "
            "circularity diagnostic (sec 4.2), never evidence for the regime "
            "hypothesis."
        ),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"written {OUT_PATH}")
    print(f"stronger reversion in backwardation: {n_stronger_backwardation}/{len(inter_commodity)}")


if __name__ == "__main__":
    main()
