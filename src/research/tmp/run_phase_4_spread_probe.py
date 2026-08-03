"""Phase 4: one cheap empirical probe (NEXT_PROMPT.md sec 4, Phase 4).

Per the pre-registration, Phase 4 runs ONLY if Phase 3's shortlist surfaces a
candidate that is (i) Tier 1/2-supported, (ii) testable with data already in
this repo, and (iii) cheap. The shortlist (see phase_3_shortlist_results.json)
identifies exactly one candidate meeting all three: structural mean-reversion
in the pre-built commodity spread series (Gate SP), evidenced by Tier 1
peer-reviewed literature (Gatev-Goetzmann-Rouwenhorst 2006; Zhu 2024) and
directly addressing hypothesis (e) - this repo has 30 pre-built spread series
(`data/market/spreads/*.parquet`) that notebook 8 explicitly declared out of
scope (Strategy E) and that have never been used for anything beyond one
descriptive plot (notebook 8 Fig. 33).

This is EXPLICITLY NOT a gated backtest. No cost model, no Sharpe, no gate
verdict is computed here - only a first-look sanity check of whether the
mechanism (mean reversion) is even present in this repo's own data, in the
spirit of notebook 3's Phase 4 IC screen. Anything that looks promising gets a
properly pre-registered gate in notebook 10 (see NEXT_PROMPT.md written by
this notebook), not a retrofitted one here.

Method (deliberately simple, not a formal unit-root test with tabulated
critical values - statsmodels is not a repo dependency, and a full ADF isn't
warranted for a screening pass):
  - drop roll-window-flagged rows (this repo's own documented discipline,
    notebook 8's "mandatory roll-window-exclusion" note on this exact data)
  - AR(1)-in-differences regression: delta_v_t = alpha + beta * v_{t-1} + eps
    (OLS via numpy lstsq); beta < 0 with a large |t-stat| is evidence of mean
    reversion; implied half-life = -ln(2) / ln(1 + beta) when beta < 0
  - a simple cross-sectional-style IC: Spearman corr(z-score_t, 5-day-forward
    spread change_t), where z-score_t = (v_t - rolling_mean_60) / rolling_std_60
    - a negative correlation is the mean-reversion sign (high z -> spread
    should fall going forward)

Writes phase_4_spread_probe_results.json.
"""

import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_lib9 import ols_ar1_diff, zscore_ic

DATA_DIR = "src/research/data/market/spreads"
OUT_PATH = "src/research/tmp/phase_4_spread_probe_results.json"

SPREADS = [
    "crack_321",  # refining margin, 3 legs (CL/HO/RB) collapsed to one series
    "gold_silver",  # cross-metal ratio spread
    "brent_wti",  # cross-crude spread
    "corn_wheat",  # cross-grain spread
    "platinum_palladium",  # cross-metal spread
    "crush_soy",  # soy processing margin
]


def main():
    results = {}
    for name in SPREADS:
        df = pl.read_parquet(f"{DATA_DIR}/{name}.parquet").sort("date")
        if "roll_window_flag" in df.columns:
            n_before = df.height
            df = df.filter(~pl.col("roll_window_flag").fill_null(False))
            n_dropped = n_before - df.height
        else:
            n_dropped = 0
        df = df.filter(pl.col("value").is_finite())
        v = df["value"].to_numpy().astype(float)
        dates = df["date"].to_numpy()
        if len(v) < 100:
            results[name] = {"error": f"insufficient data ({len(v)} rows)"}
            continue
        ar1 = ols_ar1_diff(v)
        ic = zscore_ic(v)
        results[name] = {
            "n_rows": len(v),
            "n_roll_window_rows_dropped": int(n_dropped),
            "date_range": [str(dates.min()), str(dates.max())],
            "ar1_mean_reversion": ar1,
            "zscore_5d_forward_ic": ic,
        }
        print(
            f"{name}: beta={ar1['beta']:.5f} t={ar1['t_stat_beta']:.2f} "
            f"half_life={ar1['half_life_days']} IC={ic['ic']} p={ic['p_value']}"
        )

    n_mean_reverting = sum(
        1
        for r in results.values()
        if "ar1_mean_reversion" in r and r["ar1_mean_reversion"]["mean_reverting"]
    )
    n_negative_ic_significant = sum(
        1
        for r in results.values()
        if "zscore_5d_forward_ic" in r
        and r["zscore_5d_forward_ic"]["ic"] is not None
        and r["zscore_5d_forward_ic"]["ic"] < 0
        and r["zscore_5d_forward_ic"]["p_value"] is not None
        and r["zscore_5d_forward_ic"]["p_value"] < 0.05
    )
    summary = {
        "n_spreads_tested": len(SPREADS),
        "n_mean_reverting_ar1": n_mean_reverting,
        "n_negative_significant_ic": n_negative_ic_significant,
        "note": (
            "This is a first-look descriptive screen, NOT a gated backtest: no "
            "cost model, no position sizing, no Sharpe ratio, no gate verdict. "
            "It only asks whether the mean-reversion MECHANISM Gatev-Goetzmann-"
            "Rouwenhorst and Zhu 2024 describe is even present, directionally, "
            "in this repo's own commodity spread data. A positive result here "
            "means 'worth a properly pre-registered gate in notebook 10', not "
            "'a strategy exists'."
        ),
    }
    with open(OUT_PATH, "w") as f:
        json.dump({"per_spread": results, "summary": summary}, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nwritten {OUT_PATH}")


if __name__ == "__main__":
    main()
