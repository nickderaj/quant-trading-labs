"""10a Phase 2: spread taxonomy and structure, all 30 pre-built spreads
(NEXT_PROMPT.md sec 5 Phase 2, sec 4.2, sec 4.3). Descriptive only -- no
Sharpe, no cost model, no gate verdict (sec 1 rule 1).

For each spread: taxonomy classification (sec 4.2, from leg_roles' own
product list, not a hardcoded name split), descriptive stats, rolling leg
correlation, the sec 4.3 cointegration/stationarity precondition (ADF on the
spread's own `value` series -- already the Engle-Granger residual, see
spread_lib10.adf_test's docstring), and the AR(1)/z-score-IC mean-reversion
probe extended from notebook 9's 6-spread first look to all 30.

**Sec 4.3 include/exclude decision, made here in 10a, before any 10b
backtest runs:** spreads failing the ADF test at the 5% level (t_stat does
not clear -2.86) are EXCLUDED from Gate SP/SPR's 10b backtest universe --
trading an uncointegrated pair as mean-reverting has no legitimate
statistical basis (docs/09's own stated precondition on calendar spreads,
generalised here to every spread) -- but every excluded spread's full
descriptive record (AR1, IC, ADF, taxonomy) is still reported in this
notebook, never silently dropped from the write-up.

Development window only, matching notebook 8/10a-Phase-1's own convention:
spread series are filtered to <= 2024-12-31 before any statistic is
computed; the 2025-01-01+ holdout is never read here.

Writes phase_2_10a_results.json.
"""

import json
import sys

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import numpy as np
import polars as pl
import research_lib9 as R9
import spread_lib10 as S

SPREAD_DIR = "src/research/data/market/spreads"
OUT_PATH = "src/research/tmp/phase_2_10a_results.json"
DEV_END = "2024-12-31"

SPREAD_NAMES = [
    "bean_corn", "brent_calendar", "brent_wti", "bz_cal_m1m3", "cl_cal_m2m3",
    "corn_wheat", "crack_321", "crush_soy", "es_calendar", "gasheat_rbho",
    "gasoline_crack", "gc_cal_m1m2", "gc_cal_m2m3", "gold_silver",
    "heating_oil_crack", "ho_cal_m1m2", "ho_cal_m2m3", "kc_chicago_wheat",
    "ke_cal_m1m2", "ng_cal_m1m2", "ng_cal_m2m3", "ng_calendar",
    "platinum_palladium", "rb_cal_m1m2", "rb_cal_m2m3", "wti_calendar",
    "zc_cal_m1m2", "zl_cal_m1m2", "zm_cal_m1m2", "zw_cal_m1m2",
]

ADF_MIN_ROWS = 60


def main():
    per_spread: dict = {}
    for name in SPREAD_NAMES:
        df = pl.read_parquet(f"{SPREAD_DIR}/{name}.parquet")
        df = df.filter(pl.col("date") <= pl.lit(DEV_END).str.to_date()).sort("date")
        leg_products = [r["product"] for r in df["leg_roles"][0]]
        taxonomy = S.classify_spread_taxonomy(leg_products)

        n_total = df.height
        n_roll_flagged = int(df["roll_window_flag"].sum())
        clean = df.filter(~pl.col("roll_window_flag"))

        value_clean = clean["value"].to_numpy()

        desc = {
            "mean": float(np.nanmean(value_clean)), "std": float(np.nanstd(value_clean)),
            "min": float(np.nanmin(value_clean)), "max": float(np.nanmax(value_clean)),
        }

        ar1 = R9.ols_ar1_diff(value_clean[np.isfinite(value_clean)]) if len(value_clean) >= 30 else None
        ic = R9.zscore_ic(value_clean) if len(value_clean) >= 90 else None
        adf = S.adf_test(value_clean) if len(value_clean) >= ADF_MIN_ROWS else None
        include_10b = bool(adf is not None and adf.get("stationary_5pct") is True)

        leg1 = clean["leg1_price"].to_numpy()
        leg2 = clean["leg2_price"].to_numpy()
        r1 = np.diff(np.log(leg1), prepend=np.nan)
        r2 = np.diff(np.log(leg2), prepend=np.nan)
        leg_corr_full = float(np.corrcoef(r1[1:], r2[1:])[0, 1]) if len(r1) > 30 else None
        rolling_corr = S.rolling_leg_correlation(r1, r2, window=60)
        roll_corr_finite = rolling_corr[np.isfinite(rolling_corr)]
        rolling_corr_summary = {
            "mean": float(np.mean(roll_corr_finite)) if len(roll_corr_finite) else None,
            "min": float(np.min(roll_corr_finite)) if len(roll_corr_finite) else None,
            "p10": float(np.percentile(roll_corr_finite, 10)) if len(roll_corr_finite) else None,
        }

        dates_clean = clean["date"].to_list()
        rolling_corr_series = {
            str(dates_clean[i]): float(rolling_corr[i])
            for i in range(len(dates_clean))
            if np.isfinite(rolling_corr[i]) and i % 5 == 0  # thinned to every 5th day for JSON/plot size
        }

        per_spread[name] = {
            "taxonomy": taxonomy,
            "leg_products": leg_products,
            "n_legs": len(leg_products),
            "date_range": [str(df["date"].min()), str(df["date"].max())],
            "n_rows": n_total,
            "n_roll_window_flagged": n_roll_flagged,
            "frac_roll_window_flagged": n_roll_flagged / n_total if n_total else None,
            "descriptive": desc,
            "leg_return_correlation_full_sample": leg_corr_full,
            "rolling_leg_correlation_60d": rolling_corr_summary,
            "rolling_leg_correlation_60d_series": rolling_corr_series,
            "ar1_mean_reversion": ar1,
            "zscore_5d_forward_ic": ic,
            "adf_cointegration": adf,
            "include_in_10b": include_10b,
        }

    n_inter = sum(1 for v in per_spread.values() if v["taxonomy"] == "inter_commodity")
    n_cal = sum(1 for v in per_spread.values() if v["taxonomy"] == "calendar")
    n_pass_adf = sum(1 for v in per_spread.values() if v["include_in_10b"])
    n_ar1_mr = sum(1 for v in per_spread.values() if v["ar1_mean_reversion"] and v["ar1_mean_reversion"]["mean_reverting"])
    n_ic_sig = sum(
        1 for v in per_spread.values()
        if v["zscore_5d_forward_ic"] and v["zscore_5d_forward_ic"]["ic"] is not None
        and v["zscore_5d_forward_ic"]["p_value"] < 0.05 and v["zscore_5d_forward_ic"]["ic"] < 0
    )

    disagreements = [
        name for name, v in per_spread.items()
        if v["ar1_mean_reversion"] and v["ar1_mean_reversion"]["mean_reverting"]
        and not (v["zscore_5d_forward_ic"] and v["zscore_5d_forward_ic"]["ic"] is not None
                 and v["zscore_5d_forward_ic"]["p_value"] < 0.05 and v["zscore_5d_forward_ic"]["ic"] < 0)
    ]

    results = {
        "per_spread": per_spread,
        "summary": {
            "n_spreads": len(SPREAD_NAMES),
            "n_inter_commodity": n_inter,
            "n_calendar": n_cal,
            "n_pass_adf_5pct": n_pass_adf,
            "n_ar1_mean_reverting": n_ar1_mr,
            "n_ic_significant_negative": n_ic_sig,
            "ar1_vs_ic_disagreements": disagreements,
        },
        "_sec_4_3_decision": (
            "Spreads failing the ADF cointegration/stationarity test at the 5% level "
            "(stationary_5pct == False) are EXCLUDED from Gate SP/SPR's 10b backtest "
            "universe. Made here in 10a, before any 10b backtest runs, per sec 4.3. "
            "Excluded spreads' full descriptive record remains reported above, never "
            "silently dropped."
        ),
        "_dev_window_end": DEV_END,
        "_note": "descriptive only -- no strategy verdicts, no Sharpe, no gate (NEXT_PROMPT.md sec 1 rule 1).",
    }
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"written {OUT_PATH}")
    print(f"inter_commodity={n_inter} calendar={n_cal} pass_adf={n_pass_adf}/{len(SPREAD_NAMES)}")
    print(f"ar1_mr={n_ar1_mr} ic_sig_neg={n_ic_sig} disagreements={disagreements}")


if __name__ == "__main__":
    main()
