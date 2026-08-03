"""10a Phase 1: term-structure regime atlas, all 16 products (NEXT_PROMPT.md
sec 5 Phase 1). Descriptive only -- no strategy verdicts, no Sharpe, no gate
(sec 1 rule 1). Development window only (2010-06-06 to 2024-12-31, ES from
2018-01-01, KE from 2013-12-16, matching notebook 8's own convention) --
the 2025-01-01 -> 2026-07-28 holdout is never read here, per sec 8's "holdout
stays frozen" and this notebook's own house rule that even a purely
descriptive atlas has no legitimate reason to touch it.

Writes phase_1_10a_results.json.
"""

import json
import sys

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import commod_lib8 as C
import numpy as np
import polars as pl
import spread_lib10 as S

CURVE_DIR = "src/research/tmp/phase_0_curves"
OUT_PATH = "src/research/tmp/phase_1_10a_results.json"

DEV_START = {"__default__": "2010-06-06", "ES": "2018-01-01", "KE": "2013-12-16"}
DEV_END = "2024-12-31"


def load_curve(product: str) -> pl.DataFrame:
    curve = pl.read_parquet(f"{CURVE_DIR}/{product}.parquet")
    dev_start = DEV_START.get(product, DEV_START["__default__"])
    sub = curve.filter(
        (pl.col("date") >= pl.lit(dev_start).str.to_date())
        & (pl.col("date") <= pl.lit(DEV_END).str.to_date())
    ).sort("date")
    return sub


def main():
    per_product: dict = {}
    by_sector: dict[str, list[dict]] = {}

    for product in C.PRODUCTS:
        curve = load_curve(product)
        if curve.height < 100:
            per_product[product] = {"error": "insufficient history"}
            continue

        ts = C.term_structure_state(
            curve.select(["date", "close_f1", "dte_f1", "close_f2", "dte_f2"])
        )
        curve = curve.join(ts, on="date", how="left")

        state = curve["term_structure_state"]
        n_total = state.len()
        n_backward = int((state == "backwardation").sum())
        n_contango = int((state == "contango").sum())
        n_null = int(state.is_null().sum())

        persistence = S.regime_state_persistence(state)

        slope = curve["roll_slope_annualized"].drop_nulls().to_numpy()
        slope_stats = {
            "mean": float(np.mean(slope)) if len(slope) else None,
            "median": float(np.median(slope)) if len(slope) else None,
            "std": float(np.std(slope)) if len(slope) else None,
            "p10": float(np.percentile(slope, 10)) if len(slope) else None,
            "p90": float(np.percentile(slope, 90)) if len(slope) else None,
        }

        dates = curve["date"].to_list()
        months = np.array([d.month for d in dates])
        month_regime: dict[int, dict[str, float]] = {}
        state_arr = state.to_list()
        for m in range(1, 13):
            sel = [
                state_arr[i]
                for i in range(len(state_arr))
                if months[i] == m and state_arr[i] is not None
            ]
            if len(sel) < 5:
                continue
            month_regime[m] = {
                "frac_backwardation": float(
                    np.mean([s == "backwardation" for s in sel])
                ),
                "n": len(sel),
            }

        entry = {
            "sector": C.SECTOR[product],
            "n_obs": n_total,
            "n_backwardation": n_backward,
            "n_contango": n_contango,
            "n_null_flat_slope": n_null,
            "frac_backwardation": n_backward / n_total if n_total else None,
            "frac_contango": n_contango / n_total if n_total else None,
            "roll_slope_annualized_stats": slope_stats,
            "persistence": persistence,
            "month_of_year_regime": month_regime,
            "date_range": [str(dates[0]), str(dates[-1])],
        }
        per_product[product] = entry
        by_sector.setdefault(C.SECTOR[product], []).append(
            {"product": product, "frac_backwardation": entry["frac_backwardation"]}
        )

    sector_summary = {
        sector: {
            "mean_frac_backwardation": float(
                np.mean(
                    [
                        p["frac_backwardation"]
                        for p in members
                        if p["frac_backwardation"] is not None
                    ]
                )
            ),
            "products": members,
        }
        for sector, members in by_sector.items()
    }

    # Term-structure curve snapshots: a deep-contango and a deep-backwardation
    # day for a representative energy product (CL) and a representative
    # grain (ZC), for sec 7's "curve snapshot" chart -- picked as the date
    # with the most extreme roll_slope_annualized in each direction.
    snapshots = {}
    for product in ["CL", "NG", "ZC"]:
        curve = load_curve(product)
        ts = C.term_structure_state(
            curve.select(["date", "close_f1", "dte_f1", "close_f2", "dte_f2"])
        )
        curve = curve.join(ts, on="date", how="left").drop_nulls(
            subset=["roll_slope_annualized"]
        )
        if curve.height == 0:
            continue
        deep_contango_row = (
            curve.sort("roll_slope_annualized", descending=True).head(1).to_dicts()[0]
        )
        deep_backward_row = (
            curve.sort("roll_slope_annualized", descending=False).head(1).to_dicts()[0]
        )
        snapshots[product] = {
            "deep_contango": {
                "date": str(deep_contango_row["date"]),
                "close_f1": deep_contango_row["close_f1"],
                "dte_f1": deep_contango_row["dte_f1"],
                "close_f2": deep_contango_row["close_f2"],
                "dte_f2": deep_contango_row["dte_f2"],
                "close_f3": deep_contango_row.get("close_f3"),
                "dte_f3": deep_contango_row.get("dte_f3"),
                "roll_slope_annualized": deep_contango_row["roll_slope_annualized"],
            },
            "deep_backwardation": {
                "date": str(deep_backward_row["date"]),
                "close_f1": deep_backward_row["close_f1"],
                "dte_f1": deep_backward_row["dte_f1"],
                "close_f2": deep_backward_row["close_f2"],
                "dte_f2": deep_backward_row["dte_f2"],
                "close_f3": deep_backward_row.get("close_f3"),
                "dte_f3": deep_backward_row.get("dte_f3"),
                "roll_slope_annualized": deep_backward_row["roll_slope_annualized"],
            },
        }

    results = {
        "per_product": per_product,
        "sector_summary": sector_summary,
        "curve_snapshots": snapshots,
        "_dev_window": {"start": DEV_START, "end": DEV_END},
        "_note": "descriptive only -- no strategy verdicts, no Sharpe, no gate (NEXT_PROMPT.md sec 1 rule 1). Holdout (2025-01-01+) not read.",
    }
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"written {OUT_PATH}")


if __name__ == "__main__":
    main()
