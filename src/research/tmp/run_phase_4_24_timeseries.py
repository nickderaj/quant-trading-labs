"""Phase 4 -- the time dimension (Section B).

rolling_maturity_vol with a 63-day causal window, per product, for buckets
(0,60),(60,180),(180,365),(365,730),(730,3650). Derives the front/deferred
ratio time series, crisis-window vs calm-window means, rolling 126-day
correlation between front and deferred daily returns for CL and NG, and
per-calendar-year vol_by_bucket tables for CL and NG.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib24

TMP = Path(__file__).resolve().parent

BUCKETS = [(0, 60), (60, 180), (180, 365), (365, 730), (730, 3650)]


def crisis_vs_calm(rolling: pd.DataFrame) -> dict:
    out = {}
    for name, (start, end) in lib24.CRISIS_WINDOWS.items():
        mask = (rolling["date"] >= start) & (rolling["date"] <= end)
        window = rolling[mask]
        if len(window) == 0:
            continue
        out[name] = {
            "front_mean_0_60": float(window["vol_0_60"].mean()),
            "deferred_mean_365_730": float(window["vol_365_730"].mean()),
            "ratio_mean": float(
                (window["vol_0_60"] / window["vol_365_730"])
                .replace([np.inf, -np.inf], np.nan)
                .mean()
            ),
        }
    return out


def rolling_correlation(panel: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    front = panel[panel["dte"] < 60][["date", "r"]].groupby("date")["r"].mean()
    deferred_df = panel[(panel["dte"] >= 365) & (panel["dte"] < 730)][["date", "r"]]
    deferred = deferred_df.groupby("date")["r"].mean()
    both = pd.DataFrame({"front": front, "deferred": deferred}).dropna()
    both = both.sort_index()
    corr = both["front"].rolling(window, min_periods=window // 2).corr(both["deferred"])
    return pd.DataFrame({"date": corr.index, "rolling_corr_126d": corr.to_numpy()})


def main() -> None:
    full_panel = pd.read_parquet(TMP / "phase_1_24_returns.parquet")
    report: dict = {"products": {}}

    for product in lib24.PRODUCTS:
        panel = full_panel[full_panel["product"] == product]
        usable = lib24.usable_returns(panel, max_gap=1, min_volume=0)
        if len(usable) < 200:
            continue

        rolling = lib24.rolling_maturity_vol(usable, window=63, buckets=BUCKETS)
        rolling["date"] = pd.to_datetime(rolling["date"])
        rolling["ratio_front_deferred"] = (
            rolling["vol_0_60"] / rolling["vol_365_730"]
        ).replace([np.inf, -np.inf], np.nan)

        rolling_path = TMP / f"phase_4_24_rolling_{product}.parquet"
        rolling.to_parquet(rolling_path, index=False)

        prod_report = {
            "n_dates": len(rolling),
            "date_range": [
                str(rolling["date"].min().date()),
                str(rolling["date"].max().date()),
            ],
            "ratio_overall_mean": float(rolling["ratio_front_deferred"].mean()),
            "ratio_overall_median": float(rolling["ratio_front_deferred"].median()),
            "crisis_vs_calm": crisis_vs_calm(rolling),
            "vol_of_vol_by_bucket": {
                f"{lo}_{hi}": float(rolling[f"vol_{lo}_{hi}"].std())
                for lo, hi in BUCKETS
            },
            "rolling_parquet": str(rolling_path.name),
        }
        report["products"][product] = prod_report

    # Rolling 126-day correlation and annual small multiples: CL and NG only.
    report["correlation_and_annual"] = {}
    for product in ("CL", "NG"):
        panel = full_panel[full_panel["product"] == product]
        usable = lib24.usable_returns(panel, max_gap=1, min_volume=0)

        corr = rolling_correlation(usable, window=126)
        corr_path = TMP / f"phase_4_24_corr_{product}.parquet"
        corr.to_parquet(corr_path, index=False)

        usable = usable.copy()
        usable["year"] = pd.to_datetime(usable["date"]).dt.year
        annual = {}
        for year, g in usable.groupby("year"):
            bucket = lib24.vol_by_bucket(g, lib24.mad_vol, min_obs=20)
            if len(bucket):
                annual[str(year)] = bucket.to_dict(orient="records")

        report["correlation_and_annual"][product] = {
            "corr_parquet": str(corr_path.name),
            "corr_mean": float(corr["rolling_corr_126d"].mean()),
            "corr_min": float(corr["rolling_corr_126d"].min()),
            "annual_vol_by_bucket": annual,
        }

    out = TMP / "phase_4_24_timeseries.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
