"""Phase 5: Seasonality analysis for the Samuelson effect (notebook 024).

Analyses:
1. NG delivery-month split (winter vs summer) with full dte buckets
2. Grains delivery-month split (1-12) with front-bucket only
3. NG and CL calendar-month x maturity-bucket heatmap matrix
4. GC control (same as NG, should show no seasonality)
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib24


def extract_month(contract_month_str):
    """Extract month (1-12) from contract_month string (YYYY-MM format)."""
    if pd.isna(contract_month_str):
        return None
    s = str(contract_month_str).strip()
    if "-" not in s:
        return None
    try:
        return int(s.split("-")[1])
    except (ValueError, IndexError):
        return None


def main():
    # Load the cleaned panel from Phase 1
    panel_path = Path(__file__).resolve().parent / "phase_1_24_returns.parquet"
    full_panel = pd.read_parquet(panel_path)

    report = {}

    # ========================================================================
    # 1. NG delivery-month split: winter vs summer
    # ========================================================================
    ng_panel = full_panel[full_panel["product"] == "NG"].copy()
    ng_panel["delivery_month"] = ng_panel["contract_month"].apply(extract_month)
    ng_panel = ng_panel.dropna(subset=["delivery_month"])

    winter_mask = ng_panel["delivery_month"].isin([12, 1, 2, 3])
    summer_mask = ng_panel["delivery_month"].isin(range(4, 11))

    ng_winter = ng_panel[winter_mask]
    ng_summer = ng_panel[summer_mask]

    ng_winter_usable = lib24.usable_returns(ng_winter, max_gap=1, min_volume=0)
    ng_summer_usable = lib24.usable_returns(ng_summer, max_gap=1, min_volume=0)

    ng_winter_vols = lib24.vol_by_bucket(
        ng_winter_usable, lib24.mad_vol, bins=lib24.DTE_BINS, min_obs=20
    )
    ng_summer_vols = lib24.vol_by_bucket(
        ng_summer_usable, lib24.mad_vol, bins=lib24.DTE_BINS, min_obs=20
    )

    report["ng_seasonality"] = {
        "winter": ng_winter_vols.to_dict(orient="records"),
        "summer": ng_summer_vols.to_dict(orient="records"),
    }

    # ========================================================================
    # 2. Grains delivery-month split (ZC, ZS, ZW, KE)
    # ========================================================================
    grains = ["ZC", "ZS", "ZW", "KE"]
    report["grains_seasonality"] = {}

    for product in grains:
        prod_panel = full_panel[full_panel["product"] == product].copy()
        prod_panel["delivery_month"] = prod_panel["contract_month"].apply(extract_month)
        prod_panel = prod_panel.dropna(subset=["delivery_month"])

        month_vols = {}
        for month in range(1, 13):
            month_data = prod_panel[prod_panel["delivery_month"] == month]
            month_usable = lib24.usable_returns(month_data, max_gap=1, min_volume=0)

            if len(month_usable) >= 15:
                month_vol_df = lib24.vol_by_bucket(
                    month_usable,
                    lib24.mad_vol,
                    bins=[0, 90, 10000],
                    min_obs=15,
                )
                # Extract front bucket (dte_lo == 0)
                front = month_vol_df[month_vol_df["dte_lo"] == 0.0]
                if len(front) > 0:
                    vol = float(front["vol"].iloc[0])
                    month_vols[str(month)] = vol
                else:
                    month_vols[str(month)] = None
            else:
                month_vols[str(month)] = None

        report["grains_seasonality"][product] = month_vols

    # ========================================================================
    # 3. NG and CL calendar-month x maturity-bucket heatmap
    # ========================================================================
    report["heatmap_matrix"] = {}

    for product in ["NG", "CL"]:
        prod_panel = full_panel[full_panel["product"] == product].copy()
        prod_panel["calendar_month"] = pd.to_datetime(prod_panel["date"]).dt.month
        prod_usable = lib24.usable_returns(prod_panel, max_gap=1, min_volume=0)

        heatmap = {}
        for cal_month in range(1, 13):
            for bucket_label, (lo, hi) in [
                ("front_0_90", (0, 90)),
                ("mid_90_365", (90, 365)),
                ("back_365plus", (365, 10000)),
            ]:
                mask = (
                    (prod_usable["calendar_month"] == cal_month)
                    & (prod_usable["dte"] >= lo)
                    & (prod_usable["dte"] < hi)
                )
                cell_data = prod_usable[mask]

                if len(cell_data) >= 15:
                    vol = lib24.mad_vol(cell_data["r"].to_numpy())
                    heatmap[f"{cal_month}_{bucket_label}"] = vol
                else:
                    heatmap[f"{cal_month}_{bucket_label}"] = None

        report["heatmap_matrix"][product] = heatmap

    # ========================================================================
    # 4. GC control (same as NG)
    # ========================================================================
    gc_panel = full_panel[full_panel["product"] == "GC"].copy()
    gc_panel["delivery_month"] = gc_panel["contract_month"].apply(extract_month)
    gc_panel = gc_panel.dropna(subset=["delivery_month"])

    gc_winter_mask = gc_panel["delivery_month"].isin([12, 1, 2, 3])
    gc_summer_mask = gc_panel["delivery_month"].isin(range(4, 11))

    gc_winter = gc_panel[gc_winter_mask]
    gc_summer = gc_panel[gc_summer_mask]

    gc_winter_usable = lib24.usable_returns(gc_winter, max_gap=1, min_volume=0)
    gc_summer_usable = lib24.usable_returns(gc_summer, max_gap=1, min_volume=0)

    gc_winter_vols = lib24.vol_by_bucket(
        gc_winter_usable, lib24.mad_vol, bins=lib24.DTE_BINS, min_obs=20
    )
    gc_summer_vols = lib24.vol_by_bucket(
        gc_summer_usable, lib24.mad_vol, bins=lib24.DTE_BINS, min_obs=20
    )

    report["gc_control"] = {
        "winter": gc_winter_vols.to_dict(orient="records"),
        "summer": gc_summer_vols.to_dict(orient="records"),
    }

    # Write output
    output_path = Path(__file__).resolve().parent / "phase_5_24_seasonality.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Output written to {output_path}")


if __name__ == "__main__":
    main()
