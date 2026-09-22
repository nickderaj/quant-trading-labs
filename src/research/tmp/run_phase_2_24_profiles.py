"""Phase 2: compute volatility profiles across 16 products, 3 estimators, 3 liquidity variants."""

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib24

# Load the full cleaned panel (Phase 1 output)
full_panel = pd.read_parquet(
    Path(__file__).resolve().parent / "phase_1_24_returns.parquet"
)

# Initialize report structure
report: dict[str, Any] = {"products": {}}

# All parameter combinations
products = lib24.PRODUCTS
estimators = ["mad", "std", "winsor"]
liquidity_variants = [0, 100, 1000]
gap_variants = [1, 3]

# Compute all combinations
for product in products:
    report["products"][product] = {}
    product_panel = full_panel[full_panel["product"] == product]

    for estimator_name in estimators:
        estimator_fn = lib24.ESTIMATORS[estimator_name]
        report["products"][product][estimator_name] = {}

        for max_gap in gap_variants:
            for min_volume in liquidity_variants:
                key = f"gap{max_gap}_vol{min_volume}"

                # Filter to usable returns
                usable = lib24.usable_returns(
                    product_panel, max_gap=max_gap, min_volume=min_volume
                )

                # Compute vol_by_bucket
                bucket_df = lib24.vol_by_bucket(
                    usable, estimator_fn, bins=lib24.DTE_BINS, min_obs=50
                )

                # Convert to list of dicts
                bucket_list = bucket_df.to_dict("records")
                report["products"][product][estimator_name][key] = bucket_list

    # Compute front/deferred ratio for primary config: mad, gap=1, vol=0
    primary_usable = lib24.usable_returns(product_panel, max_gap=1, min_volume=0)
    primary_bins = [0, 30, 60, 90, 120, 180, 270, 365, 547, 730]
    primary_bucket = lib24.vol_by_bucket(
        primary_usable,
        lib24.ESTIMATORS["mad"],
        bins=primary_bins,
        min_obs=50,
    )

    # Find front (0-30) and deferred (365-730) buckets
    front_row = primary_bucket[primary_bucket["dte_lo"] == 0]
    deferred_row = primary_bucket[primary_bucket["dte_lo"] == 365]

    if len(front_row) > 0 and len(deferred_row) > 0:
        front_vol = front_row.iloc[0]["vol"]
        deferred_vol = deferred_row.iloc[0]["vol"]
        if deferred_vol > 0:
            ratio = front_vol / deferred_vol
        else:
            ratio = float("nan")
    else:
        ratio = float("nan")

    report["products"][product]["front_deferred_ratio_primary"] = ratio

# Write to JSON
output_path = Path(__file__).resolve().parent / "phase_2_24_profiles.json"
with open(output_path, "w") as f:
    json.dump(report, f, indent=2, default=str)

print(f"Written to {output_path}")

# Verify valid JSON and report counts
with open(output_path, "r") as f:
    verified = json.load(f)

n_products = len(verified["products"])
print(f"Products: {n_products}")

for product in list(verified["products"].keys())[:1]:  # Check first product
    n_estimators = len(verified["products"][product]) - 1  # -1 for the ratio
    print(f"Estimators per product: {n_estimators}")
    for est in estimators:
        n_variant_keys = len(verified["products"][product][est])
        print(f"  {est}: {n_variant_keys} gap/vol variant keys")

# Print smoke test front/deferred ratios
print("\nFront/deferred ratios (primary config: mad, gap=1, vol=0):")
for prod in ["NG", "CL", "ES", "GC", "PA", "PL"]:
    ratio = verified["products"][prod]["front_deferred_ratio_primary"]
    print(
        f"{prod}: {ratio:.2f}"
        if isinstance(ratio, (int, float))
        else f"{prod}: {ratio}"
    )
