"""Phase 3: Samuelson slope estimation across 16 futures markets.

Compute log-log slopes of volatility vs. time-to-expiry for each product,
estimator, and liquidity variant. Investigate the metals-inversion anomaly
(PL and PA). Write results to phase_3_24_slopes.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib24


def compute_primary_slopes(full_panel: pd.DataFrame) -> dict:
    """Compute slopes for all products × estimators × min_volume variants."""
    slopes: dict[str, dict] = {}
    for product in lib24.PRODUCTS:
        print(f"Loading {product}...")
        product_panel = full_panel[full_panel["product"] == product]

        slopes[product] = {}
        for estimator_name, estimator_fn in lib24.ESTIMATORS.items():
            slopes[product][estimator_name] = {}
            for min_vol in [0, 100, 1000]:
                usable = lib24.usable_returns(
                    product_panel, max_gap=1, min_volume=min_vol
                )
                slope_result = lib24.samuelson_slope(
                    usable,
                    estimator_fn,
                    n_boot=300,
                    seed=24,
                    bins=lib24.DTE_BINS,
                    min_obs=50,
                )
                slopes[product][estimator_name][f"vol{min_vol}"] = slope_result
                print(
                    f"  {estimator_name} vol{min_vol}: slope={slope_result['slope']:.4f}"
                )

    return slopes


def rank_by_primary_slope(slopes: dict) -> list:
    """Rank products by slope (mad estimator, min_volume=0)."""
    ranking = []
    for product in lib24.PRODUCTS:
        slope_dict = slopes[product]["mad"]["vol0"]
        ranking.append(
            {
                "product": product,
                "slope": slope_dict["slope"],
                "ci_lo": slope_dict["ci_lo"],
                "ci_hi": slope_dict["ci_hi"],
            }
        )
    ranking.sort(key=lambda x: x["slope"])
    return ranking


def investigate_metals(slopes: dict, full_panel: pd.DataFrame) -> dict:
    """Investigate metals inversion (PL and PA) under various constraints."""
    metals_inversion: dict[str, dict] = {}

    for product in ["PL", "PA"]:
        print(f"\nInvestigating {product}...")
        product_panel = full_panel[full_panel["product"] == product]
        metals_inversion[product] = {}

        # Compute with volume > 1000
        usable_vol1000 = lib24.usable_returns(product_panel, max_gap=1, min_volume=1000)
        slope_vol1000 = lib24.samuelson_slope(
            usable_vol1000,
            lib24.mad_vol,
            n_boot=300,
            seed=24,
            bins=lib24.DTE_BINS,
            min_obs=50,
        )
        metals_inversion[product]["vol1000"] = slope_vol1000

        # Compute with max_gap <= 3
        usable_gap3 = lib24.usable_returns(product_panel, max_gap=3, min_volume=0)
        slope_gap3 = lib24.samuelson_slope(
            usable_gap3,
            lib24.mad_vol,
            n_boot=300,
            seed=24,
            bins=lib24.DTE_BINS,
            min_obs=50,
        )
        metals_inversion[product]["gap3"] = slope_gap3

        # Copy primary (vol0, gap1, mad)
        usable_primary = lib24.usable_returns(product_panel, max_gap=1, min_volume=0)
        slope_primary = lib24.samuelson_slope(
            usable_primary,
            lib24.mad_vol,
            n_boot=300,
            seed=24,
            bins=lib24.DTE_BINS,
            min_obs=50,
        )
        metals_inversion[product]["primary"] = slope_primary

        # Compute 365-730 bucket stats
        bucket_data = lib24.vol_by_bucket(
            usable_primary, lib24.mad_vol, bins=lib24.DTE_BINS, min_obs=1
        )
        # Find rows where dte_mid is between 365 and 730
        bucket_365_730 = bucket_data[
            (bucket_data["dte_mid"] >= 365) & (bucket_data["dte_mid"] <= 730)
        ]
        if len(bucket_365_730) > 0:
            n_contracts = int(bucket_365_730["n_contracts"].sum())
            n_obs = int(bucket_365_730["n_obs"].sum())
        else:
            n_contracts = 0
            n_obs = 0

        metals_inversion[product]["bucket_365_730_stats"] = {
            "n_contracts": n_contracts,
            "n_obs": n_obs,
        }
        print(f"  365-730 bucket: n_contracts={n_contracts}, n_obs={n_obs}")

    return metals_inversion


def metals_inversion_with_contract_floor(
    full_panel: pd.DataFrame,
) -> dict:
    """Recompute metals slopes (GC, SI, PL, PA) restricted to n_contracts >= 10."""
    metals_contract_floor = {}

    for product in ["GC", "SI", "PL", "PA"]:
        print(f"\nComputing {product} with n_contracts >= 10 restriction...")
        product_panel = full_panel[full_panel["product"] == product]
        usable = lib24.usable_returns(product_panel, max_gap=1, min_volume=0)

        # Get bucket data
        bucket_data = lib24.vol_by_bucket(
            usable, lib24.mad_vol, bins=lib24.DTE_BINS, min_obs=50
        )

        # Filter to rows with n_contracts >= 10
        filtered = bucket_data[bucket_data["n_contracts"] >= 10].copy()
        n_buckets_used = len(filtered)

        if len(filtered) < 2:
            metals_contract_floor[product] = {
                "slope": float("nan"),
                "n_buckets_used": n_buckets_used,
            }
            print(f"  Not enough buckets (need 2, have {n_buckets_used})")
            continue

        # Manually fit weighted log-log OLS
        x = np.log(filtered["dte_mid"].to_numpy())
        y = np.log(filtered["vol"].to_numpy())
        w = np.sqrt(filtered["n_obs"].to_numpy())

        X = np.column_stack([np.ones_like(x), x])
        Wm = np.diag(w)

        coef, *_ = np.linalg.lstsq(Wm @ X, Wm @ y, rcond=None)
        slope = float(coef[1])

        metals_contract_floor[product] = {
            "slope": slope,
            "n_buckets_used": n_buckets_used,
        }
        print(
            f"  {product} slope (n_contracts>=10): {slope:.4f} ({n_buckets_used} buckets)"
        )

    return metals_contract_floor


def generate_verdicts(metals_inversion: dict, metals_contract_floor: dict) -> dict:
    """Generate one-sentence verdicts for PL and PA metals inversion."""
    verdicts = {}

    for product in ["PL", "PA"]:
        vol1000_slope = metals_inversion[product]["vol1000"]["slope"]
        gap3_slope = metals_inversion[product]["gap3"]["slope"]
        contract_floor_slope = metals_contract_floor[product]["slope"]

        # Check if inversion (positive slope) survives
        survives_vol1000 = vol1000_slope > 0 if np.isfinite(vol1000_slope) else False
        survives_gap3 = gap3_slope > 0 if np.isfinite(gap3_slope) else False
        survives_contract_floor = (
            contract_floor_slope > 0 if np.isfinite(contract_floor_slope) else False
        )

        # Generate verdict
        if survives_vol1000 and survives_gap3 and survives_contract_floor:
            verdict = f"{product} shows a persistent positive (inverted Samuelson) slope that survives volume>1000, max_gap<=3, and n_contracts>=10 filters."
        else:
            failed_filters = []
            if not survives_vol1000:
                failed_filters.append("volume>1000")
            if not survives_gap3:
                failed_filters.append("max_gap<=3")
            if not survives_contract_floor:
                failed_filters.append("n_contracts>=10")

            verdict = f"{product} slope inversion appears to be a liquidity artefact from thin far-dated contracts, as it does not survive {', '.join(failed_filters)} filters."

        verdicts[product] = verdict

    return verdicts


def main():
    print("Phase 3: Computing Samuelson slopes...")
    print()

    # Load full panel once
    full_panel = pd.read_parquet(
        Path(__file__).resolve().parent / "phase_1_24_returns.parquet"
    )

    # Compute primary slopes
    slopes = compute_primary_slopes(full_panel)

    # Rank by primary slope
    ranking = rank_by_primary_slope(slopes)

    # Investigate metals inversion
    metals_inversion = investigate_metals(slopes, full_panel)

    # Metals with contract floor
    metals_contract_floor = metals_inversion_with_contract_floor(full_panel)

    # Generate verdicts
    verdicts = generate_verdicts(metals_inversion, metals_contract_floor)
    for product in ["PL", "PA"]:
        metals_inversion[product]["verdict"] = verdicts[product]

    # Build report
    report = {
        "slopes": slopes,
        "ranking_primary": ranking,
        "metals_inversion": metals_inversion,
        "metals_inversion_contract_floor": metals_contract_floor,
    }

    # Write to JSON
    output_path = Path(__file__).resolve().parent / "phase_3_24_slopes.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nWrote {output_path}")
    print()
    print("Top 3 (most negative):")
    for i in range(min(3, len(ranking))):
        r = ranking[i]
        print(f"  {i + 1}. {r['product']}: {r['slope']:.4f}")
    print()
    print("Bottom 3 (least negative/most positive):")
    for i in range(max(0, len(ranking) - 3), len(ranking)):
        r = ranking[i]
        print(f"  {len(ranking) - i}. {r['product']}: {r['slope']:.4f}")
    print()
    print("Metals inversion verdicts:")
    for product in ["PL", "PA"]:
        print(f"  {product}: {verdicts[product]}")


if __name__ == "__main__":
    main()
