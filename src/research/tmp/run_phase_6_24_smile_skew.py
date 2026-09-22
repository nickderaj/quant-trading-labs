"""Phase 6 of Notebook 024: realised smile, skew, and moments.

The realised smile is the volatility surface across moneyness (log basis
against the front contract). The realised skew compares volatility under
contango vs backwardation. All output is realised volatility, never implied.
"""

import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib24


def main() -> None:
    # Load full panel
    full_panel = pd.read_parquet(
        Path(__file__).resolve().parent / "phase_1_24_returns.parquet"
    )

    report: dict[str, Any] = {
        "moments": {},
        "smile": {},
        "skew": {},
        "densities": {},
    }

    # ========================================================================
    # 1. Realised moments by bucket for all 16 products
    # ========================================================================
    for product in lib24.PRODUCTS:
        panel = full_panel[full_panel["product"] == product]
        usable = lib24.usable_returns(panel, max_gap=1)
        moments = lib24.realised_moments_by_bucket(usable, min_obs=50)
        report["moments"][product] = moments.to_dict(orient="records")

    # ========================================================================
    # 2. Realised "smile" for CL and NG
    # ========================================================================
    for product in ["CL", "NG"]:
        panel = full_panel[full_panel["product"] == product]
        usable = lib24.usable_returns(panel, max_gap=1)
        with_moneyness = lib24.log_moneyness(usable)

        moneyness_edges = np.array(
            [-0.5, -0.2, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.2, 0.5]
        )

        # Pooled smile across all maturities
        pooled_smile: list[dict[str, Any]] = []
        with_moneyness["moneyness_bucket"] = pd.cut(  # type: ignore[call-overload]
            with_moneyness["log_moneyness"],
            bins=moneyness_edges,
            right=False,
        )
        for interval, group in with_moneyness.groupby(
            "moneyness_bucket", observed=True
        ):
            if len(group) < 200:
                continue
            interval_cast = cast(pd.Interval, interval)
            r_values = group["r"].dropna().to_numpy()
            pooled_smile.append(
                {
                    "moneyness_lo": float(interval_cast.left),
                    "moneyness_hi": float(interval_cast.right),
                    "moneyness_mid": float(
                        (interval_cast.left + interval_cast.right) / 2
                    ),
                    "vol": lib24.mad_vol(r_values),
                    "n_obs": len(r_values),
                }
            )

        # Smile by broad maturity bins
        maturity_bins = {
            "0_90": (0, 90),
            "90_365": (90, 365),
            "365_plus": (365, 3650),
        }
        by_maturity: dict[str, list[dict[str, Any]]] = {}
        for bin_name, (dte_lo, dte_hi) in maturity_bins.items():
            bin_data = with_moneyness[
                (with_moneyness["dte"] >= dte_lo) & (with_moneyness["dte"] < dte_hi)
            ]
            bin_data["moneyness_bucket"] = pd.cut(  # type: ignore[call-overload]
                bin_data["log_moneyness"],
                bins=moneyness_edges,
                right=False,
            )
            smile_records: list[dict[str, Any]] = []
            for interval, group in bin_data.groupby("moneyness_bucket", observed=True):
                if len(group) < 200:
                    continue
                interval_cast = cast(pd.Interval, interval)
                r_values = group["r"].dropna().to_numpy()
                smile_records.append(
                    {
                        "moneyness_lo": float(interval_cast.left),
                        "moneyness_hi": float(interval_cast.right),
                        "moneyness_mid": float(
                            (interval_cast.left + interval_cast.right) / 2
                        ),
                        "vol": lib24.mad_vol(r_values),
                        "n_obs": len(r_values),
                    }
                )
            by_maturity[bin_name] = smile_records

        report["smile"][product] = {
            "pooled": pooled_smile,
            "by_maturity_bin": by_maturity,
        }

    # ========================================================================
    # 3. Realised "skew" for CL and NG
    # ========================================================================
    for product in ["CL", "NG"]:
        panel = full_panel[full_panel["product"] == product]
        curve = lib24.curve_state(panel)
        usable = lib24.usable_returns(panel, max_gap=1)
        merged = usable.merge(curve[["date", "curve_state"]], on="date", how="left")

        skew_result: dict[str, list[dict[str, Any]]] = {}
        for state in ["contango", "backwardation"]:
            subset = merged[merged["curve_state"] == state]
            vol_by_state = lib24.vol_by_bucket(subset, lib24.mad_vol, min_obs=30)
            skew_result[state] = cast(
                list[dict[str, Any]], vol_by_state.to_dict(orient="records")
            )

        report["skew"][product] = skew_result

    # ========================================================================
    # 4. Front vs deferred return densities for CL and NG
    # ========================================================================
    for product in ["CL", "NG"]:
        panel = full_panel[full_panel["product"] == product]
        usable = lib24.usable_returns(panel, max_gap=1)

        # Front bucket: dte 0-60
        front = usable[(usable["dte"] >= 0) & (usable["dte"] < 60)]
        front_r = front["r"].dropna().to_numpy()

        # Deferred bucket: dte 365-730
        deferred = usable[(usable["dte"] >= 365) & (usable["dte"] < 730)]
        deferred_r = deferred["r"].dropna().to_numpy()

        # Downsample to at most 3000 points with fixed seed
        rng = np.random.default_rng(24)
        if len(front_r) > 3000:
            indices = rng.choice(len(front_r), size=3000, replace=False)
            front_r = front_r[indices]
        if len(deferred_r) > 3000:
            indices = rng.choice(len(deferred_r), size=3000, replace=False)
            deferred_r = deferred_r[indices]

        report["densities"][product] = {
            "front": front_r.tolist(),
            "deferred": deferred_r.tolist(),
        }

    # Write output
    output_path = Path(__file__).resolve().parent / "phase_6_24_smile_skew.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Output written to {output_path}")


if __name__ == "__main__":
    main()
