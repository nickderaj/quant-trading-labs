"""Phase 1 -- market inputs for notebook 025.

For each of CL, HO, RB, NG, ZW, KE, BZ: the observed forward curve on a set
of representative dates plus a Nelson-Siegel interpolation, the realised vol
term structure (checked against 024's published Samuelson slopes), the
curve-state split of the same, discount-rate convention, FX inputs for the
quanto case study, and cross-product correlations for the spread pricers.

Writes phase_1_25_inputs.json (summary numbers) and phase_1_25_curves.parquet
(the persisted curve points, long format) so no later phase re-does this work.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib24
import lib25

TMP = Path(__file__).resolve().parent

PRODUCTS = ["CL", "HO", "RB", "NG", "ZW", "KE", "BZ"]

# 024's published full-sample Samuelson slopes (95% CI), MAD estimator,
# max_gap=1, min_volume=0, min_contracts>=5. NEXT_PROMPT.md section 1 / 024
# write-up's table. The Phase 1 fit below must land inside a widened band
# around these -- a miss is a wiring bug, not a discovery.
PUBLISHED_024_SLOPES: dict[str, dict[str, float | list[float]]] = {
    "NG": {"slope": -0.284, "ci": [-0.330, -0.242]},
    "CL": {"slope": -0.174, "ci": [-0.197, -0.150]},
    "ZW": {"slope": -0.122, "ci": [-0.156, -0.020]},
    "KE": {"slope": -0.115, "ci": [-0.146, -0.082]},
    "HO": {"slope": -0.112, "ci": [-0.134, -0.089]},
    "BZ": {"slope": -0.107, "ci": [-0.131, -0.076]},
    "RB": {"slope": -0.098, "ci": [-0.120, -0.058]},
}

CALM_DATE = pd.Timestamp("2017-06-04")
CRISIS_DATE = pd.Timestamp("2020-04-21")

DGS2_CONVENTION = (
    "DGS2 (2-year Treasury constant maturity, FRED), forward-filled over "
    "non-trading days: the most recent published observation on or before "
    "the valuation date is used flat across the curve. Sensible default for "
    "the <=2y tenors used throughout this notebook; DFF is not used here "
    "since no instrument in scope has a tenor short enough to need it."
)


def month_end_dates(panel: pd.DataFrame) -> list[pd.Timestamp]:
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    s = pd.Series(dates, index=dates)
    month_ends = s.groupby([dates.year, dates.month]).max()
    return [pd.Timestamp(d) for d in month_ends]


def find_state_dates(panel: pd.DataFrame) -> dict:
    states = lib24.curve_state(panel)
    backwardated = states[states["curve_state"] == "backwardation"]
    contango = states[states["curve_state"] == "contango"]
    out = {}
    if len(backwardated):
        idx = backwardated["basis"].idxmax()
        out["backwardated"] = pd.Timestamp(str(backwardated.loc[idx, "date"]))
    if len(contango):
        idx = contango["basis"].idxmin()
        out["steep_contango"] = pd.Timestamp(str(contango.loc[idx, "date"]))
    return out


def curve_rows_from_panel(
    panel: pd.DataFrame, product: str, date: pd.Timestamp, tag: str
) -> list[dict]:
    """Same logic as `lib25.forward_curve`, but against an already-loaded
    panel -- avoids re-reading the parquet file per date (hundreds of
    month-end dates per product otherwise reload the whole panel each time)."""
    day = panel[panel["date"] == date]
    day = day[day["dte"] > 0].sort_values("dte")
    rows = []
    for _, row in day.iterrows():
        rows.append(
            {
                "product": product,
                "date": date,
                "tag": tag,
                "tau": float(row["dte"]) / lib25.DAYS_PER_YEAR,
                "F": float(row["close"]),
                "contract_id": int(row["contract_id"]),
                "ticker": row["ticker"],
            }
        )
    return rows


def main() -> None:
    report: dict = {"products": {}, "rates": {}, "fx": {}, "correlations": {}}
    curve_records: list[dict] = []

    for product in PRODUCTS:
        panel = lib24.load_panel(product)
        panel_usable = lib24.usable_returns(panel)
        last_date = pd.Timestamp(panel["date"].max())

        # --- curve: representative dates plus the whole month-end series
        rep_dates = {"calm": CALM_DATE, "crisis": CRISIS_DATE}
        rep_dates.update(find_state_dates(panel))
        for tag, date in rep_dates.items():
            avail = panel[panel["date"] <= date]["date"]
            if len(avail) == 0:
                continue
            actual_date = pd.Timestamp(avail.max())
            curve_records.extend(
                curve_rows_from_panel(panel, product, actual_date, tag)
            )

        for me_date in month_end_dates(panel):
            curve_records.extend(
                curve_rows_from_panel(panel, product, me_date, "month_end")
            )

        # --- vol term structure: full-sample fit, checked against 024
        fit = lib24.samuelson_slope(panel_usable, lib24.mad_vol, n_boot=300, seed=25)
        published = PUBLISHED_024_SLOPES.get(product)
        within_band = None
        if published is not None and np.isfinite(fit["slope"]):
            ci = published["ci"]
            assert isinstance(ci, list)
            lo, hi = ci
            width = hi - lo
            band = (lo - 2 * width, hi + 2 * width)
            within_band = bool(band[0] <= fit["slope"] <= band[1])
            assert within_band, (
                f"{product}: fitted slope {fit['slope']:.3f} outside widened "
                f"band {band} around 024's published CI {published['ci']} -- "
                "this is a wiring bug, not a new finding"
            )

        # --- curve-state split (024 D4)
        vol_by_state = lib25.vol_term_structure(
            product, last_date, window=100_000, by_state=True
        )

        product_report = {
            "n_rows": len(panel),
            "n_usable_returns": len(panel_usable),
            "representative_dates": {k: str(v) for k, v in rep_dates.items()},
            "samuelson_fit_full_sample": {
                "slope": fit["slope"],
                "ci_lo": fit["ci_lo"],
                "ci_hi": fit["ci_hi"],
                "r2": fit["r2"],
            },
            "published_024": published,
            "within_widened_band_of_024": within_band,
            "curve_state_vol": {
                state: (
                    {
                        "sigma_1": vol_by_state[state]["sigma_1"],
                        "k": vol_by_state[state]["k"],
                        "r2": vol_by_state[state]["r2"],
                    }
                    if vol_by_state.get(state) is not None
                    else None
                )
                for state in ("contango", "backwardation")
            },
        }
        report["products"][product] = product_report
        print(
            f"{product}: slope={fit['slope']:.3f} CI=[{fit['ci_lo']:.3f},"
            f"{fit['ci_hi']:.3f}] within_band={within_band}"
        )

    # --- rates
    dgs2 = pd.read_parquet(f"{lib25.FRED_DIR}/DGS2.parquet").sort_values("date")
    report["rates"] = {
        "series": "DGS2",
        "convention": DGS2_CONVENTION,
        "date_range": [str(dgs2["date"].min()), str(dgs2["date"].max())],
    }

    # --- FX: 6E returns, vol, rolling correlation with CL
    fx = lib25.fx_series("6E")
    fx_returns = np.log(fx).diff().dropna()
    fx_vol = lib24.mad_vol(fx_returns.to_numpy())
    corr_cl_fx = lib25.corr_rolling("CL", "6E", window=126).dropna()
    report["fx"] = {
        "pair": "6E",
        "date_range": [str(fx.index.min()), str(fx.index.max())],
        "annualised_mad_vol": fx_vol,
        "rolling_corr_with_CL": {
            "window": 126,
            "mean": float(corr_cl_fx.mean()),
            "std": float(corr_cl_fx.std()),
            "min": float(corr_cl_fx.min()),
            "max": float(corr_cl_fx.max()),
            "n": len(corr_cl_fx),
        },
    }

    # --- correlations for the spread pricers
    for pair in [("CL", "HO"), ("CL", "RB"), ("CL", "BZ")]:
        corr = lib25.corr_rolling(*pair, window=126).dropna()
        report["correlations"][f"{pair[0]}_{pair[1]}"] = {
            "window": 126,
            "mean": float(corr.mean()),
            "std": float(corr.std()),
            "n": len(corr),
        }

    out_json = TMP / "phase_1_25_inputs.json"
    out_json.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {out_json}")

    curves_df = pd.DataFrame(curve_records)
    out_parquet = TMP / "phase_1_25_curves.parquet"
    curves_df.to_parquet(out_parquet)
    print(f"wrote {out_parquet} ({len(curves_df)} rows)")


if __name__ == "__main__":
    main()
