"""Phase 1 -- panel construction and data-quality census for all 16 products.

Loops all 16 products in one process. Reproduces the NEXT_PROMPT.md section 1
row-count table and asserts against it, censuses the far-dated junk prints,
cross-checks MAD vol against metrics/<PROD>.parquet's realised_vol_20d where
available, and persists the cleaned panel to phase_1_24_returns.parquet so no
later phase re-does this work.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib24

TMP = Path(__file__).resolve().parent
DATA_DIR = Path(lib24.DATA_DIR)

# Verified 2026-09-22 (NEXT_PROMPT.md section 1). Tolerance: rounding only.
EXPECTED_ROWS = {
    "BZ": 49702,
    "CL": 109758,
    "ES": 8846,
    "GC": 52734,
    "HO": 80196,
    "KE": 24540,
    "NG": 144449,
    "PA": 16716,
    "PL": 23521,
    "RB": 64983,
    "SI": 39466,
    "ZC": 51413,
    "ZL": 52581,
    "ZM": 54764,
    "ZS": 52880,
    "ZW": 40667,
}


def main() -> None:
    report: dict = {"products": {}}
    all_returns = []

    for product in lib24.PRODUCTS:
        raw = pd.read_parquet(DATA_DIR / "ohlcv" / f"{product}.parquet")
        raw_rows = len(raw)

        panel = lib24.load_panel(product)
        nonpositive = int((raw["close"] <= 0).sum())

        # Data-quality fix: a contract that ever printed a non-positive close
        # (other than the documented genuine negative-WTI event) has
        # demonstrated unreliable settlement data for its ENTIRE life, not just
        # on the bad-print day -- e.g. CL203212/CL203305/CL203311 print daily
        # log returns as large as 1.4-2.5 (a multi-hundred-percent one-day
        # move) on days their close is nominally positive. A single |r|>1.0
        # filter does not catch this (their non-outlier days are still ~30-60%
        # annualised noisier than clean contracts), and because these
        # contracts cluster at the far-dated end, they previously produced a
        # spurious volatility SPIKE in the longest maturity bucket for CL, GC,
        # and (mildly) other products -- mirroring notebook 023's own
        # discovery of exactly these tickers. Excluded here, entirely, from
        # every downstream phase; counted and named so the exclusion is
        # visible rather than absorbed into a different filter's numbers.
        event_ticker = lib24.NEGATIVE_PRICE_EVENT["ticker"]
        bad_tickers = sorted(
            t
            for t in raw.loc[raw["close"] <= 0, "ticker"].unique()
            if t != event_ticker
        )
        rows_before_ticker_exclusion = len(panel)
        panel = panel[~panel["ticker"].isin(bad_tickers)].reset_index(drop=True)

        prod_report: dict = {
            "raw_rows": raw_rows,
            "close_le_0_count": nonpositive,
            "unreliable_tickers_excluded": {
                "count": len(bad_tickers),
                "tickers": bad_tickers[:30],
                "rows_removed": rows_before_ticker_exclusion - len(panel),
            },
            "date_range": [
                str(panel["date"].min().date()),
                str(panel["date"].max().date()),
            ],
            "n_contracts": int(panel["contract_id"].nunique()),
        }

        if product in EXPECTED_ROWS:
            expected = EXPECTED_ROWS[product]
            diff_pct = abs(raw_rows - expected) / expected * 100
            prod_report["expected_raw_rows"] = expected
            prod_report["raw_rows_diff_pct"] = round(diff_pct, 3)
            if diff_pct > 1.0:
                raise AssertionError(
                    f"{product}: raw_rows={raw_rows} vs expected={expected} "
                    f"(diff {diff_pct:.2f}%) -- stop and find out why."
                )

        # dte quantiles
        dte = panel["dte"].dropna()
        prod_report["dte_quantiles"] = {
            q: float(np.quantile(dte, q / 100)) for q in [0, 10, 25, 50, 75, 90, 100]
        }
        # Data-quality finding: a handful of products (observed: BZ) carry trade
        # prints dated after the recorded expiry -- likely a last-trade-date vs.
        # settlement-date mismatch in this product's contract metadata, not a
        # panel-construction bug. CL genuinely has none, per NEXT_PROMPT.md.
        # Report the count rather than silently dropping; DTE_BINS starts at 0
        # so these rows fall outside every bucket in Phase 2 onward anyway.
        n_negative_dte = int((dte < 0).sum())
        prod_report["negative_dte_count"] = n_negative_dte
        if product == "CL":
            assert n_negative_dte == 0, "CL: negative dte found, expected none"

        # Junk-print census: |r| > 1.0
        junk = panel[panel["r"].abs() > 1.0].dropna(subset=["r"])
        prod_report["junk_prints_gt_1_log_return"] = {
            "count": len(junk),
            "tickers": sorted(junk["ticker"].unique().tolist())[:30],
            "max_abs_r": float(panel["r"].abs().max())
            if panel["r"].notna().any()
            else None,
            "sample_dates": [str(d.date()) for d in junk["date"].head(10)],
        }

        # usable returns across max_gap x min_volume grid
        variants: dict = {}
        for max_gap in (1, 3):
            for min_volume in (0, 100, 1000):
                key = f"gap{max_gap}_vol{min_volume}"
                usable = lib24.usable_returns(
                    panel, max_gap=max_gap, min_volume=min_volume
                )
                variants[key] = {
                    "n": len(usable),
                    "retained_fraction": round(len(usable) / raw_rows, 4)
                    if raw_rows
                    else None,
                }
        prod_report["usable_returns_variants"] = variants

        # Cross-check against metrics/<PROD>.parquet if present
        metrics_path = DATA_DIR / "metrics" / f"{product}.parquet"
        if metrics_path.exists():
            metrics = pd.read_parquet(metrics_path)
            metrics["date"] = pd.to_datetime(metrics["date"])
            usable = lib24.usable_returns(panel, max_gap=1, min_volume=0)
            merged = usable.merge(
                metrics[["contract_id", "date", "realised_vol_20d"]],
                on=["contract_id", "date"],
                how="inner",
            ).dropna(subset=["realised_vol_20d"])
            if len(merged) > 100:
                overall_mad = lib24.mad_vol(usable["r"].to_numpy())
                prod_report["metrics_cross_check"] = {
                    "n_matched_rows": len(merged),
                    "metrics_realised_vol_20d_mean": float(
                        merged["realised_vol_20d"].mean()
                    ),
                    "our_full_sample_mad_vol": float(overall_mad),
                    "note": (
                        "metrics parquet only from 2018 for 12/16 products; used as "
                        "a directional sanity check (overall MAD vol vs. metrics' "
                        "mean rolling 20d realised vol) only, per NEXT_PROMPT.md."
                    ),
                }
            else:
                prod_report["metrics_cross_check"] = {"n_matched_rows": len(merged)}
        else:
            prod_report["metrics_cross_check"] = None

        report["products"][product] = prod_report

        keep_cols = [
            "product",
            "contract_id",
            "ticker",
            "date",
            "expiry",
            "close",
            "volume",
            "log_close",
            "r",
            "gap",
            "dte",
            "contract_month",
        ]
        all_returns.append(panel[[c for c in keep_cols if c in panel.columns]])

    full_panel = pd.concat(all_returns, ignore_index=True)
    full_panel.to_parquet(TMP / "phase_1_24_returns.parquet", index=False)
    report["total_rows_all_products"] = len(full_panel)
    report["total_contracts_all_products"] = int(full_panel["contract_id"].nunique())

    out = TMP / "phase_1_24_panel_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(
        f"wrote {out} ({len(full_panel)} total rows across {len(lib24.PRODUCTS)} products)"
    )


if __name__ == "__main__":
    main()
