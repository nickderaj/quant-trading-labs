"""Gate DC: the data contract bites (NEXT_PROMPT.md sec 4, sec 10).

`risk.hygiene.assert_risk_inputs` must reject a synthetic frame carrying the
*symptom* of each of 008 Phase 0's four separately-discovered bugs (restated
in NEXT_PROMPT.md sec 4's table), with a named error, and must accept the
real, already-hygiene-passed 16-product frame with zero false rejections.

Threshold (sec 10): 4/4 rejected with the correct named error; 0 false
rejections on the clean 16-product frame. Hard gate.

Writes `src/research/tmp/run_risk_01_data_contract_results.json`.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from typing import Any

sys.path.insert(0, "src")

import numpy as np
import polars as pl

from risk import hygiene as H

CURVE_DIR = "src/research/tmp/phase_0_curves"
OUT_PATH = "src/research/tmp/run_risk_01_data_contract_results.json"
PRODUCTS = [
    "CL", "BZ", "NG", "HO", "RB", "GC", "SI", "PL", "PA",
    "ZC", "ZW", "KE", "ZS", "ZL", "ZM", "ES",
]  # fmt: skip


def _clean_synthetic_curve(n: int = 300, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    close = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    curve = pl.DataFrame(
        {
            "date": dates,
            "close_f1": close,
            "contract_month_f1": ["2020-01"] * n,
        }
    )
    log_ret = np.diff(np.log(close), prepend=np.nan)
    curve = curve.with_columns(pl.Series("log_return", log_ret))
    setattr(curve, H.PROVENANCE_ATTR, H.PROVENANCE_VALUE)
    return curve


def bug1_contamination_reaches_front_month() -> pl.DataFrame:
    """Symptom of 008 Bug 1 (mislabeled spread/differential contract, e.g.
    NG202507): a near-zero close for an extended block -- a differential
    series mislabeled as an outright, not a single bad day. If
    `flag_contaminated_rows` failed to catch it, this is what would reach
    the final curve: a frozen near-zero block."""
    curve = _clean_synthetic_curve()
    ret = curve["log_return"].to_numpy().copy()
    ret[100 : 100 + H.REALIZED_VOL_WINDOW + 5] = 1e-9
    curve = curve.with_columns(pl.Series("log_return", ret))
    setattr(curve, H.PROVENANCE_ATTR, H.PROVENANCE_VALUE)
    return curve


def bug2_stale_quote_not_screened() -> pl.DataFrame:
    """Symptom of 008 Bug 2 (front-month selection on a stale single quote):
    a quiet day misread as price discovery repeats an unchanged close for
    several days running -- the `liquidity_screen` failure mode."""
    curve = _clean_synthetic_curve()
    close = curve["close_f1"].to_numpy().copy()
    close[150:155] = close[150]  # a run of 5, past the observed ceiling of 3
    curve = curve.with_columns(pl.Series("close_f1", close))
    setattr(curve, H.PROVENANCE_ATTR, H.PROVENANCE_VALUE)
    return curve


def bug3_near_dead_contract_months() -> pl.DataFrame:
    """Symptom of 008 Bug 3 (roll_calendar lists nominally-listed but
    near-dead contract months, e.g. PL's F1 57% null): rolling into a
    near-dead month collapses the usable series far below the fitting
    floor."""
    return _clean_synthetic_curve(n=60)


def bug4_additive_backadj_crosses_zero() -> pl.DataFrame:
    """Symptom of 008 Bug 4 (additive back-adjustment crosses zero over a
    long history): ~200 rolls accumulate an offset that sends early prices
    negative, manufacturing a >100% single-day 'return' that is pure splice
    artifact. Modelled directly: one day's log_return blows past the 0.5
    threshold, exactly as `log_return_backadj` would if used in place of
    `log_return_ratioadj`."""
    curve = _clean_synthetic_curve()
    ret = curve["log_return"].to_numpy().copy()
    ret[200] = 1.35  # log(4) ~= a >100% single-day move
    curve = curve.with_columns(pl.Series("log_return", ret))
    setattr(curve, H.PROVENANCE_ATTR, H.PROVENANCE_VALUE)
    return curve


BUG_SCENARIOS = {
    "bug1_contamination": bug1_contamination_reaches_front_month,
    "bug2_stale_quote": bug2_stale_quote_not_screened,
    "bug3_near_dead_months": bug3_near_dead_contract_months,
    "bug4_backadj_crosses_zero": bug4_additive_backadj_crosses_zero,
}


def main() -> None:
    results: dict[str, Any] = {"bug_rejections": {}, "clean_frame_check": {}}

    n_rejected = 0
    for name, build in BUG_SCENARIOS.items():
        curve = build()
        try:
            H.assert_risk_inputs(curve)
            results["bug_rejections"][name] = {"rejected": False, "error": None}
            print(f"{name}: NOT REJECTED (gate failure)", flush=True)
        except H.RiskInputError as e:
            n_rejected += 1
            results["bug_rejections"][name] = {"rejected": True, "error": str(e)}
            print(f"{name}: rejected ({e})", flush=True)

    n_false_rejections = 0
    for p in PRODUCTS:
        df = pl.read_parquet(f"{CURVE_DIR}/{p}.parquet")
        curve = df.with_columns(pl.col("log_return_ratioadj").alias("log_return"))
        setattr(curve, H.PROVENANCE_ATTR, H.PROVENANCE_VALUE)
        try:
            H.assert_risk_inputs(curve)
            results["clean_frame_check"][p] = {"accepted": True, "error": None}
        except H.RiskInputError as e:
            n_false_rejections += 1
            results["clean_frame_check"][p] = {"accepted": False, "error": str(e)}
            print(f"{p}: FALSE REJECTION ({e})", flush=True)

    results["gate_DC"] = {
        "n_bugs_rejected": n_rejected,
        "n_bugs_total": len(BUG_SCENARIOS),
        "n_false_rejections": n_false_rejections,
        "n_clean_products": len(PRODUCTS),
        "fires": n_rejected == len(BUG_SCENARIOS) and n_false_rejections == 0,
    }
    print(f"\ngate DC: {results['gate_DC']}", flush=True)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"written {OUT_PATH}")


if __name__ == "__main__":
    main()
