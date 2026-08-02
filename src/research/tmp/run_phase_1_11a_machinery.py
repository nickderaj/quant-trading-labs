"""11a Phase 1: port and sanity-check the primitives (NEXT_PROMPT.md sec 3
Phase 1). Descriptive only -- no gate verdict. The full correctness proof
for these primitives lives in tests/test_spread_lib11.py; this script
exercises them once each on real spread data as a smoke check and reports
the numbers that feed the sec 0.2 half-life corroboration table (already
computed in Phase 0 from phase_2_10a_results.json -- restated here as the
Phase-1-machinery view of the same numbers, via this module's own
`rolling_half_life`/`rolling_stability`, as an independent cross-check of
`research_lib9.ols_ar1_diff` reused inside `rolling_stability`).

Writes phase_1_11a_results.json.
"""

import json
import sys

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import numpy as np
import polars as pl
import spread_lib11 as S11

SPREAD_DIR = "src/research/data/market/spreads"
OUT_PATH = "src/research/tmp/phase_1_11a_results.json"
DEV_END = "2024-12-31"

CHECK_SPREADS = [
    "brent_calendar",
    "brent_wti",
    "corn_wheat",
    "bean_corn",
    "kc_chicago_wheat",
]


def main() -> None:
    checks: dict[str, dict] = {}
    for name in CHECK_SPREADS:
        df = pl.read_parquet(f"{SPREAD_DIR}/{name}.parquet")
        df = df.filter(pl.col("date") <= pl.lit(DEV_END).str.to_date()).sort("date")
        clean = df.filter(~pl.col("roll_window_flag"))
        value = clean["value"].to_numpy()
        value = value[np.isfinite(value)]

        z = S11.compute_zscore(value, 60)
        atr = S11.compute_atr_series(value, 14)
        stability = S11.rolling_stability(value)
        hurst = S11.hurst_exponent(value)
        vr5 = S11.variance_ratio(value, 5)

        leg2_med = float(np.median(clean["leg2_price"].to_numpy()))
        full_carry = S11.compute_carry_fv(
            leg2_med, S11.STORAGE_MID, S11.FINANCING_RATE_APPROX
        )
        value_med = float(np.median(value))
        c_ratio = S11.carry_ratio(value_med, full_carry)
        ts_labels = S11.label_ts_regime(value, flat_band=0.05)
        n_backward = int((ts_labels == "backwardation").sum())
        n_contango = int((ts_labels == "contango").sum())

        checks[name] = {
            "n_obs": len(value),
            "zscore_no_lookahead_shift_applied": bool(np.isnan(z[0])),
            "n_finite_zscore": int(np.isfinite(z).sum()),
            "n_finite_atr": int(np.isfinite(atr).sum()),
            "rolling_stability": stability,
            "hurst_exponent": hurst,
            "variance_ratio_q5": vr5,
            "full_carry_mid": full_carry,
            "carry_ratio_at_median_value": c_ratio,
            "label_ts_regime_n_backwardation": n_backward,
            "label_ts_regime_n_contango": n_contango,
        }

    out = {
        "per_spread_checks": checks,
        "_note": (
            "Full-sample half-life here (rolling_stability.full_sample_half_life) is expected "
            "to match phase_2_10a_results.json's ar1_mean_reversion.half_life_days for the same "
            "spread -- both call research_lib9.ols_ar1_diff on the same roll-window-excluded "
            "value series; this script's smoke check confirms the reimplemented Phase-1 wiring "
            "(rolling_stability) doesn't silently diverge from 10a's own AR(1) call."
        ),
        "_dev_window_end": DEV_END,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)

    for name in CHECK_SPREADS:
        hl = checks[name]["rolling_stability"]["full_sample_half_life"]
        print(
            f"Phase 1 {name}: half_life={hl:.1f}d hurst={checks[name]['hurst_exponent']:.3f}",
            end="  ",
        )
    print()


if __name__ == "__main__":
    main()
