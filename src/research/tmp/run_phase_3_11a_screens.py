"""11a Phase 3: rebuild and calibrate both screens (NEXT_PROMPT.md sec 3
Phase 3). Descriptive only -- no gate verdict, no cost model, no Sharpe.

Old screen: ADF on 30-day deviations, run on 20 synthetic pure random walks
(must fail to reject, i.e. pass them as "not stationary" -- a screen that
CAN reject a random walk carries no information; theirs passed all 20 at
median p ~ 1e-19, i.e. it correctly failed to reject none of them as
mean-reverting).

New screen: ADF on the level + variance ratio (q=5, q=20, one-sided
z=1.645) + Hurst < 0.5 + half-life stability (full-sample half-life in
3-60d band AND >= 3/4 sub-periods also in band). Variance-ratio test
calibrated on 500 seeded random walks for its empirical false-positive
rate against its nominal 5%.

Applies both screens to all 30 of our spreads (dev window, <= 2024-12-31)
and tabulates against 10a's own ADF verdicts (phase_2_10a_results.json).

Writes phase_3_11a_results.json.
"""

import json
import sys

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import numpy as np
import polars as pl
import spread_lib10 as S10
import spread_lib11 as S11

SPREAD_DIR = "src/research/data/market/spreads"
OUT_PATH = "src/research/tmp/phase_3_11a_results.json"
DEV_END = "2024-12-31"

SPREAD_NAMES = [
    "bean_corn",
    "brent_calendar",
    "brent_wti",
    "bz_cal_m1m3",
    "cl_cal_m2m3",
    "corn_wheat",
    "crack_321",
    "crush_soy",
    "es_calendar",
    "gasheat_rbho",
    "gasoline_crack",
    "gc_cal_m1m2",
    "gc_cal_m2m3",
    "gold_silver",
    "heating_oil_crack",
    "ho_cal_m1m2",
    "ho_cal_m2m3",
    "kc_chicago_wheat",
    "ke_cal_m1m2",
    "ng_cal_m1m2",
    "ng_cal_m2m3",
    "ng_calendar",
    "platinum_palladium",
    "rb_cal_m1m2",
    "rb_cal_m2m3",
    "wti_calendar",
    "zc_cal_m1m2",
    "zl_cal_m1m2",
    "zm_cal_m1m2",
    "zw_cal_m1m2",
]

Z_CRIT_ONE_SIDED = 1.645


def old_screen_on_random_walks(
    n_rw: int = 20, n_obs: int = 2000, seed: int = 0
) -> dict:
    """Old screen: ADF on the 30-day deviation of the level from its own
    30-day rolling mean, on `n_rw` synthetic pure random walks.

    NEXT_PROMPT.md sec 3 Phase 3: "confirm it passes them" -- i.e. confirm
    this screen construction FALSELY flags pure random walks as stationary
    (theirs did, on all 20, at median p ~ 1e-19). This is the demonstrated
    flaw motivating the new screen below, not a correctness check: 30-day-
    detrending a random walk manufactures a bounded, spuriously stationary
    residual almost by construction, so a screen built this way carries no
    information about genuine mean reversion.
    """
    rng = np.random.default_rng(seed)
    results = []
    for i in range(n_rw):
        level = np.cumsum(rng.standard_normal(n_obs))
        s = pl.Series(level)
        deviation = (s - s.rolling_mean(window_size=30)).drop_nulls().to_numpy()
        adf = S10.adf_test(deviation)
        results.append(
            {"t_stat": adf["t_stat"], "stationary_5pct": adf["stationary_5pct"]}
        )
    n_falsely_stationary = sum(1 for r in results if r["stationary_5pct"])
    return {
        "n_random_walks": n_rw,
        "n_obs": n_obs,
        "median_t_stat": float(np.median([r["t_stat"] for r in results])),
        "n_falsely_flagged_stationary_5pct": n_falsely_stationary,
        "false_positive_rate": n_falsely_stationary / n_rw,
        "passes_random_walk_check": n_falsely_stationary == 0,
    }


def variance_ratio_calibration(
    n_rw: int = 500, n_obs: int = 2000, seed: int = 1
) -> dict:
    """Empirical false-positive rate of the one-sided VR(q) test (reject
    random walk if z < -1.645) against its nominal 5%, on `n_rw` seeded
    random walks, for q=5 and q=20.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for q in (5, 20):
        n_reject = 0
        for i in range(n_rw):
            level = np.cumsum(rng.standard_normal(n_obs))
            vr = S11.variance_ratio(level, q)
            if np.isfinite(vr["z_stat"]) and vr["z_stat"] < -Z_CRIT_ONE_SIDED:
                n_reject += 1
        out[f"q_{q}"] = {
            "n_random_walks": n_rw,
            "n_rejected_5pct_one_sided": n_reject,
            "empirical_false_positive_rate": n_reject / n_rw,
            "nominal_rate": 0.05,
        }
    return out


def new_screen(value: np.ndarray) -> dict:
    adf = S10.adf_test(value) if len(value) >= 60 else None
    vr5 = S11.variance_ratio(value, 5)
    vr20 = S11.variance_ratio(value, 20)
    hurst = S11.hurst_exponent(value)
    stability = S11.rolling_stability(value)
    adf_pass = bool(adf is not None and adf["stationary_5pct"])
    vr5_pass = bool(np.isfinite(vr5["z_stat"]) and vr5["z_stat"] < -Z_CRIT_ONE_SIDED)
    vr20_pass = bool(np.isfinite(vr20["z_stat"]) and vr20["z_stat"] < -Z_CRIT_ONE_SIDED)
    hurst_pass = bool(np.isfinite(hurst) and hurst < 0.5)
    stability_pass = bool(stability["stable"])
    passes = adf_pass and vr5_pass and vr20_pass and hurst_pass and stability_pass
    return {
        "adf_t_stat": adf["t_stat"] if adf else None,
        "adf_pass": adf_pass,
        "vr5_z": vr5["z_stat"],
        "vr5_pass": vr5_pass,
        "vr20_z": vr20["z_stat"],
        "vr20_pass": vr20_pass,
        "hurst": hurst,
        "hurst_pass": hurst_pass,
        "stability": stability,
        "stability_pass": stability_pass,
        "new_screen_pass": passes,
    }


def main() -> None:
    old_rw = old_screen_on_random_walks()
    vr_calibration = variance_ratio_calibration()

    with open("src/research/tmp/phase_2_10a_results.json") as f:
        d10a = json.load(f)
    per_spread_10a = d10a["per_spread"]

    per_spread: dict = {}
    for name in SPREAD_NAMES:
        df = pl.read_parquet(f"{SPREAD_DIR}/{name}.parquet")
        df = df.filter(pl.col("date") <= pl.lit(DEV_END).str.to_date()).sort("date")
        clean = df.filter(~pl.col("roll_window_flag"))
        value = clean["value"].to_numpy()
        value = value[np.isfinite(value)]

        old_pass = bool(len(value) >= 60 and S10.adf_test(value)["stationary_5pct"])
        new = new_screen(value)
        old_10a_pass = bool(
            per_spread_10a[name]["adf_cointegration"]["stationary_5pct"]
        )

        per_spread[name] = {
            "old_screen_pass": old_pass,
            "new_screen": new,
            "agrees_with_10a_adf_verdict": old_pass == old_10a_pass,
        }

    n_old_pass = sum(1 for v in per_spread.values() if v["old_screen_pass"])
    n_new_pass = sum(
        1 for v in per_spread.values() if v["new_screen"]["new_screen_pass"]
    )
    n_disagree_old_vs_10a = sum(
        1 for v in per_spread.values() if not v["agrees_with_10a_adf_verdict"]
    )

    out = {
        "old_screen_random_walk_check": old_rw,
        "variance_ratio_calibration": vr_calibration,
        "per_spread": per_spread,
        "summary": {
            "n_spreads": len(SPREAD_NAMES),
            "n_old_screen_pass": n_old_pass,
            "n_new_screen_pass": n_new_pass,
            "n_disagreements_old_vs_10a_adf": n_disagree_old_vs_10a,
        },
        "_dev_window_end": DEV_END,
        "_note": (
            "Old screen here reuses 10a's own ADF-on-level test as the old-screen "
            "baseline (10a Phase 2's sec 4.3 test IS this repo's pre-existing old "
            "screen); the random-walk check is run on synthetic 30-day-deviation "
            "series per NEXT_PROMPT.md's literal spec for the old screen, which is "
            "a stricter construction than plain level-ADF and is reported separately "
            "in old_screen_random_walk_check, not conflated with old_screen_pass above."
        ),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(
            out,
            f,
            indent=2,
            default=lambda o: bool(o) if isinstance(o, np.bool_) else o,
        )
    print(
        f"Phase 3: old_screen RW false-positive={old_rw['false_positive_rate']:.2f}, "
        f"VR5 FP={vr_calibration['q_5']['empirical_false_positive_rate']:.3f}, "
        f"VR20 FP={vr_calibration['q_20']['empirical_false_positive_rate']:.3f}, "
        f"n_old_pass={n_old_pass}/30, n_new_pass={n_new_pass}/30, "
        f"n_disagree_vs_10a={n_disagree_old_vs_10a}"
    )


if __name__ == "__main__":
    main()
