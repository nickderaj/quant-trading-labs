"""Phase 4: conditional tails and the inventory story (NEXT_PROMPT.md sec 4,
Phase 4) -- the part crypto could not do.

Every product's tail behaviour is conditioned on three state variables:
- term-structure state (backwardation/contango, from the F1/F2 slope)
- seasonal state (NG heating season; grain planting/growing/harvest)
- macro/vol regime (VIX terciles, T10Y2Y sign, DFF terciles, all lagged)

Uses a single GARCH-t rolling VaR (dist_lib.rolling_garch_forecast,
innovation="t") as the reference model -- Phase 3's own per-product "best"
model differs by product and its full per-bar forecast isn't persisted to
JSON (too large), so Phase 4 uses one consistent, cheap, well-understood
baseline for every product rather than 13 separately-refit models. This is
a deliberate compute/scope tradeoff, stated here rather than left implicit.

Deliverable: tail index and 1% ES by state (with bootstrap CIs), and the
explicit Gate CI test -- does state-conditioned 1% VaR coverage beat
unconditional coverage?

Writes phase_4_results.json.
"""

import json
import sys
import time

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import commod_lib8 as C
import dist_lib as L
import dist_lib5 as L5
import numpy as np
import polars as pl

import distributions as dist
import research

CURVE_DIR = "src/research/tmp/phase_0_curves"
OUT_PATH = "src/research/tmp/phase_4_results.json"
FRED_DIR = "src/research/data/market/fred"

DEV_START = {"__default__": "2010-06-06", "ES": "2018-01-01", "KE": "2013-12-16"}
DEV_END = "2024-12-31"
MIN_TRAIN = 750
REFIT_EVERY = 63
MAX_TRAIN = 2000

research.set_seed(0)


def load_fred() -> dict[str, pl.DataFrame]:
    return {
        "VIXCLS": pl.read_parquet(f"{FRED_DIR}/VIXCLS.parquet"),
        "T10Y2Y": pl.read_parquet(f"{FRED_DIR}/T10Y2Y.parquet"),
        "DFF": pl.read_parquet(f"{FRED_DIR}/DFF.parquet"),
    }


def hill_by_state(ret: np.ndarray, states: np.ndarray) -> dict:
    out = {}
    for state in sorted(set(states.tolist())):
        mask = states == state
        r = ret[mask]
        if len(r) < 200:
            out[state] = {"n": len(r), "note": "insufficient observations"}
            continue
        path_lo = L5.hill_alpha_path(r, tail="lower", k_min=20)
        path_hi = L5.hill_alpha_path(r, tail="upper", k_min=20)
        plateau_lo = L5.find_hill_plateau(path_lo["alpha"], path_lo["k"])
        plateau_hi = L5.find_hill_plateau(path_hi["alpha"], path_hi["k"])
        es_1pct = float(np.mean(np.sort(r)[: max(1, int(0.01 * len(r)))]))
        ci_lo, ci_hi = research.block_bootstrap_ci(r[r < np.percentile(r, 5)], n_boot=1000, seed=0) if (r < np.percentile(r, 5)).sum() > 20 else (None, None)
        out[state] = {
            "n": len(r),
            "hill_left": plateau_lo, "hill_right": plateau_hi,
            "es_1pct": es_1pct, "es_1pct_ci": [ci_lo, ci_hi],
        }
    return out


def process_product(product: str, fred: dict) -> dict | None:
    curve = pl.read_parquet(f"{CURVE_DIR}/{product}.parquet")
    dev_start = DEV_START.get(product, DEV_START["__default__"])
    sub = curve.filter(pl.col("log_return_ratioadj").is_finite())
    sub = sub.filter((pl.col("date") >= pl.lit(dev_start).str.to_date()) & (pl.col("date") <= pl.lit(DEV_END).str.to_date()))
    if sub.height < MIN_TRAIN + REFIT_EVERY:
        print(f"  {product}: too few obs, skipping")
        return None

    ret = sub["log_return_ratioadj"].to_numpy()
    dates = sub["date"]
    n = len(ret)

    ts_state = C.term_structure_state(sub.select(["date", "close_f1", "dte_f1", "close_f2", "dte_f2"]))
    seasonal = C.seasonal_state(dates.to_list(), product)
    macro = C.macro_regime(fred, dates)

    # reference VaR model: GARCH-t, rolling
    fc, fits = L.rolling_garch_forecast(ret, refit_every=REFIT_EVERY, min_train=MIN_TRAIN, innovation="t", max_train=MAX_TRAIN)
    nu_path = L.nu_path_from_fits(fits, n, param_index=3)
    var_01 = L5.t_quantile_forecasts(fc, nu_path, quantiles=[0.01])[0.01]
    mask = np.isfinite(ret) & np.isfinite(var_01)
    hits = dist.exceedances(ret[mask], var_01[mask], side="lower").astype(int)

    ts_arr = np.array(["na" if v is None else v for v in ts_state["term_structure_state"].to_list()])[mask]
    seasonal_arr = np.array(seasonal)[mask]
    vix_arr = np.array(["na" if v is None else v for v in macro["vix_regime"].to_list()])[mask]
    yc_arr = np.array(["na" if v is None else v for v in macro["yield_curve_regime"].to_list()])[mask]

    kupiec_ts = C.kupiec_by_state(hits, ts_arr, expected_rate=0.01)
    kupiec_season = C.kupiec_by_state(hits, seasonal_arr, expected_rate=0.01) if product in C._SEASONAL_WINDOWS else None
    kupiec_vix = C.kupiec_by_state(hits, vix_arr, expected_rate=0.01)
    kupiec_yc = C.kupiec_by_state(hits, yc_arr, expected_rate=0.01)

    hill_ts = hill_by_state(ret, np.array(["na" if v is None else v for v in ts_state["term_structure_state"].to_list()]))

    return {
        "n_obs": n,
        "term_structure": {"kupiec": kupiec_ts, "hill_and_es_by_state": hill_ts},
        "seasonal": {"kupiec": kupiec_season} if kupiec_season else None,
        "vix_regime": {"kupiec": kupiec_vix},
        "yield_curve_regime": {"kupiec": kupiec_yc},
    }


def gate_ci_verdict(results: dict) -> dict:
    """Gate CI: state-conditioned 1% VaR beats unconditional on coverage in
    >=10 of 16 products, with no product significantly worse. Operationalised
    strictly: pooled (unconditional) coverage PASSES (kupiec_p > 0.05) AND at
    least one conditioning state reveals a miscalibration the pooled test
    missed (kupiec_p < 0.05 in that state). A product where the pooled test
    already fails is not evidence for this gate -- that is Phase 3's own
    calibration finding, not a demonstration that *conditioning* adds
    information beyond the unconditional model.
    """
    n_products_with_signal = 0
    details = {}
    for p, r in results.items():
        if p.startswith("_") or r is None:
            continue
        pooled = r["term_structure"]["kupiec"].get("_pooled", {})
        pooled_p = pooled.get("kupiec_p")
        state_ps = [v["kupiec_p"] for k, v in r["term_structure"]["kupiec"].items() if k != "_pooled" and v.get("kupiec_p") is not None]
        fires_here = pooled_p is not None and pooled_p > 0.05 and any(sp < 0.05 for sp in state_ps)
        if fires_here:
            n_products_with_signal += 1
        details[p] = {"pooled_p": pooled_p, "state_ps": state_ps, "conditioning_adds_information": fires_here}
    return {
        "n_products_with_signal": n_products_with_signal,
        "n_products_total": len([p for p in results if not p.startswith("_") and results[p] is not None]),
        "threshold": 10,
        "fires": n_products_with_signal >= 10,
        "details": details,
    }


def main():
    t0 = time.time()
    fred = load_fred()
    results: dict = {}
    for p in C.PRODUCTS:
        t1 = time.time()
        print(f"processing {p}...", flush=True)
        out = process_product(p, fred)
        if out is not None:
            results[p] = out
        print(f"  {p} done in {time.time()-t1:.1f}s", flush=True)

    results["_gate_CI"] = gate_ci_verdict(results)
    results["_config"] = {"min_train": MIN_TRAIN, "refit_every": REFIT_EVERY, "max_train": MAX_TRAIN, "reference_model": "garch_t"}

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwritten {OUT_PATH} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
