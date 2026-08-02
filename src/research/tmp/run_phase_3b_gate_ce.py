"""Phase 3b: Gate CE formal test -- Acerbi-Szekely bootstrap p-values for the
normal-innovation GARCH model, both tails, at 1% and 2.5%, BH-adjusted across
the 16 products. Phase 3's own run recorded the Z-statistics but not the
bootstrap p-values (a lighter, targeted follow-up rather than adding this to
the already ~80-minute Phase 3 battery).

Gate CE: Acerbi-Szekely rejects at 5% (BH-adjusted) for normal-innovation
models in >=14 of 16 products at the 1% level, both tails.

Writes phase_3b_gate_ce_results.json.
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

import research

CURVE_DIR = "src/research/tmp/phase_0_curves"
OUT_PATH = "src/research/tmp/phase_3b_gate_ce_results.json"
DEV_START = {"__default__": "2010-06-06", "ES": "2018-01-01", "KE": "2013-12-16"}
DEV_END = "2024-12-31"
MIN_TRAIN = 750
REFIT_EVERY = 252
MAX_TRAIN = 2000
N_BOOT = 300

research.set_seed(0)


def load_ret(product: str) -> np.ndarray:
    curve = pl.read_parquet(f"{CURVE_DIR}/{product}.parquet")
    dev_start = DEV_START.get(product, DEV_START["__default__"])
    sub = curve.select(["date", pl.col("log_return_ratioadj").alias("ret")]).drop_nulls()
    sub = sub.filter(pl.col("ret").is_finite())
    sub = sub.filter((pl.col("date") >= pl.lit(dev_start).str.to_date()) & (pl.col("date") <= pl.lit(DEV_END).str.to_date()))
    return sub["ret"].to_numpy()


def make_upper_normal_acerbi_simulate_fn(sigma: np.ndarray, es_forecast_upper: np.ndarray, q: float):
    """Upper-tail mirror of dist_lib5.make_normal_acerbi_simulate_fn: reflect
    both the simulated draws and the ES forecast, run the same lower-tail
    machinery on the reflected series (upper-tail test on X == lower-tail
    test on -X)."""
    from scipy import stats as st

    def simulate(rng):
        u = rng.uniform(0.0, 1.0, len(sigma))
        sim_values = -sigma * st.norm.ppf(u)  # reflected
        return L5._simulate_z_from_uniforms(u, sim_values, -es_forecast_upper, q)

    return simulate


def process_product(product: str) -> dict:
    ret = load_ret(product)
    fc, _fits = L.rolling_garch_forecast(ret, refit_every=REFIT_EVERY, min_train=MIN_TRAIN, innovation="normal", max_train=MAX_TRAIN)
    sigma = np.sqrt(np.where(fc > 0, fc, np.nan))

    out = {}
    for q in [0.01, 0.025]:
        var_lo = L5.normal_quantile_forecasts(fc, quantiles=[q])[q]
        es_lo = L5.normal_es_forecast(fc, q)
        z_lo = L5.acerbi_szekely_z(ret, var_lo, es_lo, q)
        sim_lo = L5.make_normal_acerbi_simulate_fn(sigma, es_lo, q)
        p_lo = L5.acerbi_szekely_bootstrap_pvalue(z_lo, sim_lo, n_boot=N_BOOT, seed=0)

        var_hi = L5.normal_quantile_forecasts(fc, quantiles=[1 - q])[1 - q]
        es_hi = -es_lo  # symmetric normal: upper ES mirrors lower ES
        z_hi = L5.acerbi_szekely_z(-ret, -var_hi, -es_hi, q)
        sim_hi = make_upper_normal_acerbi_simulate_fn(sigma, es_hi, q)
        p_hi = L5.acerbi_szekely_bootstrap_pvalue(z_hi, sim_hi, n_boot=N_BOOT, seed=1)

        out[str(q)] = {"lower": {"z": z_lo, "p": p_lo}, "upper": {"z": z_hi, "p": p_hi}}
    return out


def main():
    t0 = time.time()
    results = {}
    for p in C.PRODUCTS:
        t1 = time.time()
        print(f"processing {p}...", flush=True)
        results[p] = process_product(p)
        print(f"  {p} done in {time.time()-t1:.1f}s", flush=True)

    # BH correction across the 16 products, separately per (level, tail)
    for level in ["0.01", "0.025"]:
        for tail in ["lower", "upper"]:
            pvals = {p: results[p][level][tail]["p"] for p in results}
            bh = L5.benjamini_hochberg(pvals, alpha=0.05)
            for p in results:
                results[p][level][tail]["bh"] = bh[p]

    n_reject_1pct_both_tails = sum(
        1 for p in results
        if results[p]["0.01"]["lower"]["bh"]["significant"] and results[p]["0.01"]["upper"]["bh"]["significant"]
    )
    gate_ce = {
        "n_products_rejecting_1pct_both_tails_bh": n_reject_1pct_both_tails,
        "n_products_total": len(results),
        "threshold": 14,
        "fires": n_reject_1pct_both_tails >= 14,
    }

    out = {"per_product": results, "gate_CE": gate_ce, "_config": {"n_boot": N_BOOT, "model": "garch_normal"}}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nwritten {OUT_PATH} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
