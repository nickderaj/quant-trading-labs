"""Phase 8: the holdout, spent exactly once (NEXT_PROMPT.md sec 4, Phase 8).

2025-01-01 -> 2026-07-28 is frozen from the first line of code and touched
here for the first and only time, because at least one gate fired (CT, CE,
and RE all fired -- see the results MD for the full gate table). Nothing
here is re-tuned on the holdout: every model was already fit on the
development window (2010-06-06..2024-12-31) by Phase 3/7; this script only
*evaluates* those frozen fits on data they have never seen.

- CT holdout check: Hill tail index computed fresh on holdout returns,
  compared qualitatively to the development-window estimate.
- CE holdout check: the development-fit GARCH-normal model's frozen
  variance-recursion parameters, continued forward through the holdout,
  scored via Acerbi-Szekely (bootstrap p-value) on holdout hits only.
- RE holdout check: Phase 7's already-fit RiskModel (development window),
  evaluated for 1% VaR coverage on holdout returns via a causal EWMA
  volatility path computed over the full (dev+holdout) history -- causal
  throughout, so this uses no holdout information to condition itself.

Writes phase_8_holdout_results.json.
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
OUT_PATH = "src/research/tmp/phase_8_holdout_results.json"
PHASE7_PATH = "src/research/tmp/phase_7_results.json"

DEV_START = {"__default__": "2010-06-06", "ES": "2018-01-01", "KE": "2013-12-16"}
DEV_END = "2024-12-31"
HOLDOUT_START = "2025-01-01"
HOLDOUT_END = "2026-07-28"

research.set_seed(0)


def load_full_series(product: str) -> tuple[np.ndarray, np.ndarray]:
    curve = pl.read_parquet(f"{CURVE_DIR}/{product}.parquet")
    dev_start = DEV_START.get(product, DEV_START["__default__"])
    sub = curve.select(
        ["date", pl.col("log_return_ratioadj").alias("ret")]
    ).drop_nulls()
    sub = sub.filter(pl.col("ret").is_finite())
    sub = sub.filter(
        (pl.col("date") >= pl.lit(dev_start).str.to_date())
        & (pl.col("date") <= pl.lit(HOLDOUT_END).str.to_date())
    )
    sub = sub.sort("date")
    return sub["ret"].to_numpy(), sub["date"].to_numpy()


def ct_holdout_check(ret: np.ndarray, dates: np.ndarray) -> dict:
    holdout_mask = dates >= np.datetime64(HOLDOUT_START)
    r_hold = ret[holdout_mask]
    if len(r_hold) < 100:
        return {"note": "insufficient holdout observations"}
    out = {}
    for tail in ["upper", "lower"]:
        path = L5.hill_alpha_path(
            r_hold, tail=tail, k_min=15, k_max=min(150, len(r_hold) // 3)
        )
        plateau = L5.find_hill_plateau(path["alpha"], path["k"], window=20)
        out[tail] = plateau
    return out


def ce_holdout_check(ret: np.ndarray, dates: np.ndarray) -> dict:
    """Frozen dev-fit GARCH-normal, continued forward through holdout with NO
    refitting, scored on holdout hits only."""
    dev_mask = dates < np.datetime64(HOLDOUT_START)
    r_dev = ret[dev_mask]
    fit = L.fit_garch11(r_dev, innovation="normal")
    if fit is None:
        return {"note": "dev-window GARCH fit failed"}
    omega, alpha, beta = fit["omega"], fit["alpha"], fit["beta"]
    uncond = omega / max(1 - alpha - beta, 1e-6)
    sig2_full = L._garch_variance_path(omega, alpha, beta, ret, uncond)
    holdout_mask = dates >= np.datetime64(HOLDOUT_START)
    fc_hold = sig2_full[holdout_mask]
    r_hold = ret[holdout_mask]

    out = {}
    for q in [0.01, 0.025]:
        var_q = L5.normal_quantile_forecasts(fc_hold, quantiles=[q])[q]
        es_q = L5.normal_es_forecast(fc_hold, q)
        z = L5.acerbi_szekely_z(r_hold, var_q, es_q, q)
        sigma_hold = np.sqrt(np.where(fc_hold > 0, fc_hold, np.nan))
        sim = L5.make_normal_acerbi_simulate_fn(sigma_hold, es_q, q)
        p = L5.acerbi_szekely_bootstrap_pvalue(z, sim, n_boot=300, seed=0)
        out[str(q)] = {"z": z, "p": p, "n": int(holdout_mask.sum())}
    return out


def re_holdout_check(
    ret: np.ndarray, dates: np.ndarray, family: str, product: str
) -> dict:
    dev_mask = dates < np.datetime64(HOLDOUT_START)
    model = C.fit_risk_model(ret[dev_mask], product, family)
    if model is None:
        return {"note": "dev-window risk model fit failed"}
    sigma_path = C.ewma_vol(ret, lam=0.94)  # causal over full history
    holdout_mask = dates >= np.datetime64(HOLDOUT_START)
    r_hold = ret[holdout_mask]
    sigma_hold = sigma_path[holdout_mask]
    valid = np.isfinite(sigma_hold)
    hits = np.array(
        [
            float(r_hold[i] < -model.var_conditional(0.01, sigma_t=sigma_hold[i]))
            for i in range(len(r_hold))
            if valid[i]
        ]
    )
    if len(hits) < 30:
        return {"note": "insufficient holdout observations"}
    _, kp = dist.kupiec_test(hits.astype(int), 0.01)
    _, ip = dist.christoffersen_independence_test(hits.astype(int))
    return {
        "n": len(hits),
        "observed_rate": float(np.mean(hits)),
        "kupiec_p": float(kp),
        "christoffersen_independence_p": float(ip),
        "pass": bool(kp > 0.05 and ip > 0.05),
    }


def main():
    t0 = time.time()
    with open(PHASE7_PATH) as f:
        phase7 = json.load(f)
    families = phase7["family_map"]

    results = {}
    for p in C.PRODUCTS:
        if p not in families:
            continue
        print(f"processing {p}...", flush=True)
        ret, dates = load_full_series(p)
        results[p] = {
            "CT_holdout": ct_holdout_check(ret, dates),
            "CE_holdout": ce_holdout_check(ret, dates),
            "RE_holdout": re_holdout_check(ret, dates, families[p], p),
        }

    n_re_pass = sum(1 for r in results.values() if r["RE_holdout"].get("pass"))
    n_ce_reject_01 = sum(
        1
        for r in results.values()
        if r["CE_holdout"].get("0.01", {}).get("p") is not None
        and r["CE_holdout"]["0.01"]["p"] < 0.05
    )
    summary = {
        "n_products": len(results),
        "RE_holdout_pass_count": n_re_pass,
        "CE_holdout_reject_1pct_count": n_ce_reject_01,
    }

    out = {
        "per_product": results,
        "summary": summary,
        "_config": {"holdout_start": HOLDOUT_START, "holdout_end": HOLDOUT_END},
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nwritten {OUT_PATH} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
