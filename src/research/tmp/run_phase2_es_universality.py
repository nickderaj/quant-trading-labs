"""Phase 2 driver: is the ES-underestimation finding universal?

Full grid: every model (d0-d9, 10 total - d8/d9 GARCH-EVT/GJR-EVT included
here even though they sat out the Phase 1/3 log-score contest, because their
quantile/ES forecasts are well-defined from their GPD fits, exactly as in
notebook 5's own Phase 4) x every interval (1h/4h/12h/1d) x every quantile
level (1%/2.5%/5% primary lower tail, 95%/97.5%/99% secondary upper tail) x
every symbol. Computes the Acerbi-Szekely Z statistic and its bootstrap
p-value via dist_lib5.acerbi_szekely_z / acerbi_szekely_bootstrap_pvalue
(already sign-checked in notebook 5 - see docs/06-scoring-rules-and-
calibration.md#acerbi-székely, not re-derived here).

Run per-symbol: `python run_phase2_es_universality.py SYMBOL`. Writes
`phase2_es_universality_{symbol}.json`, one file per symbol so subagents
fanned out by symbol never write-conflict (same convention as Phase 1's
driver). The orchestrator merges all six afterwards and applies Gate U /
Gate U-fat - never a subagent's job.

Power caveat (mandatory, NEXT_RUN_PROMPT.md section 4 Phase 2): any cell
with under 10 violations is marked underpowered here and must be excluded
from Gate U's primary percentages downstream.
"""

import sys
import time
import json

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import numpy as np  # noqa: E402

import dist_lib as L  # noqa: E402
import dist_lib5 as L5  # noqa: E402
import dist_lib6 as L6  # noqa: E402
import distributions as dist  # noqa: E402
import research  # noqa: E402

ES_BOOTSTRAP_N = 200
LEVELS = L5.QUANTILES  # [0.01, 0.025, 0.05, 0.95, 0.975, 0.99]
INTERVAL_ORDER = ["12h", "4h", "1d", "1h"]  # cheapest-ish first; 1h last (most expensive)


def run_one_interval(symbol: str, interval: str) -> dict:
    t0 = time.time()
    df = L.build_asset_frame(symbol, interval, end=research.HOLDOUT_START)
    n = len(df)
    ret = df["log_return"].fill_null(0.0).to_numpy()

    variance_fc, nu_paths, fits = L6.build_gate_a_forecasts(df, interval, ret)

    gpd_paths_d8, _ = L5.rolling_gpd_paths(ret, fits["d4"], model="garch", max_train=L6.MLE_MAX_TRAIN, tail_frac=0.10)
    gpd_paths_d9, _ = L5.rolling_gpd_paths(ret, fits["d6"], model="gjr", max_train=L6.MLE_MAX_TRAIN, tail_frac=0.10)

    all_models = dict(variance_fc)
    all_models["d8_garch_evt"] = variance_fc["d4_garch_normal"]  # sigma path only; tail comes from gpd_paths_d8
    all_models["d9_gjr_evt"] = variance_fc["d6_gjr_normal"]

    sigma = {name: np.sqrt(np.where(fc > 0, fc, np.nan)) for name, fc in all_models.items()}

    cells = {}
    for name in all_models:
        for q in LEVELS:
            fc = all_models[name]
            if name == "d5_garch_t":
                qf = L5.t_quantile_forecasts(fc, nu_paths["d5_garch_t"], [q])[q]
                es = L5.t_es_forecast(fc, nu_paths["d5_garch_t"], q)
                sim_fn = L5.make_t_acerbi_simulate_fn(sigma[name], nu_paths["d5_garch_t"], es, q)
            elif name == "d7_gjr_t":
                qf = L5.t_quantile_forecasts(fc, nu_paths["d7_gjr_t"], [q])[q]
                es = L5.t_es_forecast(fc, nu_paths["d7_gjr_t"], q)
                sim_fn = L5.make_t_acerbi_simulate_fn(sigma[name], nu_paths["d7_gjr_t"], es, q)
            elif name == "d8_garch_evt":
                qf = L5.gpd_quantile_forecasts(fc, gpd_paths_d8, [q])[q]
                es = L5.gpd_es_forecast(fc, gpd_paths_d8, q)
                sim_fn = L5.make_gpd_acerbi_simulate_fn(sigma[name], gpd_paths_d8, es, q)
            elif name == "d9_gjr_evt":
                qf = L5.gpd_quantile_forecasts(fc, gpd_paths_d9, [q])[q]
                es = L5.gpd_es_forecast(fc, gpd_paths_d9, q)
                sim_fn = L5.make_gpd_acerbi_simulate_fn(sigma[name], gpd_paths_d9, es, q)
            else:
                qf = L5.normal_quantile_forecasts(fc, [q])[q]
                es = L5.normal_es_forecast(fc, q)
                sim_fn = L5.make_normal_acerbi_simulate_fn(sigma[name], es, q)

            side = "lower" if q < 0.5 else "upper"
            mask = np.isfinite(ret) & np.isfinite(qf)
            hits = dist.exceedances(ret[mask], qf[mask], side=side)
            n_violations = int(hits.sum())

            z = L5.acerbi_szekely_z(ret, qf, es, q)
            p = L5.acerbi_szekely_bootstrap_pvalue(z, sim_fn, n_boot=ES_BOOTSTRAP_N, seed=0) if np.isfinite(z) else float("nan")

            cells[f"{name}__{q}"] = {
                "model": name, "level": q, "z": z, "bootstrap_pvalue": p,
                "n": int(mask.sum()), "n_violations": n_violations,
                "underpowered": n_violations < 10,
            }

    return {"n_obs": n, "cells": cells, "elapsed_sec": time.time() - t0}


def main():
    if len(sys.argv) < 2:
        print("usage: run_phase2_es_universality.py SYMBOL [interval ...]")
        sys.exit(1)
    symbol = sys.argv[1]
    intervals = sys.argv[2:] if len(sys.argv) > 2 else INTERVAL_ORDER

    out = {"symbol": symbol, "intervals": {}}
    for interval in intervals:
        res = run_one_interval(symbol, interval)
        out["intervals"][interval] = res
        n_underpowered = sum(1 for c in res["cells"].values() if c["underpowered"])
        print(f"{symbol} {interval}: n={res['n_obs']} elapsed={res['elapsed_sec']:.1f}s "
              f"cells={len(res['cells'])} underpowered={n_underpowered}")
        out_path = f"src/research/tmp/phase2_es_universality_{symbol}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=1, default=float)
        print(f"  (incremental write to {out_path})")


if __name__ == "__main__":
    main()
