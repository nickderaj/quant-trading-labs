"""Phase 4 driver: does the violation process look like i.i.d. Bernoulli, or
does it cluster in a way Kupiec/Christoffersen can't see?

Every coverage test in notebook 5 treats VaR violations as a binary sequence
and checks a first-order-Markov alternative at most (Christoffersen
independence). This driver models the violation process directly, two ways,
at the 1% level (the level this whole research programme has centred on):

4a. Counts: bin the 1% violation indicator into weekly calendar blocks
    (`block_size = 7 * bars_per_day`, fixed per NEXT_RUN_PROMPT.md's own
    instruction to state the choice and stick to it) and fit Poisson (the
    i.i.d. null) vs. negative binomial (MLE, not method-of-moments - see
    dist_lib6.fit_nb_counts's own docstring) via a boundary-corrected LR test
    (dist_lib6.boundary_lr_test - the 50:50 chi2_0/chi2_1 mixture, not a
    plain chi2_1, documented in docs/03-statistical-inference.md).

4b. Durations: gaps between consecutive violations. Geometric (the discrete
    i.i.d. null) vs. discrete Weibull (beta<1 = clustering, a plain chi2_1
    LR test since beta=1 is an interior, not boundary, point).

Fit-once, descriptive (same status as notebook 5's own Hill estimator) -
not a rolling-refit forecast input. Run per-symbol, all 4 intervals x 10
models per symbol, writing phase4_violation_{symbol}.json.
"""

import json
import sys
import time
from typing import Any

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import dist_lib as L
import dist_lib5 as L5
import dist_lib6 as L6
import numpy as np

import distributions as dist
import research

Q = 0.01  # the level this research programme centres on
INTERVAL_ORDER = ["12h", "4h", "1d", "1h"]


def run_one_interval(symbol: str, interval: str) -> dict:
    t0 = time.time()
    df = L.build_asset_frame(symbol, interval, end=research.HOLDOUT_START)
    n = len(df)
    ret = df["log_return"].fill_null(0.0).to_numpy()
    bpd = L6.BARS_PER_DAY[interval]
    block_size = 7 * bpd

    variance_fc, nu_paths, fits = L6.build_gate_a_forecasts(df, interval, ret)
    gpd_paths_d8, _ = L5.rolling_gpd_paths(
        ret, fits["d4"], model="garch", max_train=L6.MLE_MAX_TRAIN, tail_frac=0.10
    )
    gpd_paths_d9, _ = L5.rolling_gpd_paths(
        ret, fits["d6"], model="gjr", max_train=L6.MLE_MAX_TRAIN, tail_frac=0.10
    )

    all_fc = dict(variance_fc)
    all_fc["d8_garch_evt"] = variance_fc["d4_garch_normal"]
    all_fc["d9_gjr_evt"] = variance_fc["d6_gjr_normal"]

    results = {}
    for name, fc in all_fc.items():
        if name == "d5_garch_t":
            qf = L5.t_quantile_forecasts(fc, nu_paths["d5_garch_t"], [Q])[Q]
        elif name == "d7_gjr_t":
            qf = L5.t_quantile_forecasts(fc, nu_paths["d7_gjr_t"], [Q])[Q]
        elif name == "d8_garch_evt":
            qf = L5.gpd_quantile_forecasts(fc, gpd_paths_d8, [Q])[Q]
        elif name == "d9_gjr_evt":
            qf = L5.gpd_quantile_forecasts(fc, gpd_paths_d9, [Q])[Q]
        else:
            qf = L5.normal_quantile_forecasts(fc, [Q])[Q]

        mask = np.isfinite(ret) & np.isfinite(qf)
        hit_masked = dist.exceedances(ret[mask], qf[mask], side="lower")
        # re-expand to full length (unmasked bars contribute no violation and
        # no exposure - dropped from block/duration counting, matching the
        # coverage battery's own masking convention)
        hit_full = np.zeros(n, dtype=bool)
        hit_full[mask] = hit_masked.astype(bool)

        counts, durations = L6.violation_blocks_and_durations(
            hit_full[mask], block_size
        )

        cell: dict[str, Any] = {
            "n_valid_bars": int(mask.sum()),
            "n_violations": int(hit_masked.sum()),
            "n_blocks": len(counts),
            "n_durations": len(durations),
        }

        pois_fit = L6.fit_poisson_counts(counts) if len(counts) else None
        nb_fit = L6.fit_nb_counts(counts) if len(counts) else None
        if pois_fit is not None and nb_fit is not None:
            lr_counts, p_counts = L6.boundary_lr_test(
                nb_fit["loglik"], pois_fit["loglik"]
            )
            cell["counts"] = {
                "poisson_mu": pois_fit["mu"],
                "nb_mu": nb_fit["mu"],
                "nb_alpha": nb_fit["alpha"],
                "lr_stat": lr_counts,
                "boundary_lr_pvalue": p_counts,
                "nb_significantly_better": p_counts < 0.05,
            }
        else:
            cell["counts"] = None

        geo_fit = L6.fit_geometric_durations(durations) if len(durations) else None
        dw_fit = (
            L6.fit_discrete_weibull_durations(durations) if len(durations) else None
        )
        if geo_fit is not None and dw_fit is not None:
            lr_dur = max(0.0, 2.0 * (dw_fit["loglik"] - geo_fit["loglik"]))
            from scipy import stats as st

            p_dur = float(st.chi2.sf(lr_dur, df=1))
            cell["durations"] = {
                "geometric_q": geo_fit["q"],
                "weibull_q": dw_fit["q"],
                "weibull_beta": dw_fit["beta"],
                "lr_stat": lr_dur,
                "pvalue": p_dur,
                "weibull_significantly_better": p_dur < 0.05,
                "clusters": dw_fit["beta"] < 1.0,
            }
        else:
            cell["durations"] = None

        results[name] = cell

    return {
        "n_obs": n,
        "block_size_bars": block_size,
        "models": results,
        "elapsed_sec": time.time() - t0,
    }


def main():
    if len(sys.argv) < 2:
        print("usage: run_phase4_violation_process.py SYMBOL [interval ...]")
        sys.exit(1)
    symbol = sys.argv[1]
    intervals = sys.argv[2:] if len(sys.argv) > 2 else INTERVAL_ORDER

    out = {"symbol": symbol, "intervals": {}}
    for interval in intervals:
        res = run_one_interval(symbol, interval)
        out["intervals"][interval] = res
        n_counts_sig = sum(
            1
            for m in res["models"].values()
            if m["counts"] and m["counts"]["nb_significantly_better"]
        )
        n_dur_sig = sum(
            1
            for m in res["models"].values()
            if m["durations"] and m["durations"]["weibull_significantly_better"]
        )
        print(
            f"{symbol} {interval}: n={res['n_obs']} elapsed={res['elapsed_sec']:.1f}s "
            f"counts_sig={n_counts_sig}/10 dur_sig={n_dur_sig}/10"
        )
        out_path = f"src/research/tmp/phase4_violation_{symbol}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=1, default=float)
        print(f"  (incremental write to {out_path})")


if __name__ == "__main__":
    main()
