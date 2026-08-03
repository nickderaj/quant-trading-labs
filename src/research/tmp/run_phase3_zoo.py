"""Phase 3 contest driver: does any new density family beat GARCH-t?

Runs the four new innovation families (GED, NIG, Johnson SU, Hansen skew-t)
through the same GARCH(1,1) variance recursion via dist_lib6's two-stage
fit (dist_lib6.fit_garch_zoo_two_stage - see its own docstring for why
two-stage rather than joint MLE), scored into the same log-score /
all-pairs-DM / BH contest machinery as Phase 1, against GARCH-t (the
incumbent champion) plus GARCH-normal as a sanity-check floor.

5-model contest per (symbol, interval): GARCH-normal, GARCH-t, GARCH-GED,
GARCH-NIG, GARCH-JSU, GARCH-Hansen-skewt (6, actually - GARCH-normal is a
floor/sanity check, not a Gate-P competitor). Gate P only needs each zoo
family's pairwise comparison against GARCH-t, but the full all-pairs table
is computed anyway so a reader can see the whole ranking, not just the one
comparison the gate cares about.

Run per-symbol: `python run_phase3_zoo.py SYMBOL`. Writes
`phase3_zoo_{symbol}.json`.
"""

import json
import sys
import time

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import dist_lib as L
import dist_lib5 as L5
import dist_lib6 as L6
import numpy as np
from densities import REGISTRY

import research

DM_BOOTSTRAP_N = 500
INTERVAL_ORDER = ["12h", "4h", "1h", "1d"]
ZOO_FAMILIES = ["ged", "nig", "johnsonsu", "hansen_skewt"]


def run_one_interval(symbol: str, interval: str) -> dict:
    t0 = time.time()
    df = L.build_asset_frame(symbol, interval, end=research.HOLDOUT_START)
    n = len(df)
    ret = df["log_return"].fill_null(0.0).to_numpy()
    bpd = L6.BARS_PER_DAY[interval]
    mle_refit_every = L6.MLE_REFIT_DAYS * bpd
    min_train = L6.MIN_TRAIN_DAYS * bpd

    variance_fc: dict[str, np.ndarray] = {}
    log_score_full: dict[str, np.ndarray] = {}
    shape_paths: dict[str, list] = {}

    variance_fc["garch_normal"], _fits_n = L.rolling_garch_forecast(
        ret,
        refit_every=mle_refit_every,
        min_train=min_train,
        innovation="normal",
        max_train=L6.MLE_MAX_TRAIN,
    )
    log_score_full["garch_normal"] = L5.vectorized_normal_scores(
        ret, variance_fc["garch_normal"]
    )["log_score"]
    mask_n = (
        np.isfinite(ret)
        & np.isfinite(variance_fc["garch_normal"])
        & (variance_fc["garch_normal"] > 0)
    )
    ls_full = np.full(n, np.nan)
    ls_full[mask_n] = log_score_full["garch_normal"]
    log_score_full["garch_normal"] = ls_full

    variance_fc["garch_t"], fits_t = L.rolling_garch_forecast(
        ret,
        refit_every=mle_refit_every,
        min_train=min_train,
        innovation="t",
        max_train=L6.MLE_MAX_TRAIN,
    )
    nu_path = L.nu_path_from_fits(fits_t, n, param_index=3)
    res_t = L5.vectorized_t_scores(ret, variance_fc["garch_t"], nu_path)
    ls_full = np.full(n, np.nan)
    ls_full[res_t["mask"]] = res_t["log_score"]
    log_score_full["garch_t"] = ls_full

    for fam_name in ZOO_FAMILIES:
        family_module = REGISTRY[fam_name]
        key = f"garch_{fam_name}"
        fc, fits = L6.rolling_garch_forecast_zoo(
            ret,
            refit_every=mle_refit_every,
            min_train=min_train,
            family_module=family_module,
            max_train=L6.MLE_MAX_TRAIN,
        )
        variance_fc[key] = fc
        log_score_full[key] = L6.score_zoo_model(ret, fc, fits, family_module)
        shape_paths[key] = [{"t": f["t"], "shape": f["shape"]} for f in fits]

    model_names = list(variance_fc.keys())
    scores = {
        name: {
            "log_score_mean": float(np.nanmean(log_score_full[name])),
            "n": int(np.isfinite(log_score_full[name]).sum()),
        }
        for name in model_names
    }

    all_pairs_dm = L6.all_pairs_dm_bh(
        model_names, log_score_full, dm_bootstrap_n=DM_BOOTSTRAP_N, seed=0
    )

    zoo_only = [f"garch_{f}" for f in ZOO_FAMILIES]
    best_zoo = max(zoo_only, key=lambda name: scores[name]["log_score_mean"])
    gate_p_per_family = {}
    for fam_key in zoo_only:
        key = (
            f"{fam_key}_vs_garch_t"
            if f"{fam_key}_vs_garch_t" in all_pairs_dm
            else f"garch_t_vs_{fam_key}"
        )
        entry = all_pairs_dm.get(key)
        beats_garch_t = False
        significant = False
        if entry is not None:
            a_is_fam = entry["a"] == fam_key
            fam_wins = (a_is_fam and entry["tstat"] < 0) or (
                not a_is_fam and entry["tstat"] > 0
            )
            significant = entry["bh_bootstrap"]["significant"]
            beats_garch_t = (
                fam_wins
                and scores[fam_key]["log_score_mean"]
                > scores["garch_t"]["log_score_mean"]
            )
        gate_p_per_family[fam_key] = {
            "beats_garch_t_on_mean": scores[fam_key]["log_score_mean"]
            > scores["garch_t"]["log_score_mean"],
            "significantly_beats_garch_t": beats_garch_t and significant,
        }

    return {
        "n_obs": n,
        "scores": scores,
        "all_pairs_dm": all_pairs_dm,
        "best_by_log_score": max(
            model_names, key=lambda name: scores[name]["log_score_mean"]
        ),
        "best_zoo_family": best_zoo,
        "gate_p_per_family": gate_p_per_family,
        "shape_paths": shape_paths,
        "elapsed_sec": time.time() - t0,
    }


def main():
    if len(sys.argv) < 2:
        print("usage: run_phase3_zoo.py SYMBOL [interval ...]")
        sys.exit(1)
    symbol = sys.argv[1]
    intervals = sys.argv[2:] if len(sys.argv) > 2 else INTERVAL_ORDER

    out = {"symbol": symbol, "intervals": {}}
    for interval in intervals:
        res = run_one_interval(symbol, interval)
        out["intervals"][interval] = res
        print(
            f"{symbol} {interval}: n={res['n_obs']} elapsed={res['elapsed_sec']:.1f}s "
            f"best={res['best_by_log_score']} best_zoo={res['best_zoo_family']} "
            f"gate_p={ {k: v['significantly_beats_garch_t'] for k, v in res['gate_p_per_family'].items()} }"
        )
        out_path = f"src/research/tmp/phase3_zoo_{symbol}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=1, default=float)
        print(f"  (incremental write to {out_path})")


if __name__ == "__main__":
    main()
