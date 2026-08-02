"""Phase 3: conditional models and the risk battery -- the replication test
(NEXT_PROMPT.md sec 4, Phase 3). Per product: GARCH(1,1) and GJR(1,1,1),
each x {normal, t, Hansen skew-t, NIG, GED, JohnsonSU}, plus one spliced-EVT
density (paired with the GARCH-normal variance process), rolling out-of-
sample on the development window. Then:

- log score + all-pairs DM/BH -> which density wins per product, and does
  one win everywhere (the crypto-replication question: notebook 6 found
  GARCH-t >~ GARCH-NIG > skew-t ~ JohnsonSU > GED >> normal).
- the full 36-test coverage battery at {0.5%, 1%, 2.5%, 5%} both tails.
- Acerbi-Szekely ES tests at 1% and 2.5%, both tails, bootstrap p-values.
- violation-count/duration PMF fits (Poisson vs NB; geometric vs discrete-
  Weibull) + boundary_lr_test, on the best model's own violations.
- GJR sign check: is the fitted leverage parameter gamma the equity sign
  (>0, down-moves raise next-bar variance more) or has the commodity
  "inverse leverage" sign (<0)?
- HAR-log-RV reported separately (QLIKE only, not a density model).

Writes phase_3_results.json.
"""

import json
import sys
import time

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import commod_lib8 as C
import dist_lib as L
import dist_lib5 as L5
import dist_lib6 as L6
import numpy as np
import polars as pl

import distributions as dist
import research

CURVE_DIR = "src/research/tmp/phase_0_curves"
OUT_PATH = "src/research/tmp/phase_3_results.json"

DEV_START = {"__default__": "2010-06-06", "ES": "2018-01-01", "KE": "2013-12-16"}
DEV_END = "2024-12-31"

MIN_TRAIN = 750
REFIT_EVERY = 252
MAX_TRAIN = 2000
DM_BOOTSTRAP_N = 100

ZOO_FAMILIES = ["ged", "nig", "johnsonsu", "hansen_skewt"]
QUANTILES = [0.005, 0.01, 0.025, 0.05, 0.95, 0.975, 0.99, 0.995]
ES_LEVELS = [0.01, 0.025]

research.set_seed(0)


def load_ret(product: str) -> tuple[np.ndarray, np.ndarray]:
    curve = pl.read_parquet(f"{CURVE_DIR}/{product}.parquet")
    dev_start = DEV_START.get(product, DEV_START["__default__"])
    sub = curve.select(["date", pl.col("log_return_ratioadj").alias("ret")]).drop_nulls()
    sub = sub.filter(pl.col("ret").is_finite())
    sub = sub.filter((pl.col("date") >= pl.lit(dev_start).str.to_date()) & (pl.col("date") <= pl.lit(DEV_END).str.to_date()))
    return sub["ret"].to_numpy(), sub["date"].to_numpy()


def build_models(ret: np.ndarray) -> dict:
    """Returns {model_name: {"variance_fc": arr, "nu_path": arr|None, "fits": list, "log_score_full": arr}}."""
    import densities

    n = len(ret)
    models: dict = {}

    fc_g_n, fits_g_n = L.rolling_garch_forecast(ret, refit_every=REFIT_EVERY, min_train=MIN_TRAIN, innovation="normal", max_train=MAX_TRAIN)
    fc_g_t, fits_g_t = L.rolling_garch_forecast(ret, refit_every=REFIT_EVERY, min_train=MIN_TRAIN, innovation="t", max_train=MAX_TRAIN)
    nu_g_t = L.nu_path_from_fits(fits_g_t, n, param_index=3)
    fc_j_n, fits_j_n = L5.rolling_gjr_forecast(ret, refit_every=REFIT_EVERY, min_train=MIN_TRAIN, innovation="normal", max_train=MAX_TRAIN)
    fc_j_t, fits_j_t = L5.rolling_gjr_forecast(ret, refit_every=REFIT_EVERY, min_train=MIN_TRAIN, innovation="t", max_train=MAX_TRAIN)
    nu_j_t = L.nu_path_from_fits(fits_j_t, n, param_index=4)

    def normal_ls(fc):
        r = L5.vectorized_normal_scores(ret, fc)
        out = np.full(n, np.nan)
        out[r["mask"]] = r["log_score"]
        return out

    def t_ls(fc, nu):
        r = L5.vectorized_t_scores(ret, fc, nu)
        out = np.full(n, np.nan)
        out[r["mask"]] = r["log_score"]
        return out

    models["garch_normal"] = {"variance_fc": fc_g_n, "nu_path": None, "fits": fits_g_n, "log_score_full": normal_ls(fc_g_n), "kind": "garch"}
    models["garch_t"] = {"variance_fc": fc_g_t, "nu_path": nu_g_t, "fits": fits_g_t, "log_score_full": t_ls(fc_g_t, nu_g_t), "kind": "garch"}
    models["gjr_normal"] = {"variance_fc": fc_j_n, "nu_path": None, "fits": fits_j_n, "log_score_full": normal_ls(fc_j_n), "kind": "gjr"}
    models["gjr_t"] = {"variance_fc": fc_j_t, "nu_path": nu_j_t, "fits": fits_j_t, "log_score_full": t_ls(fc_j_t, nu_j_t), "kind": "gjr"}

    for fam_name in ZOO_FAMILIES:
        mod = densities.REGISTRY[fam_name]
        fc_gz, fits_gz = L6.rolling_garch_forecast_zoo(ret, refit_every=REFIT_EVERY, min_train=MIN_TRAIN, family_module=mod, max_train=MAX_TRAIN)
        models[f"garch_{fam_name}"] = {
            "variance_fc": fc_gz, "nu_path": None, "fits": fits_gz,
            "log_score_full": L6.score_zoo_model(ret, fc_gz, fits_gz, mod), "kind": "garch",
        }
        fc_jz, fits_jz = C.rolling_gjr_forecast_zoo(ret, refit_every=REFIT_EVERY, min_train=MIN_TRAIN, family_module=mod, max_train=MAX_TRAIN)
        models[f"gjr_{fam_name}"] = {
            "variance_fc": fc_jz, "nu_path": None, "fits": fits_jz,
            "log_score_full": L6.score_zoo_model(ret, fc_jz, fits_jz, mod), "kind": "gjr",
        }

    spliced_fits = L6.rolling_spliced_evt_fits(ret, fits_g_n, model="garch", max_train=MAX_TRAIN)
    models["garch_spliced_evt"] = {
        "variance_fc": fc_g_n, "nu_path": None, "fits": spliced_fits,
        "log_score_full": L6.score_spliced_evt_model(ret, fc_g_n, spliced_fits), "kind": "spliced",
    }

    return models


def quantile_and_es_forecasts(model_name: str, m: dict, ret: np.ndarray) -> tuple[dict, dict]:
    import alpha_lib7 as A
    import densities

    fc = m["variance_fc"]
    quantiles: dict = {}
    es: dict = {}
    if model_name.endswith(("_normal", "_t")):
        if model_name.endswith("_normal"):
            quantiles = L5.normal_quantile_forecasts(fc, quantiles=QUANTILES)
            es_lower_fn = lambda q: L5.normal_es_forecast(fc, q)
        else:
            quantiles = L5.t_quantile_forecasts(fc, m["nu_path"], quantiles=QUANTILES)
            es_lower_fn = lambda q: L5.t_es_forecast(fc, m["nu_path"], q)
        # normal and t are both symmetric, so the upper-tail ES at exceedance
        # probability q is exactly minus the lower-tail ES at the same q.
        for q in ES_LEVELS:
            es[q] = es_lower_fn(q)
            es[1 - q] = -es_lower_fn(q)
    elif model_name.endswith("_spliced_evt"):
        for q in QUANTILES:
            out = C.spliced_evt_var_es_forecast(fc, m["fits"], q)
            quantiles[q] = out["var"]
            es[q] = out["es"]
    else:
        fam_name = model_name.split("_", 1)[1]
        mod = densities.REGISTRY[fam_name]
        for q in QUANTILES:
            quantiles[q] = L6.zoo_quantile_forecast(fc, m["fits"], mod, q)
        for q in ES_LEVELS:
            es[q] = A.zoo_es_forecast(fc, m["fits"], mod, q)
            es[1 - q] = C.zoo_es_forecast_upper(fc, m["fits"], mod, q)
    return quantiles, es


def process_product(product: str) -> dict | None:
    ret, _dates = load_ret(product)
    n = len(ret)
    if n < MIN_TRAIN + REFIT_EVERY:
        print(f"  {product}: too few obs ({n}), skipping")
        return None

    models = build_models(ret)
    log_score_full = {name: m["log_score_full"] for name, m in models.items()}
    dm = L6.all_pairs_dm_bh(list(models.keys()), log_score_full, dm_bootstrap_n=DM_BOOTSTRAP_N)
    mean_ls = {name: (float(np.nanmean(ls)) if np.isfinite(ls).any() else None) for name, ls in log_score_full.items()}
    valid = [k for k, v in mean_ls.items() if v is not None]
    ranking = sorted(valid, key=lambda k: mean_ls[k], reverse=True)
    best = ranking[0] if ranking else None
    best_significant = L6.beats_all_significantly(best, valid, dm, "bh_bootstrap") if best else False

    coverage = {}
    acerbi = {}
    for name, m in models.items():
        quantiles, es = quantile_and_es_forecasts(name, m, ret)
        quantiles = {q: v for q, v in quantiles.items() if v is not None}
        if len(quantiles) < 4:
            continue
        coverage[name] = L5.coverage_battery(ret, quantiles)
        acerbi_entry = {}
        for q in ES_LEVELS:
            var_lo, es_lo = quantiles.get(q), es.get(q)
            if var_lo is not None and es_lo is not None:
                acerbi_entry[f"lower_{q}"] = {"z": L5.acerbi_szekely_z(ret, var_lo, es_lo, q)}
            var_hi, es_hi = quantiles.get(1 - q), es.get(1 - q)
            if var_hi is not None and es_hi is not None:
                # upper-tail test = lower-tail test on the reflected series
                z_hi = L5.acerbi_szekely_z(-ret, -var_hi, -es_hi, q)
                acerbi_entry[f"upper_{q}"] = {"z": z_hi}
        acerbi[name] = acerbi_entry

    # GJR sign check
    gjr_gammas = []
    for name in ["gjr_normal", "gjr_t", *[f"gjr_{f}" for f in ZOO_FAMILIES]]:
        for f in models[name]["fits"]:
            if "gamma" in f:
                gjr_gammas.append(f["gamma"])
    gjr_sign = {
        "mean_gamma": float(np.mean(gjr_gammas)) if gjr_gammas else None,
        "frac_positive": float(np.mean(np.array(gjr_gammas) > 0)) if gjr_gammas else None,
        "n_fits": len(gjr_gammas),
        "interpretation": "positive gamma = equity leverage sign (down-moves raise next-bar var more); negative = inverse-leverage (commodity) sign",
    }

    # Violation process on the best model
    violation_process = None
    if best is not None and best in coverage:
        q01 = coverage[best].get("0.01")
        if q01 is not None:
            quantiles, _ = quantile_and_es_forecasts(best, models[best], ret)
            var01 = quantiles.get(0.01)
            mask = np.isfinite(ret) & np.isfinite(var01)
            hits = dist.exceedances(ret[mask], var01[mask], side="lower").astype(int)
            counts, durations = L6.violation_blocks_and_durations(hits, block_size=21)
            poisson_fit = L6.fit_poisson_counts(counts)
            nb_fit = L6.fit_nb_counts(counts)
            geo_fit = L6.fit_geometric_durations(durations)
            weibull_fit = L6.fit_discrete_weibull_durations(durations)
            count_lr = L6.boundary_lr_test(nb_fit["loglik"], poisson_fit["loglik"]) if (nb_fit and poisson_fit) else (None, None)
            dur_lr_stat, dur_lr_p = None, None
            if weibull_fit and geo_fit:
                # beta=1 (geometric) is an INTERIOR point of discrete-Weibull's
                # beta>0 space, not a boundary -- plain chi2_1, no Chernoff
                # mixture correction (see fit_discrete_weibull_durations).
                from scipy import stats as st

                dur_lr_stat = float(max(0.0, 2 * (weibull_fit["loglik"] - geo_fit["loglik"])))
                dur_lr_p = float(st.chi2.sf(dur_lr_stat, df=1))
            violation_process = {
                "model": best, "n_blocks": len(counts), "n_durations": len(durations),
                "poisson": poisson_fit, "nb": nb_fit, "geometric": geo_fit, "discrete_weibull": weibull_fit,
                "count_lr_stat": count_lr[0], "count_lr_pvalue": count_lr[1],
                "duration_lr_stat": dur_lr_stat, "duration_lr_pvalue": dur_lr_p,
            }

    return {
        "n_obs": n,
        "mean_log_score": mean_ls,
        "ranking": ranking,
        "best_model": best,
        "best_wins_significantly_bh": best_significant,
        "coverage_battery": coverage,
        "acerbi_szekely": acerbi,
        "gjr_sign_check": gjr_sign,
        "violation_process": violation_process,
    }


def main():
    t0 = time.time()
    results: dict = {}
    for p in C.PRODUCTS:
        t1 = time.time()
        print(f"processing {p}...", flush=True)
        out = process_product(p)
        if out is not None:
            results[p] = out
        print(f"  {p} done in {time.time()-t1:.1f}s", flush=True)

    winners = {p: results[p]["best_model"] for p in results if results[p]["best_model"]}
    results["_summary"] = {
        "winners": winners,
        "distinct_winners": sorted(set(winners.values())),
    }
    results["_config"] = {
        "min_train": MIN_TRAIN, "refit_every": REFIT_EVERY, "max_train": MAX_TRAIN,
        "quantiles": QUANTILES, "es_levels": ES_LEVELS,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwritten {OUT_PATH} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
