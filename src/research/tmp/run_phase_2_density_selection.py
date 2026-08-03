"""Phase 2: unconditional density selection per commodity (NEXT_PROMPT.md sec
4, Phase 2) -- "which pdf fits each commodity best."

Families: normal, Student-t (distributions.py's own loc/scale MLE, fit
directly on raw returns), skew-t (Hansen), NIG, GED, JohnsonSU (the
notebook-6 `densities/*.py` shape-only modules -- standardized to the
train fold's own mean/std, scored on test via the family's logpdf plus the
Jacobian term -log(std)), and spliced-EVT (dist_lib6, same standardized
convention).

OOS discipline: expanding-window walk-forward (`research.walk_forward_splits`,
mode="anchored") on the development window only (2010-06-06..2024-12-31;
ES 2018-01-01..; KE 2013-12-16..). Each family is refit once per fold on
everything up to the fold's train end, then scored on that fold's test block
-- never touching future data. AIC/BIC are reported too but are explicitly
secondary (in-sample, single full-sample fit) to the OOS log score, per the
prompt's "in-sample AIC alone is not evidence" instruction.

Writes phase_2_results.json.
"""

import json
import sys
import time
from typing import cast

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import densities
import dist_lib6 as L6
import numpy as np
import polars as pl

import distributions as dist
import research

CURVE_DIR = "src/research/tmp/phase_0_curves"
OUT_PATH = "src/research/tmp/phase_2_results.json"
BTC_PATH = "src/research/cache/BTCUSDT-klines-1d-2021-07-01-2026-07-01.parquet"

DEV_END = {"__default__": "2024-12-31"}
DEV_START = {"__default__": "2010-06-06", "ES": "2018-01-01", "KE": "2013-12-16"}

SHAPE_FAMILIES = list(densities.REGISTRY.keys())  # ged, nig, johnsonsu, hansen_skewt
ALL_FAMILIES = ["normal", "t", *SHAPE_FAMILIES, "spliced_evt"]

N_FOLDS_TARGET = 5
research.set_seed(0)


def load_series(product: str) -> pl.DataFrame:
    if product == "BTCUSDT":
        btc = pl.read_parquet(BTC_PATH).sort("datetime")
        btc = btc.with_columns(
            (pl.col("close") / pl.col("close").shift(1)).log().alias("ret")
        )
        out = btc.select(
            [pl.col("datetime").dt.date().alias("date"), "ret"]
        ).drop_nulls()
        return out.filter(pl.col("ret").is_finite())
    curve = pl.read_parquet(f"{CURVE_DIR}/{product}.parquet")
    dev_start = DEV_START.get(product, DEV_START["__default__"])
    dev_end = DEV_END["__default__"]
    # drop_nulls() alone is not enough: a null close_f1 (e.g. a day the front
    # contract itself didn't print) propagates to NaN, not polars-null, once
    # it passes through the ratio-adjustment's cumulative product/log chain.
    sub = curve.select(
        ["date", pl.col("log_return_ratioadj").alias("ret")]
    ).drop_nulls()
    sub = sub.filter(pl.col("ret").is_finite())
    sub = sub.filter(
        (pl.col("date") >= pl.lit(dev_start).str.to_date())
        & (pl.col("date") <= pl.lit(dev_end).str.to_date())
    )
    return sub


def make_folds(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    train_bars = int(n * 0.6)
    test_bars = max(50, int(n * 0.08))
    return research.walk_forward_splits(
        n, train_bars=train_bars, test_bars=test_bars, mode="anchored"
    )


def fit_and_score_family(
    family: str, r_train: np.ndarray, r_test: np.ndarray
) -> dict | None:
    if family == "normal":
        params = dist._fit_normal(r_train)
        if params is None:
            return None
        d = dist.frozen_dist("normal", params)
        log_score = d.logpdf(r_test)
        n_params = 2
        ll_train = float(np.sum(d.logpdf(r_train)))
    elif family == "t":
        params = dist._fit_t(r_train)
        if params is None:
            return None
        d = dist.frozen_dist("t", params)
        log_score = d.logpdf(r_test)
        n_params = 3
        ll_train = float(np.sum(d.logpdf(r_train)))
    else:
        mean, std = float(np.mean(r_train)), float(np.std(r_train))
        if std <= 1e-12:
            return None
        z_train = (r_train - mean) / std
        z_test = (r_test - mean) / std
        if family == "spliced_evt":
            fit = L6.fit_spliced_evt_density(z_train)
            if fit is None:
                return None
            log_score = L6.spliced_evt_logpdf(z_test, fit) - np.log(std)
            n_params = 6  # 2 GPD tails (2 params each) + KDE bandwidth (approx)
            ll_train = float(np.sum(L6.spliced_evt_logpdf(z_train, fit) - np.log(std)))
        else:
            mod = densities.REGISTRY[family]
            shape = mod.fit(z_train)
            if shape is None:
                return None
            log_score = mod.logpdf(z_test, shape) - np.log(std)
            n_params = 2 + mod.N_SHAPE  # mean, std, + shape params
            ll_train = float(np.sum(mod.logpdf(z_train, shape) - np.log(std)))

    finite = np.isfinite(log_score)
    aic = 2 * n_params - 2 * ll_train
    bic = n_params * np.log(len(r_train)) - 2 * ll_train
    return {
        "log_score": log_score,
        "finite_mask": finite,
        "aic": aic,
        "bic": bic,
        "n_params": n_params,
    }


def pit_for_family(
    family: str, r_train: np.ndarray, r_pool: np.ndarray
) -> np.ndarray | None:
    """PIT of a pooled OOS sample under a family fit once on the full train
    prefix ending just before the pool (used only for the calibration plot,
    not for family ranking)."""
    if family == "normal":
        params = dist._fit_normal(r_train)
        if params is None:
            return None
        return dist.frozen_dist("normal", params).cdf(r_pool)
    if family == "t":
        params = dist._fit_t(r_train)
        if params is None:
            return None
        return dist.frozen_dist("t", params).cdf(r_pool)
    mean, std = float(np.mean(r_train)), float(np.std(r_train))
    if std <= 1e-12:
        return None
    z_train = (r_train - mean) / std
    z_pool = (r_pool - mean) / std
    if family == "spliced_evt":
        fit = L6.fit_spliced_evt_density(z_train)
        if fit is None:
            return None
        return C_numerical_pit(lambda z: L6.spliced_evt_logpdf(z, fit), z_pool)
    mod = densities.REGISTRY[family]
    shape = mod.fit(z_train)
    if shape is None:
        return None
    return C_numerical_pit(lambda z: mod.logpdf(z, shape), z_pool)


def C_numerical_pit(logpdf_fn, z_values):
    import commod_lib8 as C

    return C.numerical_pit(logpdf_fn, z_values)


def process_product(product: str) -> dict | None:
    df = load_series(product)
    ret = df["ret"].to_numpy()
    n = len(ret)
    if n < 500:
        print(f"  {product}: too few obs ({n}) for a real walk-forward, skipping")
        return None
    folds = make_folds(n)
    if len(folds) == 0:
        print(f"  {product}: no folds produced, skipping")
        return None

    per_family_log_score_full: dict[str, np.ndarray] = {
        f: np.full(n, np.nan) for f in ALL_FAMILIES
    }
    per_family_aic: dict[str, list[float]] = {f: [] for f in ALL_FAMILIES}
    per_family_bic: dict[str, list[float]] = {f: [] for f in ALL_FAMILIES}

    for train_idx, test_idx in folds:
        r_train, r_test = ret[train_idx], ret[test_idx]
        for family in ALL_FAMILIES:
            out = fit_and_score_family(family, r_train, r_test)
            if out is None:
                continue
            per_family_log_score_full[family][test_idx] = out["log_score"]
            per_family_aic[family].append(out["aic"])
            per_family_bic[family].append(out["bic"])

    # DM + BH across all pairs, on the pooled OOS log-score arrays
    dm = L6.all_pairs_dm_bh(ALL_FAMILIES, per_family_log_score_full)

    mean_log_score = {
        f: float(np.nanmean(per_family_log_score_full[f]))
        if np.isfinite(per_family_log_score_full[f]).any()
        else None
        for f in ALL_FAMILIES
    }
    valid_families = [f for f, v in mean_log_score.items() if v is not None]
    ranked = sorted(
        valid_families, key=lambda f: cast(float, mean_log_score[f]), reverse=True
    )
    best = ranked[0] if ranked else None
    best_wins_significantly = (
        L6.beats_all_significantly(best, valid_families, dm, "bh_bootstrap")
        if best
        else False
    )

    # PIT + KS on the last fold's test pool (most-informed fit, held-out data)
    last_train_idx, last_test_idx = folds[-1]
    r_train_last, r_test_last = ret[last_train_idx], ret[last_test_idx]
    pit_ks = {}
    for family in ALL_FAMILIES:
        pit = pit_for_family(family, r_train_last, r_test_last)
        if pit is None:
            continue
        pit = pit[np.isfinite(pit)]
        if len(pit) < 20:
            continue
        from scipy import stats as st

        ks_stat, ks_p = st.kstest(pit, "uniform")
        pit_ks[family] = {
            "ks_stat": float(ks_stat),
            "ks_p": float(ks_p),
            "n": len(pit),
            "pit_sample": pit[:500].tolist(),
        }

    return {
        "n_obs": n,
        "n_folds": len(folds),
        "mean_log_score": mean_log_score,
        "aic_mean": {
            f: float(np.mean(v)) if v else None for f, v in per_family_aic.items()
        },
        "bic_mean": {
            f: float(np.mean(v)) if v else None for f, v in per_family_bic.items()
        },
        "ranking": ranked,
        "best_family": best,
        "best_wins_significantly_bh": best_wins_significantly,
        "dm_all_pairs": dm,
        "pit_ks": pit_ks,
    }


def main():
    t0 = time.time()
    results: dict = {}
    import commod_lib8 as C

    products = [*C.PRODUCTS, "BTCUSDT"]
    for p in products:
        print(f"processing {p}...")
        out = process_product(p)
        if out is not None:
            results[p] = out

    # Family map: which family wins where, and is it the same family
    # everywhere (Gate CD's question)?
    winners = {
        p: results[p]["best_family"] for p in results if results[p]["best_family"]
    }
    distinct_winners = sorted(set(winners.values()))
    results["_family_map_summary"] = {
        "winners": winners,
        "distinct_families_that_win_somewhere": distinct_winners,
        "n_distinct": len(distinct_winners),
    }
    results["_config"] = {
        "families": ALL_FAMILIES,
        "dev_start": DEV_START,
        "dev_end": DEV_END,
        "n_folds_target": N_FOLDS_TARGET,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwritten {OUT_PATH} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
