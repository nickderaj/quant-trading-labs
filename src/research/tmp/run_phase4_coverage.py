"""Phase 4 driver: the tail calibration battery. For every model in Phase 3
(d0-d9 - ALL TEN here, unlike Phase 3's density contest: d8/d9's quantile/ES
forecasts are well-defined from their GPD fits even though a full normalized
density for log-score/CRPS was the part deferred, per Phase 3's own scope
note), at every interval, at all six quantile levels: Kupiec, Christoffersen
independence, Christoffersen conditional coverage (Gate B's own 36-test-per-
interval bar), plus an Acerbi-Szekely ES backtest wherever a model can
produce ES (all ten can: d0-d7 analytically, d8/d9 from their GPD fits).

Re-fits every model from scratch (same convention as run_phase1_tails.py):
results JSON files in this repo persist aggregated scores, not raw per-bar
forecast arrays, so every driver that needs the underlying forecasts rebuilds
them with the same construction rather than depending on another driver's
in-memory state.
"""

import json
import sys
import time

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import dist_lib as L
import dist_lib5 as L5
import numpy as np

import research

SYMBOL = "BTCUSDT"
INTERVALS = ["1h", "4h", "12h", "1d"]
BARS_PER_DAY = {"1h": 24, "4h": 6, "12h": 2, "1d": 1}
MIN_TRAIN_DAYS = 90
CHEAP_REFIT_DAYS = 7
MLE_REFIT_DAYS = 30
MLE_MAX_TRAIN = 500
ES_BOOTSTRAP_N = 300
QUANTILES = L5.QUANTILES

out: dict = {"symbol": SYMBOL, "intervals": {}}

for interval in INTERVALS:
    t0 = time.time()
    bpd = BARS_PER_DAY[interval]
    cheap_refit_every = CHEAP_REFIT_DAYS * bpd
    mle_refit_every = MLE_REFIT_DAYS * bpd
    min_train = MIN_TRAIN_DAYS * bpd

    df = L.build_asset_frame(SYMBOL, interval, end=research.HOLDOUT_START)
    n = len(df)
    rv = df["rv_target"].to_numpy()
    ret = df["log_return"].fill_null(0.0).to_numpy()

    variance_fc: dict[str, np.ndarray] = {}

    trailing_candidates = {
        f"trailing_{w}": L.rung0_trailing_std(df, w).to_numpy() for w in [8, 24, 96]
    }

    def _q(fc, rv=rv):
        import distributions as dist

        m = np.isfinite(fc) & (fc > 0) & (rv > 0)
        return np.nanmean(dist.qlike(rv[m], fc[m])) if m.sum() > 10 else np.inf

    _, variance_fc["d0_trailing_std"] = min(
        trailing_candidates.items(), key=lambda kv: _q(kv[1])
    )

    har_df = L.make_har_features(df, interval)
    variance_fc["d1_har_rv"] = L.rolling_ols_refit(
        har_df,
        ["rv_d", "rv_w", "rv_m"],
        "rv_target",
        refit_every=cheap_refit_every,
        min_train=min_train,
    )
    variance_fc["d2_har_log_rv"] = L5.har_log_rv_forecast(
        df, interval, cheap_refit_every, min_train
    )

    range_df = L.range_estimator_forecasts(df, window=bpd if bpd > 1 else 24)
    range_candidates = {
        name: range_df[col].to_numpy()
        for name, col in [
            ("parkinson", "fc_parkinson"),
            ("gk", "fc_gk"),
            ("rs", "fc_rs"),
            ("yz", "fc_yz"),
        ]
    }
    _, variance_fc["d3_range"] = min(range_candidates.items(), key=lambda kv: _q(kv[1]))

    variance_fc["d4_garch_normal"], fits_d4 = L.rolling_garch_forecast(
        ret,
        refit_every=mle_refit_every,
        min_train=min_train,
        innovation="normal",
        max_train=MLE_MAX_TRAIN,
    )
    variance_fc["d5_garch_t"], fits_d5 = L.rolling_garch_forecast(
        ret,
        refit_every=mle_refit_every,
        min_train=min_train,
        innovation="t",
        max_train=MLE_MAX_TRAIN,
    )
    nu_d5 = L.nu_path_from_fits(fits_d5, n, param_index=3)

    variance_fc["d6_gjr_normal"], fits_d6 = L5.rolling_gjr_forecast(
        ret,
        refit_every=mle_refit_every,
        min_train=min_train,
        innovation="normal",
        max_train=MLE_MAX_TRAIN,
    )
    variance_fc["d7_gjr_t"], fits_d7 = L5.rolling_gjr_forecast(
        ret,
        refit_every=mle_refit_every,
        min_train=min_train,
        innovation="t",
        max_train=MLE_MAX_TRAIN,
    )
    nu_d7 = L.nu_path_from_fits(fits_d7, n, param_index=4)

    # d8/d9: GARCH-EVT / GJR-EVT - GPD tails on standardized residuals from
    # the d4/d6 (normal-innovation) variance fits, causal and refit-aligned.
    gpd_paths_d8, _gpd_fits_d8 = L5.rolling_gpd_paths(
        ret, fits_d4, model="garch", max_train=MLE_MAX_TRAIN, tail_frac=0.10
    )
    gpd_paths_d9, _gpd_fits_d9 = L5.rolling_gpd_paths(
        ret, fits_d6, model="gjr", max_train=MLE_MAX_TRAIN, tail_frac=0.10
    )
    variance_fc["d8_garch_evt"] = variance_fc["d4_garch_normal"]
    variance_fc["d9_gjr_evt"] = variance_fc["d6_gjr_normal"]

    # ---- quantile forecasts per model ----
    quantile_forecasts: dict[str, dict] = {}
    es_forecasts: dict[str, dict] = {}
    for name in [
        "d0_trailing_std",
        "d1_har_rv",
        "d2_har_log_rv",
        "d3_range",
        "d4_garch_normal",
        "d6_gjr_normal",
    ]:
        quantile_forecasts[name] = L5.normal_quantile_forecasts(
            variance_fc[name], QUANTILES
        )
        es_forecasts[name] = {
            q: L5.normal_es_forecast(variance_fc[name], q) for q in QUANTILES
        }
    quantile_forecasts["d5_garch_t"] = L5.t_quantile_forecasts(
        variance_fc["d5_garch_t"], nu_d5, QUANTILES
    )
    es_forecasts["d5_garch_t"] = {
        q: L5.t_es_forecast(variance_fc["d5_garch_t"], nu_d5, q) for q in QUANTILES
    }
    quantile_forecasts["d7_gjr_t"] = L5.t_quantile_forecasts(
        variance_fc["d7_gjr_t"], nu_d7, QUANTILES
    )
    es_forecasts["d7_gjr_t"] = {
        q: L5.t_es_forecast(variance_fc["d7_gjr_t"], nu_d7, q) for q in QUANTILES
    }
    quantile_forecasts["d8_garch_evt"] = L5.gpd_quantile_forecasts(
        variance_fc["d4_garch_normal"], gpd_paths_d8, QUANTILES
    )
    es_forecasts["d8_garch_evt"] = {
        q: L5.gpd_es_forecast(variance_fc["d4_garch_normal"], gpd_paths_d8, q)
        for q in QUANTILES
    }
    quantile_forecasts["d9_gjr_evt"] = L5.gpd_quantile_forecasts(
        variance_fc["d6_gjr_normal"], gpd_paths_d9, QUANTILES
    )
    es_forecasts["d9_gjr_evt"] = {
        q: L5.gpd_es_forecast(variance_fc["d6_gjr_normal"], gpd_paths_d9, q)
        for q in QUANTILES
    }

    # ---- coverage battery + Gate B verdict ----
    coverage: dict[str, dict] = {}
    gate_b: dict[str, bool] = {}
    for name, qf in quantile_forecasts.items():
        battery = L5.coverage_battery(ret, qf)
        coverage[name] = battery
        never_rejected = all(
            (level["kupiec_p"] > 0.05)
            and (level["indep_p"] > 0.05)
            and (level["cc_p"] > 0.05)
            for level in battery.values()
        )
        underpowered_levels = [
            q for q, level in battery.items() if level["n_violations"] < 10
        ]
        gate_b[name] = bool(never_rejected)
        coverage[name]["_gate_b_clears_all_36"] = bool(never_rejected)
        coverage[name]["_underpowered_levels"] = underpowered_levels

    # ---- ES backtest (Acerbi-Szekely), one-sided tails 0.01/0.05 lower and upper ----
    es_backtest: dict[str, dict] = {}
    underlying_variance_col = {
        "d8_garch_evt": "d4_garch_normal",
        "d9_gjr_evt": "d6_gjr_normal",
    }
    sigma_by_model = {
        name: np.sqrt(
            np.where(
                variance_fc[underlying_variance_col.get(name, name)] > 0,
                variance_fc[underlying_variance_col.get(name, name)],
                np.nan,
            )
        )
        for name in quantile_forecasts
    }
    for name in quantile_forecasts:
        es_backtest[name] = {}
        for q in [0.01, 0.05]:
            var_fc_q = quantile_forecasts[name][q]
            es_fc_q = es_forecasts[name][q]
            z_obs = L5.acerbi_szekely_z(ret, var_fc_q, es_fc_q, q)
            sigma = sigma_by_model[name]
            if name == "d5_garch_t":
                sim_fn = L5.make_t_acerbi_simulate_fn(sigma, nu_d5, es_fc_q, q)
            elif name == "d7_gjr_t":
                sim_fn = L5.make_t_acerbi_simulate_fn(sigma, nu_d7, es_fc_q, q)
            elif name == "d8_garch_evt":
                sim_fn = L5.make_gpd_acerbi_simulate_fn(sigma, gpd_paths_d8, es_fc_q, q)
            elif name == "d9_gjr_evt":
                sim_fn = L5.make_gpd_acerbi_simulate_fn(sigma, gpd_paths_d9, es_fc_q, q)
            else:
                sim_fn = L5.make_normal_acerbi_simulate_fn(sigma, es_fc_q, q)
            p_boot = L5.acerbi_szekely_bootstrap_pvalue(
                z_obs, sim_fn, n_boot=ES_BOOTSTRAP_N, seed=0
            )
            es_backtest[name][str(q)] = {"z": z_obs, "bootstrap_pvalue": p_boot}

    res_out = {
        "n_obs": n,
        "coverage": coverage,
        "gate_b_verdict": gate_b,
        "es_backtest": es_backtest,
        "elapsed_sec": time.time() - t0,
    }
    out["intervals"][interval] = res_out
    any_gate_b = [name for name, v in gate_b.items() if v]
    print(
        f"{interval}: n={n} elapsed={res_out['elapsed_sec']:.1f}s "
        f"gate_b_clears={any_gate_b if any_gate_b else 'NONE'}"
    )

n_any_gate_b_interval = sum(
    1 for iv in out["intervals"].values() if any(iv["gate_b_verdict"].values())
)
out["gate_b_summary"] = {
    "n_intervals_with_any_clearer": n_any_gate_b_interval,
    "n_intervals": len(INTERVALS),
}
print(
    f"Gate B: {n_any_gate_b_interval}/{len(INTERVALS)} intervals have at least one model clearing all 36 tests"
)

with open("src/research/tmp/phase4_coverage_results.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print("written phase4_coverage_results.json")
