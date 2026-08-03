"""Phase 5 driver: transfer / stability. ETH/SOL/DOGE/BNB/XRP at 1d only -
same scoping rationale as notebook 4 (wall clock on a Raspberry Pi): BTC
already got the full 4-interval treatment in checkpoints 6-7, this checks
whether that treatment's *findings* generalize, at lighter cost.

A real scoping tension, stated plainly rather than glossed over: BTC's own
Gate A win (GARCH-t, significant at 1h/4h/12h) is NOT at 1d - 1d was the one
interval where BTC's own contest found no significant winner (best was
HAR-log-RV, not significantly). So this driver cannot directly test whether
the 1h/4h/12h Gate A finding itself replicates cross-sectionally - only
whether the SAME pattern found on BTC at 1d (a near-tie, GARCH-t/GJR-t/HAR-
log-RV close together, nothing significant) also holds on the transfer
symbols at 1d. That is still a meaningful stability check (per notebook 4's
own framing: tail-shape/ranking patterns, not just point "winners," are
worth checking for replication) - just a narrower one than "does GARCH-t's
win replicate," which the compute budget on this hardware does not allow
checking directly across all 4 intervals x 5 symbols.

Runs the same 8-model Gate A contest (log score, all-pairs DM, BH-adjusted)
and 10-model Gate B coverage battery as checkpoints 6-7, per symbol, at 1d.
"""

import itertools
import json
import sys
import time
from typing import Any

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import dist_lib as L
import dist_lib5 as L5
import numpy as np

import distributions as dist
import research

TRANSFER_SYMBOLS = ["ETHUSDT", "SOLUSDT", "DOGEUSDT", "BNBUSDT", "XRPUSDT"]
INTERVAL = "1d"
BPD = 1
MIN_TRAIN_DAYS = 90
CHEAP_REFIT_DAYS = 7
MLE_REFIT_DAYS = 30
MLE_MAX_TRAIN = 500
DM_BOOTSTRAP_N = 500
ES_BOOTSTRAP_N = 300
QUANTILES = L5.QUANTILES

out: dict = {"interval": INTERVAL, "symbols": {}}


def run_one_symbol(symbol: str) -> dict:
    t0 = time.time()
    bpd = BPD
    cheap_refit_every = CHEAP_REFIT_DAYS * bpd
    mle_refit_every = MLE_REFIT_DAYS * bpd
    min_train = MIN_TRAIN_DAYS * bpd

    df = L.build_asset_frame(symbol, INTERVAL, end=research.HOLDOUT_START)
    n = len(df)
    rv = df["rv_target"].to_numpy()
    ret = df["log_return"].fill_null(0.0).to_numpy()

    variance_fc: dict[str, np.ndarray] = {}
    nu_paths: dict[str, np.ndarray] = {}

    trailing_candidates = {
        f"trailing_{w}": L.rung0_trailing_std(df, w).to_numpy() for w in [8, 24, 96]
    }

    def _q(fc):
        m = np.isfinite(fc) & (fc > 0) & (rv > 0)
        return np.nanmean(dist.qlike(rv[m], fc[m])) if m.sum() > 10 else np.inf

    _, variance_fc["d0_trailing_std"] = min(
        trailing_candidates.items(), key=lambda kv: _q(kv[1])
    )

    har_df = L.make_har_features(df, INTERVAL)
    variance_fc["d1_har_rv"] = L.rolling_ols_refit(
        har_df,
        ["rv_d", "rv_w", "rv_m"],
        "rv_target",
        refit_every=cheap_refit_every,
        min_train=min_train,
    )
    variance_fc["d2_har_log_rv"] = L5.har_log_rv_forecast(
        df, INTERVAL, cheap_refit_every, min_train
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
    nu_paths["d5_garch_t"] = L.nu_path_from_fits(fits_d5, n, param_index=3)

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
    nu_paths["d7_gjr_t"] = L.nu_path_from_fits(fits_d7, n, param_index=4)

    gpd_paths_d8, _ = L5.rolling_gpd_paths(
        ret, fits_d4, model="garch", max_train=MLE_MAX_TRAIN, tail_frac=0.10
    )
    gpd_paths_d9, _ = L5.rolling_gpd_paths(
        ret, fits_d6, model="gjr", max_train=MLE_MAX_TRAIN, tail_frac=0.10
    )
    variance_fc["d8_garch_evt"] = variance_fc["d4_garch_normal"]
    variance_fc["d9_gjr_evt"] = variance_fc["d6_gjr_normal"]

    gjr_normal_lr = [
        f["lr_gamma0_pvalue"] for f in fits_d6 if f.get("lr_gamma0_pvalue") is not None
    ]
    gjr_t_lr = [
        f["lr_gamma0_pvalue"] for f in fits_d7 if f.get("lr_gamma0_pvalue") is not None
    ]
    frac_leverage_normal = (
        float(np.mean([p < 0.05 for p in gjr_normal_lr])) if gjr_normal_lr else None
    )
    frac_leverage_t = float(np.mean([p < 0.05 for p in gjr_t_lr])) if gjr_t_lr else None

    # ---- Gate A: 8-model log-score contest ----
    t_models = {"d5_garch_t", "d7_gjr_t"}
    log_score_full: dict[str, np.ndarray] = {}
    scores: dict[str, dict] = {}
    gate_a_models = [
        name for name in variance_fc if name not in ("d8_garch_evt", "d9_gjr_evt")
    ]
    for name in gate_a_models:
        fc = variance_fc[name]
        if name in t_models:
            r = L5.vectorized_t_scores(ret, fc, nu_paths[name])
        else:
            r = L5.vectorized_normal_scores(ret, fc)
        mask, ls = r["mask"], r["log_score"]
        ls_full = np.full(n, np.nan)
        ls_full[mask] = ls
        log_score_full[name] = ls_full
        scores[name] = {"log_score_mean": float(np.nanmean(ls)), "n": int(mask.sum())}

    all_pairs_dm: dict[str, dict[str, Any]] = {}
    normal_pvalues, boot_pvalues = {}, {}
    for a, b in itertools.combinations(gate_a_models, 2):
        loss_a, loss_b = -log_score_full[a], -log_score_full[b]
        both = np.isfinite(loss_a) & np.isfinite(loss_b)
        if both.sum() < 30:
            continue
        d = (loss_a - loss_b)[both]
        tstat, p_normal = L.diebold_mariano(loss_a[both], loss_b[both])
        p_boot = research.block_bootstrap_pvalue(
            d, null_value=0.0, n_boot=DM_BOOTSTRAP_N, seed=0
        )
        key = f"{a}_vs_{b}"
        all_pairs_dm[key] = {
            "a": a,
            "b": b,
            "tstat": tstat,
            "normal_pvalue": p_normal,
            "bootstrap_pvalue": p_boot,
        }
        normal_pvalues[key] = p_normal
        boot_pvalues[key] = p_boot

    bh_boot = L5.benjamini_hochberg(boot_pvalues, alpha=0.05)
    for key, entry in all_pairs_dm.items():
        entry["bh_bootstrap"] = bh_boot[key]

    mean_log_score = {name: scores[name]["log_score_mean"] for name in gate_a_models}
    best_name = max(mean_log_score, key=lambda n: mean_log_score[n])
    gate_a_fires = True
    for other in gate_a_models:
        if other == best_name:
            continue
        key = (
            f"{best_name}_vs_{other}"
            if f"{best_name}_vs_{other}" in all_pairs_dm
            else f"{other}_vs_{best_name}"
        )
        pair_entry = all_pairs_dm.get(key)
        if pair_entry is None:
            continue
        bh = pair_entry["bh_bootstrap"]
        a_is_best = pair_entry["a"] == best_name
        best_wins_sig = bh["significant"] and (
            (a_is_best and pair_entry["tstat"] < 0)
            or (not a_is_best and pair_entry["tstat"] > 0)
        )
        if not best_wins_sig:
            gate_a_fires = False
            break

    # ---- Gate B: 10-model coverage battery ----
    quantile_forecasts: dict[str, dict] = {}
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
    quantile_forecasts["d5_garch_t"] = L5.t_quantile_forecasts(
        variance_fc["d5_garch_t"], nu_paths["d5_garch_t"], QUANTILES
    )
    quantile_forecasts["d7_gjr_t"] = L5.t_quantile_forecasts(
        variance_fc["d7_gjr_t"], nu_paths["d7_gjr_t"], QUANTILES
    )
    quantile_forecasts["d8_garch_evt"] = L5.gpd_quantile_forecasts(
        variance_fc["d4_garch_normal"], gpd_paths_d8, QUANTILES
    )
    quantile_forecasts["d9_gjr_evt"] = L5.gpd_quantile_forecasts(
        variance_fc["d6_gjr_normal"], gpd_paths_d9, QUANTILES
    )

    gate_b_by_model = {}
    for name, qf in quantile_forecasts.items():
        battery = L5.coverage_battery(ret, qf)
        gate_b_by_model[name] = all(
            (lvl["kupiec_p"] > 0.05)
            and (lvl["indep_p"] > 0.05)
            and (lvl["cc_p"] > 0.05)
            for lvl in battery.values()
        )
    gate_b_any = any(gate_b_by_model.values())

    return {
        "n_obs": n,
        "scores": scores,
        "best_by_log_score": best_name,
        "gate_a_fires": gate_a_fires,
        "gate_b_by_model": gate_b_by_model,
        "gate_b_any_clears": gate_b_any,
        "gjr_leverage": {
            "frac_refits_significant_gamma0_normal": frac_leverage_normal,
            "frac_refits_significant_gamma0_t": frac_leverage_t,
        },
        "elapsed_sec": time.time() - t0,
    }


for symbol in TRANSFER_SYMBOLS:
    res = run_one_symbol(symbol)
    out["symbols"][symbol] = res
    print(
        f"{symbol}: n={res['n_obs']} elapsed={res['elapsed_sec']:.1f}s "
        f"best={res['best_by_log_score']} gate_a={res['gate_a_fires']} gate_b_any={res['gate_b_any_clears']} "
        f"leverage_frac_normal={res['gjr_leverage']['frac_refits_significant_gamma0_normal']}"
    )

# ---- Gate C stability verdict: compare against BTC's own 1d result
# (phase3_density_results.json / phase4_coverage_results.json) ----
with open("src/research/tmp/phase3_density_results.json") as _f:
    btc_density = json.load(_f)["intervals"]["1d"]
with open("src/research/tmp/phase4_coverage_results.json") as _f:
    btc_coverage = json.load(_f)["intervals"]["1d"]
btc_gate_a = btc_density["gate_a_verdict"][
    "beats_every_other_significantly_bootstrap_bh"
]
btc_gate_b = btc_coverage.get("gate_b_summary", None)
btc_best = btc_density["gate_a_verdict"]["best_by_log_score"]

all_symbols_gate_a = [btc_gate_a] + [
    out["symbols"][s]["gate_a_fires"] for s in TRANSFER_SYMBOLS
]
n_gate_a_fire = sum(all_symbols_gate_a)
out["gate_c_verdict"] = {
    "btc_1d_best_model": btc_best,
    "btc_1d_gate_a_fires": btc_gate_a,
    "n_of_6_symbols_gate_a_fires": n_gate_a_fire,
    "stable_no_winner_pattern": n_gate_a_fire
    <= 1,  # BTC didn't fire; check if transfer mostly agrees
    "transfer_best_models": {
        s: out["symbols"][s]["best_by_log_score"] for s in TRANSFER_SYMBOLS
    },
}
print(
    f"Gate C: {n_gate_a_fire}/6 symbols (incl. BTC) have a significant Gate A winner at 1d; "
    f"BTC's own best={btc_best} (not significant)"
)

with open("src/research/tmp/phase5_transfer_results.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print("written phase5_transfer_results.json")
