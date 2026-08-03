"""Phase 3 driver: the density contest. Log score is PRIMARY (QLIKE kept as a
secondary column only, for continuity with notebook 4 - see
NEXT_RUN_PROMPT.md's own framing of why the criterion changed).

Competitor set (d0-d9 in NEXT_RUN_PROMPT.md's own table). d8/d9 (GARCH-EVT,
GJR-EVT) are semiparametric (empirical body + GPD tails) and are NOT entered
in this log-score/CRPS contest - continuously normalizing that density proved
exactly as fiddly as the runbook anticipated it might, and the runbook itself
sanctions the fallback: "evaluate d8/d9 on Gate B (tail calibration) only and
state plainly that they were not entered in the log-score contest - an honest
partial entry beats a hand-waved density." d8/d9's rolling GARCH/GJR fits are
still produced here (needed by Phase 4's coverage battery) but not scored.

So this is an 8-model (d0-d7) contest: C(8,2)=28 pairs x 4 intervals = 112
tests, not the 45x4=180 a full 10-model contest would need - reported
honestly as a documented, sanctioned scope reduction, not silently smaller.

All-pairs DM (same non-transitive logic as notebook 4's run_phase3.py) on the
per-bar log-score loss differential (loss = -log_score, DM's own lower-is-
better convention), reported both via DM's normal-approximation p-value and a
block-bootstrap p-value (per Phase 1b's own finding that these can disagree,
worth carrying through Phase 3 rather than trusting the normal approximation
alone) - Gate A's determination uses the BH-adjusted BOOTSTRAP p-values as
primary, with the normal-approximation/BH pair reported alongside.
"""

import itertools
import json
import sys
import time

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import dist_lib as L
import dist_lib5 as L5
import numpy as np

import distributions as dist
import research

SYMBOL = "BTCUSDT"
INTERVALS = ["1h", "4h", "12h", "1d"]
BARS_PER_DAY = {"1h": 24, "4h": 6, "12h": 2, "1d": 1}
MIN_TRAIN_DAYS = 90
CHEAP_REFIT_DAYS = 7
MLE_REFIT_DAYS = 30
MLE_MAX_TRAIN = 500
DM_BOOTSTRAP_N = 500  # reduced from Phase 1's 2000 - 112 pairwise tests total

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
    nu_paths: dict[str, np.ndarray] = {}

    # d0: trailing std (best window, chosen by QLIKE - same 3 candidates as
    # notebook 4's rung0)
    trailing_candidates = {
        f"trailing_{w}": L.rung0_trailing_std(df, w).to_numpy() for w in [8, 24, 96]
    }
    d0_name, variance_fc["d0_trailing_std"] = min(
        trailing_candidates.items(),
        key=lambda kv: (
            np.nanmean(
                dist.qlike(
                    rv[(rv > 0) & (kv[1] > 0) & np.isfinite(kv[1])],
                    kv[1][(rv > 0) & (kv[1] > 0) & np.isfinite(kv[1])],
                )
            )
            if np.isfinite(kv[1]).any()
            else np.inf
        ),
    )

    # d1: HAR-RV (levels)
    har_df = L.make_har_features(df, interval)
    variance_fc["d1_har_rv"] = L.rolling_ols_refit(
        har_df,
        ["rv_d", "rv_w", "rv_m"],
        "rv_target",
        refit_every=cheap_refit_every,
        min_train=min_train,
    )

    # d2: HAR-log-RV (Phase 1c)
    variance_fc["d2_har_log_rv"] = L5.har_log_rv_forecast(
        df, interval, cheap_refit_every, min_train
    )

    # d3: best range estimator (chosen by QLIKE)
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

    def _q(fc, rv=rv):
        m = np.isfinite(fc) & (fc > 0) & (rv > 0)
        return np.nanmean(dist.qlike(rv[m], fc[m])) if m.sum() > 10 else np.inf

    best_range_name, variance_fc["d3_range"] = min(
        range_candidates.items(), key=lambda kv: _q(kv[1])
    )

    # d4/d5: GARCH(1,1) normal/t
    variance_fc["d4_garch_normal"], _fits_d4 = L.rolling_garch_forecast(
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

    # d6/d7: GJR-GARCH normal/t
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

    # d8/d9: GARCH-EVT / GJR-EVT - fits produced (needed by Phase 4), NOT
    # scored here (see module docstring). Report the LR gamma=0 test from the
    # GJR fits here too, since it's a one-line, already-computed byproduct.
    gjr_normal_lr = [
        f["lr_gamma0_pvalue"] for f in fits_d6 if f.get("lr_gamma0_pvalue") is not None
    ]
    gjr_t_lr = [
        f["lr_gamma0_pvalue"] for f in fits_d7 if f.get("lr_gamma0_pvalue") is not None
    ]
    frac_leverage_sig_normal = (
        float(np.mean([p < 0.05 for p in gjr_normal_lr])) if gjr_normal_lr else None
    )
    frac_leverage_sig_t = (
        float(np.mean([p < 0.05 for p in gjr_t_lr])) if gjr_t_lr else None
    )

    # ---- score d0-d7 ----
    t_models = {"d5_garch_t", "d7_gjr_t"}
    log_score_full: dict[str, np.ndarray] = {}
    crps_full: dict[str, np.ndarray] = {}
    scores: dict[str, dict] = {}
    for name, fc in variance_fc.items():
        if name in t_models:
            res = L5.vectorized_t_scores(ret, fc, nu_paths[name])
        else:
            res = L5.vectorized_normal_scores(ret, fc)
        mask, ls, cr = res["mask"], res["log_score"], res["crps"]
        ls_full = np.full(n, np.nan)
        ls_full[mask] = ls
        cr_full = np.full(n, np.nan)
        cr_full[mask] = cr
        log_score_full[name] = ls_full
        crps_full[name] = cr_full

        qmask = np.isfinite(fc) & (fc > 0) & (rv > 0)
        qlike_mean = (
            float(np.nanmean(dist.qlike(rv[qmask], fc[qmask])))
            if qmask.sum() > 10
            else float("nan")
        )
        scores[name] = {
            "log_score_mean": float(np.nanmean(ls)),
            "crps_mean": float(np.nanmean(cr)),
            "qlike_mean": qlike_mean,
            "n": int(mask.sum()),
        }

    # ---- all-pairs DM on log-score loss differential (loss = -log_score) ----
    model_names = list(variance_fc.keys())
    all_pairs_dm = {}
    normal_pvalues, boot_pvalues = {}, {}
    for a, b in itertools.combinations(model_names, 2):
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
            "log_score_a": float(np.nanmean(-loss_a[both])),
            "log_score_b": float(np.nanmean(-loss_b[both])),
            "n": int(both.sum()),
        }
        normal_pvalues[key] = p_normal
        boot_pvalues[key] = p_boot

    bh_normal = L5.benjamini_hochberg(normal_pvalues, alpha=0.05)
    bh_boot = L5.benjamini_hochberg(boot_pvalues, alpha=0.05)
    for key, entry in all_pairs_dm.items():
        entry["bh_normal"] = bh_normal[key]
        entry["bh_bootstrap"] = bh_boot[key]

    # ---- Gate A verdict (primary: BH-adjusted bootstrap p-values) ----
    mean_log_score = {name: scores[name]["log_score_mean"] for name in model_names}
    best_name = max(mean_log_score, key=lambda n: mean_log_score[n])

    def _beats_all_significantly(
        best_name: str,
        bh_field: str,
        model_names=model_names,
        all_pairs_dm=all_pairs_dm,
    ) -> bool:
        for other in model_names:
            if other == best_name:
                continue
            key = (
                f"{best_name}_vs_{other}"
                if f"{best_name}_vs_{other}" in all_pairs_dm
                else f"{other}_vs_{best_name}"
            )
            entry = all_pairs_dm.get(key)
            if entry is None:
                continue
            a_is_best = entry["a"] == best_name
            bh = entry[bh_field]
            if not bh["significant"]:
                return False
            best_wins_sig = (a_is_best and entry["tstat"] < 0) or (
                not a_is_best and entry["tstat"] > 0
            )
            if not best_wins_sig:
                return False
        return True

    gate_a_bootstrap = _beats_all_significantly(best_name, "bh_bootstrap")
    gate_a_normal = _beats_all_significantly(best_name, "bh_normal")

    res_out = {
        "n_obs": n,
        "scope_note": "8-model contest (d0-d7); d8/d9 GARCH-EVT/GJR-EVT deferred to Gate B only",
        "best_range_estimator": best_range_name,
        "best_trailing_window": d0_name,
        "scores": scores,
        "all_pairs_dm": all_pairs_dm,
        "gate_a_verdict": {
            "best_by_log_score": best_name,
            "beats_every_other_significantly_bootstrap_bh": gate_a_bootstrap,
            "beats_every_other_significantly_normal_bh": gate_a_normal,
        },
        "gjr_leverage": {
            "frac_refits_significant_gamma0_normal": frac_leverage_sig_normal,
            "frac_refits_significant_gamma0_t": frac_leverage_sig_t,
            "n_refits_normal": len(fits_d6),
            "n_refits_t": len(fits_d7),
        },
        "elapsed_sec": time.time() - t0,
    }
    out["intervals"][interval] = res_out
    print(
        f"{interval}: n={n} elapsed={res_out['elapsed_sec']:.1f}s "
        f"best={best_name} gate_a(boot)={gate_a_bootstrap} gate_a(normal)={gate_a_normal} "
        f"leverage_frac_normal={frac_leverage_sig_normal} leverage_frac_t={frac_leverage_sig_t}"
    )

with open("src/research/tmp/phase3_density_results.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print("written phase3_density_results.json")
