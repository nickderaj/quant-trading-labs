"""Phase 1 driver: does the variance even exist?

Foundational, cheap, and it gates the interpretation of everything else in
notebook 5. Per NEXT_RUN_PROMPT.md, fit-once on the full pre-holdout history
is acceptable here (Phase 1 is characterization, same status as notebook 4's
own Phase 1) - Phase 2 onward is where everything becomes rolling/causal.

1a. Hill estimator of the tail index, both tails, all 4 intervals, with a
    Hill plot (plateau-read, not a single point estimate) and a block-
    bootstrap CI at the chosen k.
1b. Does the Diebold-Mariano test's own CLT hold here? Compare its normal-
    approximation p-value against a block-bootstrap p-value on the same
    QLIKE loss-differential series, for the closest pair found in notebook
    4's Phase 3 all-pairs DM at each interval.
1c. Is log-RV the better-behaved object? Fit normal/t/skew-t to rv and to
    log(rv), compare PIT/KS calibration.

Gate E (foundational, from NEXT_RUN_PROMPT.md #3): if alpha <= 2 with a CI
excluding 2 at most intervals, notebook 5's results file gets a prominent
caveat that notebook 4's variance-forecasting framework rests on a moment
that may not exist. Fires independently of everything else and is reported
regardless of outcome.
"""

import json
import sys
import time

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import numpy as np
import polars as pl

import dist_lib as L
import dist_lib5 as L5
import distributions as dist
import research

SYMBOL = "BTCUSDT"
INTERVALS = ["1h", "4h", "12h", "1d"]
BARS_PER_DAY = {"1h": 24, "4h": 6, "12h": 2, "1d": 1}
MIN_TRAIN_DAYS = 90
CHEAP_REFIT_DAYS = 7
MLE_REFIT_DAYS = 30
MLE_MAX_TRAIN = 500

# The closest (least-significant) pair from notebook 4's Phase 3 all-pairs DM
# at each interval (src/research/tmp/phase3_results.json's own all_pairs_dm),
# used for the Phase 1b DM-validity check. Recorded explicitly here (not
# re-derived from the JSON at runtime) so this driver's own intent is legible
# without cross-referencing another file.
CLOSEST_PAIR = {
    "1h": ("rung3_fc_gk", "rung5_garch_normal"),
    "4h": ("rung3_fc_gk", "rung5_garch_normal"),
    "12h": ("rung1_ewma", "rung3_fc_gk"),
    "1d": ("rung0_trailing_96", "rung3_fc_parkinson"),
}

out: dict = {"symbol": SYMBOL, "intervals": {}}


def qlike_loss_series(rv: np.ndarray, fc: np.ndarray) -> np.ndarray:
    mask = np.isfinite(rv) & np.isfinite(fc) & (fc > 0) & (rv > 0)
    out_arr = np.full(len(rv), np.nan)
    out_arr[mask] = dist.qlike(rv[mask], fc[mask])
    return out_arr


def build_rung_forecast(name: str, df, rv, ret, bpd) -> np.ndarray:
    """Recompute a single named Phase-3 rung's forecast, for the DM-validity
    check's loss-differential series - not persisted by run_phase3.py, so
    it's rebuilt here with the exact same construction/cadence."""
    cheap_refit_every = CHEAP_REFIT_DAYS * bpd
    mle_refit_every = MLE_REFIT_DAYS * bpd
    min_train = MIN_TRAIN_DAYS * bpd
    if name.startswith("rung0_trailing_"):
        w = int(name.rsplit("_", 1)[1])
        return L.rung0_trailing_std(df, w).to_numpy()
    if name == "rung1_ewma":
        return L.rung1_ewma(df).to_numpy()
    if name.startswith("rung3_"):
        range_df = L.range_estimator_forecasts(df, window=bpd if bpd > 1 else 24)
        col = {"rung3_fc_parkinson": "fc_parkinson", "rung3_fc_gk": "fc_gk",
               "rung3_fc_rs": "fc_rs", "rung3_fc_yz": "fc_yz"}[name]
        return range_df[col].to_numpy()
    if name.startswith("rung5_garch_"):
        innov = name.rsplit("_", 1)[1]
        fc, _fits = L.rolling_garch_forecast(
            ret, refit_every=mle_refit_every, min_train=min_train, innovation=innov,
            max_train=MLE_MAX_TRAIN,
        )
        return fc
    raise ValueError(f"unhandled rung name for this driver: {name!r}")


for interval in INTERVALS:
    t0 = time.time()
    bpd = BARS_PER_DAY[interval]
    min_train = MIN_TRAIN_DAYS * bpd

    df = L.build_asset_frame(SYMBOL, interval, end=research.HOLDOUT_START)
    n = len(df)
    rv = df["rv_target"].to_numpy()
    ret = df["log_return"].fill_null(0.0).to_numpy()

    res: dict = {}

    # ---- 1a. Hill estimator, both tails ----
    hill = {}
    for tail in ["upper", "lower"]:
        path = L5.hill_alpha_path(ret, tail=tail, k_min=20, k_max=max(20, n // 10))
        plateau = L5.find_hill_plateau(path["alpha"], path["k"], window=50, rel_tol=0.10)
        entry = {"k_grid_min": int(path["k"].min()) if len(path["k"]) else None,
                  "k_grid_max": int(path["k"].max()) if len(path["k"]) else None,
                  "plateau": plateau}
        if plateau.get("found"):
            k_chosen = plateau["k_chosen"]
            lo, hi = research.block_bootstrap_ci(
                ret, n_boot=1000, seed=0,
                statistic=lambda x, k=k_chosen, t=tail: L5.hill_estimator(x, k=k, tail=t),
            )
            entry["k_chosen"] = k_chosen
            entry["alpha_point"] = L5.hill_estimator(ret, k_chosen, tail=tail)
            entry["ci_95"] = [lo, hi]
        hill[tail] = entry
    res["hill"] = hill

    # ---- Gate E signal for this interval: alpha <= 2 with CI excluding 2 ----
    upper = hill["upper"]
    gate_e_fires_here = (
        upper.get("plateau", {}).get("found")
        and upper.get("alpha_point", np.inf) <= 2.0
        and upper.get("ci_95", [np.nan, np.nan])[1] < 2.0
    )
    res["gate_e_fires_here"] = bool(gate_e_fires_here)

    # ---- 1b. DM-validity: bootstrap p-value vs normal-approx p-value ----
    name_a, name_b = CLOSEST_PAIR[interval]
    fc_a = build_rung_forecast(name_a, df, rv, ret, bpd)
    fc_b = build_rung_forecast(name_b, df, rv, ret, bpd)
    la = qlike_loss_series(rv, fc_a)
    lb = qlike_loss_series(rv, fc_b)
    both = np.isfinite(la) & np.isfinite(lb)
    d = (la - lb)[both]
    tstat, normal_pvalue = L.diebold_mariano(la[both], lb[both])
    boot_pvalue = research.block_bootstrap_pvalue(d, null_value=0.0, n_boot=2000, seed=0)
    d_hill_upper = L5.hill_estimator(d - np.mean(d), k=max(20, int(len(d) * 0.05)), tail="upper")
    d_hill_lower = L5.hill_estimator(d - np.mean(d), k=max(20, int(len(d) * 0.05)), tail="lower")
    res["dm_validity"] = {
        "pair": [name_a, name_b], "n": int(both.sum()),
        "tstat": tstat, "normal_pvalue": normal_pvalue, "bootstrap_pvalue": boot_pvalue,
        "materially_disagree": bool(np.isfinite(normal_pvalue) and np.isfinite(boot_pvalue)
                                     and abs(normal_pvalue - boot_pvalue) > 0.05),
        "loss_diff_hill_alpha_upper": d_hill_upper, "loss_diff_hill_alpha_lower": d_hill_lower,
    }

    # ---- 1c. log-RV vs RV calibration ----
    rv_pos = rv[np.isfinite(rv) & (rv > 0)]
    log_rv_frame = pl.DataFrame({"log_rv": np.log(rv_pos)})
    rv_frame = pl.DataFrame({"rv": rv_pos})
    logrv_cal, rv_cal = {}, {}
    for fam in ["normal", "t", "skewt"]:
        p_log = L.fit_once(log_rv_frame, "log_rv", fam)
        p_rv = L.fit_once(rv_frame, "rv", fam)
        if p_log is not None:
            d_log = dist.frozen_dist(fam, p_log)
            _, ks_p_log = dist.pit_ks_test(d_log, np.log(rv_pos))
        else:
            ks_p_log = None
        if p_rv is not None:
            d_rv = dist.frozen_dist(fam, p_rv)
            _, ks_p_rv = dist.pit_ks_test(d_rv, rv_pos)
        else:
            ks_p_rv = None
        logrv_cal[fam] = ks_p_log
        rv_cal[fam] = ks_p_rv
    res["log_rv_vs_rv"] = {"log_rv_ks_pvalue": logrv_cal, "rv_ks_pvalue": rv_cal}

    res["elapsed_sec"] = time.time() - t0
    out["intervals"][interval] = res
    print(f"{interval}: n={n} elapsed={res['elapsed_sec']:.1f}s "
          f"hill_upper_alpha={hill['upper'].get('alpha_point')} "
          f"gate_e_fires={gate_e_fires_here} "
          f"dm_validity(normal={normal_pvalue:.4f}, boot={boot_pvalue:.4f})")

n_fire = sum(1 for iv in out["intervals"].values() if iv["gate_e_fires_here"])
out["gate_e_verdict"] = {
    "n_intervals_firing": n_fire, "n_intervals_total": len(INTERVALS),
    "fires_at_most_intervals": n_fire > len(INTERVALS) / 2,
}
print(f"Gate E: fires at {n_fire}/{len(INTERVALS)} intervals -> "
      f"{'CAVEAT REQUIRED' if out['gate_e_verdict']['fires_at_most_intervals'] else 'no prominent caveat triggered'}")

with open("src/research/tmp/phase1_tails_results.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print("written phase1_tails_results.json")
