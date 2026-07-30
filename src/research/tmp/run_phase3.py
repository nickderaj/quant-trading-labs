"""Phase 3 driver: volatility forecasting contest, full benchmark ladder,
BTC across all 4 intervals, rolling causal refits (never full-sample fits).

Refit cadence is declared in calendar days (not bars), so the number of
refits is bounded independent of bar resolution: cheap rungs (HAR-RV,
activity - a single lstsq call) refit weekly; the MLE rungs (RV
distribution fits, GARCH) refit monthly, on a trailing window capped at
500 bars, because a from-scratch skew-t GARCH MLE costs ~0.3-1s per fit on
this machine and refitting more often buys little (daily refit of a GARCH
that only actually reacts to information over weeks is cosmetic). All of
this is still causal - parameters are forward-filled between refits, so
the forecast used at bar t is whatever was last fit at or before t.

Evaluated on the full pre-holdout sample (rolling out-of-sample, not the
frozen holdout - see NEW_PROMPT's "A note on the holdout": Phase 3 has
thousands of scored observations so its power does not depend on holdout
purity the way a return-prediction backtest's does).
"""

import json
import sys
import time

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import numpy as np

import dist_lib as L
import research

SYMBOL = "BTCUSDT"
INTERVALS = ["1h", "4h", "12h", "1d"]
BARS_PER_DAY = {"1h": 24, "4h": 6, "12h": 2, "1d": 1}
MIN_TRAIN_DAYS = 90  # ~3 months of history before the first refit

# Refit cadence, in calendar days, so the *number* of refits is bounded
# independent of bar resolution (an MLE fit costs the same whether the
# window holds 500 hourly bars or 500 daily bars, but a naive "refit every
# k bars" with k fixed in bar units would do 24x more MLE fits at 1h than
# at 1d for the same wall-clock history). Cheap rungs (HAR/activity are a
# single lstsq call) can afford a tighter cadence than the MLE rungs
# (GARCH, RV distribution fits) on this machine (a Raspberry Pi shared with
# other processes) - a from-scratch skew-t GARCH MLE costs ~0.3-1s per fit.
CHEAP_REFIT_DAYS = 7  # HAR-RV, activity: weekly refit
MLE_REFIT_DAYS = 30  # GARCH, RV-distribution fits: monthly refit
MLE_MAX_TRAIN = 500  # cap the MLE fit's trailing window (bars), bounds cost

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

    forecasts: dict[str, np.ndarray] = {}

    # rung 0: trailing rolling std, 3 windows (in bars)
    for w in [8, 24, 96]:
        forecasts[f"rung0_trailing_{w}"] = L.rung0_trailing_std(df, w).to_numpy()

    # rung 1: EWMA / RiskMetrics
    forecasts["rung1_ewma"] = L.rung1_ewma(df).to_numpy()

    # rung 2: HAR-RV (daily/weekly/monthly RV components, rolling-refit OLS)
    har_df = L.make_har_features(df, interval)
    har_fc = L.rolling_ols_refit(
        har_df, ["rv_d", "rv_w", "rv_m"], "rv_target",
        refit_every=cheap_refit_every, min_train=min_train,
    )
    forecasts["rung2_har_rv"] = har_fc

    # rung 3: range estimators (Parkinson, GK, RS, YZ), rolling-mean, window = 1 day
    range_df = L.range_estimator_forecasts(df, window=bpd if bpd > 1 else 24)
    for name in ["fc_parkinson", "fc_gk", "fc_rs", "fc_yz"]:
        forecasts[f"rung3_{name}"] = range_df[name].to_numpy()

    # rung 4: distributional fits on RV (gamma/invgamma/lognorm), rolling refit
    for fam in ["gamma", "invgamma", "lognorm"]:
        forecasts[f"rung4_{fam}"] = L.rolling_rv_dist_forecast(
            rv, fam, refit_every=mle_refit_every, min_train=min_train, max_train=MLE_MAX_TRAIN,
        )

    # rung 5: GARCH(1,1), normal / t / skewt innovations, rolling refit
    garch_fits_summary = {}
    t_fits = []
    for innov in ["normal", "t", "skewt"]:
        fc, fits = L.rolling_garch_forecast(
            ret, refit_every=mle_refit_every, min_train=min_train, innovation=innov,
            max_train=MLE_MAX_TRAIN,
        )
        forecasts[f"rung5_garch_{innov}"] = fc
        if innov == "t":
            t_fits = fits
        garch_fits_summary[innov] = {
            "n_refits": len(fits),
            "last_params": {k: v for k, v in fits[-1].items() if k != "params"} if fits else None,
            # descriptive only ("what did the final fit look like") - NEVER used
            # for scoring below, since fits[-1] is only estimable from data at
            # the end of the sample and using it to score earlier bars would be
            # lookahead. See L.nu_path_from_fits for the causal path used to score.
            "last_nu": (fits[-1]["params"][3] if innov == "t" and fits else None),
        }

    # rung 6: activity-based (count / dispersion index -> RV), rolling-refit OLS
    forecasts["rung6_activity"] = L.activity_forecast(df, window=bpd if bpd > 1 else 24).to_numpy()  # uses cheap_refit_every internally via window

    # ---- score every rung: QLIKE (primary) + MSE, Mincer-Zarnowitz ----
    scores = {}
    for name, fc in forecasts.items():
        qm = L.qlike_mse(rv, fc)
        mz = L.mincer_zarnowitz(rv, fc)
        scores[name] = {**qm, "mz": mz}

    # ---- density scoring: normal-density scores for every rung's variance
    # forecast (fair common comparison); GARCH-t/skewt also scored under
    # their own fitted innovation distribution where available ----
    density = {}
    for name, fc in forecasts.items():
        density[name] = L.density_scores(ret, fc, family="normal")
    # GARCH-t under its own distribution, scored with a causal, forward-filled
    # df path (the df in force at bar t is whatever was last fit at or before
    # t - the same forward-fill the variance forecast already uses). Using
    # fits[-1]'s single final-window df to score the whole evaluation period
    # would be lookahead (bar 500 scored under a shape parameter estimated
    # from data through the end of the sample) - this is what nu_path_from_fits
    # fixes; see docs/02-estimation-and-fitting.md#forward-filling-parameters.
    if t_fits:
        nu_path = L.nu_path_from_fits(t_fits, n)
        density["rung5_garch_t_own_dist"] = L.density_scores(
            ret, forecasts["rung5_garch_t"], family="t", extra_params=(nu_path,)
        )

    # ---- ladder ordering: pick the single best-QLIKE representative of
    # each rung group for the Diebold-Mariano progression ----
    def best_in_group(prefix: str) -> str | None:
        cands = [k for k in scores if k.startswith(prefix) and np.isfinite(scores[k]["qlike"])]
        if not cands:
            return None
        return min(cands, key=lambda k: scores[k]["qlike"])

    ladder_reps = {
        "rung0": best_in_group("rung0_"),
        "rung1": best_in_group("rung1_"),
        "rung2": best_in_group("rung2_"),
        "rung3": best_in_group("rung3_"),
        "rung4": best_in_group("rung4_"),
        "rung5": best_in_group("rung5_"),
        "rung6": best_in_group("rung6_"),
    }

    # QLIKE loss series per representative, DM test on adjacent pairs
    def qlike_loss_series(fc: np.ndarray) -> np.ndarray:
        # actual > 0, not >= 0: QLIKE's log(ratio) term is undefined at
        # actual == 0 (see dist_lib.qlike_mse's docstring comment - same
        # frozen-price-bar bug).
        mask = np.isfinite(rv) & np.isfinite(fc) & (fc > 0) & (rv > 0)
        out_arr = np.full(len(rv), np.nan)
        import distributions as dist
        out_arr[mask] = dist.qlike(rv[mask], fc[mask])
        return out_arr

    rung_order = ["rung0", "rung1", "rung2", "rung3", "rung4", "rung5", "rung6"]
    dm_tests = {}
    prev_rung = None
    for r in rung_order:
        rep = ladder_reps[r]
        if rep is None:
            continue
        if prev_rung is not None and ladder_reps[prev_rung] is not None:
            la = qlike_loss_series(forecasts[ladder_reps[prev_rung]])
            lb = qlike_loss_series(forecasts[rep])
            both = np.isfinite(la) & np.isfinite(lb)
            if both.sum() > 30:
                tstat, pval = L.diebold_mariano(la[both], lb[both])
                dm_tests[f"{prev_rung}_vs_{r}"] = {
                    "prev": ladder_reps[prev_rung], "rung": rep,
                    "tstat": tstat, "pvalue": pval,
                    "qlike_prev": float(np.nanmean(la[both])),
                    "qlike_rung": float(np.nanmean(lb[both])),
                    "n": int(both.sum()),
                }
        prev_rung = r

    # ---- all-pairs DM among the 7 ladder representatives ----
    # Adjacent-only DM tests establish "beats the immediately preceding
    # rung" but the ladder's own rule ("a model is only reported as a
    # winner if it beats *every* rung below it") is not transitive from
    # adjacent comparisons alone (rung5 beating rung4, and rung3 beating
    # rung4, says nothing about rung5 vs rung3). All C(7,2)=21 pairs are
    # tested directly so "is there an actual winner" can be answered
    # honestly rather than assumed from the ladder order.
    all_pairs_dm = {}
    present_rungs = [r for r in rung_order if ladder_reps[r] is not None]
    loss_cache = {r: qlike_loss_series(forecasts[ladder_reps[r]]) for r in present_rungs}
    for i, ra in enumerate(present_rungs):
        for rb in present_rungs[i + 1:]:
            la, lb = loss_cache[ra], loss_cache[rb]
            both = np.isfinite(la) & np.isfinite(lb)
            if both.sum() > 30:
                tstat, pval = L.diebold_mariano(la[both], lb[both])
                all_pairs_dm[f"{ra}_vs_{rb}"] = {
                    "a": ladder_reps[ra], "b": ladder_reps[rb],
                    "tstat": tstat, "pvalue": pval,
                    "qlike_a": float(np.nanmean(la[both])), "qlike_b": float(np.nanmean(lb[both])),
                    "n": int(both.sum()),
                }

    # A rung is an honest "winner" only if its QLIKE is strictly lower than
    # every other rung AND every one of those pairwise differences is
    # significant (p < 0.05) in its favour.
    qlike_by_rung = {r: float(np.nanmean(loss_cache[r])) for r in present_rungs}
    best_rung = min(qlike_by_rung, key=qlike_by_rung.get)
    beats_all = True
    for r in present_rungs:
        if r == best_rung:
            continue
        key = f"{best_rung}_vs_{r}" if f"{best_rung}_vs_{r}" in all_pairs_dm else f"{r}_vs_{best_rung}"
        entry = all_pairs_dm.get(key)
        if entry is None:
            continue
        # sign convention: tstat < 0 means "a" (first-listed) has lower loss
        a_is_best = entry["a"] == ladder_reps[best_rung]
        best_wins_sig = entry["pvalue"] < 0.05 and (
            (a_is_best and entry["tstat"] < 0) or (not a_is_best and entry["tstat"] > 0)
        )
        if not best_wins_sig:
            beats_all = False
            break
    winner_verdict = {
        "best_by_qlike": best_rung, "best_rep": ladder_reps[best_rung],
        "qlike_by_rung": qlike_by_rung, "beats_every_other_rung_significantly": beats_all,
    }

    res = {
        "n_obs": n,
        "cheap_refit_every_bars": cheap_refit_every,
        "mle_refit_every_bars": mle_refit_every,
        "min_train_bars": min_train,
        "scores": scores,
        "density": density,
        "ladder_reps": ladder_reps,
        "dm_tests": dm_tests,
        "all_pairs_dm": all_pairs_dm,
        "winner_verdict": winner_verdict,
        "garch_fits_summary": garch_fits_summary,
        "elapsed_sec": time.time() - t0,
    }
    out["intervals"][interval] = res
    print(f"{interval}: n={n} elapsed={res['elapsed_sec']:.1f}s ladder_reps={ladder_reps}")

with open("src/research/tmp/phase3_results.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print("written phase3_results.json")
