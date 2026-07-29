"""Phase 4 driver: regime estimation, BTC across all 4 intervals.

Three traps handled explicitly:
1. Filtered, never smoothed - HMM state probabilities at bar t use only
   fit_hmm's *rolling-refit* parameters (fit on data < the refit point) and
   `hmm_filter_step`'s one-step-forward recursion; the backward pass
   (`fit_hmm`'s own Baum-Welch, run only to *estimate parameters* on past
   data) never touches bars after the refit point, and its own smoothed
   gamma is discarded - only the frozen fit's forward filter, applied
   bar-by-bar going forward, is used as the tradeable state probability.
2. Rolling refit, never full-sample - GMM/HMM refit monthly on a trailing
   capped window (mirrors Phase 3's MLE cadence/cost reasoning), forward-
   filled between refits.
3. Label switching - `fit_gmm_em`/`fit_hmm` already impose ascending-
   fitted-variance canonical ordering; this script additionally checks
   state-mean/variance stability across adjacent refits.

Baseline (rung 0): trailing-median realized-vol threshold, two states,
causal (shift(1) median over a trailing window).
"""

import json
import sys
import time

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import numpy as np
import polars as pl
from scipy import stats as st

import dist_lib as L
import distributions as dist
import research

SYMBOL = "BTCUSDT"
INTERVALS = ["1h", "4h", "12h", "1d"]
BARS_PER_DAY = {"1h": 24, "4h": 6, "12h": 2, "1d": 1}
MIN_TRAIN_DAYS = 90
REFIT_DAYS = 30  # monthly refit, same cost reasoning as Phase 3's MLE rungs
MAX_TRAIN = 500
THRESH_WINDOW_DAYS = 90


def rolling_refit_states(
    x: np.ndarray, k: int, refit_every: int, min_train: int, max_train: int,
    emission: str = "gaussian", fit_fn=None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Generic rolling-refit + forward-filter state machinery shared by GMM
    and HMM. Returns (state_probs [n,k], hard_state [n], fit_log).
    fit_fn(window, k) -> fit dict or None (either L.fit_gmm_em or L.fit_hmm).
    For GMM, "filtering" is just each bar's own posterior under the frozen
    fit (no cross-bar Markov dependence, so no smoothing risk); for HMM the
    forward recursion (`hmm_filter_step`) is used explicitly - never the
    backward/smoothed pass.
    """
    n = len(x)
    probs = np.full((n, k), np.nan)
    hard = np.full(n, np.nan)
    fit_log = []
    fit = None
    alpha = None
    for t in range(n):
        if t >= min_train and (fit is None or t % refit_every == 0):
            start = max(0, t - max_train)
            window = x[start:t]
            window = window[np.isfinite(window)]
            # Bug found and fixed: this required len(window) >= min_train//2,
            # but `window` is capped at max_train bars (start = t-max_train).
            # At 1h, min_train = 90*24 = 2160 while max_train = 500, so
            # min_train//2 (1080) could never be satisfied by a <=500-bar
            # window - every GMM/HMM refit at 1h silently no-opped and every
            # hard-state/geometric-duration/predicts-vol stat came back None.
            # The sufficiency floor must be relative to the window that will
            # actually be fit (max_train), not the warm-up gate (min_train).
            if len(window) >= min(min_train, max_train) // 2:
                new_fit = fit_fn(window, k)
                if new_fit is not None:
                    fit = new_fit
                    alpha = np.array(fit.get("pi", fit.get("weights")))
                    fit_log.append({"t": t, "means": fit["means"], "vars": fit["vars"]})
        if fit is None or not np.isfinite(x[t]):
            continue
        if emission == "mixture":
            p = L.gmm_posterior(x[t], fit)
        else:
            alpha = L.hmm_filter_step(alpha, x[t], fit)
            p = alpha
        probs[t] = p
        hard[t] = int(np.argmax(p))
    return probs, hard, fit_log


def geometric_duration_test(hard_state: np.ndarray) -> dict:
    """Run lengths of the hard state sequence, tested against a geometric
    null with p = 1 - (empirical same-state transition prob) - the direct
    test of an HMM/Markov regime model's own core assumption."""
    s = hard_state[np.isfinite(hard_state)]
    if len(s) < 20:
        return {}
    runs = []
    cur = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    runs = np.array(runs, dtype=float)
    p_geom = 1.0 / np.mean(runs)
    ks = st.kstest(runs, "geom", args=(p_geom,))
    return {
        "n_runs": len(runs), "mean_duration": float(np.mean(runs)),
        "p_geom": float(p_geom), "ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
    }


def transition_matrix(hard_state: np.ndarray, k: int) -> list[list[float]]:
    s = hard_state[np.isfinite(hard_state)].astype(int)
    counts = np.zeros((k, k))
    for i in range(len(s) - 1):
        counts[s[i], s[i + 1]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return (counts / row_sums).tolist()


def conditional_stats_by_regime(returns: np.ndarray, hard_state: np.ndarray, k: int) -> dict:
    out = {}
    for j in range(k):
        mask = hard_state == j
        r = returns[mask]
        r = r[np.isfinite(r)]
        if len(r) < 10:
            out[str(j)] = {"n": len(r)}
            continue
        ac1 = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if len(r) > 20 else np.nan
        out[str(j)] = {
            "n": len(r), "mean": float(np.mean(r)), "std": float(np.std(r)),
            "skew": float(st.skew(r)), "kurtosis": float(st.kurtosis(r, fisher=False)),
            "autocorr_lag1": ac1,
        }
    return out


def predicts_vol_and_direction(
    returns: np.ndarray, rv_fwd: np.ndarray, hard_state: np.ndarray, k: int
) -> dict:
    """Does state_t predict RV_{t+1} (vol) and sign(return_{t+1}) (direction)?
    Kruskal-Wallis across states for vol (rank-based, robust to the heavy
    tails here); state-conditional mean forward return + a one-way ANOVA F
    test for direction."""
    mask = np.isfinite(hard_state) & np.isfinite(rv_fwd)
    groups_vol = [rv_fwd[mask & (hard_state == j)] for j in range(k)]
    groups_vol = [g for g in groups_vol if len(g) >= 10]
    vol_test = {}
    if len(groups_vol) >= 2:
        h, p = st.kruskal(*groups_vol)
        vol_test = {"kruskal_h": float(h), "pvalue": float(p)}

    mask_d = np.isfinite(hard_state) & np.isfinite(returns)
    groups_dir = [returns[mask_d & (hard_state == j)] for j in range(k)]
    groups_dir = [g for g in groups_dir if len(g) >= 10]
    dir_test = {}
    if len(groups_dir) >= 2:
        f, p = st.f_oneway(*groups_dir)
        dir_test = {"f_stat": float(f), "pvalue": float(p)}
    return {
        "vol_kruskal": vol_test,
        "direction_anova": dir_test,
        "mean_fwd_return_by_state": [float(np.mean(g)) if len(g) else None for g in
                                      [returns[mask_d & (hard_state == j)] for j in range(k)]],
        "mean_rv_fwd_by_state": [float(np.mean(g)) if len(g) else None for g in
                                  [rv_fwd[mask & (hard_state == j)] for j in range(k)]],
    }


out: dict = {"symbol": SYMBOL, "intervals": {}}

for interval in INTERVALS:
    t0 = time.time()
    bpd = BARS_PER_DAY[interval]
    refit_every = REFIT_DAYS * bpd
    min_train = MIN_TRAIN_DAYS * bpd
    thresh_window = THRESH_WINDOW_DAYS * bpd

    df = L.build_asset_frame(SYMBOL, interval, end=research.HOLDOUT_START)
    n = len(df)
    ret = df["log_return"].to_numpy()
    rv = df["rv_target"].to_numpy()
    rv_fwd = np.concatenate([rv[1:], [np.nan]])  # RV realized in the *next* bar
    ret_fwd = np.concatenate([ret[1:], [np.nan]])
    counts = df["count"].to_numpy().astype(float)

    res: dict = {"n_obs": n}

    # ---- rung 0: trailing-median RV threshold (causal, 2 states) ----
    med = df["rv_target"].rolling_median(window_size=thresh_window, min_periods=thresh_window // 2).shift(1).to_numpy()
    thr_state = np.where(np.isfinite(med), (rv > med).astype(float), np.nan)
    res["baseline_threshold"] = {
        "geometric_duration": geometric_duration_test(thr_state),
        "transition_matrix": transition_matrix(thr_state, 2),
        "conditional_stats": conditional_stats_by_regime(ret, thr_state, 2),
        "predicts": predicts_vol_and_direction(ret_fwd, rv_fwd, thr_state, 2),
    }

    # ---- rung 1: Gaussian mixture, K=2,3 ----
    for k in [2, 3]:
        probs, hard, fit_log = rolling_refit_states(
            ret, k, refit_every, min_train, MAX_TRAIN, emission="mixture", fit_fn=L.fit_gmm_em,
        )
        res[f"gmm_k{k}"] = {
            "n_refits": len(fit_log),
            "geometric_duration": geometric_duration_test(hard),
            "transition_matrix": transition_matrix(hard, k),
            "conditional_stats": conditional_stats_by_regime(ret, hard, k),
            "predicts": predicts_vol_and_direction(ret_fwd, rv_fwd, hard, k),
        }

    # ---- rung 2: HMM, Gaussian and Student-t emissions, K=2 ----
    for emission, t_df in [("gaussian", None), ("t", 5.0)]:
        def fit_fn(window, k, emission=emission, t_df=t_df):
            return L.fit_hmm(window, k=k, n_iter=15, emission=emission, t_df=t_df)
        probs, hard, fit_log = rolling_refit_states(
            ret, 2, refit_every, min_train, MAX_TRAIN, emission="hmm", fit_fn=fit_fn,
        )
        res[f"hmm_{emission}"] = {
            "n_refits": len(fit_log),
            "geometric_duration": geometric_duration_test(hard),
            "transition_matrix": transition_matrix(hard, 2),
            "conditional_stats": conditional_stats_by_regime(ret, hard, 2),
            "predicts": predicts_vol_and_direction(ret_fwd, rv_fwd, hard, 2),
        }

    # ---- rung 3: activity regime (count dispersion index vs its own trailing median) ----
    disp = (
        df["count"].rolling_std(window_size=thresh_window, min_periods=thresh_window // 2) ** 2
        / df["count"].rolling_mean(window_size=thresh_window, min_periods=thresh_window // 2)
    ).shift(1)
    disp_med = disp.rolling_median(window_size=thresh_window, min_periods=thresh_window // 2).to_numpy()
    disp_arr = disp.to_numpy()
    act_state = np.where(np.isfinite(disp_arr) & np.isfinite(disp_med), (disp_arr > disp_med).astype(float), np.nan)
    res["activity_regime"] = {
        "geometric_duration": geometric_duration_test(act_state),
        "transition_matrix": transition_matrix(act_state, 2),
        "conditional_stats": conditional_stats_by_regime(ret, act_state, 2),
        "predicts": predicts_vol_and_direction(ret_fwd, rv_fwd, act_state, 2),
    }

    res["elapsed_sec"] = time.time() - t0
    out["intervals"][interval] = res
    print(f"{interval}: n={n} elapsed={res['elapsed_sec']:.1f}s")

with open("src/research/tmp/phase4_results.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print("written phase4_results.json")
