"""Phase 4 frozen-transfer check: HMM-Gaussian (the model that showed the
clearest persistence/vol-discrimination gain over the baseline on BTC) and
the baseline threshold, unchanged, on ETH/SOL/DOGE/BNB/XRP at 1d. Scoped
down the same way as the Phase 3 transfer check.
"""

import json
import sys
import time

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import numpy as np

import dist_lib as L
import research
from scipy import stats as st

# Inlined from run_phase4.py rather than imported: that module runs its full
# BTC 4-interval driver at import time (script-style, not `if __name__`-
# guarded), so importing it would silently re-run several minutes of
# compute. Kept in sync by hand - same constants/functions, unchanged.

MIN_TRAIN_DAYS = 90
REFIT_DAYS = 30
MAX_TRAIN = 500
THRESH_WINDOW_DAYS = 90


def rolling_refit_states(x, k, refit_every, min_train, max_train, emission="gaussian", fit_fn=None):
    n = len(x)
    hard = np.full(n, np.nan)
    fit_log = []
    fit = None
    alpha = None
    for t in range(n):
        if t >= min_train and (fit is None or t % refit_every == 0):
            start = max(0, t - max_train)
            window = x[start:t]
            window = window[np.isfinite(window)]
            if len(window) >= min(min_train, max_train) // 2:
                new_fit = fit_fn(window, k)
                if new_fit is not None:
                    fit = new_fit
                    alpha = np.array(fit.get("pi", fit.get("weights")))
                    fit_log.append({"t": t})
        if fit is None or not np.isfinite(x[t]):
            continue
        if emission == "mixture":
            p = L.gmm_posterior(x[t], fit)
        else:
            alpha = L.hmm_filter_step(alpha, x[t], fit)
            p = alpha
        hard[t] = int(np.argmax(p))
    return None, hard, fit_log


def geometric_duration_test(hard_state):
    s = hard_state[np.isfinite(hard_state)]
    if len(s) < 20:
        return {}
    runs, cur = [], 1
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
    return {"n_runs": len(runs), "mean_duration": float(np.mean(runs)),
            "ks_pvalue": float(ks.pvalue)}


def predicts_vol_and_direction(returns, rv_fwd, hard_state, k):
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
    return {"vol_kruskal": vol_test, "direction_anova": dir_test}


SYMBOLS = ["ETHUSDT", "SOLUSDT", "DOGEUSDT", "BNBUSDT", "XRPUSDT"]
INTERVAL = "1d"
bpd = 1
refit_every = REFIT_DAYS * bpd
min_train = MIN_TRAIN_DAYS * bpd
thresh_window = THRESH_WINDOW_DAYS * bpd

out: dict = {"interval": INTERVAL, "symbols": {}}

for symbol in SYMBOLS:
    t0 = time.time()
    df = L.build_asset_frame(symbol, INTERVAL, end=research.HOLDOUT_START)
    n = len(df)
    ret = df["log_return"].to_numpy()
    rv = df["rv_target"].to_numpy()
    rv_fwd = np.concatenate([rv[1:], [np.nan]])
    ret_fwd = np.concatenate([ret[1:], [np.nan]])

    med = df["rv_target"].rolling_median(window_size=thresh_window, min_samples=thresh_window // 2).shift(1).to_numpy()
    thr_state = np.where(np.isfinite(med), (rv > med).astype(float), np.nan)
    baseline = {
        "geometric_duration": geometric_duration_test(thr_state),
        "predicts": predicts_vol_and_direction(ret_fwd, rv_fwd, thr_state, 2),
    }

    def fit_fn(window, k):
        return L.fit_hmm(window, k=k, n_iter=15, emission="gaussian", t_df=None)
    probs, hard, fit_log = rolling_refit_states(
        ret, 2, refit_every, min_train, MAX_TRAIN, emission="hmm", fit_fn=fit_fn,
    )
    hmm = {
        "n_refits": len(fit_log),
        "geometric_duration": geometric_duration_test(hard),
        "predicts": predicts_vol_and_direction(ret_fwd, rv_fwd, hard, 2),
    }

    out["symbols"][symbol] = {
        "n_obs": n, "baseline_threshold": baseline, "hmm_gaussian": hmm,
        "elapsed_sec": time.time() - t0,
    }
    print(f"{symbol}: n={n} elapsed={time.time()-t0:.1f}s "
          f"baseline_vol_p={baseline['predicts']['vol_kruskal'].get('pvalue')} "
          f"hmm_vol_p={hmm['predicts']['vol_kruskal'].get('pvalue')} "
          f"baseline_dir_p={baseline['predicts']['direction_anova'].get('pvalue')} "
          f"hmm_dir_p={hmm['predicts']['direction_anova'].get('pvalue')}")

with open("src/research/tmp/phase4_transfer_results.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print("written phase4_transfer_results.json")
