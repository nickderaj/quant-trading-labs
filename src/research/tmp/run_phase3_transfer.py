"""Phase 3 frozen-transfer check: the exact same ladder, unchanged, on
ETH/SOL/DOGE/BNB/XRP. Scoped down to 1d only (not all 4 intervals) per
NEW_PROMPT's own "reasonable time-boxing" allowance for the transfer check -
BTC already got the full 4-interval treatment in run_phase3.py; this checks
whether BTC's "no clear winner, HAR-RV/range/GARCH-normal all cluster
together, gamma/invgamma/lognorm distribution-fit rung is reliably worst"
pattern is a property of crypto generally or an artifact of BTC specifically.
"""

import json
import sys
import time

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import dist_lib as L
import numpy as np

import research

SYMBOLS = ["ETHUSDT", "SOLUSDT", "DOGEUSDT", "BNBUSDT", "XRPUSDT"]
INTERVAL = "1d"
MIN_TRAIN_DAYS = 90
CHEAP_REFIT_DAYS = 7
MLE_REFIT_DAYS = 30
MLE_MAX_TRAIN = 500

out: dict = {"interval": INTERVAL, "symbols": {}}

for symbol in SYMBOLS:
    t0 = time.time()
    df = L.build_asset_frame(symbol, INTERVAL, end=research.HOLDOUT_START)
    n = len(df)
    rv = df["rv_target"].to_numpy()
    ret = df["log_return"].fill_null(0.0).to_numpy()

    forecasts: dict[str, np.ndarray] = {}
    for w in [8, 24, 96]:
        forecasts[f"rung0_trailing_{w}"] = L.rung0_trailing_std(df, w).to_numpy()
    forecasts["rung1_ewma"] = L.rung1_ewma(df).to_numpy()

    har_df = L.make_har_features(df, INTERVAL)
    forecasts["rung2_har_rv"] = L.rolling_ols_refit(
        har_df,
        ["rv_d", "rv_w", "rv_m"],
        "rv_target",
        refit_every=CHEAP_REFIT_DAYS,
        min_train=MIN_TRAIN_DAYS,
    )

    range_df = L.range_estimator_forecasts(df, window=24)
    for name in ["fc_parkinson", "fc_gk", "fc_rs", "fc_yz"]:
        forecasts[f"rung3_{name}"] = range_df[name].to_numpy()

    for fam in ["gamma", "invgamma", "lognorm"]:
        forecasts[f"rung4_{fam}"] = L.rolling_rv_dist_forecast(
            rv,
            fam,
            refit_every=MLE_REFIT_DAYS,
            min_train=MIN_TRAIN_DAYS,
            max_train=MLE_MAX_TRAIN,
        )

    for innov in ["normal", "t", "skewt"]:
        fc, _fits = L.rolling_garch_forecast(
            ret,
            refit_every=MLE_REFIT_DAYS,
            min_train=MIN_TRAIN_DAYS,
            innovation=innov,
            max_train=MLE_MAX_TRAIN,
        )
        forecasts[f"rung5_garch_{innov}"] = fc

    forecasts["rung6_activity"] = L.activity_forecast(df, window=24).to_numpy()

    scores = {name: L.qlike_mse(rv, fc) for name, fc in forecasts.items()}

    def best_in_group(prefix, scores=scores):
        cands = [
            k
            for k in scores
            if k.startswith(prefix) and np.isfinite(scores[k]["qlike"])
        ]
        return min(cands, key=lambda k: scores[k]["qlike"]) if cands else None

    ladder_reps = {f"rung{i}": best_in_group(f"rung{i}_") for i in range(7)}

    def qlike_loss_series(fc, rv=rv):
        import distributions as dist

        mask = np.isfinite(rv) & np.isfinite(fc) & (fc > 0) & (rv > 0)
        arr = np.full(len(rv), np.nan)
        arr[mask] = dist.qlike(rv[mask], fc[mask])
        return arr

    present = [r for r in ladder_reps if ladder_reps[r] is not None]
    loss_cache = {r: qlike_loss_series(forecasts[ladder_reps[r]]) for r in present}
    qlike_by_rung = {r: float(np.nanmean(loss_cache[r])) for r in present}
    best_rung = min(qlike_by_rung, key=lambda r: qlike_by_rung[r])

    beats_all = True
    for r in present:
        if r == best_rung:
            continue
        la, lb = loss_cache[best_rung], loss_cache[r]
        both = np.isfinite(la) & np.isfinite(lb)
        if both.sum() <= 30:
            continue
        tstat, pval = L.diebold_mariano(la[both], lb[both])
        if not (pval < 0.05 and tstat < 0):
            beats_all = False

    out["symbols"][symbol] = {
        "n_obs": n,
        "qlike_by_rung": qlike_by_rung,
        "best_rung": best_rung,
        "best_rep": ladder_reps[best_rung],
        "beats_every_other_rung_significantly": beats_all,
        "elapsed_sec": time.time() - t0,
    }
    print(
        f"{symbol}: n={n} best={best_rung}/{ladder_reps[best_rung]} "
        f"beats_all={beats_all} elapsed={time.time() - t0:.1f}s"
    )

with open("src/research/tmp/phase3_transfer_results.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print("written phase3_transfer_results.json")
