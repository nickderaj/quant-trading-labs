"""Phase 7 -- volatility identity eq (23), an independent falsification check.
Uses the fitted (sigma_S, sigma_L, rho, beta) from the kalman config (train sample)
to predict sigma_F(tau) and compares against realised volatility by maturity
bucket, computed from the panel's own log-returns per near-constant-maturity
bucket. This is a free check: the price fit never saw realised vol.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib23

TMP = Path(__file__).resolve().parent
VARIANT = "all"
L_NAME = "kalman"
BUCKETS = [(0, 1), (1, 2), (2, 4), (4, 9)]

panel = pd.read_parquet(TMP / f"phase_1_23_panel_{VARIANT}.parquet")
panel["date"] = pd.to_datetime(panel["date"])
panel = panel[(panel["date"] >= lib23.TRAIN_START) & (panel["date"] <= lib23.TRAIN_END)]
fit26 = pd.read_parquet(TMP / f"phase_3_23_fit26_{VARIANT}_{L_NAME}.parquet")

beta_med = float(fit26["beta"].median())
nu_med = float(fit26["nu"].median())
sl_report = json.loads((TMP / "phase_2_23_SL_report.json").read_text())["variants"][
    VARIANT
]["joint_ls_daily" if L_NAME == "joint_ls_daily" else L_NAME]
sigma_S = sl_report["sigma_S_annualised"]
sigma_L = sl_report["sigma_L_annualised"]
rho = sl_report["rho"]

t0 = time.time()

# Realised vol by maturity bucket: for each contract ticker, compute log-return
# series of its own close price while tau sits in that bucket, pool across
# contracts and tickers, annualise.
panel = panel.sort_values(["ticker", "date"])
panel["log_ret"] = panel.groupby("ticker")["log_close"].diff()

results = {}
for lo, hi in BUCKETS:
    mask = (panel["tau"] >= lo) & (panel["tau"] < hi) & panel["log_ret"].notna()
    rets = panel.loc[mask, "log_ret"]
    realised_vol = float(rets.std() * np.sqrt(252)) if len(rets) > 30 else None
    tau_mid = 0.5 * (lo + hi)
    predicted_vol = float(
        lib23.vol_identity_sigma_F(
            np.array([tau_mid]), beta_med, sigma_S, sigma_L, rho
        )[0]
    )
    results[f"{lo}-{hi}y"] = {
        "n_obs": int(mask.sum()),
        "realised_vol_annualised": realised_vol,
        "predicted_vol_annualised": predicted_vol,
        "abs_gap": (
            abs(realised_vol - predicted_vol) if realised_vol is not None else None
        ),
    }

report = {
    "config": {
        "variant": VARIANT,
        "L_estimator": L_NAME,
        "beta_median": beta_med,
        "nu_median": nu_med,
        "sigma_S": sigma_S,
        "sigma_L": sigma_L,
        "rho": rho,
    },
    "by_bucket": results,
    "wall_time_seconds": time.time() - t0,
}
out = TMP / "phase_7_23_vol_identity_report.json"
out.write_text(json.dumps(report, indent=2, default=str))
print(json.dumps(report, indent=2))
print(f"wrote {out}")
