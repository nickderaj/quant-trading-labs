"""Phase 2 -- estimate the spot proxy S and long-term anchor L, three estimators,
for each liquidity variant. Fits are computed mechanically over the full sample
(needed downstream at Phase 9), but every reported statistic in this phase's JSON
is restricted to the TRAIN window only -- the holdout is not inspected here.
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
VARIANTS = ["all", "vol_gt_0", "vol_gt_100"]

# Gabillon's own reported figures, for comparison in the write-up.
PAPER_FIGURES = {
    "method_a": {"sigma_L": 0.176, "sigma_S": 0.464, "rho": 0.21},
    "method_b_raw": {"sigma_L": 0.581, "rho": 0.42},
    "method_b_filtered": {"sigma_L": 0.337},
}


def summarize(S: pd.Series, L: pd.Series, label: str) -> dict:
    df = pd.concat([S.rename("S"), L.rename("L")], axis=1).dropna()
    df = df[(df.index >= lib23.TRAIN_START) & (df.index <= lib23.TRAIN_END)]
    if len(df) < 30:
        return {"label": label, "n": len(df), "insufficient": True}
    lnS, lnL = np.log(df["S"]), np.log(df["L"])
    rS, rL = lnS.diff().dropna(), lnL.diff().dropna()
    joined = pd.concat([rS.rename("rS"), rL.rename("rL")], axis=1).dropna()
    sigma_S = float(rS.std() * np.sqrt(252))
    sigma_L = float(rL.std() * np.sqrt(252))
    rho = float(joined["rS"].corr(joined["rL"]))
    return {
        "label": label,
        "n": len(df),
        "sigma_S_annualised": sigma_S,
        "sigma_L_annualised": sigma_L,
        "rho": rho,
    }


report: dict = {"paper_figures": PAPER_FIGURES, "variants": {}}
t_start = time.time()

for variant in VARIANTS:
    panel = pd.read_parquet(TMP / f"phase_1_23_panel_{variant}.parquet")
    panel["date"] = pd.to_datetime(panel["date"])

    S = lib23.compute_spot_series(panel)
    S.to_frame("S").to_parquet(TMP / f"phase_2_23_spot_{variant}.parquet")

    var_report = {}

    # (a) joint LS, monthly-fixed and daily-free
    fit_daily = lib23.fit_L_joint_ls_panel(panel, monthly_fixed=False)
    fit_monthly = lib23.fit_L_joint_ls_panel(panel, monthly_fixed=True)
    fit_daily.to_parquet(TMP / f"phase_2_23_L_joint_ls_daily_{variant}.parquet")
    fit_monthly.to_parquet(TMP / f"phase_2_23_L_joint_ls_monthly_{variant}.parquet")
    var_report["joint_ls_daily"] = summarize(
        fit_daily["S"], fit_daily["L"], "joint_ls_daily"
    )
    var_report["joint_ls_monthly"] = summarize(
        fit_monthly["S"], fit_monthly["L"], "joint_ls_monthly"
    )

    # (b) long-end extrapolation
    fit_longend = lib23.fit_L_longend(panel)
    fit_longend.to_parquet(TMP / f"phase_2_23_L_longend_{variant}.parquet")
    n_infinite = int((~fit_longend["valid"]).sum())
    valid_frac = float(fit_longend["valid"].mean())
    var_report["longend"] = summarize(S, fit_longend["L"], "longend")
    var_report["longend"]["n_carried_forward_infinite_L"] = n_infinite
    var_report["longend"]["valid_fraction"] = valid_frac

    # (c) Kalman filter
    fit_kalman = lib23.kalman_S_L(panel, S)
    fit_kalman.to_parquet(TMP / f"phase_2_23_L_kalman_{variant}.parquet")
    var_report["kalman"] = summarize(fit_kalman["S"], fit_kalman["L"], "kalman")

    report["variants"][variant] = var_report
    print(f"{variant}: done at t={time.time() - t_start:.1f}s")

report["wall_time_seconds"] = time.time() - t_start
out = TMP / "phase_2_23_SL_report.json"
out.write_text(json.dumps(report, indent=2, default=str))
print(f"wrote {out}")
