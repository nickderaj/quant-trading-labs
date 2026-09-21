"""Phase 5 -- model (28), the decaying short-term shock. Scored against model (26)
in the pre-registered crisis windows (2014-15 collapse, 2020 COVID, 2022) and a
calm-period control sample, train-sample dates only. Primary config: kalman L
estimator, "all" liquidity variant (Phase 3/4's numerically sound survivor).
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib23

TMP = Path(__file__).resolve().parent
VARIANT = "all"
L_NAME = "kalman"

prereg = json.loads((TMP / "phase_0_23_preregistration.json").read_text())
windows = prereg["declared_windows_for_phase5"]

panel = pd.read_parquet(TMP / f"phase_1_23_panel_{VARIANT}.parquet")
panel["date"] = pd.to_datetime(panel["date"])
fit26 = pd.read_parquet(TMP / f"phase_3_23_fit26_{VARIANT}_{L_NAME}.parquet")
fit26["date"] = pd.to_datetime(fit26["date"])
fit26 = fit26.set_index("date")

# t (calendar time in years since train start) for the exp(-eta*t) shock term.
t0_date = lib23.TRAIN_START


def fit_model28_day(
    g: pd.DataFrame, s: float, ell: float, beta: float, nu: float, t: float
) -> dict[str, float] | None:
    tau = g.sort_values("tau")["tau"].to_numpy()
    actual_ln = g.sort_values("tau")["log_close"].to_numpy()

    def resid(params: np.ndarray) -> np.ndarray:
        theta, log_eta = params
        eta = np.exp(log_eta)
        pred = lib23.model28_lnF(s, ell, tau, beta, nu, theta, eta, t=t)
        return pred - actual_ln

    x0 = np.array([0.0, np.log(max(beta * 2, 0.5))])
    try:
        sol = least_squares(resid, x0, max_nfev=200)
    except (RuntimeError, ValueError):
        return None
    theta, log_eta = sol.x
    eta = float(np.exp(log_eta))
    pred_ln = lib23.model28_lnF(s, ell, tau, beta, nu, theta, eta, t=t)
    pred = np.exp(pred_ln)
    actual = np.exp(actual_ln)
    rmse28 = float(np.sqrt(np.mean((pred - actual) ** 2)))
    return {"theta": float(theta), "eta": eta, "rmse28": rmse28}


def run_window(label: str, start: str, end: str) -> dict[str, Any]:
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    dates = [d for d in fit26.index if lo <= d <= hi and d <= lib23.TRAIN_END]
    rows = []
    for date in dates:
        g = panel[panel["date"] == date]
        if len(g) < 8:
            continue
        s = float(cast(float, fit26.loc[date, "S"]))
        ell = float(cast(float, fit26.loc[date, "L"]))
        beta = float(cast(float, fit26.loc[date, "beta"]))
        nu = float(cast(float, fit26.loc[date, "nu"]))
        rmse26 = float(cast(float, fit26.loc[date, "rmse"]))
        t = (date - t0_date).days / 365.25
        res28 = fit_model28_day(g, s, ell, beta, nu, t)
        if res28 is None:
            continue
        rows.append({"date": str(date.date()), "rmse26": float(rmse26), **res28})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return {"label": label, "n_days": 0}
    return {
        "label": label,
        "n_days": len(df),
        "median_rmse26": float(df["rmse26"].median()),
        "median_rmse28": float(df["rmse28"].median()),
        "pct_days_28_better": float((df["rmse28"] < df["rmse26"]).mean()),
        "median_theta": float(df["theta"].median()),
        "median_eta": float(df["eta"].median()),
    }


# Calm-period control: train dates NOT in any declared crisis window.
crisis_mask = pd.Series(False, index=fit26.index)
for w in windows.values():
    lo, hi = pd.Timestamp(w[0]), pd.Timestamp(w[1])
    crisis_mask |= (fit26.index >= lo) & (fit26.index <= hi)
calm_dates = fit26.index[~crisis_mask]
rng = np.random.default_rng(23)
calm_sample = pd.Index(
    rng.choice(calm_dates, size=min(150, len(calm_dates)), replace=False)
).sort_values()

report: dict[str, Any] = {}
t_start = time.time()
for label, (start, end) in windows.items():
    report[label] = run_window(label, start, end)
    print(label, report[label], f"t={time.time() - t_start:.1f}s")

# calm control
rows = []
for date in calm_sample:
    g = panel[panel["date"] == date]
    if len(g) < 8:
        continue
    s = float(cast(float, fit26.loc[date, "S"]))
    ell = float(cast(float, fit26.loc[date, "L"]))
    beta = float(cast(float, fit26.loc[date, "beta"]))
    nu = float(cast(float, fit26.loc[date, "nu"]))
    rmse26 = float(cast(float, fit26.loc[date, "rmse"]))
    t = (date - t0_date).days / 365.25
    res28 = fit_model28_day(g, s, ell, beta, nu, t)
    if res28 is None:
        continue
    rows.append({"date": str(date.date()), "rmse26": float(rmse26), **res28})
df_calm = pd.DataFrame(rows)
report["calm_control"] = {
    "label": "calm_control",
    "n_days": len(df_calm),
    "median_rmse26": float(df_calm["rmse26"].median()) if len(df_calm) else None,
    "median_rmse28": float(df_calm["rmse28"].median()) if len(df_calm) else None,
    "pct_days_28_better": float((df_calm["rmse28"] < df_calm["rmse26"]).mean())
    if len(df_calm)
    else None,
}
print("calm_control", report["calm_control"], f"t={time.time() - t_start:.1f}s")

report["wall_time_seconds"] = time.time() - t_start
out = TMP / "phase_5_23_model28_report.json"
out.write_text(json.dumps(report, indent=2, default=str))
print(f"wrote {out}")
