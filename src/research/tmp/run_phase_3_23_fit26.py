"""Phase 3 -- fit model (26) per day, train sample only, for every pre-registered
(liquidity variant, L estimator) combination. beta is refit per day given the L
estimator's (S, L); nu comes from a trailing rolling realised-vol estimate of the
same (S, L) series, causal (no lookahead: nu at day t uses only returns strictly
before t).
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib23

TMP = Path(__file__).resolve().parent
VARIANTS = ["all", "vol_gt_0", "vol_gt_100"]
L_ESTIMATORS = {
    "joint_ls_monthly": "phase_2_23_L_joint_ls_monthly_{v}.parquet",
    "joint_ls_daily": "phase_2_23_L_joint_ls_daily_{v}.parquet",
    "longend": "phase_2_23_L_longend_{v}.parquet",
    "kalman": "phase_2_23_L_kalman_{v}.parquet",
}
ROLL_WINDOW = 63
BUCKETS = [(0, 1), (1, 2), (2, 4), (4, 100)]


def bucket_label(lo: int, hi: int) -> str:
    return f"{lo}-{hi}y" if hi < 100 else f"{lo}y+"


def compute_nu_series(S: pd.Series, L: pd.Series) -> pd.DataFrame:
    lnS, lnL = np.log(S), np.log(L)
    rS, rL = lnS.diff(), lnL.diff()
    sig_S2 = (rS.rolling(ROLL_WINDOW).var() * 252).shift(1)
    sig_L2 = (rL.rolling(ROLL_WINDOW).var() * 252).shift(1)
    joined = pd.concat([rS, rL], axis=1)
    joined.columns = ["rS", "rL"]
    rho = joined["rS"].rolling(ROLL_WINDOW).corr(joined["rL"]).shift(1)
    sig_S = np.sqrt(sig_S2)
    sig_L = np.sqrt(sig_L2)
    nu = sig_S2 + sig_L2 - 2 * rho.fillna(0.0) * sig_S * sig_L
    return pd.DataFrame({"nu": nu, "sigma_S": sig_S, "sigma_L": sig_L, "rho": rho})


def run_one(variant: str, l_name: str) -> dict[str, Any]:
    panel = pd.read_parquet(TMP / f"phase_1_23_panel_{variant}.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[
        (panel["date"] >= lib23.TRAIN_START) & (panel["date"] <= lib23.TRAIN_END)
    ]

    S = pd.read_parquet(TMP / f"phase_2_23_spot_{variant}.parquet")["S"]
    L_df = pd.read_parquet(TMP / L_ESTIMATORS[l_name].format(v=variant))
    L = L_df["L"].ffill()

    nu_df = compute_nu_series(S, L)

    rows = []
    for date_key, g in panel.groupby("date"):
        date = cast(pd.Timestamp, date_key)
        if date not in S.index or date not in L.index or date not in nu_df.index:
            continue
        s, ell = float(S.loc[date]), float(L.loc[date])
        if not (np.isfinite(s) and np.isfinite(ell) and s > 0 and ell > 0):
            continue
        nu = float(cast(float, nu_df.loc[date, "nu"]))
        if not np.isfinite(nu) or nu <= 0:
            continue
        beta, _sse = lib23.fit_beta_given_SL(g, s, ell)
        tau = g.sort_values("tau")["tau"].to_numpy()
        actual = g.sort_values("tau")["close"].to_numpy()
        pred = np.exp(lib23.model26_lnF(s, ell, tau, beta, nu))
        err = pred - actual
        rmse = float(np.sqrt(np.mean(err**2)))
        maxabs = float(np.max(np.abs(err)))
        bucket_rmse: dict[str, float] = {}
        for lo, hi in BUCKETS:
            mask = (tau >= lo) & (tau < hi)
            if mask.sum() > 0:
                bucket_rmse[bucket_label(lo, hi)] = float(
                    np.sqrt(np.mean(err[mask] ** 2))
                )
        rows.append(
            {
                "date": str(date.date()),
                "beta": beta,
                "L": float(ell),
                "S": float(s),
                "nu": float(nu),
                "rmse": rmse,
                "max_abs_error": maxabs,
                "bucket_rmse": bucket_rmse,
            }
        )

    df = pd.DataFrame(rows)
    out_parquet = TMP / f"phase_3_23_fit26_{variant}_{l_name}.parquet"
    df.to_parquet(out_parquet)
    if len(df) == 0:
        return {"variant": variant, "L_estimator": l_name, "n_days": 0}
    return {
        "variant": variant,
        "L_estimator": l_name,
        "n_days": len(df),
        "median_rmse": float(df["rmse"].median()),
        "mean_rmse": float(df["rmse"].mean()),
        "p90_rmse": float(df["rmse"].quantile(0.9)),
        "median_beta": float(df["beta"].median()),
        "median_max_abs_error": float(df["max_abs_error"].median()),
    }


report: dict[str, Any] = {"combos": []}
t0 = time.time()
for variant in VARIANTS:
    for l_name in L_ESTIMATORS:
        res = run_one(variant, l_name)
        report["combos"].append(res)
        print(
            res.get("variant"),
            res.get("L_estimator"),
            "median_rmse=",
            res.get("median_rmse"),
            f"t={time.time() - t0:.1f}s",
        )

report["wall_time_seconds"] = time.time() - t0
out = TMP / "phase_3_23_fit26_report.json"
out.write_text(json.dumps(report, indent=2, default=str))
print(f"wrote {out}")
