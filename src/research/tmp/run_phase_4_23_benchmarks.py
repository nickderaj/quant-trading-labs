"""Phase 4 -- benchmark comparison, full-curve in-sample RMSE, train sample.

Scoped to the two configurations Phase 3 identified as numerically sound (kalman
and joint_ls_monthly L estimators, "all" liquidity variant) -- Phase 3 showed
joint_ls_daily numerically unstable (median RMSE 1898 $/bbl, occasional float
overflow) and longend only valid on 22% of days, so neither is a fair benchmark
opponent. Both discarded configs are still charged in the multiple-testing count
(Phase 0: 24 configs tried total).

Note: cubic-spline and Nelson-Siegel are near-interpolants through the very points
they are scored on here, so their in-sample RMSE is close to zero by construction --
this is expected and not informative on its own. The decisive comparison is the
held-out-maturity test in Phase 6.
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
VARIANT = "all"
L_ESTIMATORS = ["kalman", "joint_ls_monthly"]


def rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def run(l_name: str) -> dict[str, Any]:
    panel = pd.read_parquet(TMP / f"phase_1_23_panel_{VARIANT}.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[
        (panel["date"] >= lib23.TRAIN_START) & (panel["date"] <= lib23.TRAIN_END)
    ]
    fit26 = pd.read_parquet(TMP / f"phase_3_23_fit26_{VARIANT}_{l_name}.parquet")
    fit26["date"] = pd.to_datetime(fit26["date"])
    fit26 = fit26.set_index("date")

    grid = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
    pca_train_grid = lib23.build_pca_train_grid(panel, grid)

    dates = sorted(set(panel["date"].unique()) & set(fit26.index))
    prev_day = None
    results: dict[str, list[float]] = {
        name: [] for name in ["gabillon26", "flat", "prevcurve", "spline", "ns", "pca"]
    }

    for i, date_key in enumerate(dates):
        date = cast(pd.Timestamp, date_key)
        g = panel[panel["date"] == date].sort_values("tau")
        tau = g["tau"].to_numpy()
        actual = g["close"].to_numpy()
        if date not in fit26.index:
            prev_day = g
            continue
        s = float(cast(float, fit26.loc[date, "S"]))
        ell = float(cast(float, fit26.loc[date, "L"]))
        beta = float(cast(float, fit26.loc[date, "beta"]))
        nu = float(cast(float, fit26.loc[date, "nu"]))
        pred_g = np.exp(lib23.model26_lnF(s, ell, tau, beta, nu))
        results["gabillon26"].append(rmse(pred_g, actual))

        pred_flat = lib23.bench_flat_forward(g, s, tau)
        results["flat"].append(rmse(pred_flat, actual))

        if prev_day is not None:
            pred_prev = lib23.bench_previous_curve(prev_day, tau)
            if pred_prev is not None:
                results["prevcurve"].append(rmse(pred_prev, actual))

        pred_spline = lib23.bench_cubic_spline(g, tau)
        if pred_spline is not None:
            results["spline"].append(rmse(pred_spline, actual))

        pred_ns = lib23.bench_nelson_siegel(g, tau)
        if pred_ns is not None:
            results["ns"].append(rmse(pred_ns, actual))

        if i % 5 == 0:  # subsample PCA for compute budget; SVD is the costly step
            pred_pca = lib23.bench_pca_2factor(pca_train_grid, g, tau)
            if pred_pca is not None:
                results["pca"].append(rmse(pred_pca, actual))

        prev_day = g

    summary: dict[str, Any] = {}
    for name, vals in results.items():
        arr = np.array(vals)
        arr = arr[np.isfinite(arr)]
        summary[name] = {
            "n": len(arr),
            "median_rmse": float(np.median(arr)) if len(arr) else None,
            "mean_rmse": float(np.mean(arr)) if len(arr) else None,
        }
    return summary


report: dict[str, Any] = {}
t0 = time.time()
for l_name in L_ESTIMATORS:
    report[l_name] = run(l_name)
    print(l_name, json.dumps(report[l_name], indent=2), f"t={time.time() - t0:.1f}s")

out = TMP / "phase_4_23_benchmarks_report.json"
out.write_text(json.dumps(report, indent=2, default=str))
print(f"wrote {out}")
