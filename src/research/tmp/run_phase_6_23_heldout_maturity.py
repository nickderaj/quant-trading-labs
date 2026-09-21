"""Phase 6 -- THE PRIMARY GATE. Fit every model/benchmark using only tau<=2y
contracts, predict tau in (2y, 9y], score RMSE against actual settlement. Train
sample only (holdout dates untouched -- spent once in Phase 9). Broken out by
year and by curve state (backwardation/contango via sign of ln(F2/F1)).

Also runs leave-one-contract-out across the full strip as a secondary check.
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
SHORT_CUTOFF = 2.0
LONG_MAX = 9.0


def rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def curve_state(g: pd.DataFrame) -> str:
    d = g.sort_values("tau")
    if len(d) < 2:
        return "unknown"
    f1, f2 = d["close"].iloc[0], d["close"].iloc[1]
    return "backwardation" if np.log(f2 / f1) < 0 else "contango"


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
    short_panel = panel[panel["tau"] <= SHORT_CUTOFF]
    pca_train_grid = lib23.build_pca_train_grid(short_panel, grid)

    rows = []
    for date_key, g in panel.groupby("date"):
        date = cast(pd.Timestamp, date_key)
        g = g.sort_values("tau")
        short = g[g["tau"] <= SHORT_CUTOFF]
        long = g[(g["tau"] > SHORT_CUTOFF) & (g["tau"] <= LONG_MAX)]
        if len(short) < 4 or len(long) < 1 or date not in fit26.index:
            continue
        pred_tau = long["tau"].to_numpy()
        actual = long["close"].to_numpy()
        s = float(cast(float, fit26.loc[date, "S"]))
        ell = float(cast(float, fit26.loc[date, "L"]))
        nu = float(cast(float, fit26.loc[date, "nu"]))
        beta, _ = lib23.fit_beta_given_SL(short, s, ell)
        pred_g = np.exp(lib23.model26_lnF(s, ell, pred_tau, beta, nu))

        pred_flat = lib23.bench_flat_forward(short, s, pred_tau)
        pred_spline = lib23.bench_cubic_spline(short, pred_tau)
        pred_ns = lib23.bench_nelson_siegel(short, pred_tau)
        pred_pca = lib23.bench_pca_2factor(pca_train_grid, short, pred_tau)

        rec: dict[str, Any] = {
            "date": str(date.date()),
            "year": int(date.year),
            "curve_state": curve_state(g),
            "n_long": len(long),
            "gabillon26": rmse(pred_g, actual),
            "flat": rmse(pred_flat, actual),
        }
        if pred_spline is not None:
            rec["spline"] = rmse(pred_spline, actual)
        if pred_ns is not None:
            rec["ns"] = rmse(pred_ns, actual)
        if pred_pca is not None:
            rec["pca"] = rmse(pred_pca, actual)
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_parquet(TMP / f"phase_6_23_heldout_{l_name}.parquet")

    def summarize(sub: pd.DataFrame) -> dict[str, Any]:
        out: dict[str, Any] = {"n_days": len(sub)}
        for col in ["gabillon26", "flat", "spline", "ns", "pca"]:
            if col in sub.columns:
                vals = sub[col].dropna()
                out[f"median_{col}"] = float(vals.median()) if len(vals) else None
        return out

    result: dict[str, Any] = {"overall": summarize(df)}
    result["by_year"] = {
        int(cast(int, y)): summarize(sub) for y, sub in df.groupby("year")
    }
    result["by_curve_state"] = {
        s: summarize(sub) for s, sub in df.groupby("curve_state")
    }

    # paired bootstrap CI on (gabillon26 - best_benchmark) per day
    bench_cols = [c for c in ["flat", "spline", "ns", "pca"] if c in df.columns]
    df["best_bench"] = df[bench_cols].min(axis=1)
    diff = (df["gabillon26"] - df["best_bench"]).dropna().to_numpy()
    rng = np.random.default_rng(23)
    boot = [
        np.mean(rng.choice(diff, size=len(diff), replace=True)) for _ in range(2000)
    ]
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    result["paired_gap_gabillon_minus_best_bench"] = {
        "point_estimate": float(np.mean(diff)),
        "bootstrap_95ci": ci,
        "gabillon_wins_pct_days": float((df["gabillon26"] < df["best_bench"]).mean()),
    }
    return result


report: dict[str, Any] = {}
t0 = time.time()
for l_name in L_ESTIMATORS:
    report[l_name] = run(l_name)
    print(l_name, "overall:", report[l_name]["overall"], f"t={time.time() - t0:.1f}s")
    print(l_name, "paired gap:", report[l_name]["paired_gap_gabillon_minus_best_bench"])

# leave-one-contract-out across the full strip (kalman config only, for compute)
panel = pd.read_parquet(TMP / f"phase_1_23_panel_{VARIANT}.parquet")
panel["date"] = pd.to_datetime(panel["date"])
panel = panel[(panel["date"] >= lib23.TRAIN_START) & (panel["date"] <= lib23.TRAIN_END)]
fit26 = pd.read_parquet(TMP / f"phase_3_23_fit26_{VARIANT}_kalman.parquet")
fit26["date"] = pd.to_datetime(fit26["date"])
fit26 = fit26.set_index("date")
rng = np.random.default_rng(23)
sample_dates = pd.Index(
    rng.choice(
        sorted(set(panel["date"].unique()) & set(fit26.index)), size=250, replace=False
    )
).sort_values()
looc_err_list: list[float] = []
for date in sample_dates:
    g = panel[panel["date"] == date].sort_values("tau").reset_index(drop=True)
    if len(g) < 9 or date not in fit26.index:
        continue
    s = float(cast(float, fit26.loc[date, "S"]))
    ell = float(cast(float, fit26.loc[date, "L"]))
    nu = float(cast(float, fit26.loc[date, "nu"]))
    for i in range(len(g)):
        rest = g.drop(index=i)
        beta, _ = lib23.fit_beta_given_SL(rest, s, ell)
        pred = np.exp(
            lib23.model26_lnF(s, ell, np.array([g["tau"].iloc[i]]), beta, nu)
        )[0]
        looc_err_list.append(float(pred - g["close"].iloc[i]))
looc_errs = np.array(looc_err_list)
looc_result = {
    "n_obs": len(looc_errs),
    "n_days_sampled": len(sample_dates),
    "rmse": float(np.sqrt(np.mean(looc_errs**2))),
    "median_abs_error": float(np.median(np.abs(looc_errs))),
}
print("leave-one-contract-out (kalman):", looc_result)
report["leave_one_contract_out_kalman"] = looc_result

report["wall_time_seconds"] = time.time() - t0
out = TMP / "phase_6_23_heldout_report.json"
out.write_text(json.dumps(report, indent=2, default=str))
print(f"wrote {out}")
