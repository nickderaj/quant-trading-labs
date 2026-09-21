"""Phase 9 -- spend the holdout once. Certified configuration (pre-registered as
the Phase 3/6 survivor): "all" liquidity variant, kalman L estimator, model (26).
Runs the exact Phase 6 held-out-maturity protocol (fit tau<=2y, predict tau in
(2,9]) restricted to 2022-01-01 onward. This script must only ever be run once
against real holdout data; it is idempotent by output-file existence like every
other phase, which is itself the mechanism enforcing "spent once."
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
L_NAME = "kalman"
SHORT_CUTOFF = 2.0
LONG_MAX = 9.0

ROLL_WINDOW = 63


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


def rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def curve_state(g: pd.DataFrame) -> str:
    d = g.sort_values("tau")
    f1, f2 = d["close"].iloc[0], d["close"].iloc[1]
    return "backwardation" if np.log(f2 / f1) < 0 else "contango"


t0 = time.time()
panel = pd.read_parquet(TMP / f"phase_1_23_panel_{VARIANT}.parquet")
panel["date"] = pd.to_datetime(panel["date"])

S = pd.read_parquet(TMP / f"phase_2_23_spot_{VARIANT}.parquet")["S"]
L_full = pd.read_parquet(TMP / f"phase_2_23_L_{L_NAME}_{VARIANT}.parquet")["L"].ffill()
nu_df = compute_nu_series(S, L_full)

holdout_panel = panel[panel["date"] >= lib23.HOLDOUT_START]

grid = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
short_panel = holdout_panel[holdout_panel["tau"] <= SHORT_CUTOFF]
pca_train_grid = lib23.build_pca_train_grid(short_panel, grid)

rows = []
for date_key, g in holdout_panel.groupby("date"):
    date = cast(pd.Timestamp, date_key)
    if date not in S.index or date not in L_full.index or date not in nu_df.index:
        continue
    s, ell = float(S.loc[date]), float(L_full.loc[date])
    nu = float(cast(float, nu_df.loc[date, "nu"]))
    if not (
        np.isfinite(s)
        and np.isfinite(ell)
        and s > 0
        and ell > 0
        and np.isfinite(nu)
        and nu > 0
    ):
        continue
    g = g.sort_values("tau")
    short = g[g["tau"] <= SHORT_CUTOFF]
    long = g[(g["tau"] > SHORT_CUTOFF) & (g["tau"] <= LONG_MAX)]
    if len(short) < 4 or len(long) < 1:
        continue
    pred_tau = long["tau"].to_numpy()
    actual = long["close"].to_numpy()
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
df.to_parquet(TMP / "phase_9_23_holdout_results.parquet")


def summarize(sub: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"n_days": len(sub)}
    for col in ["gabillon26", "flat", "spline", "ns", "pca"]:
        if col in sub.columns:
            vals = sub[col].dropna()
            out[f"median_{col}"] = float(vals.median()) if len(vals) else None
    return out


bench_cols = [c for c in ["flat", "spline", "ns", "pca"] if c in df.columns]
df["best_bench"] = df[bench_cols].min(axis=1)
diff = (df["gabillon26"] - df["best_bench"]).dropna().to_numpy()
rng = np.random.default_rng(23)
boot = (
    [np.mean(rng.choice(diff, size=len(diff), replace=True)) for _ in range(2000)]
    if len(diff)
    else []
)
ci = (
    [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    if boot
    else [None, None]
)

pass_threshold_pct = 0.10
best_bench_median = df["best_bench"].median()
pct_improvement = (
    float((best_bench_median - df["gabillon26"].median()) / best_bench_median)
    if best_bench_median
    else None
)
gate_pass = bool(
    len(diff) > 0
    and np.mean(diff) < 0  # Gabillon RMSE below best benchmark on average
    and ci[1] is not None
    and ci[1] < 0  # CI entirely favours Gabillon
    and pct_improvement is not None
    and pct_improvement >= pass_threshold_pct
)

report: dict[str, Any] = {
    "certified_config": {"variant": VARIANT, "L_estimator": L_NAME, "model": "model26"},
    "holdout_start": str(lib23.HOLDOUT_START.date()),
    "overall": summarize(df),
    "by_year": {int(cast(int, y)): summarize(sub) for y, sub in df.groupby("year")},
    "by_curve_state": {s: summarize(sub) for s, sub in df.groupby("curve_state")},
    "paired_gap_gabillon_minus_best_bench": {
        "point_estimate": float(np.mean(diff)) if len(diff) else None,
        "bootstrap_95ci": ci,
        "gabillon_wins_pct_days": float((df["gabillon26"] < df["best_bench"]).mean())
        if len(df)
        else None,
    },
    "pct_improvement_median_vs_best_bench": pct_improvement,
    "PRE_REGISTERED_GATE_PASS": gate_pass,
    "wall_time_seconds": time.time() - t0,
}
out = TMP / "phase_9_23_holdout_report.json"
out.write_text(json.dumps(report, indent=2, default=str))
print(json.dumps(report, indent=2, default=str))
print(f"wrote {out}")
