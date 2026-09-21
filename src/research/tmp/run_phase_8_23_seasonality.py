"""Phase 8 -- seasonality / cross-product. Fit model (26) to NG (strong
seasonality), HO and RB (moderate), BZ (like WTI, the frozen-transfer control),
"all" liquidity variant, kalman L estimator, using the same held-out-maturity
protocol as Phase 6 (fit tau<=2y, predict tau in (2, 9]). If NG degrades sharply
relative to BZ, Gabillon's own stated limitation reproduces.
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
PRODUCTS = ["BZ", "HO", "RB", "NG"]
SHORT_CUTOFF = 2.0
LONG_MAX = 9.0


def rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def run_product(product: str) -> dict[str, Any]:
    raw = lib23.load_raw_panel(product)
    clean, hyg = lib23.apply_hygiene(raw, "all")
    clean = clean[
        (clean["date"] >= lib23.TRAIN_START) & (clean["date"] <= lib23.TRAIN_END)
    ]
    if clean["date"].nunique() < 30:
        return {"product": product, "n_days": 0, "note": "insufficient curve depth"}

    S = lib23.compute_spot_series(clean)
    fit_kalman = lib23.kalman_S_L(clean, S)

    rows = []
    for date_key, g in clean.groupby("date"):
        date = cast(pd.Timestamp, date_key)
        g = g.sort_values("tau")
        short = g[g["tau"] <= SHORT_CUTOFF]
        long = g[(g["tau"] > SHORT_CUTOFF) & (g["tau"] <= LONG_MAX)]
        if len(short) < 4 or len(long) < 1 or date not in fit_kalman.index:
            continue
        s = float(cast(float, fit_kalman.loc[date, "S"]))
        ell = float(cast(float, fit_kalman.loc[date, "L"]))
        if not (np.isfinite(s) and np.isfinite(ell) and s > 0 and ell > 0):
            continue
        beta, _ = lib23.fit_beta_given_SL(short, s, ell)
        nu = 0.05  # fixed nominal nu (small effect on the (1-B)-weighted shape term)
        pred_tau = long["tau"].to_numpy()
        actual = long["close"].to_numpy()
        pred = np.exp(lib23.model26_lnF(s, ell, pred_tau, beta, nu))
        rows.append(
            {"date": str(date.date()), "rmse": rmse(pred, actual), "beta": beta}
        )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return {"product": product, "n_days": 0, "note": "no fittable days"}
    median_rmse = float(df["rmse"].median())
    # Relative RMSE, normalised to each product's own price level -- raw $/bbl
    # (BZ), $/gallon (HO, RB) and $/mmbtu (NG) are not directly comparable.
    median_price_level = float(clean["close"].median())
    return {
        "product": product,
        "n_days": len(df),
        "median_rmse": median_rmse,
        "mean_rmse": float(df["rmse"].mean()),
        "median_beta": float(df["beta"].median()),
        "days_retained_hygiene": hyg["days_retained"],
        "median_price_level": median_price_level,
        "relative_rmse_pct": 100 * median_rmse / median_price_level,
    }


report: dict[str, Any] = {}
t0 = time.time()
for p in PRODUCTS:
    report[p] = run_product(p)
    print(p, report[p], f"t={time.time() - t0:.1f}s")

report["wall_time_seconds"] = time.time() - t0
out = TMP / "phase_8_23_seasonality_report.json"
out.write_text(json.dumps(report, indent=2, default=str))
print(f"wrote {out}")
