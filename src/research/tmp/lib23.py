"""Shared logic for notebook 023 -- the Gabillon (1991) two-factor curve model.

Implements: curve-panel construction with data hygiene, the spot proxy S, the three
L estimators (joint-LS, long-end extrapolation, Kalman filter), model (26) and model
(28) pricing, the volatility identity (23), and the Phase 4 benchmark curves.

See NEXT_PROMPT.md for the full specification and src/research/reference/README.md
for the paper citation and page map.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DATA_DIR = "src/research/data/market/databento"

TRAIN_START = pd.Timestamp("2010-06-06")
TRAIN_END = pd.Timestamp("2021-12-31")
HOLDOUT_START = pd.Timestamp("2022-01-01")

# The single real structural-limitation event: CL202005 settled at -2.67 on
# 2020-04-20 (negative WTI). Excluded from every fit (ln undefined) but never
# silently dropped from the record -- see the write-up.
NEGATIVE_PRICE_EVENT = {
    "ticker": "CL202005",
    "date": pd.Timestamp("2020-04-20"),
    "close": -2.67,
}

MIN_CONTRACTS_PER_DAY = 8


# --------------------------------------------------------------------------- #
# Phase 1 -- curve panel construction
# --------------------------------------------------------------------------- #


def load_raw_panel(product: str = "CL") -> pd.DataFrame:
    """Join ohlcv to contracts and compute tau in years. No hygiene applied yet."""
    ohlcv = pd.read_parquet(f"{DATA_DIR}/ohlcv/{product}.parquet")
    contracts = pd.read_parquet(f"{DATA_DIR}/contracts.parquet")
    contracts = contracts[contracts["product"] == product][
        ["contract_id", "ticker", "expiry"]
    ]
    df = ohlcv.merge(contracts, on="contract_id", suffixes=("", "_c"))
    df["ticker"] = df["ticker"].where(df["ticker"].notna(), df["ticker_c"])
    df = df.drop(columns=["ticker_c"])
    df["date"] = pd.to_datetime(df["date"])
    df["expiry"] = pd.to_datetime(df["expiry"])
    df["tau"] = (df["expiry"] - df["date"]).dt.days / 365.25
    return df


def apply_hygiene(
    df: pd.DataFrame, liquidity_variant: str
) -> tuple[pd.DataFrame, dict]:
    """Apply data hygiene rules and one liquidity variant. Returns (clean_df, report).

    liquidity_variant: one of "all", "vol_gt_0", "vol_gt_100".
    """
    report: dict = {"input_rows": len(df), "dropped": {}}

    # Rule 1a: the real negative-price event -- exclude from fit, record explicitly.
    is_event = (df["ticker"] == NEGATIVE_PRICE_EVENT["ticker"]) & (
        df["date"] == NEGATIVE_PRICE_EVENT["date"]
    )
    report["dropped"]["negative_price_event_real"] = int(is_event.sum())
    df = df[~is_event]

    # Rule 1b: any remaining non-positive close is junk (illiquid far-dated stub
    # contracts printing erratic sign-flipped settlements). Filter unconditionally
    # since ln() is undefined -- these cannot enter any fit under any liquidity
    # variant. This is a documented deviation from the prompt's "at least 8 rows,
    # CL203212 only" description: the actual panel has 390 non-positive rows across
    # 7 tickers (CL202707, CL202903, CL202904, CL203012, CL203212, CL203305,
    # CL203311), all far-dated stubs. See write-up.
    is_nonpositive = df["close"] <= 0
    report["dropped"]["nonpositive_junk"] = int(is_nonpositive.sum())
    report["dropped"]["nonpositive_junk_tickers"] = sorted(
        df.loc[is_nonpositive, "ticker"].unique().tolist()
    )
    df = df[~is_nonpositive]

    # Rule 2: liquidity variant.
    if liquidity_variant == "all":
        pass
    elif liquidity_variant == "vol_gt_0":
        n = int((df["volume"] <= 0).sum())
        report["dropped"]["volume_le_0"] = n
        df = df[df["volume"] > 0]
    elif liquidity_variant == "vol_gt_100":
        n = int((df["volume"] <= 100).sum())
        report["dropped"]["volume_le_100"] = n
        df = df[df["volume"] > 100]
    else:
        raise ValueError(liquidity_variant)

    # Non-positive tau (expired but still recorded) or NaN tau.
    bad_tau = (~df["tau"].notna()) | (df["tau"] <= 0)
    report["dropped"]["bad_tau"] = int(bad_tau.sum())
    df = df[~bad_tau]

    df = df.sort_values(["date", "tau"]).reset_index(drop=True)
    df["log_close"] = np.log(df["close"])

    # Rule 3: minimum curve width per day.
    counts = df.groupby("date")["ticker"].nunique()
    good_days = counts[counts >= MIN_CONTRACTS_PER_DAY].index
    skipped_days = counts[counts < MIN_CONTRACTS_PER_DAY]
    report["skipped_days_thin_curve"] = len(skipped_days)
    report["skipped_days_thin_curve_daylist_sample"] = [
        str(d.date()) for d in skipped_days.index[:10]
    ]
    df = df[df["date"].isin(good_days)]

    report["output_rows"] = len(df)
    report["days_retained"] = int(df["date"].nunique())
    report["mean_contracts_per_day"] = float(
        df.groupby("date")["ticker"].nunique().mean()
    )
    df["year"] = df["date"].dt.year
    report["max_tau_by_year"] = df.groupby("year")["tau"].max().round(3).to_dict()
    report["max_tau_by_year"] = {
        int(k): float(v) for k, v in report["max_tau_by_year"].items()
    }
    return df[["date", "ticker", "tau", "close", "volume", "log_close"]].reset_index(
        drop=True
    ), report


# --------------------------------------------------------------------------- #
# Spot proxy S
# --------------------------------------------------------------------------- #


def spot_proxy_for_day(day_df: pd.DataFrame) -> float | None:
    """S = F1 - tau1*(F2-F1)/(tau2-tau1) from the two nearest-maturity contracts."""
    d = day_df.sort_values("tau")
    if len(d) < 2:
        return None
    f1, tau1 = d["close"].iloc[0], d["tau"].iloc[0]
    f2, tau2 = d["close"].iloc[1], d["tau"].iloc[1]
    if tau2 == tau1:
        return None
    s = f1 - tau1 * (f2 - f1) / (tau2 - tau1)
    return float(s) if s > 0 else None


def compute_spot_series(panel: pd.DataFrame) -> pd.Series:
    """Per-day spot proxy S, indexed by date."""
    out = {}
    for date, g in panel.groupby("date"):
        s = spot_proxy_for_day(g)
        if s is not None:
            out[date] = s
    return pd.Series(out).sort_index()


# --------------------------------------------------------------------------- #
# L estimators
# --------------------------------------------------------------------------- #


def fit_L_joint_ls(day_df: pd.DataFrame) -> tuple[float, float, float] | None:
    """Method (a): joint least squares on ln F = beta-implied blend of ln S, ln L.

    We fit beta and L by nonlinear least squares on a single day's log-curve using
    the model-26 functional form with nu=0 during the *shape* fit (nu only affects
    the convexity term A(tau), not the beta/L blend weight), i.e. we regress
        ln F(tau) - ln S = (1 - B(tau)) * (ln L - ln S)
    over a 1-D search on beta (B(tau)=exp(-beta*tau)), solving ln L in closed form
    per beta by OLS, then grid+refine on beta to minimise SSE. Returns
    (beta, L, sse) or None if the day cannot be fit (not enough distinct tau).
    """
    d = day_df.sort_values("tau")
    tau = d["tau"].to_numpy()
    lnF = d["log_close"].to_numpy()
    if len(d) < 3 or (tau.max() - tau.min()) < 0.1:
        return None
    s = spot_proxy_for_day(d)
    if s is None:
        return None
    lnS = np.log(s)

    # L is economically a long-term anchor; bound ln(L/S) to a generous +-log(5)
    # band. Without this, small-beta days (where the (1-B) weight vanishes for
    # every observed tau and L becomes numerically unidentified) produce a
    # closed-form L that explodes to absurd values (observed: L up to 1e62)
    # purely from noise amplification, not signal. Documented, pragmatic guard.
    coef_bound = np.log(5.0)

    def sse_for_beta(beta: float) -> tuple[float, float]:
        B = np.exp(-beta * tau)
        w = 1 - B
        y = lnF - lnS
        # y = w * (lnL - lnS)  =>  closed-form lnL - lnS via weighted OLS through origin
        denom = np.sum(w * w)
        if denom < 1e-8:
            return np.inf, np.nan
        coef = np.sum(w * y) / denom
        coef = float(np.clip(coef, -coef_bound, coef_bound))
        resid = y - w * coef
        return float(np.sum(resid**2)), float(lnS + coef)

    betas = np.linspace(0.01, 8.0, 400)
    sses = np.array([sse_for_beta(b)[0] for b in betas])
    b0 = betas[np.argmin(sses)]
    lo, hi = max(0.001, b0 - 0.05), b0 + 0.05
    refine = np.linspace(lo, hi, 200)
    sses_r = np.array([sse_for_beta(b)[0] for b in refine])
    best_beta = refine[np.argmin(sses_r)]
    best_sse, best_lnL = sse_for_beta(best_beta)
    if not np.isfinite(best_lnL):
        return None
    return float(best_beta), float(np.exp(best_lnL)), float(best_sse)


def fit_L_joint_ls_panel(panel: pd.DataFrame, monthly_fixed: bool) -> pd.DataFrame:
    """Per-day (or monthly-fixed-L) joint LS fit. Returns DataFrame indexed by date
    with columns beta, L, S, sse.
    """
    rows = []
    for date, g in panel.groupby("date"):
        s = spot_proxy_for_day(g)
        fit = fit_L_joint_ls(g)
        if fit is None or s is None:
            continue
        beta, ell, sse = fit
        rows.append({"date": date, "beta": beta, "L": ell, "S": s, "sse": sse})
    df = pd.DataFrame(rows).set_index("date").sort_index()
    if monthly_fixed and len(df):
        month = pd.DatetimeIndex(df.index).to_period("M")
        df["L"] = df.groupby(month)["L"].transform("mean")
    return df


def fit_L_longend(panel: pd.DataFrame) -> pd.DataFrame:
    """Method (b): long-end extrapolation, paper section 5.2.

    f(tau) = exp(a1*exp(-a2*tau) + a3); match value, f', f'' at the long end.
    f' proxy: slope between the two farthest contracts.
    f'' proxy: change in slope between the 9th-nearest contract and the farthest
    traded contract (assumption, since the paper calls this proxy arbitrary --
    documented in the write-up).
    L = exp(a3), valid only when a2 > 0; else carry forward the most recent valid L.
    """
    rows = []
    last_valid_L = None
    for date, g in panel.groupby("date"):
        d = g.sort_values("tau").reset_index(drop=True)
        if len(d) < 10:
            rows.append({"date": date, "L": last_valid_L, "valid": False})
            continue
        tau = d["tau"].to_numpy()
        lnF = d["log_close"].to_numpy()
        # f' from the two farthest contracts
        t_far2, t_far1 = tau[-2], tau[-1]
        f_far2, f_far1 = lnF[-2], lnF[-1]
        fprime = (f_far1 - f_far2) / (t_far1 - t_far2) if t_far1 != t_far2 else np.nan
        # f'' from the change in slope between the 9th-nearby contract and the
        # farthest traded contract (documented assumption).
        idx9 = min(8, len(d) - 2)
        t9, f9 = tau[idx9], lnF[idx9]
        slope_near = (
            (lnF[idx9 + 1] - f9) / (tau[idx9 + 1] - t9)
            if tau[idx9 + 1] != t9
            else np.nan
        )
        slope_far = fprime
        fpp = (slope_far - slope_near) / (t_far1 - t9) if t_far1 != t9 else np.nan

        if not (np.isfinite(fprime) and np.isfinite(fpp)) or fpp >= 0 or fprime == 0:
            rows.append({"date": date, "L": last_valid_L, "valid": False})
            continue
        # f(t) = a1*exp(-a2 t) + a3 (working in log-price directly, f == lnF)
        # f'(t) = -a1*a2*exp(-a2 t); f''(t) = a1*a2^2*exp(-a2 t) = -a2 * f'(t)
        a2 = -fpp / fprime
        if not np.isfinite(a2) or a2 <= 0 or a2 * t_far1 > 50:
            rows.append({"date": date, "L": last_valid_L, "valid": False})
            continue
        a1_exp = fprime / (-a2)  # a1*exp(-a2*t_far1)
        a1 = a1_exp * np.exp(a2 * t_far1)
        a3 = f_far1 - a1 * np.exp(-a2 * t_far1)
        ell = float(np.exp(a3))
        last_valid_L = ell
        rows.append({"date": date, "L": ell, "valid": True})
    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df


def kalman_S_L(
    panel: pd.DataFrame,
    spot: pd.Series,
    q_S: float = 1e-4,
    q_L: float = 1e-6,
    r_obs: float = 1e-3,
) -> pd.DataFrame:
    """Method (c): a plain linear Kalman filter on latent state x=(ln S, ln L)_t.

    Observation model per contract on a day (with a provisional beta held fixed at
    a rough calibration estimate, since (26) is linear in (lnS, lnL) given beta):
        ln F(tau) = B(tau) * lnS + (1-B(tau)) * lnL + noise
    State transition: random walk in both components.
    Uses a single global beta (pre-calibrated by the joint-LS median) since the
    Kalman step here targets S, L only -- beta itself is re-fit downstream per day
    in Phase 3 using the filtered L as an anchor.
    """
    dates = sorted(panel["date"].unique())
    if len(dates) == 0:
        return pd.DataFrame(columns=["S", "L"])

    # Rough global beta from a cheap day-by-day joint LS median, for the
    # observation matrix only.
    sample_dates = dates[:: max(1, len(dates) // 60)][:60]
    betas = []
    for dt in sample_dates:
        fit = fit_L_joint_ls(panel[panel["date"] == dt])
        if fit is not None:
            betas.append(fit[0])
    beta0 = float(np.median(betas)) if betas else 0.3

    s0 = spot.iloc[0] if len(spot) else 60.0
    x = np.array([np.log(s0), np.log(s0)])
    P = np.eye(2) * 1.0
    Q = np.diag([q_S, q_L])

    rows = []
    for dt in dates:
        g = panel[panel["date"] == dt].sort_values("tau")
        tau = g["tau"].to_numpy()
        y = g["log_close"].to_numpy()
        if len(y) == 0:
            continue
        # predict
        x_pred = x
        P_pred = P + Q
        B = np.exp(-beta0 * tau)
        H = np.column_stack([B, 1 - B])
        R = np.eye(len(y)) * r_obs
        S_cov = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.pinv(S_cov)
        resid = y - H @ x_pred
        x = x_pred + K @ resid
        P = (np.eye(2) - K @ H) @ P_pred
        rows.append({"date": dt, "S": np.exp(x[0]), "L": np.exp(x[1])})
    return pd.DataFrame(rows).set_index("date")


# --------------------------------------------------------------------------- #
# Model (26) / (28) pricing
# --------------------------------------------------------------------------- #


def model26_B(tau: np.ndarray, beta: float) -> np.ndarray:
    return np.exp(-beta * tau)


def model26_lnF(
    S: float, L: float, tau: np.ndarray, beta: float, nu: float
) -> np.ndarray:
    B = model26_B(tau, beta)
    A_log = (nu / (4 * beta)) * (np.exp(-beta * tau) - np.exp(-2 * beta * tau))
    return A_log + B * np.log(S) + (1 - B) * np.log(L)


def model28_lnF(
    S: float,
    L: float,
    tau: np.ndarray,
    beta: float,
    nu: float,
    theta: float,
    eta: float,
    t: float = 0.0,
) -> np.ndarray:
    B = np.exp(-beta * tau)
    A_log = (nu / (4 * beta)) * (np.exp(-beta * tau) - np.exp(-2 * beta * tau))
    if abs(beta - eta) < 1e-6:
        shock = -theta * tau * np.exp(-beta * tau) * np.exp(-eta * t)
    else:
        shock = (
            -(theta / (beta - eta))
            * np.exp(-eta * t)
            * (1 - np.exp(-(beta - eta) * tau))
        )
    return A_log + shock + B * np.log(S) + (1 - B) * np.log(L)


def fit_beta_given_SL(day_df: pd.DataFrame, S: float, L: float) -> tuple[float, float]:
    """Given fixed S, L for the day, fit beta by 1-D search minimising SSE on ln F.
    Returns (beta, sse). nu is fit separately from rolling vol estimates upstream.
    """
    d = day_df.sort_values("tau")
    tau = d["tau"].to_numpy()
    lnF = d["log_close"].to_numpy()
    lnS, lnL = np.log(S), np.log(L)

    def sse(beta: float) -> float:
        B = np.exp(-beta * tau)
        pred = B * lnS + (1 - B) * lnL
        return float(np.sum((lnF - pred) ** 2))

    betas = np.linspace(0.01, 8.0, 400)
    sses = np.array([sse(b) for b in betas])
    b0 = betas[np.argmin(sses)]
    refine = np.linspace(max(0.001, b0 - 0.05), b0 + 0.05, 200)
    sses_r = np.array([sse(b) for b in refine])
    best = refine[np.argmin(sses_r)]
    return float(best), float(sse(best))


def rolling_nu(log_returns: pd.Series, window: int = 63) -> pd.Series:
    """Rolling annualised variance, for nu components."""
    return log_returns.rolling(window).var() * 252


# --------------------------------------------------------------------------- #
# Benchmarks (Phase 4)
# --------------------------------------------------------------------------- #


def bench_flat_forward(
    day_df: pd.DataFrame, s: float, pred_tau: np.ndarray
) -> np.ndarray:
    return np.full(len(pred_tau), s)


def bench_previous_curve(
    prev_day_df: pd.DataFrame, pred_tau: np.ndarray
) -> np.ndarray | None:
    """Yesterday's observed curve, interpolated (carried forward) to today's tau grid."""
    d = prev_day_df.sort_values("tau")
    if len(d) < 2:
        return None
    return np.interp(pred_tau, d["tau"].to_numpy(), d["close"].to_numpy())


def bench_cubic_spline(day_df: pd.DataFrame, pred_tau: np.ndarray) -> np.ndarray | None:
    from scipy.interpolate import CubicSpline

    d = day_df.sort_values("tau").drop_duplicates("tau")
    if len(d) < 4:
        return None
    cs = CubicSpline(d["tau"].to_numpy(), d["log_close"].to_numpy(), extrapolate=True)
    return np.exp(cs(pred_tau))


def bench_nelson_siegel(
    day_df: pd.DataFrame, pred_tau: np.ndarray
) -> np.ndarray | None:

    d = day_df.sort_values("tau")
    tau = d["tau"].to_numpy()
    y = d["log_close"].to_numpy()
    if len(d) < 5:
        return None
    lam = 1.5

    def basis(tau: np.ndarray, lam: float) -> np.ndarray:
        z = tau / lam
        b1 = np.ones_like(tau)
        with np.errstate(divide="ignore", invalid="ignore"):
            b2 = np.where(z > 1e-8, (1 - np.exp(-z)) / z, 1.0)
        b3 = b2 - np.exp(-z)
        return np.column_stack([b1, b2, b3])

    X = basis(tau, lam)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    Xp = basis(pred_tau, lam)
    return np.exp(Xp @ coef)


def bench_pca_2factor(
    train_curves: pd.DataFrame, day_df: pd.DataFrame, pred_tau: np.ndarray
) -> np.ndarray | None:
    """2-factor PCA on the log-curve, fit on a common tau grid from train_curves
    (rows=dates, cols=tau grid), then project today's observed short end onto the
    first 2 PCs by least squares and reconstruct at pred_tau.
    """
    grid = train_curves.columns.to_numpy(dtype=float)
    X = train_curves.to_numpy()
    mean = np.nanmean(X, axis=0)
    Xc = np.nan_to_num(X - mean, nan=0.0)
    try:
        _U, _Sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    pcs = Vt[:2]  # (2, n_grid)

    d = day_df.sort_values("tau")
    tau_obs = d["tau"].to_numpy()
    y_obs = d["log_close"].to_numpy()
    mean_obs = np.interp(tau_obs, grid, mean)
    pcs_obs = np.column_stack(
        [np.interp(tau_obs, grid, pcs[0]), np.interp(tau_obs, grid, pcs[1])]
    )
    y_c = y_obs - mean_obs
    coef, *_ = np.linalg.lstsq(pcs_obs, y_c, rcond=None)

    mean_pred = np.interp(pred_tau, grid, mean)
    pcs_pred = np.column_stack(
        [np.interp(pred_tau, grid, pcs[0]), np.interp(pred_tau, grid, pcs[1])]
    )
    return np.exp(mean_pred + pcs_pred @ coef)


def build_pca_train_grid(panel: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Interpolate each day's observed log-curve onto a common tau grid for PCA."""
    rows = {}
    for date, g in panel.groupby("date"):
        d = g.sort_values("tau").drop_duplicates("tau")
        if len(d) < 4:
            continue
        vals = np.interp(grid, d["tau"].to_numpy(), d["log_close"].to_numpy())
        rows[date] = vals
    return pd.DataFrame.from_dict(rows, orient="index", columns=grid)


# --------------------------------------------------------------------------- #
# Volatility identity (23)
# --------------------------------------------------------------------------- #


def vol_identity_sigma_F(
    tau: np.ndarray, beta: float, sigma_S: float, sigma_L: float, rho: float
) -> np.ndarray:
    B = np.exp(-beta * tau)
    return np.sqrt(
        sigma_S**2 * B**2
        + sigma_L**2 * (1 - B) ** 2
        + 2 * rho * sigma_S * sigma_L * B * (1 - B)
    )


# --------------------------------------------------------------------------- #
# Multiple-testing correction
# --------------------------------------------------------------------------- #


def bonferroni_alpha(n_configs: int, alpha: float = 0.05) -> float:
    return alpha / n_configs
