"""Notebook 9 (external research review) - the one piece of genuinely new
computation this notebook needed: a lightweight AR(1)-in-differences
mean-reversion test and a rolling z-score information coefficient, used only
by Phase 4's spread-mean-reversion probe (`run_phase_4_spread_probe.py`).

Deliberately NOT a formal Dickey-Fuller test (no tabulated critical values -
statsmodels is not a dependency of this repo) - a first-look descriptive
screen only, matching Phase 4's own stated scope (sec 4 Phase 4: "not a gated
backtest... a first look").
"""

import numpy as np
import polars as pl
from scipy import stats as st


def ols_ar1_diff(v: np.ndarray) -> dict:
    """Fit delta_v_t = alpha + beta * v_{t-1} + eps by OLS.

    beta < 0 with a large |t-stat| is evidence of mean reversion (a positive
    level predicts a subsequent decline, and vice versa). Half-life is the
    number of periods for a unit deviation to decay by half under the
    implied AR(1) recursion v_t = (1+beta) * v_{t-1} + ..., defined only when
    -1 < beta < 0 (a stationary, mean-reverting process).
    """
    if len(v) < 3:
        raise ValueError("need at least 3 observations")
    v_lag = v[:-1]
    dv = np.diff(v)
    X = np.column_stack([np.ones_like(v_lag), v_lag])
    coef, *_ = np.linalg.lstsq(X, dv, rcond=None)
    alpha, beta = coef
    resid = dv - X @ coef
    n, k = X.shape
    dof = n - k
    sigma2 = (resid @ resid) / dof if dof > 0 else np.nan
    xtx_inv = np.linalg.inv(X.T @ X)
    se_beta = np.sqrt(sigma2 * xtx_inv[1, 1]) if np.isfinite(sigma2) else np.nan
    t_beta = beta / se_beta if se_beta and se_beta > 0 else np.nan
    half_life = (-np.log(2) / np.log(1 + beta)) if -1 < beta < 0 else None
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "t_stat_beta": float(t_beta),
        "n_obs": int(n),
        "half_life_days": half_life,
        "mean_reverting": bool(beta < 0 and np.isfinite(t_beta) and abs(t_beta) > 2.0),
    }


def zscore_ic(v: np.ndarray, window: int = 60, horizon: int = 5) -> dict:
    """Spearman IC between a rolling z-score and the horizon-ahead change in
    level. A negative, significant IC is the mean-reversion sign (a currently
    high z-score predicts a subsequent decline).
    """
    s = pl.Series("v", v)
    roll_mean = s.rolling_mean(window_size=window)
    roll_std = s.rolling_std(window_size=window)
    z = ((s - roll_mean) / roll_std).to_numpy()
    if horizon > 0:
        fwd_change = np.concatenate([v[horizon:] - v[:-horizon], [np.nan] * horizon])
    else:
        fwd_change = np.zeros_like(v)
    mask = np.isfinite(z) & np.isfinite(fwd_change)
    if mask.sum() < 30:
        return {"ic": None, "p_value": None, "n": int(mask.sum())}
    rho, p = st.spearmanr(z[mask], fwd_change[mask])
    return {"ic": float(rho), "p_value": float(p), "n": int(mask.sum())}
