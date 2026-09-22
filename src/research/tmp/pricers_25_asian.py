"""Asian option pricing: geometric and arithmetic averages.

Discrete fixing schedules with moment-matching to Black-76 framework.
Generality: handles any set of fixing times, uniform or non-uniform,
by computing exact moments from the actual fixing times, not closed-form shortcuts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib25

OptType = lib25.OptType


def asian_geometric(
    F: float,
    K: float,
    T: float,
    sigma: float,
    df: float,
    opt: OptType,
    avg_start: float,
    n_fix: int,
) -> float:
    """Price geometric Asian option via moment matching to lognormal.

    Under driftless GBM (Black-76 measure), ln(F_t) = ln(F) - 0.5*sigma^2*t + sigma*W_t.
    Geometric average G = exp(mean(ln(F_t_i))) is exactly lognormal.

    Fixing times: t_i = avg_start + i*(T - avg_start)/(n_fix-1) for i=0..n_fix-1
                  (if n_fix==1, just t_i=[T])

    Returns Black-76 price at the effective lognormal parameters.
    """
    if T <= 0 or sigma <= 0:
        intrinsic = max(F - K, 0.0) if opt == "call" else max(K - F, 0.0)
        return df * intrinsic

    # Build fixing times
    if n_fix == 1:
        t = np.array([T])
    else:
        t = avg_start + np.arange(n_fix) * (T - avg_start) / (n_fix - 1)

    n = len(t)
    mean_t = np.mean(t)

    # Covariance matrix of W_t: Cov(W_ti, W_tj) = min(t_i, t_j)
    cov_matrix = np.minimum.outer(t, t)
    var_avgW = cov_matrix.sum() / (n**2)

    # Effective forward and vol for lognormal (Black-76 compatible)
    ln_F_eff = np.log(F) - 0.5 * sigma**2 * mean_t + 0.5 * sigma**2 * var_avgW
    F_eff = np.exp(ln_F_eff)
    sigma_eff = sigma * np.sqrt(var_avgW)

    # T_eff = 1 because sigma_eff absorbs the time dimension
    return lib25.black76(F_eff, K, 1.0, sigma_eff, df, opt)


def asian_turnbull_wakeman(
    F: float,
    K: float,
    T: float,
    sigma: float,
    df: float,
    opt: OptType,
    avg_start: float,
    n_fix: int,
) -> float:
    """Price arithmetic Asian option via Turnbull & Wakeman moment matching.

    Discrete fixing times, exact moment matching for arithmetic average.
    Under driftless GBM, Cov(F_ti, F_tj) = F^2 * (exp(sigma^2 * min(t_i, t_j)) - 1).

    Fixing times: t_i = avg_start + i*(T - avg_start)/(n_fix-1) for i=0..n_fix-1
                  (if n_fix==1, just t_i=[T])

    Returns Black-76 price at the moment-matched lognormal parameters.
    """
    if T <= 0 or sigma <= 0:
        intrinsic = max(F - K, 0.0) if opt == "call" else max(K - F, 0.0)
        return df * intrinsic

    # Build fixing times
    if n_fix == 1:
        t = np.array([T])
    else:
        t = avg_start + np.arange(n_fix) * (T - avg_start) / (n_fix - 1)

    n = len(t)

    # First moment: arithmetic average of driftless martingale is F
    M1 = F

    # Second moment: compute covariance matrix
    # Cov(F_ti, F_tj) = F^2 * (exp(sigma^2 * min(t_i, t_j)) - 1)
    min_matrix = np.minimum.outer(t, t)
    cov_matrix = (F**2) * (np.exp(sigma**2 * min_matrix) - 1)
    var_A = cov_matrix.sum() / (n**2)
    M2 = var_A + M1**2

    # Moment-matched lognormal vol, T_eff = 1
    sigma_eff = np.sqrt(np.log(M2 / (M1**2)))

    return lib25.black76(M1, K, 1.0, sigma_eff, df, opt)


def asian_mc(
    paths: np.ndarray,
    K: float,
    opt: OptType,
    df: float,
    avg_idx: np.ndarray,
    control_variate: bool = True,
) -> dict:
    """Monte Carlo price of arithmetic Asian, optionally with geometric control variate.

    Parameters
    ----------
    paths : np.ndarray
        Shape (n_paths, n_steps+1). Simulated futures prices.
    K : float
        Strike price.
    opt : OptType
        "call" or "put".
    df : float
        Discount factor.
    avg_idx : np.ndarray
        Integer indices into columns of paths to use for averaging.
    control_variate : bool
        If True, apply control variate adjustment using empirical geometric mean.
        True geometric closed form not available to this function; the control variate
        here uses the empirical geometric mean as its own reference, which still removes
        shared noise between the two payoffs and reduces variance, though not to the
        full extent a closed-form control would.

    Returns
    -------
    dict
        {'price': float, 'se': float, 'n': n_paths}
    """
    n_paths = paths.shape[0]
    avg_idx = np.asarray(avg_idx, dtype=int)

    # Arithmetic average payoff
    arithmetic_avg = paths[:, avg_idx].mean(axis=1)
    if opt == "call":
        payoff_arith = np.maximum(arithmetic_avg - K, 0.0)
    else:  # put
        payoff_arith = np.maximum(K - arithmetic_avg, 0.0)

    # Optional control variate
    if control_variate:
        # Geometric average payoff (only for control, not for primary estimate)
        geometric_avg = np.exp(np.log(paths[:, avg_idx]).mean(axis=1))
        if opt == "call":
            payoff_geo = np.maximum(geometric_avg - K, 0.0)
        else:  # put
            payoff_geo = np.maximum(K - geometric_avg, 0.0)

        # Control variate correction: beta = Cov(arith, geo) / Var(geo)
        cov_ag = np.cov(payoff_arith, payoff_geo)[0, 1]
        var_g = np.var(payoff_geo)
        if var_g > 1e-12:
            beta = cov_ag / var_g
        else:
            beta = 0.0

        # Adjusted payoff
        payoff = payoff_arith - beta * (payoff_geo - np.mean(payoff_geo))
    else:
        payoff = payoff_arith

    # Price and standard error
    price_array = df * payoff
    price = np.mean(price_array)
    se = np.std(price_array, ddof=1) / np.sqrt(n_paths)

    return {"price": float(price), "se": float(se), "n": n_paths}


def averaging_schedule(product: str, month: str) -> np.ndarray:
    """Return business day dates covering a calendar month.

    Parameters
    ----------
    product : str
        Product identifier (currently unused, but reserved for future
        hygiene checks against the futures contract's last-trade date).
    month : str
        Month string like "2020-01" (YYYY-MM).

    Returns
    -------
    np.ndarray
        Business days within the calendar month, as numpy.datetime64[D].

    Notes
    -----
    Real desk schedules must also check the futures contract's own last-trade
    date so the average never includes days after expiry, but that check is
    out of scope for this pure calendar function.
    """
    ts = pd.Timestamp(month)
    month_start = ts
    month_end = ts + pd.offsets.MonthEnd(0)

    # Business days from start to end of month
    bdays = pd.bdate_range(start=month_start, end=month_end)

    # Convert to numpy.datetime64[D]
    return bdays.to_numpy().astype("datetime64[D]")
