"""Notebook-5-local machinery: tail risk and conditional non-normality.

Mirrors dist_lib.py's own convention (`import dist_lib as L`, not a fork of it):
this module imports from dist_lib.py and reuses its causal, rolling-refit
building blocks (build_asset_frame, rolling_garch_forecast, refit-cadence
constants) rather than re-implementing them. Everything here is either new
machinery notebook 4 never needed (Hill estimator, GJR-GARCH, GPD/POT) or a
thin notebook-5-specific composition of dist_lib's existing pieces.

Run as a script from the repo root (sys.path.insert(0, "src")), and imported
from the notebook the same way dist_lib.py is (sys.path.insert(0, "tmp") from
src/research/).
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

import numpy as np
import polars as pl
from scipy import stats as st
from scipy.optimize import minimize

sys.path.insert(0, "src/research/tmp")
import dist_lib as L

import distributions as dist

# NEXT_PROMPT.md sec 3.3: acerbi_szekely_z/acerbi_szekely_bootstrap_pvalue have
# been promoted to src/risk/calibration.py as durable, tested, production code
# (008 Gate CE: 15/16 development, 11/16 holdout; the strongest result in the
# programme). Re-imported and re-exported here so this notebook and
# tests/test_dist_lib5.py keep working unchanged — they are no longer
# *defined* here. See docs/10-risk-engine.md.
from risk.calibration import (  # noqa: F401
    acerbi_szekely_bootstrap_pvalue,
    acerbi_szekely_z,
)

# --------------------------------------------------------------------------
# Phase 1: Hill estimator (tail index), independent of any parametric MLE fit
# --------------------------------------------------------------------------


def hill_estimator(x: np.ndarray, k: int, tail: str = "upper") -> float:
    """Hill estimate of the tail index alpha for the top-k order statistics.

    alpha = 1 / xi, where xi is the extreme-value (Pareto) shape. alpha < 2
    means infinite variance; alpha < 1 means infinite mean. Crypto log returns
    are expected somewhere in 2-4; the whole point of computing it is that
    notebook 4's fitted t-df of ~2 sits exactly on the finite-variance boundary
    and was produced by an optimizer we already know can pin at a bound.
    """
    x = x[np.isfinite(x)]
    y = np.sort(np.abs(x[x > 0]) if tail == "upper" else np.abs(x[x < 0]))
    n = len(y)
    if k >= n or k < 10:
        return np.nan
    top = y[n - k :]
    thresh = y[n - k - 1]
    xi = float(np.mean(np.log(top / thresh)))
    return 1.0 / xi if xi > 0 else np.nan


def hill_alpha_path(
    x: np.ndarray, tail: str = "upper", k_min: int = 20, k_max: int | None = None
) -> dict:
    """Vectorized Hill alpha-hat for every k in [k_min, k_max] at once (a
    fast, O(n) equivalent of calling hill_estimator(x, k, tail) once per k -
    that function alone would cost O(n log n) per call, i.e. O(n^2 log n)
    across a full k-grid, far too slow at 35k observations).

    Same underlying formula as hill_estimator: sort the nonzero one-sided
    values ascending (y), let L = log(y). For the top k values,
    xi(k) = mean(L[n-k:n]) - L[n-k-1]. mean(L[n-k:n]) is a suffix mean,
    computed for every k at once via a reversed cumulative sum.

    Returns {"k": array, "alpha": array} with alpha[i] = NaN wherever the
    same guards as hill_estimator would reject that k (k>=n, k<10, xi<=0).
    Verified in run_phase1_tails.py to agree with hill_estimator at spot-
    checked k values before being trusted for the Hill plot.
    """
    x = x[np.isfinite(x)]
    y = np.sort(np.abs(x[x > 0]) if tail == "upper" else np.abs(x[x < 0]))
    n = len(y)
    if k_max is None:
        k_max = max(k_min, n // 10)
    k_max = min(k_max, n - 1)
    if k_max < k_min:
        return {"k": np.array([], dtype=int), "alpha": np.array([])}

    L = np.log(y, out=np.full_like(y, -np.inf, dtype=float), where=(y > 0))
    ks = np.arange(k_min, k_max + 1)
    # suffix sum of top-k logs, for every k in ks at once: reverse L, cumsum,
    # cumsum[k-1] = sum of the k largest elements' logs.
    cumsum_from_top = np.cumsum(L[::-1])
    top_sum = cumsum_from_top[ks - 1]
    suffix_mean = top_sum / ks
    thresh_log = L[n - ks - 1]
    xi = suffix_mean - thresh_log
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.where((xi > 0) & (ks < n) & (ks >= 10), 1.0 / xi, np.nan)
    return {"k": ks, "alpha": alpha}


def find_hill_plateau(
    alpha: np.ndarray, ks: np.ndarray, window: int = 50, rel_tol: float = 0.10
) -> dict:
    """Read a Hill plot for a stable plateau, honestly.

    Heuristic (documented, not a universal statistical result): slide a
    window of `window` consecutive k-values across the alpha-hat path;
    for each window compute (max-min)/median as a relative-spread score.
    The plateau is the widest contiguous run of windows whose score is
    below rel_tol, read left-to-right; ties broken by lowest average score.
    If no window anywhere satisfies rel_tol, no plateau is reported - callers
    must treat every downstream tail-index claim at that interval/tail as
    provisional (NEXT_RUN_PROMPT.md's own tripwire for this case).
    """
    valid = np.isfinite(alpha)
    if valid.sum() < window:
        return {"found": False, "reason": "insufficient finite alpha estimates"}
    a, k = alpha[valid], ks[valid]
    n = len(a)
    scores = np.full(n - window + 1, np.nan)
    for i in range(n - window + 1):
        seg = a[i : i + window]
        med = np.median(seg)
        if med > 0:
            scores[i] = (seg.max() - seg.min()) / med
    stable = np.where(scores < rel_tol)[0]
    if len(stable) == 0:
        return {
            "found": False,
            "reason": f"no {window}-wide window has relative spread < {rel_tol}",
        }
    # widest contiguous run of stable window-start-indices
    runs, run_start = [], stable[0]
    for i in range(1, len(stable)):
        if stable[i] != stable[i - 1] + 1:
            runs.append((run_start, stable[i - 1]))
            run_start = stable[i]
    runs.append((run_start, stable[-1]))
    best_run = max(runs, key=lambda r: r[1] - r[0])
    k_lo, k_hi = k[best_run[0]], k[best_run[1] + window - 1]
    plateau_mask = (k >= k_lo) & (k <= k_hi)
    return {
        "found": True,
        "k_lo": int(k_lo),
        "k_hi": int(k_hi),
        "alpha_median": float(np.median(a[plateau_mask])),
        "alpha_min": float(np.min(a[plateau_mask])),
        "alpha_max": float(np.max(a[plateau_mask])),
        "k_chosen": int((k_lo + k_hi) // 2),
    }


# --------------------------------------------------------------------------
# Phase 2a: GJR-GARCH(1,1,1) - leverage. Nests dist_lib's plain GARCH(1,1)
# exactly at gamma=0, which is what makes the LR test on gamma=0 meaningful.
# --------------------------------------------------------------------------


def _gjr_variance_path(
    omega: float, alpha: float, gamma: float, beta: float, r: np.ndarray, sig2_0: float
) -> np.ndarray:
    """GJR(1,1,1): sig2_t = omega + (alpha + gamma*1{r_{t-1}<0}) * r_{t-1}^2
    + beta*sig2_{t-1}. gamma > 0 is the leverage effect (down-moves raise
    next-bar variance more than up-moves of the same size). gamma == 0
    collapses exactly to the GARCH(1,1) already in dist_lib, which makes the
    two directly nested and testable by likelihood ratio.
    """
    n = len(r)
    sig2 = np.empty(n)
    sig2[0] = sig2_0
    for t in range(1, n):
        shock = r[t - 1] ** 2
        lev = gamma * shock if r[t - 1] < 0.0 else 0.0
        sig2[t] = omega + alpha * shock + lev + beta * sig2[t - 1]
    return sig2


def _gjr_negloglik(params: np.ndarray, r: np.ndarray, innovation: str) -> float:
    extra: list[float]
    if innovation == "normal":
        omega, alpha, gamma, beta = params
        extra = []
    elif innovation == "t":
        omega, alpha, gamma, beta, nu = params
        if nu <= 2.1:
            return 1e10
        extra = [nu]
    else:
        raise ValueError(
            f"GJR only supports normal/t innovations (skew-t skipped as "
            f"over-parameterized - see NEXT_RUN_PROMPT.md #Phase 2a), got {innovation!r}"
        )

    if omega <= 1e-12 or alpha < 0 or beta < 0 or (alpha + gamma) < 0:
        return 1e10
    # stationarity: alpha + gamma/2 + beta < 1 (the gamma/2 reflects the
    # leverage indicator firing about half the time under a roughly
    # symmetric innovation) - same guard-rail convention as dist_lib's own
    # GARCH(1,1), which rejects candidates at exactly this kind of boundary
    # rather than letting the optimizer wander into a non-stationary region.
    if alpha + gamma / 2.0 + beta >= 0.999:
        return 1e10

    uncond = omega / max(1 - alpha - gamma / 2.0 - beta, 1e-6)
    sig2 = _gjr_variance_path(omega, alpha, gamma, beta, r, uncond)
    if np.any(sig2 <= 0) or not np.all(np.isfinite(sig2)):
        return 1e10
    z = r / np.sqrt(sig2)

    if innovation == "normal":
        ll = -0.5 * np.log(2 * np.pi * sig2) - 0.5 * z**2
    else:
        nu = extra[0]
        c = np.sqrt(nu / (nu - 2))
        zt = z * c
        ll = st.t.logpdf(zt, df=nu) + np.log(c) - 0.5 * np.log(sig2)

    if not np.all(np.isfinite(ll)):
        return 1e10
    return -float(np.sum(ll))


def fit_gjr11(r: np.ndarray, innovation: str = "normal") -> dict | None:
    """MLE GJR-GARCH(1,1,1), normal or t innovations only (see
    _gjr_negloglik's docstring for why skew-t is skipped). Same
    "null rather than propagate junk" convention as dist_lib.fit_garch11:
    returns None on non-convergence or a degenerate window.

    Also runs a likelihood-ratio test of gamma=0 against dist_lib's own
    plain GARCH(1,1) fit on the *same* window - a direct, one-number-per-fit
    answer to "is there a leverage effect here," since GJR nests GARCH
    exactly at gamma=0 (see docs/02-estimation-and-fitting.md#nested-models).
    """
    r = r[np.isfinite(r)]
    if len(r) < 60 or np.std(r) <= 1e-10:
        return None
    var0 = float(np.var(r))
    if innovation == "normal":
        x0 = np.array([0.1 * var0, 0.03, 0.05, 0.85])
        bounds = [(1e-10, None), (0, 1), (-1, 1), (0, 1)]
    elif innovation == "t":
        x0 = np.array([0.1 * var0, 0.03, 0.05, 0.85, 8.0])
        bounds = [(1e-10, None), (0, 1), (-1, 1), (0, 1), (2.2, 60)]
    else:
        return None

    try:
        res = minimize(
            _gjr_negloglik,
            x0,
            args=(r, innovation),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200},
        )
    except Exception:  # noqa: BLE001 - optimizer can raise arbitrary errors; convention is None on any failure
        return None
    if not np.all(np.isfinite(res.x)):
        return None
    if np.allclose(res.x, x0, atol=1e-9):
        return None  # optimizer never moved -> treat as unconverged

    params = res.x
    omega, alpha, gamma, beta = params[0], params[1], params[2], params[3]
    if (alpha + gamma) < 0 or alpha + gamma / 2.0 + beta >= 0.999:
        return None
    uncond = omega / max(1 - alpha - gamma / 2.0 - beta, 1e-6)
    sig2 = _gjr_variance_path(omega, alpha, gamma, beta, r, uncond)
    last_shock = r[-1] ** 2
    lev_last = gamma * last_shock if r[-1] < 0.0 else 0.0
    next_sig2 = omega + alpha * last_shock + lev_last + beta * sig2[-1]
    ll_gjr = -_gjr_negloglik(params, r, innovation)

    lr_stat, lr_pvalue = None, None
    garch_fit = L.fit_garch11(r, innovation=innovation)
    if garch_fit is not None:
        garch_params = np.array(garch_fit["params"])
        ll_garch = -L._garch_negloglik(garch_params, r, innovation)
        lr_stat = max(0.0, -2.0 * (ll_garch - ll_gjr))  # LR>=0 since GJR nests GARCH
        lr_pvalue = float(st.chi2.sf(lr_stat, df=1))

    return {
        "omega": float(omega),
        "alpha": float(alpha),
        "gamma": float(gamma),
        "beta": float(beta),
        "params": params.tolist(),
        "next_var": float(next_sig2),
        "innovation": innovation,
        "lr_gamma0_stat": lr_stat,
        "lr_gamma0_pvalue": lr_pvalue,
    }


def rolling_gjr_forecast(
    returns: np.ndarray,
    refit_every: int,
    min_train: int,
    innovation: str = "normal",
    max_train: int = 1500,
) -> tuple[np.ndarray, list[dict]]:
    """Rolling-refit GJR-GARCH(1,1,1) one-step-ahead variance forecast.

    Structurally identical to dist_lib.rolling_garch_forecast (same
    refit-then-forward-fill loop, same between-refit re-rolling of the
    fitted model's own recursion on realized returns) - generalized here to
    the GJR variance recursion (with its extra leverage term) rather than
    copy-pasting a second, drifting implementation. Watch this loop as
    carefully as dist_lib's own: it is the same subtle, correct logic.
    """
    n = len(returns)
    forecast = np.full(n, np.nan)
    fits = []
    fit = None
    sig2_state = np.nan
    for t in range(n):
        if t >= min_train and t % refit_every == 0:
            start = max(0, t - max_train)
            window = returns[start:t]
            new_fit = fit_gjr11(window, innovation)
            if new_fit is not None:
                fit = new_fit
                sig2_state = fit["omega"] / max(
                    1 - fit["alpha"] - fit["gamma"] / 2.0 - fit["beta"], 1e-6
                )
                fits.append({"t": t, **fit})
        if fit is not None:
            forecast[t] = sig2_state
            if t + 1 < n and np.isfinite(returns[t]):
                shock = returns[t] ** 2
                lev = fit["gamma"] * shock if returns[t] < 0.0 else 0.0
                sig2_state = (
                    fit["omega"] + fit["alpha"] * shock + lev + fit["beta"] * sig2_state
                )
    return forecast, fits


# --------------------------------------------------------------------------
# Phase 2b: Conditional EVT (McNeil-Frey two-stage) - peaks-over-threshold
# GPD on standardized residuals from an already-fit conditional variance model
# --------------------------------------------------------------------------


def fit_gpd_tail(
    z: np.ndarray, tail_frac: float = 0.10, tail: str = "lower"
) -> dict | None:
    """Peaks-over-threshold GPD fit to standardized residuals.

    z: standardized residuals r_t / sigma_t from an already-fit conditional
    variance model, computed on the TRAINING window only. Lower tail is
    handled by negating, so the same upper-tail machinery serves both.

    tail_frac: fraction of observations treated as "tail". 10% is the
    conventional default and is FIXED IN ADVANCE - do not tune it to improve
    a score. Report sensitivity at 5% and 15% as a robustness check only.
    """
    z = z[np.isfinite(z)]
    y = -z if tail == "lower" else z
    n = len(y)
    k = int(np.floor(tail_frac * n))
    if k < 30:
        return None
    ys = np.sort(y)
    u = ys[n - k - 1]  # threshold
    excess = ys[n - k :] - u  # exceedances over threshold
    excess = excess[excess > 0]
    if len(excess) < 30:
        return None
    xi, _loc, beta = st.genpareto.fit(excess, floc=0.0)
    if not np.isfinite(xi) or beta <= 0:
        return None
    return {
        "xi": float(xi),
        "beta": float(beta),
        "u": float(u),
        "n_exceed": len(excess),
        "n": int(n),
        "tail": tail,
    }


def gpd_var_es(fit: dict, q: float) -> tuple[float, float]:
    """Conditional VaR and ES at level q (an EXCEEDANCE probability, e.g.
    0.01 for a 1% tail), in units of the standardized residual.

    Standard POT tail estimator:
        z_q = u + (beta/xi) * [ (q * n / n_exceed)^(-xi) - 1 ]
    Expected shortfall beyond that quantile (finite only when xi < 1):
        ES_q = z_q/(1-xi) + (beta - xi*u)/(1-xi)

    Both are returned as POSITIVE magnitudes in the tail's own orientation;
    the caller re-signs for a lower tail. ES is the reason to do EVT at all -
    it is the coherent risk measure VaR is not, and no other rung in this
    research programme can produce one.
    """
    xi, beta, u = fit["xi"], fit["beta"], fit["u"]
    ratio = q * fit["n"] / fit["n_exceed"]
    z_q = (
        u + (beta / xi) * (ratio ** (-xi) - 1.0)
        if abs(xi) > 1e-8
        else u - beta * np.log(ratio)
    )
    es = np.nan
    if xi < 1.0:
        es = z_q / (1.0 - xi) + (beta - xi * u) / (1.0 - xi)
    return float(z_q), float(es)


def _variance_path_for_fit(fit: dict, window: np.ndarray, model: str) -> np.ndarray:
    """In-sample sigma^2 path for a single training window, recomputed from
    an already-fit GARCH or GJR model's own parameters - used to build the
    standardized residuals a GPD tail fit needs. `model` selects which
    variance recursion the fit's parameters belong to."""
    if model == "garch":
        omega, alpha, beta = fit["omega"], fit["alpha"], fit["beta"]
        uncond = omega / max(1 - alpha - beta, 1e-6)
        return L._garch_variance_path(omega, alpha, beta, window, uncond)
    if model == "gjr":
        omega, alpha, gamma, beta = (
            fit["omega"],
            fit["alpha"],
            fit["gamma"],
            fit["beta"],
        )
        uncond = omega / max(1 - alpha - gamma / 2.0 - beta, 1e-6)
        return _gjr_variance_path(omega, alpha, gamma, beta, window, uncond)
    raise ValueError(f"unknown model: {model!r}")


def rolling_gpd_paths(
    returns: np.ndarray,
    variance_fits: list[dict],
    model: str,
    max_train: int,
    tail_frac: float = 0.10,
) -> tuple[dict, dict]:
    """Causal, forward-filled GPD tail fits (both tails), refit exactly when
    the underlying GARCH/GJR variance model refits - using that SAME
    training window's own fitted variance recursion to standardize
    residuals, so the GPD threshold/xi/beta can never drift out of sync with
    the variance model's own refits (the critical causality note in
    NEXT_RUN_PROMPT.md's Phase 2b: fitting the GPD once on the whole sample,
    or refitting it on a different cadence than the variance model, would be
    the same class of lookahead bug as #1a, one level deeper).

    variance_fits: the `fits` list from rolling_garch_forecast /
    rolling_gjr_forecast (each entry has "t" - the refit bar index - and
    that model's own fitted parameters).

    Returns (paths, gpd_fits): paths is {"upper": {...}, "lower": {...}},
    each a dict of forward-filled arrays (xi, beta, u, n_exceed), aligned to
    `returns`' own length/index, exactly like nu_path_from_fits. gpd_fits
    mirrors variance_fits' own list-of-refit-records structure, per tail.
    """
    n = len(returns)
    paths = {
        tail: {key: np.full(n, np.nan) for key in ["xi", "beta", "u", "n_exceed"]}
        for tail in ["upper", "lower"]
    }
    gpd_fits: dict[str, list[dict]] = {"upper": [], "lower": []}
    for vf in variance_fits:
        t = vf["t"]
        start = max(0, t - max_train)
        window = returns[start:t]
        window = window[np.isfinite(window)]
        if len(window) < 60:
            continue
        sig2 = _variance_path_for_fit(vf, window, model)
        if np.any(sig2 <= 0) or not np.all(np.isfinite(sig2)):
            continue
        z = window / np.sqrt(sig2)
        for tail in ["upper", "lower"]:
            gfit = fit_gpd_tail(z, tail_frac=tail_frac, tail=tail)
            if gfit is None:
                continue
            gpd_fits[tail].append({"t": t, **gfit})
            for key in ["xi", "beta", "u", "n_exceed"]:
                paths[tail][key][t:] = gfit[key]
    return paths, gpd_fits


# --------------------------------------------------------------------------
# Phase 1c / Phase 3 d2: HAR-log-RV (fit HAR-style features on log(rv)
# instead of rv levels, exponentiate back with the lognormal mean correction)
# --------------------------------------------------------------------------


def har_log_rv_forecast(
    df: pl.DataFrame,
    interval: str,
    refit_every: int,
    min_train: int,
    resid_window: int = 250,
) -> np.ndarray:
    """HAR-style variance forecast built in log(rv) space rather than rv
    levels - Phase 1c found log-RV materially better PIT/KS-calibrated than
    raw RV at every interval, and NEXT_RUN_PROMPT.md asks for this as a
    ~10-line rung addition rather than letting it grow further.

    Fits the same daily/weekly/monthly trailing-mean-of-log-rv features as
    dist_lib.make_har_features, but on log(rv_target), via the same causal
    rolling_ols_refit used for rung2. The regression forecasts E[log(rv)];
    converting that back to a variance forecast needs the lognormal mean
    correction exp(mu + sigma^2/2), so a causal, forward-looking-safe
    residual variance is estimated as a trailing (shift(1)) rolling std of
    the model's own past residuals (log_rv_t - mu_hat_t, only ever using
    bars < t) - never a full-sample residual variance, which would leak.
    """
    bpd = max(1, 24 // L.INTERVAL_HOURS[interval])
    rv = df["rv_target"].to_numpy()
    log_rv = np.where(rv > 0, np.log(np.where(rv > 0, rv, 1.0)), np.nan)
    log_rv_df = df.with_columns(pl.Series("log_rv_target", log_rv))
    lrv = pl.col("log_rv_target")
    log_rv_df = log_rv_df.with_columns(
        lrv.rolling_mean(window_size=bpd, min_samples=1).shift(1).alias("log_rv_d"),
        lrv.rolling_mean(window_size=bpd * 5, min_samples=bpd)
        .shift(1)
        .alias("log_rv_w"),
        lrv.rolling_mean(window_size=bpd * 22, min_samples=bpd)
        .shift(1)
        .alias("log_rv_m"),
    )
    mu_hat = L.rolling_ols_refit(
        log_rv_df,
        ["log_rv_d", "log_rv_w", "log_rv_m"],
        "log_rv_target",
        refit_every=refit_every,
        min_train=min_train,
    )
    resid = log_rv - mu_hat
    sigma2_resid = (
        (
            pl.Series("resid", resid).rolling_std(
                window_size=resid_window, min_samples=resid_window // 4
            )
            ** 2
        )
        .shift(1)
        .to_numpy()
    )
    with np.errstate(invalid="ignore"):
        forecast = np.exp(mu_hat + sigma2_resid / 2.0)
    forecast[~np.isfinite(mu_hat) | ~np.isfinite(sigma2_resid)] = np.nan
    return forecast


# --------------------------------------------------------------------------
# Phase 3: vectorized density scoring (normal / t), using the closed-form
# CRPS from #1b instead of distributions.crps's numerical-grid version, and
# manual vectorized log-density formulas instead of building a scipy frozen
# distribution object per observation - both purely a speed concern (10s of
# thousands of bars x 8 models x 4 intervals), not a change in what's scored.
# --------------------------------------------------------------------------


def vectorized_normal_scores(actual: np.ndarray, variance_forecast: np.ndarray) -> dict:
    mask = (
        np.isfinite(actual) & np.isfinite(variance_forecast) & (variance_forecast > 0)
    )
    a, v = actual[mask], variance_forecast[mask]
    log_score = -0.5 * np.log(2 * np.pi * v) - 0.5 * (a**2) / v
    crps = dist.crps_normal_closed_form(a, loc=0.0, scale=np.sqrt(v))
    return {"mask": mask, "log_score": log_score, "crps": crps}


def vectorized_t_scores(
    actual: np.ndarray, variance_forecast: np.ndarray, nu: np.ndarray
) -> dict:
    nu = np.broadcast_to(np.asarray(nu, dtype=float), actual.shape)
    mask = (
        np.isfinite(actual)
        & np.isfinite(variance_forecast)
        & (variance_forecast > 0)
        & np.isfinite(nu)
        & (nu > 2)
    )
    a, v, n_ = actual[mask], variance_forecast[mask], nu[mask]
    c = np.sqrt(n_ / (n_ - 2))
    scale = np.sqrt(v) / c
    log_score = st.t.logpdf(a / scale, df=n_) - np.log(scale)
    crps = dist.crps_t_closed_form(a, df=n_, loc=0.0, scale=scale)
    return {"mask": mask, "log_score": log_score, "crps": crps}


def benjamini_hochberg(pvalues: dict, alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg FDR adjustment. `pvalues`: {key: pvalue}. Returns
    {key: {"pvalue": ..., "rank": ..., "bh_threshold": ..., "significant": bool}},
    where "significant" applies the standard BH decision rule (reject every
    hypothesis up to and including the largest rank k with p_(k) <= k/m*alpha).
    """
    items = sorted(
        pvalues.items(), key=lambda kv: np.inf if not np.isfinite(kv[1]) else kv[1]
    )
    finite = [kv for kv in items if np.isfinite(kv[1])]
    m_finite = len(finite)
    # find the largest rank k (1-indexed among finite p-values) satisfying BH's rule
    k_star = 0
    for rank, (_key, p) in enumerate(finite, start=1):
        if p <= (rank / m_finite) * alpha:
            k_star = rank
    out = {}
    for rank, (key, p) in enumerate(items, start=1):
        is_finite = np.isfinite(p)
        entry: dict[str, float | int | bool | None]
        if is_finite:
            bh_rank = finite.index((key, p)) + 1
            entry = {
                "pvalue": float(p),
                "rank": bh_rank,
                "bh_threshold": float((bh_rank / m_finite) * alpha),
                "significant": bool(bh_rank <= k_star),
            }
        else:
            entry = {
                "pvalue": None,
                "rank": None,
                "bh_threshold": None,
                "significant": False,
            }
        out[key] = entry
    return out


# --------------------------------------------------------------------------
# Phase 4: quantile forecasts + the coverage battery
# --------------------------------------------------------------------------

QUANTILES = [0.01, 0.025, 0.05, 0.95, 0.975, 0.99]


def normal_quantile_forecasts(
    variance_forecast: np.ndarray, quantiles=QUANTILES
) -> dict:
    """VaR forecast at every quantile level, normal innovations, causal
    (uses only variance_forecast, itself already causal)."""
    sigma = np.sqrt(np.where(variance_forecast > 0, variance_forecast, np.nan))
    return {q: st.norm.ppf(q, loc=0.0, scale=sigma) for q in quantiles}


def t_quantile_forecasts(
    variance_forecast: np.ndarray, nu_path: np.ndarray, quantiles=QUANTILES
) -> dict:
    """VaR forecast at every quantile level under a causal, per-bar Student-t
    shape path (same nu_path convention as nu_path_from_fits)."""
    nu = np.asarray(nu_path, dtype=float)
    valid = np.isfinite(nu) & (nu > 2) & (variance_forecast > 0)
    c = np.where(valid, np.sqrt(nu / np.where(nu > 2, nu - 2, np.nan)), np.nan)
    scale = np.where(
        valid,
        np.sqrt(np.where(variance_forecast > 0, variance_forecast, np.nan)) / c,
        np.nan,
    )
    out = {}
    for q in quantiles:
        qf = np.full(len(nu), np.nan)
        qf[valid] = st.t.ppf(q, df=nu[valid], loc=0.0, scale=scale[valid])
        out[q] = qf
    return out


def gpd_quantile_forecasts(
    variance_forecast: np.ndarray, gpd_paths: dict, quantiles=QUANTILES
) -> dict:
    """VaR forecast at every quantile level from causal, forward-filled GPD
    tail fits (rolling_gpd_paths' own output) combined with the underlying
    model's causal sigma_t - VaR_t(q) = -sigma_t * z_q for the lower tail,
    +sigma_t * z_q for the upper (gpd_var_es returns POSITIVE magnitudes in
    the tail's own orientation; re-signed here for the caller's quantile
    convention). Quantile levels below 0.5 use the lower-tail GPD fit at
    exceedance probability q; levels above 0.5 use the upper-tail fit at
    exceedance probability (1-q).
    """
    sigma = np.sqrt(np.where(variance_forecast > 0, variance_forecast, np.nan))
    n = len(sigma)
    out = {}
    for q in quantiles:
        tail = "lower" if q < 0.5 else "upper"
        exceed_q = q if q < 0.5 else 1.0 - q
        p = gpd_paths[tail]
        qf = np.full(n, np.nan)
        for t in range(n):
            xi, beta, u, n_exceed = (
                p["xi"][t],
                p["beta"][t],
                p["u"][t],
                p["n_exceed"][t],
            )
            if not (
                np.isfinite(xi)
                and np.isfinite(beta)
                and np.isfinite(u)
                and np.isfinite(n_exceed)
            ):
                continue
            fit = {
                "xi": xi,
                "beta": beta,
                "u": u,
                "n_exceed": int(n_exceed),
                "n": int(n_exceed / 0.10),
            }
            z_q, _es_q = gpd_var_es(fit, exceed_q)
            qf[t] = -sigma[t] * z_q if tail == "lower" else sigma[t] * z_q
        out[q] = qf
    return out


def coverage_battery(actual: np.ndarray, quantile_forecasts: dict) -> dict:
    """Full coverage grid: Kupiec (unconditional), Christoffersen independence
    (do violations cluster?), and conditional coverage (joint) at every level.

    Kupiec alone counts violations; it cannot see that they all arrived in the
    same week. Given Phase 1 of notebook 4 measured gamma waiting-time shapes
    of 0.52-0.85 (violations demonstrably cluster), the independence test is
    where a normal-innovation model is expected to break, and it is precisely
    the test notebook 4 ran for exactly one model at exactly one level.
    """
    out = {}
    for q, qf in quantile_forecasts.items():
        side = "lower" if q < 0.5 else "upper"
        rate = q if q < 0.5 else 1.0 - q
        mask = np.isfinite(actual) & np.isfinite(qf)
        hits = dist.exceedances(actual[mask], qf[mask], side=side)
        _, kp = dist.kupiec_test(hits, rate)
        _, ip = dist.christoffersen_independence_test(hits)
        _, cp = dist.christoffersen_conditional_coverage_test(hits, rate)
        out[str(q)] = {
            "kupiec_p": float(kp),
            "indep_p": float(ip),
            "cc_p": float(cp),
            "observed_rate": float(np.mean(hits)),
            "expected_rate": rate,
            "n": int(mask.sum()),
            "n_violations": int(hits.sum()),
        }
    return out


# --------------------------------------------------------------------------
# Expected shortfall forecasts + Acerbi-Szekely null simulation
#
# Analytic ES formulas for normal/t (verified numerically against Monte
# Carlo integration before use): ES_q = -sigma*phi(z_q)/q (normal),
# ES_q = -scale*(nu+z_q^2)/((nu-1)*q)*f_nu(z_q) (Student-t), z_q the
# q-quantile of the standard family. Both are the standard closed forms
# (Patton 2019 and elsewhere); re-derived and checked here rather than only
# cited, per this repo's own "read the actual numbers" discipline.
# --------------------------------------------------------------------------


def normal_es_forecast(variance_forecast: np.ndarray, q: float) -> np.ndarray:
    sigma = np.sqrt(np.where(variance_forecast > 0, variance_forecast, np.nan))
    z_q = st.norm.ppf(q)
    return -sigma * st.norm.pdf(z_q) / q


def t_es_forecast(
    variance_forecast: np.ndarray, nu_path: np.ndarray, q: float
) -> np.ndarray:
    nu = np.asarray(nu_path, dtype=float)
    valid = np.isfinite(nu) & (nu > 2) & (variance_forecast > 0)
    c = np.where(valid, np.sqrt(nu / np.where(nu > 2, nu - 2, np.nan)), np.nan)
    scale = np.where(
        valid,
        np.sqrt(np.where(variance_forecast > 0, variance_forecast, np.nan)) / c,
        np.nan,
    )
    z_q = st.t.ppf(q, df=np.where(valid, nu, np.nan))
    f_zq = st.t.pdf(z_q, df=np.where(valid, nu, np.nan))
    return -scale * (nu + z_q**2) / (nu - 1) / q * f_zq


def gpd_es_forecast(
    variance_forecast: np.ndarray, gpd_paths: dict, q: float
) -> np.ndarray:
    """ES companion to gpd_quantile_forecasts, at exceedance probability q
    (q<0.5: lower tail directly; q>0.5 callers should pass 1-q and re-sign,
    matching gpd_quantile_forecasts' own convention - kept separate here
    since VaR and ES are needed together at the SAME tail/level for the
    Acerbi-Szekely backtest)."""
    sigma = np.sqrt(np.where(variance_forecast > 0, variance_forecast, np.nan))
    tail = "lower" if q < 0.5 else "upper"
    exceed_q = q if q < 0.5 else 1.0 - q
    p = gpd_paths[tail]
    n = len(sigma)
    es = np.full(n, np.nan)
    for t in range(n):
        xi, beta, u, n_exceed = p["xi"][t], p["beta"][t], p["u"][t], p["n_exceed"][t]
        if not (
            np.isfinite(xi)
            and np.isfinite(beta)
            and np.isfinite(u)
            and np.isfinite(n_exceed)
        ):
            continue
        fit = {
            "xi": xi,
            "beta": beta,
            "u": u,
            "n_exceed": int(n_exceed),
            "n": int(n_exceed / 0.10),
        }
        _z_q, es_q = gpd_var_es(fit, exceed_q)
        if np.isfinite(es_q):
            es[t] = -sigma[t] * es_q if tail == "lower" else sigma[t] * es_q
    return es


def _simulate_z_from_uniforms(
    u: np.ndarray, sim_values: np.ndarray, es_forecast: np.ndarray, q: float
) -> float:
    """Shared Z-statistic assembly: given each bar's own draw u~Uniform(0,1),
    a bar counts as a simulated hit iff u<q (exactly reproducing hit
    probability q under the null), and its simulated value is whatever the
    model's own inverse-CDF gives at that u - so hit bars automatically carry
    the correct conditional-on-hit distribution without a separate
    conditional draw. Non-hit bars contribute 0 to the sum, matching
    acerbi_szekely_z's own indicator-masked formula exactly (including the
    "-1", not "+1" - see acerbi_szekely_z's own docstring for why)."""
    n = len(u)
    hit = u < q
    mask = hit & np.isfinite(es_forecast) & (es_forecast != 0) & np.isfinite(sim_values)
    if mask.sum() == 0:
        return float("nan")
    return float(
        (1.0 / (n * q)) * np.sum(np.where(mask, sim_values / es_forecast, 0.0)) - 1.0
    )


def make_normal_acerbi_simulate_fn(
    sigma: np.ndarray, es_forecast: np.ndarray, q: float
):
    def simulate(rng):
        u = rng.uniform(0.0, 1.0, len(sigma))
        sim_values = sigma * st.norm.ppf(u)
        return _simulate_z_from_uniforms(u, sim_values, es_forecast, q)

    return simulate


def make_t_acerbi_simulate_fn(
    sigma: np.ndarray, nu: np.ndarray, es_forecast: np.ndarray, q: float
):
    valid = np.isfinite(nu) & (nu > 2)
    c = np.where(valid, np.sqrt(nu / np.where(nu > 2, nu - 2, np.nan)), np.nan)
    scale = sigma / c

    def simulate(rng):
        u = rng.uniform(0.0, 1.0, len(sigma))
        sim_values = np.where(
            valid, scale * st.t.ppf(u, df=np.where(valid, nu, np.nan)), np.nan
        )
        return _simulate_z_from_uniforms(u, sim_values, es_forecast, q)

    return simulate


def make_gpd_acerbi_simulate_fn(
    sigma: np.ndarray, gpd_paths: dict, es_forecast: np.ndarray, q: float
):
    tail = "lower" if q < 0.5 else "upper"
    p = gpd_paths[tail]
    xi, beta, u_thr, n_exceed = p["xi"], p["beta"], p["u"], p["n_exceed"]

    def simulate(rng):
        n = len(sigma)
        u = rng.uniform(0.0, 1.0, n)
        sim_values = np.full(n, np.nan)
        hit = u < q
        for t in np.where(hit)[0]:
            if not (
                np.isfinite(xi[t])
                and np.isfinite(beta[t])
                and np.isfinite(u_thr[t])
                and np.isfinite(n_exceed[t])
            ):
                continue
            fit = {
                "xi": xi[t],
                "beta": beta[t],
                "u": u_thr[t],
                "n_exceed": int(n_exceed[t]),
                "n": int(n_exceed[t] / 0.10),
            }
            z_u, _es = gpd_var_es(fit, u[t])
            sim_values[t] = -sigma[t] * z_u if tail == "lower" else sigma[t] * z_u
        return _simulate_z_from_uniforms(u, sim_values, es_forecast, q)

    return simulate
