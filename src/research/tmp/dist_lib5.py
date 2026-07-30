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
from scipy import stats as st
from scipy.optimize import minimize

sys.path.insert(0, "src/research/tmp")
import dist_lib as L  # noqa: E402  (path must be set up first)

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
    top = y[n - k:]
    thresh = y[n - k - 1]
    xi = float(np.mean(np.log(top / thresh)))
    return 1.0 / xi if xi > 0 else np.nan


def hill_alpha_path(x: np.ndarray, tail: str = "upper", k_min: int = 20, k_max: int | None = None) -> dict:
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


def find_hill_plateau(alpha: np.ndarray, ks: np.ndarray, window: int = 50, rel_tol: float = 0.10) -> dict:
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
        return {"found": False, "reason": f"no {window}-wide window has relative spread < {rel_tol}"}
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
        "k_lo": int(k_lo), "k_hi": int(k_hi),
        "alpha_median": float(np.median(a[plateau_mask])),
        "alpha_min": float(np.min(a[plateau_mask])),
        "alpha_max": float(np.max(a[plateau_mask])),
        "k_chosen": int((k_lo + k_hi) // 2),
    }


# --------------------------------------------------------------------------
# Phase 2a: GJR-GARCH(1,1,1) - leverage. Nests dist_lib's plain GARCH(1,1)
# exactly at gamma=0, which is what makes the LR test on gamma=0 meaningful.
# --------------------------------------------------------------------------


def _gjr_variance_path(omega: float, alpha: float, gamma: float, beta: float, r: np.ndarray, sig2_0: float) -> np.ndarray:
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
    if innovation == "normal":
        omega, alpha, gamma, beta = params
        extra = ()
    elif innovation == "t":
        omega, alpha, gamma, beta, nu = params
        if nu <= 2.1:
            return 1e10
        extra = (nu,)
    else:
        raise ValueError(f"GJR only supports normal/t innovations (skew-t skipped as "
                          f"over-parameterized - see NEXT_RUN_PROMPT.md #Phase 2a), got {innovation!r}")

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
        (nu,) = extra
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
            _gjr_negloglik, x0, args=(r, innovation), method="L-BFGS-B",
            bounds=bounds, options={"maxiter": 200},
        )
    except Exception:
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
        "omega": float(omega), "alpha": float(alpha), "gamma": float(gamma), "beta": float(beta),
        "params": params.tolist(), "next_var": float(next_sig2), "innovation": innovation,
        "lr_gamma0_stat": lr_stat, "lr_gamma0_pvalue": lr_pvalue,
    }


def rolling_gjr_forecast(
    returns: np.ndarray, refit_every: int, min_train: int, innovation: str = "normal",
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
                sig2_state = fit["omega"] / max(1 - fit["alpha"] - fit["gamma"] / 2.0 - fit["beta"], 1e-6)
                fits.append({"t": t, **fit})
        if fit is not None:
            forecast[t] = sig2_state
            if t + 1 < n and np.isfinite(returns[t]):
                shock = returns[t] ** 2
                lev = fit["gamma"] * shock if returns[t] < 0.0 else 0.0
                sig2_state = fit["omega"] + fit["alpha"] * shock + lev + fit["beta"] * sig2_state
    return forecast, fits


# --------------------------------------------------------------------------
# Phase 2b: Conditional EVT (McNeil-Frey two-stage) - peaks-over-threshold
# GPD on standardized residuals from an already-fit conditional variance model
# --------------------------------------------------------------------------


def fit_gpd_tail(z: np.ndarray, tail_frac: float = 0.10, tail: str = "lower") -> dict | None:
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
    u = ys[n - k - 1]                      # threshold
    excess = ys[n - k:] - u                # exceedances over threshold
    excess = excess[excess > 0]
    if len(excess) < 30:
        return None
    xi, _loc, beta = st.genpareto.fit(excess, floc=0.0)
    if not np.isfinite(xi) or beta <= 0:
        return None
    return {"xi": float(xi), "beta": float(beta), "u": float(u),
            "n_exceed": int(len(excess)), "n": int(n), "tail": tail}


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
    z_q = u + (beta / xi) * (ratio ** (-xi) - 1.0) if abs(xi) > 1e-8 \
        else u - beta * np.log(ratio)
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
        omega, alpha, gamma, beta = fit["omega"], fit["alpha"], fit["gamma"], fit["beta"]
        uncond = omega / max(1 - alpha - gamma / 2.0 - beta, 1e-6)
        return _gjr_variance_path(omega, alpha, gamma, beta, window, uncond)
    raise ValueError(f"unknown model: {model!r}")


def rolling_gpd_paths(
    returns: np.ndarray, variance_fits: list[dict], model: str, max_train: int,
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
