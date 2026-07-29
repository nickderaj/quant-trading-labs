"""Shared machinery for notebook 4 (distributional models for volatility and
regime). Single-asset, single-interval throughout - no cross-sectional/pooled
code from notebook 3.

Run as a script from the repo root (`sys.path.insert(0, "src")`), and
imported from the notebook itself in the same way build_notebook.py's
predecessors were (`sys.path.insert(0, "tmp")` from `src/research/`).
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime

sys.path.insert(0, "src")

import numpy as np
import polars as pl
from scipy import stats as st
from scipy.optimize import minimize

import data
import distributions as dist
import research

CACHE_DIR = "src/research/cache"
DOWNLOAD_DIR = "src/research/tmp"
START = datetime(2021, 7, 1, tzinfo=UTC)
FULL_END = datetime(2026, 6, 30, tzinfo=UTC)
HOLDOUT_START = research.HOLDOUT_START

INTERVAL_HOURS = {"1h": 1, "4h": 4, "12h": 12, "1d": 24}

# --------------------------------------------------------------------------
# Loading + feature engineering (full OHLCV bar, not just close)
# --------------------------------------------------------------------------


def load_klines(symbol: str, interval: str, end: datetime = FULL_END) -> pl.DataFrame:
    """Cache-only load (data is fully cached; this must not hit the network)."""
    return data.download_klines_range(
        symbol, interval, START, end, download_dir=DOWNLOAD_DIR, cache_dir=CACHE_DIR
    )


def realized_variance_from_subbars(symbol: str, interval: str, end: datetime = FULL_END) -> pl.DataFrame:
    """RV proxy for a coarse bar built from higher-frequency (1h) sub-bars
    where available, per NEW_PROMPT's "computed from higher-frequency bars
    where available" - a materially less noisy proxy than the coarse bar's
    own single squared return for 4h/12h/1d. 1h itself has no finer cached
    series, so its RV proxy is unavoidably the single-bar squared return
    (noted explicitly wherever it's used).
    """
    hrs = INTERVAL_HOURS[interval]
    if hrs == 1:
        raise ValueError("1h has no finer sub-bar series cached; use bar_squared_return instead")
    base = load_klines(symbol, "1h", end).sort("datetime")
    base = base.with_columns((pl.col("close") / pl.col("close").shift(1)).log().alias("r1h"))
    base = base.with_columns(
        pl.col("datetime").dt.truncate(f"{hrs}h").alias("bucket")
    )
    rv = base.group_by("bucket").agg(
        (pl.col("r1h") ** 2).sum().alias("rv_subbar"),
        pl.col("r1h").count().alias("n_subbars"),
    ).rename({"bucket": "datetime"}).sort("datetime")
    return rv


def build_asset_frame(symbol: str, interval: str, end: datetime = FULL_END) -> pl.DataFrame:
    """One symbol/interval's causal feature frame: returns, gap/intrabar
    decomposition, range-estimator per-bar components, activity variables,
    and the RV target (from sub-bars where available). Everything here is a
    function of bar t and earlier only (log_return_fwd is the one
    deliberately-forward column, used as the *target*, never as a feature).
    """
    df = load_klines(symbol, interval, end).sort("datetime")
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1)).log().alias("log_return"),
        (pl.col("open") / pl.col("close").shift(1)).log().alias("gap_return"),
        (pl.col("close") / pl.col("open")).log().alias("intrabar_return"),
        (pl.col("high") / pl.col("low")).log().alias("hl_log"),
        ((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low"))).alias(
            "intrabar_close_pos"
        ),
        (pl.col("taker_buy_volume") / pl.col("volume")).alias("taker_buy_ratio"),
    )
    # per-bar range-estimator variance contributions (Parkinson / Garman-Klass / Rogers-Satchell)
    ln2 = float(np.log(2.0))
    df = df.with_columns(
        (pl.col("hl_log") ** 2 / (4.0 * ln2)).alias("pk_var"),
        (
            0.5 * pl.col("hl_log") ** 2
            - (2.0 * ln2 - 1.0) * pl.col("intrabar_return") ** 2
        ).alias("gk_var"),
        (
            (pl.col("high") / pl.col("close")).log() * (pl.col("high") / pl.col("open")).log()
            + (pl.col("low") / pl.col("close")).log() * (pl.col("low") / pl.col("open")).log()
        ).alias("rs_var"),
        (pl.col("log_return") ** 2).alias("bar_squared_return"),
    )
    if INTERVAL_HOURS[interval] > 1:
        rv = realized_variance_from_subbars(symbol, interval, end)
        df = df.join(rv, on="datetime", how="left")
        df = df.with_columns(pl.coalesce(["rv_subbar", "bar_squared_return"]).alias("rv_target"))
    else:
        df = df.with_columns(pl.col("bar_squared_return").alias("rv_target"))
    return df


# --------------------------------------------------------------------------
# Phase 1 helpers: descriptive fits (fit once, causal-to-date -> use
# fit_rolling with window == len(series) so only the last row actually
# fits, per distributions.py's own min_periods semantics)
# --------------------------------------------------------------------------


def fit_once(df: pl.DataFrame, col: str, family: str) -> tuple[float, ...] | None:
    """One causal-to-date fit, reusing distributions.fit_rolling rather than
    reinventing per-family fitting: window == len(df) means every row before
    the last is `n_insufficient_history` and only the final row is actually
    fit, so this is O(1) fits, not O(n).
    """
    sub = df.select(col).drop_nulls()
    n = len(sub)
    if n == 0:
        return None
    result = dist.fit_rolling(sub, col, family, window=n, min_periods=n)
    row = result.frame.tail(1)
    params = dist.FAMILY_PARAMS[family]
    vals = [row[f"{col}_{family}_{p}"][0] for p in params]
    if any(v is None for v in vals):
        return None
    return tuple(float(v) for v in vals)


def waiting_times_between_k_sigma(returns: np.ndarray, k: float) -> np.ndarray:
    """Bar-count gaps between successive |z| >= k events, z = return
    standardized by the full-sample std (descriptive, not causal - Phase 1
    is explicitly the "fit once on the whole history" phase)."""
    z = (returns - np.nanmean(returns)) / np.nanstd(returns)
    idx = np.where(np.abs(z) >= k)[0]
    if len(idx) < 2:
        return np.array([])
    return np.diff(idx).astype(float)


def run_lengths(returns: np.ndarray) -> np.ndarray:
    """Bars until the sign of the return flips (run length of same-sign
    returns), zero returns dropped (sign undefined)."""
    signs = np.sign(returns[returns != 0])
    if len(signs) < 2:
        return np.array([])
    runs = []
    cur = 1
    for i in range(1, len(signs)):
        if signs[i] == signs[i - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return np.array(runs, dtype=float)


# --------------------------------------------------------------------------
# Phase 3: vol forecasting ladder
# --------------------------------------------------------------------------


def rung0_trailing_std(df: pl.DataFrame, window: int) -> pl.Series:
    """Trailing rolling std of log_return, variance forecast = std^2,
    causal (shifted by 1 so bar t's forecast uses data < t)."""
    return (
        df["log_return"].rolling_std(window_size=window, min_periods=window // 2 + 1) ** 2
    ).shift(1)


def rung1_ewma(df: pl.DataFrame, lam: float = 0.94) -> pl.Series:
    """RiskMetrics EWMA variance: sigma2_t = lam*sigma2_{t-1} + (1-lam)*r_{t-1}^2.
    Pure recursion (no MLE), cheap enough for a plain python loop even at 1h."""
    r2 = df["log_return"].fill_null(0.0).to_numpy() ** 2
    n = len(r2)
    sig2 = np.full(n, np.nan)
    sig2[0] = np.nanvar(r2[: min(50, n)]) if n else np.nan
    for t in range(1, n):
        prev = sig2[t - 1] if np.isfinite(sig2[t - 1]) else r2[t - 1]
        sig2[t] = lam * prev + (1 - lam) * r2[t - 1]
    return pl.Series("ewma_var", sig2)


def har_rv_features(df: pl.DataFrame) -> pl.DataFrame:
    """Daily/weekly/monthly RV components in the bar's own units (window
    sizes scaled by how many bars/day this interval has)."""
    bars_per_day = 24 // INTERVAL_HOURS[list(INTERVAL_HOURS.keys())[0]]  # placeholder unused
    return df


def make_har_features(df: pl.DataFrame, interval: str) -> pl.DataFrame:
    """Daily/weekly/monthly trailing RV components, HAR-RV style.

    Bug found and fixed: the rolling means here were not shifted, so at
    bpd=1 (the 1d interval) `rv_d` (window_size=1) was *literally identical*
    to `rv_target` for the same bar - the rolling-OLS forecast built from it
    was regressing rv_target on rv_target, a same-bar target leak, not a
    forecast. This is exactly the lookahead trap NEW_PROMPT.md warns about
    ("future-informed standardisation... any result that looks implausibly
    good"), and it was visible as HAR-RV's QLIKE loss coming back ~0.000000
    at 1d. All three windows are now shifted by 1 bar, so the feature used
    to forecast bar t's rv_target only ever contains bars < t.
    """
    bpd = max(1, 24 // INTERVAL_HOURS[interval])
    rv = pl.col("rv_target")
    return df.with_columns(
        rv.rolling_mean(window_size=bpd, min_periods=1).shift(1).alias("rv_d"),
        rv.rolling_mean(window_size=bpd * 5, min_periods=bpd).shift(1).alias("rv_w"),
        rv.rolling_mean(window_size=bpd * 22, min_periods=bpd).shift(1).alias("rv_m"),
    )


def rolling_ols_refit(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str,
    refit_every: int,
    min_train: int,
) -> np.ndarray:
    """Rolling-refit OLS forecast: at each refit point, fit y ~ 1 + X on all
    causal history to date, forward-fill the coefficients until the next
    refit, predict every bar in between with those coefficients applied to
    that bar's *own* (already-causal, already-lagged) features. Returns the
    forecast array (same length as df, NaN before the first refit)."""
    n = len(df)
    X_full = df.select(feature_cols).to_numpy()
    y_full = df[target_col].to_numpy()
    forecast = np.full(n, np.nan)
    beta = None
    for t in range(n):
        if t >= min_train and (beta is None or t % refit_every == 0):
            train_mask = np.arange(0, t)
            Xt = X_full[train_mask]
            yt = y_full[train_mask]
            valid = np.all(np.isfinite(Xt), axis=1) & np.isfinite(yt)
            if valid.sum() > len(feature_cols) + 5:
                Xd = np.column_stack([np.ones(valid.sum()), Xt[valid]])
                try:
                    beta, *_ = np.linalg.lstsq(Xd, yt[valid], rcond=None)
                except np.linalg.LinAlgError:
                    beta = None
        if beta is not None and np.all(np.isfinite(X_full[t])):
            forecast[t] = beta[0] + X_full[t] @ beta[1:]
    return forecast


def range_estimator_forecasts(df: pl.DataFrame, window: int) -> pl.DataFrame:
    """Rolling-mean forecasts (shift(1), causal) of the Parkinson /
    Garman-Klass / Rogers-Satchell per-bar variance components, plus
    Yang-Zhang combining overnight + open + RS variance over the same
    window - all vectorized with polars rolling ops, no python loop."""
    n = window
    k = 0.34 / (1.34 + (n + 1) / max(n - 1, 1))
    out = df.with_columns(
        pl.col("pk_var").rolling_mean(window_size=n, min_periods=n // 2 + 1).shift(1).alias("fc_parkinson"),
        pl.col("gk_var").rolling_mean(window_size=n, min_periods=n // 2 + 1).shift(1).alias("fc_gk"),
        pl.col("rs_var").rolling_mean(window_size=n, min_periods=n // 2 + 1).shift(1).alias("fc_rs"),
        pl.col("gap_return").rolling_var(window_size=n, min_periods=n // 2 + 1).shift(1).alias("v_o"),
        pl.col("intrabar_return").rolling_var(window_size=n, min_periods=n // 2 + 1).shift(1).alias("v_c"),
    )
    out = out.with_columns(
        (pl.col("v_o") + k * pl.col("v_c") + (1 - k) * pl.col("fc_rs")).alias("fc_yz")
    )
    return out


# --------------------------------------------------------------------------
# GARCH(1,1): manual MLE (normal / t / skew-t innovations), rolling refit
# --------------------------------------------------------------------------


def _garch_variance_path(omega: float, alpha: float, beta: float, r: np.ndarray, sig2_0: float) -> np.ndarray:
    n = len(r)
    sig2 = np.empty(n)
    sig2[0] = sig2_0
    for t in range(1, n):
        sig2[t] = omega + alpha * r[t - 1] ** 2 + beta * sig2[t - 1]
    return sig2


def _garch_negloglik(params: np.ndarray, r: np.ndarray, innovation: str) -> float:
    if innovation == "normal":
        omega, alpha, beta = params
        extra = ()
    elif innovation == "t":
        omega, alpha, beta, nu = params
        if nu <= 2.1:
            return 1e10
        extra = (nu,)
    else:  # skewt: jf_skew_t(a, b), standardized innovations
        omega, alpha, beta, a, b = params
        if a <= 0.5 or b <= 0.5:
            return 1e10
        extra = (a, b)

    if omega <= 1e-12 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return 1e10

    uncond = omega / max(1 - alpha - beta, 1e-6)
    sig2 = _garch_variance_path(omega, alpha, beta, r, uncond)
    if np.any(sig2 <= 0) or not np.all(np.isfinite(sig2)):
        return 1e10
    z = r / np.sqrt(sig2)

    if innovation == "normal":
        ll = -0.5 * np.log(2 * np.pi * sig2) - 0.5 * z**2
    elif innovation == "t":
        (nu,) = extra
        # standardized (unit-variance) Student-t density
        c = np.sqrt(nu / (nu - 2))
        zt = z * c
        ll = st.t.logpdf(zt, df=nu) + np.log(c) - 0.5 * np.log(sig2)
    else:
        a, b = extra
        d = st.jf_skew_t(a=a, b=b)
        m, v = d.mean(), d.var()
        if not (np.isfinite(m) and np.isfinite(v) and v > 1e-8):
            return 1e10
        zs = z * np.sqrt(v) + m
        ll = d.logpdf(zs) + 0.5 * np.log(v) - 0.5 * np.log(sig2)

    if not np.all(np.isfinite(ll)):
        return 1e10
    return -float(np.sum(ll))


def fit_garch11(r: np.ndarray, innovation: str = "normal") -> dict | None:
    """MLE GARCH(1,1). No arch/statsmodels dependency in this environment,
    so this is a from-scratch scipy.optimize.minimize fit: variance
    recursion in a plain python loop (inherently sequential, can't
    vectorize away), log-likelihood under normal / Student-t / skew-t
    (jf_skew_t, standardized to zero-mean/unit-variance) innovations.
    Returns None on non-convergence or a degenerate (near-zero variance)
    window, same "null rather than propagate junk" convention as
    distributions.py.
    """
    r = r[np.isfinite(r)]
    if len(r) < 60 or np.std(r) <= 1e-10:
        return None
    var0 = float(np.var(r))
    if innovation == "normal":
        x0 = np.array([0.1 * var0, 0.05, 0.9])
        bounds = [(1e-10, None), (0, 1), (0, 1)]
    elif innovation == "t":
        x0 = np.array([0.1 * var0, 0.05, 0.9, 8.0])
        bounds = [(1e-10, None), (0, 1), (0, 1), (2.2, 60)]
    else:
        x0 = np.array([0.1 * var0, 0.05, 0.9, 2.0, 2.0])
        bounds = [(1e-10, None), (0, 1), (0, 1), (0.6, 30), (0.6, 30)]

    try:
        res = minimize(
            _garch_negloglik, x0, args=(r, innovation), method="L-BFGS-B",
            bounds=bounds, options={"maxiter": 200},
        )
    except Exception:
        return None
    if not res.success and res.status not in (0, 1):
        # status 1 = max iterations reached but often still usable; anything else -> reject
        pass
    if not np.all(np.isfinite(res.x)):
        return None
    if np.allclose(res.x, x0, atol=1e-9):
        return None  # optimizer never moved -> treat as unconverged

    params = res.x
    omega, alpha, beta = params[0], params[1], params[2]
    if alpha + beta >= 0.999:
        return None
    uncond = omega / max(1 - alpha - beta, 1e-6)
    sig2 = _garch_variance_path(omega, alpha, beta, r, uncond)
    next_sig2 = omega + alpha * r[-1] ** 2 + beta * sig2[-1]
    return {
        "omega": float(omega), "alpha": float(alpha), "beta": float(beta),
        "params": params.tolist(), "next_var": float(next_sig2),
        "innovation": innovation,
    }


def rolling_garch_forecast(
    returns: np.ndarray, refit_every: int, min_train: int, innovation: str = "normal",
    max_train: int = 1500,
) -> tuple[np.ndarray, list[dict]]:
    """Rolling-refit GARCH(1,1) one-step-ahead variance forecast. Refits
    every `refit_every` bars on a trailing window capped at `max_train`
    observations (bounds MLE cost independent of series length), forward-
    fills the fitted model's one-step-ahead forecast between refits by
    re-rolling its own variance recursion forward on realized returns (so
    the forecast used at bar t only ever depends on returns < t)."""
    n = len(returns)
    forecast = np.full(n, np.nan)
    fits = []
    fit = None
    sig2_state = np.nan
    for t in range(n):
        if t >= min_train and t % refit_every == 0:
            start = max(0, t - max_train)
            window = returns[start:t]
            new_fit = fit_garch11(window, innovation)
            if new_fit is not None:
                fit = new_fit
                sig2_state = fit["omega"] / max(1 - fit["alpha"] - fit["beta"], 1e-6)
                fits.append({"t": t, **fit})
        if fit is not None:
            forecast[t] = sig2_state
            if t + 1 < n and np.isfinite(returns[t]):
                sig2_state = fit["omega"] + fit["alpha"] * returns[t] ** 2 + fit["beta"] * sig2_state
    # forecast[t] currently holds the variance *used to forecast bar t's own
    # realization*; that's exactly the causal one-step-ahead forecast wanted.
    return forecast, fits


# --------------------------------------------------------------------------
# Rolling distributional fits on RV (gamma / inverse-gamma / lognormal)
# --------------------------------------------------------------------------


def rolling_rv_dist_forecast(rv: np.ndarray, family: str, refit_every: int, min_train: int, max_train: int = 2000) -> np.ndarray:
    """Rolling-refit scipy MLE fit of `family` to trailing RV history;
    forecast = fitted distribution's mean, forward-filled between refits."""
    n = len(rv)
    forecast = np.full(n, np.nan)
    mean_val = np.nan
    for t in range(n):
        if t >= min_train and t % refit_every == 0:
            start = max(0, t - max_train)
            window = rv[start:t]
            window = window[np.isfinite(window) & (window > 0)]
            if len(window) >= 30:
                try:
                    if family == "gamma":
                        a, loc, scale = st.gamma.fit(window, floc=0)
                        mean_val = a * scale
                    elif family == "invgamma":
                        a, loc, scale = st.invgamma.fit(window, floc=0)
                        mean_val = scale / (a - 1) if a > 1 else np.nan
                    elif family == "lognorm":
                        s, loc, scale = st.lognorm.fit(window, floc=0)
                        mean_val = scale * np.exp(s**2 / 2)
                    else:
                        raise ValueError(family)
                    if not (np.isfinite(mean_val) and mean_val > 0):
                        mean_val = np.nan
                except Exception:
                    mean_val = np.nan
        forecast[t] = mean_val
    return forecast


# --------------------------------------------------------------------------
# Activity-based (rung 6): count / dispersion index -> RV
# --------------------------------------------------------------------------


def activity_forecast(df: pl.DataFrame, window: int) -> pl.Series:
    """Rolling-refit linear regression of RV on trailing-window count and
    count-dispersion-index (mean-forward-filled coefficients between
    refits, same rolling_ols_refit machinery as HAR)."""
    feats = df.with_columns(
        pl.col("count").rolling_mean(window_size=window, min_periods=window // 2).shift(1).alias("count_mean"),
        (
            pl.col("count").rolling_std(window_size=window, min_periods=window // 2) ** 2
            / pl.col("count").rolling_mean(window_size=window, min_periods=window // 2)
        ).shift(1).alias("count_dispersion"),
    )
    fc = rolling_ols_refit(feats, ["count_mean", "count_dispersion"], "rv_target", refit_every=window, min_train=window * 3)
    return pl.Series("fc_activity", fc)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def qlike_mse(actual: np.ndarray, forecast: np.ndarray) -> dict:
    # QLIKE = actual/predicted - log(actual/predicted) - 1 is undefined at
    # actual == 0 (log(ratio) -> -inf as ratio -> 0), not just at negative
    # actual. A handful of exactly-zero-realized-variance bars (frozen-price
    # bars - the same failure mode notebook 3's realized_vol_24==0 bug was,
    # and flagged in NEW_PROMPT.md as certain to recur) previously slipped
    # through an `actual >= 0` mask and poisoned the whole-series mean QLIKE
    # to +inf for every rung at 1h. MSE has no such issue (a zero-actual bar
    # is a perfectly normal, finite squared-error term), so it's masked
    # separately with the original (actual >= 0) condition.
    mse_mask = np.isfinite(actual) & np.isfinite(forecast) & (forecast > 0) & (actual >= 0)
    qlike_mask = mse_mask & (actual > 0)
    a_mse, f_mse = actual[mse_mask], forecast[mse_mask]
    a_q, f_q = actual[qlike_mask], forecast[qlike_mask]
    if len(a_mse) < 10:
        return {"n": len(a_mse), "n_qlike": len(a_q), "qlike": np.nan, "mse": np.nan}
    mse = float(np.mean((a_mse - f_mse) ** 2))
    if len(a_q) < 10:
        return {"n": len(a_mse), "n_qlike": len(a_q), "qlike": np.nan, "mse": mse}
    q = dist.qlike(a_q, f_q)
    return {"n": len(a_mse), "n_qlike": len(a_q), "qlike": float(np.mean(q)), "mse": mse}


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray) -> tuple[float, float]:
    """DM test on a pointwise loss differential d = loss_a - loss_b
    (H0: equal predictive accuracy), Newey-West HAC variance at lag = n^(1/3).

    Bug found and fixed: `research.newey_west_tstat` returns `(mean, tstat)`,
    not `(tstat, pvalue)` - this function originally unpacked its result as
    `tstat, pvalue = newey_west_tstat(...)`, silently assigning the series
    *mean* to `tstat` and the real *tstat* to `pvalue`. Every DM p-value
    reported anywhere in Phase 3 was actually a raw HAC t-statistic (hence
    values >1 and negative "p-values"). Fixed by taking the real tstat and
    converting it to a two-sided p-value via the normal approximation
    (valid here: DM loss-differential series are always long, n>>30).
    """
    d = loss_a - loss_b
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 20:
        return float("nan"), float("nan")
    lag = max(1, int(round(n ** (1 / 3))))
    _mean, tstat = research.newey_west_tstat(d, lag)
    if not np.isfinite(tstat):
        return float(tstat), float("nan")
    pvalue = float(2 * st.norm.sf(abs(tstat)))
    return float(tstat), pvalue


def mincer_zarnowitz(actual: np.ndarray, forecast: np.ndarray) -> dict:
    mask = np.isfinite(actual) & np.isfinite(forecast)
    a, f = actual[mask], forecast[mask]
    if len(a) < 20 or np.std(f) < 1e-14:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan}
    X = np.column_stack([np.ones(len(f)), f])
    beta, *_ = np.linalg.lstsq(X, a, rcond=None)
    resid = a - X @ beta
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"intercept": float(beta[0]), "slope": float(beta[1]), "r2": float(r2)}


def density_scores(actual_returns: np.ndarray, variance_forecast: np.ndarray, family: str = "normal", extra_params=None) -> dict:
    """CRPS/log score of the return itself under N(0, forecast_var) (or a
    scaled-t if extra_params=(df,) given), plus 5%/95% quantile coverage
    (Kupiec + Christoffersen independence)."""
    mask = np.isfinite(actual_returns) & np.isfinite(variance_forecast) & (variance_forecast > 0)
    a, v = actual_returns[mask], variance_forecast[mask]
    if len(a) < 20:
        return {"log_score": np.nan, "crps": np.nan, "kupiec_p": np.nan, "christoffersen_p": np.nan}
    if family == "normal":
        dists = [st.norm(loc=0, scale=np.sqrt(vi)) for vi in v]
        q05 = st.norm(loc=0, scale=np.sqrt(v)).ppf(0.05)
    else:
        nu = extra_params[0]
        c = np.sqrt(nu / (nu - 2))
        dists = [st.t(df=nu, loc=0, scale=np.sqrt(vi) / c) for vi in v]
        q05 = st.t(df=nu, loc=0, scale=np.sqrt(v) / c).ppf(0.05)
    ls = dist.log_score(dists, a)
    cr = dist.crps(dists, a, n_points=400)
    hits = dist.exceedances(a, q05, side="lower")
    _, kp = dist.kupiec_test(hits, 0.05)
    _, cp = dist.christoffersen_independence_test(hits)
    return {
        "log_score": float(np.nanmean(ls)), "crps": float(np.nanmean(cr)),
        "kupiec_p": float(kp), "christoffersen_p": float(cp),
        "n": int(len(a)),
    }


# --------------------------------------------------------------------------
# Phase 4: regime models
# --------------------------------------------------------------------------


def fit_gmm_em(x: np.ndarray, k: int, n_iter: int = 50) -> dict | None:
    """1-D Gaussian mixture via plain-numpy EM (no sklearn in this
    environment). Canonical ordering imposed at the end: ascending fitted
    variance, so component 0 is always "low vol" - this is what makes state
    labels comparable across refit windows (label switching, guardrail #3)."""
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10 * k:
        return None
    qs = np.linspace(0.15, 0.85, k)
    means = np.quantile(x, qs)
    vars_ = np.full(k, np.var(x) + 1e-12)
    weights = np.full(k, 1.0 / k)
    for _ in range(n_iter):
        resp = np.column_stack(
            [weights[j] * st.norm.pdf(x, means[j], np.sqrt(vars_[j])) for j in range(k)]
        )
        s = resp.sum(axis=1, keepdims=True)
        s[s < 1e-300] = 1e-300
        resp = resp / s
        nk = np.maximum(resp.sum(axis=0), 1e-8)
        means = (resp * x[:, None]).sum(axis=0) / nk
        vars_ = np.maximum((resp * (x[:, None] - means[None, :]) ** 2).sum(axis=0) / nk, 1e-12)
        weights = nk / n
    order = np.argsort(vars_)
    return {"means": means[order].tolist(), "vars": vars_[order].tolist(), "weights": weights[order].tolist()}


def gmm_posterior(x: float, fit: dict) -> np.ndarray:
    means, vars_, weights = np.array(fit["means"]), np.array(fit["vars"]), np.array(fit["weights"])
    p = weights * st.norm.pdf(x, means, np.sqrt(vars_))
    s = p.sum()
    return p / s if s > 1e-300 else np.full(len(means), 1.0 / len(means))


def fit_hmm(x: np.ndarray, k: int = 2, n_iter: int = 30, emission: str = "gaussian", t_df: float | None = None) -> dict | None:
    """2/3-state HMM via Baum-Welch (forward-backward EM), from scratch
    (no hmmlearn in this environment). Gaussian emissions: exact EM M-step.
    Student-t emissions: `t_df` is estimated once (globally, held fixed -
    exact t M-step for location/scale has no closed form) and the M-step
    uses the responsibility-weighted mean/std as a method-of-moments
    location/scale update - a documented simplification, not full t-MLE.
    Canonical ascending-variance ordering imposed at the end.
    """
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20 * k:
        return None
    qs = np.linspace(0.15, 0.85, k)
    means = np.quantile(x, qs)
    vars_ = np.full(k, np.var(x) + 1e-12)
    A = np.full((k, k), 0.1 / (k - 1) if k > 1 else 1.0)
    np.fill_diagonal(A, 0.9)
    pi = np.full(k, 1.0 / k)

    def emis(xv):
        if emission == "gaussian":
            return np.column_stack([st.norm.pdf(xv, means[j], np.sqrt(vars_[j])) for j in range(k)])
        c = np.sqrt(t_df / (t_df - 2))
        return np.column_stack(
            [st.t.pdf(xv * c, df=t_df, loc=means[j] * c, scale=np.sqrt(vars_[j]) * c) * c for j in range(k)]
        )

    for _ in range(n_iter):
        B = np.clip(emis(x), 1e-300, None)
        alpha = np.zeros((n, k))
        c_scale = np.zeros(n)
        alpha[0] = pi * B[0]
        c_scale[0] = max(alpha[0].sum(), 1e-300)
        alpha[0] /= c_scale[0]
        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ A) * B[t]
            c_scale[t] = max(alpha[t].sum(), 1e-300)
            alpha[t] /= c_scale[t]
        beta = np.zeros((n, k))
        beta[-1] = 1.0
        for t in range(n - 2, -1, -1):
            beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c_scale[t + 1]
        gamma = alpha * beta
        gamma = gamma / np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
        xi_sum = np.zeros((k, k))
        for t in range(n - 1):
            xi_sum += (alpha[t][:, None] * A * B[t + 1][None, :] * beta[t + 1][None, :]) / c_scale[t + 1]
        pi = gamma[0]
        row_sums = np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-8)
        A = xi_sum / row_sums
        nk = np.maximum(gamma.sum(axis=0), 1e-8)
        means = (gamma * x[:, None]).sum(axis=0) / nk
        vars_ = np.maximum((gamma * (x[:, None] - means[None, :]) ** 2).sum(axis=0) / nk, 1e-12)

    if not (np.all(np.isfinite(means)) and np.all(np.isfinite(vars_)) and np.all(np.isfinite(A))):
        return None
    order = np.argsort(vars_)
    means, vars_ = means[order], vars_[order]
    A = A[np.ix_(order, order)]
    pi = pi[order]
    return {
        "means": means.tolist(), "vars": vars_.tolist(), "A": A.tolist(), "pi": pi.tolist(),
        "emission": emission, "t_df": t_df,
    }


def hmm_filter_step(alpha_prev: np.ndarray, x: float, fit: dict) -> np.ndarray:
    """One step of the forward filter using an already-fit (frozen) HMM:
    alpha_t propto (alpha_{t-1} @ A) * emission(x_t). Never uses future
    data or a backward pass - this is the filtered, tradeable probability,
    not the smoothed one (guardrail #1)."""
    means, vars_, A = np.array(fit["means"]), np.array(fit["vars"]), np.array(fit["A"])
    if fit["emission"] == "gaussian":
        b = st.norm.pdf(x, means, np.sqrt(vars_))
    else:
        nu = fit["t_df"]
        c = np.sqrt(nu / (nu - 2))
        b = st.t.pdf(x * c, df=nu, loc=means * c, scale=np.sqrt(vars_) * c) * c
    a = (alpha_prev @ A) * b
    s = a.sum()
    return a / s if s > 1e-300 else np.full(len(means), 1.0 / len(means))
