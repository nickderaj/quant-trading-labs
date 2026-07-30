"""Distributional-modelling machinery: rolling causal parameter fits and
proper scoring rules, for notebook 4 (volatility and regime).

Mirrors the conventions in features.py: polars in/out, strictly causal
(every fitted row uses only data <= that row), and degenerate windows are
handled defensively rather than allowed to raise or to silently report junk.

Two failure modes this module exists to guard against, both drawn from bugs
already found in this repo:

1. The `realized_vol_24 == 0` bug from notebook 3 (a frozen-price window with
   zero variance breaks every distribution's variance-based moments) -
   *every* fit function here detects a degenerate window and returns None
   rather than propagating NaN/inf parameters silently. See
   `RollingFitResult` for how dropped windows are counted so this is
   inspectable, not silently swallowed.
2. A scipy MLE optimizer that fails to converge and just hands back its own
   starting guess as though it were the fit. `_mle_fit_with_convergence_check`
   below checks scipy's own optimizer exit flag (via a custom `optimizer=`
   callback into `fmin`'s `full_output`) and, as a second, independent check,
   compares the returned parameters to the initial guess actually used - if
   the "fit" is suspiciously close to where it started, it is treated as
   unconverged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats as st
from scipy.optimize import fmin
from scipy.special import beta as _betafn
from scipy.special import xlogy

# --------------------------------------------------------------------------
# Family registry
# --------------------------------------------------------------------------

# Parameter names, in the order each family's fit function returns them and
# in the order frozen_dist expects them.
FAMILY_PARAMS: dict[str, tuple[str, ...]] = {
    "normal": ("loc", "scale"),
    "t": ("df", "loc", "scale"),
    "skewt": ("a", "b", "loc", "scale"),
    "poisson": ("mu",),
    "nbinom": ("n", "p"),
    "beta": ("a", "b"),
}

# Minimum raw observations (post NaN-drop) before a fit is even attempted.
# Below this, a "fit" is really just reporting the starting guess, so these
# windows are counted as insufficient history rather than fit at all.
_MIN_OBS = {
    "normal": 2,
    "t": 10,
    "skewt": 15,
    "poisson": 1,
    "nbinom": 2,
    "beta": 2,
}


# --------------------------------------------------------------------------
# MLE convergence detection (used by the t and skew-t fits)
# --------------------------------------------------------------------------


def _mle_fit_with_convergence_check(
    dist: st.rv_continuous, x: np.ndarray
) -> tuple[float, ...] | None:
    """scipy.stats `<dist>.fit(x)` MLE, with the resulting optimizer state
    inspected rather than trusted blindly.

    Two independent checks, either of which nulls the fit:
    1. `scipy.optimize.fmin`'s own `warnflag` (exposed via `full_output`,
       passed through a custom `optimizer=` callback) - nonzero means fmin
       itself declared it did not converge (hit maxiter/maxfun or the
       simplex didn't shrink).
    2. The returned parameters are (numerically) identical to the initial
       guess `fmin` was started from - the exact symptom of an optimizer
       that silently returns x0 without ever actually searching.
    """
    state: dict[str, object] = {}

    def _optimizer(func, x0, args=(), disp=0):
        x0 = np.asarray(x0, dtype=float)
        xopt, _fopt, _niter, _ncalls, warnflag = fmin(
            func, x0, args=args, disp=0, full_output=True
        )
        state["warnflag"] = warnflag
        state["x0"] = x0
        state["xopt"] = np.asarray(xopt, dtype=float)
        return xopt

    try:
        params = dist.fit(x, optimizer=_optimizer)
    except Exception:
        return None

    if state.get("warnflag") != 0:
        return None
    x0, xopt = state["x0"], state["xopt"]
    if x0.shape == xopt.shape and np.allclose(x0, xopt, atol=1e-8, rtol=1e-8):
        return None
    if not np.all(np.isfinite(params)):
        return None
    return tuple(float(p) for p in params)


# --------------------------------------------------------------------------
# Per-family fit functions: window (1D numpy array, already NaN-dropped) ->
# parameter tuple, or None if the window is degenerate / unfit-able.
# --------------------------------------------------------------------------


def _fit_normal(x: np.ndarray) -> tuple[float, ...] | None:
    """Closed-form MLE: sample mean/std. Exact, no optimizer, so there is
    nothing to fail to converge - the only failure mode is a degenerate
    (zero-variance) window, which is nulled explicitly.
    """
    if len(x) < _MIN_OBS["normal"]:
        return None
    std = float(np.std(x, ddof=0))
    if not np.isfinite(std) or std <= 1e-12:
        return None
    return (float(np.mean(x)), std)


def _fit_t(x: np.ndarray) -> tuple[float, ...] | None:
    """Student-t: MLE via scipy (`t.fit`), because the degrees-of-freedom
    parameter (the whole point of using t over normal) has no simple
    closed-form estimator - method-of-moments on the first two moments alone
    can't identify it. Convergence is checked (see
    `_mle_fit_with_convergence_check`) since t.fit's simplex search is known
    to stall on flat/degenerate likelihoods.
    """
    if len(x) < _MIN_OBS["t"] or float(np.std(x)) <= 1e-12:
        return None
    params = _mle_fit_with_convergence_check(st.t, x)
    if params is None:
        return None
    df, loc, scale = params
    if not (df > 0 and scale > 0):
        return None
    return (df, loc, scale)


def _fit_skewt(x: np.ndarray) -> tuple[float, ...] | None:
    """Skewed-t (Jones & Faddy `jf_skew_t`): MLE via scipy, same rationale
    and same convergence check as Student-t - a 4-parameter shape+skew fit
    has no useful closed form.
    """
    if len(x) < _MIN_OBS["skewt"] or float(np.std(x)) <= 1e-12:
        return None
    params = _mle_fit_with_convergence_check(st.jf_skew_t, x)
    if params is None:
        return None
    a, b, loc, scale = params
    if not (a > 0 and b > 0 and scale > 0):
        return None
    return (a, b, loc, scale)


def _fit_poisson(x: np.ndarray) -> tuple[float, ...] | None:
    """Poisson: MLE and method-of-moments coincide (mean = rate) and are
    both just the sample mean - closed form, exact, nothing to converge.
    An all-zero-counts window is the count-data analogue of the frozen-price
    bug and is nulled rather than reported as a valid mu=0 fit.
    """
    if len(x) < _MIN_OBS["poisson"]:
        return None
    if np.all(x == 0):
        return None
    mu = float(np.mean(x))
    if not np.isfinite(mu) or mu <= 0:
        return None
    return (mu,)


def _fit_nbinom(x: np.ndarray) -> tuple[float, ...] | None:
    """Negative binomial: method-of-moments (mean/variance -> n, p) rather
    than MLE. NB's MLE requires numerically solving a digamma equation for
    the size parameter - too slow and, on the small windows a rolling fit
    uses, too unstable (easily diverges when a window is only mildly
    overdispersed) to run at every bar. MOM needs var > mean (overdispersion)
    to be well-defined at all; an equi- or under-dispersed window has no
    valid NB fit and is nulled rather than clipped to a degenerate n.
    """
    if len(x) < _MIN_OBS["nbinom"]:
        return None
    if np.all(x == 0):
        return None
    mean = float(np.mean(x))
    var = float(np.var(x, ddof=0))
    if mean <= 0 or var <= mean:
        return None
    p = mean / var
    n = mean * p / (1.0 - p)
    if not (np.isfinite(n) and np.isfinite(p)) or n <= 0 or not (0.0 < p < 1.0):
        return None
    return (n, p)


def _fit_beta(x: np.ndarray) -> tuple[float, ...] | None:
    """Beta: method-of-moments (mean/variance -> alpha, beta), not scipy's
    MLE `beta.fit`. MLE for Beta on small rolling windows is badly behaved
    right where crypto ratio data (taker-buy ratio, intrabar close position)
    actually lives - near the [0, 1] boundary - and scipy's general-purpose
    optimizer frequently fails to converge there. The MOM closed form is
    exact given the first two moments and has no optimizer to fail.
    """
    if len(x) < _MIN_OBS["beta"]:
        return None
    if np.any(x <= 0.0) or np.any(x >= 1.0):
        return None
    mean = float(np.mean(x))
    var = float(np.var(x, ddof=0))
    if var <= 1e-12:
        return None
    common = mean * (1.0 - mean) / var - 1.0
    if common <= 0 or not np.isfinite(common):
        return None
    a = mean * common
    b = (1.0 - mean) * common
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0:
        return None
    return (a, b)


_FIT_FUNCS = {
    "normal": _fit_normal,
    "t": _fit_t,
    "skewt": _fit_skewt,
    "poisson": _fit_poisson,
    "nbinom": _fit_nbinom,
    "beta": _fit_beta,
}


def frozen_dist(family: str, params: Sequence[float]):
    """Build a scipy frozen distribution from a fitted parameter tuple, in
    the same order fit_rolling / the _fit_* functions return them. Used by
    the scoring functions below and by callers who want to evaluate a fit
    directly (pdf/cdf/ppf/rvs) rather than just read the parameter columns.
    """
    if family == "normal":
        return st.norm(loc=params[0], scale=params[1])
    if family == "t":
        return st.t(df=params[0], loc=params[1], scale=params[2])
    if family == "skewt":
        return st.jf_skew_t(a=params[0], b=params[1], loc=params[2], scale=params[3])
    if family == "poisson":
        return st.poisson(mu=params[0])
    if family == "nbinom":
        return st.nbinom(n=params[0], p=params[1])
    if family == "beta":
        return st.beta(a=params[0], b=params[1])
    raise ValueError(f"unknown family: {family!r}")


# --------------------------------------------------------------------------
# fit_rolling
# --------------------------------------------------------------------------


@dataclass
class RollingFitResult:
    """`fit_rolling`'s return value: the fitted frame plus counts of every
    way a window can fail to produce a real fit, so a caller can inspect
    (and a test can assert on) how much of the series was dropped rather
    than that being silently absorbed into a column of NaNs.
    """

    frame: pl.DataFrame
    n_rows: int
    n_insufficient_history: int  # window not yet full (start-of-series edge)
    n_degenerate: int  # window was full but zero-variance / all-zero / etc.
    n_fit: int  # windows that produced a real, converged fit

    @property
    def n_dropped(self) -> int:
        return self.n_insufficient_history + self.n_degenerate


def fit_rolling(
    df: pl.DataFrame,
    col: str,
    family: str,
    window: int,
    min_periods: int | None = None,
) -> RollingFitResult:
    """Rolling causal distribution fit: for every row t, fit `family` to
    `col`'s trailing `window` bars ending at (and including) t, using only
    rows <= t. Adds one column per parameter, named
    f"{col}_{family}_{param_name}".

    Call this on a single symbol's frame, sorted by datetime (same
    convention as features.py's per-symbol builders) - group a multi-symbol
    panel with features.apply_per_symbol first so windows never cross a
    symbol boundary.

    min_periods (default: window) is the minimum number of trailing
    observations required before a fit is attempted at all; rows before
    that are null and counted under n_insufficient_history, not
    n_degenerate. A window that reaches full size but is degenerate
    (zero-variance, all-zero counts, an MLE that fails to converge) is
    still nulled, but counted separately under n_degenerate - see
    RollingFitResult.
    """
    if family not in _FIT_FUNCS:
        raise ValueError(f"unknown family: {family!r}. choose from {list(_FIT_FUNCS)}")

    values = df[col].to_numpy().astype(float)
    n = len(values)
    min_periods = window if min_periods is None else min_periods
    fit_fn = _FIT_FUNCS[family]
    param_names = FAMILY_PARAMS[family]

    out_cols = {name: np.full(n, np.nan) for name in param_names}
    n_insufficient_history = 0
    n_degenerate = 0
    n_fit = 0

    for t in range(n):
        start = max(0, t + 1 - window)
        window_vals = values[start : t + 1]
        window_vals = window_vals[np.isfinite(window_vals)]
        if len(window_vals) < min_periods:
            n_insufficient_history += 1
            continue
        params = fit_fn(window_vals)
        if params is None:
            n_degenerate += 1
            continue
        n_fit += 1
        for name, val in zip(param_names, params):
            out_cols[name][t] = val

    out = df.with_columns(
        [
            pl.Series(f"{col}_{family}_{name}", out_cols[name]).fill_nan(None)
            for name in param_names
        ]
    )
    return RollingFitResult(
        frame=out,
        n_rows=n,
        n_insufficient_history=n_insufficient_history,
        n_degenerate=n_degenerate,
        n_fit=n_fit,
    )


# --------------------------------------------------------------------------
# Scoring rules
# --------------------------------------------------------------------------

_DistOrList = "st.rv_frozen | Sequence[st.rv_frozen]"


def _as_dist_seq(dists, n: int) -> list:
    if isinstance(dists, (list, tuple)):
        if len(dists) != n:
            raise ValueError("dists sequence must be the same length as actual")
        return list(dists)
    return [dists] * n


def _is_discrete(dist) -> bool:
    return hasattr(dist, "pmf")


def log_score(dists, actual: np.ndarray) -> np.ndarray:
    """Out-of-sample log score: log density (continuous families) or log
    mass (discrete families) of each observed value under its fitted
    distribution. Higher is better - this is a proper scoring rule, so the
    true generating distribution has the highest expected score.

    `dists` is either a single frozen scipy distribution (broadcast to every
    observation) or a sequence of one frozen distribution per observation
    (the usual out-of-sample case: a different rolling fit at each t).
    """
    actual = np.asarray(actual, dtype=float)
    seq = _as_dist_seq(dists, len(actual))
    out = np.empty(len(actual))
    for i, (dist, y) in enumerate(zip(seq, actual)):
        with np.errstate(divide="ignore"):
            out[i] = dist.logpmf(y) if _is_discrete(dist) else dist.logpdf(y)
    return out


def crps(dists, actual: np.ndarray, n_points: int = 2000) -> np.ndarray:
    """Continuous Ranked Probability Score via direct numerical integration
    of the standard identity CRPS(F, y) = integral (F(x) - 1{x >= y})^2 dx.

    Works uniformly for every family here (continuous or discrete: a
    discrete distribution's CDF is just a step function, and integrating a
    fine grid across it approximates the sum-form CRPS closely enough for
    scoring-rule comparisons). Lower is better.
    """
    actual = np.asarray(actual, dtype=float)
    seq = _as_dist_seq(dists, len(actual))
    out = np.empty(len(actual))
    for i, (dist, y) in enumerate(zip(seq, actual)):
        lo, hi = dist.ppf(1e-6), dist.ppf(1 - 1e-6)
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            out[i] = np.nan
            continue
        # widen a touch so the observed value itself is inside the grid
        span = hi - lo
        lo, hi = lo - 0.01 * span, hi + 0.01 * span
        xs = np.linspace(lo, hi, n_points)
        cdf = dist.cdf(xs)
        indicator = (xs >= y).astype(float)
        out[i] = np.trapezoid((cdf - indicator) ** 2, xs)
    return out


def crps_normal_closed_form(
    actual: np.ndarray, loc: np.ndarray | float = 0.0, scale: np.ndarray | float = 1.0
) -> np.ndarray:
    """Closed-form CRPS for a normal predictive distribution (Gneiting & Raftery
    2007): CRPS(N(loc, scale^2), y) = scale * [z(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)],
    z = (y - loc) / scale. Exact - no integration grid, so unlike `crps` above it
    carries no resolution trade-off at all. `loc`/`scale` broadcast against
    `actual`, so a full per-bar variance path can be scored in one vectorized call.
    """
    actual = np.asarray(actual, dtype=float)
    z = (actual - loc) / scale
    return scale * (z * (2.0 * st.norm.cdf(z) - 1.0) + 2.0 * st.norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def crps_t_closed_form(
    actual: np.ndarray, df: np.ndarray | float, loc: np.ndarray | float = 0.0,
    scale: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Closed-form CRPS for a (non-standardized) Student-t predictive
    distribution, following the Jordan/Krueger/Lerch formula also implemented in
    R's `scoringRules::crps_t` (itself following Gneiting & Raftery 2007's
    supplement). For standard (loc=0, scale=1) t with nu>1 degrees of freedom and
    standard CDF/PDF F_nu, f_nu:

        CRPS(t_nu, z) = z*(2*F_nu(z)-1) + 2*f_nu(z)*(nu+z^2)/(nu-1)
                        - (2*sqrt(nu)/(nu-1)) * B(1/2, nu-1/2) / B(1/2, nu/2)^2

    (B is the Beta function.) A location-scale t's CRPS is `scale` times this,
    evaluated at the standardized z = (y-loc)/scale - exact, no integration grid.
    This is why it exists: `crps`'s numerical grid is built as
    linspace(ppf(1e-6), ppf(1-1e-6), n_points), which for a heavy-tailed t spans
    a vastly wider range than for a normal (t(2)'s grid is ~1400 units wide vs a
    normal's ~10), so cross-family CRPS comparisons using `crps` at a fixed
    n_points partly measure integration resolution, not forecast quality - see
    NEXT_RUN_PROMPT.md #1b / docs/06-scoring-rules-and-calibration.md#crps.
    Undefined (NaN) for nu<=1, the same boundary where a Student-t's own mean
    stops existing.
    """
    actual = np.asarray(actual, dtype=float)
    nu = np.asarray(df, dtype=float)
    z = (actual - loc) / scale
    Fz = st.t.cdf(z, df=nu)
    fz = st.t.pdf(z, df=nu)
    with np.errstate(invalid="ignore", divide="ignore"):
        const = (2.0 * np.sqrt(nu) / (nu - 1.0)) * (_betafn(0.5, nu - 0.5) / _betafn(0.5, nu / 2.0) ** 2)
        out = scale * (z * (2.0 * Fz - 1.0) + 2.0 * fz * (nu + z**2) / (nu - 1.0) - const)
    return np.where(nu > 1.0, out, np.nan)


def pit_values(dists, actual: np.ndarray) -> np.ndarray:
    """Probability integral transform: F(actual) under each observation's
    fitted distribution. If the fits are correct, these are Uniform(0, 1) -
    that's the only role Uniform plays anywhere in this module (a PIT null,
    never a candidate model)."""
    actual = np.asarray(actual, dtype=float)
    seq = _as_dist_seq(dists, len(actual))
    return np.array([dist.cdf(y) for dist, y in zip(seq, actual)])


def pit_ks_test(dists, actual: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov test of the PIT values against Uniform(0, 1).
    Returns (statistic, p-value); a small p-value rejects "this family's
    fits are well calibrated on this data".
    """
    u = pit_values(dists, actual)
    u = u[np.isfinite(u)]
    result = st.kstest(u, "uniform")
    return float(result.statistic), float(result.pvalue)


def qlike(actual_variance, predicted_variance) -> np.ndarray:
    """QLIKE loss for a variance forecast: actual/predicted - log(actual/predicted) - 1.

    Preferred over MSE for variance forecasts because it is a proper scoring
    rule for a noisy variance *proxy* (Patton 2011) - minimized in
    expectation by the true conditional variance even when the realized
    variance used as `actual` is itself a noisy estimate of it, which MSE is
    not robust to. Always >= 0, minimized (= 0) exactly when
    predicted == actual.
    """
    actual_variance = np.asarray(actual_variance, dtype=float)
    predicted_variance = np.asarray(predicted_variance, dtype=float)
    ratio = actual_variance / predicted_variance
    return ratio - np.log(ratio) - 1.0


# --------------------------------------------------------------------------
# Quantile / VaR coverage tests
# --------------------------------------------------------------------------


def exceedances(actual: np.ndarray, quantile_forecast: np.ndarray, side: str = "lower") -> np.ndarray:
    """Boolean exceedance indicator for VaR-style backtesting: True where
    the realized value breached the forecast quantile. side="lower" (the
    usual VaR case) flags actual < quantile_forecast; side="upper" flags
    actual > quantile_forecast.
    """
    actual = np.asarray(actual, dtype=float)
    quantile_forecast = np.asarray(quantile_forecast, dtype=float)
    if side == "lower":
        return actual < quantile_forecast
    if side == "upper":
        return actual > quantile_forecast
    raise ValueError("side must be 'lower' or 'upper'")

def kupiec_test(hits: np.ndarray, expected_rate: float) -> tuple[float, float]:
    """Kupiec (1995) unconditional coverage test: a likelihood-ratio test of
    whether the observed exceedance rate matches the rate a correctly
    calibrated quantile forecast should produce.

    H0: true exceedance probability == expected_rate.
    LR = -2 * [ (x*log(p) + (n-x)*log(1-p)) - (x*log(x/n) + (n-x)*log(1-x/n)) ]
    ~ chi2(1) under H0, using the xlogy convention 0*log(0) = 0 so a zero
    exceedance count doesn't blow up.

    Returns (LR statistic, p-value).
    """
    hits = np.asarray(hits, dtype=bool)
    n = len(hits)
    x = int(hits.sum())
    p = expected_rate
    pihat = x / n
    ll_null = xlogy(x, p) + xlogy(n - x, 1.0 - p)
    ll_alt = xlogy(x, pihat) + xlogy(n - x, 1.0 - pihat)
    lr = -2.0 * (ll_null - ll_alt)
    pvalue = float(st.chi2.sf(lr, df=1))
    return float(lr), pvalue


def christoffersen_independence_test(hits: np.ndarray) -> tuple[float, float]:
    """Christoffersen (1998) independence test: a likelihood-ratio test of
    whether exceedances are serially independent (a first-order 2-state
    Markov chain with a single exceedance probability) against the
    alternative that the transition probability P(hit_t | hit_{t-1}) differs
    from P(hit_t | no hit_{t-1}) - i.e. exceedances cluster.

    Complements kupiec_test: a model can have exactly the right unconditional
    exceedance rate while still failing badly (e.g. all its breaches
    clustered in one crash) - that failure only shows up here.

    Returns (LR statistic, p-value), LR ~ chi2(1) under H0 (independence).
    """
    hits = np.asarray(hits, dtype=int)
    prev, cur = hits[:-1], hits[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    n0, n1 = n00 + n01, n10 + n11
    pi01 = n01 / n0 if n0 > 0 else 0.0
    pi11 = n11 / n1 if n1 > 0 else 0.0
    pi = (n01 + n11) / (n0 + n1) if (n0 + n1) > 0 else 0.0

    ll_null = xlogy(n01 + n11, pi) + xlogy(n00 + n10, 1.0 - pi)
    ll_alt = (
        xlogy(n01, pi01)
        + xlogy(n00, 1.0 - pi01)
        + xlogy(n11, pi11)
        + xlogy(n10, 1.0 - pi11)
    )
    lr = -2.0 * (ll_null - ll_alt)
    pvalue = float(st.chi2.sf(lr, df=1))
    return float(lr), pvalue


def christoffersen_conditional_coverage_test(
    hits: np.ndarray, expected_rate: float
) -> tuple[float, float]:
    """Combined conditional-coverage test: Kupiec's unconditional-coverage
    LR plus Christoffersen's independence LR, jointly chi2(2) under the null
    that the forecast is both correctly sized and its breaches independent.
    """
    lr_uc, _ = kupiec_test(hits, expected_rate)
    lr_ind, _ = christoffersen_independence_test(hits)
    lr_cc = lr_uc + lr_ind
    pvalue = float(st.chi2.sf(lr_cc, df=2))
    return lr_cc, pvalue
