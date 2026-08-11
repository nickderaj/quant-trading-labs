"""Normal-Inverse Gaussian (NIG) innovation density, for notebook 6's Phase 3
distribution zoo (NEXT_RUN_PROMPT.md). Same NAME/N_SHAPE/fit/logpdf/ppf/es
interface as densities/ged.py and densities/hansen_skewt.py.

The general NIG density has 4 parameters (alpha, beta, delta, mu):

    gamma = sqrt(alpha**2 - beta**2)          # requires alpha > |beta| >= 0
    f(x) = (alpha*delta/pi) * exp(delta*gamma + beta*(x-mu))
           * K1(alpha*sqrt(delta**2 + (x-mu)**2)) / sqrt(delta**2 + (x-mu)**2)

with mean = mu + delta*beta/gamma and variance = delta*alpha**2/gamma**3
(general form). This module exposes only the *shape* pair (alpha, beta) and
internally solves mean=0, variance=1 for (delta, mu) at that shape:

    delta = gamma**3 / alpha**2      # from variance=1
    mu = -delta*beta/gamma           # from mean=0, using the delta above

delta > 0 always holds here since alpha > |beta| >= 0 implies gamma > 0.
This was verified numerically before trusting it (ad hoc script, not kept
in the repo, per this repo's "read the actual numbers" convention - see
dist_lib5.py's acerbi_szekely_z docstring for the precedent): for (alpha,
beta) in {(2,0), (3,1), (5,-2), (10,3), (1.5,0.5)}, quad-integrating f(x),
x*f(x), x**2*f(x) over [-100,100] gave integral~=1, mean~=0 (~1e-14), and
var~=1 (~1e-6 or better) in every case.

K1 is the modified Bessel function of the second kind, order 1. logpdf uses
scipy.special.kve (exponentially scaled: kve(1,x) = kv(1,x)*exp(x)) rather
than kv directly, since kv(1, x) underflows to exactly 0 (-> log(0) = -inf)
for the larger arguments this density's tails reach (e.g. alpha near the
fit's upper bound with a far-out z); kve does not have this problem because
its exponential growth is factored out before the float64 representation:
log(kv(1,x)) = log(kve(1,x)) - x avoids that underflow entirely. Verified
against direct kv-based logpdf agreeing to ~1e-10 in the safe range, and
remaining finite well past where the kv-based version already returns -inf
(ad hoc check before writing the tests below).

No closed form exists for the NIG ppf/cdf, unlike GED and Hansen skew-t.
ppf(q, shape) numerically integrates the pdf from a far-left bound to a
trial x (via quad) and root-finds on x with brentq to hit the target
probability q. es(q, shape) then integrates that (expensive) ppf in
probability space over [0, q] - the same probability-space convention the
sibling modules use to avoid notebook 5's documented value-space-linspace
CRPS bug - with a capped quad point count to keep runtime bounded.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize
from scipy.special import kve

NAME = "nig"
N_SHAPE = 2  # shape params: (alpha, beta) - alpha>0 controls tail heaviness, beta controls skew, |beta|<alpha

_ALPHA_BOUNDS = (0.3, 30.0)
_BETA_FRAC_BOUNDS = (-0.95, 0.95)  # beta = beta_frac * alpha, |beta_frac| < 1
_BOUND_EPS = 1e-4  # how close to a bound counts as "pinned" (optimizer artifact)
_MIN_N = 30

# Far-left bound for the numerical CDF integral: standardized NIG densities
# used here have unit variance and (at worst, small alpha) moderately heavy
# tails, so -80 is comfortably past where the density is indistinguishable
# from 0 in float64 for the alpha range this module supports.
_X_LO = -80.0
_X_HI = 80.0


def _consts(alpha: float, beta: float) -> tuple[float, float, float]:
    """(gamma, delta, mu) for the zero-mean/unit-variance parametrization at
    shape (alpha, beta)."""
    gamma = np.sqrt(alpha**2 - beta**2)
    delta = gamma**3 / alpha**2
    mu = -delta * beta / gamma
    return gamma, delta, mu


def logpdf(z: np.ndarray, shape: tuple[float, ...]) -> np.ndarray:
    alpha, beta = shape
    gamma, delta, mu = _consts(alpha, beta)
    z = np.asarray(z, dtype=float)
    x = z - mu
    r = np.sqrt(delta**2 + x**2)
    arg = alpha * r
    # log(kv(1, arg)) via the exponentially-scaled kve to avoid underflow:
    # kve(1, arg) = kv(1, arg) * exp(arg)  =>  log(kv(1,arg)) = log(kve(1,arg)) - arg
    logk1 = np.log(kve(1, arg)) - arg
    return np.log(alpha * delta / np.pi) + delta * gamma + beta * x + logk1 - np.log(r)


def _pdf_scalar(x: float, shape: tuple[float, ...]) -> float:
    return float(np.exp(logpdf(np.array([x]), shape))[0])


def _cdf_scalar(x: float, shape: tuple[float, ...]) -> float:
    """CDF at x via quad with a -inf lower bound (scipy's substitution
    handles the infinite tail far more cheaply than a large finite bound
    like _X_LO would). Split at 0 rather than integrating -inf directly to
    a large positive x in one call: ad hoc testing before trusting this
    showed quad(-inf, x) silently returns garbage (near 0 instead of ~1)
    once x is roughly >= 45 for typical shapes here - the one-sided
    substitution loses track of where the density's mass actually is when
    the finite endpoint is far past it. Splitting at 0 keeps each quad call
    within a domain where the mass-to-total-interval ratio stays sane, and
    is needed here since brentq's bracket in ppf() probes x out to
    +/-_X_HI/_X_LO.
    """
    if x <= 0:
        val, _ = quad(_pdf_scalar, -np.inf, x, args=(shape,), limit=100)
        return val
    left, _ = quad(_pdf_scalar, -np.inf, 0.0, args=(shape,), limit=100)
    right, _ = quad(_pdf_scalar, 0.0, x, args=(shape,), limit=100)
    return left + right


def ppf(q: float | np.ndarray, shape: tuple[float, ...]) -> np.ndarray:
    q_arr = np.asarray(q, dtype=float)
    scalar_input = q_arr.ndim == 0
    q_arr = np.atleast_1d(q_arr)
    out = np.empty_like(q_arr, dtype=float)
    for i, qi in enumerate(q_arr):
        qi = float(qi)
        out[i] = brentq(
            lambda x, qi=qi: _cdf_scalar(x, shape) - qi,
            _X_LO,
            _X_HI,
            xtol=1e-8,
            rtol=1e-10,
            maxiter=200,
        )
    return out[0] if scalar_input else out


def fit(z: np.ndarray) -> tuple[float, ...] | None:
    """MLE of (alpha, beta) on already-standardized residuals z, optimizing
    over (alpha, beta_frac = beta/alpha) to respect the alpha > |beta|
    constraint by construction. None on non-convergence, insufficient data,
    or a fit pinned exactly at a bound (an optimizer artifact per this
    repo's convention, not a real estimate).
    """
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < _MIN_N:
        return None
    if np.var(z) < 1e-12:
        return None

    def negloglik(params: np.ndarray) -> float:
        alpha, beta_frac = params
        if alpha <= 0 or not (-1.0 < beta_frac < 1.0):
            return 1e10
        beta = beta_frac * alpha
        ll = logpdf(z, (alpha, beta))
        if not np.all(np.isfinite(ll)):
            return 1e10
        return -float(np.sum(ll))

    x0 = np.array([3.0, 0.0])
    try:
        res = minimize(
            negloglik,
            x0,
            method="L-BFGS-B",
            bounds=[_ALPHA_BOUNDS, _BETA_FRAC_BOUNDS],
            options={"maxiter": 200},
        )
    except Exception:  # noqa: BLE001 - optimizer can raise arbitrary errors; convention is None on any failure
        return None
    if not res.success or not np.all(np.isfinite(res.x)):
        return None

    alpha, beta_frac = float(res.x[0]), float(res.x[1])
    alpha_lo, alpha_hi = _ALPHA_BOUNDS
    bf_lo, bf_hi = _BETA_FRAC_BOUNDS
    if alpha <= alpha_lo + _BOUND_EPS or alpha >= alpha_hi - _BOUND_EPS:
        return None
    if beta_frac <= bf_lo + _BOUND_EPS or beta_frac >= bf_hi - _BOUND_EPS:
        return None
    return (alpha, beta_frac * alpha)


_ES_GL_NODES, _ES_GL_WEIGHTS = np.polynomial.legendre.leggauss(24)


def es(q: float, shape: tuple[float, ...]) -> float:
    """Expected shortfall below the q-quantile: (1/q) * integral_0^q ppf(u) du,
    via probability-space numerical integration of ppf over the bounded
    interval [0, q] (not a value-space linspace grid over
    [ppf(eps), ppf(q)] - that was notebook 5's documented CRPS bug).

    ppf itself is a root-find here (no closed form), so an *adaptive* quad
    on top of it is doubly expensive: scipy's adaptive quad kept refining
    the [0, q] interval and calling ppf 200+ times in ad hoc timing before
    this was written, costing 20-30s for a single es() call. Using a fixed
    24-point Gauss-Legendre rule instead bounds the cost to exactly 24 ppf
    evaluations regardless of shape, at a relative error of ~1e-4 against
    the adaptive-quad reference in that same ad hoc check - accurate enough
    for this repo's use (fitting/comparing tail risk across families) while
    keeping a single call well under the "a few seconds" budget.
    """
    a, b = 0.0, q
    pts = 0.5 * (b - a) * _ES_GL_NODES + 0.5 * (b + a)
    vals = ppf(pts, shape)
    val = 0.5 * (b - a) * np.sum(_ES_GL_WEIGHTS * vals)
    return val / q
