"""Johnson SU innovation density, standardized to unit variance.

Johnson SU is a four-parameter family built from a shifted-and-scaled
hyperbolic-sine transform of a standard normal: if ``Z ~ N(0,1)`` then
``X = xi + lambda * sinh((Z - gamma) / delta)`` is Johnson SU with shape
params ``(gamma, delta)``, location ``xi`` and scale ``lambda``. It supports
skew (``gamma != 0``) and heavier-than-normal tails, and nests something
close to normal as ``delta -> infinity`` with ``gamma = 0``. `scipy.stats`
already implements the density (as ``johnsonsu(a, b, loc, scale)`` with
``a = gamma``, ``b = delta``) and its exact ppf, so this module is mostly a
standardization wrapper around scipy plus an MLE fit routine, matching the
fixed Phase-3 interface (`NAME`, `N_SHAPE`, `fit`, `logpdf`, `ppf`, `es`)
declared in `NEXT_RUN_PROMPT.md` section 4 for the wider density zoo.

Standardization: scipy's "standard" Johnson SU at given shape (a, b) with
loc=0, scale=1 has mean mu0 and variance sigma0^2 that are generally *not*
0 and 1 (unlike, say, the Student-t, whose only rescaling is the classic
sqrt(nu/(nu-2)) constant). We solve for the loc/scale that undoes this:
loc = -mu0/sigma0, scale = 1/sigma0, so that the resulting distribution has
mean 0 and variance 1 for any shape. This is verified numerically in this
repo's test suite (`tests/test_dist_lib6_johnsonsu.py`), not just asserted.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize
from scipy import stats as st

NAME = "johnsonsu"
N_SHAPE = 2  # shape params: (gamma, delta) matching scipy.stats.johnsonsu's (a, b)

_GAMMA_BOUNDS = (-10.0, 10.0)
_DELTA_BOUNDS = (0.1, 20.0)
_MIN_N = 30


def _loc_scale(gamma: float, delta: float) -> tuple[float, float]:
    """loc/scale that standardize scipy's johnsonsu(gamma, delta) to mean 0, var 1."""
    mu0, var0 = st.johnsonsu(gamma, delta).stats(moments="mv")
    sigma0 = np.sqrt(var0)
    return float(-mu0 / sigma0), float(1.0 / sigma0)


def logpdf(z: np.ndarray, shape: tuple[float, ...]) -> np.ndarray:
    gamma, delta = shape
    loc, scale = _loc_scale(gamma, delta)
    return st.johnsonsu.logpdf(z, gamma, delta, loc=loc, scale=scale)


def ppf(q: float | np.ndarray, shape: tuple[float, ...]) -> np.ndarray:
    gamma, delta = shape
    loc, scale = _loc_scale(gamma, delta)
    return np.asarray(st.johnsonsu.ppf(q, gamma, delta, loc=loc, scale=scale))


def es(q: float, shape: tuple[float, ...]) -> float:
    """Expected shortfall below the q-quantile, via probability-space integration.

    ES_q = (1/q) * integral_0^q ppf(u) du. `ppf` is closed-form (scipy), so we
    integrate directly in probability space with `scipy.integrate.quad` rather
    than building a value-space linspace grid over z, which breaks down on
    heavy-tailed distributions where the tail is not well represented by a
    finite z-grid (see `distributions.crps`'s old numerical-grid bug in this
    repo's history for exactly this failure mode).
    """
    _gamma, _delta = shape

    def integrand(u):
        return ppf(u, shape)

    value, _ = integrate.quad(integrand, 0.0, q, limit=200)
    return float(value / q)


def _negloglik(params: np.ndarray, z: np.ndarray) -> float:
    gamma, delta = params
    ll = logpdf(z, (gamma, delta))
    if not np.all(np.isfinite(ll)):
        return np.inf
    return -np.sum(ll)


def fit(z: np.ndarray) -> tuple[float, ...] | None:
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < _MIN_N:
        return None

    # a few starting points: symmetric-ish and mildly skewed, to reduce the
    # chance of getting stuck in a bad local optimum.
    starts = [(0.0, 2.0), (0.5, 3.0), (-0.5, 3.0), (0.0, 5.0)]

    best = None
    for g0, d0 in starts:
        res = optimize.minimize(
            _negloglik,
            x0=np.array([g0, d0]),
            args=(z,),
            method="L-BFGS-B",
            bounds=[_GAMMA_BOUNDS, _DELTA_BOUNDS],
        )
        if not res.success or not np.isfinite(res.fun):
            continue
        if best is None or res.fun < best.fun:
            best = res

    if best is None:
        return None

    gamma, delta = best.x

    # reject a fit pinned exactly at a bound - that means the optimizer ran
    # off the edge of the feasible region rather than converging to an
    # interior optimum.
    eps = 1e-6
    if (
        abs(gamma - _GAMMA_BOUNDS[0]) < eps
        or abs(gamma - _GAMMA_BOUNDS[1]) < eps
        or abs(delta - _DELTA_BOUNDS[0]) < eps
        or abs(delta - _DELTA_BOUNDS[1]) < eps
    ):
        return None

    return float(gamma), float(delta)
