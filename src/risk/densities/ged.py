"""Generalized Error Distribution (GED / generalized normal) innovation
family for notebook 6's Phase 3 distribution zoo (NEXT_RUN_PROMPT.md).

Standardized to unit variance throughout: `scipy.stats.gennorm(beta=kappa,
scale=1)` has Var = Gamma(3/kappa)/Gamma(1/kappa) at scale=1, which is NOT 1
in general, so every call here rescales by
`s(kappa) = sqrt(Gamma(1/kappa) / Gamma(3/kappa))` to force unit variance
before use - the same discipline `_garch_negloglik` applies to the Student-t
scale (`sqrt(nu/(nu-2))`).

kappa=2 nests the normal; kappa=1 nests the Laplace. kappa<2 is
sharper-peaked/heavier-shouldered than normal but every moment is still
finite (unlike Student-t) - that's the whole point of trying this family.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as st
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import gamma as gammafn

NAME = "ged"
N_SHAPE = 1  # shape param: kappa (kappa=2 -> normal, kappa=1 -> Laplace)

_KAPPA_BOUNDS = (0.3, 12.0)
_BOUND_EPS = 1e-4  # how close to a bound counts as "pinned" (optimizer artifact)
_MIN_N = 30


def _unit_scale(kappa: float) -> float:
    """s(kappa) such that gennorm(beta=kappa, scale=s(kappa)) has Var == 1."""
    return np.sqrt(gammafn(1.0 / kappa) / gammafn(3.0 / kappa))


def logpdf(z: np.ndarray, shape: tuple[float, ...]) -> np.ndarray:
    (kappa,) = shape
    s = _unit_scale(kappa)
    return st.gennorm.logpdf(z, beta=kappa, scale=s)


def ppf(q: float | np.ndarray, shape: tuple[float, ...]) -> np.ndarray:
    (kappa,) = shape
    s = _unit_scale(kappa)
    return np.asarray(st.gennorm.ppf(q, beta=kappa, scale=s))


def fit(z: np.ndarray) -> tuple[float, ...] | None:
    """MLE of kappa on already-standardized residuals z. None on
    non-convergence, insufficient data, or a fit pinned exactly at a bound
    (an optimizer artifact per this repo's convention, not a real estimate).
    """
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < _MIN_N:
        return None
    if np.var(z) < 1e-12:
        return None

    def negloglik(kappa: float) -> float:
        s = _unit_scale(kappa)
        ll = st.gennorm.logpdf(z, beta=kappa, scale=s)
        if not np.all(np.isfinite(ll)):
            return np.inf
        return -np.sum(ll)

    res = minimize_scalar(
        negloglik,
        bounds=_KAPPA_BOUNDS,
        method="bounded",
        options={"xatol": 1e-6},
    )
    if not res.success:
        return None
    kappa = float(res.x)
    lo, hi = _KAPPA_BOUNDS
    if kappa <= lo + _BOUND_EPS or kappa >= hi - _BOUND_EPS:
        return None
    return (kappa,)


def es(q: float, shape: tuple[float, ...]) -> float:
    """Expected shortfall below the q-quantile: (1/q) * integral_0^q ppf(u) du,
    via probability-space numerical integration of ppf over the bounded
    interval [0, q] (not a value-space linspace grid over
    [ppf(eps), ppf(q)] - that was notebook 5's documented CRPS bug).
    """
    (kappa,) = shape
    s = _unit_scale(kappa)

    def integrand(u: float) -> float:
        return st.gennorm.ppf(u, beta=kappa, scale=s)

    val, _ = quad(integrand, 0.0, q, limit=200)
    return val / q
