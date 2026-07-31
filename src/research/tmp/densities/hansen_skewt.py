"""Hansen's (1994) skewed Student-t innovation density, for notebook 6's
Phase 3 distribution zoo (NEXT_RUN_PROMPT.md). Same NAME/N_SHAPE/fit/logpdf/
ppf/es interface as densities/ged.py.

Unlike GED (which needs an explicit unit-variance rescale of scipy's
gennorm), Hansen's construction bakes zero-mean/unit-variance directly into
the density's own parameterization via the (a, b) shift-and-scale constants
below - no separate standardization step is needed here. That claim is not
taken on faith: it is verified numerically in this repo's tests
(tests/test_dist_lib6_hansen_skewt.py) by numerically integrating
z*f(z) and z^2*f(z) to confirm mean~=0, var~=1 for several (nu, lam) pairs,
per this repo's "read the actual numbers" convention (see dist_lib5.py's
acerbi_szekely_z docstring for the precedent of a prior sign-error catch on
exactly this kind of formula).

Density (nu = degrees of freedom > 2, lam = skewness in (-1, 1)):

    c = Gamma((nu+1)/2) / (sqrt(pi*(nu-2)) * Gamma(nu/2))
    a = 4 * lam * c * (nu-2) / (nu-1)
    b = sqrt(1 + 3*lam**2 - a**2)

    f(z) = b*c * (1 + (1/(nu-2)) * ((b*z+a) / (1 -+ lam))**2) ** (-(nu+1)/2)

using (1-lam) below the threshold -a/b and (1+lam) at/above it. lam=0
collapses a=0, b=1, reducing exactly to a standardized Student-t (verified
below to match st.t.logpdf(z * sqrt(nu/(nu-2)), df=nu) + log(sqrt(nu/(nu-2)))
- the same standardization dist_lib5.py's GJR-t branch already uses - to
machine precision, max abs diff ~1e-15 in an ad hoc numerical check run
before writing this module).

Quantile function (piecewise via the standard Student-t ppf), split at the
CDF value of the density's own threshold, q0 = (1-lam)/2:

    q < q0:  ppf(q) = (1/b) * ((1-lam) * sqrt((nu-2)/nu) * t.ppf(q/(1-lam), df=nu) - a)
    q >= q0: ppf(q) = (1/b) * ((1+lam) * sqrt((nu-2)/nu) * t.ppf((q-q0)/(1+lam) + 0.5, df=nu) - a)

Both the pdf and the ppf formulas above were numerically verified before
trusting them (ad hoc script, not kept in the repo): pdf integrates to 1 and
gives mean~=0/var~=1 for (nu,lam) in {(5,.3),(8,-.3),(10,0),(4.5,.5)} to
~1e-4 or better (quad's own numerical floor on a heavy, skewed tail); ppf
round-trips through quad(pdf, -inf, ppf(q)) to q at values spanning both
sides of (1-lam)/2 to ~1e-9; and the lam=0 nesting check above passed to
~1e-15. No sign error was found this time (unlike the acerbi_szekely_z
precedent) - the formulas as commonly stated for this distribution are
correct as given.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as st
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import gammaln

NAME = "hansen_skewt"
N_SHAPE = 2  # shape params: (nu, lam) - nu = degrees of freedom (>2), lam = skewness in (-1,1)

_NU_BOUNDS = (2.1, 60.0)
_LAM_BOUNDS = (-0.98, 0.98)
_BOUND_EPS = 1e-4  # how close to a bound counts as "pinned" (optimizer artifact)
_MIN_N = 30


def _consts(nu: float, lam: float) -> tuple[float, float, float]:
    """(c, a, b) per Hansen (1994); c computed via gammaln to avoid overflow
    at large nu."""
    c = np.exp(gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0)) / np.sqrt(np.pi * (nu - 2.0))
    a = 4.0 * lam * c * (nu - 2.0) / (nu - 1.0)
    b = np.sqrt(1.0 + 3.0 * lam**2 - a**2)
    return c, a, b


def logpdf(z: np.ndarray, shape: tuple[float, ...]) -> np.ndarray:
    nu, lam = shape
    c, a, b = _consts(nu, lam)
    z = np.asarray(z, dtype=float)
    thresh = -a / b
    denom = np.where(z < thresh, 1.0 - lam, 1.0 + lam)
    inner = 1.0 + (1.0 / (nu - 2.0)) * ((b * z + a) / denom) ** 2
    return np.log(b) + np.log(c) - (nu + 1.0) / 2.0 * np.log(inner)


def ppf(q: float | np.ndarray, shape: tuple[float, ...]) -> np.ndarray:
    nu, lam = shape
    c, a, b = _consts(nu, lam)
    q = np.asarray(q, dtype=float)
    scalar_input = q.ndim == 0
    q = np.atleast_1d(q)
    q0 = (1.0 - lam) / 2.0
    cst = np.sqrt((nu - 2.0) / nu)
    out = np.empty_like(q, dtype=float)
    left = q < q0
    right = ~left
    out[left] = (1.0 / b) * ((1.0 - lam) * cst * st.t.ppf(q[left] / (1.0 - lam), df=nu) - a)
    out[right] = (1.0 / b) * (
        (1.0 + lam) * cst * st.t.ppf((q[right] - q0) / (1.0 + lam) + 0.5, df=nu) - a
    )
    return out[0] if scalar_input else out


def fit(z: np.ndarray) -> tuple[float, ...] | None:
    """MLE of (nu, lam) on already-standardized residuals z. None on
    non-convergence, insufficient data, or a fit pinned exactly at a bound
    (an optimizer artifact per this repo's convention, not a real estimate).
    """
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < _MIN_N:
        return None
    if np.var(z) < 1e-12:
        return None

    def negloglik(params: np.ndarray) -> float:
        nu, lam = params
        if nu <= 2.0 or not (-1.0 < lam < 1.0):
            return 1e10
        ll = logpdf(z, (nu, lam))
        if not np.all(np.isfinite(ll)):
            return 1e10
        return -float(np.sum(ll))

    x0 = np.array([8.0, 0.0])
    try:
        res = minimize(
            negloglik, x0, method="L-BFGS-B",
            bounds=[_NU_BOUNDS, _LAM_BOUNDS],
            options={"maxiter": 200},
        )
    except Exception:
        return None
    if not res.success or not np.all(np.isfinite(res.x)):
        return None

    nu, lam = float(res.x[0]), float(res.x[1])
    nu_lo, nu_hi = _NU_BOUNDS
    lam_lo, lam_hi = _LAM_BOUNDS
    if nu <= nu_lo + _BOUND_EPS or nu >= nu_hi - _BOUND_EPS:
        return None
    if lam <= lam_lo + _BOUND_EPS or lam >= lam_hi - _BOUND_EPS:
        return None
    return (nu, lam)


def es(q: float, shape: tuple[float, ...]) -> float:
    """Expected shortfall below the q-quantile: (1/q) * integral_0^q ppf(u) du,
    via probability-space numerical integration of ppf over the bounded
    interval [0, q] (not a value-space linspace grid over
    [ppf(eps), ppf(q)] - that was notebook 5's documented CRPS bug).
    """
    nu, lam = shape

    def integrand(u: float) -> float:
        return float(ppf(u, (nu, lam)))

    val, _ = quad(integrand, 0.0, q, limit=200)
    return val / q
