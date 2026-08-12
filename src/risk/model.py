"""The per-product risk model: `RiskModel`, `ewma_vol`, `fit_risk_model`, and
the numerical PIT/quantile machinery `RiskModel` needs for its
non-closed-form density families.

Ported verbatim from `src/research/tmp/commod_lib8.py` (see
`docs/10-risk-engine.md`). `commod_lib8.py` re-imports these names and
re-exports them so notebook 008 and its tests keep passing unchanged against
the promoted code (NEXT_PROMPT.md sec 3.3).

`numerical_cdf_grid`/`numerical_pit`/`numerical_ppf` are not listed in
NEXT_PROMPT.md's move table (they sit in commod_lib8.py's Phase 2 section),
but `RiskModel.ppf_from_u`/`cdf_from_x` call them directly, so leaving them
behind would make `src/risk/` depend on `commod_lib8.py` -- exactly the
reverse dependency NEXT_PROMPT.md sec 12 forbids ("`risk/` importing
anything from `commod_lib8.py` is not [fine]"). They move here as an implied
consequence of that rule, verbatim, with the same re-export shim treatment
(`run_phase_2_density_selection.py`'s `C.numerical_pit` keeps working).

**Naming note** (NEXT_PROMPT.md sec 1): 008's prose calls the winning models
`garch_ged`, `gjr_ged`, etc, because Phase 3 ranked *GARCH-family* models.
`fit_risk_model` here strips the variance-process prefix and fits only the
**density family** unconditionally on the full return series; time-variation
is supplied separately by the caller via `ewma_vol` + `RiskModel.var_conditional`.
A `family_map` value of `"ged"` therefore does NOT mean "the GARCH-GED model was
promoted" -- it means "the GED innovation density was promoted, fit
unconditionally, and conditioned at call time by a caller-supplied EWMA scale."
Reading `family_map` as "these are GARCH models" misdescribes what this engine
does.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from scipy import stats as st

import distributions as dist
from risk import densities

__all__ = [
    "RiskModel",
    "StressResult",
    "ewma_vol",
    "fit_risk_model",
    "numerical_cdf_grid",
    "numerical_pit",
    "numerical_ppf",
]


class StressResult(TypedDict):
    """`RiskModel.stress`'s return shape (NEXT_PROMPT.md sec 3.5's TypedDict
    convention, applied here the same way it was to `PortfolioRisk`)."""

    cum_return: float | None
    worst_day: float | None
    n_days: int


def numerical_cdf_grid(
    logpdf_fn: object, grid_lo: float = -20.0, grid_hi: float = 20.0, n_grid: int = 4001
) -> dict[str, np.ndarray]:
    """Build a numerical CDF over `grid_lo..grid_hi` from a logpdf callable
    (z: np.ndarray) -> np.ndarray, via cumulative trapezoidal integration.
    Returns {"grid": z-grid, "cdf": F(z)}; F is normalised to end at 1.0 so a
    logpdf that isn't perfectly normalised over the truncated grid still
    yields a valid PIT.
    """
    grid = np.linspace(grid_lo, grid_hi, n_grid)
    pdf = np.exp(logpdf_fn(grid))  # type: ignore[operator]
    pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
    cdf = np.concatenate([[0.0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * np.diff(grid))])
    total = cdf[-1]
    if total > 0:
        cdf = cdf / total
    return {"grid": grid, "cdf": cdf}


def numerical_pit(
    logpdf_fn: object,
    z_values: np.ndarray,
    grid_lo: float = -20.0,
    grid_hi: float = 20.0,
) -> np.ndarray:
    """PIT values for `z_values` under a shape-only family's logpdf, via
    `numerical_cdf_grid` + linear interpolation."""
    g = numerical_cdf_grid(logpdf_fn, grid_lo, grid_hi)
    z = np.clip(z_values, grid_lo, grid_hi)
    return np.interp(z, g["grid"], g["cdf"])


def numerical_ppf(
    logpdf_fn: object, u: np.ndarray, grid_lo: float = -20.0, grid_hi: float = 20.0
) -> np.ndarray:
    """Inverse-CDF (quantile function) for `u` under a shape-only family's
    logpdf, via one `numerical_cdf_grid` build + interpolation -- the
    `numerical_pit` inverse. Exists because nig.ppf has no closed form and
    root-finds its CDF per point (~50ms/point measured); a 20,000-point
    Monte Carlo draw for portfolio simulation would take ~15+ minutes per
    asset called that way. This is O(n_grid) once, then O(len(u)), regardless
    of how many draws are needed.
    """
    g = numerical_cdf_grid(logpdf_fn, grid_lo, grid_hi)
    # cdf must be strictly increasing for interp's x-array; de-duplicate flat
    # (near-zero-density) regions by keeping only strictly increasing points.
    cdf, grid = g["cdf"], g["grid"]
    keep = np.concatenate([[True], np.diff(cdf) > 0])
    return np.interp(np.clip(u, cdf[keep][0], cdf[keep][-1]), cdf[keep], grid[keep])


class RiskModel:
    """A fitted per-product density, with VaR/ES/simulate/stress. `kind` is
    "loc_scale" (normal/t: params already on the return scale) or
    "standardized" (ged/nig/johnsonsu/hansen_skewt/spliced_evt: shape fit on
    (r-mean)/std, so VaR/ES need the mean/std Jacobian applied explicitly).

    Certified by `src/results/008_commodity_tails_and_risk.md` Phase 7/8
    (Gate RE: 15/16 development, 14/16 holdout 1% VaR coverage).
    """

    def __init__(
        self,
        product: str,
        family: str,
        kind: str,
        mean: float,
        std: float,
        params: tuple[float, ...],
    ):
        self.product = product
        self.family = family
        self.kind = kind
        self.mean = mean
        self.std = std
        self.params = params

    def _lower_q(self, alpha: float) -> float:
        if self.kind == "loc_scale":
            d = dist.frozen_dist(self.family, self.params)
            return float(d.ppf(alpha))
        mod = densities.REGISTRY[self.family]
        return float(self.mean + self.std * mod.ppf(alpha, self.params))

    def _lower_es(self, alpha: float) -> float:
        if self.family == "normal":
            z = st.norm.ppf(alpha)
            return float(self.mean - self.std * st.norm.pdf(z) / alpha)
        if self.family == "t":
            df = self.params[0]
            z = st.t.ppf(alpha, df=df)
            es_z = -st.t.pdf(z, df=df) * (df + z**2) / (df - 1) / alpha
            return float(self.mean + self.std * es_z)
        mod = densities.REGISTRY[self.family]
        return float(self.mean + self.std * mod.es(alpha, self.params))

    def var(self, alpha: float, horizon: int = 1) -> float:
        """Value at Risk: the alpha-quantile loss, positive horizon scaled by
        sqrt(horizon) (the standard, and limited, iid scaling -- documented
        here rather than silently assumed)."""
        q = self._lower_q(alpha)
        return float(-q * np.sqrt(horizon))

    def es(self, alpha: float, horizon: int = 1) -> float:
        """Expected Shortfall: the average loss beyond VaR at level alpha."""
        e = self._lower_es(alpha)
        return float(-e * np.sqrt(horizon))

    def var_conditional(self, alpha: float, sigma_t: float, horizon: int = 1) -> float:
        """VaR using a caller-supplied *current* volatility `sigma_t`
        (e.g. from `ewma_vol`) instead of the model's static full-sample std
        -- the shape (skew/kurtosis) stays as fitted, only the scale is
        time-varying. This is the "conditioning" half of `fit_risk_model`'s
        spec: a full GARCH refit per day is what Phase 3 already validated;
        this is the cheap, still-causal middle ground the risk *engine* uses
        so its own VaR isn't frozen at one full-sample volatility level for
        years at a time.
        """
        q = self._lower_q_at_scale(alpha, sigma_t)
        return float(-q * np.sqrt(horizon))

    def es_conditional(self, alpha: float, sigma_t: float, horizon: int = 1) -> float:
        e = self._lower_es_at_scale(alpha, sigma_t)
        return float(-e * np.sqrt(horizon))

    def _lower_q_at_scale(self, alpha: float, sigma_t: float) -> float:
        base = self._lower_q(alpha)
        return (
            float(self.mean + (base - self.mean) * (sigma_t / self.std))
            if self.std > 0
            else base
        )

    def _lower_es_at_scale(self, alpha: float, sigma_t: float) -> float:
        base = self._lower_es(alpha)
        return (
            float(self.mean + (base - self.mean) * (sigma_t / self.std))
            if self.std > 0
            else base
        )

    def simulate(self, n: int, seed: int = 0) -> np.ndarray:
        """Draw n i.i.d. returns from the fitted marginal (used to build
        portfolio-level Monte Carlo scenarios and copula transforms)."""
        rng = np.random.default_rng(seed)
        u = rng.uniform(1e-6, 1 - 1e-6, n)
        return self.ppf_from_u(u)

    def ppf_from_u(self, u: np.ndarray) -> np.ndarray:
        if self.kind == "loc_scale":
            d = dist.frozen_dist(self.family, self.params)
            return np.asarray(d.ppf(u))
        mod = densities.REGISTRY[self.family]
        # nig.ppf has no closed form and root-finds per point (~50ms/point) --
        # far too slow for a Monte Carlo draw of thousands of points, so a
        # numerical grid-inversion is used for every standardized family here
        # (fast, and consistent regardless of which family is fitted).
        z = numerical_ppf(lambda zz: mod.logpdf(zz, self.params), u)
        return self.mean + self.std * z

    def cdf_from_x(self, x: np.ndarray) -> np.ndarray:
        if self.kind == "loc_scale":
            d = dist.frozen_dist(self.family, self.params)
            return np.asarray(d.cdf(x))
        mod = densities.REGISTRY[self.family]
        z = (x - self.mean) / self.std
        return numerical_pit(lambda zz: mod.logpdf(zz, self.params), z)

    def stress(self, scenario_returns: np.ndarray) -> StressResult:
        """Replay a named historical event's realized return path against
        this model: cumulative P&L and the worst single-day loss, both on the
        return scale (a $1 notional). `scenario_returns` is a log-return
        path (e.g. the product's own returns during a Phase 1 named event
        window)."""
        scenario_returns = scenario_returns[np.isfinite(scenario_returns)]
        if len(scenario_returns) == 0:
            return {"cum_return": None, "worst_day": None, "n_days": 0}
        return {
            "cum_return": float(np.sum(scenario_returns)),
            "worst_day": float(np.min(scenario_returns)),
            "n_days": len(scenario_returns),
        }


def ewma_vol(
    returns: np.ndarray, lam: float = 0.94, seed_window: int = 20
) -> np.ndarray:
    """Causal RiskMetrics-style EWMA volatility path: sigma2_t = lam *
    sigma2_{t-1} + (1-lam) * r_{t-1}^2, seeded by the sample variance of the
    first `seed_window` observations. sigma_t (the output) is only ever a
    function of r_0..r_{t-1} -- never r_t itself -- so it is safe to use as
    "today's" conditioning volatility for a VaR forecast made before the
    close.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    sigma2 = np.full(n, np.nan)
    if n < seed_window + 1:
        return np.sqrt(np.maximum(sigma2, 0))
    sigma2[seed_window] = float(np.var(r[:seed_window]))
    for t in range(seed_window + 1, n):
        sigma2[t] = lam * sigma2[t - 1] + (1 - lam) * r[t - 1] ** 2
    return np.sqrt(np.maximum(sigma2, 0))


def fit_risk_model(returns: np.ndarray, product: str, family: str) -> RiskModel | None:
    """Fit `family` fresh on the full `returns` series. `family` is expected
    to come from the caller's own family-selection JSON (Phase 2/3 output) --
    this function does not choose a family itself, per sec 4 Phase 7's
    "driven by Phase 2/3 results, not hardcoded" requirement.

    This is the low-level, verbatim-ported fit routine: it takes a plain
    ndarray and applies no data-contract check, exactly like its
    `commod_lib8.py` original, because the reproduction gate (NEXT_PROMPT.md
    sec 6.1) calls it exactly this way and must see bit-identical behaviour.
    The contract-enforcing entry point is `risk.fit()` (`src/risk/__init__.py`),
    which calls `hygiene.assert_risk_inputs` on a `build_risk_inputs` frame
    before delegating here -- see NEXT_PROMPT.md sec 4.1's "fit_risk_model
    should refuse to fit a frame that did not come through build_risk_inputs",
    which is about that higher-level entry point, not this one.
    """
    returns = returns[np.isfinite(returns)]
    if len(returns) < 100:
        return None
    if family in ("normal", "t"):
        fit_fn = dist._fit_normal if family == "normal" else dist._fit_t
        params = fit_fn(returns)
        if params is None:
            return None
        return RiskModel(
            product,
            family,
            "loc_scale",
            mean=float(np.mean(returns)),
            std=float(np.std(returns)),
            params=params,
        )
    if family == "spliced_evt":
        return None  # not supported as a standalone RiskModel (needs a variance process); use Phase 3's own path instead
    mod = densities.REGISTRY[family]
    mean, std = float(np.mean(returns)), float(np.std(returns))
    z = (returns - mean) / std
    shape = mod.fit(z)
    if shape is None:
        return None
    return RiskModel(product, family, "standardized", mean=mean, std=std, params=shape)
