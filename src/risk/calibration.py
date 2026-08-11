"""Calibration and monitoring battery.

`kupiec_by_state` is ported verbatim from `src/research/tmp/commod_lib8.py`
(lines 1328-1370 as of `7641ee4`). `acerbi_szekely_z` /
`acerbi_szekely_bootstrap_pvalue` are ported verbatim from
`src/research/tmp/dist_lib5.py` (lines 735, 785) -- the single strongest
result in the programme (008 Gate CE: 15/16 development, 11/16 holdout; 005's
"every non-fat-tailed model has a significantly positive Z at the 1% level,
at every interval, with zero exceptions"). Both source files re-import these
names and re-export them so notebooks 005/008 and their tests keep passing
unchanged against the promoted code (NEXT_PROMPT.md sec 3.3).

`CalibrationMonitor` (below the ported section) is new: the continuous
monitoring battery NEXT_PROMPT.md sec 6.3 specifies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

import distributions as dist
from risk.model import RiskModel

__all__ = [
    "CalibrationMonitor",
    "CalibrationStatus",
    "acerbi_szekely_bootstrap_pvalue",
    "acerbi_szekely_z",
    "kupiec_by_state",
]


# ---------------------------------------------------------------------------
# Ported verbatim from commod_lib8.py:1328-1370.
# ---------------------------------------------------------------------------


def kupiec_by_state(
    hits: np.ndarray, states: list[str] | np.ndarray, expected_rate: float
) -> dict:
    """Kupiec unconditional-coverage test run separately within each state
    label, plus the pooled (unconditional) test, for a direct
    state-conditioned-vs-unconditional coverage comparison (Gate CI)."""
    hits = np.asarray(hits)
    states = np.asarray(states)
    out: dict = {}
    for state in sorted(set(states.tolist())):
        mask = states == state
        h = hits[mask]
        if len(h) < 20:
            out[state] = {"n": len(h), "kupiec_p": None, "observed_rate": None}
            continue
        _, p = dist.kupiec_test(h, expected_rate)
        out[state] = {
            "n": len(h),
            "kupiec_p": float(p),
            "observed_rate": float(np.mean(h)),
        }
    _, p_all = dist.kupiec_test(hits, expected_rate)
    out["_pooled"] = {
        "n": len(hits),
        "kupiec_p": float(p_all),
        "observed_rate": float(np.mean(hits)),
    }
    return out


# ---------------------------------------------------------------------------
# Ported verbatim from dist_lib5.py:735 (acerbi_szekely_z) and :785
# (acerbi_szekely_bootstrap_pvalue).
# ---------------------------------------------------------------------------


def acerbi_szekely_z(
    actual: np.ndarray, var_forecast: np.ndarray, es_forecast: np.ndarray, q: float
) -> float:
    """Acerbi-Szekely (2014) Test 2 statistic for expected-shortfall
    calibration: Z ~= 0 means well-calibrated ES; Z > 0 means realized tail
    losses are WORSE (more extreme) than the model's own ES predicted - the
    failure mode that matters, i.e. the model understates tail risk. Z < 0
    means the model was overly conservative (realized losses milder than
    predicted). var_forecast/es_forecast are signed (negative for a
    lower-tail loss threshold), aligned to actual.

    Two corrections to NEXT_RUN_PROMPT.md's own pseudocode, both verified
    numerically (not just re-derived on paper) before trusting them:

    1. Sign of the "+/-1" term. The runbook writes "... + 1"; a 20M-draw
       Monte Carlo check (a correctly-specified normal model) showed the
       "+1" form gives E[Z] ~= 2 for a perfectly calibrated model, not 0 -
       inconsistent with its own stated interpretation. "- 1" (matching
       Acerbi & Szekely 2014's published Test 2) gives E[Z] ~= 0 under
       correct calibration, confirmed by the same check. Implemented as
       "- 1" here.
    2. Direction of the failure mode. The runbook states "Z < 0 means
       realized tail losses exceed the model's own prediction" - checked
       directly with a synthetic mis-specified model (true volatility 2-3x
       the model's assumed volatility, i.e. a model that understates risk)
       and found this produces a strongly POSITIVE Z (+9 to +14 in two
       separate checks), not negative. The algebraic reason: for a hit bar,
       actual and es_forecast are both negative; realized losses more
       extreme than predicted means |actual| > |es_forecast|, so
       actual/es_forecast > 1 (a ratio of two negatives, larger-magnitude
       numerator), pushing the mean ratio - and hence Z - positive. Both
       corrections are exactly the kind of "read the actual numbers, don't
       trust pseudocode blindly" check this repo's own history has
       repeatedly needed (see the six bugs documented in
       src/results/004_distributional_models.md).
    """
    mask = (
        np.isfinite(actual)
        & np.isfinite(var_forecast)
        & np.isfinite(es_forecast)
        & (es_forecast != 0)
    )
    a, v, e = actual[mask], var_forecast[mask], es_forecast[mask]
    hit = a < v
    n = len(a)
    if n == 0 or hit.sum() == 0:
        return float("nan")
    return float((1.0 / (n * q)) * np.sum(np.where(hit, a / e, 0.0)) - 1.0)


def acerbi_szekely_bootstrap_pvalue(
    z_observed: float,
    simulate_fn: object,
    n_boot: int = 200,
    seed: int = 0,
) -> float:
    """Bootstrap p-value for the Acerbi-Szekely Z statistic: no closed form
    exists (per NEXT_RUN_PROMPT.md), so the null distribution of Z under
    "the model is correctly specified" is built by drawing n_boot simulated
    return paths from the model's OWN predictive distribution (via
    `simulate_fn()`, which the caller provides - a full draw of one
    simulated actual/var/es-aligned Z value) and comparing the observed Z's
    rank against them. Two-sided: Z>0 (worse than predicted - the model
    understates tail risk) is the failure mode that matters, but a two-sided
    test is reported for completeness.
    """
    rng = np.random.default_rng(seed)
    boot_z = np.array([simulate_fn(rng) for _ in range(n_boot)])  # type: ignore[operator]
    boot_z = boot_z[np.isfinite(boot_z)]
    if len(boot_z) == 0:
        return float("nan")
    p_lo = float(np.mean(boot_z <= z_observed))
    p_hi = float(np.mean(boot_z >= z_observed))
    return float(min(1.0, 2.0 * min(p_lo, p_hi)))


# ---------------------------------------------------------------------------
# CalibrationMonitor (NEXT_PROMPT.md sec 6.3-6.4) -- new code, not a port.
# Thresholds are pre-registered in risk_engine_preregistration.json BEFORE
# this monitor ever runs on real data (sec 12: "monitoring thresholds tuned
# after seeing the alerts" is the failure mode this discipline prevents).
# ---------------------------------------------------------------------------


def _load_prereg_thresholds() -> dict[str, Any]:
    # research/tmp/ is a scratch directory, not an importable package (no
    # __init__.py), so this is a plain filesystem path relative to src/
    # (this module's grandparent), not an importlib.resources lookup.
    src_root = Path(__file__).resolve().parents[1]
    candidate = src_root / "research" / "tmp" / "risk_engine_preregistration.json"
    try:
        with candidate.open("r") as f:
            payload: dict[str, Any] = json.load(f)
        return dict(payload["calibration_monitor"])
    except (FileNotFoundError, OSError, KeyError) as exc:
        raise FileNotFoundError(
            f"{candidate} not found or malformed -- CalibrationMonitor "
            "thresholds must be pre-registered before use (NEXT_PROMPT.md "
            "sec 12)"
        ) from exc


@dataclass(frozen=True)
class LevelResult:
    """One test level's (e.g. 1%) calibration battery for one evaluation
    window."""

    level: float
    n: int
    observed_rate: float
    expected_rate: float
    kupiec_p: float
    independence_p: float
    cc_p: float
    max_cluster_length: int
    coverage_breach: bool
    clustering_breach: bool


@dataclass(frozen=True)
class AcerbiResult:
    level: float
    z_lower: float
    p_lower: float
    z_upper: float
    p_upper: float
    shape_breach: bool


@dataclass(frozen=True)
class CalibrationStatus:
    """A single evaluation window's verdict for one product (NEXT_PROMPT.md
    sec 6.3): `status` in {"ok", "warn", "breach"}, `failure_mode` in
    {"coverage", "clustering", "shape", "both", None}."""

    product: str
    levels: dict[float, LevelResult]
    acerbi: AcerbiResult | None
    status: str
    failure_mode: str | None


def _max_violation_cluster_length(hits: np.ndarray) -> int:
    hits = np.asarray(hits, dtype=bool)
    if len(hits) == 0:
        return 0
    max_run = 0
    cur = 0
    for h in hits:
        if h:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run


def _conditional_quantile_series(
    model: RiskModel, sigma_t: np.ndarray, alpha: float
) -> np.ndarray:
    """Signed conditional quantile at level `alpha`, for every finite
    `sigma_t`; NaN elsewhere. Uses the same scale-conditioning
    `RiskModel.var_conditional` does internally, but returns the *signed*
    quantile (negative for a lower-tail loss) rather than a positive VaR
    magnitude, which is what `acerbi_szekely_z`'s signed-input convention
    (docstring above) needs."""
    out = np.full(len(sigma_t), np.nan)
    finite = np.isfinite(sigma_t)
    for i in np.where(finite)[0]:
        out[i] = model._lower_q_at_scale(alpha, sigma_t[i])
    return out


def _standardized_ppf(model: RiskModel, u: np.ndarray) -> np.ndarray:
    """`u` -> standardized (mean 0, std 1 scale) quantiles of `model`'s fitted
    shape, regardless of `kind`. `RiskModel._lower_q`/`_lower_es` bake in the
    fitted mean/std for `alpha < 0.5` only (a lower-tail ES formula is not a
    general "ES at level alpha" formula for alpha > 0.5); this is the shared
    building block both `_conditional_es_series`'s upper tail and the
    Acerbi-Szekely null simulator use to stay valid at any alpha."""
    if model.kind == "loc_scale":
        d = dist.frozen_dist(model.family, model.params)
        raw = np.asarray(d.ppf(u), dtype=float)
        return (raw - model.mean) / model.std if model.std > 0 else np.zeros_like(raw)
    from risk import densities

    mod = densities.REGISTRY[model.family]
    return np.asarray(mod.ppf(u, model.params), dtype=float)


def _upper_tail_es_z(model: RiskModel, q: float, n_points: int = 200) -> float:
    """Standardized-scale average of the top-`q` fraction of `model`'s fitted
    shape -- the upper-tail analogue of `commod_lib8.zoo_es_forecast_upper`,
    generalized to loc_scale families too (numerical integration over the
    model's own ppf, since a family's `es(q, shape)` is defined as the
    average of the *bottom* q-fraction only and is not reusable for the top
    tail on a family that may be skewed)."""
    u = np.linspace(1 - q, 1 - 1e-6, n_points)
    return float(np.mean(_standardized_ppf(model, u)))


def _conditional_es_series(
    model: RiskModel, sigma_t: np.ndarray, alpha: float, tail: str = "lower"
) -> np.ndarray:
    """`alpha` is always the *quantile level* passed to `_lower_q_at_scale`
    elsewhere for the same tail (e.g. 0.01 for the lower tail, 0.99 for the
    upper tail) -- matching `_conditional_quantile_series`'s convention.
    `_upper_tail_es_z` instead needs the *exceedance probability* (the tail
    mass fraction, `1 - alpha` for the upper tail), which is derived here."""
    out = np.full(len(sigma_t), np.nan)
    finite = np.isfinite(sigma_t)
    if tail == "lower":
        for i in np.where(finite)[0]:
            out[i] = model._lower_es_at_scale(alpha, sigma_t[i])
    else:
        es_z = _upper_tail_es_z(model, 1 - alpha)
        out[finite] = model.mean + es_z * sigma_t[finite]
    return out


def _conditional_acerbi_simulate_fn(
    model: RiskModel, sigma_t: np.ndarray, q: float, tail: str = "lower"
):
    """A `simulate_fn(rng)` for `acerbi_szekely_bootstrap_pvalue`: draws one
    simulated conditional return path from the model's OWN fitted shape,
    rescaled to the same time-varying `sigma_t` path used for the observed
    forecast, then returns that draw's own Acerbi-Szekely Z -- the null
    distribution of Z "if this model is exactly correctly specified".

    `q` is always the exceedance probability (e.g. 0.01), matching
    `acerbi_szekely_z`'s own `q` argument; for `tail="upper"` the quantile
    level used internally is `1 - q`, and the comparison is made on the
    sign-reflected series, exactly mirroring how `z_upper` itself is
    computed in `CalibrationMonitor.evaluate`.
    """
    finite = np.isfinite(sigma_t)
    sigma_used = sigma_t[finite]
    n = len(sigma_used)
    quantile_level = q if tail == "lower" else 1 - q
    var_signed = np.array(
        [model._lower_q_at_scale(quantile_level, s) for s in sigma_used]
    )
    if tail == "lower":
        es_signed = np.array(
            [model._lower_es_at_scale(quantile_level, s) for s in sigma_used]
        )
    else:
        es_z = _upper_tail_es_z(model, q)
        es_signed = model.mean + es_z * sigma_used

    def simulate_fn(rng: np.random.Generator) -> float:
        u = rng.uniform(1e-6, 1 - 1e-6, n)
        z = _standardized_ppf(model, u)
        simulated = model.mean + z * sigma_used
        if tail == "lower":
            return acerbi_szekely_z(simulated, var_signed, es_signed, q)
        return acerbi_szekely_z(-simulated, -var_signed, -es_signed, q)

    return simulate_fn


class CalibrationMonitor:
    """Given a product's fitted `RiskModel`, its realised return series and a
    causal conditioning-volatility path (e.g. `ewma_vol`), computes the full
    calibration battery on a rolling window and emits an `ok`/`warn`/`breach`
    status distinguishing `coverage`/`clustering`/`shape`/`both` failure
    modes (NEXT_PROMPT.md sec 6.3).

    Thresholds are loaded from `risk_engine_preregistration.json`'s
    `calibration_monitor` block by default; pass `thresholds=` to override
    (e.g. in a test), but production use must go through the pre-registered
    file.
    """

    def __init__(self, thresholds: dict[str, Any] | None = None):
        self.thresholds = thresholds or _load_prereg_thresholds()

    def _level_battery(self, hits: np.ndarray, level: float) -> LevelResult:
        hits = np.asarray(hits, dtype=bool)
        n = len(hits)
        if n < 20:
            return LevelResult(
                level=level,
                n=n,
                observed_rate=float(np.mean(hits)) if n else float("nan"),
                expected_rate=level,
                kupiec_p=float("nan"),
                independence_p=float("nan"),
                cc_p=float("nan"),
                max_cluster_length=_max_violation_cluster_length(hits),
                coverage_breach=False,
                clustering_breach=False,
            )
        _, kupiec_p = dist.kupiec_test(hits, level)
        _, independence_p = dist.christoffersen_independence_test(hits)
        _, cc_p = dist.christoffersen_conditional_coverage_test(hits, level)
        p_threshold = float(self.thresholds["p_value_threshold"])
        return LevelResult(
            level=level,
            n=n,
            observed_rate=float(np.mean(hits)),
            expected_rate=level,
            kupiec_p=float(kupiec_p),
            independence_p=float(independence_p),
            cc_p=float(cc_p),
            max_cluster_length=_max_violation_cluster_length(hits),
            coverage_breach=kupiec_p < p_threshold,
            clustering_breach=independence_p < p_threshold,
        )

    def evaluate_from_hits(
        self,
        product: str,
        hits_by_level: dict[float, np.ndarray],
        acerbi: AcerbiResult | None = None,
    ) -> CalibrationStatus:
        """Coverage/clustering battery from precomputed boolean exceedance
        (hit) arrays, one per level -- for callers (e.g. a walk-forward
        backtest that refits a different model per fold) that already have
        correctly-computed hits and would otherwise have to force them
        through a single `(model, sigma_t)` pair `evaluate()` assumes one
        consistent fit for. Acerbi-Szekely (which needs a specific model's
        VaR/ES forecast, not just a hit indicator) is optional and omitted
        by default; pass a precomputed `AcerbiResult` if available."""
        level_results = {
            level: self._level_battery(hits, level)
            for level, hits in hits_by_level.items()
        }
        status, failure_mode = self._verdict(level_results, acerbi)
        return CalibrationStatus(
            product=product,
            levels=level_results,
            acerbi=acerbi,
            status=status,
            failure_mode=failure_mode,
        )

    def evaluate(
        self,
        product: str,
        model: RiskModel,
        returns: np.ndarray,
        sigma_t: np.ndarray,
        levels: tuple[float, ...] = (0.01, 0.025),
        acerbi_level: float = 0.01,
        acerbi_n_boot: int = 200,
        acerbi_seed: int = 0,
        compute_acerbi: bool = True,
    ) -> CalibrationStatus:
        """One evaluation window (a full backtest period, or one rolling
        slice of live history -- the caller decides what `returns`/`sigma_t`
        cover; this function only computes the battery over whatever it is
        given), for the single-fit case: one `RiskModel` scores the whole
        window. For a walk-forward window spanning multiple refits, use
        `evaluate_from_hits` with precomputed per-fold hits instead."""
        returns = np.asarray(returns, dtype=float)
        sigma_t = np.asarray(sigma_t, dtype=float)

        level_results: dict[float, LevelResult] = {}
        for level in levels:
            var_signed = _conditional_quantile_series(model, sigma_t, level)
            valid = np.isfinite(returns) & np.isfinite(var_signed)
            hits = dist.exceedances(returns[valid], var_signed[valid], side="lower")
            level_results[level] = self._level_battery(hits, level)

        acerbi_result: AcerbiResult | None = None
        if compute_acerbi:
            var_lo = _conditional_quantile_series(model, sigma_t, acerbi_level)
            es_lo = _conditional_es_series(model, sigma_t, acerbi_level)
            valid = np.isfinite(returns) & np.isfinite(var_lo) & np.isfinite(es_lo)
            z_lower = acerbi_szekely_z(
                returns[valid], var_lo[valid], es_lo[valid], acerbi_level
            )
            sim_lower = _conditional_acerbi_simulate_fn(model, sigma_t, acerbi_level)
            p_lower = acerbi_szekely_bootstrap_pvalue(
                z_lower, sim_lower, n_boot=acerbi_n_boot, seed=acerbi_seed
            )

            upper_level = 1 - acerbi_level
            var_hi = _conditional_quantile_series(model, sigma_t, upper_level)
            es_hi = _conditional_es_series(model, sigma_t, upper_level, tail="upper")
            valid_hi = np.isfinite(returns) & np.isfinite(var_hi) & np.isfinite(es_hi)
            z_upper = acerbi_szekely_z(
                -returns[valid_hi], -var_hi[valid_hi], -es_hi[valid_hi], acerbi_level
            )
            sim_upper = _conditional_acerbi_simulate_fn(
                model, sigma_t, acerbi_level, tail="upper"
            )
            p_upper = acerbi_szekely_bootstrap_pvalue(
                z_upper, sim_upper, n_boot=acerbi_n_boot, seed=acerbi_seed
            )

            p_threshold = float(self.thresholds["p_value_threshold"])
            shape_breach = (
                np.isfinite(z_lower)
                and z_lower > 0
                and p_lower < p_threshold
                or (np.isfinite(z_upper) and z_upper > 0 and p_upper < p_threshold)
            )
            acerbi_result = AcerbiResult(
                level=acerbi_level,
                z_lower=z_lower,
                p_lower=p_lower,
                z_upper=z_upper,
                p_upper=p_upper,
                shape_breach=bool(shape_breach),
            )

        status, failure_mode = self._verdict(level_results, acerbi_result)
        return CalibrationStatus(
            product=product,
            levels=level_results,
            acerbi=acerbi_result,
            status=status,
            failure_mode=failure_mode,
        )

    def _verdict(
        self,
        level_results: dict[float, LevelResult],
        acerbi_result: AcerbiResult | None,
    ) -> tuple[str, str | None]:
        """NEXT_PROMPT.md sec 6.3's failure-mode table:

        | failure_mode | signature                                    |
        |--------------|-----------------------------------------------|
        | coverage     | Kupiec fails, independence passes            |
        | clustering   | Kupiec passes, independence fails            |
        | shape        | coverage fine, Acerbi-Szekely Z significantly positive |
        | both         | Kupiec and independence both fail -- escalate |

        Any level's breach counts; the most severe applicable mode is
        reported (both > clustering/coverage individually > shape).
        """
        any_coverage = any(r.coverage_breach for r in level_results.values())
        any_clustering = any(r.clustering_breach for r in level_results.values())
        any_shape = bool(acerbi_result and acerbi_result.shape_breach)

        vr = self.thresholds.get("violation_rate_thresholds", {})
        warn_ratio = float(vr.get("warn_ratio_observed_over_expected", 1.5))
        breach_ratio = float(vr.get("breach_ratio_observed_over_expected", 2.0))
        cl = self.thresholds.get("max_cluster_length_thresholds", {})
        warn_cluster = int(cl.get("warn", 4))
        breach_cluster = int(cl.get("breach", 6))

        rate_breach = False
        cluster_breach = False
        rate_warn = False
        cluster_warn = False
        for r in level_results.values():
            if r.expected_rate <= 0 or not np.isfinite(r.observed_rate):
                continue
            ratio = r.observed_rate / r.expected_rate
            if ratio >= breach_ratio:
                rate_breach = True
            elif ratio >= warn_ratio:
                rate_warn = True
            if r.max_cluster_length >= breach_cluster:
                cluster_breach = True
            elif r.max_cluster_length >= warn_cluster:
                cluster_warn = True

        if any_coverage and any_clustering:
            return "breach", "both"
        if any_coverage or rate_breach:
            return "breach", "coverage"
        if any_clustering or cluster_breach:
            return "breach", "clustering"
        if any_shape:
            return "breach", "shape"
        if rate_warn or cluster_warn:
            return "warn", None
        return "ok", None

    def evaluate_batch(
        self,
        product_inputs: dict[str, tuple[RiskModel, np.ndarray, np.ndarray]],
        levels: tuple[float, ...] = (0.01, 0.025),
        acerbi_level: float = 0.01,
        acerbi_n_boot: int = 200,
        acerbi_seed: int = 0,
        compute_acerbi: bool = True,
    ) -> dict[str, CalibrationStatus]:
        """The production entry point: evaluates every product in
        `product_inputs` (each a `(model, returns, sigma_t)` triple, one
        evaluation window) and applies Benjamini-Hochberg correction to each
        test's p-values *across products* before deciding breach status
        (NEXT_PROMPT.md sec 6.3: "16 products x 2 levels x 3 tests run
        repeatedly will produce false alarms at any fixed alpha... Specify
        the correction (BH across products within each run is the natural
        choice, matching 008's own procedure)"). `evaluate()` alone uses
        uncorrected per-product p-values and is for single-product,
        one-off use.
        """
        raw = {
            product: self.evaluate(
                product,
                model,
                returns,
                sigma_t,
                levels=levels,
                acerbi_level=acerbi_level,
                acerbi_n_boot=acerbi_n_boot,
                acerbi_seed=acerbi_seed,
                compute_acerbi=compute_acerbi,
            )
            for product, (model, returns, sigma_t) in product_inputs.items()
        }
        return self._apply_bh_correction(raw, levels)

    def evaluate_batch_from_hits(
        self,
        product_hits: dict[str, dict[float, np.ndarray]],
        product_acerbi: dict[str, AcerbiResult] | None = None,
    ) -> dict[str, CalibrationStatus]:
        """`evaluate_batch`'s counterpart for `evaluate_from_hits` -- BH
        correction across products, applied to precomputed per-fold hit
        arrays (e.g. a walk-forward backtest that refit a different model
        per fold, so no single `(model, sigma_t)` pair is meaningful)."""
        product_acerbi = product_acerbi or {}
        levels = tuple(next(iter(product_hits.values())).keys()) if product_hits else ()
        raw = {
            product: self.evaluate_from_hits(
                product, hits_by_level, acerbi=product_acerbi.get(product)
            )
            for product, hits_by_level in product_hits.items()
        }
        return self._apply_bh_correction(raw, levels)

    def _apply_bh_correction(
        self, raw: dict[str, CalibrationStatus], levels: tuple[float, ...]
    ) -> dict[str, CalibrationStatus]:
        alpha = float(self.thresholds["p_value_threshold"])

        kupiec_sig_by_level: dict[float, dict[str, bool]] = {}
        indep_sig_by_level: dict[float, dict[str, bool]] = {}
        for level in levels:
            kupiec_p = {
                p: s.levels[level].kupiec_p for p, s in raw.items() if level in s.levels
            }
            indep_p = {
                p: s.levels[level].independence_p
                for p, s in raw.items()
                if level in s.levels
            }
            kupiec_sig_by_level[level] = _benjamini_hochberg_significant(
                kupiec_p, alpha
            )
            indep_sig_by_level[level] = _benjamini_hochberg_significant(indep_p, alpha)

        lower_sig: dict[str, bool] = {}
        upper_sig: dict[str, bool] = {}
        have_acerbi = any(s.acerbi is not None for s in raw.values())
        if have_acerbi:
            p_lower = {
                p: s.acerbi.p_lower for p, s in raw.items() if s.acerbi is not None
            }
            p_upper = {
                p: s.acerbi.p_upper for p, s in raw.items() if s.acerbi is not None
            }
            lower_sig = _benjamini_hochberg_significant(p_lower, alpha)
            upper_sig = _benjamini_hochberg_significant(p_upper, alpha)

        corrected: dict[str, CalibrationStatus] = {}
        for product, status in raw.items():
            new_levels = {
                level: replace(
                    lr,
                    coverage_breach=kupiec_sig_by_level[level].get(product, False),
                    clustering_breach=indep_sig_by_level[level].get(product, False),
                )
                for level, lr in status.levels.items()
            }
            new_acerbi = status.acerbi
            if status.acerbi is not None:
                z_lower_positive = np.isfinite(status.acerbi.z_lower) and (
                    status.acerbi.z_lower > 0
                )
                z_upper_positive = np.isfinite(status.acerbi.z_upper) and (
                    status.acerbi.z_upper > 0
                )
                shape_breach = (z_lower_positive and lower_sig.get(product, False)) or (
                    z_upper_positive and upper_sig.get(product, False)
                )
                new_acerbi = replace(status.acerbi, shape_breach=bool(shape_breach))
            verdict_status, failure_mode = self._verdict(new_levels, new_acerbi)
            corrected[product] = CalibrationStatus(
                product=product,
                levels=new_levels,
                acerbi=new_acerbi,
                status=verdict_status,
                failure_mode=failure_mode,
            )
        return corrected


def _benjamini_hochberg_significant(
    pvalues: dict[str, float], alpha: float
) -> dict[str, bool]:
    """Standard BH step-up procedure: reject every hypothesis up to and
    including the largest rank k with p_(k) <= (k/m)*alpha, m = number of
    finite p-values. Returns {key: significant} for every key in `pvalues`
    (non-finite p-values are never significant)."""
    finite = [(k, p) for k, p in pvalues.items() if np.isfinite(p)]
    finite.sort(key=lambda kv: kv[1])
    m = len(finite)
    if m == 0:
        return dict.fromkeys(pvalues, False)
    k_star = 0
    for rank, (_key, p) in enumerate(finite, start=1):
        if p <= (rank / m) * alpha:
            k_star = rank
    significant_keys = {k for k, _p in finite[:k_star]}
    return {k: (k in significant_keys) for k in pvalues}
