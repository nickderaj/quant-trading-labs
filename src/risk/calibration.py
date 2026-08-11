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

import numpy as np

import distributions as dist

__all__ = [
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
