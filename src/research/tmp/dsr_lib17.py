"""Notebook 017 library (NEXT_PROMPT.md sec 7.1).

Implements the Bailey-Lopez de Prado Deflated Sharpe Ratio's expected-maximum
bracket and three candidate repairs of `research.deflated_sharpe_prob`'s
sec 1 defect (the deflation benchmark's scale term uses the sampling SE of
one Sharpe instead of the family's cross-sectional dispersion). The public
surface mirrors sec 7.1 exactly: `expected_max_sharpe`, `dsr_variant`,
`psr_upper_bound`, `mc_cell`, plus the seed/moment-generation plumbing Phase
2 and Phase 3 both need.

`_sr_se` below is the SAME formula as `research.deflated_sharpe_prob`'s
`sr_std` line (sec 7.1's "PSR z-score and moment-adjusted SE, factored
out" reuse rule) -- Test 1 in tests/test_dsr_lib17.py is the executable
proof that dsr_variant(variant="v0") reproduces research.py's published
numbers bit for bit.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import jf_skew_t, norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import distributions

EULER_MASCHERONI = 0.5772156649

# --------------------------------------------------------------------------
# sec 4.1's three moment regimes.
#
# Gaussian (0.0, 3.0) is exact. The other two are targets frozen in Phase 0
# (skew, TOTAL kurtosis, kurtosis=3 is the no-excess-kurtosis case, matching
# research.deflated_sharpe_prob's own convention).
#
# "moderate_nongaussian" (-1.5, 6.0) is drawn from distributions.frozen_dist
# ("skewt" = st.jf_skew_t) per sec 4.1's instruction. Its (a, b) were solved
# by scipy.optimize (Nelder-Mead, minimizing squared error in (skew, excess
# kurtosis)) and the fit is NOT exact -- explored numerically before this
# grid ran: at fixed b -> infinity, jf_skew_t's skew magnitude and kurtosis
# both grow with decreasing a, but skew=-1.5 forces excess kurtosis >= ~5,
# it cannot land at kurtosis=6 exactly alongside skew=-1.5. The closest fit
# holding kurtosis close to the target lands at skew=-1.208 (kurtosis=6.05,
# vs. targets skew=-1.5/kurtosis=6.0). Achieved values are recorded in every
# cell's output so this is measured, not asserted.
#
# "018_measured" (-11.5, 817.0) cannot be reached by jf_skew_t AT ALL: the
# same boundary search (a -> 2+, b -> infinity, the family's kurtosis-
# existence limit) shows |skew| saturates near 5.1-5.2 as kurtosis
# diverges to infinity -- skew=-11.5 is outside jf_skew_t's feasible region
# at ANY finite kurtosis, let alone 817. A two-point jump mixture (rare
# deterministic downward jump + Gaussian bulk, mean 0, var 1) is used
# instead: this reaches the target EXACTLY (method-of-moments solve,
# verified below), and is a more honest model of how 018's own 0.01746
# per-period Sharpe book actually produced sample skew=-11.5/kurtosis=817
# in the first place -- a few extreme funding-rate bars, not a smoothly
# heavy-tailed continuous return series. This deviation from
# distributions.frozen_dist is disclosed here and in the sec 6 write-up.
# --------------------------------------------------------------------------

GAUSSIAN_MOMENTS = (0.0, 3.0)
MODERATE_MOMENTS = (-1.5, 6.0)
EXTREME_MOMENTS = (-11.5, 817.0)

# solved via scipy.optimize.minimize (Nelder-Mead) against
# st.jf_skew_t.stats(a=a, b=b, moments="mvsk"); see module docstring.
MODERATE_SKEWT_PARAMS = {"a": 6.257592, "b": 1_000_000.0}

# solved via scipy.optimize.fsolve, exact to float precision (see module
# docstring): a two-point mixture, prob (1-p) ~ N(loc, sigma^2), prob p a
# deterministic jump at -k. loc is fixed by the mean=0 constraint.
_EXTREME_P = 3.23859054e-05
_EXTREME_K = 70.8249809
_EXTREME_SIGMA = 0.915187631
_EXTREME_LOC = _EXTREME_P * _EXTREME_K / (1 - _EXTREME_P)


def _sr_se(
    sharpe: float | np.ndarray,
    n_obs: int,
    skew: float | np.ndarray,
    kurtosis: float | np.ndarray,
) -> Any:
    """Lo (2002) moment-adjusted sampling SE of one Sharpe estimate. Bit for
    bit the `sr_std` line in research.deflated_sharpe_prob (src/research.py)."""
    return np.sqrt((1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe**2) / (n_obs - 1))


def expected_max_sharpe(
    dispersion: float | np.ndarray, n_trials: float | np.ndarray
) -> Any:
    """The Bailey-LdP expected-maximum-of-N-Gaussians bracket, times a scale.
    Pure, no policy: what that scale IS (sec 2.1's V0/V1/V2) is `dsr_variant`'s
    job, not this function's. n_trials may be an array (V2's per-replication,
    continuous N_eff), elementwise-safe via np.where rather than a Python
    `if`.

    The expected maximum of N mean-zero draws is never negative for any
    N >= 1 (trivially: at N=1 it's E[Z]=0; more trials can only raise a
    max, never lower it below that). The asymptotic EVT bracket this
    formula is built from is a large-N approximation and is NOT well
    behaved as N -> 1 from above (it diverges toward -infinity rather than
    the true limit of 0) -- this matters here because V2's continuous
    N_eff can land anywhere in (1, 2) at high correlation, a regime V1/V0's
    integer N never visits. Flooring at 0 is the correct value at the
    true boundary (N=1), not a fudge; result is `>= 0` for every input, and
    Test 7 (sec 7.3) is what actually exercises this regime for V2."""
    n_arr = np.asarray(n_trials, dtype=float)
    n_safe = np.where(n_arr > 1, n_arr, 2.0)  # avoid log/ppf domain errors where n<=1
    bracket = (1 - EULER_MASCHERONI) * norm.ppf(
        1 - 1 / n_safe
    ) + EULER_MASCHERONI * norm.ppf(1 - 1 / (n_safe * np.e))
    result = np.where(n_arr > 1, np.maximum(dispersion * bracket, 0.0), 0.0)
    return (
        result
        if isinstance(dispersion, np.ndarray) or isinstance(n_trials, np.ndarray)
        else float(result)
    )


def dsr_variant(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    *,
    variant: str = "v0",
    trial_sharpes: Sequence[float] | None = None,
    mean_pairwise_corr: float | None = None,
    shrinkage_c: float | None = None,
) -> dict[str, float | str | bool | int | None]:
    """One DSR evaluation under one of sec 2.1's variants. Returns the
    probability AND the working (dispersion used, SR*, n_trials used in the
    bracket, family size, family_mismatch) so Phase 5's ledger is auditable
    (sec 7.1).

    sharpe, n_obs, skew, kurtosis: same units/meaning as
    research.deflated_sharpe_prob (sharpe is PER-PERIOD, not annualized --
    sec 7.2's trap).
    n_trials: the honest trial count, always used for the bracket's N (sec
    2.2 rule 1 -- never silently replaced by len(trial_sharpes)).
    """
    if n_obs <= 1:
        return {"probability": float("nan"), "variant": variant}

    sr_se = float(_sr_se(sharpe, n_obs, skew, kurtosis))
    if sr_se == 0:
        return {
            "probability": 1.0,
            "variant": variant,
            "dispersion_used": 0.0,
            "sr_star": 0.0,
            "sr_se": 0.0,
            "n_trials_declared": n_trials,
            "n_trials_used_in_bracket": n_trials,
            "n_trial_sharpes_provided": None
            if trial_sharpes is None
            else len(trial_sharpes),
            "family_mismatch": False,
        }

    n_trial_sharpes_provided = None if trial_sharpes is None else len(trial_sharpes)
    family_mismatch = trial_sharpes is not None and n_trial_sharpes_provided != n_trials
    n_bracket: float = n_trials

    if variant == "v0":
        dispersion = sr_se
    elif variant in ("v1", "v1b"):
        if trial_sharpes is None or len(trial_sharpes) < 2:
            raise ValueError(f"{variant} requires >=2 trial_sharpes")
        disp_obs = float(np.std(np.asarray(trial_sharpes, dtype=float), ddof=1))
        if variant == "v1":
            dispersion = disp_obs
        else:
            if shrinkage_c not in (0.25, 0.5):
                raise ValueError("v1b requires shrinkage_c in {0.25, 0.5}")
            dispersion = max(disp_obs, shrinkage_c * sr_se)
    elif variant == "v2":
        if mean_pairwise_corr is None:
            raise ValueError("v2 requires mean_pairwise_corr")
        rho_bar = max(mean_pairwise_corr, -1.0 / max(n_trials - 1, 1) + 1e-9)
        n_eff = n_trials / (1 + (n_trials - 1) * rho_bar)
        n_bracket = max(n_eff, 1.0 + 1e-9)
        dispersion = sr_se
    else:
        raise ValueError(f"unknown variant: {variant!r}")

    sr_star = float(expected_max_sharpe(dispersion, n_bracket))
    probability = float(norm.cdf((sharpe - sr_star) / sr_se))

    return {
        "probability": probability,
        "variant": variant,
        "dispersion_used": dispersion,
        "sr_star": sr_star,
        "sr_se": sr_se,
        "n_trials_declared": n_trials,
        "n_trials_used_in_bracket": n_bracket,
        "n_trial_sharpes_provided": n_trial_sharpes_provided,
        "family_mismatch": family_mismatch,
    }


def psr_upper_bound(
    sharpe: float, n_obs: int, skew: float = 0.0, kurtosis: float = 3.0
) -> float:
    """The rho -> 1 limit (sec 5.4 item 2): the maximum DSR any
    dispersion-based repair can produce for these inputs, since dispersion
    -> 0 as the family becomes perfectly correlated, so SR* -> 0."""
    if n_obs <= 1:
        return float("nan")
    sr_se = float(_sr_se(sharpe, n_obs, skew, kurtosis))
    if sr_se == 0:
        return 1.0
    return float(norm.cdf(sharpe / sr_se))


def seed_for_cell(key: tuple[Any, ...]) -> int:
    """Deterministic seed derived from a hash of the cell's own key (sec
    7.2 rule 2 / test 6): never a running counter, so resuming an
    interrupted grid produces bit-identical results to an uninterrupted
    one."""
    digest = hashlib.sha256(repr(key).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def _moments_label(moments: tuple[float, float]) -> str:
    if moments == GAUSSIAN_MOMENTS:
        return "gaussian"
    if moments == MODERATE_MOMENTS:
        return "moderate_nongaussian"
    if moments == EXTREME_MOMENTS:
        return "018_measured"
    raise ValueError(f"unregistered moments target: {moments!r}")


def _draw_correlated_uniforms(
    rng: np.random.Generator, n_reps: int, n_obs: int, n_trials: int, rho: float
) -> np.ndarray:
    """Equicorrelated Gaussian copula: z = sqrt(rho)*common + sqrt(1-rho)*idio
    gives pairwise correlation exactly rho between trial columns for
    Gaussian marginals, and an approximate (NORTA) rank correlation for any
    marginal reached via u=Phi(z) below. Returns U in (0,1), shape
    (n_reps, n_obs, n_trials)."""
    zc = rng.standard_normal((n_reps, n_obs, 1))
    zi = rng.standard_normal((n_reps, n_obs, n_trials))
    z = np.sqrt(rho) * zc + np.sqrt(1 - rho) * zi
    u = norm.cdf(z)
    return np.clip(u, 1e-12, 1 - 1e-12)


_moderate_ppf_lookup: tuple[np.ndarray, np.ndarray] | None = None


def _build_moderate_ppf_lookup(n_grid: int = 4001) -> tuple[np.ndarray, np.ndarray]:
    """scipy's jf_skew_t.ppf is a slow generic numerical inversion (no
    closed form) -- calling it per-replication made the moderate-moments
    cells the grid's dominant cost by ~60x over the Gaussian cells at the
    same (N, T, M). Built once per process instead: a dense monotonic
    (u, x) lookup table (extra density in both tails, since the target
    excess kurtosis is 3.0 and tail behavior matters), then every actual
    draw is `np.interp` -- linear interpolation error here is negligible
    against Monte Carlo sampling noise at the M this grid runs at."""
    a, b = MODERATE_SKEWT_PARAMS["a"], MODERATE_SKEWT_PARAMS["b"]
    m, v = jf_skew_t.stats(a=a, b=b, moments="mv")  # type: ignore[call-overload]  # scipy-stubs expects moment=, runtime scipy takes moments=
    scale = 1.0 / float(np.sqrt(v))
    loc = -scale * float(m)
    dist = distributions.frozen_dist("skewt", [a, b, loc, scale])
    u_grid = np.unique(
        np.concatenate(
            [
                np.geomspace(1e-12, 0.01, 400, endpoint=False),
                np.linspace(0.01, 0.99, n_grid),
                1 - np.geomspace(1e-12, 0.01, 400, endpoint=False)[::-1],
            ]
        )
    )
    x_grid = dist.ppf(u_grid)
    return u_grid, x_grid


def _moderate_ppf(u: np.ndarray) -> np.ndarray:
    global _moderate_ppf_lookup
    if _moderate_ppf_lookup is None:
        _moderate_ppf_lookup = _build_moderate_ppf_lookup()
    u_grid, x_grid = _moderate_ppf_lookup
    return np.interp(u, u_grid, x_grid)


def _extreme_ppf(u: np.ndarray) -> np.ndarray:
    is_jump = u < _EXTREME_P
    normal_branch = _EXTREME_LOC + _EXTREME_SIGMA * norm.ppf(
        np.clip((u - _EXTREME_P) / (1 - _EXTREME_P), 1e-12, 1 - 1e-12)
    )
    return np.where(is_jump, -_EXTREME_K, normal_branch)


def draw_trial_returns(
    rng: np.random.Generator,
    n_reps: int,
    n_obs: int,
    n_trials: int,
    rho: float,
    moments: tuple[float, float],
) -> np.ndarray:
    """(n_reps, n_obs, n_trials) draws, each trial column standardized to
    mean 0 / var 1 marginally and equicorrelated at (approximately, for
    non-Gaussian marginals) rho. Chunked by the caller for memory (sec 4.1:
    a full (M, T, N) array can be tens of GB)."""
    label = _moments_label(moments)
    if label == "gaussian":
        zc = rng.standard_normal((n_reps, n_obs, 1))
        zi = rng.standard_normal((n_reps, n_obs, n_trials))
        return np.sqrt(rho) * zc + np.sqrt(1 - rho) * zi
    u = _draw_correlated_uniforms(rng, n_reps, n_obs, n_trials, rho)
    if label == "moderate_nongaussian":
        return _moderate_ppf(u)
    return _extreme_ppf(u)


def mean_pairwise_corr_estimate(x: np.ndarray) -> np.ndarray:
    """Cheap O(n_trials * n_obs) mean-pairwise-correlation estimator for
    equicorrelated-style designs, avoiding an O(n_trials^2) correlation
    matrix: Var(mean series) = Var(single)/N * (1 + (N-1)*rho_bar), so
    rho_bar = (N*Var(mean) - Var(single)) / ((N-1)*Var(single)).
    x: (n_reps, n_obs, n_trials). Returns (n_reps,)."""
    n_trials = x.shape[-1]
    var_i = x.var(axis=1, ddof=1)  # (n_reps, n_trials)
    mean_var_i = var_i.mean(axis=1)  # (n_reps,)
    mean_series = x.mean(axis=2)  # (n_reps, n_obs)
    var_mean_series = mean_series.var(axis=1, ddof=1)  # (n_reps,)
    if n_trials <= 1:
        return np.zeros_like(mean_var_i)
    rho_bar = (n_trials * var_mean_series - mean_var_i) / ((n_trials - 1) * mean_var_i)
    return rho_bar


VARIANT_SPECS: tuple[tuple[str, str, float | None], ...] = (
    ("v0", "v0", None),
    ("v1", "v1", None),
    ("v2", "v2", None),
    ("v1b_c0.25", "v1b", 0.25),
    ("v1b_c0.5", "v1b", 0.5),
)


def mc_cell(
    n_trials: int,
    n_obs: int,
    rho: float,
    moments: tuple[float, float],
    n_reps: int,
    seed: int,
    true_sharpe: float = 0.0,
    *,
    chunk: int = 200,
) -> dict[str, Any]:
    """One grid cell (sec 7.1): all 5 variants, null (true_sharpe=0, for
    DS-1/DS-2) or injected-edge (true_sharpe>0, for DS-3). Deterministic in
    (n_trials, n_obs, rho, moments, n_reps, seed, true_sharpe) -- test 6."""
    rng = np.random.default_rng(seed)
    threshold = 0.95

    hits: dict[str, int] = {name: 0 for name, _, _ in VARIANT_SPECS}
    sample_skews: list[float] = []
    sample_kurts: list[float] = []
    achieved_rho: list[float] = []

    done = 0
    while done < n_reps:
        m = min(chunk, n_reps - done)
        x = draw_trial_returns(rng, m, n_obs, n_trials, rho, moments)
        if true_sharpe != 0.0:
            x = x.copy()
            x[:, :, 0] = x[:, :, 0] + true_sharpe

        mean = x.mean(axis=1)  # (m, n_trials)
        std = x.std(axis=1, ddof=1)
        srs = mean / std  # (m, n_trials) per-period Sharpes

        # Only the WINNING trial's skew/kurtosis is ever used (the
        # formula's skew/kurtosis inputs, like a real notebook's, are the
        # selected trial's own sample moments -- sec 4.1). Computing 3rd/4th
        # central moments for all n_trials columns and discarding n_trials-1
        # of them was the grid's dominant cost at large N (95, 122): this
        # cuts that part from O(m*T*n_trials) to O(m*T), independent of N.
        best_idx = srs.argmax(axis=1)
        rows = np.arange(m)
        best_sharpe = srs[rows, best_idx]
        best_series = x[rows, :, best_idx]  # (m, n_obs)
        best_mean = mean[rows, best_idx]  # (m,)
        centered_best = best_series - best_mean[:, None]
        m2 = (centered_best**2).mean(axis=1)
        m3 = (centered_best**3).mean(axis=1)
        m4 = (centered_best**4).mean(axis=1)
        best_skew = m3 / np.maximum(m2, 1e-300) ** 1.5
        best_kurt = m4 / np.maximum(m2, 1e-300) ** 2  # TOTAL kurtosis (3.0 = normal)
        sample_skews.extend(best_skew.tolist())
        sample_kurts.extend(best_kurt.tolist())

        # mean-pairwise-corr estimator (sec 7.1), reusing `std` (var_i =
        # std**2) rather than recomputing x.var(axis=1) from scratch.
        var_i = std**2  # (m, n_trials), ddof=1 already
        mean_var_i = var_i.mean(axis=1)  # (m,)
        mean_series = x.mean(axis=2)  # (m, n_obs)
        var_mean_series = mean_series.var(axis=1, ddof=1)  # (m,)
        if n_trials > 1:
            rho_bar = (n_trials * var_mean_series - mean_var_i) / (
                (n_trials - 1) * mean_var_i
            )
        else:
            rho_bar = np.zeros(m)
        achieved_rho.extend(rho_bar.tolist())

        # Fully vectorized over the (m,) chunk -- sec 4.1's explicit
        # instruction ("norm.cdf on the whole (M,) array"). dsr_variant
        # (scalar, one call per input) exists for tests and single-value
        # use (Phase 4/5); this hot loop reimplements the same formulas
        # elementwise rather than calling it m*5 times per chunk.
        sr_se_arr = _sr_se(best_sharpe, n_obs, best_skew, best_kurt)
        disp_v1 = srs.std(axis=1, ddof=1)  # (m,)
        n_eff = n_trials / (
            1 + (n_trials - 1) * np.maximum(rho_bar, -1.0 / max(n_trials - 1, 1) + 1e-9)
        )
        n_eff = np.maximum(n_eff, 1.0 + 1e-9)

        dispersions = {
            "v0": sr_se_arr,
            "v1": disp_v1,
            "v2": sr_se_arr,
            "v1b_c0.25": np.maximum(disp_v1, 0.25 * sr_se_arr),
            "v1b_c0.5": np.maximum(disp_v1, 0.5 * sr_se_arr),
        }
        n_brackets: dict[str, float | np.ndarray] = {
            "v0": n_trials,
            "v1": n_trials,
            "v2": n_eff,
            "v1b_c0.25": n_trials,
            "v1b_c0.5": n_trials,
        }
        for name, _, _ in VARIANT_SPECS:
            sr_star = expected_max_sharpe(dispersions[name], n_brackets[name])
            probs = norm.cdf((best_sharpe - sr_star) / sr_se_arr)
            hits[name] += int((probs > threshold).sum())

        done += m

    rate = {name: hits[name] / n_reps for name in hits}
    mc_se = {
        name: float(np.sqrt(rate[name] * (1 - rate[name]) / n_reps)) for name in hits
    }

    return {
        "n_trials": n_trials,
        "n_obs": n_obs,
        "rho": rho,
        "moments": list(moments),
        "moments_label": _moments_label(moments),
        "n_reps": n_reps,
        "seed": seed,
        "true_sharpe": true_sharpe,
        "rate": rate,
        "mc_se": mc_se,
        "achieved_moments": {
            "mean_sample_skew_of_winner": float(np.mean(sample_skews)),
            "mean_sample_kurtosis_of_winner": float(np.mean(sample_kurts)),
            "mean_estimated_pairwise_corr": float(np.mean(achieved_rho)),
        },
    }
