"""Notebook-5-local machinery: tail risk and conditional non-normality.

Mirrors dist_lib.py's own convention (`import dist_lib as L`, not a fork of it):
this module imports from dist_lib.py and reuses its causal, rolling-refit
building blocks (build_asset_frame, rolling_garch_forecast, refit-cadence
constants) rather than re-implementing them. Everything here is either new
machinery notebook 4 never needed (Hill estimator, GJR-GARCH, GPD/POT) or a
thin notebook-5-specific composition of dist_lib's existing pieces.

Run as a script from the repo root (sys.path.insert(0, "src")), and imported
from the notebook the same way dist_lib.py is (sys.path.insert(0, "tmp") from
src/research/).
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

import numpy as np
from scipy import stats as st

# --------------------------------------------------------------------------
# Phase 1: Hill estimator (tail index), independent of any parametric MLE fit
# --------------------------------------------------------------------------


def hill_estimator(x: np.ndarray, k: int, tail: str = "upper") -> float:
    """Hill estimate of the tail index alpha for the top-k order statistics.

    alpha = 1 / xi, where xi is the extreme-value (Pareto) shape. alpha < 2
    means infinite variance; alpha < 1 means infinite mean. Crypto log returns
    are expected somewhere in 2-4; the whole point of computing it is that
    notebook 4's fitted t-df of ~2 sits exactly on the finite-variance boundary
    and was produced by an optimizer we already know can pin at a bound.
    """
    x = x[np.isfinite(x)]
    y = np.sort(np.abs(x[x > 0]) if tail == "upper" else np.abs(x[x < 0]))
    n = len(y)
    if k >= n or k < 10:
        return np.nan
    top = y[n - k:]
    thresh = y[n - k - 1]
    xi = float(np.mean(np.log(top / thresh)))
    return 1.0 / xi if xi > 0 else np.nan


def hill_alpha_path(x: np.ndarray, tail: str = "upper", k_min: int = 20, k_max: int | None = None) -> dict:
    """Vectorized Hill alpha-hat for every k in [k_min, k_max] at once (a
    fast, O(n) equivalent of calling hill_estimator(x, k, tail) once per k -
    that function alone would cost O(n log n) per call, i.e. O(n^2 log n)
    across a full k-grid, far too slow at 35k observations).

    Same underlying formula as hill_estimator: sort the nonzero one-sided
    values ascending (y), let L = log(y). For the top k values,
    xi(k) = mean(L[n-k:n]) - L[n-k-1]. mean(L[n-k:n]) is a suffix mean,
    computed for every k at once via a reversed cumulative sum.

    Returns {"k": array, "alpha": array} with alpha[i] = NaN wherever the
    same guards as hill_estimator would reject that k (k>=n, k<10, xi<=0).
    Verified in run_phase1_tails.py to agree with hill_estimator at spot-
    checked k values before being trusted for the Hill plot.
    """
    x = x[np.isfinite(x)]
    y = np.sort(np.abs(x[x > 0]) if tail == "upper" else np.abs(x[x < 0]))
    n = len(y)
    if k_max is None:
        k_max = max(k_min, n // 10)
    k_max = min(k_max, n - 1)
    if k_max < k_min:
        return {"k": np.array([], dtype=int), "alpha": np.array([])}

    L = np.log(y, out=np.full_like(y, -np.inf, dtype=float), where=(y > 0))
    ks = np.arange(k_min, k_max + 1)
    # suffix sum of top-k logs, for every k in ks at once: reverse L, cumsum,
    # cumsum[k-1] = sum of the k largest elements' logs.
    cumsum_from_top = np.cumsum(L[::-1])
    top_sum = cumsum_from_top[ks - 1]
    suffix_mean = top_sum / ks
    thresh_log = L[n - ks - 1]
    xi = suffix_mean - thresh_log
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.where((xi > 0) & (ks < n) & (ks >= 10), 1.0 / xi, np.nan)
    return {"k": ks, "alpha": alpha}


def find_hill_plateau(alpha: np.ndarray, ks: np.ndarray, window: int = 50, rel_tol: float = 0.10) -> dict:
    """Read a Hill plot for a stable plateau, honestly.

    Heuristic (documented, not a universal statistical result): slide a
    window of `window` consecutive k-values across the alpha-hat path;
    for each window compute (max-min)/median as a relative-spread score.
    The plateau is the widest contiguous run of windows whose score is
    below rel_tol, read left-to-right; ties broken by lowest average score.
    If no window anywhere satisfies rel_tol, no plateau is reported - callers
    must treat every downstream tail-index claim at that interval/tail as
    provisional (NEXT_RUN_PROMPT.md's own tripwire for this case).
    """
    valid = np.isfinite(alpha)
    if valid.sum() < window:
        return {"found": False, "reason": "insufficient finite alpha estimates"}
    a, k = alpha[valid], ks[valid]
    n = len(a)
    scores = np.full(n - window + 1, np.nan)
    for i in range(n - window + 1):
        seg = a[i : i + window]
        med = np.median(seg)
        if med > 0:
            scores[i] = (seg.max() - seg.min()) / med
    stable = np.where(scores < rel_tol)[0]
    if len(stable) == 0:
        return {"found": False, "reason": f"no {window}-wide window has relative spread < {rel_tol}"}
    # widest contiguous run of stable window-start-indices
    runs, run_start = [], stable[0]
    for i in range(1, len(stable)):
        if stable[i] != stable[i - 1] + 1:
            runs.append((run_start, stable[i - 1]))
            run_start = stable[i]
    runs.append((run_start, stable[-1]))
    best_run = max(runs, key=lambda r: r[1] - r[0])
    k_lo, k_hi = k[best_run[0]], k[best_run[1] + window - 1]
    plateau_mask = (k >= k_lo) & (k <= k_hi)
    return {
        "found": True,
        "k_lo": int(k_lo), "k_hi": int(k_hi),
        "alpha_median": float(np.median(a[plateau_mask])),
        "alpha_min": float(np.min(a[plateau_mask])),
        "alpha_max": float(np.max(a[plateau_mask])),
        "k_chosen": int((k_lo + k_hi) // 2),
    }
