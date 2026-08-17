"""Notebook 017, disclosed pilot (NEXT_PROMPT sec 3).

Run: uv run python scratch/017/pilot_dsr.py

Four cells, M=400, N=18, T=3840, Gaussian returns, seed 7. Indicative only --
MC SE is about +-0.011 at p=0.05. Phase 2 must reproduce this at M >= 20000
across the full rho axis; if Phase 2 disagrees, Phase 2 wins.
"""

import numpy as np
from scipy.stats import norm

EM = 0.5772156649


def emax(disp, N):
    if N <= 1:
        return 0.0
    return disp * ((1 - EM) * norm.ppf(1 - 1 / N) + EM * norm.ppf(1 - 1 / (N * np.e)))


def dsr(sr, N, T, disp, skew=0.0, kurt=3.0):
    se = np.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr**2) / (T - 1))
    return norm.cdf((sr - emax(disp, N)) / se)


def cell(N, T, rho, M, seed):
    """Equicorrelated null: every trial has true Sharpe 0. Returns (V0 FPR, V1 FPR)."""
    rng = np.random.default_rng(seed)
    cur, fix = np.empty(M), np.empty(M)
    i = 0
    while i < M:  # chunked: (M, T, N) does not fit whole
        m = min(50, M - i)
        zc = rng.standard_normal((m, T, 1))
        zi = rng.standard_normal((m, T, N))
        x = np.sqrt(rho) * zc + np.sqrt(1 - rho) * zi
        srs = x.mean(axis=1) / x.std(axis=1, ddof=1)  # (m, N)
        best = srs.max(axis=1)
        se1 = np.sqrt(1.0 / (T - 1))  # V0 scale
        disp_obs = srs.std(axis=1, ddof=1)  # V1 scale
        for j in range(m):
            cur[i + j] = dsr(best[j], N, T, se1)
            fix[i + j] = dsr(best[j], N, T, disp_obs[j])
        i += m
    return (cur > 0.95).mean(), (fix > 0.95).mean()


if __name__ == "__main__":
    print(f"{'rho':>5}  {'V0 (current)':>12}  {'V1 (dispersion)':>15}")
    for rho in (0.0, 0.5, 0.9, 0.99):
        v0, v1 = cell(18, 3840, rho, 400, 7)
        print(f"{rho:>5}  {v0:>12.4f}  {v1:>15.4f}")
