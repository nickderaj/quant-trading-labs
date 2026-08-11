"""Portfolio-level risk: dependence-mode Monte Carlo, tail-dependence helpers.

Ported verbatim from `src/research/tmp/commod_lib8.py` (see
`docs/10-risk-engine.md`). `commod_lib8.py` re-imports these names and
re-exports them so notebook 008 and its tests keep passing unchanged against
the promoted code (NEXT_PROMPT.md sec 3.3).
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from scipy import stats as st

from risk.model import RiskModel

__all__ = [
    "PortfolioRisk",
    "empirical_lower_tail_dependence",
    "portfolio_risk",
    "to_pseudo_uniform",
]


class PortfolioRisk(TypedDict):
    """`portfolio_risk`'s return shape (NEXT_PROMPT.md sec 3.5): a caller
    needs to know `var_01`/`es_01`/`lower_tail_dependence` etc are always
    present, which a bare `dict` return type does not communicate."""

    dependence: str
    n_sims: int
    var_01: float
    var_05: float
    es_01: float
    es_05: float
    lower_tail_dependence: dict[str, float]


def empirical_lower_tail_dependence(
    u_i: np.ndarray, u_j: np.ndarray, q: float = 0.1
) -> float:
    """Empirical lower-tail dependence coefficient at threshold q:
    P(U_j <= q | U_i <= q), from rank-transformed (pseudo-uniform) marginals.
    """
    mask_i = u_i <= q
    if mask_i.sum() == 0:
        return float("nan")
    return float(np.mean(u_j[mask_i] <= q))


def to_pseudo_uniform(x: np.ndarray) -> np.ndarray:
    """Rank-transform to pseudo-uniform marginals (empirical CDF ranks in
    (0,1)), the standard first step before any copula fit or tail-dependence
    estimate."""
    order = np.argsort(np.argsort(x))
    return (order + 0.5) / len(x)


def portfolio_risk(
    models: dict[str, RiskModel],
    weights: dict[str, float],
    dependence: str = "empirical",
    historical_returns: dict[str, np.ndarray] | None = None,
    n_sims: int = 20000,
    t_df: float = 5.0,
    seed: int = 0,
) -> PortfolioRisk:
    """Portfolio-level VaR/ES via Monte Carlo, under one of three explicit
    dependence assumptions:

    - "empirical": bootstrap-resample historical JOINT return vectors
      (requires `historical_returns`, aligned by date/index across products)
      -- captures whatever dependence (incl. tail dependence) is actually in
      the data.
    - "gaussian": simulate a Gaussian copula from the empirical (pseudo-
      uniform) correlation matrix, transform each margin through its own
      RiskModel.ppf_from_u -- has ZERO asymptotic tail dependence by
      construction, regardless of the correlation used.
    - "t": simulate a Student-t copula (same correlation, `t_df` degrees of
      freedom) -- has positive tail dependence, growing as t_df falls.

    Reports portfolio VaR/ES at 1%/5%, plus empirical lower-tail dependence
    per pair for both the real (empirical-dependence) simulation and the
    Gaussian-copula simulation, so any understatement (sec 3.4) is a
    reported, quantified comparison, not an assertion.

    NEXT_PROMPT.md sec 8.2.3: the three dependence modes are a reported
    comparison, not a choice to be defaulted away -- report all three; if a
    caller needs one default, use "empirical", never "gaussian".
    """
    products = list(models.keys())
    n_assets = len(products)
    rng = np.random.default_rng(seed)
    w = np.array([weights[p] for p in products])

    if dependence == "empirical":
        if historical_returns is None:
            raise ValueError("empirical dependence requires historical_returns")
        mat = np.column_stack([historical_returns[p] for p in products])
        mat = mat[np.all(np.isfinite(mat), axis=1)]
        idx = rng.integers(0, len(mat), n_sims)
        sims = mat[idx]
    else:
        if historical_returns is not None:
            mat = np.column_stack([historical_returns[p] for p in products])
            mat = mat[np.all(np.isfinite(mat), axis=1)]
            pseudo = np.column_stack(
                [to_pseudo_uniform(mat[:, i]) for i in range(n_assets)]
            )
            corr = np.corrcoef(pseudo, rowvar=False)
        else:
            corr = np.eye(n_assets)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        chol = np.linalg.cholesky(corr + 1e-8 * np.eye(n_assets))
        if dependence == "gaussian":
            z = rng.standard_normal((n_sims, n_assets)) @ chol.T
            u = _norm_cdf(z)
        elif dependence == "t":
            g = rng.standard_normal((n_sims, n_assets)) @ chol.T
            chi2 = rng.chisquare(t_df, n_sims)
            t_draws = g / np.sqrt(chi2 / t_df)[:, None]
            u = _t_cdf(t_draws, t_df)
        else:
            raise ValueError(f"unknown dependence: {dependence!r}")
        sims = np.column_stack(
            [models[p].ppf_from_u(u[:, i]) for i, p in enumerate(products)]
        )

    port_ret = sims @ w
    var_01 = float(-np.percentile(port_ret, 1))
    var_05 = float(-np.percentile(port_ret, 5))
    es_01 = float(-np.mean(port_ret[port_ret <= np.percentile(port_ret, 1)]))
    es_05 = float(-np.mean(port_ret[port_ret <= np.percentile(port_ret, 5)]))

    tail_dep = {}
    pseudo_sims = np.column_stack(
        [to_pseudo_uniform(sims[:, i]) for i in range(n_assets)]
    )
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            key = f"{products[i]}_{products[j]}"
            tail_dep[key] = empirical_lower_tail_dependence(
                pseudo_sims[:, i], pseudo_sims[:, j]
            )

    return {
        "dependence": dependence,
        "n_sims": n_sims,
        "var_01": var_01,
        "var_05": var_05,
        "es_01": es_01,
        "es_05": es_05,
        "lower_tail_dependence": tail_dep,
    }


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return st.norm.cdf(z)


def _t_cdf(x: np.ndarray, df: float) -> np.ndarray:
    return st.t.cdf(x, df=df)
