"""Unit tests for `src/risk/portfolio.py`: `portfolio_risk` under the three
dependence assumptions, and the tail-dependence helpers.

Split from `tests/test_commod_lib8.py` (NEXT_PROMPT.md sec 3.6) when the
underlying functions were promoted to `src/risk/portfolio.py`. New tests
added per sec 3.6's list: determinism (same seed -> bit-identical output)
and the copula tail-dependence ordering (gaussian <= t <~ empirical).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from risk.model import fit_risk_model
from risk.portfolio import (
    empirical_lower_tail_dependence,
    portfolio_risk,
    to_pseudo_uniform,
)

SEED = 0


def _two_asset_setup(corr=0.6, seed=SEED):
    rng = np.random.default_rng(seed)
    n = 3000
    z1 = rng.standard_normal(n)
    z2 = corr * z1 + np.sqrt(1 - corr**2) * rng.standard_normal(n)
    r1, r2 = 0.02 * z1, 0.02 * z2
    m1 = fit_risk_model(r1, "A", "normal")
    m2 = fit_risk_model(r2, "B", "normal")
    return {"A": m1, "B": m2}, {"A": r1, "B": r2}


class TestPortfolioRisk:
    def test_gaussian_copula_var_positive(self):
        models, hist = _two_asset_setup()
        result = portfolio_risk(
            models,
            {"A": 0.5, "B": 0.5},
            dependence="gaussian",
            historical_returns=hist,
            n_sims=5000,
            seed=0,
        )
        assert result["var_01"] > 0
        assert result["es_01"] >= result["var_01"]

    def test_empirical_dependence_runs(self):
        models, hist = _two_asset_setup()
        result = portfolio_risk(
            models,
            {"A": 0.5, "B": 0.5},
            dependence="empirical",
            historical_returns=hist,
            n_sims=5000,
            seed=0,
        )
        assert result["var_01"] > 0

    def test_t_copula_shows_more_tail_dependence_than_gaussian(self):
        # low correlation but fat-tailed joint -> t-copula should show
        # materially higher lower-tail dependence than Gaussian at the same
        # correlation, since Gaussian tail dependence -> 0 asymptotically.
        models, hist = _two_asset_setup(corr=0.3)
        gauss = portfolio_risk(
            models,
            {"A": 0.5, "B": 0.5},
            dependence="gaussian",
            historical_returns=hist,
            n_sims=20000,
            seed=0,
        )
        t_cop = portfolio_risk(
            models,
            {"A": 0.5, "B": 0.5},
            dependence="t",
            historical_returns=hist,
            t_df=3.0,
            n_sims=20000,
            seed=0,
        )
        gauss_td = gauss["lower_tail_dependence"]["A_B"]
        t_td = t_cop["lower_tail_dependence"]["A_B"]
        assert t_td > gauss_td

    def test_dependence_ordering_gaussian_le_t_le_empirical(self):
        # NEXT_PROMPT.md sec 3.6: the structural claim 008 rests on -- with
        # the same marginals and correlation, the Gaussian copula's estimated
        # lower-tail dependence must be lower than the t-copula's, which must
        # be lower than or comparable to the empirical. Uses a strongly
        # co-crashing synthetic joint (shared fat-tailed shock) so the
        # empirical tail dependence is unambiguously the highest of the three.
        rng = np.random.default_rng(SEED)
        n = 6000
        shock = rng.standard_t(3, n)
        r1 = 0.02 * shock + 0.005 * rng.standard_normal(n)
        r2 = 0.02 * shock + 0.005 * rng.standard_normal(n)
        m1 = fit_risk_model(r1, "A", "t")
        m2 = fit_risk_model(r2, "B", "t")
        models = {"A": m1, "B": m2}
        hist = {"A": r1, "B": r2}

        gauss = portfolio_risk(
            models,
            {"A": 0.5, "B": 0.5},
            dependence="gaussian",
            historical_returns=hist,
            n_sims=20000,
            seed=0,
        )
        t_cop = portfolio_risk(
            models,
            {"A": 0.5, "B": 0.5},
            dependence="t",
            historical_returns=hist,
            t_df=5.0,
            n_sims=20000,
            seed=0,
        )
        empirical = portfolio_risk(
            models,
            {"A": 0.5, "B": 0.5},
            dependence="empirical",
            historical_returns=hist,
            n_sims=20000,
            seed=0,
        )
        gauss_td = gauss["lower_tail_dependence"]["A_B"]
        t_td = t_cop["lower_tail_dependence"]["A_B"]
        emp_td = empirical["lower_tail_dependence"]["A_B"]
        assert gauss_td < t_td
        assert t_td <= emp_td + 0.05  # "lower than or comparable to"

    def test_determinism_same_seed_bit_identical(self):
        models, hist = _two_asset_setup()
        kwargs = {
            "dependence": "t",
            "historical_returns": hist,
            "n_sims": 8000,
            "t_df": 5.0,
            "seed": 0,
        }
        r1 = portfolio_risk(models, {"A": 0.5, "B": 0.5}, **kwargs)
        r2 = portfolio_risk(models, {"A": 0.5, "B": 0.5}, **kwargs)
        assert r1 == r2


class TestTailDependenceHelpers:
    def test_to_pseudo_uniform_is_in_unit_interval(self):
        rng = np.random.default_rng(SEED)
        x = rng.standard_normal(500)
        u = to_pseudo_uniform(x)
        assert u.min() > 0
        assert u.max() < 1

    def test_empirical_tail_dependence_high_for_comonotonic(self):
        import pytest

        rng = np.random.default_rng(SEED)
        x = rng.standard_normal(2000)
        u = to_pseudo_uniform(x)
        td = empirical_lower_tail_dependence(
            u, u, q=0.1
        )  # perfectly comonotonic with itself
        assert td == pytest.approx(1.0)

    def test_empirical_tail_dependence_low_for_independent(self):
        import pytest

        rng = np.random.default_rng(SEED)
        u1 = to_pseudo_uniform(rng.standard_normal(5000))
        u2 = to_pseudo_uniform(rng.standard_normal(5000))
        td = empirical_lower_tail_dependence(u1, u2, q=0.1)
        assert td == pytest.approx(0.1, abs=0.05)
