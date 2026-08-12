"""The productionised commodity risk engine -- the only module a production
caller should import from. Every other submodule (`risk.model`,
`risk.portfolio`, `risk.hygiene`, `risk.densities`, `risk.calibration`,
`risk.families`, `risk.ingest`, `risk.serve`) is implementation detail.

Ports `src/research/tmp/commod_lib8.py`'s Phase 7 risk engine -- certified by
`src/results/008_commodity_tails_and_risk.md` (Gate RE: 15/16 development,
14/16 holdout 1% VaR coverage; Gate CE: normal-model ES rejected 15/16 dev,
11/16 holdout; Gate CT: Hill alpha < 5, 16/16) -- into durable, tested,
monitorable software. See `docs/10-risk-engine.md` for the full operator
document, **including the validated envelope this module must not be used
outside of**: 16 daily commodity/equity-index futures (`family_map_v1.json`),
nothing else.

**No alpha, no positions, no Sharpe, no backtest, no equity curve.** Every
function below answers "how much could this position lose," never "should I
hold it" (NEXT_PROMPT.md sec 13, on evidence: risk-gating was tested twice
and hurt net Sharpe both times). `size()` returns a notional, never a
direction or a signal.

Design points preserved rather than tidied away (sec 8.2), each because a
notebook 008 finding depends on it staying visible, not hidden behind a
convenient default:

1. `sigma_t` is always caller-supplied to `var`/`es` (never model-internal
   state) -- a static full-sample VaR failed OOS coverage in Phase 7;
   `ewma_vol` is the validated, causal source, but the call stays explicit.
2. `horizon` scaling is `sqrt(horizon)`, an i.i.d. assumption documented on
   `RiskModel.var`/`es` themselves, not silently assumed.
3. `portfolio()` reports all three dependence modes together when asked for
   all three; if a single default is needed, it is `"empirical"`, never
   `"gaussian"` (the Gaussian copula has zero asymptotic tail dependence by
   construction and understates joint tail risk).
4. `size()` is `risk_budget / var(...)`, nothing else -- no direction, no
   position, no strategy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from risk.calibration import CalibrationMonitor, CalibrationStatus
from risk.families import load_family_map
from risk.hygiene import assert_not_holdout, assert_risk_inputs
from risk.ingest import IngestReport
from risk.ingest import refresh as _refresh
from risk.model import RiskModel, StressResult, ewma_vol, fit_risk_model
from risk.portfolio import PortfolioRisk, portfolio_risk
from risk.serve import build_snapshot as _build_snapshot

__all__ = [
    "CalibrationStatus",
    "IngestReport",
    "PortfolioRisk",
    "RiskModel",
    "StressResult",
    "es",
    "ewma_vol",
    "fit",
    "monitor",
    "portfolio",
    "refresh",
    "size",
    "snapshot",
    "stress",
    "var",
]


def fit(product: str, returns_frame: pl.DataFrame) -> RiskModel:
    """Give me a fitted model for this product, from contract-checked
    inputs. `returns_frame` must be `hygiene.build_risk_inputs`'s output
    (or something carrying its provenance stamp) -- `assert_risk_inputs`
    is enforced here, at the public boundary, per NEXT_PROMPT.md sec 4.1:
    "fit_risk_model should refuse to fit a frame that did not come through
    build_risk_inputs." The family is read from the frozen `family_map_v1`
    (never guessed); a product outside its validated envelope raises
    (`risk.families.UnseenProductError`) rather than silently defaulting to
    a family. Also refuses (`risk.hygiene.HoldoutLeakError`) a frame whose
    dates extend past the spent futures holdout boundary (sec 2 ground rule
    1, sec 12) -- the fitting path must never re-spend it, even though the
    ingestion/dashboard path is explicitly allowed to see current dates
    (sec 7.4).
    """
    assert_not_holdout(returns_frame)
    assert_risk_inputs(returns_frame)
    family_map = load_family_map("v1")
    family = family_map.family_for(product)
    returns = returns_frame["log_return"].to_numpy()
    model = fit_risk_model(returns, product, family)
    if model is None:
        raise ValueError(
            f"fit_risk_model returned None for {product!r} (family={family!r}) -- "
            "fewer than 100 finite observations, or the fit itself failed"
        )
    return model


def var(model: RiskModel, alpha: float, sigma_t: float, horizon: int = 1) -> float:
    """What is the alpha-VaR today, at today's volatility `sigma_t` (e.g.
    from `ewma_vol`)?"""
    return model.var_conditional(alpha, sigma_t=sigma_t, horizon=horizon)


def es(model: RiskModel, alpha: float, sigma_t: float, horizon: int = 1) -> float:
    """What is the alpha-ES today, at today's volatility `sigma_t`?"""
    return model.es_conditional(alpha, sigma_t=sigma_t, horizon=horizon)


def portfolio(
    models: dict[str, RiskModel],
    weights: dict[str, float],
    dependence: str = "empirical",
    historical_returns: dict[str, np.ndarray] | None = None,
    n_sims: int = 20000,
    t_df: float = 5.0,
    seed: int = 0,
) -> PortfolioRisk:
    """What is the book's VaR/ES under `dependence`? Default is
    `"empirical"`, never `"gaussian"` (sec 8.2.3) -- call this three times,
    once per mode, to get the full side-by-side comparison the dashboard
    shows; do not rely on a single call's default standing in for that
    comparison."""
    return portfolio_risk(
        models,
        weights,
        dependence=dependence,
        historical_returns=historical_returns,
        n_sims=n_sims,
        t_df=t_df,
        seed=seed,
    )


def stress(model: RiskModel, scenario_returns: np.ndarray) -> StressResult:
    """What would this position have done in a named historical event?
    `scenario_returns` is that event's realized log-return path for this
    product (e.g. `risk.serve.NAMED_EVENTS` + the product's own ingested
    returns, sliced to the event window)."""
    return model.stress(scenario_returns)


def monitor(
    product: str,
    model: RiskModel,
    returns: np.ndarray,
    sigma_t: np.ndarray,
    **kwargs: Any,
) -> CalibrationStatus:
    """Is the model still calibrated? Thin wrapper over
    `CalibrationMonitor().evaluate(...)` -- construct a `CalibrationMonitor`
    directly (and use `evaluate_batch`/`evaluate_batch_from_hits`) for
    BH-corrected, multi-product monitoring; this convenience wrapper is for
    a single product, uncorrected, one-off check."""
    return CalibrationMonitor().evaluate(product, model, returns, sigma_t, **kwargs)


def size(
    model: RiskModel, alpha: float, sigma_t: float, risk_budget: float, horizon: int = 1
) -> float:
    """What notional consumes exactly this much risk budget? `risk_budget /
    var(...)`, units of risk_budget per unit of VaR -- a risk-sizing
    utility, not a strategy: no direction, no sign, no position (sec
    8.2.4). 015 closed the directional question and nothing here reopens
    it."""
    v = var(model, alpha, sigma_t, horizon=horizon)
    if not np.isfinite(v) or v <= 0:
        raise ValueError(
            f"var(...) must be finite and positive to size against, got {v}"
        )
    return risk_budget / v


def refresh(
    products: list[str] | None = None,
    as_of: str | None = None,
    **kwargs: Any,
) -> IngestReport:
    """Pull the latest cleaned inputs. Thin wrapper over
    `risk.ingest.refresh`."""
    return _refresh(products=products, as_of=as_of, **kwargs)


def snapshot(as_of: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """Everything the dashboard needs, in one document. Thin wrapper over
    `risk.serve.build_snapshot`."""
    return _build_snapshot(as_of=as_of, **kwargs)
