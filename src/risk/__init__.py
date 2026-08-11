"""The productionised commodity risk engine.

Ports `src/research/tmp/commod_lib8.py`'s Phase 7 risk engine -- certified by
`src/results/008_commodity_tails_and_risk.md` (Gate RE: 15/16 development,
14/16 holdout 1% VaR coverage; Gate CE: normal-model ES rejected 15/16 dev,
11/16 holdout; Gate CT: Hill alpha < 5, 16/16) -- into durable, tested,
monitorable software. See `docs/10-risk-engine.md` for the full operator
document, including the validated envelope this module must not be used
outside of.

This is the only module a production caller should import from; internal
submodules (`risk.model`, `risk.portfolio`, `risk.hygiene`, `risk.densities`,
`risk.calibration`, `risk.families`, `risk.ingest`, `risk.serve`) are
implementation detail.

**No alpha, no positions, no Sharpe, no backtest, no equity curve.** This
module answers "how much could this position lose," never "should I hold
it" -- see NEXT_PROMPT.md sec 13 for why, on evidence, not assertion.
"""

from __future__ import annotations

from risk.model import RiskModel, ewma_vol, fit_risk_model
from risk.portfolio import PortfolioRisk, portfolio_risk

__all__ = [
    "PortfolioRisk",
    "RiskModel",
    "ewma_vol",
    "fit_risk_model",
    "portfolio_risk",
]
