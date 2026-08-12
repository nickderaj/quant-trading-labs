"""Gate NL (NEXT_PROMPT.md sec 8.3, sec 10): wires
`regime.evaluation.no_lookahead_check` -- 014's causality gate, run as a hard
gate across 27 symbols at four truncations -- over the `ewma_vol` ->
`var_conditional` path. This is the highest-value single import from
`regime`: it strengthens the engine's no-lookahead claim beyond the bespoke
unit test in `tests/test_risk_model.py`'s `TestEwmaVol` by reusing the exact
harness that already proved this property for a different engine.

`no_lookahead_check` is generic over anything shaped like a `RegimeEngine`
(a `.detect(inputs) -> RegimeResult`, `RegimeInputs` carrying a
row-indexed `.ohlcv` frame). `_EwmaVarEngine` below is a thin adapter, not a
new regime dimension -- it exists only so the *existing, already-validated*
harness can be reused verbatim rather than reimplemented (NEXT_PROMPT.md sec
12: "Do not modify src/regime/... import from it, don't edit it").

Per NEXT_PROMPT.md sec 8.2.1, `sigma_t` conditioning is deliberately
caller-supplied, not model-internal state -- so what this gate actually
certifies is the *composition* `ewma_vol(returns) -> var_conditional(alpha,
sigma_t)`: row t's output must depend only on returns[:t], never
returns[t:]. The `RiskModel` itself (its fitted shape/mean/std) is frozen
before this check runs and is not itself a lookahead concern -- a
full-sample fit is a separate, already-documented and accepted limitation
(sec 8.2.2's sqrt(horizon)-scaling caveat is the same kind of documented,
not-hidden assumption).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime.config import RegimeConfig
from regime.engine import RegimeEngine, RegimeInputs, RegimeResult
from regime.evaluation import no_lookahead_check
from risk.calibration import _conditional_quantile_series
from risk.model import RiskModel, ewma_vol

__all__ = ["check_no_lookahead"]

_DUMMY_CONFIG = RegimeConfig(
    version="v0",
    name="risk_ewma_var_no_lookahead_adapter",
    asset_class="commodity_futures",
    dimensions=[],
)


class _EwmaVarEngine(RegimeEngine):
    """Adapts `ewma_vol` -> `RiskModel.var_conditional` to the
    `RegimeEngine.detect(inputs) -> RegimeResult` shape
    `no_lookahead_check` expects (subclassed, not duck-typed, so the type
    checker sees this really does satisfy `RegimeEngine`'s interface --
    `detect` is fully overridden, `RegimeEngine.__init__`'s config is a
    fixed placeholder never consulted by this override). `model` is frozen
    (already fitted) before any check runs; only the sigma_t/VaR *path* is
    under test."""

    def __init__(self, model: RiskModel, alpha: float):
        super().__init__(_DUMMY_CONFIG)
        self.model = model
        self.alpha = alpha

    def detect(self, inputs: RegimeInputs) -> RegimeResult:
        # NOTE: uses risk.calibration's hoisted-quantile helper, not a
        # naive `[model.var_conditional(alpha, sigma_t=s) for s in sigma]`
        # loop -- for the standardized families (NIG especially, whose ppf
        # root-finds via brentq over a quad-integrated CDF per call) that
        # naive loop recomputes the same *sigma_t-independent* unconditional
        # quantile thousands of times, and no_lookahead_check calls detect()
        # five times per product. See calibration.py's
        # _conditional_quantile_series docstring for the full story (the
        # same bug, found and fixed there first).
        returns = inputs.ohlcv["log_return"].to_numpy()
        sigma = ewma_vol(returns)
        signed_q = _conditional_quantile_series(self.model, sigma, self.alpha)
        var = -signed_q  # var_conditional's sign convention: positive VaR magnitude
        idx = inputs.ohlcv.index
        frame = pd.DataFrame({"sigma_t": sigma, "var": var}, index=idx)
        empty = pd.DataFrame(index=idx)
        return RegimeResult(
            scores=frame,
            labels=frame,
            indicators=empty,
            contributions=empty,
            config=_DUMMY_CONFIG,
        )


def check_no_lookahead(
    model: RiskModel,
    returns: np.ndarray,
    dates: np.ndarray,
    alpha: float = 0.01,
    truncations: tuple[int, ...] = (1, 5, 21, 63),
) -> bool:
    """True iff every truncated re-detection of the `ewma_vol` ->
    `var_conditional` path agrees bit-identically with the full-history
    detection over the retained rows (NEXT_PROMPT.md sec 8.3/10, Gate NL).
    `dates` becomes the row index `no_lookahead_check` slices against.
    """
    engine = _EwmaVarEngine(model, alpha)
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    ohlcv = pd.DataFrame({"log_return": returns}, index=idx)
    inputs = RegimeInputs(ohlcv=ohlcv, curve=None, macro=None, cot=None)
    return no_lookahead_check(engine, inputs, truncations=truncations)
