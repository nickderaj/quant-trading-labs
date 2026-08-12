"""Tests for `src/risk/_lookahead.py` (NEXT_PROMPT.md sec 8.3, gate NL).

Gate NL's real measurement (16/16 products, real data, four truncations) is
a slow, real-data job and lives in `src/research/tmp/run_risk_05_lookahead.py`;
this file is the "run in CI" half -- fast, synthetic-data coverage of
`check_no_lookahead` itself, including a negative control (a deliberately
lookahead-leaking engine) proving the check can actually fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from regime.engine import RegimeInputs, RegimeResult
from regime.evaluation import no_lookahead_check
from risk._lookahead import _DUMMY_CONFIG, check_no_lookahead
from risk.model import fit_risk_model

SEED = 0


def _synthetic_series(n: int = 500, seed: int = SEED):
    rng = np.random.default_rng(seed)
    ret = rng.standard_t(6, n) * 0.02
    dates = np.array(["2020-01-01"], dtype="datetime64[D]") + np.arange(n)
    return ret, dates


class TestCheckNoLookahead:
    def test_ewma_var_path_is_causal(self):
        ret, dates = _synthetic_series()
        model = fit_risk_model(ret, "TEST", "ged")
        assert model is not None
        assert check_no_lookahead(model, ret, dates, alpha=0.01) is True

    def test_passes_at_the_gate_nl_truncations(self):
        ret, dates = _synthetic_series(n=800)
        model = fit_risk_model(ret, "TEST", "normal")
        assert model is not None
        assert (
            check_no_lookahead(
                model, ret, dates, alpha=0.01, truncations=(1, 5, 21, 63)
            )
            is True
        )

    def test_a_lookahead_leaking_engine_is_caught(self):
        # Negative control: an engine that peeks at the *last* row of
        # whatever history it's given (a textbook lookahead bug) must be
        # rejected by no_lookahead_check -- proving the harness bites, not
        # just that our causal path happens to pass it.
        class _LeakyEngine:
            def detect(self, inputs: RegimeInputs) -> RegimeResult:
                idx = inputs.ohlcv.index
                ret = inputs.ohlcv["log_return"].to_numpy()
                # every row sees the FINAL value in the (possibly truncated)
                # series -- changes under truncation, the leak this test
                # exists to catch.
                leaked = np.full(len(ret), ret[-1] if len(ret) else np.nan)
                frame = pd.DataFrame({"leak": leaked}, index=idx)
                empty = pd.DataFrame(index=idx)
                return RegimeResult(
                    scores=frame,
                    labels=frame,
                    indicators=empty,
                    contributions=empty,
                    config=_DUMMY_CONFIG,
                )

        ret, dates = _synthetic_series(n=200)
        idx = pd.DatetimeIndex(pd.to_datetime(dates))
        inputs = RegimeInputs(
            ohlcv=pd.DataFrame({"log_return": ret}, index=idx),
            curve=None,
            macro=None,
            cot=None,
        )
        assert no_lookahead_check(_LeakyEngine(), inputs, truncations=(1, 5)) is False

    def test_raises_on_truncation_not_shorter_than_input(self):
        ret, dates = _synthetic_series(n=150)
        model = fit_risk_model(ret, "TEST", "normal")
        assert model is not None
        with pytest.raises(ValueError):
            check_no_lookahead(model, ret, dates, truncations=(1000,))
