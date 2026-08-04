from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regime import prediction as pred
from regime.engine import RegimeEngine, RegimeInputs


def _macro_inputs(n: int = 700, phase: float = 0.0) -> RegimeInputs:
    idx = pd.date_range("2016-01-01", periods=n, freq="B")
    close = pd.Series(100 * np.exp(np.cumsum(np.sin(np.arange(n) / 17 + phase) / 100)), index=idx)
    ohlcv = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close})
    macro = pd.DataFrame(
        {
            "VIXCLS": 20 + np.sin(np.arange(n) / 13 + phase),
            "T10Y2Y": np.sin(np.arange(n) / 17 + phase),
            "T10Y3M": np.cos(np.arange(n) / 17 + phase),
            "BAMLH0A0HYM2": 4 + np.sin(np.arange(n) / 11 + phase),
            "BAMLC0A0CM": 1.5 + np.cos(np.arange(n) / 11 + phase),
            "DFF": np.arange(n) / 500,
        },
        index=idx,
    )
    cot = pd.DataFrame({"noncomm_net_pct_oi": np.sin(np.arange(n) / 23 + phase)}, index=idx)
    return RegimeInputs(ohlcv=ohlcv, macro=macro, cot=cot)


def test_horizon_to_days() -> None:
    assert pred.horizon_to_days("1w") == 5
    assert pred.horizon_to_days("1m") == 21
    assert pred.horizon_to_days("10y") == 2520
    assert pred.horizon_to_days(30) == 30
    for bad in ("", "5x", "m", "-1d"):
        try:
            pred.horizon_to_days(bad)
            raise AssertionError(f"expected failure for {bad!r}")
        except ValueError:
            pass


def test_default_weights_reproduce_engine_score() -> None:
    result = RegimeEngine.from_default("macro_default").detect(_macro_inputs())
    scaled = pred.scaled_indicator_frame(result, "risk")
    dim = pred._dimension_config(result, "risk")
    score = pred.forecast_scores(scaled, dim, pred.base_weights(result, "risk"))
    pd.testing.assert_series_equal(
        score.dropna(), result.scores["risk"].dropna(), check_names=False, atol=1e-12
    )


def test_scaled_frame_roundtrip() -> None:
    result = RegimeEngine.from_default("macro_default").detect(_macro_inputs())
    scaled = pred.scaled_indicator_frame(result, "risk")
    weights = pred.base_weights(result, "risk")
    for name, w in weights.items():
        rebuilt = scaled[name] * w
        pd.testing.assert_series_equal(
            rebuilt.dropna(),
            result.contributions[f"risk.{name}"].dropna(),
            check_names=False,
            atol=1e-12,
        )


def test_forecast_truncation_invariance() -> None:
    engine = RegimeEngine.from_default("macro_default")
    inputs = _macro_inputs()
    cfg = pred.ForecastConfig(
        dimension="risk",
        horizon="1m",
        weights={i.name: i.weight for i in engine.config.dimensions[0].indicators},
        markov_alpha=0.5,
        markov_min_history=60,
    )
    full = pred.forecast_from_result(engine.detect(inputs), cfg)
    assert inputs.macro is not None
    assert inputs.cot is not None
    for bars in (1, 5, 21):
        trunc = RegimeInputs(
            ohlcv=inputs.ohlcv.iloc[:-bars],
            macro=inputs.macro.iloc[:-bars],
            cot=inputs.cot.iloc[:-bars],
        )
        part = pred.forecast_from_result(engine.detect(trunc), cfg)
        pd.testing.assert_series_equal(full.score.iloc[:-bars], part.score, atol=1e-9)
        pd.testing.assert_series_equal(full.labels.iloc[:-bars], part.labels)
        pd.testing.assert_frame_equal(full.probs.iloc[:-bars], part.probs, atol=1e-9)


def test_markov_forecast_matches_predict_next_final_bar() -> None:
    from regime.transitions import predict_next, transition_matrix

    labels = pd.Series(list("aabbbccaab" * 40))
    states = ["a", "b", "c"]
    mf = pred.markov_forecast(labels, horizon=3, states=states, min_history=1)
    expected = (
        predict_next(str(labels.iloc[-1]), transition_matrix(labels), 3).reindex(states).fillna(0.0)
    )
    np.testing.assert_allclose(
        mf.iloc[-1].to_numpy(dtype=float), expected.to_numpy(dtype=float), atol=1e-9
    )


def test_prob_blend_endpoints_and_sum_to_one() -> None:
    labels = pd.Series(list("aabbbccaab" * 40))
    idx = pd.RangeIndex(len(labels))
    labels.index = idx
    states = ["a", "b", "c"]
    onehot = pred._one_hot(labels, states)
    row = onehot.dropna().iloc[0]
    assert row.sum() == 1.0 and row.max() == 1.0
    mf = pred.markov_forecast(labels, 2, states, min_history=10).dropna()
    assert np.allclose(mf.sum(axis=1), 1.0)


def test_nonzero_shift_rejected() -> None:
    engine = RegimeEngine.from_default("macro_default")
    result = engine.detect(_macro_inputs())
    object.__setattr__(result.config, "shift", 1)
    try:
        pred.scaled_indicator_frame(result, "risk")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_basket_scaled_frame() -> None:
    engine = RegimeEngine.from_default("macro_default")
    result_a = engine.detect(_macro_inputs())
    result_b = engine.detect(_macro_inputs(phase=0.7))
    results = {"A": result_a, "B": result_b}

    basket = pred.basket_scaled_frame(results, "risk")
    frame_a = pred.scaled_indicator_frame(result_a, "risk")
    frame_b = pred.scaled_indicator_frame(result_b, "risk")

    for col in ("macro.vix", "macro.yield_curve"):
        expected = pd.concat([frame_a[col], frame_b[col]], axis=1).mean(axis=1, skipna=True)
        pd.testing.assert_series_equal(
            basket[col].dropna(), expected.dropna(), check_names=False, atol=1e-12
        )
