"""Shared synthetic fixtures for the notebook-014 port-fidelity check
(Phase 0). Imported by both `fidelity_source_14.py` (runs in ../ultron's
`ultron_finance` venv, via `sys.path.insert` to this file's absolute path)
and `fidelity_port_14.py` (runs in this repo's venv) -- this module only
uses numpy/pandas, present in both environments, so the fixture
construction is identical by import rather than by copy-paste.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def macro_inputs(n: int = 700):
    idx = pd.date_range("2016-01-01", periods=n, freq="B")
    close = pd.Series(100 * np.exp(np.cumsum(np.sin(np.arange(n) / 17) / 100)), index=idx)
    ohlcv = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close}
    )
    macro = pd.DataFrame(
        {
            "VIXCLS": 20 + np.sin(np.arange(n) / 13),
            "T10Y2Y": np.sin(np.arange(n) / 17),
            "T10Y3M": np.cos(np.arange(n) / 17),
            "BAMLH0A0HYM2": 4 + np.sin(np.arange(n) / 11),
            "BAMLC0A0CM": 1.5 + np.cos(np.arange(n) / 11),
            "DFF": np.arange(n) / 500,
        },
        index=idx,
    )
    cot = pd.DataFrame({"noncomm_net_pct_oi": np.sin(np.arange(n) / 23)}, index=idx)
    return ohlcv, macro, cot


def commodity_inputs(n: int = 700):
    idx = pd.date_range("2016-01-01", periods=n, freq="B")
    close = pd.Series(100.0 + np.arange(n) * 0.013 + 3 * np.sin(np.arange(n) / 31), index=idx)
    ohlcv = pd.DataFrame(
        {"open": close, "high": close * 1.008, "low": close * 0.992, "close": close}
    )
    front = close * (1 + 0.01 * np.sin(np.arange(n) / 41))
    deferred = close * (1 + 0.015 * np.sin(np.arange(n) / 41 + 0.6))
    third = close * (1 + 0.02 * np.sin(np.arange(n) / 41 + 1.1))
    curve = pd.DataFrame(
        {
            "close_f1": front,
            "dte_f1": 30.0,
            "close_f2": deferred,
            "dte_f2": 60.0,
            "close_f3": third,
            "dte_f3": 90.0,
        },
        index=idx,
    )
    return ohlcv, curve
