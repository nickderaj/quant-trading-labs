"""Parquet adapters: the sole polars-to-pandas conversion boundary for the
regime engine.

The engine (``src/regime/``) is pandas end to end, ported verbatim from a
production codebase that never touches polars. This repo's data layer is
polars. Every function here reads a polars parquet file and returns a
pandas object with a sorted, tz-naive `DatetimeIndex` -- nothing downstream
of this module should call `.to_pandas()` again.

Data sources (all already present in this repo, nothing is fetched):
- bars:  ``src/research/data/market/yfinance/daily/<SYMBOL>.parquet``
- FRED:  ``src/research/data/market/fred/<SERIES>.parquet``
- COT:   ``src/research/data/market/cot/067651.parquet`` (crude only)
- curve: ``src/research/data/market/research/{cl,gc,hg,ng,si}_curve.parquet``
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT = _REPO_ROOT / "src" / "research" / "data" / "market"

BARS_DIR = _DATA_ROOT / "yfinance" / "daily"
FRED_DIR = _DATA_ROOT / "fred"
COT_PATH = _DATA_ROOT / "cot" / "067651.parquet"
CURVE_DIR = _DATA_ROOT / "research"

# Exactly ultron_finance.data.fred.FRED_V1_SERIES (NEXT_PROMPT.md Sec3.2).
FRED_SERIES = (
    "VIXCLS",
    "T10Y2Y",
    "T10Y3M",
    "BAMLH0A0HYM2",
    "BAMLC0A0CM",
    "DFF",
    "DGS2",
    "DGS10",
)

# yfinance/CME symbol -> curve parquet stem. Exactly the five symbols the
# source's DATABENTO_CURVE_SYMBOLS lists (NEXT_PROMPT.md Sec3.4); the other
# fifteen commodity legs have no curve file and get curve=None.
CURVE_SYMBOLS: dict[str, str] = {
    "CL=F": "cl",
    "NG=F": "ng",
    "GC=F": "gc",
    "SI=F": "si",
    "HG=F": "hg",
}

# The one COT file this repo has: 067651 = CRUDE OIL, LIGHT SWEET. Only
# wired into the oil_products basket, behind an explicit opt-in flag
# (NEXT_PROMPT.md Sec3.3) -- the default run stays bug-for-bug identical to
# production, which has no COT data for any of these commodity legs.
COT_SYMBOL = "CL=F"


def _to_pandas_indexed(frame: pl.DataFrame, time_column: str) -> pd.DataFrame:
    pdf = frame.to_pandas()
    pdf[time_column] = pd.to_datetime(pdf[time_column]).dt.tz_localize(None)
    pdf = pdf.sort_values(time_column).set_index(time_column)
    pdf.index.name = None
    return pdf


def load_bars(symbol: str) -> pd.DataFrame:
    """Full daily OHLCV history for ``symbol`` (e.g. ``"ES=F"``)."""
    path = BARS_DIR / f"{symbol}.parquet"
    frame = pl.read_parquet(path).filter(pl.col("interval") == "1d")
    pdf = _to_pandas_indexed(frame, "timestamp")
    return pdf[["open", "high", "low", "close", "volume"]]


def load_fred_frame(series: tuple[str, ...] = FRED_SERIES) -> pd.DataFrame:
    """Wide FRED frame, one column per series, indexed by publication date
    (midnight-indexed, not yet aligned to any bar calendar -- align via
    ``regime.align.align_frame_to_daily`` at the call site, exactly as
    production's ``builder.py`` does)."""
    columns: dict[str, pd.Series] = {}
    for name in series:
        frame = pl.read_parquet(FRED_DIR / f"{name}.parquet")
        pdf = _to_pandas_indexed(frame, "date")
        columns[name] = pdf[name]
    return pd.DataFrame(columns)


def load_cot_raw() -> pd.DataFrame:
    """Raw CFTC legacy report rows for crude oil (067651), indexed by
    ``report_date``. Reuses the parquet this repo already loads in
    ``src/research/tmp/run_phase_4_10a_cot.py``."""
    frame = pl.read_parquet(COT_PATH)
    return _to_pandas_indexed(frame, "report_date")


def net_positioning(df: pd.DataFrame) -> pd.DataFrame:
    """Add commercial/non-commercial net positioning columns.

    Ported verbatim from ``ultron_finance.data.cot.CotClient.net_positioning``
    (``../ultron/libs/finance/src/ultron_finance/data/cot.py:132``).
    """
    result = df.copy()
    result["noncomm_net"] = (
        result["noncomm_positions_long_all"] - result["noncomm_positions_short_all"]
    )
    result["comm_net"] = result["comm_positions_long_all"] - result["comm_positions_short_all"]
    open_interest = result["open_interest_all"].replace(0, float("nan"))
    result["noncomm_net_pct_oi"] = result["noncomm_net"] / open_interest
    result["comm_net_pct_oi"] = result["comm_net"] / open_interest
    return result


def load_curve(symbol: str) -> pd.DataFrame | None:
    """Futures curve for ``symbol`` if this repo has one, else ``None``.

    Schema ``[close_f1, dte_f1, close_f2, dte_f2, close_f3, dte_f3]``,
    indexed by date -- three legs only, no f12 (NEXT_PROMPT.md Sec3.4
    landmine #1: callers must not rely on ``curve_slope``'s default
    ``far="close_f12"``; ``regime.dimensions.term_structure`` already pins
    ``far="close_f3"`` explicitly).
    """
    stem = CURVE_SYMBOLS.get(symbol)
    if stem is None:
        return None
    path = CURVE_DIR / f"{stem}_curve.parquet"
    frame = pl.read_parquet(path)
    return _to_pandas_indexed(frame, "date")
