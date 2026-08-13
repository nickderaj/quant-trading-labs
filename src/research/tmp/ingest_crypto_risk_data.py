"""Builds `src/risk/data/crypto/{SYMBOL}.parquet` -- the crypto-panel
counterpart of `risk.ingest`'s per-product output -- from the raw daily
klines already cached at `src/research/cache/{SYMBOL}-klines-1d-*.parquet`.

Crypto perpetuals have no contract roll (unlike the futures curves
`risk.ingest` builds), so this is deliberately much simpler than
`risk.ingest.refresh()`: no roll-adjustment, no multi-contract stitching,
just `date` + `log_return` from the spot/perp close, which is everything
`risk.serve._load_product_curve`/`_product_snapshot` actually read from a
curve file (`close_f1` is carried along for parity with the futures schema
and dashboard hover text, but nothing computes off it here).

Usage: `uv run python src/research/tmp/ingest_crypto_risk_data.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import polars as pl

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "BNBUSDT", "XRPUSDT"]
CACHE_DIR = Path("src/research/cache")
OUT_DIR = Path("src/risk/data/crypto")

# The consolidated multi-year file per symbol (vs. the monthly-chunked ones
# also present in the cache) -- longest single continuous range available.
KLINES_GLOB = "{symbol}-klines-1d-*-*.parquet"


def _find_consolidated_file(symbol: str) -> Path:
    candidates = sorted(CACHE_DIR.glob(KLINES_GLOB.format(symbol=symbol)))
    # prefer the file covering the longest date range (widest start-end span)
    if not candidates:
        raise FileNotFoundError(f"no klines cache found for {symbol!r} in {CACHE_DIR}")
    return max(candidates, key=lambda p: p.stat().st_size)


def build_curve(symbol: str) -> pl.DataFrame:
    path = _find_consolidated_file(symbol)
    df = pl.read_parquet(path).sort("datetime")
    close = df["close"].to_numpy().astype(float)
    log_return = np.full(len(close), np.nan)
    log_return[1:] = np.log(close[1:] / close[:-1])
    return pl.DataFrame(
        {
            "date": df["datetime"].cast(pl.Date),
            "close_f1": close,
            "log_return": log_return,
        }
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for symbol in SYMBOLS:
        curve = build_curve(symbol)
        out_path = OUT_DIR / f"{symbol}.parquet"
        curve.write_parquet(out_path)
        n_finite = int(np.isfinite(curve["log_return"].to_numpy()).sum())
        print(
            f"wrote {out_path} ({len(curve)} rows, {n_finite} finite returns, "
            f"{curve['date'][0]} -> {curve['date'][-1]})"
        )


if __name__ == "__main__":
    main()
