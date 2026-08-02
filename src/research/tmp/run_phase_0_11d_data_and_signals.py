"""Notebook 11d Phase 0 -- load the crypto and equity universes, trim to the
dev window, build the regime gates, and sanity-check the breakout signal's
firing rate before any backtest runs (NEXT_PROMPT.md sec 7, sec 8 holdout
discipline: dev window only, 2025-01-01 onward untouched).
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import spread_lib11 as S11

import research

research.set_seed(0)

DEV_END = "2024-12-31"
CRYPTO_CACHE = "src/research/cache"
EQUITY_DIR = "src/research/data/market/yfinance/daily"

CRYPTO_SYMBOLS = sorted(
    {
        p.split("/")[-1].split("-klines-1d-")[0]
        for p in glob.glob(f"{CRYPTO_CACHE}/*-klines-1d-*.parquet")
    }
)
EQUITY_SYMBOLS = sorted(p.stem for p in Path(EQUITY_DIR).glob("*.parquet"))
CRYPTO_REGIME_REF = ["BTCUSDT"]
EQUITY_REGIME_REF = ["SPY", "QQQ", "IWM"]

DELISTED_CRYPTO = {"LUNAUSDT", "FTTUSDT"}


def load_crypto_daily(symbol: str) -> dict:
    files = sorted(glob.glob(f"{CRYPTO_CACHE}/{symbol}-klines-1d-*.parquet"))
    df = pl.concat([pl.read_parquet(f) for f in files])
    df = (
        df.unique(subset=["datetime"], keep="first")
        .sort("datetime")
        .filter(pl.col("datetime") <= pl.lit(DEV_END).str.to_datetime())
    )
    return {
        "dates": df["datetime"].to_numpy(),
        "open": df["open"].to_numpy().astype(float),
        "high": df["high"].to_numpy().astype(float),
        "low": df["low"].to_numpy().astype(float),
        "close": df["close"].to_numpy().astype(float),
    }


def load_equity_daily(symbol: str) -> dict:
    df = pl.read_parquet(f"{EQUITY_DIR}/{symbol}.parquet")
    df = (
        df.unique(subset=["timestamp"], keep="first")
        .sort("timestamp")
        .filter(pl.col("timestamp") <= pl.lit(DEV_END).str.to_datetime())
    )
    return {
        "dates": df["timestamp"].to_numpy(),
        "open": df["open"].to_numpy().astype(float),
        "high": df["high"].to_numpy().astype(float),
        "low": df["low"].to_numpy().astype(float),
        "close": df["close"].to_numpy().astype(float),
    }


def build_crypto_universe() -> dict[str, dict]:
    return {s: load_crypto_daily(s) for s in CRYPTO_SYMBOLS}


def build_equity_universe() -> dict[str, dict]:
    return {s: load_equity_daily(s) for s in EQUITY_SYMBOLS}


def build_regime_series(
    universe: dict[str, dict], ref_symbols: list[str], min_confirm: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    all_dates = np.array(
        sorted(set().union(*[set(f["dates"]) for f in universe.values()]))
    )
    closes = {s: universe[s]["close"] for s in ref_symbols if s in universe}
    dates = {s: universe[s]["dates"] for s in ref_symbols if s in universe}
    gate = S11.regime_gate(closes, dates, all_dates, min_confirm=min_confirm)
    date_index = {d: i for i, d in enumerate(all_dates)}
    per_symbol = {
        s: gate[[date_index[d] for d in f["dates"]]] for s, f in universe.items()
    }
    return all_dates, per_symbol


def signal_firing_diagnostics(universe: dict[str, dict]) -> dict:
    p = S11.BreakoutParams()
    out = {}
    for s, f in universe.items():
        sig = S11.compute_breakout_entries(f["open"], f["high"], f["low"], f["close"], p)
        out[s] = {"n_bars": len(f["close"]), "n_raw_signals": int(sig.sum())}
    return out


def main() -> None:
    crypto = build_crypto_universe()
    equity = build_equity_universe()

    _, crypto_regime = build_regime_series(crypto, CRYPTO_REGIME_REF, min_confirm=1)
    _, equity_regime = build_regime_series(equity, EQUITY_REGIME_REF, min_confirm=2)

    crypto_sig = signal_firing_diagnostics(crypto)
    equity_sig = signal_firing_diagnostics(equity)

    out = {
        "dev_end": DEV_END,
        "n_crypto_symbols": len(CRYPTO_SYMBOLS),
        "n_equity_symbols": len(EQUITY_SYMBOLS),
        "crypto_symbols": CRYPTO_SYMBOLS,
        "equity_symbols": EQUITY_SYMBOLS,
        "delisted_crypto_present": sorted(DELISTED_CRYPTO & set(CRYPTO_SYMBOLS)),
        "crypto_date_range": {
            s: [str(f["dates"][0])[:10], str(f["dates"][-1])[:10]]
            for s, f in crypto.items()
        },
        "equity_date_range": {
            s: [str(f["dates"][0])[:10], str(f["dates"][-1])[:10]]
            for s, f in equity.items()
        },
        "crypto_raw_signal_counts": crypto_sig,
        "equity_raw_signal_counts": equity_sig,
        "total_crypto_raw_signals": sum(v["n_raw_signals"] for v in crypto_sig.values()),
        "total_equity_raw_signals": sum(v["n_raw_signals"] for v in equity_sig.values()),
        "crypto_regime_ok_frac": float(
            np.mean([crypto_regime[s].mean() for s in crypto if len(crypto_regime[s])])
        ),
        "equity_regime_ok_frac": float(
            np.mean([equity_regime[s].mean() for s in equity if len(equity_regime[s])])
        ),
    }
    Path("src/research/tmp/phase_0_11d_results.json").write_text(json.dumps(out, indent=2))
    print(
        f"Phase 0 11d: {out['n_crypto_symbols']} crypto symbols "
        f"(raw signals={out['total_crypto_raw_signals']}, "
        f"regime_ok_frac={out['crypto_regime_ok_frac']:.3f}, "
        f"delisted_present={out['delisted_crypto_present']}), "
        f"{out['n_equity_symbols']} equity symbols "
        f"(raw signals={out['total_equity_raw_signals']}, "
        f"regime_ok_frac={out['equity_regime_ok_frac']:.3f})"
    )


if __name__ == "__main__":
    main()
