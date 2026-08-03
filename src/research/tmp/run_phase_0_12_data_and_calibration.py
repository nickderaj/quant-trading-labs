"""Notebook 12 Phase 0 -- load all three asset-class universes (crypto
perpetuals, commodity-equity/ETF, and databento futures), trim to the dev
window, build each class's fail-closed trend regime gate, and derive the
single frozen set of Gate VB thresholds from a per-instrument calibration
window (NEXT_PROMPT.md sec 2: "derive the thresholds once from the first
2-3 years of history [out of sample of the test period], declare them, and
never re-tune after seeing a backtest"). No backtest of any kind runs in
this phase -- only the raw-signal diagnostics needed to freeze the
threshold numbers.
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import commod_lib8 as C8
import spread_lib11 as S11

import research

research.set_seed(0)

DEV_END = "2024-12-31"
CALIBRATION_YEARS = 3
CRYPTO_CACHE = "src/research/cache"
EQUITY_DIR = "src/research/data/market/yfinance/daily"
DATABENTO_DIR = "src/research/data/market/databento"

CRYPTO_SYMBOLS = sorted(
    {
        p.split("/")[-1].split("-klines-1d-")[0]
        for p in glob.glob(f"{CRYPTO_CACHE}/*-klines-1d-*.parquet")
    }
)
# Excludes yfinance's *=F FX/futures proxy tickers: NEXT_PROMPT.md sec 1
# flags their volume as unreliable ("either exclude those tickers or
# declare the check that keeps them") -- this notebook excludes them.
EQUITY_SYMBOLS = sorted(
    p.stem for p in Path(EQUITY_DIR).glob("*.parquet") if "=" not in p.stem
)
FUTURES_PRODUCTS = [
    "BZ",
    "CL",
    "ES",
    "GC",
    "HO",
    "KE",
    "NG",
    "PA",
    "PL",
    "RB",
    "SI",
    "ZC",
    "ZL",
    "ZM",
    "ZS",
    "ZW",
]

CRYPTO_REGIME_REF = ["BTCUSDT"]
EQUITY_REGIME_REF = ["SPY", "QQQ", "IWM"]
FUTURES_REGIME_REF = ["CL", "GC", "ES"]  # energy/metal/equity-index vote, same
# 3-reference/min_confirm=2 construction as the equity class -- a declared
# structural choice, not fitted to this rule's own performance.

DELISTED_CRYPTO = {"LUNAUSDT", "FTTUSDT"}

BASE_P = S11.VolBreakoutParams()  # only fixed structural fields used pre-freeze
CALIB_PARAMS = {
    "prior_run_lookback": BASE_P.prior_run_lookback,
    "base_window": BASE_P.base_window,
    "atr_window": BASE_P.atr_window,
    "vol_window": BASE_P.vol_window,
}


def load_crypto_daily(symbol: str) -> dict:
    files = sorted(glob.glob(f"{CRYPTO_CACHE}/{symbol}-klines-1d-*.parquet"))
    df = pl.concat([pl.read_parquet(f) for f in files])
    df = (
        df.unique(subset=["datetime"], keep="first")
        .sort("datetime")
        .filter(pl.col("datetime") <= pl.lit(DEV_END).str.to_datetime())
    )
    return {
        "dates": df["datetime"].to_numpy().astype("datetime64[D]"),
        "open": df["open"].to_numpy().astype(float),
        "high": df["high"].to_numpy().astype(float),
        "low": df["low"].to_numpy().astype(float),
        "close": df["close"].to_numpy().astype(float),
        "volume": df["volume"].to_numpy().astype(float),
        "is_roll": np.zeros(len(df), dtype=bool),  # perpetual: no rolls
        "asset_class": "crypto",
    }


def load_equity_daily(symbol: str) -> dict:
    df = pl.read_parquet(f"{EQUITY_DIR}/{symbol}.parquet")
    df = (
        df.unique(subset=["timestamp"], keep="first")
        .sort("timestamp")
        .filter(pl.col("timestamp") <= pl.lit(DEV_END).str.to_datetime())
    )
    return {
        "dates": df["timestamp"].to_numpy().astype("datetime64[D]"),
        "open": df["open"].to_numpy().astype(float),
        "high": df["high"].to_numpy().astype(float),
        "low": df["low"].to_numpy().astype(float),
        "close": df["close"].to_numpy().astype(float),
        "volume": df["volume"].to_numpy().astype(float),
        "is_roll": np.zeros(len(df), dtype=bool),  # single continuous instrument
        "asset_class": "equity",
    }


def load_futures_ohlcv_cache() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    ohlcv = pl.read_parquet(f"{DATABENTO_DIR}/ohlcv/*.parquet")
    contracts = pl.read_parquet(f"{DATABENTO_DIR}/contracts.parquet")
    roll_calendar = pl.read_parquet(f"{DATABENTO_DIR}/roll_calendar.parquet")
    return ohlcv, contracts, roll_calendar


def load_futures_product(
    product: str,
    ohlcv: pl.DataFrame,
    contracts: pl.DataFrame,
    roll_calendar: pl.DataFrame,
) -> dict:
    curve = C8.build_continuous_series_ohlcv(ohlcv, contracts, roll_calendar, product)
    curve = curve.filter(pl.col("date") <= pl.lit(DEV_END).str.to_date()).drop_nulls(
        subset=["close_backadj"]
    )
    return {
        "dates": curve["date"].to_numpy().astype("datetime64[D]"),
        "open": curve["open_backadj"].to_numpy().astype(float),
        "high": curve["high_backadj"].to_numpy().astype(float),
        "low": curve["low_backadj"].to_numpy().astype(float),
        "close": curve["close_backadj"].to_numpy().astype(float),
        "volume": curve["volume"].to_numpy().astype(float),
        "is_roll": curve["is_roll"].to_numpy().astype(bool),
        "asset_class": "futures",
    }


def build_crypto_universe() -> dict[str, dict]:
    return {s: load_crypto_daily(s) for s in CRYPTO_SYMBOLS}


def build_equity_universe() -> dict[str, dict]:
    return {s: load_equity_daily(s) for s in EQUITY_SYMBOLS}


def build_futures_universe() -> dict[str, dict]:
    ohlcv, contracts, roll_calendar = load_futures_ohlcv_cache()
    return {
        p: load_futures_product(p, ohlcv, contracts, roll_calendar)
        for p in FUTURES_PRODUCTS
    }


def build_regime_series(
    universe: dict[str, dict], ref_symbols: list[str], min_confirm: int
) -> dict[str, np.ndarray]:
    all_dates = np.array(
        sorted(set().union(*[set(f["dates"]) for f in universe.values()]))
    )
    closes = {s: universe[s]["close"] for s in ref_symbols if s in universe}
    dates = {s: universe[s]["dates"] for s in ref_symbols if s in universe}
    gate = S11.regime_gate(closes, dates, all_dates, min_confirm=min_confirm)
    date_index = {d: i for i, d in enumerate(all_dates)}
    return {s: gate[[date_index[d] for d in f["dates"]]] for s, f in universe.items()}


def calibration_cutoff(dates: np.ndarray) -> np.datetime64:
    """Per-instrument calibration/backtest split: the instrument's own
    first `CALIBRATION_YEARS` of history (or its full history if shorter,
    capped at the dev-window end), held out of the backtest and used only
    to freeze the Gate VB thresholds below."""
    start = dates[0]
    cutoff = start + np.timedelta64(365 * CALIBRATION_YEARS, "D")
    return min(cutoff, np.datetime64(DEV_END))


def raw_diagnostics(f: dict, calib_end: np.datetime64) -> dict:
    """Base-range (ATR multiples), prior-run (ATR-normalized, signed), and
    volume-ratio-to-trailing-median arrays, restricted to this
    instrument's own calibration window only -- the pooled inputs to
    freezing Gate VB's thresholds. Mirrors the internal arithmetic of
    `compute_vol_breakout_entries` without needing a frozen threshold to
    already exist (this IS how those thresholds get frozen)."""
    high, low, close = f["high"], f["low"], f["close"]
    atr = S11.true_atr_series(high, low, close, CALIB_PARAMS["atr_window"])
    bw = CALIB_PARAMS["base_window"]
    lookback = CALIB_PARAMS["prior_run_lookback"]
    n = len(close)
    base_high = pl.Series(high).rolling_max(window_size=bw).to_numpy()
    base_low = pl.Series(low).rolling_min(window_size=bw).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        base_range_atr = (base_high - base_low) / atr

    idx = np.arange(n)
    prior_close = np.full(n, np.nan)
    valid = idx - bw - lookback >= 0
    prior_close[valid] = close[idx[valid] - bw - lookback]
    base_start_close = np.full(n, np.nan)
    base_start_atr = np.full(n, np.nan)
    valid2 = idx - bw >= 0
    base_start_close[valid2] = close[idx[valid2] - bw]
    base_start_atr[valid2] = atr[idx[valid2] - bw]
    with np.errstate(invalid="ignore", divide="ignore"):
        prior_run_ret = base_start_close / prior_close - 1.0
        expected_pct_move = base_start_atr / base_start_close
        prior_run_atr_mult = prior_run_ret / expected_pct_move

    vol_window = CALIB_PARAMS["vol_window"]
    median_vol_1 = (
        pl.Series(f["volume"])
        .rolling_median(window_size=vol_window)
        .shift(1)
        .to_numpy()
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        vol_ratio = f["volume"] / median_vol_1

    since_roll = S11._bars_since_roll(f["is_roll"])
    clean_vol = (since_roll > vol_window) & ~f["is_roll"]

    in_calib = f["dates"] < calib_end
    return {
        "base_range_atr": base_range_atr[in_calib],
        "prior_run_atr_mult": prior_run_atr_mult[in_calib],
        "vol_ratio": vol_ratio[in_calib & clean_vol],
    }


def freeze_thresholds(universe_by_class: dict[str, dict[str, dict]]) -> dict:
    base_ranges, prior_runs, vol_ratios = [], [], []
    for uni in universe_by_class.values():
        for f in uni.values():
            calib_end = calibration_cutoff(f["dates"])
            d = raw_diagnostics(f, calib_end)
            base_ranges.append(d["base_range_atr"])
            prior_runs.append(d["prior_run_atr_mult"])
            vol_ratios.append(d["vol_ratio"])
    all_base_ranges = np.concatenate(base_ranges)
    all_base_ranges = all_base_ranges[np.isfinite(all_base_ranges)]
    all_prior_runs = np.concatenate(prior_runs)
    all_prior_runs = all_prior_runs[np.isfinite(all_prior_runs)]
    all_vol_ratios = np.concatenate(vol_ratios)
    all_vol_ratios = all_vol_ratios[np.isfinite(all_vol_ratios) & (all_vol_ratios > 0)]

    # Declared percentiles, chosen once from the calibration pool's own
    # shape (NEXT_PROMPT.md sec 2: "not by trying 1.2, 1.5, 2.0 and keeping
    # the best"): p30 of base range picks out the "tight" tail; p70 of
    # |prior run| picks out a genuine trend, not a random walk's typical
    # ATR-normalized wander; p75 of the volume ratio is the "meaningfully
    # above normal" tail of each instrument's own volume distribution.
    base_max_range_atr_mult = float(np.percentile(all_base_ranges, 30))
    prior_run_min_atr_mult = float(np.percentile(np.abs(all_prior_runs), 70))
    vol_k = float(np.percentile(all_vol_ratios, 75))

    return {
        "n_calibration_bars": {
            "base_range": len(all_base_ranges),
            "prior_run": len(all_prior_runs),
            "vol_ratio": len(all_vol_ratios),
        },
        "base_max_range_atr_mult_raw": base_max_range_atr_mult,
        "prior_run_min_atr_mult_raw": prior_run_min_atr_mult,
        "vol_k_raw": vol_k,
        # Frozen, declared values used everywhere from Phase 1 onward --
        # rounded to 2 significant figures (a legibility rounding of the
        # measured percentile, not a re-tune: see notebook text).
        "base_max_range_atr_mult": round(base_max_range_atr_mult, 1),
        "prior_run_min_atr_mult": round(prior_run_min_atr_mult, 1),
        "vol_k": round(vol_k, 2),
    }


def main() -> None:
    crypto = build_crypto_universe()
    equity = build_equity_universe()
    futures = build_futures_universe()

    crypto_regime = build_regime_series(crypto, CRYPTO_REGIME_REF, min_confirm=1)
    equity_regime = build_regime_series(equity, EQUITY_REGIME_REF, min_confirm=2)
    futures_regime = build_regime_series(futures, FUTURES_REGIME_REF, min_confirm=2)

    universe_by_class = {"crypto": crypto, "equity": equity, "futures": futures}
    thresholds = freeze_thresholds(universe_by_class)

    n_instruments = {
        "crypto": len(crypto),
        "equity": len(equity),
        "futures": len(futures),
    }
    out = {
        "dev_end": DEV_END,
        "calibration_years": CALIBRATION_YEARS,
        "n_instruments": n_instruments,
        "total_instruments": len(crypto) + len(equity) + len(futures),
        "crypto_symbols": CRYPTO_SYMBOLS,
        "equity_symbols": EQUITY_SYMBOLS,
        "futures_products": FUTURES_PRODUCTS,
        "delisted_crypto_present": sorted(DELISTED_CRYPTO & set(CRYPTO_SYMBOLS)),
        "regime_ok_frac": {
            "crypto": float(np.mean([crypto_regime[s].mean() for s in crypto])),
            "equity": float(np.mean([equity_regime[s].mean() for s in equity])),
            "futures": float(np.mean([futures_regime[s].mean() for s in futures])),
        },
        "thresholds": thresholds,
    }
    Path("src/research/tmp/phase_0_12_results.json").write_text(
        json.dumps(out, indent=2)
    )
    print(
        f"Phase 0 12: {out['total_instruments']} instruments "
        f"(crypto={n_instruments['crypto']}, equity={n_instruments['equity']}, "
        f"futures={n_instruments['futures']}); "
        f"frozen thresholds: base_max_range_atr_mult={thresholds['base_max_range_atr_mult']}, "
        f"prior_run_min_atr_mult={thresholds['prior_run_min_atr_mult']}, "
        f"vol_k={thresholds['vol_k']}"
    )


if __name__ == "__main__":
    main()
