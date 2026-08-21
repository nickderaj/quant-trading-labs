"""Notebook 020, Phase 1b (NEXT_PROMPT.md sec 4.3): fetch Bybit perp klines
and funding-rate history for the Binance(spot-usable)-intersect-Bybit
universe, into src/research/cache/bybit20/{dev,holdout}/.

Caches RAW native-interval data only (4h klines -- interval=480 is silently
rejected by Bybit's kline endpoint, sec 4.2/phase1a_probe.json -- and
native-cadence funding, which is 240min or 480min per symbol). The 4h->8h
kline aggregation and the funding sum-into-8h-bucket resampling (sec 4.4) are
NOT done here: they live in basis_lib20.py as testable, pinned functions
(test_bybit_funding_resample_to_8h). This script's only job is a faithful,
resumable, rate-limited copy of what Bybit's API returns.

Single-threaded, <=5 req/s, 3 retries with exponential backoff (5s/10s/15s)
-- sec 4.3's politeness rule, learned from 018 Phase 1's DNS-blip lesson.
A 404/empty result across a whole window for a symbol/series is DATA, not an
error, and is recorded as such in the manifest, not raised.

Usage:
    uv run python src/research/tmp/run_phase_1_20_fetch_bybit.py [--smoke]
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

BYBIT_HOST = "https://api.bybit.com"
CACHE_DIR = REPO_ROOT / "src" / "research" / "cache" / "bybit20"
UNIVERSE_SEED_PATH = (
    REPO_ROOT / "src" / "research" / "tmp" / "design_c_v2_universe.json"
)
MANIFEST_PATH = REPO_ROOT / "scratch" / "020" / "phase1_manifest.json"
STATUS_PATH = REPO_ROOT / "scratch" / "020" / "status.json"
PROBE_PATH = REPO_ROOT / "scratch" / "020" / "phase1a_probe.json"

DEV_START = datetime(2021, 7, 1, tzinfo=UTC)
DEV_END = datetime(2025, 6, 30, tzinfo=UTC)
HOLDOUT_START = datetime(2025, 7, 1, tzinfo=UTC)
HOLDOUT_END = datetime.now(UTC)

KLINE_INTERVAL_MIN = 240  # sec 4.2: 480 is silently rejected by Bybit
KLINE_CHUNK_MS = 1000 * KLINE_INTERVAL_MIN * 60_000  # 1000 bars/request cap
FUNDING_CHUNK_MS = (
    200 * KLINE_INTERVAL_MIN * 60_000
)  # 200 entries/request cap, worst-case 4h cadence

MAX_REQ_PER_SEC = 5.0
MIN_INTERVAL_S = 1.0 / MAX_REQ_PER_SEC
RETRY_DELAYS_S = [5, 10, 15]
TIME_BOX_S = 90 * 60

_last_request_time = 0.0


def _rate_limited_get(url: str) -> dict:
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < MIN_INTERVAL_S:
        time.sleep(MIN_INTERVAL_S - elapsed)
    last_exc: Exception | None = None
    for attempt, delay in enumerate([0, *RETRY_DELAYS_S]):
        if delay:
            time.sleep(delay)
        try:
            _last_request_time = time.monotonic()
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
            if data.get("retCode") != 0:
                raise ValueError(
                    f"retCode={data.get('retCode')} retMsg={data.get('retMsg')}"
                )
            return data
        except (urllib.error.URLError, ValueError, TimeoutError) as exc:
            last_exc = exc
            print(f"  retry {attempt}/{len(RETRY_DELAYS_S)} for {url}: {exc}")
    raise RuntimeError(f"exhausted retries for {url}") from last_exc


def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def fetch_kline_4h_chunked(symbol: str, start: datetime, end: datetime) -> list[list]:
    rows: list[list] = []
    chunk_start = _to_ms(start)
    end_ms = _to_ms(end)
    while chunk_start < end_ms:
        chunk_end = min(chunk_start + KLINE_CHUNK_MS, end_ms)
        url = (
            f"{BYBIT_HOST}/v5/market/kline?category=linear&symbol={symbol}"
            f"&interval={KLINE_INTERVAL_MIN}&start={chunk_start}&end={chunk_end}&limit=1000"
        )
        data = _rate_limited_get(url)
        rows.extend(data["result"]["list"])
        chunk_start = chunk_end
    return rows


def fetch_funding_chunked(symbol: str, start: datetime, end: datetime) -> list[dict]:
    rows: list[dict] = []
    chunk_start = _to_ms(start)
    end_ms = _to_ms(end)
    while chunk_start < end_ms:
        chunk_end = min(chunk_start + FUNDING_CHUNK_MS, end_ms)
        url = (
            f"{BYBIT_HOST}/v5/market/funding/history?category=linear&symbol={symbol}"
            f"&startTime={chunk_start}&endTime={chunk_end}&limit=200"
        )
        data = _rate_limited_get(url)
        rows.extend(data["result"]["list"])
        chunk_start = chunk_end
    return rows


def kline_rows_to_df(rows: list[list]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={
                "datetime": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "turnover": pl.Float64,
            }
        )
    df = pl.DataFrame(
        rows,
        schema=["start_ms", "open", "high", "low", "close", "volume", "turnover"],
        orient="row",
    )
    return (
        df.with_columns(
            pl.col("start_ms").cast(pl.Int64).alias("start_ms"),
            *[
                pl.col(c).cast(pl.Float64)
                for c in ("open", "high", "low", "close", "volume", "turnover")
            ],
        )
        .with_columns(pl.from_epoch("start_ms", time_unit="ms").alias("datetime"))
        .select("datetime", "open", "high", "low", "close", "volume", "turnover")
        .unique(subset=["datetime"])
        .sort("datetime")
    )


def funding_rows_to_df(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={"datetime": pl.Datetime, "funding_rate": pl.Float64}
        )
    df = pl.DataFrame(rows)
    return (
        df.with_columns(
            pl.col("fundingRateTimestamp").cast(pl.Int64),
            pl.col("fundingRate").cast(pl.Float64).alias("funding_rate"),
        )
        .with_columns(
            pl.from_epoch("fundingRateTimestamp", time_unit="ms").alias("datetime")
        )
        .select("datetime", "funding_rate")
        .unique(subset=["datetime"])
        .sort("datetime")
    )


def fetch_symbol_window(
    symbol: str, start: datetime, end: datetime, window_dir: Path
) -> dict[str, object]:
    window_dir.mkdir(parents=True, exist_ok=True)
    kline_path = (
        window_dir / f"{symbol}-kline4h-{start:%Y-%m-%d}-{end:%Y-%m-%d}.parquet"
    )
    funding_path = (
        window_dir / f"{symbol}-funding-{start:%Y-%m-%d}-{end:%Y-%m-%d}.parquet"
    )

    if kline_path.exists():
        kline_df = pl.read_parquet(kline_path)
    else:
        kline_rows = fetch_kline_4h_chunked(symbol, start, end)
        kline_df = kline_rows_to_df(kline_rows)
        kline_df.write_parquet(kline_path)

    if funding_path.exists():
        funding_df = pl.read_parquet(funding_path)
    else:
        funding_rows = fetch_funding_chunked(symbol, start, end)
        funding_df = funding_rows_to_df(funding_rows)
        funding_df.write_parquet(funding_path)

    return {
        "kline_rows": len(kline_df),
        "kline_first": str(kline_df["datetime"].min()) if len(kline_df) else None,
        "kline_last": str(kline_df["datetime"].max()) if len(kline_df) else None,
        "funding_rows": len(funding_df),
        "funding_first": str(funding_df["datetime"].min()) if len(funding_df) else None,
        "funding_last": str(funding_df["datetime"].max()) if len(funding_df) else None,
        "status": "ok" if len(kline_df) and len(funding_df) else "empty",
    }


def _write_status(done: int, total: int, state: str, extra: dict | None = None) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": "1b_fetch_bybit",
        "done": done,
        "total": total,
        "state": state,
        "updated_utc": datetime.now(UTC).isoformat(),
    }
    if extra:
        payload.update(extra)
    with open(STATUS_PATH, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    smoke = "--smoke" in sys.argv

    with open(PROBE_PATH) as f:
        probe = json.load(f)
    symbols = list(probe["universe_intersection"].get("_intersection_list", []))
    if not symbols:
        # probe JSON stores counts, not the list; recompute the same
        # intersection it used, deterministically, from the same inputs.
        with open(UNIVERSE_SEED_PATH) as f:
            universe = json.load(f)
        usable_126 = [s for s in universe if s not in ("1000SHIBUSDT", "1000XECUSDT")]
        url = f"{BYBIT_HOST}/v5/market/instruments-info?category=linear&limit=1000"
        data = _rate_limited_get(url)
        bybit_symbols = {
            it["symbol"]: it
            for it in data["result"]["list"]
            if it["quoteCoin"] == "USDT" and it["contractType"] == "LinearPerpetual"
        }
        symbols = [s for s in usable_126 if s in bybit_symbols]

    dev_start, dev_end = DEV_START, DEV_END
    holdout_start, holdout_end = HOLDOUT_START, HOLDOUT_END
    if smoke:
        symbols = symbols[:3]
        from datetime import timedelta

        dev_end = dev_start + timedelta(days=182)
        holdout_end = holdout_start + timedelta(days=30)

    total = len(symbols)
    symbols_manifest: dict[str, dict[str, object]] = {}
    truncated = False
    truncated_after_n_symbols: int | None = None
    _write_status(0, total, "running")

    t0 = time.monotonic()
    for i, symbol in enumerate(symbols):
        if time.monotonic() - t0 > TIME_BOX_S:
            truncated = True
            truncated_after_n_symbols = i
            print(f"90-minute time box hit after {i}/{total} symbols -- stopping")
            break
        try:
            dev_result = fetch_symbol_window(
                symbol, dev_start, dev_end, CACHE_DIR / "dev"
            )
            holdout_result = fetch_symbol_window(
                symbol, holdout_start, holdout_end, CACHE_DIR / "holdout"
            )
            symbols_manifest[symbol] = {"dev": dev_result, "holdout": holdout_result}
            print(
                f"[{i + 1}/{total}] {symbol}: dev kline={dev_result['kline_rows']} "
                f"funding={dev_result['funding_rows']}, holdout kline={holdout_result['kline_rows']} "
                f"funding={holdout_result['funding_rows']}"
            )
        except Exception as exc:  # noqa: BLE001 -- a genuine per-symbol fetch failure is data-quality info
            symbols_manifest[symbol] = {"status": "error", "error": str(exc)}
            print(f"[{i + 1}/{total}] {symbol}: ERROR {exc}")
        _write_status(i + 1, total, "running")

    def _both_windows_ok(v: dict[str, object]) -> bool:
        dev = v.get("dev")
        holdout = v.get("holdout")
        return (
            isinstance(dev, dict)
            and dev.get("status") == "ok"
            and isinstance(holdout, dict)
            and holdout.get("status") == "ok"
        )

    n_ok = sum(1 for v in symbols_manifest.values() if _both_windows_ok(v))
    manifest: dict[str, object] = {
        "started_utc": datetime.now(UTC).isoformat(),
        "smoke": smoke,
        "n_symbols": total,
        "dev_window": [str(dev_start), str(dev_end)],
        "holdout_window": [str(holdout_start), str(holdout_end)],
        "symbols": symbols_manifest,
        "truncated": truncated,
        "truncated_after_n_symbols": truncated_after_n_symbols,
        "finished_utc": datetime.now(UTC).isoformat(),
        "n_ok_both_windows": n_ok,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    _write_status(len(symbols_manifest), total, "done", {"n_ok_both_windows": n_ok})
    print(f"Wrote {MANIFEST_PATH}: {n_ok}/{total} symbols ok in both windows")


if __name__ == "__main__":
    main()
