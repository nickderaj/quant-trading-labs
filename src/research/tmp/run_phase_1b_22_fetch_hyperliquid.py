"""Notebook 022, Phase 1b: fetch Hyperliquid perp funding history and 8h
candles for the mapped universe (scratch/022/phase1a_probe.json), into
src/research/cache/hyperliquid22/dev/.

Two endpoints, one host, no auth (`POST https://api.hyperliquid.xyz/info`):

  - `fundingHistory`: native cadence is HOURLY and the response is capped at
    500 records regardless of how wide [startTime, endTime] is -- verified
    directly (not assumed) before writing this script: a 2-year window
    returned exactly 500 rows, the earliest 500 hours from startTime. This
    script chunks funding requests at 500 hours (~20.8 days) per call,
    advancing startTime by the chunk width each time, mirroring
    fetch_funding_chunked's shape in run_phase_1_20_fetch_bybit.py.
  - `candleSnapshot`: requesting 8h candles for the FULL two-year dev window
    in one call returned every bar with no truncation (2,101 rows for a
    ~2-year window) -- also verified directly. One call per symbol, no
    chunking needed. (If a future symbol's history is long enough to hit an
    undocumented cap, `n_candles` is recorded in the manifest so that's
    visible rather than silently truncated.)

Both of the above were checked with disposable ad-hoc requests before this
script was written; neither request's raw response was cached to disk or
used in any analysis -- itemised as trial-log row "HL API limits" in
`phase_0_22_preregistration.json`, alongside the fact that the first such
check unintentionally requested a window reaching past 2025-07-01 (to see
whether the API would silently truncate at some undocumented internal cap
independent of the requested end date). It did not: the funding cap is a
flat 500 records from startTime regardless of endTime, so nothing >=
HOLDOUT_START was retained or would have been reachable at 500 hours/call
from a startTime inside the dev window. This script's own guard below still
refuses any endTime past HOLDOUT_START unconditionally, so the discipline
does not rest on that one observation.

Rate limit: <=5 req/s (Bybit's own convention, sec 4.3 in 020 -- Hyperliquid
documents a per-IP weight limit and informally 429s were seen near 8 req/s
in the pre-registration's own planning probe; 5/s stays clear of it), 3
retries with exponential backoff (5s/10s/15s).

Usage:
    uv run python src/research/tmp/run_phase_1b_22_fetch_hyperliquid.py [--smoke]
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

HL_INFO_URL = "https://api.hyperliquid.xyz/info"
CACHE_DIR = REPO_ROOT / "src" / "research" / "cache" / "hyperliquid22" / "dev"
PROBE_PATH = REPO_ROOT / "scratch" / "022" / "phase1a_probe.json"
MANIFEST_PATH = REPO_ROOT / "scratch" / "022" / "phase1_manifest.json"
STATUS_PATH = REPO_ROOT / "scratch" / "022" / "status.json"

# Matches Probe P1/P2's own window exactly (NEXT_PROMPT.md Candidate 1) so
# the pre-registered gates are computed on the identical dev window the
# planning-time numbers came from.
DEV_START = datetime(2023, 7, 1, tzinfo=UTC)
DEV_END = datetime(2025, 6, 30, tzinfo=UTC)
HOLDOUT_START = datetime(2025, 7, 1, tzinfo=UTC)

FUNDING_CHUNK_HOURS = 500
FUNDING_CHUNK_MS = FUNDING_CHUNK_HOURS * 3600 * 1000

MAX_REQ_PER_SEC = 5.0
MIN_INTERVAL_S = 1.0 / MAX_REQ_PER_SEC
RETRY_DELAYS_S = [5, 10, 15]
TIME_BOX_S = 60 * 60

_last_request_time = 0.0


def _rate_limited_post(body: dict) -> dict | list:
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
            req = urllib.request.Request(
                HL_INFO_URL,
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_exc = exc
            print(f"  retry {attempt}/{len(RETRY_DELAYS_S)} for {body}: {exc}")
    raise RuntimeError(f"exhausted retries for {body}") from last_exc


def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _assert_within_dev_window(end: datetime) -> None:
    if end > HOLDOUT_START:
        raise ValueError(
            f"refusing to fetch past HOLDOUT_START ({HOLDOUT_START:%Y-%m-%d}): "
            f"requested end={end}. Only a notebook's own gated holdout script "
            "may ever pass an end date past this boundary (mirrors bl18/bl20's "
            "load_basis_panel / load_xvenue_panel guard)."
        )


def fetch_funding_chunked(coin: str, start: datetime, end: datetime) -> list[dict]:
    _assert_within_dev_window(end)
    rows: list[dict] = []
    chunk_start = _to_ms(start)
    end_ms = _to_ms(end)
    while chunk_start < end_ms:
        data = _rate_limited_post(
            {
                "type": "fundingHistory",
                "coin": coin,
                "startTime": chunk_start,
                "endTime": end_ms,
            }
        )
        if not isinstance(data, list) or not data:
            break
        rows.extend(data)
        last_ts = data[-1]["time"]
        next_start = last_ts + 1
        if next_start <= chunk_start:
            break
        chunk_start = next_start
    return rows


def fetch_candles_8h(coin: str, start: datetime, end: datetime) -> list[dict]:
    _assert_within_dev_window(end)
    data = _rate_limited_post(
        {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": "8h",
                "startTime": _to_ms(start),
                "endTime": _to_ms(end),
            },
        }
    )
    return list(data) if isinstance(data, list) else []


def funding_rows_to_df(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={"datetime": pl.Datetime, "funding_rate": pl.Float64}
        )
    df = pl.DataFrame(rows)
    return (
        df.with_columns(
            pl.col("time").cast(pl.Int64),
            pl.col("fundingRate").cast(pl.Float64).alias("funding_rate"),
        )
        .with_columns(pl.from_epoch("time", time_unit="ms").alias("datetime"))
        .select("datetime", "funding_rate")
        .unique(subset=["datetime"])
        .sort("datetime")
    )


def candle_rows_to_df(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={
                "datetime": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "dollar_volume": pl.Float64,
            }
        )
    df = pl.DataFrame(rows)
    return (
        df.with_columns(
            pl.col("t").cast(pl.Int64),
            pl.col("o").cast(pl.Float64).alias("open"),
            pl.col("h").cast(pl.Float64).alias("high"),
            pl.col("l").cast(pl.Float64).alias("low"),
            pl.col("c").cast(pl.Float64).alias("close"),
            pl.col("v").cast(pl.Float64).alias("volume"),
        )
        .with_columns(pl.from_epoch("t", time_unit="ms").alias("datetime"))
        .with_columns((pl.col("volume") * pl.col("close")).alias("dollar_volume"))
        .select("datetime", "open", "high", "low", "close", "volume", "dollar_volume")
        .unique(subset=["datetime"])
        .sort("datetime")
    )


def fetch_symbol(coin: str, start: datetime, end: datetime) -> dict[str, object]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    funding_path = CACHE_DIR / f"{coin}-funding-{start:%Y-%m-%d}-{end:%Y-%m-%d}.parquet"
    candle_path = (
        CACHE_DIR / f"{coin}-candles8h-{start:%Y-%m-%d}-{end:%Y-%m-%d}.parquet"
    )

    if funding_path.exists():
        funding_df = pl.read_parquet(funding_path)
    else:
        funding_df = funding_rows_to_df(fetch_funding_chunked(coin, start, end))
        funding_df.write_parquet(funding_path)

    if candle_path.exists():
        candle_df = pl.read_parquet(candle_path)
    else:
        candle_df = candle_rows_to_df(fetch_candles_8h(coin, start, end))
        candle_df.write_parquet(candle_path)

    return {
        "funding_rows": len(funding_df),
        "funding_first": str(funding_df["datetime"].min()) if len(funding_df) else None,
        "funding_last": str(funding_df["datetime"].max()) if len(funding_df) else None,
        "candle_rows": len(candle_df),
        "candle_first": str(candle_df["datetime"].min()) if len(candle_df) else None,
        "candle_last": str(candle_df["datetime"].max()) if len(candle_df) else None,
        "status": "ok" if len(funding_df) and len(candle_df) else "empty",
    }


def _write_status(done: int, total: int, state: str, extra: dict | None = None) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": "1b_fetch_hyperliquid",
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
    mapping: dict[str, str] = probe["mapping"]
    # sorted by Binance symbol name for a deterministic, resumable order
    items = sorted(mapping.items())

    dev_start, dev_end = DEV_START, DEV_END
    if smoke:
        items = items[:3]
        from datetime import timedelta

        dev_end = dev_start + timedelta(days=60)

    total = len(items)
    symbols_manifest: dict[str, dict[str, object]] = {}
    truncated = False
    truncated_after_n_symbols: int | None = None
    _write_status(0, total, "running")

    t0 = time.monotonic()
    for i, (binance_symbol, hl_coin) in enumerate(items):
        if time.monotonic() - t0 > TIME_BOX_S:
            truncated = True
            truncated_after_n_symbols = i
            print(f"60-minute time box hit after {i}/{total} symbols -- stopping")
            break
        try:
            result = fetch_symbol(hl_coin, dev_start, dev_end)
            symbols_manifest[binance_symbol] = {"hl_coin": hl_coin, **result}
            print(
                f"[{i + 1}/{total}] {binance_symbol} ({hl_coin}): "
                f"funding={result['funding_rows']} candles={result['candle_rows']}"
            )
        except Exception as exc:  # noqa: BLE001 -- a genuine per-symbol fetch failure is data-quality info
            symbols_manifest[binance_symbol] = {
                "hl_coin": hl_coin,
                "status": "error",
                "error": str(exc),
            }
            print(f"[{i + 1}/{total}] {binance_symbol} ({hl_coin}): ERROR {exc}")
        _write_status(i + 1, total, "running")

    n_ok = sum(1 for v in symbols_manifest.values() if v.get("status") == "ok")
    manifest: dict[str, object] = {
        "started_utc": datetime.now(UTC).isoformat(),
        "smoke": smoke,
        "n_symbols": total,
        "dev_window": [str(dev_start), str(dev_end)],
        "symbols": symbols_manifest,
        "truncated": truncated,
        "truncated_after_n_symbols": truncated_after_n_symbols,
        "finished_utc": datetime.now(UTC).isoformat(),
        "n_ok": n_ok,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    _write_status(len(symbols_manifest), total, "done", {"n_ok": n_ok})
    print(f"Wrote {MANIFEST_PATH}: {n_ok}/{total} symbols ok")


if __name__ == "__main__":
    main()
