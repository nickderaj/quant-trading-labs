"""Notebook 018, Phase 1: data acquisition (spot + perp + premiumIndexKlines,
8h, for the 128-symbol universe seed).

Run as a script (not imported): `uv run python src/research/tmp/run_phase_1_18_fetch.py`.
Idempotent -- every fetch goes through basis_lib18's own per-month parquet
caching, so a killed-and-restarted run skips everything already on disk.

Fetches BOTH the development window (2021-07-01..2025-06-30) and the
holdout window (2025-07-01..) in this one pass, into two separate cache
directories (basis18/dev, basis18/holdout) -- per the run instructions:
"fetch the holdout window to basis18/holdout/ in the same pass -- Phase 4/5
cannot read that directory, so fetching it early costs nothing and saves a
second download later." This script only ever performs raw I/O (download
and cache spot/perp/premium klines and funding rate history); it computes
no returns, no gate statistic, and never calls allow_holdout=True on
anything -- basis_lib18.load_basis_panel (Phase 3/4/5's only entry point)
still cannot read basis18/holdout/, and run_phase_6_18_holdout.py remains
the only file that ever turns this pre-fetched holdout data into a backtest
result (NEXT_PROMPT sec 9.3).

Writes scratch/018/status.json (heartbeat, at least every 30s) and
scratch/018/phase1.log (full log). A 404 across every month for a
symbol/series is recorded in the per-window manifest as data, not an error
(sec 4.1/9.2).
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "research" / "tmp"))

import basis_lib18 as bl
import requests

import data

STATUS_PATH = REPO_ROOT / "scratch" / "018" / "status.json"
LOG_PATH = REPO_ROOT / "scratch" / "018" / "phase1.log"
MANIFEST_PATH = REPO_ROOT / "scratch" / "018" / "phase1_manifest.json"

HOLDOUT_START = datetime(2025, 7, 1, tzinfo=UTC)
HOLDOUT_END = datetime(2026, 8, 1, tzinfo=UTC)
HOLDOUT_CACHE_DIR = "src/research/cache/basis18/holdout"
HOLDOUT_DOWNLOAD_DIR = "src/research/tmp_dl/basis18/holdout"

CONCURRENCY = 8

_status_lock_state: dict[str, object] = {}


def _log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now(UTC).isoformat()}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def _write_status(**kwargs: object) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _status_lock_state.update(kwargs)
    _status_lock_state["last_heartbeat"] = datetime.now(UTC).isoformat()
    tmp_path = STATUS_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(_status_lock_state, f, indent=2)
    tmp_path.replace(STATUS_PATH)


def check_availability() -> dict[str, int | str]:
    """Re-run the sec 4.1 availability checks live, before building anything
    on the cached snapshot's assumptions.
    """
    checks = {
        "spot_klines_8h": "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/8h/BTCUSDT-8h-2021-08.zip",
        "futures_klines_8h": "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/8h/BTCUSDT-8h-2021-08.zip",
        "premium_index_klines_8h": "https://data.binance.vision/data/futures/um/monthly/premiumIndexKlines/BTCUSDT/8h/BTCUSDT-8h-2021-08.zip",
        "spot_klines_holdout_month": "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/8h/BTCUSDT-8h-2025-09.zip",
    }
    results: dict[str, int | str] = {}
    for name, url in checks.items():
        try:
            resp = requests.head(url, timeout=15, allow_redirects=True)
            results[name] = resp.status_code
        except requests.RequestException as e:
            results[name] = f"ERROR: {e}"
    return results


def fetch_symbol_window(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    cache_dir: str,
    download_dir: str,
) -> str:
    """Fetch spot/perp/premium for one symbol/window; for the holdout window
    also fetch funding rate history (not already cached, unlike dev). Returns
    a manifest status string.
    """
    try:
        series = bl.fetch_symbol_series(
            symbol, start_date, end_date, download_dir, cache_dir
        )
    except Exception as e:  # noqa: BLE001 -- one bad symbol must not kill the batch
        return f"ERROR: {e}"

    if start_date >= HOLDOUT_START:
        try:
            data.download_funding_rate_range(
                symbol, start_date, end_date, cache_dir=cache_dir
            )
        except Exception as e:  # noqa: BLE001
            return f"ERROR (funding): {e}"

    if series["spot"] is None:
        return "no_spot"
    if series["perp"] is None:
        return "no_perp"
    return "ok"


def run_window(
    window_name: str,
    start_date: datetime,
    end_date: datetime,
    cache_dir: str,
    download_dir: str,
    symbols: list[str],
    manifest: dict[str, dict[str, str]],
    total_units: int,
    done_counter: list[int],
) -> None:
    _log(
        f"=== window={window_name} start={start_date.date()} end={end_date.date()} symbols={len(symbols)} ==="
    )
    manifest[window_name] = {}
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {
            pool.submit(
                fetch_symbol_window, sym, start_date, end_date, cache_dir, download_dir
            ): sym
            for sym in symbols
        }
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                outcome = fut.result()
            except Exception as e:  # noqa: BLE001
                outcome = f"ERROR: {e}"
            manifest[window_name][sym] = outcome
            done_counter[0] += 1
            # "availability" is a str -> (int | str) dict, not a per-symbol
            # outcome map -- excluded here so its int status codes never hit
            # .startswith.
            failed = sum(
                1
                for window, w in manifest.items()
                if window != "availability"
                for v in w.values()
                if v.startswith(("ERROR", "no_"))
            )
            t0 = _status_lock_state["_t0"]
            assert isinstance(t0, float)
            elapsed = time.monotonic() - t0
            rate = done_counter[0] / elapsed if elapsed > 0 else 0.0
            eta = (total_units - done_counter[0]) / rate if rate > 0 else None
            _write_status(
                phase="1",
                state="running",
                done=done_counter[0],
                total=total_units,
                failed=failed,
                eta_seconds=eta,
                current_window=window_name,
                current_symbol=sym,
                current_outcome=outcome,
            )
            if done_counter[0] % 10 == 0 or outcome != "ok":
                _log(
                    f"[{window_name}] {sym}: {outcome} ({done_counter[0]}/{total_units})"
                )
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    _status_lock_state["_t0"] = time.monotonic()
    _write_status(
        phase="1", state="starting", done=0, total=0, failed=0, eta_seconds=None
    )

    _log("Re-running sec 4.1 availability checks live...")
    availability = check_availability()
    for name, code in availability.items():
        _log(f"  {name}: {code}")
    if availability.get("spot_klines_8h") != 200:
        _log(
            "WARNING: spot 8h klines availability check did not return 200 -- "
            "the 009 blocker may not actually be resolved. Continuing anyway "
            "per the run instructions (record and proceed)."
        )

    symbols = bl.load_universe_seed()
    _log(f"Universe seed: {len(symbols)} symbols from {bl.UNIVERSE_SEED_PATH}")

    total_units = len(symbols) * 2  # dev + holdout, one unit per symbol per window
    done_counter = [0]
    manifest: dict[str, dict[str, str]] = {"availability": availability}  # type: ignore[dict-item]

    Path(bl.DEV_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    Path(HOLDOUT_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    run_window(
        "dev",
        bl.DEV_START,
        bl.DEV_END,
        bl.DEV_CACHE_DIR,
        bl.DEV_DOWNLOAD_DIR,
        symbols,
        manifest,
        total_units,
        done_counter,
    )
    run_window(
        "holdout",
        HOLDOUT_START,
        HOLDOUT_END,
        HOLDOUT_CACHE_DIR,
        HOLDOUT_DOWNLOAD_DIR,
        symbols,
        manifest,
        total_units,
        done_counter,
    )

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    n_no_spot = sum(1 for v in manifest["dev"].values() if v == "no_spot")
    n_ok = sum(1 for v in manifest["dev"].values() if v == "ok")
    _log(f"Phase 1 done. dev: {n_ok}/{len(symbols)} ok, {n_no_spot} no_spot.")
    _write_status(
        phase="1",
        state="done",
        done=total_units,
        total=total_units,
        failed=sum(
            1
            for w in manifest.values()
            if isinstance(w, dict)
            for v in w.values()
            if isinstance(v, str) and (v.startswith(("ERROR", "no_")))
        ),
        eta_seconds=0,
    )


if __name__ == "__main__":
    main()
