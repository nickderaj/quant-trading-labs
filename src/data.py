import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import requests
from tqdm import tqdm


def download_and_unzip(
    symbol: str,
    date: str | datetime,
    download_dir: str = "tmp",
    cache_dir: str = "cache",
    read_cache: bool = True,
) -> pl.DataFrame | None:
    """
    Download and unzip Binance futures trade data for a given symbol and date.
    Caches results as parquet files to avoid repeated downloads.

    read_cache=False skips reading an existing cache file into memory when the
    caller only needs the download/cache side effect, not the data itself.
    """
    # Normalize date to string
    date_str = date.strftime("%Y-%m-%d") if isinstance(date, datetime) else date

    # parents=True so this also works for nested dirs that don't exist yet
    cache_path_dir = Path(cache_dir)
    cache_path_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_path_dir / f"{symbol}-trades-{date_str}.parquet"

    if cache_path.exists():
        return pl.read_parquet(cache_path) if read_cache else None

    url = f"https://data.binance.vision/data/futures/um/daily/trades/{symbol}/{symbol}-trades-{date_str}.zip"

    download_path_dir = Path(download_dir)
    download_path_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_path_dir / f"{symbol}-trades-{date_str}.zip"
    csv_path = download_path_dir / f"{symbol}-trades-{date_str}.csv"

    try:
        # Download zip. Timeout so a stalled connection doesn't hang forever.
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            f.writelines(response.iter_content(chunk_size=8192))

        # Extract
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(download_path_dir)

        # Load into Polars
        df = pl.read_csv(
            csv_path,
            schema={
                "id": pl.Int64,
                "price": pl.Float64,
                "qty": pl.Float64,
                "quoteQty": pl.Float64,
                "time": pl.Int64,
                "isBuyerMaker": pl.Boolean,
            },
        ).with_columns(pl.from_epoch("time", time_unit="ms").alias("datetime"))

        # Cache result
        df.write_parquet(cache_path)
    finally:
        # Always clean up the temp zip/csv, even if something above failed,
        # so a retry doesn't trip over stale partial files.
        zip_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)

    return df


def download_trades(
    symbol: str,
    no_days: int,
    download_dir: str = "tmp",
    cache_dir: str = "cache",
    return_trades: bool = False,
) -> pl.DataFrame | None:
    """
    Download trades for the last N days up to yesterday with a progress bar.
    """
    # Binance file names use UTC dates, so anchor "yesterday" to UTC too
    yesterday = datetime.now(UTC) - timedelta(days=1)
    start_date = yesterday - timedelta(days=no_days - 1)

    dfs: list[pl.DataFrame] = []
    for i in tqdm(range(no_days), desc=f"Downloading {symbol}"):
        current_date = start_date + timedelta(days=i)
        try:
            if return_trades:
                df = download_and_unzip(symbol, current_date, download_dir, cache_dir)
                if df is not None:
                    dfs.append(df)
            else:
                download_and_unzip(
                    symbol, current_date, download_dir, cache_dir, read_cache=False
                )
        except Exception as e:  # noqa: BLE001 - one bad day shouldn't stop the whole batch
            tqdm.write(f"[ERROR] {symbol} {current_date.date()}: {e}")

    # Only concat if there's something to concat; an empty list would raise
    return pl.concat(dfs) if return_trades and dfs else None


def download_date_range(
    symbol: str,
    start_date: str | datetime,
    end_date: str | datetime,
    download_dir: str = "tmp",
    cache_dir: str = "cache",
) -> None:
    """
    Download trade data for a range of dates with a progress bar.
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)

    num_days = (end_date - start_date).days + 1

    for i in tqdm(range(num_days), desc=f"Downloading {symbol}"):
        current_date = start_date + timedelta(days=i)
        try:
            download_and_unzip(
                symbol, current_date, download_dir, cache_dir, read_cache=False
            )
        except Exception as e:  # noqa: BLE001 - one bad day shouldn't stop the whole batch
            tqdm.write(f"[ERROR] {symbol} {current_date.date()}: {e}")
