import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import requests
from tqdm import tqdm

# Binance's raw kline CSV columns, in file order. "ignore" is an unused
# reserved column Binance ships in every kline file.
KLINE_SCHEMA = {
    "open_time": pl.Int64,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "close_time": pl.Int64,
    "quote_volume": pl.Float64,
    "count": pl.Int64,
    "taker_buy_volume": pl.Float64,
    "taker_buy_quote_volume": pl.Float64,
    "ignore": pl.Int64,
}


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


def _month_range(start_date: datetime, end_date: datetime) -> list[str]:
    """List of 'YYYY-MM' month strings spanning start_date..end_date inclusive."""
    months = []
    cur = start_date.replace(day=1)
    while cur <= end_date:
        months.append(cur.strftime("%Y-%m"))
        # Advance to the first of next month without a calendar dependency
        cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
    return months


def download_and_unzip_klines(
    symbol: str,
    interval: str,
    month: str,
    download_dir: str = "tmp",
    cache_dir: str = "cache",
) -> pl.DataFrame | None:
    """
    Download and unzip one month of Binance USDS-M futures klines (OHLCV bars).

    Unlike download_and_unzip (raw trades), Binance has already aggregated
    these into bars, so there is no group_by_dynamic step: one row in, one
    row out. A missing month (symbol not yet listed) returns None rather
    than raising, so a caller can skip pre-listing months for newer symbols.
    """
    cache_path_dir = Path(cache_dir)
    cache_path_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_path_dir / f"{symbol}-klines-{interval}-{month}.parquet"

    if cache_path.exists():
        return pl.read_parquet(cache_path)

    url = (
        f"https://data.binance.vision/data/futures/um/monthly/klines/"
        f"{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
    )

    download_path_dir = Path(download_dir)
    download_path_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_path_dir / f"{symbol}-{interval}-{month}.zip"
    csv_path = download_path_dir / f"{symbol}-{interval}-{month}.csv"

    df = None
    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 404:
            # Symbol wasn't listed yet in this month; not an error.
            return None
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            f.writelines(response.iter_content(chunk_size=8192))

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(download_path_dir)

        # Some archives ship a header row, some don't; detect and skip it.
        with open(csv_path) as f:
            first_line = f.readline()
        has_header = first_line.startswith("open_time")

        # schema_overrides forces dtypes during parsing itself. Relying on
        # inference (or a post-hoc .cast()) breaks on months where the first
        # rows of "volume"/"quote_volume" happen to be integer-valued, since
        # polars then infers Int64 and chokes on a later decimal value.
        df = pl.read_csv(
            csv_path,
            has_header=has_header,
            new_columns=None if has_header else list(KLINE_SCHEMA.keys()),
            schema_overrides=KLINE_SCHEMA,
        )

        df = df.with_columns(
            pl.from_epoch("open_time", time_unit="ms").alias("datetime")
        ).select(
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "count",
            "taker_buy_volume",
        )
        df.write_parquet(cache_path)
    finally:
        zip_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)

    return df


def download_klines_range(
    symbol: str,
    interval: str,
    start_date: str | datetime,
    end_date: str | datetime,
    download_dir: str = "tmp",
    cache_dir: str = "cache",
) -> pl.DataFrame:
    """
    Download (or load from cache) one symbol/interval's OHLCV bars for a date
    range, from Binance's pre-aggregated monthly klines files.

    This is the bar series directly, not raw trades: a year of hourly bars
    is tens of KB per file, vs. ~1GB/month for the raw trades feed the rest
    of this module downloads. Returned columns (datetime/open/high/low/close/
    volume) match research.load_ohlc_timeseries_range's output so downstream
    feature/model code doesn't need to know which path produced the bars.

    Caches one combined parquet per symbol/interval/date-range, alongside the
    per-month cache files download_and_unzip_klines already writes.
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)

    if start_date > end_date:
        raise ValueError("start_date must be before or equal to end_date")

    range_cache_path = (
        Path(cache_dir) / f"{symbol}-klines-{interval}-"
        f"{start_date.strftime('%Y-%m-%d')}-{end_date.strftime('%Y-%m-%d')}.parquet"
    )
    if range_cache_path.exists():
        return pl.read_parquet(range_cache_path)

    months = _month_range(start_date, end_date)
    dfs = []
    for month in tqdm(months, desc=f"Downloading {symbol} {interval} klines"):
        try:
            df = download_and_unzip_klines(
                symbol, interval, month, download_dir, cache_dir
            )
            if df is not None:
                dfs.append(df)
        except Exception as e:  # noqa: BLE001 - one bad month shouldn't stop the whole range
            tqdm.write(f"[ERROR] {symbol} {interval} {month}: {e}")

    if not dfs:
        raise ValueError(
            f"No kline data found for {symbol} {interval} in range "
            f"{start_date} to {end_date}"
        )

    result = (
        pl.concat(dfs)
        .sort("datetime")
        .unique(subset=["datetime"])
        .filter(
            (pl.col("datetime") >= start_date.replace(tzinfo=None))
            & (pl.col("datetime") <= end_date.replace(tzinfo=None))
        )
    )
    result.write_parquet(range_cache_path)
    return result


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
