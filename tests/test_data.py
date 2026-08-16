"""Tests for src/data.py's Binance fetch helpers.

Network-free: `requests.get` is monkeypatched to a fake that records the
requested URL and returns a tiny in-memory zip, so these tests check URL
construction and cache-path behaviour, not real downloads.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import data


class _FakeResponse:
    def __init__(self, url: str, status_code: int, content: bytes):
        self.url = url
        self.status_code = status_code
        self._content = content

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int = 8192):
        yield self._content


def _fake_kline_zip(csv_body: str) -> bytes:
    """Binance's real zips always name their internal CSV member
    '{symbol}-{interval}-{month}.csv', with no market tag -- mirroring
    that here is what caught the real extraction-path bug this test
    module exists to guard against.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("BTCUSDT-8h-2021-08.csv", csv_body)
    return buf.getvalue()


_ONE_ROW_CSV = (
    "1627776000000,41461.84,42599.00,41120.00,41701.26,16582.71,"
    "1627804799999,695087910.37,537592,8437.33,353800700.80,0\n"
)


def test_download_and_unzip_klines_default_market_url_is_unchanged(
    tmp_path: Path, monkeypatch
) -> None:
    """market defaults to 'futures/um', so the URL and cache filename this
    function has always used must not move -- every existing futures/um
    cache file on disk depends on this staying byte-identical.
    """
    seen_urls: list[str] = []

    def fake_get(url: str, stream: bool = True, timeout: int = 30):
        seen_urls.append(url)
        return _FakeResponse(url, 200, _fake_kline_zip(_ONE_ROW_CSV))

    monkeypatch.setattr(data.requests, "get", fake_get)

    df = data.download_and_unzip_klines(
        "BTCUSDT",
        "8h",
        "2021-08",
        download_dir=str(tmp_path / "tmp"),
        cache_dir=str(tmp_path / "cache"),
    )

    assert seen_urls == [
        "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/8h/BTCUSDT-8h-2021-08.zip"
    ]
    assert df is not None and len(df) == 1
    assert (tmp_path / "cache" / "BTCUSDT-klines-8h-2021-08.parquet").exists()


def test_download_and_unzip_klines_market_spot_hits_spot_tree(
    tmp_path: Path, monkeypatch
) -> None:
    """market='spot' must read from the spot tree, not futures/um, and must
    cache under a distinct filename so a later futures/um call for the same
    symbol/interval/month is not served stale spot data (the 009 blocker
    this notebook exists to resolve depends on the two never colliding).
    """
    seen_urls: list[str] = []

    def fake_get(url: str, stream: bool = True, timeout: int = 30):
        seen_urls.append(url)
        return _FakeResponse(url, 200, _fake_kline_zip(_ONE_ROW_CSV))

    monkeypatch.setattr(data.requests, "get", fake_get)

    df = data.download_and_unzip_klines(
        "BTCUSDT",
        "8h",
        "2021-08",
        download_dir=str(tmp_path / "tmp"),
        cache_dir=str(tmp_path / "cache"),
        market="spot",
    )

    assert seen_urls == [
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/8h/BTCUSDT-8h-2021-08.zip"
    ]
    assert df is not None and len(df) == 1
    spot_cache = tmp_path / "cache" / "BTCUSDT-klines-spot-8h-2021-08.parquet"
    futures_cache = tmp_path / "cache" / "BTCUSDT-klines-8h-2021-08.parquet"
    assert spot_cache.exists()
    assert not futures_cache.exists()


def test_download_and_unzip_klines_market_cache_hit_skips_network(
    tmp_path: Path, monkeypatch
) -> None:
    """A pre-existing spot-tagged cache file must be served without any
    network call at all -- the resumability property Phase 1 depends on.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    import polars as pl

    pl.DataFrame({"datetime": [1], "open": [1.0]}).write_parquet(
        cache_dir / "BTCUSDT-klines-spot-8h-2021-08.parquet"
    )

    def fake_get(*args, **kwargs):
        raise AssertionError("network should not be called on a cache hit")

    monkeypatch.setattr(data.requests, "get", fake_get)

    df = data.download_and_unzip_klines(
        "BTCUSDT",
        "8h",
        "2021-08",
        download_dir=str(tmp_path / "tmp"),
        cache_dir=str(cache_dir),
        market="spot",
    )
    assert df is not None and len(df) == 1


def test_download_and_unzip_klines_404_returns_none_for_any_market(
    tmp_path: Path, monkeypatch
) -> None:
    """A 404 (symbol not listed that month) is data, not an error, for
    spot exactly as it already is for futures/um.
    """

    def fake_get(url: str, stream: bool = True, timeout: int = 30):
        return _FakeResponse(url, 404, b"")

    monkeypatch.setattr(data.requests, "get", fake_get)

    df = data.download_and_unzip_klines(
        "NOSUCHSYM",
        "8h",
        "2021-08",
        download_dir=str(tmp_path / "tmp"),
        cache_dir=str(tmp_path / "cache"),
        market="spot",
    )
    assert df is None
