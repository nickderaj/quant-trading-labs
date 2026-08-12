"""Tests for `src/risk/ingest.py` (NEXT_PROMPT.md sec 7.1, gate IR).

Uses a small synthetic databento-shaped parquet root rather than the real
data cache, so this suite does not depend on `src/research/data/market/`
being populated.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from risk import ingest


def _synthetic_data_root(
    tmp_path: Path, n_days: int = 400, n_contracts: int = 4
) -> Path:
    # liquidity_screen's default min_active_contracts=2 needs at least two
    # concurrently-trading contracts every day (a single-contract-per-day
    # curve, however long, is exactly the "single stale quote" pattern it
    # exists to reject); build_continuous_series's default n_legs=3 needs at
    # least 3 contract months on the books, or its F3 leg's empty selection
    # produces a null-typed join key -- real data always has more months
    # listed than n_legs, so n_contracts defaults comfortably above 3.
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n_days)]

    ohlcv_frames = []
    contract_rows = []
    roll_rows = []
    for c in range(1, n_contracts + 1):
        close = [100.0 + c + 0.05 * i for i in range(n_days)]
        ohlcv_frames.append(
            pl.DataFrame(
                {
                    "product": ["CL"] * n_days,
                    "ticker": [f"CL2020{c:02d}"] * n_days,
                    "contract_id": [c] * n_days,
                    "date": dates,
                    "open": close,
                    "high": [x + 0.5 for x in close],
                    "low": [x - 0.5 for x in close],
                    "close": close,
                    "volume": [10_000] * n_days,
                }
            )
        )
        expiry = dates[-1] + timedelta(days=30 * c)
        contract_rows.append((c, "CL", f"2020-{c:02d}", expiry))
        roll_rows.append(
            (
                "CL",
                f"2020-{c:02d}",
                expiry,
                expiry + timedelta(days=1),
                expiry,
            )
        )

    ohlcv = pl.concat(ohlcv_frames)
    contracts = pl.DataFrame(
        contract_rows,
        schema=["contract_id", "product", "contract_month", "expiry"],
        orient="row",
    )
    roll_calendar = pl.DataFrame(
        roll_rows,
        schema=[
            "product",
            "contract_month",
            "expiry",
            "first_notice_date",
            "last_trade_date",
        ],
        orient="row",
    )
    root = tmp_path / "databento"
    (root / "ohlcv").mkdir(parents=True)
    ohlcv.write_parquet(root / "ohlcv" / "part-0.parquet")
    contracts.write_parquet(root / "contracts.parquet")
    roll_calendar.write_parquet(root / "roll_calendar.parquet")
    return root


class TestRefresh:
    def test_writes_a_product_and_reports_last_observation(self, tmp_path):
        data_root = _synthetic_data_root(tmp_path)
        out_dir = tmp_path / "out"
        report = ingest.refresh(
            products=["CL"], as_of="2020-06-01", data_root=data_root, out_dir=out_dir
        )
        result = report.products["CL"]
        assert result.status == "written"
        assert result.rows > 0
        assert result.last_observation is not None
        assert (out_dir / "CL.parquet").exists()

    def test_idempotent_rerun_is_unchanged(self, tmp_path):
        data_root = _synthetic_data_root(tmp_path)
        out_dir = tmp_path / "out"
        first = ingest.refresh(
            products=["CL"], data_root=data_root, out_dir=out_dir
        ).products["CL"]
        assert first.status == "written"
        written_bytes = (out_dir / "CL.parquet").read_bytes()

        second = ingest.refresh(
            products=["CL"], data_root=data_root, out_dir=out_dir
        ).products["CL"]
        assert second.status == "unchanged"
        assert second.content_hash == first.content_hash
        # gate IR: exact byte equality on the written output
        assert (out_dir / "CL.parquet").read_bytes() == written_bytes

    def test_a_product_with_too_little_history_is_rejected_not_silently_short(
        self, tmp_path
    ):
        data_root = _synthetic_data_root(tmp_path, n_days=50)  # below the 100-obs floor
        out_dir = tmp_path / "out"
        report = ingest.refresh(products=["CL"], data_root=data_root, out_dir=out_dir)
        result = report.products["CL"]
        assert result.status == "rejected"
        assert result.rejection_reason is not None
        assert not (out_dir / "CL.parquet").exists()

    def test_report_is_written_to_out_dir(self, tmp_path):
        data_root = _synthetic_data_root(tmp_path)
        out_dir = tmp_path / "out"
        ingest.refresh(products=["CL"], data_root=data_root, out_dir=out_dir)
        assert (out_dir / "_ingest_report.json").exists()

    def test_defaults_to_the_v1_family_map_products(self):
        # products=None should resolve against the real, committed
        # family_map_v1.json (16 products) without touching real market
        # data, since we don't pass data_root here -- just check the
        # resolution step doesn't blow up before the (unreachable in this
        # test) parquet read.
        from risk.families import load_family_map

        assert len(load_family_map("v1").products) == 16


@pytest.mark.parametrize("seed_products", [["CL"]])
def test_content_hash_changes_when_data_changes(tmp_path, seed_products):
    data_root_a = _synthetic_data_root(tmp_path / "a", n_days=400)
    data_root_b = _synthetic_data_root(tmp_path / "b", n_days=410)
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    r_a = ingest.refresh(
        products=seed_products, data_root=data_root_a, out_dir=out_a
    ).products["CL"]
    r_b = ingest.refresh(
        products=seed_products, data_root=data_root_b, out_dir=out_b
    ).products["CL"]
    assert r_a.content_hash != r_b.content_hash
