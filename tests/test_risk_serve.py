"""Tests for `src/risk/serve.py` (NEXT_PROMPT.md sec 7.2).

Uses synthetic per-product parquet files shaped like `risk.ingest`'s output
(a `date`/`log_return` frame), so this suite is fast and does not depend on
`risk.ingest.refresh()` having been run against real market data.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from risk.families import load_family_map
from risk.serve import build_snapshot, render_dashboard

SEED = 0


def _write_synthetic_product(data_dir: Path, product: str, family: str, n: int = 1000):
    rng = np.random.default_rng(hash(product) % (2**31))
    if family in ("ged", "hansen_skewt", "johnsonsu", "nig"):
        ret = rng.standard_t(6, n) * 0.02
    else:
        ret = rng.standard_normal(n) * 0.02
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    ret[0] = float(
        "nan"
    )  # first-day null, matching build_continuous_series's convention
    curve = pl.DataFrame(
        {
            "date": dates,
            "close_f1": 100 * np.exp(np.cumsum(np.nan_to_num(ret))),
            "log_return": ret,
        }
    )
    curve.write_parquet(data_dir / f"{product}.parquet")


class TestBuildSnapshot:
    def test_snapshot_covers_every_v1_product(self, tmp_path):
        fm = load_family_map("v1")
        for product, entry in fm.products.items():
            _write_synthetic_product(tmp_path, product, entry["family"])
        snap = build_snapshot(as_of="2026-08-11", data_dir=tmp_path)
        assert set(snap["products"].keys()) == set(fm.products.keys())
        assert snap["as_of"] == "2026-08-11"
        assert snap["family_map_version"] == "v1"

    def test_missing_product_data_is_reported_not_silently_dropped(self, tmp_path):
        fm = load_family_map("v1")
        # only ship data for two products
        for product in ["CL", "GC"]:
            _write_synthetic_product(tmp_path, product, fm.products[product]["family"])
        snap = build_snapshot(as_of=None, data_dir=tmp_path)
        assert snap["products"]["CL"]["status"] == "ok"
        assert snap["products"]["ZC"]["status"] == "no_data"
        assert (
            "family" in snap["products"]["ZC"]
        )  # still names the family, just no data

    def test_product_snapshot_has_var_es_and_monitor_status(self, tmp_path):
        fm = load_family_map("v1")
        _write_synthetic_product(tmp_path, "CL", fm.products["CL"]["family"])
        snap = build_snapshot(data_dir=tmp_path)
        cl = snap["products"]["CL"]
        assert cl["status"] == "ok"
        assert "0.01" in cl["var_es"]
        for h in (1, 5, 10):
            assert f"var_h{h}" in cl["var_es"]["0.01"]
            assert f"es_h{h}" in cl["var_es"]["0.01"]
            assert cl["var_es"]["0.01"][f"es_h{h}"] >= cl["var_es"]["0.01"][f"var_h{h}"]
        assert cl["monitor"]["status"] in ("ok", "warn", "breach")
        assert len(cl["recent_series"]["returns"]) <= 250
        assert len(cl["recent_series"]["dates"]) == len(cl["recent_series"]["returns"])

    def test_book_level_block_present_with_multiple_products(self, tmp_path):
        fm = load_family_map("v1")
        for product in ["CL", "GC", "SI"]:
            _write_synthetic_product(tmp_path, product, fm.products[product]["family"])
        snap = build_snapshot(data_dir=tmp_path)
        book = snap["book"]
        assert book["n_products"] == 3
        assert set(book["portfolio_risk"].keys()) == {"empirical", "gaussian", "t"}
        for dep in ("empirical", "gaussian", "t"):
            assert book["portfolio_risk"][dep]["var_01"] > 0

    def test_book_omitted_with_fewer_than_two_products(self, tmp_path):
        fm = load_family_map("v1")
        _write_synthetic_product(tmp_path, "CL", fm.products["CL"]["family"])
        snap = build_snapshot(data_dir=tmp_path)
        assert snap["book"]["n_products"] <= 1
        assert "portfolio_risk" not in snap["book"]

    def test_validated_envelope_states_the_16_products(self, tmp_path):
        snap = build_snapshot(data_dir=tmp_path)
        assert len(snap["validated_envelope"]["products"]) == 16
        assert "claim" in snap["validated_envelope"]


class TestRenderDashboard:
    def test_placeholder_is_replaced_with_real_json(self, tmp_path):
        fm = load_family_map("v1")
        for product in ["CL", "GC"]:
            _write_synthetic_product(tmp_path, product, fm.products[product]["family"])
        snap = build_snapshot(as_of="2026-08-11", data_dir=tmp_path)

        html = render_dashboard(snapshot=snap)
        assert "__RISK_SNAPSHOT_JSON__" not in html
        assert '<script id="risk-snapshot" type="application/json">' in html

    def test_inlined_payload_round_trips_through_json(self, tmp_path):
        import json
        import re

        fm = load_family_map("v1")
        for product in ["CL", "GC", "SI"]:
            _write_synthetic_product(tmp_path, product, fm.products[product]["family"])
        snap = build_snapshot(data_dir=tmp_path)

        html = render_dashboard(snapshot=snap)
        m = re.search(
            r'<script id="risk-snapshot" type="application/json">(.*?)</script>',
            html,
            re.DOTALL,
        )
        assert m is not None
        parsed = json.loads(m.group(1))
        assert set(parsed["products"].keys()) == set(snap["products"].keys())
        assert parsed["book"]["n_products"] == snap["book"]["n_products"]

    def test_self_contained_no_external_references(self):
        html = render_dashboard(snapshot={"as_of": None, "products": {}, "book": {}})
        lowered = html.lower()
        for forbidden in ("cdn.", "http://", "https://", '<script src="'):
            assert forbidden not in lowered, forbidden

    def test_writes_to_out_path(self, tmp_path):
        out = tmp_path / "dashboard.html"
        render_dashboard(
            snapshot={"as_of": None, "products": {}, "book": {}}, out_path=out
        )
        assert out.exists()
        assert "__RISK_SNAPSHOT_JSON__" not in out.read_text()

    def test_raises_if_template_missing_placeholder(self, tmp_path):
        bad_template = tmp_path / "bad.html"
        bad_template.write_text("<html><body>no placeholder here</body></html>")
        with pytest.raises(ValueError, match="placeholder"):
            render_dashboard(
                snapshot={"as_of": None, "products": {}, "book": {}},
                template_path=bad_template,
            )
