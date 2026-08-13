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
from risk.serve import CRYPTO_FAMILY_MAP_VERSION, build_snapshot, render_dashboard

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

    def test_a_single_breaching_window_never_shows_as_breach(self, tmp_path):
        """Persistence rule (risk_engine_preregistration.json): a product
        can't reach "breach" on the very first monitoring run against a
        fresh data_dir -- it takes k=2 consecutive breaching runs. Regardless
        of what the raw, uncorrected battery says about this synthetic
        (iid-normal, well-specified) data, the first-ever run must not
        report "breach" for anything."""
        fm = load_family_map("v1")
        for product, entry in fm.products.items():
            _write_synthetic_product(tmp_path, product, entry["family"])
        snap = build_snapshot(data_dir=tmp_path)
        statuses = {
            p: v["monitor"]["status"]
            for p, v in snap["products"].items()
            if v["status"] == "ok"
        }
        assert "breach" not in statuses.values()

    def test_two_consecutive_runs_with_the_same_raw_breach_escalate(
        self, tmp_path, monkeypatch
    ):
        """Wiring test for `build_snapshot` -> `_attach_monitor_status` ->
        `apply_persistence`, independent of whether any particular synthetic
        seed happens to breach: force `CalibrationMonitor.evaluate_batch` to
        report a raw CL coverage breach every call, and check that streak
        state actually survives across two separate `build_snapshot` calls
        against the same `data_dir` (a fresh `data_dir`/`tmp_path` per test
        must not carry over another test's history, but *this* directory's
        own state must persist run to run)."""
        import risk.calibration as calibration_mod
        from risk.calibration import CalibrationStatus, LevelResult

        fm = load_family_map("v1")
        for product, entry in fm.products.items():
            _write_synthetic_product(tmp_path, product, entry["family"])

        real_evaluate_batch = calibration_mod.CalibrationMonitor.evaluate_batch

        def _forced_evaluate_batch(self, product_inputs, **kwargs):
            raw = real_evaluate_batch(self, product_inputs, **kwargs)
            lr = LevelResult(
                level=0.01,
                n=raw["CL"].levels[0.01].n,
                observed_rate=0.05,
                expected_rate=0.01,
                kupiec_p=0.001,
                independence_p=0.8,
                cc_p=0.01,
                max_cluster_length=2,
                coverage_breach=True,
                clustering_breach=False,
            )
            raw["CL"] = CalibrationStatus(
                product="CL",
                levels={0.01: lr},
                acerbi=None,
                status="breach",
                failure_mode="coverage",
            )
            return raw

        monkeypatch.setattr(
            calibration_mod.CalibrationMonitor, "evaluate_batch", _forced_evaluate_batch
        )

        snap1 = build_snapshot(data_dir=tmp_path)
        assert snap1["products"]["CL"]["monitor"]["status"] == "warn"
        assert snap1["products"]["CL"]["monitor"]["failure_mode"] == "coverage"

        snap2 = build_snapshot(data_dir=tmp_path)
        assert snap2["products"]["CL"]["monitor"]["status"] == "breach"
        assert snap2["products"]["CL"]["monitor"]["failure_mode"] == "coverage"


class TestCryptoSection:
    """The crypto panel (`family_map_crypto_v1`) is additive and must never
    dilute the validated futures envelope's own invariants."""

    def test_crypto_never_leaks_into_the_validated_products_dict(self, tmp_path):
        fm = load_family_map("v1")
        for product, entry in fm.products.items():
            _write_synthetic_product(tmp_path, product, entry["family"])
        # also drop synthetic crypto data in tmp_path/crypto -- it must still
        # never show up under snap["products"] or snap["validated_envelope"]
        crypto_fm = load_family_map(CRYPTO_FAMILY_MAP_VERSION)
        crypto_dir = tmp_path / "crypto"
        crypto_dir.mkdir()
        for symbol, entry in crypto_fm.products.items():
            _write_synthetic_product(crypto_dir, symbol, entry["family"])

        snap = build_snapshot(data_dir=tmp_path)
        assert set(snap["products"].keys()) == set(fm.products.keys())
        assert set(snap["validated_envelope"]["products"]) == set(fm.products.keys())
        assert set(snap["crypto_products"].keys()) == set(crypto_fm.products.keys())

    def test_crypto_products_report_no_data_without_ingested_files(self, tmp_path):
        # only futures data present; nothing under tmp_path/crypto
        fm = load_family_map("v1")
        _write_synthetic_product(tmp_path, "CL", fm.products["CL"]["family"])
        snap = build_snapshot(data_dir=tmp_path)
        crypto_fm = load_family_map(CRYPTO_FAMILY_MAP_VERSION)
        assert set(snap["crypto_envelope"]["products"]) == set(
            crypto_fm.products.keys()
        )
        for symbol in crypto_fm.products:
            assert snap["crypto_products"][symbol]["status"] == "no_data"
            assert snap["crypto_products"][symbol]["family"] == crypto_fm.family_for(
                symbol
            )

    def test_crypto_envelope_claim_says_it_is_not_validated(self, tmp_path):
        snap = build_snapshot(data_dir=tmp_path)
        claim = snap["crypto_envelope"]["claim"]
        assert "NOT the validated envelope" in claim

    def test_a_crypto_product_with_data_gets_fitted_and_monitored(self, tmp_path):
        crypto_fm = load_family_map(CRYPTO_FAMILY_MAP_VERSION)
        crypto_dir = tmp_path / "crypto"
        crypto_dir.mkdir()
        _write_synthetic_product(crypto_dir, "BTCUSDT", crypto_fm.family_for("BTCUSDT"))
        snap = build_snapshot(data_dir=tmp_path)
        btc = snap["crypto_products"]["BTCUSDT"]
        assert btc["status"] == "ok"
        assert "0.01" in btc["var_es"]
        assert btc["monitor"]["status"] in ("ok", "warn", "breach")
        # untouched sibling symbols with no data still report honestly
        assert snap["crypto_products"]["ETHUSDT"]["status"] == "no_data"

    def test_crypto_data_dir_can_be_overridden_independently(self, tmp_path):
        futures_dir = tmp_path / "futures"
        futures_dir.mkdir()
        crypto_dir = tmp_path / "somewhere_else"
        crypto_dir.mkdir()
        fm = load_family_map("v1")
        _write_synthetic_product(futures_dir, "CL", fm.products["CL"]["family"])
        crypto_fm = load_family_map(CRYPTO_FAMILY_MAP_VERSION)
        _write_synthetic_product(crypto_dir, "ETHUSDT", crypto_fm.family_for("ETHUSDT"))
        snap = build_snapshot(data_dir=futures_dir, crypto_data_dir=crypto_dir)
        assert snap["crypto_products"]["ETHUSDT"]["status"] == "ok"


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
        assert set(parsed["crypto_products"].keys()) == set(
            snap["crypto_products"].keys()
        )

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
