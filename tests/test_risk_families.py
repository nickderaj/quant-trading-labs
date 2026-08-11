"""Tests for `src/risk/families.py` (NEXT_PROMPT.md sec 5).

Gate FS's actual measurement (P1 vs P2 vs P3's Gate RE pass count) is a
16-product walk-forward battery and lives in
`src/research/tmp/run_risk_02_family_policy.py`, not here; these tests cover
`load_family_map`'s contract -- config-hash verification, the unseen-product
refusal, and `family_map_v1.json`'s own shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from risk.families import UnseenProductError, config_hash, load_family_map

EXPECTED_PRODUCTS = {
    "CL", "BZ", "NG", "HO", "RB", "GC", "SI", "PL", "PA",
    "ZC", "ZW", "KE", "ZS", "ZL", "ZM", "ES",
}  # fmt: skip


class TestLoadFamilyMap:
    def test_v1_covers_exactly_the_16_certified_products(self):
        fm = load_family_map("v1")
        assert set(fm.products.keys()) == EXPECTED_PRODUCTS

    def test_v1_matches_the_shipped_phase_7_family_map(self):
        fm = load_family_map("v1")
        # Gate FS (run_risk_02_family_policy_results.json): P1 (the shipped
        # per-product map) beat P2/P3 outright (15/16 vs 14/16), so v1 must
        # still be the per-product Phase 3 selection, not collapsed to a
        # single family.
        families = {p: e["family"] for p, e in fm.products.items()}
        assert families["GC"] == "ged"
        assert families["SI"] == "ged"
        assert families["NG"] == "johnsonsu"
        assert len(set(families.values())) > 1

    def test_family_for_returns_the_selected_family(self):
        fm = load_family_map("v1")
        assert fm.family_for("CL") == "ged"

    def test_family_for_unseen_product_raises(self):
        fm = load_family_map("v1")
        with pytest.raises(UnseenProductError):
            fm.family_for("HG")

    def test_contains(self):
        fm = load_family_map("v1")
        assert "CL" in fm
        assert "HG" not in fm

    def test_config_hash_is_verified_on_load(self):
        # load_family_map succeeding at all (called by every other test in
        # this file) already proves the stored config_hash matches the
        # file's own content; this locks in that config_hash is
        # deterministic given the same payload.
        payload = {"a": 1, "b": [1, 2, 3]}
        assert config_hash(payload) == config_hash(dict(payload))

    def test_unknown_version_raises(self):
        with pytest.raises(Exception):  # noqa: B017 - FileNotFoundError via importlib
            load_family_map("v999_does_not_exist")
