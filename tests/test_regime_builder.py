"""Integration tests for regime.builder over this repo's real parquet data.

Unlike the source's `test_builder.py` (which mocks a DataStore + FredClient),
this repo has no database -- builder.py reads real parquet directly, so
these tests exercise it against a deliberately tiny slice of the actual
universe (one basket, or the macro sector alone) rather than a fixture.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from regime.builder import build_regime_report
from regime.universe import load_regime_universe

UNIVERSE_YAML = (
    Path(__file__).resolve().parents[1] / "src" / "regime" / "configs" / "universe.yaml"
)


def _cfg():
    return load_regime_universe(yaml.safe_load(UNIVERSE_YAML.read_text()))


def test_macro_sector_builds_from_real_bars_and_fred() -> None:
    cfg = dataclasses.replace(_cfg(), baskets=())
    report = build_regime_report(cfg)

    assert report.errors == []
    assert len(report.sectors) == 1
    macro = report.sectors[0]
    assert macro.kind == "macro"
    assert {"risk", "yield_curve", "credit"} <= set(macro.latest)
    assert not macro.score_history.empty


def test_basket_sector_wires_curve_for_symbols_that_have_one() -> None:
    cfg = _cfg()
    small = dataclasses.replace(
        cfg, baskets=tuple(b for b in cfg.baskets if b.name == "base metals")
    )
    report = build_regime_report(small)

    assert report.errors == []
    basket = next(s for s in report.sectors if s.kind == "basket")
    assert basket.symbols_used == ["HG=F"]
    # HG=F has a curve file (regime.loaders.CURVE_SYMBOLS), so term_structure
    # and carry should have scored bars in the curve's covered window.
    assert basket.score_history["term_structure"].notna().any()
    assert basket.score_history["carry"].notna().any()


def test_oil_products_cot_is_opt_in_and_off_by_default() -> None:
    cfg = _cfg()
    small = dataclasses.replace(
        cfg, baskets=tuple(b for b in cfg.baskets if b.name == "oil products")
    )
    default_report = build_regime_report(small)
    cot_report = build_regime_report(small, include_oil_products_cot=True)

    assert default_report.errors == cot_report.errors == []
    default_basket = next(s for s in default_report.sectors if s.kind == "basket")
    cot_basket = next(s for s in cot_report.sectors if s.kind == "basket")
    # The macro.cot_noncomm indicator only lives in macro_default's `risk`
    # dimension, not commodity_default -- so wiring COT into oil_products
    # (a commodity_default basket) cannot change its scores. This asserts
    # the flag is a genuine no-op for this basket's config, matching
    # NEXT_PROMPT.md Sec3.3's bug-for-bug preservation intent.
    assert default_basket.score_history.equals(cot_basket.score_history)
