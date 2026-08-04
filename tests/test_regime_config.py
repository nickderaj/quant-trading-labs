from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regime.config import RegimeConfig

CONFIGS = Path(__file__).resolve().parents[1] / "src" / "regime" / "configs"


def test_yaml_round_trip_hash_and_disabled_stub() -> None:
    config = RegimeConfig.from_yaml(CONFIGS / "commodity_default.yaml")
    assert config.name == "commodity_default"
    assert any(not item.enabled for item in config.dimensions)
    assert config.config_hash() == config.config_hash()


def test_rejects_non_exhaustive_bands(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "version: '1'\nname: bad\nasset_class: test\ndimensions:\n"
        "  - key: trend\n    indicators: [{name: trend.adx}]\n"
        "    bands: [{label: bear, lower: -1, upper: 0}, {label: bull, lower: 0}]\n"
    )
    with pytest.raises(ValueError, match="ordered, contiguous"):
        RegimeConfig.from_yaml(path)
