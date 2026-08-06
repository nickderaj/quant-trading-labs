from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regime.universe import load_regime_universe


def test_load_regime_universe_parses_macro_baskets_and_micro_complexes() -> None:
    universe = load_regime_universe(
        {
            "regime": {
                "report": {
                    "enabled": True,
                    "history_start": "2019-01-01",
                    "min_bars_per_symbol": 300,
                },
                "macro": {
                    "config": "macro_default",
                    "index_symbol": "ES=F",
                    "cot_market": "E-MINI",
                },
                "macro_commodities": {
                    "config": "commodity_default",
                    "symbols": ["CL=F"],
                },
                "macro_fx": {"config": "fx_default", "symbols": ["6E=F"]},
                "micro": {
                    "oil_products": {
                        "config": "commodity_default",
                        "symbols": ["CL=F", "RB=F"],
                    }
                },
            }
        }
    )

    assert universe.history_start == date(2019, 1, 1)
    assert universe.macro.index_symbol == "ES=F"
    assert [basket.name for basket in universe.baskets] == [
        "Commodities",
        "FX",
        "oil products",
    ]
    assert universe.symbols == {"ES=F", "CL=F", "RB=F", "6E=F"}


def test_load_regime_universe_matches_this_repos_configs_yaml() -> None:
    import yaml

    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "regime"
        / "configs"
        / "universe.yaml"
    )
    universe = load_regime_universe(yaml.safe_load(path.read_text()))

    assert universe.macro.index_symbol == "ES=F"
    assert universe.macro.cot_market == "E-MINI S&P 500"
    assert {basket.name for basket in universe.baskets} == {
        "Commodities",
        "FX",
        "oil products",
        "natgas",
        "soy complex",
        "grains",
        "softs",
        "precious",
        "base metals",
        "meats",
    }
    assert len(universe.symbols) == 27
