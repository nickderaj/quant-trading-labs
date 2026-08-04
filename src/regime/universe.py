"""Validated declarative universe for the daily regime report.

Adapted from ``ultron.apps.trading-labs.common.regime_report.universe``
(``../ultron/apps/trading-labs/common/regime_report/universe.py``). The
dataclasses and parsing logic are kept as-is; ``DATABENTO_CURVE_SYMBOLS`` is
dropped -- this repo's curves are already-built parquet
(``src/research/data/market/research/{cl,gc,hg,ng,si}_curve.parquet``), not
fetched live via Databento.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class BasketDefinition:
    name: str
    config: str
    symbols: tuple[str, ...]


@dataclass(frozen=True)
class MacroDefinition:
    config: str
    index_symbol: str
    cot_market: str


@dataclass(frozen=True)
class RegimeUniverse:
    enabled: bool
    history_start: date
    min_bars_per_symbol: int
    macro: MacroDefinition
    baskets: tuple[BasketDefinition, ...]

    @property
    def symbols(self) -> set[str]:
        return {symbol for basket in self.baskets for symbol in basket.symbols} | {
            self.macro.index_symbol
        }


def _basket(name: str, raw: dict[str, Any]) -> BasketDefinition:
    return BasketDefinition(name, str(raw["config"]), tuple(map(str, raw["symbols"])))


def load_regime_universe(config: dict[str, Any]) -> RegimeUniverse:
    """Parse the ``regime`` section (e.g. loaded from ``configs/universe.yaml``)."""
    raw = config.get("regime", config)
    report = raw["report"]
    macro_raw = raw["macro"]
    macro = MacroDefinition(
        str(macro_raw["config"]), str(macro_raw["index_symbol"]), str(macro_raw["cot_market"])
    )
    baskets = [
        _basket("Commodities", raw["macro_commodities"]),
        _basket("FX", raw["macro_fx"]),
    ]
    baskets.extend(_basket(name.replace("_", " "), value) for name, value in raw["micro"].items())
    return RegimeUniverse(
        bool(report.get("enabled", True)),
        date.fromisoformat(str(report["history_start"])),
        int(report["min_bars_per_symbol"]),
        macro,
        tuple(baskets),
    )
