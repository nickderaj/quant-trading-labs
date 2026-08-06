"""Registry for pluggable regime indicators.

Ported verbatim from ``ultron_finance.regime.registry``
(``../ultron/libs/finance/src/ultron_finance/regime/registry.py``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import pandas as pd

if TYPE_CHECKING:
    from regime.engine import RegimeInputs


class RegimeIndicator(Protocol):
    def __call__(self, inputs: RegimeInputs, **params: Any) -> pd.Series: ...


@dataclass(frozen=True)
class IndicatorMeta:
    key: str
    requires: frozenset[str]
    default_params: Mapping[str, Any]
    description: str


F = TypeVar("F", bound=Callable[..., pd.Series])
_REGISTRY: dict[str, tuple[RegimeIndicator, IndicatorMeta]] = {}


def register(
    key: str, requires: Iterable[str], description: str = "", **default_params: Any
) -> Callable[[F], F]:
    def decorator(function: F) -> F:
        if key in _REGISTRY:
            raise ValueError(f"Regime indicator already registered: {key}")
        _REGISTRY[key] = (
            function,
            IndicatorMeta(key, frozenset(requires), default_params, description),
        )
        return function

    return decorator


def get(key: str) -> tuple[RegimeIndicator, IndicatorMeta]:
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown regime indicator {key!r}; available: {sorted(_REGISTRY)}"
        ) from exc


def available() -> dict[str, IndicatorMeta]:
    return {key: meta for key, (_, meta) in _REGISTRY.items()}
