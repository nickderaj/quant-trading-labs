"""Build sector regimes over this repo's parquet data.

Rewritten from ``ultron.apps.trading-labs.common.regime_report.builder``
(``../ultron/apps/trading-labs/common/regime_report/builder.py``) and
``common/prediction/data.py``'s ``load_basket_results`` (the source's only
code path that actually wires futures curves into per-symbol detection --
the report builder's own ``_basket_sector`` never passes a curve, which
would leave ``term_structure``/``carry`` permanently NaN). Same control flow
as the source (macro sector first, then baskets, errors collected rather
than raised so partial output survives) but sourcing bars/FRED/COT/curves
from ``regime.loaders`` (parquet) instead of ``DataStore``/``FredClient``/
``CotClient``/``DatabentoProvider``.

Deviation from production, disclosed per NEXT_PROMPT.md Sec3.3: this repo
has no COT data for the macro sector's ``cot_market`` ("E-MINI S&P 500"), so
the macro sector's ``cot`` input is always ``None`` here -- which, combined
with the source's ``macro.cot_noncomm`` bug (``requires={"macro"}`` but reads
``inputs.cot``), means the ``risk`` dimension is quietly reweighted onto its
other five indicators rather than skipped outright. This is preserved
bug-for-bug intentionally. Crude COT (the one series this repo does have)
is optionally wired into the ``oil_products`` basket via
``include_oil_products_cot``, off by default so the base run stays
comparable to production.

``cfg.history_start`` is accepted for interface parity with the source but
not used to bound any fetch here: ``regime.loaders`` reads each parquet's
full history unconditionally (there is nothing to fetch), and
``align_frame_to_daily`` handles arbitrary excess history ahead of a bar
index safely, so filtering it out first would only add complexity, not
change results.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import pandas as pd

from regime.aggregate import aggregate_scores, basket_labels
from regime.align import align_frame_to_daily
from regime.engine import RegimeEngine, RegimeInputs, RegimeResult
from regime.loaders import (
    COT_SYMBOL,
    load_bars,
    load_cot_raw,
    load_curve,
    load_fred_frame,
    net_positioning,
)
from regime.universe import BasketDefinition, RegimeUniverse


@dataclass(frozen=True)
class SectorRegime:
    name: str
    kind: Literal["macro", "basket"]
    latest: dict[str, tuple[str, float]]
    label_history: pd.DataFrame
    score_history: pd.DataFrame
    symbols_used: list[str]
    symbols_skipped: list[str]


@dataclass(frozen=True)
class RegimeReport:
    as_of: date | None
    sectors: list[SectorRegime]
    errors: list[str]


def _bars(symbol: str, as_of: date | None) -> pd.DataFrame:
    bars = load_bars(symbol)
    if as_of is not None:
        bars = bars.loc[: pd.Timestamp(as_of)]
    return bars


def _latest(scores: pd.DataFrame, labels: pd.DataFrame) -> dict[str, tuple[str, float]]:
    result: dict[str, tuple[str, float]] = {}
    for dimension in scores:
        score = scores[dimension].dropna()
        label = (
            labels[dimension].dropna()
            if dimension in labels
            else pd.Series(dtype="string")
        )
        if not score.empty and not label.empty:
            result[str(dimension)] = (str(label.iloc[-1]), float(score.iloc[-1]))
    return result


def _oil_products_cot(bar_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Crude COT, net-positioning derived and aligned with a 3-business-day
    publication lag, matching production's ``lags=3`` convention."""
    raw = net_positioning(load_cot_raw())
    return align_frame_to_daily(bar_index, raw[["noncomm_net_pct_oi"]], lags=3)


def _basket_sector(
    as_of: date | None,
    definition: BasketDefinition,
    min_bars: int,
    results: dict[tuple[str, str, bool], RegimeResult],
    include_oil_products_cot: bool,
) -> SectorRegime:
    engine = RegimeEngine.from_default(definition.config)
    per_symbol: dict[str, pd.DataFrame] = {}
    used: list[str] = []
    skipped: list[str] = []
    wire_cot = include_oil_products_cot and definition.name == "oil_products"
    for symbol in definition.symbols:
        key = (symbol, definition.config, wire_cot and symbol == COT_SYMBOL)
        detected = results.get(key)
        if detected is None:
            bars = _bars(symbol, as_of)
            if len(bars) < min_bars:
                skipped.append(symbol)
                continue
            curve = load_curve(symbol)
            if curve is not None:
                curve = align_frame_to_daily(pd.DatetimeIndex(bars.index), curve)
            cot = None
            if wire_cot and symbol == COT_SYMBOL:
                cot = _oil_products_cot(pd.DatetimeIndex(bars.index))
            detected = engine.detect(RegimeInputs(ohlcv=bars, curve=curve, cot=cot))
            results[key] = detected
        per_symbol[symbol] = detected.scores
        used.append(symbol)
    aggregate = aggregate_scores(per_symbol, min_coverage=0.5)
    labels = basket_labels(aggregate, engine.config)
    return SectorRegime(
        definition.name,
        "basket",
        _latest(aggregate, labels),
        labels,
        aggregate,
        used,
        skipped,
    )


def build_regime_report(
    cfg: RegimeUniverse,
    as_of: date | None = None,
    include_oil_products_cot: bool = False,
) -> RegimeReport:
    """Build every sector in ``cfg``, preserving partial output on failures.

    ``as_of=None`` (the default) uses each symbol's full available history --
    what Phase 1 needs to build the historical panel. Pass an explicit
    ``as_of`` to reproduce a single daily-report snapshot.
    """
    errors: list[str] = []
    sectors: list[SectorRegime] = []
    results: dict[tuple[str, str, bool], RegimeResult] = {}
    try:
        ohlcv = _bars(cfg.macro.index_symbol, as_of)
        if len(ohlcv) < cfg.min_bars_per_symbol:
            raise ValueError(f"{cfg.macro.index_symbol} has only {len(ohlcv)} bars")
        # FRED observations have midnight date indices whereas provider bars
        # carry market-close timestamps; align/fill onto the OHLCV index
        # before handing it to the regime engine (see regime.align docstring).
        macro = align_frame_to_daily(pd.DatetimeIndex(ohlcv.index), load_fred_frame())
        # This repo has no COT series for cfg.macro.cot_market ("E-MINI
        # S&P 500"); the macro sector's cot input is therefore always None
        # -- see module docstring for the resulting cot_noncomm behaviour.
        detected = RegimeEngine.from_default(cfg.macro.config).detect(
            RegimeInputs(ohlcv=ohlcv, macro=macro, cot=None)
        )
        sectors.append(
            SectorRegime(
                "Macro",
                "macro",
                detected.latest(),
                detected.labels,
                detected.scores,
                [cfg.macro.index_symbol],
                [],
            )
        )
    except Exception as exc:  # noqa: BLE001 - one bad sector shouldn't stop the whole report
        errors.append(f"Macro: {exc}")
    for definition in cfg.baskets:
        try:
            sectors.append(
                _basket_sector(
                    as_of,
                    definition,
                    cfg.min_bars_per_symbol,
                    results,
                    include_oil_products_cot,
                )
            )
        except Exception as exc:  # noqa: BLE001 - one bad sector shouldn't stop the whole report
            errors.append(f"{definition.name}: {exc}")
    return RegimeReport(as_of, sectors, errors)
