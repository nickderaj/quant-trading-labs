"""Shared library for notebook 015 (trend ceiling and independent
validation of the market-regime engine's yield_curve/term_structure/carry
dimensions). NEXT_PROMPT.md is the spec; this module holds the pieces every
phase script reuses so the truncation/panel/target logic isn't duplicated
and can't drift between phases.

Ground rule 1 (NEXT_PROMPT.md sec2): every dataset in this notebook is
truncated at 2024-12-31 inclusive, before any model sees it, asserted once
here rather than trusted to each phase.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast, overload

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm

from regime.loaders import load_bars

TRUNCATION = pd.Timestamp("2024-12-31")  # inclusive; both holdouts start after this
DATABENTO_DIR = Path("src/research/data/market/databento/ohlcv")


# --------------------------------------------------------------------------- #
# Ground rule 1: truncation
# --------------------------------------------------------------------------- #
@overload
def truncate(obj: pd.Series) -> pd.Series: ...
@overload
def truncate(obj: pd.DataFrame) -> pd.DataFrame: ...
def truncate(obj: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Truncate a date-indexed series/frame to <= TRUNCATION, and assert it."""
    out = obj[obj.index <= TRUNCATION]
    assert out.index.max() <= TRUNCATION, "truncation invariant violated"
    return out


def assert_truncated(*objs: pd.Series | pd.DataFrame) -> None:
    for obj in objs:
        if len(obj) == 0:
            continue
        assert obj.index.max() <= TRUNCATION, (
            f"holdout leak: index extends to {obj.index.max()}, past {TRUNCATION.date()}"
        )


# --------------------------------------------------------------------------- #
# Universe (NEXT_PROMPT.md sec5.1, sec3.1) -- identical to 014's 8 micro
# baskets, from src/regime/configs/universe.yaml
# --------------------------------------------------------------------------- #
BASKET_SYMBOLS: dict[str, list[str]] = {
    "oil products": ["CL=F", "BZ=F", "RB=F", "HO=F"],
    "natgas": ["NG=F"],
    "soy complex": ["ZS=F", "ZM=F", "ZL=F"],
    "grains": ["ZC=F", "ZW=F"],
    "softs": ["KC=F", "SB=F", "CT=F"],
    "precious": ["GC=F", "SI=F", "PL=F", "PA=F"],
    "base metals": ["HG=F"],
    "meats": ["LE=F", "HE=F"],
}
PANEL_L_SYMBOLS: list[str] = [s for syms in BASKET_SYMBOLS.values() for s in syms]
SYMBOL_TO_BASKET: dict[str, str] = {
    s: b for b, syms in BASKET_SYMBOLS.items() for s in syms
}

# Panel-D: the 16 databento products present (NEXT_PROMPT.md sec5.1). ES is
# dropped for the trend target (equity index, not a commodity -- sec5.1);
# it stays in PANEL_D_PRODUCTS_ALL for completeness/disclosure but is
# excluded from PANEL_D_SYMBOLS, which is what Track B actually pools.
PANEL_D_PRODUCTS_ALL: list[str] = [
    "BZ",
    "CL",
    "ES",
    "GC",
    "HO",
    "KE",
    "NG",
    "PA",
    "PL",
    "RB",
    "SI",
    "ZC",
    "ZL",
    "ZM",
    "ZS",
    "ZW",
]
PANEL_D_SYMBOLS: list[str] = [p for p in PANEL_D_PRODUCTS_ALL if p != "ES"]

# yfinance-equivalent symbol for a databento product, where one exists (all
# but KE -- HRW wheat has no yfinance continuous series in this repo).
PANEL_D_TO_YFINANCE: dict[str, str] = {
    "BZ": "BZ=F",
    "CL": "CL=F",
    "ES": "ES=F",
    "GC": "GC=F",
    "HO": "HO=F",
    "NG": "NG=F",
    "PA": "PA=F",
    "PL": "PL=F",
    "RB": "RB=F",
    "SI": "SI=F",
    "ZC": "ZC=F",
    "ZL": "ZL=F",
    "ZM": "ZM=F",
    "ZS": "ZS=F",
    "ZW": "ZW=F",
}


# --------------------------------------------------------------------------- #
# Panel-D: front-month continuous OHLCV and curve shape from per-contract
# databento data
# --------------------------------------------------------------------------- #
def _expiry_from_ticker(ticker: str, as_of: pd.Timestamp) -> float:
    year, month = int(ticker[-6:-2]), int(ticker[-2:])
    expiry = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return float((expiry - as_of).days)


def load_databento_front_month_ohlcv(product: str) -> pd.DataFrame:
    """Front-month continuous OHLCV, one row per date: for each date, the
    bar of the contract with the nearest (smallest) YYYYMM expiry among
    contracts trading that date -- tickers are literally f"{product}{YYYYMM}"
    so this is a lexicographic-ticker min.

    Disclosed limitation (NEXT_PROMPT.md sec5.1's price-only-target trap,
    applied here rather than re-derived per caller): this series is not
    roll-adjusted, so returns computed across a roll date contain a genuine
    price discontinuity (the front contract changing, not the commodity
    moving). Track B prefers the yfinance continuous series wherever one
    exists (PANEL_D_TO_YFINANCE); this is used as-is only for symbols with
    no yfinance equivalent (KE) and for building Panel-D's curve-shape (F3)
    features, which have no yfinance analogue at all.
    """
    frame = pl.read_parquet(DATABENTO_DIR / f"{product}.parquet")
    frame = frame.filter(pl.col("ticker").str.contains(r"^[A-Z]+\d{6}$"))
    front = (
        frame.sort(["date", "ticker"])
        .group_by("date", maintain_order=False)
        .first()
        .sort("date")
    )
    pdf = front.to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"])
    return pdf.set_index("date")[
        ["open", "high", "low", "close", "volume"]
    ].sort_index()


def load_databento_curve_frame(product: str) -> pd.DataFrame | None:
    """Front-three settlement prices and days-to-expiry per date, schema
    matching regime.loaders.load_curve exactly (close_f1, dte_f1, close_f2,
    dte_f2, close_f3, dte_f3) so it plugs straight into RegimeInputs.curve
    and the existing term_structure/carry dimensions run on Panel-D too
    (NEXT_PROMPT.md sec5.3's F3). None if fewer than 2 contracts ever trade
    simultaneously (no curve shape to speak of)."""
    frame = pl.read_parquet(DATABENTO_DIR / f"{product}.parquet").select(
        "date", "ticker", "close"
    )
    # A handful of rows carry a continuous-contract ticker like "PA=F"
    # rather than the per-contract f"{product}{YYYYMM}" form -- exclude
    # those before ranking legs by ticker, or an illiquid date with fewer
    # than 3 real contracts can pull one in as a "leg" and crash expiry
    # parsing (or worse, silently misrank the curve).
    frame = frame.filter(pl.col("ticker").str.contains(r"^[A-Z]+\d{6}$"))
    pdf = frame.to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"])
    rows = []
    for date_key, grp in pdf.groupby("date"):
        date = cast(pd.Timestamp, date_key)
        grp = grp.sort_values("ticker")
        if len(grp) < 2:
            continue
        legs = grp.iloc[:3]
        row: dict[str, object] = {"date": date}
        for i in range(3):
            if i < len(legs):
                leg = legs.iloc[i]
                row[f"close_f{i + 1}"] = leg["close"]
                row[f"dte_f{i + 1}"] = _expiry_from_ticker(str(leg["ticker"]), date)
            else:
                row[f"close_f{i + 1}"] = np.nan
                row[f"dte_f{i + 1}"] = np.nan
        rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows).set_index("date").sort_index()


# --------------------------------------------------------------------------- #
# Track A: shared-input disjointness (NEXT_PROMPT.md sec4.1) -- INPUTS(x) as
# raw-column sets, derived by reading regime/dimensions/{macro,term_structure,
# carry}.py and regime/configs/{macro_default,commodity_default}.yaml
# directly (see those files' docstrings for the exact lines each entry below
# cites).
# --------------------------------------------------------------------------- #
INDICATOR_INPUTS: dict[str, set[str]] = {
    # regime/dimensions/macro.py: _frame_column(inputs, "macro", column)
    "macro.yield_curve": {"FRED:T10Y2Y"},
    "macro.yield_curve_3m10y": {"FRED:T10Y3M"},
    # regime/dimensions/term_structure.py
    "ts.curve_slope": {
        "curve:close_f1",
        "curve:close_f3",
    },  # curve_slope(far="close_f3")
    "ts.calendar_spread_z": {"curve:close_f1", "curve:close_f2"},
    "ts.ann_roll_yield": {
        "curve:close_f1",
        "curve:close_f2",
        "curve:dte_f1",
        "curve:dte_f2",
    },
    "ts.excess_spread": {"curve:close_f1", "curve:close_f2"},
    # regime/dimensions/carry.py
    "carry.ann_roll_yield": {
        "curve:close_f1",
        "curve:close_f2",
        "curve:dte_f1",
        "curve:dte_f2",
    },
    "carry.vol_scaled": {  # vol_scaled_carry(ann_roll_yield(...), realized_vol(ohlcv.close))
        "curve:close_f1",
        "curve:close_f2",
        "curve:dte_f1",
        "curve:dte_f2",
        "bars:close",
    },
}

# Dimension -> {indicator: weight} from configs/{macro_default,commodity_default}.yaml
DIMENSION_INDICATOR_WEIGHTS: dict[str, dict[str, float]] = {
    "yield_curve": {"macro.yield_curve": 0.50, "macro.yield_curve_3m10y": 0.50},
    "term_structure": {
        "ts.curve_slope": 0.30,
        "ts.calendar_spread_z": 0.20,
        "ts.ann_roll_yield": 0.30,
        "ts.excess_spread": 0.20,
    },
    "carry": {"carry.ann_roll_yield": 0.60, "carry.vol_scaled": 0.40},
    "carry_roll_yield_only": {
        "carry.ann_roll_yield": 1.0
    },  # sec4.2's measurement-only variant
}


def dimension_inputs(dimension: str) -> set[str]:
    """Union of INPUTS(indicator) across a dimension's indicators."""
    out: set[str] = set()
    for indicator in DIMENSION_INDICATOR_WEIGHTS[dimension]:
        out |= INDICATOR_INPUTS[indicator]
    return out


# Track A targets: (name, INPUTS(target), claim). NEXT_PROMPT.md sec4.2.
TARGET_INPUTS: dict[str, set[str]] = {
    "A1_dff_fwd126": {"FRED:DFF"},
    "A2_es_drawdown_fwd126": {"bars:ES=F"},
    "A3_hy_oas_fwd63": {"FRED:BAMLH0A0HYM2"},
    "A4_front_month_return_21_63": {"bars:close"},
    "A5_cross_sectional_carry_spread": {"bars:close"},
    "A6_cot_positioning_fwd21": {"cot:noncomm_net_pct_oi"},
}
TARGET_DIMENSION: dict[str, str] = {
    "A1_dff_fwd126": "yield_curve",
    "A2_es_drawdown_fwd126": "yield_curve",
    "A3_hy_oas_fwd63": "yield_curve",
    "A4_front_month_return_21_63": "term_structure",  # and carry (partial overlap, disclosed)
    "A5_cross_sectional_carry_spread": "term_structure",
    "A6_cot_positioning_fwd21": "term_structure",
}


def build_disjointness_table() -> dict[str, Any]:
    """The Track A ID-gate deliverable: dimension -> indicator -> INPUTS,
    target -> INPUTS, and the intersection for every scored pair."""
    dims = ["yield_curve", "term_structure", "carry", "carry_roll_yield_only"]
    table: dict[str, Any] = {"dimensions": {}, "targets": {}, "pairs": []}
    for dim in dims:
        table["dimensions"][dim] = {
            "indicators": {
                name: sorted(INDICATOR_INPUTS[name])
                for name in DIMENSION_INDICATOR_WEIGHTS[dim]
            },
            "inputs_union": sorted(dimension_inputs(dim)),
        }
    for target, inputs in TARGET_INPUTS.items():
        table["targets"][target] = sorted(inputs)

    pairs = [
        ("yield_curve", "A1_dff_fwd126"),
        ("yield_curve", "A2_es_drawdown_fwd126"),
        ("yield_curve", "A3_hy_oas_fwd63"),
        ("term_structure", "A4_front_month_return_21_63"),
        ("term_structure", "A5_cross_sectional_carry_spread"),
        ("term_structure", "A6_cot_positioning_fwd21"),
        ("carry", "A4_front_month_return_21_63"),
        ("carry", "A5_cross_sectional_carry_spread"),
        ("carry_roll_yield_only", "A4_front_month_return_21_63"),
        ("carry_roll_yield_only", "A5_cross_sectional_carry_spread"),
    ]
    for dim, target in pairs:
        dim_inputs = dimension_inputs(dim)
        tgt_inputs = TARGET_INPUTS[target]
        intersection = sorted(dim_inputs & tgt_inputs)
        pair_carry_weight = None
        if dim == "carry" and intersection:
            # carry.vol_scaled (weight 0.40) is the only carry indicator
            # touching bars:close -- quantify the overlap's weight rather
            # than just flagging it (sec4.1's "disclose ... and its weight").
            pair_carry_weight = sum(
                w
                for name, w in DIMENSION_INDICATOR_WEIGHTS[dim].items()
                if intersection and (INDICATOR_INPUTS[name] & tgt_inputs)
            )
        pairs_entry = {
            "dimension": dim,
            "target": target,
            "dimension_inputs": sorted(dim_inputs),
            "target_inputs": sorted(tgt_inputs),
            "intersection": intersection,
            "disqualified": bool(intersection),
            "partial_overlap_weight": pair_carry_weight,
        }
        table["pairs"].append(pairs_entry)
    return table


# --------------------------------------------------------------------------- #
# Track C: effective sample size and minimum detectable effect
# (NEXT_PROMPT.md sec6.3)
# --------------------------------------------------------------------------- #
def bars_per_symbol_panel_l() -> dict[str, int]:
    out = {}
    for symbol in PANEL_L_SYMBOLS:
        close = truncate(load_bars(symbol)["close"])
        out[symbol] = len(close)
    return out


def bars_per_symbol_panel_d() -> dict[str, int]:
    out = {}
    for product in PANEL_D_SYMBOLS:
        close = truncate(load_databento_front_month_ohlcv(product)["close"])
        out[product] = len(close)
    return out


def non_overlapping_forward_returns(close: pd.Series, horizon: int) -> pd.Series:
    """log(close[t+h]/close[t]) sampled on a non-overlapping h-day grid."""
    with np.errstate(invalid="ignore", divide="ignore"):
        fwd = cast(pd.Series, np.log(close.shift(-horizon) / close))
    return fwd.iloc[::horizon].dropna()


def mean_pairwise_correlation(
    symbol_closes: dict[str, pd.Series], horizon: int
) -> float:
    """rho-bar: mean pairwise correlation of non-overlapping forward h-day
    returns across the panel's symbols (sec6.3 bullet 3).

    Symbols have different start dates, so their own non-overlapping grids
    don't share calendar dates -- correlating them directly on an inner
    join of two independently-offset grids returns an empty frame. Instead,
    every symbol's close is first reindexed onto one shared business-day
    calendar (ffilled, matching how `align.py` treats lower-frequency data
    elsewhere in this repo), then the *same* non-overlapping positions are
    sampled from that shared calendar for every symbol, so a "window" means
    the same span of calendar time for all of them.
    """
    if len(symbol_closes) < 2:
        return float("nan")
    calendar = pd.date_range(
        min(c.index.min() for c in symbol_closes.values()),
        max(c.index.max() for c in symbol_closes.values()),
        freq="B",
    )
    aligned = {
        symbol: close.reindex(calendar).ffill()
        for symbol, close in symbol_closes.items()
    }
    frame = pd.DataFrame(aligned)
    with np.errstate(invalid="ignore", divide="ignore"):
        fwd = cast(pd.DataFrame, np.log(frame.shift(-horizon) / frame))
    fwd = fwd.iloc[::horizon]
    corr = fwd.corr(min_periods=10).to_numpy()
    n = corr.shape[0]
    off_diag = corr[~np.eye(n, dtype=bool)]
    if np.all(np.isnan(off_diag)):
        return float("nan")
    return float(np.nanmean(off_diag))


def kish_effective_n(
    n_windows_per_symbol: float, n_symbols: int, rho_bar: float
) -> float:
    """N_eff = N*n / (1 + (n-1)*rho_bar) (sec6.3 bullet 4)."""
    if not np.isfinite(rho_bar):
        return float("nan")
    denom = 1 + (n_symbols - 1) * rho_bar
    if denom <= 0:
        return float("inf")
    return float(n_windows_per_symbol * n_symbols / denom)


def minimum_detectable_effect(
    n_eff: float, alpha: float, power: float = 0.8, p: float = 0.5
) -> float:
    """Balanced-accuracy improvement detectable at `power` given `n_eff`
    effective observations and a Bonferroni-corrected `alpha`, treating the
    comparison as a two-proportion test with variance p(1-p) on each side
    (p=0.5 is the conservative, maximum-variance choice, appropriate near a
    balanced-accuracy null of 0.5)."""
    if not np.isfinite(n_eff) or n_eff <= 0:
        return float("nan")
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    return float((z_alpha + z_power) * np.sqrt(2 * p * (1 - p) / n_eff))


def track_c_power_budget(
    horizons: tuple[int, ...] = (5, 21, 63), alpha: float = 0.05 / 34
) -> dict:
    panel_l_bars = bars_per_symbol_panel_l()
    panel_d_bars = bars_per_symbol_panel_d()
    panel_l_closes = {
        s: cast(pd.Series, truncate(load_bars(s)["close"])) for s in PANEL_L_SYMBOLS
    }
    panel_d_closes = {
        p: cast(pd.Series, truncate(load_databento_front_month_ohlcv(p)["close"]))
        for p in PANEL_D_SYMBOLS
    }

    out = {}
    for panel_name, bars, closes in (
        ("Panel-L", panel_l_bars, panel_l_closes),
        ("Panel-D", panel_d_bars, panel_d_closes),
    ):
        for h in horizons:
            n_symbols = len(bars)
            avg_bars = float(np.mean(list(bars.values())))
            non_overlap_per_symbol = avg_bars / h
            non_overlap_total = non_overlap_per_symbol * n_symbols
            rho_bar = mean_pairwise_correlation(closes, h)
            n_eff = kish_effective_n(non_overlap_per_symbol, n_symbols, rho_bar)
            mde = minimum_detectable_effect(n_eff, alpha)
            out[f"{panel_name}_h{h}"] = {
                "panel": panel_name,
                "horizon": h,
                "n_symbols": n_symbols,
                "raw_rows_pooled": int(sum(bars.values())),
                "non_overlapping_windows_per_symbol": round(non_overlap_per_symbol, 1),
                "non_overlapping_windows_total": round(non_overlap_total, 1),
                "mean_pairwise_correlation": round(rho_bar, 4)
                if np.isfinite(rho_bar)
                else None,
                "n_eff": round(n_eff, 1) if np.isfinite(n_eff) else None,
                "underpowered": (not np.isfinite(n_eff)) or n_eff < 200,
                "minimum_detectable_effect_balanced_accuracy": round(mde, 4)
                if np.isfinite(mde)
                else None,
            }
    return out


__all__ = [
    "BASKET_SYMBOLS",
    "PANEL_D_PRODUCTS_ALL",
    "PANEL_D_SYMBOLS",
    "PANEL_D_TO_YFINANCE",
    "PANEL_L_SYMBOLS",
    "SYMBOL_TO_BASKET",
    "TRUNCATION",
    "assert_truncated",
    "bars_per_symbol_panel_d",
    "bars_per_symbol_panel_l",
    "build_disjointness_table",
    "kish_effective_n",
    "load_databento_curve_frame",
    "load_databento_front_month_ohlcv",
    "mean_pairwise_correlation",
    "minimum_detectable_effect",
    "track_c_power_budget",
    "truncate",
]
