"""Notebook 015 Track B: the feature ladder (F0-F3), the model ladder
(M0a-M3), and the purged/embargoed pooled walk-forward pipeline that scores
them (NEXT_PROMPT.md sec5). Shared by the shuffle control (Phase 1) and the
real Track B run (Phase 3) so the *entire* pipeline -- folds, purge,
embargo, features, models, pooling -- is identical between them, which is
the whole point of running the shuffle control first (sec5.6).
"""

from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass
from typing import cast

sys.path.insert(0, "src")
sys.path.insert(0, "src/research/tmp")

import lib15 as lib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from regime.engine import RegimeEngine, RegimeInputs
from regime.forecast_eval import purged_embargoed_walk_forward_splits
from regime.prediction import scaled_indicator_frame

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.*")
logging.getLogger("regime.engine").setLevel(logging.ERROR)

PANEL_PATH = "src/research/data/market/research/regime_panel.parquet"
F1_INDICATORS = [
    "trend.price_vs_ma", "trend.ma_slope", "trend.nday_log_return", "trend.adx", "trend.efficiency_ratio",
]
VOL_INDICATORS = ["vol.atr_percentile", "vol.realized_vol_percentile", "vol.vol_of_vol"]
MR_INDICATORS = ["mr.autocorr", "mr.variance_ratio", "mr.half_life"]
TS_INDICATORS = ["ts.curve_slope", "ts.calendar_spread_z", "ts.ann_roll_yield", "ts.excess_spread"]
CARRY_INDICATORS = ["carry.ann_roll_yield", "carry.vol_scaled"]

M3_PARAMS: dict[str, object] = {
    "max_iter": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_samples_leaf": 200,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "random_state": 0,
}


# --------------------------------------------------------------------------- #
# Per-symbol engine results and feature frames
# --------------------------------------------------------------------------- #
@dataclass
class SymbolData:
    symbol: str
    basket: str | None
    close: pd.Series  # target-construction close (yfinance where available)
    result: RegimeInputs
    features: pd.DataFrame  # F1 + volatility + mean_reversion + [term_structure + carry]


def _basket_for(symbol_or_product: str, panel: str) -> str | None:
    if panel == "Panel-L":
        return lib.SYMBOL_TO_BASKET.get(symbol_or_product)
    yfin = lib.PANEL_D_TO_YFINANCE.get(symbol_or_product)
    return lib.SYMBOL_TO_BASKET.get(yfin) if yfin else None


def _engine_features(ohlcv: pd.DataFrame, curve: pd.DataFrame | None) -> pd.DataFrame:
    """scaled_indicator_frame's columns are already the indicator's full
    registered name (e.g. "trend.price_vs_ma", "ts.curve_slope") -- see its
    docstring -- so no re-prefixing is needed or correct here."""
    inputs = RegimeInputs(ohlcv=ohlcv, curve=curve)
    result = RegimeEngine.from_default("commodity_default").detect(inputs)
    frames = [
        scaled_indicator_frame(result, dim)
        for dim in ("trend", "volatility", "mean_reversion", "term_structure", "carry")
    ]
    return pd.concat(frames, axis=1)


_SYMBOL_DATA_CACHE: dict[tuple[str, str], SymbolData] = {}


def build_symbol_data(symbol_or_product: str, panel: str) -> SymbolData:
    key = (symbol_or_product, panel)
    if key in _SYMBOL_DATA_CACHE:
        return _SYMBOL_DATA_CACHE[key]
    data = _build_symbol_data_uncached(symbol_or_product, panel)
    _SYMBOL_DATA_CACHE[key] = data
    return data


def _build_symbol_data_uncached(symbol_or_product: str, panel: str) -> SymbolData:
    if panel == "Panel-L":
        ohlcv = lib.truncate(lib.load_bars(symbol_or_product))
        curve_symbol = symbol_or_product if symbol_or_product in ("CL=F", "NG=F", "GC=F", "SI=F", "HG=F") else None
        curve = None
        if curve_symbol is not None:
            from regime.loaders import load_curve

            raw_curve = load_curve(curve_symbol)
            assert raw_curve is not None  # curve_symbol is pinned to a known CURVE_SYMBOLS key
            curve = lib.truncate(raw_curve)
        close = ohlcv["close"]
    else:  # Panel-D
        yfin = lib.PANEL_D_TO_YFINANCE.get(symbol_or_product)
        ohlcv = lib.truncate(lib.load_databento_front_month_ohlcv(symbol_or_product))
        curve = lib.load_databento_curve_frame(symbol_or_product)
        if curve is not None:
            curve = lib.truncate(curve)
        close = lib.truncate(lib.load_bars(yfin)["close"]) if yfin else ohlcv["close"]

    basket = _basket_for(symbol_or_product, panel)
    features = _engine_features(ohlcv, curve)
    return SymbolData(
        symbol=symbol_or_product, basket=basket, close=close, result=RegimeInputs(ohlcv=ohlcv, curve=curve),
        features=features,
    )


# --------------------------------------------------------------------------- #
# F0: the engine's own trend label, read from regime_panel.parquet
# --------------------------------------------------------------------------- #
def load_f0_labels() -> dict[str, pd.Series]:
    """basket -> Series of trend labels, truncated. Commodities/trend is
    excluded (014 Phase 2: 90.24% single-label, disqualified as an
    incumbent -- NEXT_PROMPT.md sec9)."""
    panel = pl.read_parquet(PANEL_PATH)
    frame = (
        panel.filter((pl.col("dimension") == "trend") & (pl.col("kind") == "basket"))
        .filter(pl.col("sector") != "Commodities")
        .sort("date")
        .select("sector", "date", "label")
        .to_pandas()
    )
    frame["date"] = pd.to_datetime(frame["date"])
    out: dict[str, pd.Series] = {}
    for basket_key, grp in frame.groupby("sector"):
        basket = cast(str, basket_key)
        s = grp.set_index("date")["label"].astype("string").rename(None)
        out[basket] = lib.truncate(s)
    return out


# --------------------------------------------------------------------------- #
# Target: sign(forward h-day log return), zero-return rows dropped
# --------------------------------------------------------------------------- #
def build_target(close: pd.Series, horizon: int) -> pd.Series:
    with np.errstate(invalid="ignore", divide="ignore"):
        fwd = cast(pd.Series, np.log(close.shift(-horizon) / close))
    return fwd.where(fwd.abs() >= 1e-12)  # frozen-bar rule


# --------------------------------------------------------------------------- #
# Panel assembly: pool all symbols into one long frame with cross-sectional
# features computed *within date* (safe) and calendar features.
# --------------------------------------------------------------------------- #
def build_pooled_panel(panel_name: str, horizon: int, feature_set: str) -> tuple[pd.DataFrame, list[str]]:
    """Returns (long_frame, feature_columns). long_frame is indexed by
    (date, symbol) with columns: fwd_return, y (+-1), f0_label, basket, and
    every feature column for `feature_set` ("F1", "F2", or "F3")."""
    symbols = lib.PANEL_L_SYMBOLS if panel_name == "Panel-L" else lib.PANEL_D_SYMBOLS
    f0_by_basket = load_f0_labels()

    rows = []
    for sym in symbols:
        data = build_symbol_data(sym, panel_name)
        fwd = build_target(data.close, horizon)
        frame = data.features.copy()
        frame["fwd_return"] = fwd.reindex(frame.index)
        frame = frame.dropna(subset=["fwd_return"])
        if frame.empty:
            continue
        frame["y"] = np.sign(frame["fwd_return"]).astype(int)
        frame["symbol"] = sym
        frame["basket"] = data.basket
        if data.basket is not None and data.basket in f0_by_basket:
            frame["f0_label"] = f0_by_basket[data.basket].reindex(frame.index)
        else:
            frame["f0_label"] = pd.NA
        # M0b/M0c: causal, available at t.
        with np.errstate(invalid="ignore", divide="ignore"):
            realized_1d = cast(pd.Series, np.sign(data.close.diff(1)))
            trailing_60d = cast(pd.Series, np.sign(np.log(data.close / data.close.shift(60))))
            frame["realized_1d_sign"] = realized_1d.reindex(frame.index)
            frame["trailing_60d_return_sign"] = trailing_60d.reindex(frame.index)
        rows.append(frame)

    panel = pd.concat(rows, axis=0)
    panel.index.name = "date"
    panel = panel.reset_index().set_index(["date", "symbol"]).sort_index()

    feature_cols = list(F1_INDICATORS)
    if feature_set in ("F2", "F3"):
        feature_cols += VOL_INDICATORS + MR_INDICATORS + TS_INDICATORS + CARRY_INDICATORS

        # Cross-sectional ranks (within date -- safe) on each F1 feature,
        # within basket and within the full panel.
        by_date = panel.reset_index()
        for col in F1_INDICATORS:
            by_date[f"rank_panel.{col}"] = by_date.groupby("date")[col].rank(pct=True)
            by_date[f"rank_basket.{col}"] = by_date.groupby(["date", "basket"])[col].rank(pct=True)
            feature_cols += [f"rank_panel.{col}", f"rank_basket.{col}"]

        # Calendar: day-of-week, month, cyclical.
        dow = by_date["date"].dt.dayofweek.to_numpy()
        month = by_date["date"].dt.month.to_numpy()
        by_date["cal.dow_sin"] = np.sin(2 * np.pi * dow / 5)
        by_date["cal.dow_cos"] = np.cos(2 * np.pi * dow / 5)
        by_date["cal.month_sin"] = np.sin(2 * np.pi * month / 12)
        by_date["cal.month_cos"] = np.cos(2 * np.pi * month / 12)
        feature_cols += ["cal.dow_sin", "cal.dow_cos", "cal.month_sin", "cal.month_cos"]
        panel = by_date.set_index(["date", "symbol"]).sort_index()

    if feature_set == "F3" and panel_name == "Panel-D":
        curve_rows = []
        for sym in symbols:
            curve = lib.load_databento_curve_frame(sym)
            if curve is None:
                continue
            curve = lib.truncate(curve).copy()
            curve["f3.spread_12"] = curve["close_f1"] - curve["close_f2"]
            curve["f3.spread_23"] = curve["close_f2"] - curve["close_f3"]
            curve["symbol"] = sym
            curve_rows.append(
                curve[["symbol", "f3.spread_12", "f3.spread_23", "dte_f1", "dte_f2", "dte_f3"]]
                .rename(columns={"dte_f1": "f3.dte_f1", "dte_f2": "f3.dte_f2", "dte_f3": "f3.dte_f3"})
            )
        if curve_rows:
            curve_panel = pd.concat(curve_rows)
            curve_panel.index.name = "date"
            curve_panel = curve_panel.reset_index().set_index(["date", "symbol"])
            panel = panel.join(curve_panel, how="left")
            feature_cols += ["f3.spread_12", "f3.spread_23", "f3.dte_f1", "f3.dte_f2", "f3.dte_f3"]

    return panel, feature_cols


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def _majority_class(y: pd.Series) -> int:
    counts = y.value_counts()
    return int(counts.idxmax()) if len(counts) else 0


def fit_predict_all_models(
    train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]
) -> dict[str, pd.Series]:
    """Fits M0a-M3 on `train`, predicts on `test`; returns {model_id: pred
    Series in {-1,+1}, indexed like test} (M0d also emits NaN where no F0
    label exists, e.g. no basket mapping)."""
    preds: dict[str, pd.Series] = {}
    majority = _majority_class(train["y"])

    preds["M0a"] = pd.Series(majority, index=test.index)
    preds["M0b"] = test["realized_1d_sign"].fillna(0).astype(int)
    preds["M0c"] = test["trailing_60d_return_sign"].fillna(0).astype(int)

    f0_map = {"bull": 1, "bear": -1}
    f0_pred = test["f0_label"].map(f0_map)
    f0_pred = f0_pred.where(test["f0_label"] != "sideways", majority)
    preds["M0d"] = f0_pred.where(test["f0_label"].notna())

    # A feature column with zero non-NaN values in this fold's training
    # window (e.g. term_structure/carry before any curve symbol's history
    # has started) breaks HistGradientBoostingClassifier's internal
    # quantile binning outright, not just LogisticRegressionCV's finite-
    # value check -- drop it for this fold rather than let either model see
    # it.
    usable_cols = [c for c in feature_cols if train[c].notna().any()]
    X_train_raw = train[usable_cols]
    X_test_raw = test[usable_cols]
    median = X_train_raw.median()
    # A column that is entirely NaN within this fold's training window (e.g.
    # term_structure/carry for the 15/20 Panel-L symbols with no curve, in
    # a fold where even the 5 curve symbols haven't started yet) has a NaN
    # median too -- fall back to 0 (post-standardization-neutral) so
    # LogisticRegressionCV, which has no native NaN support, never sees one.
    X_train = X_train_raw.fillna(median).fillna(0.0)
    X_test = X_test_raw.fillna(median).fillna(0.0)
    y_train = train["y"].to_numpy()

    if "M1" in _MODELS_TO_FIT:
        f1_cols = [c for c in usable_cols if c.startswith("trend.")]
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegressionCV(
                Cs=[0.001, 0.01, 0.1, 1.0, 10.0], cv=TimeSeriesSplit(n_splits=3),
                penalty="l2", max_iter=2000, scoring="balanced_accuracy", random_state=0,
            )),
        ])
        pipe.fit(X_train[f1_cols], y_train)
        preds["M1"] = pd.Series(pipe.predict(X_test[f1_cols]), index=test.index)

    if "M2" in _MODELS_TO_FIT:
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegressionCV(
                Cs=[0.001, 0.01, 0.1, 1.0, 10.0], cv=TimeSeriesSplit(n_splits=3),
                penalty="l2", max_iter=2000, scoring="balanced_accuracy", random_state=0,
            )),
        ])
        pipe.fit(X_train, y_train)
        preds["M2"] = pd.Series(pipe.predict(X_test), index=test.index)

    if "M3" in _MODELS_TO_FIT:
        clf = HistGradientBoostingClassifier(**M3_PARAMS)
        clf.fit(X_train_raw, y_train)  # HGB handles NaN natively
        preds["M3"] = pd.Series(clf.predict(X_test_raw), index=test.index)

    return preds


_MODELS_TO_FIT = {"M1", "M2", "M3"}


# --------------------------------------------------------------------------- #
# Purged/embargoed pooled walk-forward
# --------------------------------------------------------------------------- #
def pooled_folds(panel: pd.DataFrame, horizon: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fold boundaries are calendar dates shared across all symbols (sec5.5):
    build folds on the unique date index, then expand to every row sharing
    that date."""
    dates = panel.index.get_level_values("date")
    unique_dates = pd.DatetimeIndex(sorted(dates.unique()))
    date_folds = purged_embargoed_walk_forward_splits(
        unique_dates, min_train=1260, test_size=252, horizon=horizon, step=252
    )
    row_dates = dates.to_numpy()
    folds = []
    for train_dates, test_dates in date_folds:
        train_mask = np.isin(row_dates, train_dates.to_numpy())
        test_mask = np.isin(row_dates, test_dates.to_numpy())
        folds.append((np.flatnonzero(train_mask), np.flatnonzero(test_mask)))
    return folds


def block_shuffle_target(panel: pd.DataFrame, block_size: int, seed: int) -> pd.DataFrame:
    """Block-shuffle `y`/`fwd_return` per symbol, in contiguous blocks of
    `block_size` days, preserving each symbol's own autocorrelation
    structure while destroying feature-target alignment (sec5.6).

    `f0_label` is NOT shuffled here -- it is M0d's *predictor*, exactly
    analogous to every other feature column, not a target. Permuting it in
    lockstep with `y` (an earlier bug in this function) reapplied the same
    permutation to both, which leaves their pairwise relationship exactly
    intact -- the shuffle control would then correctly detect that M0d
    still "predicts" the shuffled target, but for the wrong reason: not a
    leak, but the control failing to shuffle the one thing that needed it
    least disturbed relative to itself.
    """
    rng = np.random.default_rng(seed)
    out = panel.copy()
    for idx in panel.groupby(level="symbol").groups.values():
        sub = panel.loc[idx].sort_index(level="date")
        n = len(sub)
        n_blocks = int(np.ceil(n / block_size))
        block_order = rng.permutation(n_blocks)
        perm = np.concatenate(
            [np.arange(b * block_size, min((b + 1) * block_size, n)) for b in block_order]
        )
        perm = perm[perm < n]
        for col in ("y", "fwd_return"):
            out.loc[sub.index, col] = sub[col].to_numpy()[perm]
    return out


def run_pipeline(
    panel_name: str, horizon: int, feature_set: str, shuffle_seed: int | None = None,
    block_size: int = 63,
) -> dict:
    """Runs the entire Track B pipeline for one (panel, horizon, feature_set):
    build panel -> [optional block-shuffle] -> purged/embargoed pooled
    folds -> fit+predict every model per fold -> pooled (date, pred, true)
    rows per model, from which balanced accuracy (headline metric) and a
    daily mean-hit-rate series (for the block-bootstrap significance test,
    matching 014's own convention: balanced accuracy is reported once over
    all pooled predictions, significance is tested on the raw hit-rate
    paired difference) are both derived."""
    from sklearn.metrics import balanced_accuracy_score

    panel, feature_cols = build_pooled_panel(panel_name, horizon, feature_set)
    if shuffle_seed is not None:
        panel = block_shuffle_target(panel, block_size, shuffle_seed)

    folds = pooled_folds(panel, horizon)
    panel_arr = panel.reset_index()

    model_ids = ["M0a", "M0b", "M0c", "M0d", "M1", "M2", "M3"]
    pred_records: dict[str, list[pd.DataFrame]] = {m: [] for m in model_ids}
    abstention_rates = []
    n_folds_used = 0

    for train_idx, test_idx in folds:
        train = panel_arr.iloc[train_idx]
        test = panel_arr.iloc[test_idx]
        if train.empty or test.empty:
            continue
        n_folds_used += 1
        preds = fit_predict_all_models(train, test, feature_cols)
        abstention_rates.append(float((test["f0_label"] == "sideways").mean()))
        for m in model_ids:
            pred = preds[m].reindex(test.index)
            pred_records[m].append(
                pd.DataFrame({"date": test["date"], "pred": pred, "true": test["y"]})
            )

    daily_hit_rate: dict[str, pd.Series] = {}
    balanced_acc: dict[str, float] = {}
    n_obs: dict[str, int] = {}
    predictions: dict[str, pd.DataFrame] = {}
    for m in model_ids:
        if not pred_records[m]:
            continue
        combined = pd.concat(pred_records[m], ignore_index=True).dropna(subset=["pred", "true"])
        if combined.empty:
            continue
        combined = combined.sort_values("date")
        n_obs[m] = len(combined)
        balanced_acc[m] = float(balanced_accuracy_score(combined["true"], combined["pred"]))
        hit = (combined["pred"] == combined["true"]).astype(float)
        daily_hit_rate[m] = hit.groupby(combined["date"]).mean()
        predictions[m] = combined  # date-sorted (true, pred) pairs, for a paired block bootstrap

    return {
        "panel": panel_name, "horizon": horizon, "feature_set": feature_set,
        "shuffle_seed": shuffle_seed, "n_folds": n_folds_used,
        "n_rows": len(panel_arr), "feature_cols": feature_cols,
        "daily_hit_rate": daily_hit_rate, "balanced_accuracy": balanced_acc, "n_obs": n_obs,
        "predictions": predictions,
        "abstention_rate": float(np.mean(abstention_rates)) if abstention_rates else None,
    }


__all__ = [
    "CARRY_INDICATORS",
    "F1_INDICATORS",
    "M3_PARAMS",
    "MR_INDICATORS",
    "TS_INDICATORS",
    "VOL_INDICATORS",
    "SymbolData",
    "block_shuffle_target",
    "build_pooled_panel",
    "build_symbol_data",
    "build_target",
    "fit_predict_all_models",
    "load_f0_labels",
    "pooled_folds",
    "run_pipeline",
]
