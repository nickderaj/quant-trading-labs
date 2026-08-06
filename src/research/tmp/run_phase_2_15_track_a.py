"""Notebook 015 Phase 2: Track A independent-validation scoring
(NEXT_PROMPT.md sec4). Scores yield_curve, term_structure, carry, and the
roll-yield-only carry variant against six mechanical, forward-realized
targets (A1-A6) that are provably disjoint from each dimension's own raw
inputs (gate ID, Phase 0). Does not re-score the engine against 014's own
targets and does not rebuild regime_panel.parquet -- yield_curve reads its
label straight from the panel (Macro sector); term_structure/carry are
computed fresh per curve-symbol via the already-ported, tested engine
(regime.engine, regime.prediction), which is what sec4.1's "resolve through
the registry" step requires and is a different thing from re-running 014's
own accuracy trials.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "src/research/tmp")

import lib15 as lib
import numpy as np
import pandas as pd
import polars as pl

import research
from regime.engine import RegimeEngine, RegimeInputs, RegimeResult
from regime.forecast_eval import balanced_accuracy
from regime.loaders import (
    load_bars,
    load_cot_raw,
    load_curve,
    load_fred_frame,
    net_positioning,
)
from regime.prediction import (
    ForecastConfig,
    forecast,
    markov_forecast,
    prior_forecast,
    scaled_indicator_frame,
)

TMP = "src/research/tmp"
PANEL_PATH = "src/research/data/market/research/regime_panel.parquet"
CURVE_SYMBOLS = ["CL=F", "NG=F", "GC=F", "SI=F", "HG=F"]
MIN_HISTORY = 252
N_BOOT = 2000


# --------------------------------------------------------------------------- #
# Shared scoring machinery (matches 014's run_phase_3_14.py conventions:
# persistence/markov/class-prior baselines from regime.prediction, block
# bootstrap on the hit-rate paired difference, quarterly blocks)
# --------------------------------------------------------------------------- #
def _states_for(labels: pd.Series) -> list[str]:
    return sorted(labels.dropna().unique().tolist())


def _rowwise_idxmax(probs: pd.DataFrame) -> pd.Series:
    has_value = probs.notna().any(axis=1)
    out = pd.Series(pd.NA, index=probs.index, dtype="object")
    if has_value.any():
        out.loc[has_value] = probs.loc[has_value].idxmax(axis=1)
    return out


def _baselines(labels: pd.Series) -> dict[str, pd.Series]:
    states = _states_for(labels)
    persistence = labels.shift(1)
    prior_probs = prior_forecast(labels, states, min_history=MIN_HISTORY)
    prior_pred = _rowwise_idxmax(prior_probs)
    markov_probs = markov_forecast(
        labels, horizon=1, states=states, min_history=MIN_HISTORY
    )
    markov_pred = _rowwise_idxmax(markov_probs)
    return {
        "persistence": persistence.astype("string"),
        "class_prior": prior_pred.astype("string"),
        "markov": markov_pred.astype("string"),
    }


def _score_against_target(
    pred_engine: pd.Series, baselines: dict[str, pd.Series], target: pd.Series
) -> dict:
    idx = target.dropna().index
    t = target.reindex(idx).astype("string")
    p = pred_engine.reindex(idx).astype("string")
    valid = t.notna() & p.notna()
    idx, t, p = idx[valid], t[valid], p[valid]
    if len(t) < 5:
        return {"n_obs": len(t), "insufficient_data": True}

    engine_hit = (p.to_numpy() == t.to_numpy()).astype(float)
    out: dict = {
        "n_obs": len(t),
        "engine": {
            "balanced_accuracy": balanced_accuracy(p, t),
            "hit_rate": float(engine_hit.mean()),
        },
        "baselines": {},
    }
    best_name, best_hit = None, -np.inf
    for name, series in baselines.items():
        b = series.reindex(idx).astype("string")
        b_valid = b.notna()
        if b_valid.sum() < 5:
            out["baselines"][name] = {
                "n_obs": int(b_valid.sum()),
                "insufficient_data": True,
            }
            continue
        b_hit = (b[b_valid].to_numpy() == t[b_valid].to_numpy()).astype(float)
        differ_frac = float(
            (b[b_valid].to_numpy() != p.reindex(idx)[b_valid].to_numpy()).mean()
        )
        out["baselines"][name] = {
            "n_obs": int(b_valid.sum()),
            "balanced_accuracy": balanced_accuracy(b, t),
            "hit_rate": float(b_hit.mean()),
            "fraction_days_differs_from_engine": differ_frac,
            "structurally_uninformative": differ_frac < 0.05,
        }
        if float(b_hit.mean()) > best_hit:
            best_hit, best_name = float(b_hit.mean()), name

    if best_name is not None:
        b = baselines[best_name].reindex(idx).astype("string")
        b_valid = b.notna()
        diff = engine_hit[b_valid.to_numpy()] - (
            b[b_valid].to_numpy() == t[b_valid].to_numpy()
        ).astype(float)
        if len(diff) >= 5:
            lo, hi = research.block_bootstrap_ci(diff, n_boot=N_BOOT, seed=0)
            pvalue = research.block_bootstrap_pvalue(
                diff, null_value=0.0, n_boot=N_BOOT, seed=0
            )
            out["vs_best_baseline"] = {
                "baseline": best_name,
                "mean_hit_rate_diff": float(diff.mean()),
                "ci95": [lo, hi],
                "pvalue": pvalue,
                "structurally_uninformative": out["baselines"][best_name][
                    "structurally_uninformative"
                ],
            }
    return out


def _spread_significance(spread_returns: pd.Series) -> dict:
    vals = spread_returns.dropna().to_numpy()
    if len(vals) < 5:
        return {"n_obs": len(vals), "insufficient_data": True}
    lo, hi = research.block_bootstrap_ci(vals, n_boot=N_BOOT, seed=0)
    pvalue = research.block_bootstrap_pvalue(
        vals, null_value=0.0, n_boot=N_BOOT, seed=0
    )
    return {
        "n_obs": len(vals),
        "mean": float(vals.mean()),
        "ci95": [lo, hi],
        "pvalue": pvalue,
    }


# --------------------------------------------------------------------------- #
# yield_curve targets A1-A3 (Macro sector, label from regime_panel.parquet)
# --------------------------------------------------------------------------- #
def _macro_yield_curve_labels() -> pd.Series:
    panel = pl.read_parquet(PANEL_PATH)
    frame = (
        panel.filter(
            (pl.col("sector") == "Macro") & (pl.col("dimension") == "yield_curve")
        )
        .sort("date")
        .select("date", "label")
        .to_pandas()
    )
    frame["date"] = pd.to_datetime(frame["date"])
    s = frame.set_index("date")["label"].astype("string").rename(None)
    return lib.truncate(s)


def _expanding_median_binarize(values: pd.Series, min_periods: int = 60) -> pd.Series:
    """Binarize `values` against an expanding median computed only from
    values whose own observation date is strictly before the current row
    (sec4.2's A2 lookahead trap: an in-sample median leaks the future)."""
    resolved = values.dropna()
    threshold = resolved.expanding(min_periods=min_periods).median().shift(1)
    threshold = threshold.reindex(values.index).ffill()
    return threshold


def score_yield_curve(prereg_alpha: float) -> dict:
    labels = _macro_yield_curve_labels()
    directional = labels.isin(["steep", "inverted"])
    baselines = _baselines(labels)
    out = {}

    # A1: sign of forward 126d change in DFF -- policy easing/tightening
    dff = lib.truncate(load_fred_frame(("DFF",))["DFF"]).reindex(labels.index).ffill()
    fwd_dff = dff.shift(-126) - dff
    target = pd.Series(
        np.where(fwd_dff > 0, "tighten", "ease"), index=labels.index, dtype="string"
    )
    target = target.where(fwd_dff.notna() & (fwd_dff != 0))
    pred = (
        labels.map({"steep": "tighten", "inverted": "ease"})
        .astype("string")
        .where(directional)
    )
    target = target.where(pred.notna())
    bin_baselines = {
        n: s.map({"steep": "tighten", "inverted": "ease"}).astype("string")
        for n, s in baselines.items()
    }
    out["A1_dff_fwd126"] = _score_against_target(pred, bin_baselines, target)

    # A2: forward 126d max drawdown of ES=F, binarized at its own expanding
    # (strictly-past) median -- an inverted curve forecasts equity stress.
    es_close = lib.truncate(load_bars("ES=F")["close"])
    h = 126
    dd = pd.Series(np.nan, index=es_close.index)
    arr = es_close.to_numpy()
    n = len(arr)
    for i in range(n - h):
        window = arr[i : i + h + 1]
        running_max = np.maximum.accumulate(window)
        dd.iloc[i] = float(np.min(window / running_max - 1.0))
    dd = dd.reindex(labels.index)
    threshold = _expanding_median_binarize(dd)
    target = pd.Series(
        np.where(dd <= threshold, "high_stress", "low_stress"),
        index=labels.index,
        dtype="string",
    )
    target = target.where(dd.notna() & threshold.notna())
    pred = (
        labels.map({"inverted": "high_stress", "steep": "low_stress"})
        .astype("string")
        .where(directional)
    )
    target = target.where(pred.notna())
    bin_baselines = {
        n: s.map({"inverted": "high_stress", "steep": "low_stress"}).astype("string")
        for n, s in baselines.items()
    }
    out["A2_es_drawdown_fwd126"] = _score_against_target(pred, bin_baselines, target)

    # A3: sign of forward 63d change in BAMLH0A0HYM2 -- reported, flagged
    # underpowered (2023-07-17 start; ~18mo after truncation).
    hy = (
        lib.truncate(load_fred_frame(("BAMLH0A0HYM2",))["BAMLH0A0HYM2"])
        .reindex(labels.index)
        .ffill()
    )
    fwd_hy = hy.shift(-63) - hy
    target = pd.Series(
        np.where(fwd_hy > 0, "widen", "tighten"), index=labels.index, dtype="string"
    )
    target = target.where(fwd_hy.notna() & (fwd_hy != 0))
    pred = (
        labels.map({"inverted": "widen", "steep": "tighten"})
        .astype("string")
        .where(directional)
    )
    target = target.where(pred.notna())
    bin_baselines = {
        n: s.map({"inverted": "widen", "steep": "tighten"}).astype("string")
        for n, s in baselines.items()
    }
    result = _score_against_target(pred, bin_baselines, target)
    result["underpowered"] = True
    result["underpowered_reason"] = (
        "BAMLH0A0HYM2 starts 2023-07-17; ~18mo of data before truncation"
    )
    out["A3_hy_oas_fwd63"] = result
    return out


# --------------------------------------------------------------------------- #
# term_structure / carry: per curve-symbol engine results (fresh compute,
# not a re-score of 014's own trials -- sec4.1's registry-resolution step)
# --------------------------------------------------------------------------- #
def _symbol_result(symbol: str) -> tuple[pd.Series, RegimeResult]:
    bars = lib.truncate(load_bars(symbol))
    curve = load_curve(symbol)
    assert curve is not None
    curve = lib.truncate(curve)
    inputs = RegimeInputs(ohlcv=bars, curve=curve)
    result = RegimeEngine.from_default("commodity_default").detect(inputs)
    return bars["close"], result


def _price_only_target(close: pd.Series, horizon: int) -> pd.Series:
    with np.errstate(invalid="ignore", divide="ignore"):
        fwd = pd.Series(np.log(close.shift(-horizon) / close), index=close.index)
    return pd.Series(
        np.where(fwd > 0, "up", "down"), index=close.index, dtype="string"
    ).where(fwd.notna() & (fwd != 0))


def score_term_structure_and_carry() -> dict:
    out: dict = {}
    per_symbol_results = {}
    for symbol in CURVE_SYMBOLS:
        close, result = _symbol_result(symbol)
        per_symbol_results[symbol] = (close, result)

    # A4: term_structure/carry labels (each curve symbol's own, not a
    # basket aggregate) vs price-only forward 21d/63d return sign.
    for dimension in ("term_structure", "carry"):
        for horizon in (21, 63):
            per_symbol_scored = {}
            for symbol in CURVE_SYMBOLS:
                close, result = per_symbol_results[symbol]
                labels = result.labels[dimension].astype("string")
                directional = labels.isin(
                    ["backwardation", "contango"]
                    if dimension == "term_structure"
                    else ["positive", "negative"]
                )
                pred_map = (
                    {"backwardation": "up", "contango": "down"}
                    if dimension == "term_structure"
                    else {"positive": "up", "negative": "down"}
                )
                pred = labels.map(pred_map).astype("string").where(directional)
                target = _price_only_target(close, horizon)
                target = target.where(pred.reindex(target.index).notna())
                baselines = _baselines(labels)
                bin_baselines = {
                    n: s.map(pred_map).astype("string") for n, s in baselines.items()
                }
                per_symbol_scored[symbol] = _score_against_target(
                    pred, bin_baselines, target
                )
            out[f"A4_{dimension}_price_only_h{horizon}"] = per_symbol_scored

    # carry_roll_yield_only: same A4 targets, weight=1.0 on ann_roll_yield
    # only (sec4.2's measurement-only variant -- reported alongside, not
    # instead of, the shipped config).
    for horizon in (21, 63):
        per_symbol_scored = {}
        for symbol in CURVE_SYMBOLS:
            close, result = per_symbol_results[symbol]
            dim_config = next(d for d in result.config.dimensions if d.key == "carry")
            scaled = scaled_indicator_frame(result, "carry")
            cfg = ForecastConfig(
                dimension="carry",
                horizon=1,
                weights={"carry.ann_roll_yield": 1.0},
                use_hysteresis=True,
                smoothing_span=dim_config.smoothing_span,
            )
            variant = forecast(scaled, dim_config, cfg)
            labels = variant.labels.astype("string")
            directional = labels.isin(["positive", "negative"])
            pred_map = {"positive": "up", "negative": "down"}
            pred = labels.map(pred_map).astype("string").where(directional)
            target = _price_only_target(close, horizon)
            target = target.where(pred.reindex(target.index).notna())
            baselines = _baselines(labels)
            bin_baselines = {
                n: s.map(pred_map).astype("string") for n, s in baselines.items()
            }
            per_symbol_scored[symbol] = _score_against_target(
                pred, bin_baselines, target
            )
        out[f"A4_carry_roll_yield_only_price_only_h{horizon}"] = per_symbol_scored

    # A5: cross-sectional rank spread + rank IC (all 5 curve symbols'
    # term_structure and carry scores), 21d and 63d.
    for dimension in ("term_structure", "carry"):
        scores_by_symbol = {
            sym: per_symbol_results[sym][1].scores[dimension] for sym in CURVE_SYMBOLS
        }
        closes_by_symbol = {sym: per_symbol_results[sym][0] for sym in CURVE_SYMBOLS}
        score_frame = pd.DataFrame(scores_by_symbol)
        for horizon in (21, 63):
            fwd_frame = pd.DataFrame(
                {
                    sym: np.log(
                        closes_by_symbol[sym].shift(-horizon) / closes_by_symbol[sym]
                    )
                    for sym in CURVE_SYMBOLS
                }
            )
            aligned_scores = score_frame.reindex(fwd_frame.index)
            # Non-overlapping sampling first (sec4.2/A5 doesn't need every
            # overlapping day), THEN loop -- row-wise .loc access per date
            # over the full daily history is the slow path here (6000+
            # rows x 4 combinations); subsampling by `horizon` up front
            # cuts that by the same factor before any per-row work happens.
            sampled_scores = aligned_scores.iloc[::horizon]
            sampled_fwd = fwd_frame.iloc[::horizon]
            spread = []
            for date, score_row in sampled_scores.iterrows():
                row_scores = score_row.dropna()
                if len(row_scores) < 4:
                    spread.append(np.nan)
                    continue
                row_fwd = sampled_fwd.loc[date][row_scores.index].dropna()  # type: ignore[call-overload]
                common = row_scores.index.intersection(row_fwd.index)
                if len(common) < 4:
                    spread.append(np.nan)
                    continue
                ranked = row_scores[common].sort_values()
                bottom2, top2 = ranked.index[:2], ranked.index[-2:]
                spread.append(float(row_fwd[top2].mean() - row_fwd[bottom2].mean()))
            spread_series = pd.Series(spread, index=sampled_scores.index)

            stacked = pd.concat(
                [
                    pl.DataFrame(
                        {
                            "datetime": aligned_scores.index,
                            "score": aligned_scores[sym].to_numpy(),
                            "fwd": fwd_frame[sym].to_numpy(),
                        }
                    ).to_pandas()
                    for sym in CURVE_SYMBOLS
                ],
                ignore_index=True,
            ).dropna()
            stacked_pl = pl.from_pandas(stacked)
            ic_stats = research.panel_ic(
                stacked_pl, "score", "fwd", nw_lag=horizon, datetime_col="datetime"
            )
            out[f"A5_{dimension}_cross_sectional_h{horizon}"] = {
                "spread_top2_minus_bottom2": _spread_significance(spread_series),
                "rank_ic": ic_stats,
            }

    # A6: sign of forward 21d change in noncomm_net_pct_oi (crude only) vs
    # oil products basket term_structure label (CL=F is the only curve
    # symbol in that basket, so the basket label is already a CL proxy --
    # same "representative sector price" convention 014 used).
    panel = pl.read_parquet(PANEL_PATH)
    frame = (
        panel.filter(
            (pl.col("sector") == "oil products")
            & (pl.col("dimension") == "term_structure")
        )
        .sort("date")
        .select("date", "label")
        .to_pandas()
    )
    frame["date"] = pd.to_datetime(frame["date"])
    ts_labels = lib.truncate(
        frame.set_index("date")["label"].astype("string").rename(None)
    )
    cot = net_positioning(load_cot_raw())
    cot_series = (
        lib.truncate(cot["noncomm_net_pct_oi"]).reindex(ts_labels.index).ffill()
    )
    fwd_cot = cot_series.shift(-21) - cot_series
    target = pd.Series(
        np.where(fwd_cot > 0, "increase", "decrease"),
        index=ts_labels.index,
        dtype="string",
    )
    target = target.where(fwd_cot.notna() & (fwd_cot != 0))
    directional = ts_labels.isin(["backwardation", "contango"])
    pred_map = {"backwardation": "increase", "contango": "decrease"}
    pred = ts_labels.map(pred_map).astype("string").where(directional)
    target = target.where(pred.notna())
    baselines = _baselines(ts_labels)
    bin_baselines = {n: s.map(pred_map).astype("string") for n, s in baselines.items()}
    out["A6_cot_positioning_fwd21"] = _score_against_target(pred, bin_baselines, target)

    return out


def main() -> None:
    with open(f"{TMP}/phase_0_15_preregistration.json") as f:
        prereg = json.load(f)
    alpha = prereg["significance_procedure"]["alpha_bonferroni"]

    print("Scoring yield_curve (A1, A2, A3)...")
    yc = score_yield_curve(alpha)

    print("Scoring term_structure / carry / carry_roll_yield_only (A4, A5, A6)...")
    ts_carry = score_term_structure_and_carry()

    results = {
        "alpha_bonferroni": alpha,
        "yield_curve": yc,
        "term_structure_and_carry": ts_carry,
    }
    with open(f"{TMP}/phase_2_15_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    for name, res in yc.items():
        vs = res.get("vs_best_baseline")
        print(
            f"  {name}: n={res.get('n_obs')} engine_ba={res.get('engine', {}).get('balanced_accuracy')} "
            f"vs_best={vs}"
        )
    print("Wrote phase_2_15_results.json")


if __name__ == "__main__":
    main()
