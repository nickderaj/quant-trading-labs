"""Notebook 014 Phase 3: accuracy against independently-known regime periods
(NEXT_PROMPT.md sec6 Phase 3) -- the notebook's reason to exist.

Two ground-truth sources, both frozen in Phase 0 and never touched again:
(a) the hand-labelled episode table (phase_0_14_preregistration.json)
(b) mechanical, forward-realized-quantity labels needing no human judgment

For each scorable (sector, dimension, ground-truth source) triple: balanced
accuracy of the engine's actual causal label against three baselines
already ported in regime.prediction (persistence, one-step Markov, class
prior), a block-bootstrap significance test of engine-vs-best-baseline
(matching this repo's inference-correction discipline, see
src/results/003_cross_sectional_ic.md), and a lead-lag measure for the
episode table.

Both holdouts (crypto 2025-07-01, futures 2025-01-01 -> 2026-07-28) are
untouched: every series here is truncated to dates < SCORING_CUTOFF before
any target or baseline is built.
"""

from __future__ import annotations

import json
import sys
from typing import Any

sys.path.insert(0, "src")
sys.path.insert(0, "src/research/tmp")

import numpy as np
import pandas as pd
import polars as pl

import research
from regime.forecast_eval import (
    balanced_accuracy,
    forward_log_return,
    forward_realized_vol,
)
from regime.loaders import load_bars, load_curve
from regime.prediction import markov_forecast, prior_forecast

TMP = "src/research/tmp"
PANEL_PATH = "src/research/data/market/research/regime_panel.parquet"
SCORING_CUTOFF = pd.Timestamp(
    "2025-01-01"
)  # futures holdout start; crypto holdout is N/A here
MIN_HISTORY = (
    252  # class-prior / Markov baselines need a burn-in, matches prediction.py default
)
N_BOOT = 2000


def _sector_labels(panel: pl.DataFrame, sector: str, dimension: str) -> pd.Series:
    frame = (
        panel.filter((pl.col("sector") == sector) & (pl.col("dimension") == dimension))
        .sort("date")
        .select("date", "label")
        .to_pandas()
    )
    s = frame.set_index("date")["label"].astype("string").rename(None)
    return s[s.index < SCORING_CUTOFF]


def _states_for(labels: pd.Series) -> list[str]:
    return sorted(labels.dropna().unique().tolist())


def _rowwise_idxmax(probs: pd.DataFrame) -> pd.Series:
    """idxmax(axis=1) that tolerates all-NaN rows (pandas' idxmax raises on
    an all-NaN row rather than returning NaN for just that row)."""
    has_value = probs.notna().any(axis=1)
    out = pd.Series(pd.NA, index=probs.index, dtype="object")
    if has_value.any():
        out.loc[has_value] = probs.loc[has_value].idxmax(axis=1)
    return out


def _baselines(labels: pd.Series) -> dict[str, pd.Series]:
    """Causal baseline predictions aligned to `labels`' index, reusing the
    already-ported, tested regime.prediction functions."""
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
    """Balanced accuracy of the engine and each baseline vs `target`
    (already restricted to scorable dates), plus a block-bootstrap
    significance test of engine-vs-best-baseline hit-rate difference."""
    idx = target.dropna().index
    t = target.reindex(idx).astype("string")
    p = pred_engine.reindex(idx).astype("string")
    valid = t.notna() & p.notna()
    idx, t, p = idx[valid], t[valid], p[valid]
    if len(t) < 5:
        return {"n_obs": len(t), "insufficient_data": True}

    engine_hit = (p.to_numpy() == t.to_numpy()).astype(float)
    out: dict[str, Any] = {
        "n_obs": len(t),
        "engine": {
            "balanced_accuracy": balanced_accuracy(p, t),
            "hit_rate": float(engine_hit.mean()),
        },
        "baselines": {},
    }
    best_baseline_name, best_baseline_hit = None, -np.inf
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
        out["baselines"][name] = {
            "n_obs": int(b_valid.sum()),
            "balanced_accuracy": balanced_accuracy(b, t),
            "hit_rate": float(b_hit.mean()),
        }
        if float(b_hit.mean()) > best_baseline_hit:
            best_baseline_hit, best_baseline_name = float(b_hit.mean()), name

    if best_baseline_name is not None:
        b = baselines[best_baseline_name].reindex(idx).astype("string")
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
                "baseline": best_baseline_name,
                "mean_hit_rate_diff": float(diff.mean()),
                "ci95": [lo, hi],
                "pvalue": pvalue,
            }
    return out


# --------------------------------------------------------------------------- #
# (a) Episode table
# --------------------------------------------------------------------------- #
# Concrete (sector, dimension) rows per episode, derived from
# phase_0_14_preregistration.json's episode_table -- COVID crash's
# "all volatility" is expanded to every basket sector that has a
# volatility dimension.
_ALL_VOL_SECTORS = [
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
]


def _episode_targets(episode_table: list[dict]) -> list[dict]:
    targets = []
    for ep in episode_table:
        name, start, end = ep["episode"], ep["start"], ep["end"]
        if name == "GFC":
            targets += [
                {
                    "sector": "Macro",
                    "dimension": "risk",
                    "label": "risk_off",
                    "episode": name,
                    "start": start,
                    "end": end,
                },
                {
                    "sector": "Macro",
                    "dimension": "credit",
                    "label": "wide",
                    "episode": name,
                    "start": start,
                    "end": end,
                },
            ]
        elif name == "Euro crisis":
            targets.append(
                {
                    "sector": "Macro",
                    "dimension": "risk",
                    "label": "risk_off",
                    "episode": name,
                    "start": start,
                    "end": end,
                }
            )
        elif name == "Taper tantrum":
            targets.append(
                {
                    "sector": "Macro",
                    "dimension": "yield_curve",
                    "label": "steep",
                    "episode": name,
                    "start": start,
                    "end": end,
                }
            )
        elif name == "Oil glut":
            targets += [
                {
                    "sector": "Commodities",
                    "dimension": "trend",
                    "label": "bear",
                    "episode": name,
                    "start": start,
                    "end": end,
                },
                {
                    "sector": "oil products",
                    "dimension": "term_structure",
                    "label": "contango",
                    "episode": name,
                    "start": start,
                    "end": end,
                },
            ]
        elif name == "COVID crash":
            targets.append(
                {
                    "sector": "Macro",
                    "dimension": "risk",
                    "label": "risk_off",
                    "episode": name,
                    "start": start,
                    "end": end,
                }
            )
            for sector in _ALL_VOL_SECTORS:
                targets.append(
                    {
                        "sector": sector,
                        "dimension": "volatility",
                        "label": "extreme",
                        "episode": name,
                        "start": start,
                        "end": end,
                    }
                )
        elif name == "Post-COVID commodity bull":
            targets.append(
                {
                    "sector": "Commodities",
                    "dimension": "trend",
                    "label": "bull",
                    "episode": name,
                    "start": start,
                    "end": end,
                }
            )
        elif name == "Energy backwardation":
            targets += [
                {
                    "sector": "oil products",
                    "dimension": "term_structure",
                    "label": "backwardation",
                    "episode": name,
                    "start": start,
                    "end": end,
                },
                {
                    "sector": "oil products",
                    "dimension": "carry",
                    "label": "positive",
                    "episode": name,
                    "start": start,
                    "end": end,
                },
            ]
        elif name == "Hiking cycle / inversion":
            targets += [
                {
                    "sector": "Macro",
                    "dimension": "yield_curve",
                    "label": "inverted",
                    "episode": name,
                    "start": start,
                    "end": end,
                },
                {
                    "sector": "Macro",
                    "dimension": "risk",
                    "label": "risk_off",
                    "episode": name,
                    "start": start,
                    "end": end,
                },
            ]
    return targets


def _lead_lag(labels: pd.Series, expected: str, start: str, end: str) -> float | None:
    """Signed trading days between episode onset and the label's first flip
    to `expected` within [start - 21td, end]. None if it never matches."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    window_idx = labels.index[(labels.index >= start_ts) & (labels.index <= end_ts)]
    pre_idx = labels.index[labels.index < start_ts][-21:]
    search_idx = pre_idx.append(window_idx)
    matches = labels.reindex(search_idx) == expected
    if not matches.any():
        return None
    first_match_pos = matches.to_numpy().argmax()
    onset_pos = len(pre_idx)
    return float(first_match_pos - onset_pos)


def score_episode_table(
    panel: pl.DataFrame, prereg: dict, disqualified: set[tuple[str, str]]
) -> dict:
    targets = _episode_targets(prereg["episode_table"])
    by_dim: dict[tuple[str, str], list[dict]] = {}
    for t in targets:
        by_dim.setdefault((t["sector"], t["dimension"]), []).append(t)

    results = {}
    lags = []
    for (sector, dimension), rows in by_dim.items():
        if (sector, dimension) in disqualified:
            results[f"{sector}|{dimension}"] = {
                "excluded": True,
                "reason": "disqualified in Phase 2",
            }
            continue
        labels = _sector_labels(panel, sector, dimension)
        if labels.empty:
            results[f"{sector}|{dimension}"] = {
                "excluded": True,
                "reason": "no data for this sector/dimension",
            }
            continue
        target = pd.Series(pd.NA, index=labels.index, dtype="string")
        row_detail = []
        for row in rows:
            start_ts, end_ts = pd.Timestamp(row["start"]), pd.Timestamp(row["end"])
            mask = (
                (labels.index >= start_ts)
                & (labels.index <= end_ts)
                & (labels.index < SCORING_CUTOFF)
            )
            if mask.any():
                target.loc[mask] = row["label"]
            lag = _lead_lag(labels, row["label"], row["start"], row["end"])
            row_detail.append(
                {
                    "episode": row["episode"],
                    "expected": row["label"],
                    "lead_lag_days": lag,
                }
            )
            if lag is not None:
                lags.append(lag)
        if target.notna().sum() == 0:
            results[f"{sector}|{dimension}"] = {
                "excluded": True,
                "reason": "episode windows outside available history",
            }
            continue
        baselines = _baselines(labels)
        scored = _score_against_target(labels, baselines, target)
        scored["episodes"] = row_detail
        results[f"{sector}|{dimension}"] = scored

    return {"per_sector_dimension": results, "lead_lag_days": lags}


# --------------------------------------------------------------------------- #
# (b) Mechanical labels
# --------------------------------------------------------------------------- #
def _representative_close(sector: str, sectors_meta: dict) -> pd.Series | None:
    symbols = sectors_meta[sector]["symbols_used"]
    if not symbols:
        return None
    close = load_bars(symbols[0])["close"]
    return close[close.index < SCORING_CUTOFF]


def score_mechanical_labels(
    panel: pl.DataFrame, sectors_meta: dict, disqualified: set[tuple[str, str]]
) -> dict:
    results = {}

    # volatility: forward 21d realized-vol top-tercile vs label in {high, extreme}
    for sector in sectors_meta:
        key = (sector, "volatility")
        if key in disqualified:
            continue
        labels = _sector_labels(panel, sector, "volatility")
        close = _representative_close(sector, sectors_meta)
        if labels.empty or close is None:
            continue
        fwd_vol = forward_realized_vol(close, horizon=21)
        fwd_vol = fwd_vol[fwd_vol.index < SCORING_CUTOFF].dropna()
        if len(fwd_vol) < MIN_HISTORY:
            continue
        threshold = fwd_vol.quantile(2 / 3)
        target = pd.Series(
            np.where(fwd_vol >= threshold, "high_tercile", "not_high_tercile"),
            index=fwd_vol.index,
            dtype="string",
        )
        pred = pd.Series(
            np.where(
                labels.reindex(target.index).isin(["high", "extreme"]),
                "high_tercile",
                "not_high_tercile",
            ),
            index=target.index,
            dtype="string",
        )
        pred = pred.where(labels.reindex(target.index).notna())
        baselines = _baselines(labels)
        binarized = {
            name: series.reindex(target.index)
            .isin(["high", "extreme"])
            .map({True: "high_tercile", False: "not_high_tercile"})
            .astype("string")
            .where(series.reindex(target.index).notna())
            for name, series in baselines.items()
        }
        results[f"{sector}|volatility|forward_21d_vol_tercile"] = _score_against_target(
            pred, binarized, target
        )

    # trend: sign of forward 63d return vs label in {bull vs bear} (sideways dropped)
    for sector in sectors_meta:
        key = (sector, "trend")
        if key in disqualified:
            continue
        labels = _sector_labels(panel, sector, "trend")
        close = _representative_close(sector, sectors_meta)
        if labels.empty or close is None:
            continue
        fwd_ret = forward_log_return(close, horizon=63)
        fwd_ret = fwd_ret[fwd_ret.index < SCORING_CUTOFF].dropna()
        target = pd.Series(
            np.where(fwd_ret > 0, "up", "down"), index=fwd_ret.index, dtype="string"
        )
        directional = labels.isin(["bull", "bear"])
        pred = labels.map({"bull": "up", "bear": "down"}).astype("string")
        pred = pred.where(directional)
        target = target.where(pred.reindex(target.index).notna())
        baselines = _baselines(labels)
        binarized = {
            name: series.map({"bull": "up", "bear": "down"}).astype("string")
            for name, series in baselines.items()
        }
        results[f"{sector}|trend|forward_63d_return_sign"] = _score_against_target(
            pred, binarized, target
        )

    # yield_curve (Macro only): contemporaneous sign of T10Y2Y vs {steep vs inverted}
    key = ("Macro", "yield_curve")
    if key not in disqualified:
        labels = _sector_labels(panel, "Macro", "yield_curve")
        if not labels.empty:
            from regime.loaders import load_fred_frame

            t10y2y = load_fred_frame(("T10Y2Y",))["T10Y2Y"]
            t10y2y = t10y2y[t10y2y.index < SCORING_CUTOFF].reindex(labels.index).ffill()
            directional = labels.isin(["steep", "inverted"])
            target = pd.Series(
                np.where(t10y2y > 0, "steep_side", "inverted_side"),
                index=labels.index,
                dtype="string",
            )
            target = target.where(t10y2y.notna())
            pred = labels.map(
                {"steep": "steep_side", "inverted": "inverted_side"}
            ).astype("string")
            pred = pred.where(directional)
            target = target.where(pred.notna())
            baselines = _baselines(labels)
            binarized = {
                name: series.map(
                    {"steep": "steep_side", "inverted": "inverted_side"}
                ).astype("string")
                for name, series in baselines.items()
            }
            results["Macro|yield_curve|contemporaneous_T10Y2Y_sign"] = (
                _score_against_target(pred, binarized, target)
            )

    # term_structure: contemporaneous sign of f1-f2 spread vs {backwardation vs contango}
    for sector in sectors_meta:
        key = (sector, "term_structure")
        if key in disqualified:
            continue
        labels = _sector_labels(panel, sector, "term_structure")
        if labels.empty:
            continue
        symbols = sectors_meta[sector]["symbols_used"]
        curve_symbol = next((s for s in symbols if load_curve(s) is not None), None)
        if curve_symbol is None:
            continue
        curve = load_curve(curve_symbol)
        assert curve is not None
        curve = curve[curve.index < SCORING_CUTOFF]
        spread = (curve["close_f1"] - curve["close_f2"]).reindex(labels.index).ffill()
        directional = labels.isin(["backwardation", "contango"])
        target = pd.Series(
            np.where(spread > 0, "backwardation_side", "contango_side"),
            index=labels.index,
            dtype="string",
        )
        target = target.where(spread.notna())
        pred = labels.map(
            {"backwardation": "backwardation_side", "contango": "contango_side"}
        ).astype("string")
        pred = pred.where(directional)
        target = target.where(pred.notna())
        baselines = _baselines(labels)
        binarized = {
            name: series.map(
                {"backwardation": "backwardation_side", "contango": "contango_side"}
            ).astype("string")
            for name, series in baselines.items()
        }
        results[f"{sector}|term_structure|contemporaneous_f1f2_spread_sign"] = (
            _score_against_target(pred, binarized, target)
        )

    return results


def main() -> None:
    panel = pl.read_parquet(PANEL_PATH)
    with open(f"{TMP}/phase_0_14_preregistration.json") as f:
        prereg = json.load(f)
    with open(f"{TMP}/phase_1_14_results.json") as f:
        phase1 = json.load(f)
    with open(f"{TMP}/phase_2_14_results.json") as f:
        phase2 = json.load(f)

    sectors_meta = {s["name"]: s for s in phase1["sectors"]}
    disqualified = {(d["sector"], d["dimension"]) for d in phase2["disqualified"]}

    print("Scoring episode table (ground truth a)...")
    episode_results = score_episode_table(panel, prereg, disqualified)

    print("Scoring mechanical labels (ground truth b)...")
    mechanical_results = score_mechanical_labels(panel, sectors_meta, disqualified)

    n_trials_a = sum(
        1
        for v in episode_results["per_sector_dimension"].values()
        if not v.get("excluded")
    )
    n_trials_b = sum(
        1 for v in mechanical_results.values() if not v.get("insufficient_data")
    )
    n_trials = n_trials_a + n_trials_b

    out = {
        "scoring_cutoff": str(SCORING_CUTOFF.date()),
        "disqualified_excluded": sorted(f"{s}|{d}" for s, d in disqualified),
        "n_trials": {
            "episode_table": n_trials_a,
            "mechanical_labels": n_trials_b,
            "total": n_trials,
        },
        "episode_table": episode_results,
        "mechanical_labels": mechanical_results,
    }
    with open(f"{TMP}/phase_3_14_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(
        f"n_trials: episode_table={n_trials_a} mechanical={n_trials_b} total={n_trials}"
    )
    lags = episode_results["lead_lag_days"]
    if lags:
        print(f"Median lead-lag: {np.median(lags):.1f} trading days (n={len(lags)})")
    print("Wrote phase_3_14_results.json")


if __name__ == "__main__":
    main()
