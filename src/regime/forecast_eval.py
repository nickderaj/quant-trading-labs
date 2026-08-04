"""Forward-looking targets and forecast-quality metrics for regime prediction.

Ported verbatim from ``ultron_finance.regime.forecast_eval``
(``../ultron/libs/finance/src/ultron_finance/regime/forecast_eval.py``).

All targets are indexed at the *prediction date* t and describe the interval
(t, t+h]; the last ``horizon`` rows are therefore NaN. A realized regime label at
t+h uses only data <= t+h, so it is a valid *target* -- but it must never be fed
back in as a feature. That invariant is enforced structurally here: ``.shift(-h)``
appears only in the target builders below, never in the forecaster.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------- #
# Targets (indexed at prediction date t; last ``horizon`` rows are NaN)
# --------------------------------------------------------------------------- #
def forward_labels(labels: pd.Series, horizon: int) -> pd.Series:
    """Realized regime label at t+h, indexed at t (``labels.shift(-h)``)."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    return labels.shift(-horizon).astype("string")


def forward_log_return(close: pd.Series, horizon: int) -> pd.Series:
    """Log return over (t, t+h], indexed at t."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    # A price ratio can go non-positive for real (e.g. WTI/CL=F traded negative
    # on 2020-04-20); log() of that is correctly NaN, so just suppress the warning.
    with np.errstate(invalid="ignore"):
        return cast(pd.Series, np.log(close.shift(-horizon) / close))


def forward_realized_vol(close: pd.Series, horizon: int, annualize: bool = True) -> pd.Series:
    """Realized vol of daily log returns over (t, t+h], indexed at t."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    with np.errstate(invalid="ignore"):
        daily = cast(pd.Series, np.log(close / close.shift(1)))
    vol = daily.rolling(horizon).std().shift(-horizon)
    return cast(pd.Series, vol * np.sqrt(TRADING_DAYS_PER_YEAR)) if annualize else vol


# --------------------------------------------------------------------------- #
# Classification metrics (NaN pairs dropped)
# --------------------------------------------------------------------------- #
def _aligned(pred: pd.Series, target: pd.Series) -> tuple[pd.Series, pd.Series]:
    frame = pd.DataFrame({"pred": pred, "target": target}).dropna()
    return frame["pred"].astype("string"), frame["target"].astype("string")


def hit_rate(pred: pd.Series, target: pd.Series) -> float:
    p, t = _aligned(pred, target)
    return float((p.to_numpy() == t.to_numpy()).mean()) if len(p) else float("nan")


def confusion(pred: pd.Series, target: pd.Series) -> pd.DataFrame:
    p, t = _aligned(pred, target)
    return pd.crosstab(t.rename("target"), p.rename("pred"))


def per_class_stats(pred: pd.Series, target: pd.Series) -> pd.DataFrame:
    """Precision/recall/support per target class."""
    p, t = _aligned(pred, target)
    classes = sorted(set(t.unique()) | set(p.unique()))
    rows: dict[str, dict[str, float]] = {}
    for cls in classes:
        tp = int(((p == cls) & (t == cls)).sum())
        fp = int(((p == cls) & (t != cls)).sum())
        support = int((t == cls).sum())
        rows[cls] = {
            "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
            "recall": tp / support if support else float("nan"),
            "support": float(support),
        }
    return pd.DataFrame.from_dict(rows, orient="index", columns=["precision", "recall", "support"])


def balanced_accuracy(pred: pd.Series, target: pd.Series) -> float:
    """Mean per-class recall (robust to class imbalance)."""
    stats = per_class_stats(pred, target)
    recalls = stats["recall"].dropna()
    return float(recalls.mean()) if len(recalls) else float("nan")


def transition_recall(pred: pd.Series, target: pd.Series, current: pd.Series) -> float:
    """Accuracy restricted to rows where the regime actually changes
    (target != current). Persistence scores exactly 0 here by construction --
    this is the anti-persistence stress metric."""
    frame = pd.DataFrame({"pred": pred, "target": target, "cur": current}).dropna()
    changed = frame[frame["target"].astype("string") != frame["cur"].astype("string")]
    if changed.empty:
        return float("nan")
    hits = changed["pred"].astype("string") == changed["target"].astype("string")
    return float(hits.mean())


def brier_score(probs: pd.DataFrame, target: pd.Series) -> float:
    """Multiclass Brier score: mean over rows of sum_k (p_k - 1[target==k])^2.
    Lower is better; range [0, 2]."""
    aligned = probs.dropna(how="all")
    tgt = target.reindex(aligned.index).astype("string")
    mask = tgt.notna()
    aligned, tgt = aligned[mask], tgt[mask]
    if aligned.empty:
        return float("nan")
    onehot = pd.get_dummies(tgt).reindex(columns=aligned.columns, fill_value=0)
    diff = aligned.to_numpy(dtype=float) - onehot.to_numpy(dtype=float)
    return float((diff**2).sum(axis=1).mean())


# --------------------------------------------------------------------------- #
# Continuous-score metric
# --------------------------------------------------------------------------- #
def information_coefficient(
    score: pd.Series, forward: pd.Series, non_overlapping_step: int | None = None
) -> float:
    """Spearman rank correlation of ``score`` vs a forward target.

    With ``non_overlapping_step=h`` the series is subsampled every h-th row so
    overlapping forward windows don't inflate apparent significance."""
    frame = pd.DataFrame({"score": score, "fwd": forward}).dropna()
    if non_overlapping_step and non_overlapping_step > 1:
        frame = frame.iloc[::non_overlapping_step]
    if len(frame) < 3:
        return float("nan")
    rho, _ = spearmanr(frame["score"], frame["fwd"])
    return float(rho)


# --------------------------------------------------------------------------- #
# Walk-forward splits
# --------------------------------------------------------------------------- #
def walk_forward_splits(
    index: pd.DatetimeIndex,
    min_train: int,
    test_size: int,
    step: int | None = None,
    expanding: bool = True,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Ordered (train_index, test_index) folds; test windows never overlap and
    every train end precedes its test start. ``expanding=False`` gives a rolling
    train window of length ``min_train``."""
    step = step or test_size
    folds: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    start = min_train
    while start + test_size <= len(index):
        train = index[:start] if expanding else index[start - min_train : start]
        test = index[start : start + test_size]
        folds.append((train, test))
        start += step
    return folds


# --------------------------------------------------------------------------- #
# Bundle
# --------------------------------------------------------------------------- #
def evaluate_forecast(
    pred_labels: pd.Series,
    target_labels: pd.Series,
    probs: pd.DataFrame | None = None,
    score: pd.Series | None = None,
    fwd_return: pd.Series | None = None,
    fwd_vol: pd.Series | None = None,
    current_labels: pd.Series | None = None,
    horizon: int = 1,
) -> dict[str, float]:
    """One row of metrics for a (prediction, target) pair."""
    p, _t = _aligned(pred_labels, target_labels)
    out: dict[str, float] = {
        "n_obs": float(len(p)),
        "accuracy": hit_rate(pred_labels, target_labels),
        "balanced_accuracy": balanced_accuracy(pred_labels, target_labels),
    }
    if current_labels is not None:
        out["transition_recall"] = transition_recall(pred_labels, target_labels, current_labels)
    if probs is not None:
        out["brier"] = brier_score(probs, target_labels)
    if score is not None and fwd_return is not None:
        out["ic_return"] = information_coefficient(score, fwd_return, horizon)
    if score is not None and fwd_vol is not None:
        out["ic_vol"] = information_coefficient(score, fwd_vol, horizon)
    return out


__all__ = [
    "balanced_accuracy",
    "brier_score",
    "confusion",
    "evaluate_forecast",
    "forward_labels",
    "forward_log_return",
    "forward_realized_vol",
    "hit_rate",
    "information_coefficient",
    "per_class_stats",
    "transition_recall",
    "walk_forward_splits",
]
