"""11c Phase 1: the walk-forward entry-time loss classifier, Gate LC's first
leg (NEXT_PROMPT.md sec 6 / phase_6_11a_results.json's pre-registered
criterion: "a classifier trained walk-forward achieves out-of-sample AUC >
0.60 on the stop-exit label").

Reuses this repo's own `research.walk_forward_splits` for fold indices
(same construction 002_walk_forward_multi_asset.ipynb uses for its own
train/test grid) -- unmodified, just applied over trades instead of daily
bars, since 57 trades over 14.5 years is this notebook's native sampling
unit, not bars. The external program's own methodological warning applies
double here: `walk_forward_splits(mode="anchored")` is an *expanding*
window (no lookahead across the train/test boundary), never scored
in-sample.

One pre-declared feature set (15 entry-time features from Phase 0), one
model class (logistic regression, `torch.nn.Linear` + BCE loss) -- neither
swept, matching Gate LC's DSR n_trials=4 (four origin offsets only,
cross-checked against phase_6_11a_results.json in Phase 2).

Writes phase_1_11c_results.json.
"""

import json
import sys

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import numpy as np
import spread_lib11 as S11
import torch
from torch import nn

import research

IN_PATH = "src/research/tmp/phase_0_11c_results.json"
OUT_PATH = "src/research/tmp/phase_1_11c_results.json"

TRAIN_TRADES = 30
TEST_TRADES = 5
STEP_TRADES = 5
ORIGIN_OFFSETS = [0, 1, 2, 3]
EPOCHS = 300
LR = 0.05
WEIGHT_DECAY = 1.0  # strong L2: 15 features against ~30-trade train folds


def load_matrix() -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    with open(IN_PATH) as f:
        d = json.load(f)
    feature_names = d["feature_names"]
    trades = [t for t in d["trades"] if all(t[f] is not None for f in feature_names)]
    x = np.array([[t[f] for f in feature_names] for t in trades], dtype=float)
    y = np.array([t["label_stop"] for t in trades], dtype=float)
    return x, y, feature_names, trades


def train_fold(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
) -> np.ndarray:
    research.set_seed(0)
    xt = torch.tensor(x_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    xv = torch.tensor(x_test, dtype=torch.float32)

    mean, std = research._standardize_fit(xt)
    xt_s = research._standardize_apply(xt, mean, std)
    xv_s = research._standardize_apply(xv, mean, std)

    model = nn.Linear(xt_s.shape[1], 1)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for _ in range(EPOCHS):
        opt.zero_grad()
        logits = model(xt_s)
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()

    with torch.no_grad():
        probs = torch.sigmoid(model(xv_s)).squeeze(1).numpy()
    return probs


def main() -> None:
    x, y, feature_names, _trades = load_matrix()
    n = len(y)

    results_by_offset: dict[str, dict] = {}
    stitched_by_offset: dict[str, dict] = {}
    for offset in ORIGIN_OFFSETS:
        splits = research.walk_forward_splits(
            n=n,
            train_bars=TRAIN_TRADES,
            test_bars=TEST_TRADES,
            step_bars=STEP_TRADES,
            mode="anchored",
            embargo_bars=0,
            origin_offset=offset,
        )
        oof_idx: list[int] = []
        oof_pred: list[float] = []
        oof_true: list[float] = []
        fold_aucs = []
        for train_idx, test_idx in splits:
            probs = train_fold(x[train_idx], y[train_idx], x[test_idx])
            fold_auc = S11.roc_auc_score(y[test_idx], probs)
            fold_aucs.append(None if np.isnan(fold_auc) else float(fold_auc))
            oof_idx.extend(test_idx.tolist())
            oof_pred.extend(probs.tolist())
            oof_true.extend(y[test_idx].tolist())

        stitched_auc = (
            S11.roc_auc_score(np.array(oof_true), np.array(oof_pred))
            if oof_true
            else float("nan")
        )
        results_by_offset[str(offset)] = {
            "n_folds": len(splits),
            "n_oof_trades": len(oof_idx),
            "fold_aucs": fold_aucs,
            "stitched_oof_auc": float(stitched_auc)
            if np.isfinite(stitched_auc)
            else None,
        }
        stitched_by_offset[str(offset)] = {
            "trade_idx": oof_idx,
            "pred_prob_stop": oof_pred,
            "true_label_stop": oof_true,
        }

    aucs: list[float] = [
        float(r["stitched_oof_auc"])
        for r in results_by_offset.values()
        if r["stitched_oof_auc"] is not None
    ]
    all_offsets_above_060 = len(aucs) == len(ORIGIN_OFFSETS) and all(
        a > 0.60 for a in aucs
    )

    out = {
        "n_trades_used": n,
        "n_trades_dropped_incomplete_features": 57 - n,
        "feature_names": feature_names,
        "model_class": "logistic regression (torch.nn.Linear + BCEWithLogitsLoss)",
        "split_config": {
            "train_trades": TRAIN_TRADES,
            "test_trades": TEST_TRADES,
            "step_trades": STEP_TRADES,
            "mode": "anchored",
            "embargo_bars": 0,
            "origin_offsets": ORIGIN_OFFSETS,
            "unit_note": (
                "origin_offset here is in TRADES, not calendar bars -- this "
                "notebook's own discretization of the programme's 0/7/14/21-"
                "day-offset convention, since the sampling unit for this "
                "classifier is trades (n=55), not daily bars."
            ),
        },
        "results_by_offset": results_by_offset,
        "stitched_oof_predictions_by_offset": stitched_by_offset,
        "all_offsets_auc_above_0_60": all_offsets_above_060,
        "_note": (
            "Walk-forward (anchored/expanding), no in-sample scoring. One "
            "pre-declared feature set, one model class, four origin offsets "
            "-- DSR n_trials=4, matching phase_6_11a_results.json's Gate LC "
            "pre-registration (checked in Phase 2)."
        ),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)

    auc_str = ", ".join(
        f"offset{k}={v['stitched_oof_auc']}" for k, v in results_by_offset.items()
    )
    print(
        f"Phase 1 (11c): n_used={n} {auc_str} "
        f"all_offsets_above_0.60={all_offsets_above_060}"
    )


if __name__ == "__main__":
    main()
