"""Phase 0 -- pre-registration for notebook 023 (Gabillon two-factor curve model).

Writes gates, thresholds, and the trial-count budget to disk BEFORE any fit result
is inspected. See NEXT_PROMPT.md for the full spec this implements.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib23

TMP = Path(__file__).resolve().parent

LIQUIDITY_VARIANTS = ["all", "vol_gt_0", "vol_gt_100"]
L_ESTIMATORS = ["joint_ls_monthly", "joint_ls_daily", "longend", "kalman"]
MODELS = ["model26", "model28"]
N_CONFIGS = len(LIQUIDITY_VARIANTS) * len(L_ESTIMATORS) * len(MODELS)  # 3*4*2 = 24

PREREG = {
    "written": "2026-09-21",
    "primary_metric": "held-out-maturity RMSE in $/bbl (Phase 6: fit on tau<=2y, "
    "predict tau in (2y, 9y], score against actual settlement)",
    "benchmark_set": [
        "flat_forward",
        "previous_curve_carried_forward",
        "cubic_spline",
        "nelson_siegel",
        "pca_2factor",
    ],
    "pass_threshold": (
        "Median held-out RMSE (model 26 or 28, best-performing pre-registered "
        "configuration on the train/validation sample) beats the best of the five "
        "Phase-4 benchmarks by >= 10% on a paired day-by-day comparison, with a "
        "bootstrap 95% CI on the paired RMSE gap excluding zero, after the "
        "multiple-testing deflation below. Anchored on Gabillon's own reported "
        "in-sample RMSE range of ~0.1-0.5 $/bbl calm / ~1.5 $/bbl crisis."
    ),
    "train_start": str(lib23.TRAIN_START.date()),
    "train_end": str(lib23.TRAIN_END.date()),
    "holdout_start": str(lib23.HOLDOUT_START.date()),
    "holdout_rule": "Holdout (2022-01-01 onward) is touched exactly once, in Phase 9, "
    "after every other phase and every configuration choice is frozen.",
    "liquidity_variants": LIQUIDITY_VARIANTS,
    "L_estimators": L_ESTIMATORS,
    "models": MODELS,
    "n_configs": N_CONFIGS,
    "bonferroni_alpha": lib23.bonferroni_alpha(N_CONFIGS, 0.05),
    "min_contracts_per_day": lib23.MIN_CONTRACTS_PER_DAY,
    "negative_price_event": {
        k: (str(v) if hasattr(v, "isoformat") else v)
        for k, v in lib23.NEGATIVE_PRICE_EVENT.items()
    },
    "declared_windows_for_phase5": {
        "2014_15_collapse": ["2014-06-01", "2015-12-31"],
        "2020_covid": ["2020-01-01", "2020-12-31"],
        "2020_covid_acute": ["2020-03-01", "2020-05-31"],
        "2022_russia_ukraine": ["2022-01-01", "2022-12-31"],
    },
}

out = TMP / "phase_0_23_preregistration.json"
out.write_text(json.dumps(PREREG, indent=2, default=str))
print(f"wrote {out}, n_configs={N_CONFIGS}")
