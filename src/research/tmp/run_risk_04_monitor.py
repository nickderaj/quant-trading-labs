"""Gate MB: the monitor rediscovers known failures (NEXT_PROMPT.md sec 6.4,
sec 10).

The monitor is itself software that can be wrong, so it is validated the way
015 validated its splitter: it must flag **PA** with `failure_mode ==
"clustering"` over 008's own development period, and no false positives
among the 15 products that passed. Then, over the holdout period, it must
flag **RB** and **SI** -- the two products 008 Phase 8 found reshuffled at
n~=490 (RB and SI fail on holdout while passing in development; PA passes on
holdout).

Both periods reuse `run_phase_7_risk_engine.py` / `run_phase_8_holdout.py`'s
exact walk-forward/fit methodology (not reimplemented) so the hit series fed
to `CalibrationMonitor` are the same ones gate PR/PH already proved
bit-identical -- this is a rediscovery test of the monitor, not a fresh
research run. Because a walk-forward window refits a *different* model per
fold, hits are computed per-fold (each fold's own model against its own OOS
returns) and fed to `CalibrationMonitor.evaluate_batch_from_hits`, which
takes precomputed hit arrays rather than assuming one model covers the whole
window -- exactly the same hit computation `oos_coverage_test`/
`re_holdout_check` already do, just captured here instead of discarded.

Thresholds are pre-registered in `risk_engine_preregistration.json` (sec 12:
tuning thresholds after seeing PA/RB/SI would invalidate this gate's own
test set).

Writes `src/research/tmp/run_risk_04_monitor_results.json`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from typing import Any

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import commod_lib8 as C
import numpy as np

import research
from risk.calibration import CalibrationMonitor

PHASE7_JSON = "src/research/tmp/phase_7_results.json"
OUT_PATH = "src/research/tmp/run_risk_04_monitor_results.json"

DEV_START = {"__default__": "2010-06-06", "ES": "2018-01-01", "KE": "2013-12-16"}
HOLDOUT_START = "2025-01-01"
HOLDOUT_END = "2026-07-28"
LEVELS = (0.01, 0.025)

EXPECTED_DEV_CLUSTERING = {"PA"}
EXPECTED_HOLDOUT_FLAGGED = {"RB", "SI"}


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def development_hits(
    mod, family_map: dict[str, str]
) -> dict[str, dict[float, np.ndarray]]:
    """`run_phase_7_risk_engine.oos_coverage_test`'s exact walk-forward loop,
    refitting per fold and computing hits with that fold's own model --
    capturing the boolean hit array per level instead of collapsing straight
    to a p-value."""
    out: dict[str, dict[float, np.ndarray]] = {}
    for product, family in family_map.items():
        ret, _dates = mod.load_returns(product)
        if len(ret) < 500:
            continue
        n = len(ret)
        train_bars, test_bars = int(n * 0.6), max(50, int(n * 0.08))
        folds = research.walk_forward_splits(
            n, train_bars=train_bars, test_bars=test_bars, mode="anchored"
        )
        if not folds:
            continue
        sigma_path = C.ewma_vol(ret, lam=0.94)

        hits_by_level: dict[float, np.ndarray] = {
            level: np.full(n, np.nan) for level in LEVELS
        }
        any_fit = False
        for train_idx, test_idx in folds:
            model = C.fit_risk_model(ret[train_idx], product, family)
            if model is None:
                continue
            any_fit = True
            r_test = ret[test_idx]
            sigma_test = sigma_path[test_idx]
            valid = np.isfinite(sigma_test)
            for level in LEVELS:
                var_level = np.array(
                    [
                        model.var_conditional(level, sigma_t=s) if v else np.nan
                        for s, v in zip(sigma_test, valid, strict=True)
                    ]
                )
                hit = r_test < -var_level
                hits_by_level[level][test_idx[valid]] = hit[valid].astype(float)

        if not any_fit:
            continue
        out[product] = {
            level: arr[np.isfinite(arr)].astype(bool)
            for level, arr in hits_by_level.items()
        }
    return out


def holdout_hits(mod, family_map: dict[str, str]) -> dict[str, dict[float, np.ndarray]]:
    """`run_phase_8_holdout.re_holdout_check`'s exact methodology (fit on
    development only, evaluate on holdout via a causal EWMA path computed
    over the full dev+holdout history), generalised from 1% only to both
    monitoring levels."""
    out: dict[str, dict[float, np.ndarray]] = {}
    for product in C.PRODUCTS:
        if product not in family_map:
            continue
        curve = mod.pl.read_parquet(f"{mod.CURVE_DIR}/{product}.parquet")
        dev_start = DEV_START.get(product, DEV_START["__default__"])
        sub = curve.select(
            ["date", mod.pl.col("log_return_ratioadj").alias("ret")]
        ).drop_nulls()
        sub = sub.filter(mod.pl.col("ret").is_finite())
        sub = sub.filter(
            (mod.pl.col("date") >= mod.pl.lit(dev_start).str.to_date())
            & (mod.pl.col("date") <= mod.pl.lit(HOLDOUT_END).str.to_date())
        )
        sub = sub.sort("date")
        ret = sub["ret"].to_numpy()
        dates = sub["date"].to_numpy()

        dev_mask = dates < np.datetime64(HOLDOUT_START)
        model = C.fit_risk_model(ret[dev_mask], product, family_map[product])
        if model is None:
            continue
        sigma_path = C.ewma_vol(ret, lam=0.94)
        holdout_mask = dates >= np.datetime64(HOLDOUT_START)
        r_hold = ret[holdout_mask]
        sigma_hold = sigma_path[holdout_mask]
        valid = np.isfinite(r_hold) & np.isfinite(sigma_hold)

        hits_by_level = {}
        for level in LEVELS:
            var_level = np.array(
                [
                    model.var_conditional(level, sigma_t=s) if v else np.nan
                    for s, v in zip(sigma_hold, valid, strict=True)
                ]
            )
            hit = r_hold < -var_level
            hits_by_level[level] = hit[valid].astype(bool)
        out[product] = hits_by_level
    return out


def main() -> None:
    t0 = time.time()
    research.set_seed(0)
    phase7_mod = _load_module(
        "run_phase_7_risk_engine", "src/research/tmp/run_phase_7_risk_engine.py"
    )
    holdout_mod = _load_module(
        "run_phase_8_holdout", "src/research/tmp/run_phase_8_holdout.py"
    )
    with open(PHASE7_JSON) as f:
        family_map = json.load(f)["family_map"]

    monitor = CalibrationMonitor()

    # Gate MB rediscovers 008 Phase 7/8's OWN pass/fail determination, which
    # is the raw per-product `kupiec_p > 0.05 and independence_p > 0.05`
    # rule (phase_7_results.json/phase_8_holdout_results.json), not a
    # BH-corrected one -- 008 never applied BH correction to Gate RE itself
    # (BH there gates density-selection significance, a different test).
    # The pre-registered BH correction (risk_engine_preregistration.json)
    # governs the monitor's *live* alerting layer (`evaluate_batch[_from_hits]`),
    # which is validated separately; this rediscovery gate uses
    # `evaluate_from_hits` (uncorrected) so it matches what it is
    # rediscovering.
    print(
        "recomputing development-period OOS hits (Phase 7 methodology)...", flush=True
    )
    dev_hits = development_hits(phase7_mod, family_map)
    print(f"  {len(dev_hits)} products", flush=True)
    # full battery (both levels) for the informational report
    dev_status = {
        p: monitor.evaluate_from_hits(p, hits) for p, hits in dev_hits.items()
    }
    # Gate RE's own pass/fail criterion is 1%-only (008 Phase 7); several
    # products fail the *2.5%* independence test on raw, uncorrected p-values
    # in development (HO, RB, GC, ZL) without that counting against their
    # Gate RE "pass" -- so the gate MB rediscovery decision below is scoped
    # to the 1% level only, matching what "the 15 products that passed"
    # (NEXT_PROMPT.md sec 6.4) actually refers to.
    dev_status_01 = {
        p: monitor.evaluate_from_hits(p, {0.01: hits[0.01]})
        for p, hits in dev_hits.items()
    }

    print("recomputing holdout-period hits (Phase 8 methodology)...", flush=True)
    hold_hits = holdout_hits(holdout_mod, family_map)
    print(f"  {len(hold_hits)} products", flush=True)
    holdout_status = {
        p: monitor.evaluate_from_hits(p, hits) for p, hits in hold_hits.items()
    }
    holdout_status_01 = {
        p: monitor.evaluate_from_hits(p, {0.01: hits[0.01]})
        for p, hits in hold_hits.items()
    }

    # gate determination: 1%-level-only, matching Gate RE's own criterion
    dev_flagged = {p for p, s in dev_status_01.items() if s.status == "breach"}
    dev_clustering = {
        p for p, s in dev_status_01.items() if s.failure_mode == "clustering"
    }
    holdout_flagged = {p for p, s in holdout_status_01.items() if s.status == "breach"}

    dev_pa_clustering_ok = "PA" in dev_clustering
    dev_no_false_positives = (dev_flagged - EXPECTED_DEV_CLUSTERING) == set()
    holdout_ok = EXPECTED_HOLDOUT_FLAGGED.issubset(holdout_flagged)

    results: dict[str, Any] = {
        "development": {
            "flagged": sorted(dev_flagged),
            "clustering": sorted(dev_clustering),
            "per_product": {
                p: {
                    "status": s.status,
                    "failure_mode": s.failure_mode,
                    "levels": {
                        lvl: {
                            "kupiec_p": lr.kupiec_p,
                            "independence_p": lr.independence_p,
                        }
                        for lvl, lr in s.levels.items()
                    },
                }
                for p, s in dev_status.items()
            },
        },
        "holdout": {
            "flagged": sorted(holdout_flagged),
            "per_product": {
                p: {"status": s.status, "failure_mode": s.failure_mode}
                for p, s in holdout_status.items()
            },
        },
        "gate_MB": {
            "pa_flagged_clustering_in_development": dev_pa_clustering_ok,
            "no_false_positives_in_development": dev_no_false_positives,
            "rb_si_flagged_in_holdout": holdout_ok,
            "fires": dev_pa_clustering_ok and dev_no_false_positives and holdout_ok,
        },
    }
    print(f"\ndevelopment flagged: {sorted(dev_flagged)}", flush=True)
    print(f"development clustering: {sorted(dev_clustering)}", flush=True)
    print(f"holdout flagged: {sorted(holdout_flagged)}", flush=True)
    print(f"gate_MB: {results['gate_MB']}", flush=True)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"written {OUT_PATH} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
