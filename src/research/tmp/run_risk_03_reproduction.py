"""Gates PR, PH, DT: the reproduction gate (NEXT_PROMPT.md sec 6.1, sec 10).

Nothing after this gate is built until it passes -- a port that changes a
number is a port that changed behaviour, and the whole value of `src/risk/`
is that its guarantees are 008's guarantees.

**PR** (Port reproduction). Re-runs `run_phase_7_risk_engine.py`'s exact
methodology (imported, not reimplemented) -- which calls `commod_lib8.py`'s
`fit_risk_model`/`ewma_vol`/`portfolio_risk`, all of which are now shims onto
`src/risk/` -- and compares the fresh result field-by-field against the
*stored* `phase_7_results.json` at `rtol=1e-12`. This is "the old notebook
runs the new code" (sec 3.3): the reproduction methodology is untouched: only
what it calls underneath has moved.

**PH** (Holdout reproduction). Per sec 2.1/6.1, the futures holdout is not
re-spent here. This checks the *stored* `phase_8_holdout_results.json`'s own
summary against the numbers NEXT_PROMPT.md sec 1/10 commits to
(`RE_holdout_pass_count == 14`, `CE_holdout_reject_1pct_count == 11`) --
no holdout data is read or recomputed by this script.

**DT** (Determinism). Two runs of `portfolio_risk` with the same seed, on
the real fitted 16-product book, must be bit-identical.

Writes `src/research/tmp/run_risk_03_reproduction_results.json`.
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

PHASE7_JSON = "src/research/tmp/phase_7_results.json"
PHASE8_JSON = "src/research/tmp/phase_8_holdout_results.json"
OUT_PATH = "src/research/tmp/run_risk_03_reproduction_results.json"

RTOL = 1e-12


def _load_phase7_module():
    """Import run_phase_7_risk_engine.py as a module without executing its
    __main__ block (which would overwrite phase_7_results.json)."""
    spec = importlib.util.spec_from_file_location(
        "run_phase_7_risk_engine", "src/research/tmp/run_phase_7_risk_engine.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def recompute_phase7_results(
    mod,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Reproduce run_phase_7_risk_engine.main()'s result dict without
    touching the stored JSON, using the exact same functions it uses."""
    research.set_seed(0)
    families = mod.load_family_map()

    models = {}
    returns_by_product = {}
    coverage_results = {}
    for p, family in families.items():
        ret, dates = mod.load_returns(p)
        if len(ret) < 500:
            continue
        returns_by_product[p] = (ret, dates)
        model = C.fit_risk_model(ret, p, family)
        if model is None:
            continue
        models[p] = model
        coverage_results[p] = mod.oos_coverage_test(ret, family, p)

    n_pass_01 = sum(
        1 for r in coverage_results.values() if r.get("0.01", {}).get("pass")
    )
    n_total = len(coverage_results)
    gate_re = {
        "n_products_passing_1pct_coverage": n_pass_01,
        "n_products_total": n_total,
        "threshold": 14,
        "fires": n_pass_01 >= 14,
    }

    weights = dict.fromkeys(models, 1.0 / len(models)) if models else {}
    hist_returns = {p: returns_by_product[p][0] for p in models}
    min_len = min(len(v) for v in hist_returns.values()) if hist_returns else 0
    hist_returns_aligned = {p: v[-min_len:] for p, v in hist_returns.items()}

    portfolio = {}
    if len(models) >= 2:
        for dep in ["empirical", "gaussian", "t"]:
            portfolio[dep] = C.portfolio_risk(
                models,
                weights,
                dependence=dep,
                historical_returns=hist_returns_aligned,
                n_sims=20000,
                seed=0,
                t_df=5.0,
            )

    stress = mod.stress_portfolio(models, weights, returns_by_product) if models else {}

    return (
        {
            "family_map": families,
            "oos_coverage": coverage_results,
            "gate_RE": gate_re,
            "portfolio_risk": portfolio,
            "stress_scenarios": stress,
            "_config": {"weights": weights},
        },
        models,
        hist_returns_aligned,
    )


def _compare(
    path: str, fresh: Any, stored: Any, rtol: float, mismatches: list[str]
) -> None:
    if isinstance(stored, dict):
        if not isinstance(fresh, dict):
            mismatches.append(
                f"{path}: type mismatch (stored dict, fresh {type(fresh)})"
            )
            return
        for k, v in stored.items():
            if k not in fresh:
                mismatches.append(f"{path}.{k}: missing from fresh result")
                continue
            _compare(f"{path}.{k}", fresh[k], v, rtol, mismatches)
    elif isinstance(stored, (int, float)) and not isinstance(stored, bool):
        try:
            if not np.isclose(float(fresh), float(stored), rtol=rtol, atol=1e-15):
                mismatches.append(f"{path}: fresh={fresh!r} stored={stored!r}")
        except (TypeError, ValueError):
            mismatches.append(
                f"{path}: non-numeric compare fresh={fresh!r} stored={stored!r}"
            )
    else:
        if fresh != stored:
            mismatches.append(f"{path}: fresh={fresh!r} stored={stored!r}")


def gate_pr(mod) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    fresh, models, hist_aligned = recompute_phase7_results(mod)
    with open(PHASE7_JSON) as f:
        stored = json.load(f)
    stored_no_config = {k: v for k, v in stored.items() if k != "_config"}
    fresh_no_config = {k: v for k, v in fresh.items() if k != "_config"}

    mismatches: list[str] = []
    _compare("phase7", fresh_no_config, stored_no_config, RTOL, mismatches)

    result = {
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:50],
        "gate_RE_fires": fresh["gate_RE"]["fires"],
        "gate_RE_pass_count": fresh["gate_RE"]["n_products_passing_1pct_coverage"],
        "pass": len(mismatches) == 0
        and fresh["gate_RE"]["fires"]
        and fresh["gate_RE"]["n_products_passing_1pct_coverage"] == 15,
    }
    return result, models, hist_aligned


def gate_ph() -> dict[str, Any]:
    """Per sec 6.1: 'the same field-by-field comparison ... and every
    per-product RE_holdout block' -- not only the two summary scalars.

    No holdout data is read or recomputed: everything here is derived from
    the stored `phase_8_holdout_results.json`'s own `per_product` block, the
    same file the summary scalars come from. This catches a summary that has
    drifted from the per-product detail it's supposed to aggregate (e.g. a
    hand-edited or partially-regenerated stored file), and pins down the
    specific per-product reshuffling NEXT_PROMPT.md sec 1 commits to: PA
    passes on holdout, RB and SI fail.
    """
    with open(PHASE8_JSON) as f:
        stored = json.load(f)
    summary = stored["summary"]
    per_product = stored["per_product"]

    mismatches: list[str] = []

    recomputed_re_pass = sum(1 for r in per_product.values() if r["RE_holdout"]["pass"])
    recomputed_ce_reject = sum(
        1
        for r in per_product.values()
        if r.get("CE_holdout", {}).get("0.01", {}).get("p") is not None
        and r["CE_holdout"]["0.01"]["p"] < 0.05
    )

    checks = {
        "RE_holdout_pass_count": (summary.get("RE_holdout_pass_count"), 14),
        "CE_holdout_reject_1pct_count": (
            summary.get("CE_holdout_reject_1pct_count"),
            11,
        ),
        "RE_holdout_pass_count_recomputed_from_per_product": (
            recomputed_re_pass,
            14,
        ),
        "CE_holdout_reject_1pct_count_recomputed_from_per_product": (
            recomputed_ce_reject,
            11,
        ),
    }
    for k, (got, want) in checks.items():
        if got != want:
            mismatches.append(f"{k}: stored={got} expected={want}")

    if summary.get("RE_holdout_pass_count") != recomputed_re_pass:
        mismatches.append(
            "summary.RE_holdout_pass_count "
            f"({summary.get('RE_holdout_pass_count')}) does not match a fresh "
            f"count over stored per_product RE_holdout.pass ({recomputed_re_pass})"
        )
    if summary.get("CE_holdout_reject_1pct_count") != recomputed_ce_reject:
        mismatches.append(
            "summary.CE_holdout_reject_1pct_count "
            f"({summary.get('CE_holdout_reject_1pct_count')}) does not match a "
            "fresh count over stored per_product CE_holdout['0.01'].p<0.05 "
            f"({recomputed_ce_reject})"
        )

    # NEXT_PROMPT.md sec 1: "On holdout, RB and SI fail while passing in
    # development, and PA passes" -- the specific reshuffling that makes
    # the per-product block worth checking at all, not just the aggregate.
    expected_re_pass = {"PA": True, "RB": False, "SI": False}
    for product, want_pass in expected_re_pass.items():
        if product not in per_product:
            mismatches.append(
                f"per_product.{product}: missing from stored holdout JSON"
            )
            continue
        got_pass = per_product[product]["RE_holdout"]["pass"]
        if got_pass != want_pass:
            mismatches.append(
                f"per_product.{product}.RE_holdout.pass: stored={got_pass} "
                f"expected={want_pass}"
            )

    required_fields = {
        "n",
        "observed_rate",
        "kupiec_p",
        "christoffersen_independence_p",
        "pass",
    }
    for product, block in per_product.items():
        re = block.get("RE_holdout")
        if re is None:
            mismatches.append(f"per_product.{product}.RE_holdout: missing")
            continue
        missing_fields = required_fields - set(re)
        if missing_fields:
            mismatches.append(
                f"per_product.{product}.RE_holdout: missing fields {sorted(missing_fields)}"
            )

    return {
        "checks": {k: v[0] for k, v in checks.items()},
        "per_product_re_holdout_pass": {
            p: per_product[p]["RE_holdout"]["pass"] for p in per_product
        },
        "mismatches": mismatches,
        "pass": len(mismatches) == 0,
        "note": (
            "no holdout data read or recomputed -- everything derived from the "
            "stored JSON, including a recount over its own per_product block"
        ),
    }


def gate_dt(models: dict, hist_aligned: dict) -> dict[str, Any]:
    if len(models) < 2:
        return {
            "pass": False,
            "note": "insufficient fitted models for a portfolio DT check",
        }
    weights = dict.fromkeys(models, 1.0 / len(models))
    r1 = C.portfolio_risk(
        models,
        weights,
        dependence="t",
        historical_returns=hist_aligned,
        n_sims=20000,
        t_df=5.0,
        seed=0,
    )
    r2 = C.portfolio_risk(
        models,
        weights,
        dependence="t",
        historical_returns=hist_aligned,
        n_sims=20000,
        t_df=5.0,
        seed=0,
    )
    bit_identical = r1 == r2
    return {"pass": bit_identical, "bit_identical": bit_identical}


def main() -> None:
    t0 = time.time()
    mod = _load_phase7_module()

    print("Gate PR: reproducing Phase 7 via the promoted src/risk/ code...", flush=True)
    pr_result, models, hist_aligned = gate_pr(mod)
    print(
        f"  PR pass={pr_result['pass']} n_mismatches={pr_result['n_mismatches']}",
        flush=True,
    )

    print(
        "Gate PH: checking stored holdout summary against committed numbers...",
        flush=True,
    )
    ph_result = gate_ph()
    print(f"  PH pass={ph_result['pass']}", flush=True)

    print("Gate DT: determinism check on portfolio_risk...", flush=True)
    dt_result = gate_dt(models, hist_aligned)
    print(f"  DT pass={dt_result['pass']}", flush=True)

    results = {
        "gate_PR": pr_result,
        "gate_PH": ph_result,
        "gate_DT": dt_result,
        "all_pass": pr_result["pass"] and ph_result["pass"] and dt_result["pass"],
    }
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nall_pass={results['all_pass']}")
    print(f"written {OUT_PATH} in {time.time() - t0:.1f}s")
    if not results["all_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
