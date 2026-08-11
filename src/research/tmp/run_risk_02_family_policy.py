"""Gate FS: is per-product family selection worth its complexity?
(NEXT_PROMPT.md sec 5.2, sec 10)

008 Gate CD found only 2/16 products (GC, SI) had a BH-significant Phase 3
winner, and both were won by the same family (`ged`) -- per-product
selection is therefore mostly *not* statistically distinguishable. This runs
the Phase 7 walk-forward OOS coverage battery once under three pre-specified
family policies and reports Gate RE's pass count under each:

- P1: the shipped per-product map (family_map_v1.json)
- P2: `ged` everywhere
- P3: `ged` everywhere except products with a BH-significant Phase 3 winner
  (per `family_map_v1.json`'s `best_wins_significantly_bh` flag), which keep
  their own family

n_trials = 3, pre-committed in risk_engine_preregistration.json before this
script ran: ship whichever policy has the highest Gate RE pass count; a tie
is broken in favour of the simplest policy (P2).

Writes `src/research/tmp/run_risk_02_family_policy_results.json`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import commod_lib8 as C

import research
from risk.families import load_family_map

FAMILY_MAP_JSON = "src/risk/configs/family_map_v1.json"
OUT_PATH = "src/research/tmp/run_risk_02_family_policy_results.json"
PREREG_PATH = "src/research/tmp/risk_engine_preregistration.json"


def _load_phase7_module():
    spec = importlib.util.spec_from_file_location(
        "run_phase_7_risk_engine", "src/research/tmp/run_phase_7_risk_engine.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_policies() -> dict[str, dict[str, str]]:
    fm = load_family_map("v1")
    p1 = {p: e["family"] for p, e in fm.products.items()}
    p2 = dict.fromkeys(p1, "ged")
    p3 = {
        p: (p1[p] if e.get("best_wins_significantly_bh") else "ged")
        for p, e in fm.products.items()
    }
    return {"P1": p1, "P2": p2, "P3": p3}


def run_policy(mod, family_map: dict[str, str]) -> dict:
    research.set_seed(0)
    coverage_results = {}
    for p, family in family_map.items():
        ret, _dates = mod.load_returns(p)
        if len(ret) < 500:
            continue
        model = C.fit_risk_model(ret, p, family)
        if model is None:
            continue
        coverage_results[p] = mod.oos_coverage_test(ret, family, p)

    n_pass_01 = sum(
        1 for r in coverage_results.values() if r.get("0.01", {}).get("pass")
    )
    return {
        "family_map": family_map,
        "n_products_passing_1pct_coverage": n_pass_01,
        "n_products_total": len(coverage_results),
        "per_product_pass": {
            p: r.get("0.01", {}).get("pass") for p, r in coverage_results.items()
        },
    }


def main() -> None:
    t0 = time.time()
    mod = _load_phase7_module()
    policies = build_policies()

    print(
        f"P2 == P3 (identical family assignments): {policies['P2'] == policies['P3']}"
    )

    results = {}
    for name, fam_map in policies.items():
        print(f"running policy {name}...", flush=True)
        results[name] = run_policy(mod, fam_map)
        print(
            f"  {name}: {results[name]['n_products_passing_1pct_coverage']}/"
            f"{results[name]['n_products_total']} pass",
            flush=True,
        )

    pass_counts = {
        name: r["n_products_passing_1pct_coverage"] for name, r in results.items()
    }
    best_count = max(pass_counts.values())
    tied = [name for name, c in pass_counts.items() if c == best_count]
    # pre-committed rule: highest pass count wins; tie -> simplest (P2)
    winner = "P2" if "P2" in tied else min(tied)

    out = {
        "policies": results,
        "pass_counts": pass_counts,
        "tied_policies": tied,
        "winner": winner,
        "shipping_rule": "highest Gate RE pass count; tie -> P2 (pre-committed in "
        "risk_engine_preregistration.json before this ran)",
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nwinner: {winner} ({pass_counts})")
    print(f"written {OUT_PATH} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
