"""Gate NL: no lookahead (NEXT_PROMPT.md sec 8.3, sec 10).

Wires `regime.evaluation.no_lookahead_check` -- 014's causality gate, run as
a hard gate across 27 symbols at four truncations -- over the `ewma_vol` ->
`RiskModel.var_conditional` path (`risk._lookahead.check_no_lookahead`).
This is the highest-value single import from `regime`: it strengthens the
engine's no-lookahead claim beyond the bespoke unit test with the exact
harness that already proved this property for a different engine.

Threshold (sec 10): 16/16 products pass at all four truncations
(1, 5, 21, 63). Hard gate.

Writes `src/research/tmp/run_risk_05_lookahead_results.json`.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import commod_lib8 as C  # noqa: F401  (exercises the shim path)
import polars as pl

from risk._lookahead import check_no_lookahead
from risk.families import load_family_map
from risk.model import fit_risk_model

CURVE_DIR = "src/research/tmp/phase_0_curves"
OUT_PATH = "src/research/tmp/run_risk_05_lookahead_results.json"
TRUNCATIONS = (1, 5, 21, 63)


def main() -> None:
    t0 = time.time()
    family_map = load_family_map("v1")

    results: dict[str, Any] = {}
    n_pass = 0
    for product in sorted(family_map.products.keys()):
        curve = pl.read_parquet(f"{CURVE_DIR}/{product}.parquet")
        sub = curve.select(
            ["date", pl.col("log_return_ratioadj").alias("ret")]
        ).drop_nulls()
        sub = sub.filter(pl.col("ret").is_finite()).sort("date")
        ret = sub["ret"].to_numpy()
        dates = sub["date"].to_numpy()

        family = family_map.family_for(product)
        model = fit_risk_model(ret, product, family)
        if model is None:
            results[product] = {"pass": False, "note": "fit_risk_model returned None"}
            continue

        ok = check_no_lookahead(model, ret, dates, alpha=0.01, truncations=TRUNCATIONS)
        results[product] = {"pass": bool(ok), "truncations": list(TRUNCATIONS)}
        n_pass += int(ok)
        print(f"{product} ({family}): {'pass' if ok else 'FAIL'}", flush=True)

    gate_nl = {
        "n_pass": n_pass,
        "n_total": len(family_map.products),
        "fires": n_pass == len(family_map.products),
    }
    print(f"\ngate_NL: {gate_nl}")

    out = {"per_product": results, "gate_NL": gate_nl}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"written {OUT_PATH} in {time.time() - t0:.1f}s")
    if not gate_nl["fires"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
