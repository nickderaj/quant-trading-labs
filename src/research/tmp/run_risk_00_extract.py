"""Phase 0: the move + import-shim verification (NEXT_PROMPT.md sec 3, sec
11).

Confirms, mechanically, the two claims Phase 0's extraction rests on:

1. `src/risk/` imports cleanly with no dependency on `research/tmp/` scratch
   (sec 12's "risk/ importing anything from commod_lib8.py is not [fine]" --
   verified here as well as in `tests/test_risk_import_boundary.py`, which
   runs in CI on every push; this script is the one-off, human-readable
   companion check).
2. `commod_lib8.py`/`dist_lib5.py`/`densities/`'s re-export shims resolve to
   the *exact same objects* now living in `src/risk/` -- not merely
   behaviourally-similar re-implementations. Object identity is a stronger
   claim than "produces the same numbers" (which gate PR/PH separately
   prove for the numeric surface) and is what makes "the old notebook runs
   the new code" (sec 3.3) literally true rather than a paraphrase.

Writes `src/research/tmp/run_risk_00_extract_results.json`.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from typing import Any

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

OUT_PATH = "src/research/tmp/run_risk_00_extract_results.json"

SHIMMED_IDENTITY_CHECKS = [
    # (module attribute path in the shim, module attribute path in src/risk/)
    ("commod_lib8.RiskModel", "risk.model.RiskModel"),
    ("commod_lib8.ewma_vol", "risk.model.ewma_vol"),
    ("commod_lib8.fit_risk_model", "risk.model.fit_risk_model"),
    ("commod_lib8.portfolio_risk", "risk.portfolio.portfolio_risk"),
    (
        "commod_lib8.empirical_lower_tail_dependence",
        "risk.portfolio.empirical_lower_tail_dependence",
    ),
    ("commod_lib8.to_pseudo_uniform", "risk.portfolio.to_pseudo_uniform"),
    ("commod_lib8.kupiec_by_state", "risk.calibration.kupiec_by_state"),
    ("commod_lib8.flag_contaminated_rows", "risk.hygiene.flag_contaminated_rows"),
    ("commod_lib8.apply_hygiene_filter", "risk.hygiene.apply_hygiene_filter"),
    ("commod_lib8.liquidity_screen", "risk.hygiene.liquidity_screen"),
    ("commod_lib8.build_roll_schedule", "risk.hygiene.build_roll_schedule"),
    ("commod_lib8.liquid_contract_months", "risk.hygiene.liquid_contract_months"),
    ("commod_lib8.build_continuous_series", "risk.hygiene.build_continuous_series"),
    (
        "commod_lib8.build_continuous_series_ohlcv",
        "risk.hygiene.build_continuous_series_ohlcv",
    ),
    ("dist_lib5.acerbi_szekely_z", "risk.calibration.acerbi_szekely_z"),
    (
        "dist_lib5.acerbi_szekely_bootstrap_pvalue",
        "risk.calibration.acerbi_szekely_bootstrap_pvalue",
    ),
    ("densities.REGISTRY", "risk.densities.REGISTRY"),
    ("densities.ged", "risk.densities.ged"),
    ("densities.nig", "risk.densities.nig"),
    ("densities.johnsonsu", "risk.densities.johnsonsu"),
    ("densities.hansen_skewt", "risk.densities.hansen_skewt"),
]


def _resolve(path: str) -> Any:
    """Resolve a dotted `module.submodule.attr` path via
    `importlib.import_module` on the module prefix, not attribute-chasing
    from the top-level package object. `src/risk/__init__.py`'s public API
    (sec 8.1) deliberately names a function `portfolio()`, which shadows the
    `risk.portfolio` *submodule* as a `risk`-package attribute -- correct
    per spec (that name should resolve to the callable), but it means
    `import risk; risk.portfolio` no longer reaches the submodule the way a
    naive attribute walk would assume."""
    module_path, _, attr = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def check_shim_identity() -> dict[str, Any]:
    import commod_lib8  # noqa: F401
    import densities  # noqa: F401
    import dist_lib5  # noqa: F401

    results = {}
    n_identical = 0
    for shim_path, real_path in SHIMMED_IDENTITY_CHECKS:
        shim_obj = _resolve(shim_path)
        real_obj = _resolve(real_path)
        is_identical = shim_obj is real_obj
        results[shim_path] = {"target": real_path, "identical": is_identical}
        n_identical += int(is_identical)

    return {
        "n_checked": len(SHIMMED_IDENTITY_CHECKS),
        "n_identical": n_identical,
        "all_identical": n_identical == len(SHIMMED_IDENTITY_CHECKS),
        "per_name": results,
    }


def check_no_reverse_dependency() -> dict[str, Any]:
    """Runs in a fresh subprocess with only src/ on PYTHONPATH (no
    research/tmp/) -- if risk/ secretly depended on commod_lib8.py or
    dist_lib5.py, import would fail here."""
    probe = (
        "import sys; import risk, risk.hygiene, risk.model, risk.portfolio, "
        "risk.calibration, risk.families, risk.densities; "
        "leaked = [m for m in sys.modules if m in "
        "('commod_lib8', 'dist_lib5', 'densities')]; "
        "print(leaked); sys.exit(1 if leaked else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env={"PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "returncode": result.returncode,
        "clean_import": result.returncode == 0,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip()[-2000:],
    }


def main() -> None:
    identity = check_shim_identity()
    print(
        f"shim identity: {identity['n_identical']}/{identity['n_checked']}",
        flush=True,
    )
    for name, r in identity["per_name"].items():
        if not r["identical"]:
            print(f"  NOT IDENTICAL: {name} -> {r['target']}", flush=True)

    boundary = check_no_reverse_dependency()
    print(f"no reverse dependency: {boundary['clean_import']}", flush=True)
    if not boundary["clean_import"]:
        print(boundary["stderr"], flush=True)

    fires = identity["all_identical"] and boundary["clean_import"]
    out = {
        "shim_identity": identity,
        "no_reverse_dependency": boundary,
        "fires": fires,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nfires: {fires}")
    print(f"written {OUT_PATH}")
    if not fires:
        sys.exit(1)


if __name__ == "__main__":
    main()
