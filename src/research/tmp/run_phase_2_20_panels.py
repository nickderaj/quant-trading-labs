"""Notebook 020, Phase 2b (NEXT_PROMPT.md sec 5.6): build and cache both
feature panels once, at both carry half-lives, and run the 018 reproduction
tripwire before anything downstream is allowed to proceed.

Do NOT proceed past a failing tripwire (sec 5.6 point 2, sec 12's first
row): if the 018 reproduction fails, the cache/panel/environment is
suspect and every downstream number would be uninterpretable.

Usage: uv run python src/research/tmp/run_phase_2_20_panels.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "research" / "tmp"))

import basis_lib18 as bl18
import basis_lib20 as bl20

import research

OUT_PATH = REPO_ROOT / "src" / "research" / "tmp" / "phase_2_20_panels.json"
SCRATCH_DIR = REPO_ROOT / "scratch" / "020"

EXPECTED_018_NET_SHARPE = 0.5766182328943011
TRIPWIRE_ABS_TOL = 1e-6


def _json_default(o: object) -> object:
    if isinstance(o, np.floating | np.integer):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _panel_schema_hash(panel) -> str:
    import hashlib

    schema_str = str(sorted(panel.schema.items(), key=lambda kv: kv[0]))
    return hashlib.sha256(schema_str.encode()).hexdigest()[:16]


def main() -> None:
    t_start = time.monotonic()
    result: dict[str, Any] = {}

    print("== Loading raw Binance basis18/dev panel (unchanged from 018) ==")
    raw_panel, manifest = bl18.load_basis_panel()
    n_ok = sum(1 for v in manifest.values() if v == "ok")
    print(f"Universe: {n_ok}/{len(manifest)} symbols ok")

    binance_panels = {}
    for hl in (bl20.CARRY_EWMA_HALF_LIFE, bl20.SLOW_CARRY_HALF_LIFE):
        featured = bl20.add_trade_features_v2(raw_panel, half_life=hl)
        path = SCRATCH_DIR / f"panel_dev_binance_hl{hl}.parquet"
        featured.write_parquet(path)
        binance_panels[hl] = featured
        print(f"cached {path} ({len(featured)} rows)")

    result["binance_panels"] = {
        str(hl): {
            "path": str(SCRATCH_DIR / f"panel_dev_binance_hl{hl}.parquet"),
            "n_rows": len(df),
            "schema_hash": _panel_schema_hash(df),
        }
        for hl, df in binance_panels.items()
    }
    result["universe_manifest_summary"] = {
        "total_seed_symbols": len(manifest),
        "n_ok": n_ok,
    }

    # -----------------------------------------------------------------
    # sec 5.6 point 2: the 018 reproduction tripwire. STOP if this fails.
    # -----------------------------------------------------------------
    print("== 018 reproduction tripwire ==")
    hl21_panel = binance_panels[bl20.CARRY_EWMA_HALF_LIFE]
    annualized_rate = research.sharpe_to_annualized_rate("8h")

    timed_w_018 = bl18.build_book_weights(hl21_panel, timed=True)
    timed_tf_018 = bl18.book_trade_frame(hl21_panel, timed_w_018, origin_offset=0)
    timed_costed_018 = bl18.apply_two_leg_costs(timed_tf_018)
    timed_metrics_018 = bl18.book_metrics(timed_costed_018, annualized_rate, "timed")
    reproduced_sharpe = float(timed_metrics_018["sharpe_net"])

    tripwire_passed = (
        abs(reproduced_sharpe - EXPECTED_018_NET_SHARPE) < TRIPWIRE_ABS_TOL
    )
    result["reproduction_tripwire"] = {
        "expected_net_sharpe": EXPECTED_018_NET_SHARPE,
        "reproduced_net_sharpe": reproduced_sharpe,
        "abs_diff": abs(reproduced_sharpe - EXPECTED_018_NET_SHARPE),
        "tol": TRIPWIRE_ABS_TOL,
        "passed": tripwire_passed,
    }
    print(
        f"reproduced net Sharpe: {reproduced_sharpe} (expected {EXPECTED_018_NET_SHARPE})"
    )

    if not tripwire_passed:
        result["STOPPED"] = (
            "018 reproduction tripwire FAILED -- stopping per sec 5.6/sec 12. "
            "Do not proceed to Phase 3. Investigate the cache/panel/environment."
        )
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, indent=2, default=_json_default)
        print("TRIPWIRE FAILED -- see", OUT_PATH)
        sys.exit(1)

    # -----------------------------------------------------------------
    # sec 5.6 point 3: build_book_weights_v2(n_min=1, 018 defaults) must be
    # element-wise identical to bl18.build_book_weights, on the REAL panel.
    # -----------------------------------------------------------------
    print("== n_min=1 identity check on the real panel ==")
    v2_w = bl20.build_book_weights_v2(hl21_panel, timed=True, n_min=1)
    v2_sorted = v2_w.sort(["datetime", "symbol"])
    v18_sorted = timed_w_018.sort(["datetime", "symbol"])
    identical = bool(
        v2_sorted["symbol"].to_list() == v18_sorted["symbol"].to_list()
        and np.allclose(
            v2_sorted["weight"].to_numpy(),
            v18_sorted["weight"].to_numpy(),
            atol=0,
            rtol=0,
        )
    )
    result["n_min_1_identity_check_real_panel"] = {"identical": identical}
    print(f"n_min=1 identical to 018 on real panel: {identical}")
    if not identical:
        result["STOPPED"] = (
            "build_book_weights_v2(n_min=1) does NOT reproduce bl18.build_book_weights "
            "on the real panel -- stopping per sec 5.6. The unit test passed on synthetic "
            "data but the real panel disagrees; investigate before Phase 3."
        )
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, indent=2, default=_json_default)
        print("IDENTITY CHECK FAILED -- see", OUT_PATH)
        sys.exit(1)

    # -----------------------------------------------------------------
    # sec 5.6 point 4: time one full book build, bars/sec for wall-time
    # prediction (sec 3 rule 7).
    # -----------------------------------------------------------------
    print("== Timing one book build ==")
    t0 = time.monotonic()
    _ = bl20.build_book_weights_v2(hl21_panel, timed=True, n_min=bl20.N_MIN)
    elapsed = time.monotonic() - t0
    n_bars = hl21_panel["datetime"].n_unique()
    bars_per_second = n_bars / elapsed if elapsed > 0 else float("inf")
    result["book_build_timing"] = {
        "n_bars": n_bars,
        "elapsed_s": elapsed,
        "bars_per_second": bars_per_second,
    }
    print(f"book build: {n_bars} bars in {elapsed:.2f}s ({bars_per_second:.1f} bars/s)")

    # -----------------------------------------------------------------
    # Cross-venue panel, if Bybit data is available.
    # -----------------------------------------------------------------
    print("== Cross-venue panel ==")
    try:
        xvenue_raw, xvenue_manifest = bl20.load_xvenue_panel()
        n_xv_ok = sum(1 for v in xvenue_manifest.values() if v == "ok")
        print(f"xvenue universe: {n_xv_ok}/{len(xvenue_manifest)} symbols ok")
        xvenue_panels = {}
        for hl in (bl20.CARRY_EWMA_HALF_LIFE, bl20.SLOW_CARRY_HALF_LIFE):
            xv_featured = bl20.add_xvenue_trade_features(xvenue_raw, half_life=hl)
            path = SCRATCH_DIR / f"panel_dev_xvenue_hl{hl}.parquet"
            xv_featured.write_parquet(path)
            xvenue_panels[hl] = xv_featured
            print(f"cached {path} ({len(xv_featured)} rows)")
        result["xvenue_panels"] = {
            str(hl): {
                "path": str(SCRATCH_DIR / f"panel_dev_xvenue_hl{hl}.parquet"),
                "n_rows": len(df),
                "schema_hash": _panel_schema_hash(df),
            }
            for hl, df in xvenue_panels.items()
        }
        result["xvenue_universe_manifest_summary"] = {
            "total_symbols": len(xvenue_manifest),
            "n_ok": n_xv_ok,
            "manifest": xvenue_manifest,
        }
        result["mechanism_b_available"] = n_xv_ok >= 20
        if n_xv_ok < 20:
            result["mechanism_b_blocker"] = (
                f"only {n_xv_ok} symbols survived cross-venue assembly, below the "
                "sec 4.6 floor of 20 -- Mechanism B is data-limited (reported, not stretched)."
            )
    except Exception as exc:  # noqa: BLE001 -- a genuine data blocker is a reportable outcome, sec 4.6
        result["mechanism_b_available"] = False
        result["mechanism_b_blocker"] = f"load_xvenue_panel failed: {exc}"
        print(f"Mechanism B unavailable: {exc}")

    result["elapsed_total_s"] = time.monotonic() - t_start
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
