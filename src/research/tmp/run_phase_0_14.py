"""Notebook 014 Phase 0: ground truth, port-fidelity check, no_lookahead_check
hard gate, pre-registration (NEXT_PROMPT.md sec6 Phase 0).

Writes phase_0_14_results.json (data introspection + fidelity + gate
results) and phase_0_14_preregistration.json (frozen before Phase 1 ever
runs, never edited afterward).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import pandas as pd
import polars as pl
import yaml

from regime.align import align_frame_to_daily
from regime.engine import RegimeEngine, RegimeInputs
from regime.evaluation import no_lookahead_check
from regime.loaders import (
    COT_PATH,
    COT_SYMBOL,
    CURVE_SYMBOLS,
    FRED_DIR,
    FRED_SERIES,
    load_bars,
    load_cot_raw,
    load_curve,
    load_fred_frame,
    net_positioning,
)
from regime.universe import load_regime_universe

TMP = "src/research/tmp"
ULTRON_FINANCE = "/home/nick/Documents/ultron/libs/finance"
TRUNCATIONS = (1, 5, 21, 63)


# --------------------------------------------------------------------------- #
# Data introspection (NEXT_PROMPT.md sec3)
# --------------------------------------------------------------------------- #
def introspect_data(universe) -> dict:
    bars: dict[str, dict] = {}
    for symbol in sorted(universe.symbols):
        df = load_bars(symbol)
        bars[symbol] = {
            "rows": len(df),
            "first": df.index.min().date().isoformat(),
            "last": df.index.max().date().isoformat(),
        }

    fred: dict[str, dict] = {}
    for series in FRED_SERIES:
        path = FRED_DIR / f"{series}.parquet"
        fred_df = pl.read_parquet(path)
        fred[series] = {
            "rows": fred_df.height,
            "first": str(fred_df["date"].min()),
            "last": str(fred_df["date"].max()),
        }

    cot_raw = load_cot_raw()
    cot = {
        "path": str(COT_PATH),
        "market": "CRUDE OIL, LIGHT SWEET (067651)",
        "rows": len(cot_raw),
        "first_report_date": cot_raw.index.min().date().isoformat(),
        "last_report_date": cot_raw.index.max().date().isoformat(),
    }

    curves: dict[str, dict] = {}
    for symbol, stem in CURVE_SYMBOLS.items():
        curve_df = load_curve(symbol)
        assert curve_df is not None
        curves[symbol] = {
            "stem": stem,
            "rows": len(curve_df),
            "first": curve_df.index.min().date().isoformat(),
            "last": curve_df.index.max().date().isoformat(),
            "columns": list(curve_df.columns),
        }

    gaps = [
        (
            "COT: only 067651 (CRUDE OIL, LIGHT SWEET) present; macro sector's "
            "cot_market ('E-MINI S&P 500') has no series in this repo -- macro "
            "sector cot input is always None (see regime/builder.py docstring)."
        ),
        (
            "Curves: only 5 of 20 Commodities-basket symbols have a curve file "
            "(CL, NG, GC, SI, HG); the other 15 legs are permanently curve=None, "
            "so term_structure/carry for the Commodities basket are always a "
            "5-symbol aggregate, never 20."
        ),
        (
            "Curves start 2018-01-02 while bars start as early as 2000 for some "
            "symbols; term_structure/carry are NaN for all curve symbols before "
            "their curve's start date."
        ),
        (
            "curve_slope is called with far='close_f3' (regime/dimensions/"
            "term_structure.py), not production's far='close_f12' default -- "
            "this repo's curves have only 3 legs, so term_structure/carry are "
            "scored on a 3-month slope, not a 12-month slope."
        ),
    ]

    return {
        "bars": bars,
        "fred": fred,
        "cot": cot,
        "curves": curves,
        "disclosed_gaps": gaps,
    }


# --------------------------------------------------------------------------- #
# Port-fidelity check (NEXT_PROMPT.md sec6 Phase 0, bullet 2)
# --------------------------------------------------------------------------- #
def port_fidelity_check() -> dict:
    result: dict = {"method": None, "config_hash_equal": {}, "end_to_end_equal": None}

    # (a) config_hash equality on all four YAMLs, source vs port.
    from regime.config import RegimeConfig as PortConfig

    port_hashes = {}
    for name in ("macro_default", "commodity_default", "fx_default", "equity_default"):
        cfg = PortConfig.from_yaml(Path("src/regime/configs") / f"{name}.yaml")
        port_hashes[name] = cfg.config_hash()

    source_hash_script = (
        "from ultron_finance.regime.config import RegimeConfig\n"
        "from pathlib import Path\n"
        "import json\n"
        "out = {}\n"
        "for name in ['macro_default','commodity_default','fx_default','equity_default']:\n"
        "    cfg = RegimeConfig.from_yaml(Path('src/ultron_finance/regime/configs') / f'{name}.yaml')\n"
        "    out[name] = cfg.config_hash()\n"
        "print(json.dumps(out))\n"
    )
    try:
        proc = subprocess.run(
            [
                "uv",
                "run",
                "--project",
                ULTRON_FINANCE,
                "python3",
                "-c",
                source_hash_script,
            ],
            cwd=ULTRON_FINANCE,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        source_hashes = json.loads(proc.stdout.strip().splitlines()[-1])
        for name, port_hash in port_hashes.items():
            result["config_hash_equal"][name] = source_hashes[name] == port_hash
        source_reachable = True
    except Exception as exc:  # noqa: BLE001 - fall back to hash-only if source venv unreachable
        result["source_unreachable_reason"] = str(exc)
        source_reachable = False

    if not source_reachable:
        result["method"] = "config_hash_only (source venv unreachable)"
        result["end_to_end_equal"] = None
        return result

    # (b) end-to-end run on synthetic fixtures, source vs port, bit-for-bit.
    source_out = f"{TMP}/_fidelity_source.json"
    port_out = f"{TMP}/_fidelity_port.json"
    subprocess.run(
        [
            "uv",
            "run",
            "--project",
            ULTRON_FINANCE,
            "python3",
            f"{TMP}/fidelity_source_14.py",
            source_out,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    subprocess.run(
        ["uv", "run", "python3", f"{TMP}/fidelity_port_14.py", port_out],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    with open(source_out) as f:
        source_payload = json.load(f)
    with open(port_out) as f:
        port_payload = json.load(f)

    equal = {}
    for name in ("macro_default", "commodity_default"):
        equal[name] = {
            "scores_equal": source_payload[name]["scores"]
            == port_payload[name]["scores"],
            "labels_equal": source_payload[name]["labels"]
            == port_payload[name]["labels"],
        }
    result["method"] = (
        "config_hash + end_to_end synthetic-fixture run, source venv reachable"
    )
    result["end_to_end_equal"] = equal
    return result


# --------------------------------------------------------------------------- #
# no_lookahead_check hard gate (NEXT_PROMPT.md sec6 Phase 0, bullet 3)
# --------------------------------------------------------------------------- #
def run_no_lookahead_gate(universe) -> dict:
    per_symbol: dict[str, dict] = {}
    macro_frame = load_fred_frame()

    symbol_configs: list[tuple[str, str]] = [
        (universe.macro.index_symbol, universe.macro.config)
    ]
    seen = {universe.macro.index_symbol}
    for basket in universe.baskets:
        for symbol in basket.symbols:
            if symbol not in seen:
                symbol_configs.append((symbol, basket.config))
                seen.add(symbol)

    all_passed = True
    for symbol, config_name in symbol_configs:
        bars = load_bars(symbol)
        curve = load_curve(symbol)
        if curve is not None:
            curve = align_frame_to_daily(pd.DatetimeIndex(bars.index), curve)
        macro = None
        cot = None
        if symbol == universe.macro.index_symbol:
            macro = align_frame_to_daily(pd.DatetimeIndex(bars.index), macro_frame)
        engine = RegimeEngine.from_default(config_name)
        inputs = RegimeInputs(ohlcv=bars, curve=curve, macro=macro, cot=cot)
        valid_truncations = [t for t in TRUNCATIONS if t < len(bars)]
        start = time.time()
        passed = no_lookahead_check(
            engine, inputs, truncations=tuple(valid_truncations)
        )
        elapsed = time.time() - start
        per_symbol[symbol] = {
            "config": config_name,
            "truncations_tested": valid_truncations,
            "passed": passed,
            "elapsed_sec": round(elapsed, 2),
        }
        all_passed = all_passed and passed

    return {"all_passed": all_passed, "per_symbol": per_symbol}


def run_no_lookahead_oil_products_cot(universe) -> dict:
    """Separately gate the opt-in oil_products COT wiring path (CL=F only)."""
    bars = load_bars(COT_SYMBOL)
    curve = load_curve(COT_SYMBOL)
    assert curve is not None
    curve = align_frame_to_daily(pd.DatetimeIndex(bars.index), curve)
    raw = net_positioning(load_cot_raw())
    cot = align_frame_to_daily(
        pd.DatetimeIndex(bars.index), raw[["noncomm_net_pct_oi"]], lags=3
    )
    engine = RegimeEngine.from_default("commodity_default")
    inputs = RegimeInputs(ohlcv=bars, curve=curve, cot=cot)
    valid_truncations = [t for t in TRUNCATIONS if t < len(bars)]
    passed = no_lookahead_check(engine, inputs, truncations=tuple(valid_truncations))
    return {
        "symbol": COT_SYMBOL,
        "truncations_tested": valid_truncations,
        "passed": passed,
    }


# --------------------------------------------------------------------------- #
# Pre-registration (episode table + gates + n_trials, frozen before Phase 1)
# --------------------------------------------------------------------------- #
EPISODE_TABLE = [
    {
        "episode": "GFC",
        "start": "2008-09-01",
        "end": "2009-03-01",
        "sector": "Macro",
        "dimensions": ["risk", "credit"],
        "expected_labels": {"risk": "risk_off", "credit": "wide"},
    },
    {
        "episode": "Euro crisis",
        "start": "2011-08-01",
        "end": "2011-12-01",
        "sector": "Macro",
        "dimensions": ["risk"],
        "expected_labels": {"risk": "risk_off"},
    },
    {
        "episode": "Taper tantrum",
        "start": "2013-05-01",
        "end": "2013-09-01",
        "sector": "Macro",
        "dimensions": ["yield_curve"],
        "expected_labels": {"yield_curve": "steep"},
    },
    {
        "episode": "Oil glut",
        "start": "2014-11-01",
        "end": "2016-02-01",
        "sector": "Commodities / oil products",
        "dimensions": ["trend", "term_structure"],
        "expected_labels": {"trend": "bear", "term_structure": "contango"},
    },
    {
        "episode": "COVID crash",
        "start": "2020-02-01",
        "end": "2020-04-01",
        "sector": "Macro / all commodity+FX baskets",
        "dimensions": ["risk", "volatility"],
        "expected_labels": {"risk": "risk_off", "volatility": "extreme"},
    },
    {
        "episode": "Post-COVID commodity bull",
        "start": "2020-11-01",
        "end": "2022-06-01",
        "sector": "Commodities",
        "dimensions": ["trend"],
        "expected_labels": {"trend": "bull"},
    },
    {
        "episode": "Energy backwardation",
        "start": "2021-06-01",
        "end": "2022-08-01",
        "sector": "oil products",
        "dimensions": ["term_structure", "carry"],
        "expected_labels": {"term_structure": "backwardation", "carry": "positive"},
    },
    {
        "episode": "Hiking cycle / inversion",
        "start": "2022-07-01",
        "end": "2023-10-01",
        "sector": "Macro",
        "dimensions": ["yield_curve", "risk"],
        "expected_labels": {"yield_curve": "inverted", "risk": "risk_off"},
    },
]

GATES = {
    "NL": {
        "claim": "no_lookahead_check passes at every truncation for every sector",
        "hard_gate": True,
    },
    "RA": {
        "claim": "Balanced accuracy on the episode table significantly beats the "
        "class-prior baseline",
    },
    "RM": {
        "claim": "Balanced accuracy on mechanical labels significantly beats "
        "persistence and Markov baselines",
    },
    "RL": {"claim": "Median label lag at episode onset <= 21 trading days"},
    "RS": {
        "claim": "No scored dimension exceeds 90% single-label occupancy or flips "
        "more than once per 10 bars",
    },
    "RC": {
        "claim": "Port fidelity: ported engine matches source scores where checkable"
    },
}


def build_preregistration(data_intro: dict, fidelity: dict, no_lookahead: dict) -> dict:
    return {
        "notebook": "014_market_regime_engine_and_accuracy",
        "committed_before_first_backtest": True,
        "scope": {
            "authorizes_trading": False,
            "spends_holdout": False,
            "holdouts_untouched": {
                "crypto": "2025-07-01",
                "futures": "2025-01-01 to 2026-07-28",
            },
            "note": "This notebook scores label accuracy against independent ground "
            "truth. It is not a 23rd alpha gate; no strategy is built and no "
            "holdout is spent here.",
        },
        "ground_truth_phase0_findings": data_intro,
        "port_fidelity": fidelity,
        "no_lookahead_gate": no_lookahead,
        "episode_table": EPISODE_TABLE,
        "mechanical_labels": [
            "forward 21-day realized vol terciles vs volatility dimension",
            "sign of forward 63-day return vs trend dimension",
            "sign of T10Y2Y vs yield_curve dimension",
            "sign of f1-f2 spread vs term_structure dimension",
        ],
        "gates": GATES,
        "curve_slope_substitution": "far='close_f3' not far='close_f12' (see "
        "ground_truth_phase0_findings.disclosed_gaps)",
        "cot_bug_preservation": "macro.cot_noncomm requires={'macro'} but reads "
        "inputs.cot is preserved bug-for-bug; macro sector cot is always None "
        "in this repo (no E-MINI S&P 500 COT series available)",
    }


def main() -> None:
    with open("src/regime/configs/universe.yaml") as f:
        raw = yaml.safe_load(f)
    universe = load_regime_universe(raw)

    print("Introspecting data sources...")
    data_intro = introspect_data(universe)

    print("Running port-fidelity check...")
    fidelity = port_fidelity_check()

    print(
        "Running no_lookahead_check hard gate over all",
        len(universe.symbols),
        "symbols...",
    )
    no_lookahead = run_no_lookahead_gate(universe)
    no_lookahead["oil_products_cot_opt_in"] = run_no_lookahead_oil_products_cot(
        universe
    )

    results = {
        "data_introspection": data_intro,
        "port_fidelity": fidelity,
        "no_lookahead_gate": no_lookahead,
    }
    with open(f"{TMP}/phase_0_14_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    prereg = build_preregistration(data_intro, fidelity, no_lookahead)
    with open(f"{TMP}/phase_0_14_preregistration.json", "w") as f:
        json.dump(prereg, f, indent=2, default=str)

    print("NL gate all_passed:", no_lookahead["all_passed"])
    print(
        "oil_products COT opt-in NL gate passed:",
        no_lookahead["oil_products_cot_opt_in"]["passed"],
    )
    print("Config hash equality:", fidelity.get("config_hash_equal"))
    print("Wrote phase_0_14_results.json and phase_0_14_preregistration.json")

    if not no_lookahead["all_passed"]:
        raise SystemExit(
            "HARD GATE FAILURE: no_lookahead_check failed for at least one symbol"
        )


if __name__ == "__main__":
    main()
