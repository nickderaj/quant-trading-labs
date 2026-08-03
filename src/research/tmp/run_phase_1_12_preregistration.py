"""Notebook 12 Phase 1 -- Gate VB pre-registration, written before any
pooled backtest runs and never edited afterward (same discipline as
`phase_6_11a_results.json` for the 11-series notebooks). Phase 2 asserts
its own `n_trials` and criteria against this file programmatically rather
than re-typing them.
"""

import json
from pathlib import Path

GATE_VB = {
    "notebook": "012",
    "claim": (
        "a volume-confirmed bull-flag breakout (11d's prior-run + tightening-"
        "base + breakout rule, made symmetric long/short and gated on "
        "breakout-bar volume) earns its keep over the identical rule with "
        "the volume condition switched off, on a pooled ~85-instrument "
        "basket across crypto/equity-ETF/futures, at a small honestly-"
        "counted DSR trial count"
    ),
    "fires_if": (
        "net Sharpe > 0 at every origin offset (0/7/14/21) on the volume-"
        "gated 1x-cost book AND paired block-bootstrap 95% CI on "
        "(volume-gated minus ungated) offset-0 1x-cost returns excludes "
        "zero AND DSR > 0.95 at n_trials=12 (4 offsets x 3 cost "
        "multipliers, single fixed rule and single fixed k -- no other "
        "parameter swept) AND cost stress at 3x shows a real, correctly-"
        "signed degradation AND the three-way risk gate / fail-closed "
        "regime gate carry over unchanged from 11a/11d"
    ),
    "n_trials": 12,
    "origin_offsets": [0, 7, 14, 21],
    "cost_multipliers": [1, 2, 3],
    "one_rule_one_k": True,
    "control": (
        "the byte-identical breakout rule (same signals, stops, exits, "
        "costs, bars, regime gate) with the volume condition switched off "
        "-- not buy-and-hold, not a random-entry null"
    ),
    "fundable_flag_if": (
        "Gate VB fires AND net Sharpe > 0.5 at every offset AND max "
        "drawdown (offset 0, 1x cost, volume-gated book) <= 25% of peak"
    ),
    "holdout": "2025-01-01 to 2026-07-28, untouched and unspent",
    "report_per_asset_class_trade_counts": True,
    "second_pattern_declared": False,
}


def main() -> None:
    out = {"gates": {"VB": GATE_VB}}
    path = Path("src/research/tmp/phase_1_12_preregistration.json")
    if path.exists():
        existing = json.loads(path.read_text())
        assert existing == out, (
            "phase_1_12_preregistration.json already exists and differs -- "
            "pre-registration must not be edited after the fact"
        )
        print("Phase 1 12: pre-registration already committed, unchanged.")
        return
    path.write_text(json.dumps(out, indent=2))
    print("Phase 1 12: Gate VB pre-registration committed.")


if __name__ == "__main__":
    main()
