"""Builds `src/risk/configs/family_map_crypto_v1.json`: a family map for the
frozen 6-symbol crypto panel (BTCUSDT, ETHUSDT, SOLUSDT, DOGEUSDT, BNBUSDT,
XRPUSDT -- `src/research/tmp/merge_phase4_violation.py:16`) at **daily**
frequency, sourced from each symbol's own `phase3_zoo_{SYMBOL}.json`
`intervals["1d"]` block -- the same OOS log-score density contest
(`garch_{normal,t,ged,nig,johnsonsu,hansen_skewt}`) that `build_family_map_v1.py`
used for the 16 commodity/equity-index futures, just read from the crypto
notebooks' own output instead of 008's.

**This is explicitly NOT the same kind of artifact as `family_map_v1.json`.**
`family_map_v1` is downstream of a full gated pipeline (Phase 2 family
selection -> Phase 3 OOS ranking -> Phase 7 walk-forward VaR-coverage
certification, Gate RE/CE/CT all passing at daily frequency, 16/16 or
14-15/16). No Phase-7-equivalent walk-forward coverage battery has ever been
run for this crypto panel *at daily frequency* -- notebooks 004/005 ran the
coverage battery on BTC only at 1h/4h/12h (clearing cleanly at 12h only), and
the 5-symbol daily transfer in 005 cleared 6/10 models for SOL/DOGE/BNB,
2/10 for XRP, and never for BTC or ETH. This file therefore records only a
density-family *pick*, honestly labelled `"gate_validated": false` and
`"policy": "log_score_pick_ungated"`, not a certified family map. It exists
so `risk.serve` has *some* explicit, provenance-carrying family per crypto
symbol to fit against -- per `risk.families`'s own rule (`fit_new_product`),
a family must never be silently guessed.

Selection rule: `best_by_log_score` (highest mean OOS log-score among the
six densities), matching 008 Phase 3's own convention of shipping the raw
score winner regardless of whether it beats `garch_t` *significantly*
(008 shipped all 16 winners even though only 2/16 cleared BH significance);
`best_zoo_family` is recorded alongside for reference but not used to select,
since it occasionally diverges from the log-score winner (e.g. SOLUSDT 1d)
without a documented reason to prefer it.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")

from risk.families import config_hash

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "BNBUSDT", "XRPUSDT"]
ZOO_PATH = "src/research/tmp/phase3_zoo_{symbol}.json"
OUT_PATH = "src/risk/configs/family_map_crypto_v1.json"
INTERVAL = "1d"

_FAMILY_COLLAPSE = {
    "garch_normal": "normal",
    "garch_t": "t",
    "garch_ged": "ged",
    "garch_nig": "nig",
    "garch_johnsonsu": "johnsonsu",
    "garch_hansen_skewt": "hansen_skewt",
}


def main() -> None:
    products: dict[str, dict] = {}
    for symbol in SYMBOLS:
        with open(ZOO_PATH.format(symbol=symbol)) as f:
            zoo = json.load(f)
        iv = zoo["intervals"][INTERVAL]
        winner_model = iv["best_by_log_score"]
        family = _FAMILY_COLLAPSE[winner_model]
        gate_p = iv.get("gate_p_per_family", {}).get(winner_model, {})
        products[symbol] = {
            "family": family,
            "selected_by": "phase3_zoo (OOS log-score contest, 1d interval)",
            "best_model": winner_model,
            "log_score": iv["scores"][winner_model]["log_score_mean"],
            "n_obs": iv["n_obs"],
            "best_zoo_family_agrees": iv.get("best_zoo_family") == winner_model,
            "significantly_beats_garch_t": gate_p.get("significantly_beats_garch_t"),
        }

    payload = {
        "version": "crypto_v1",
        "created": "2026-08-13",
        "source_notebook": "phase3_zoo (crypto density contest, notebooks 004-006 lineage)",
        "policy": "log_score_pick_ungated",
        "gate_validated": False,
        "validation_note": (
            "Density family chosen by OOS log-score only (same procedure as "
            "family_map_v1's Phase 3 step). No walk-forward VaR-coverage gate "
            "(family_map_v1's Phase 7 equivalent) has been run for this panel "
            "at daily frequency -- treat the live calibration monitor's output "
            "for these symbols as a first read, not a certified result."
        ),
        "products": products,
    }
    payload["config_hash"] = config_hash(payload)

    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"written {OUT_PATH} ({len(products)} symbols)")
    for symbol, entry in products.items():
        print(f"  {symbol}: {entry['family']} (log_score={entry['log_score']:.4f})")


if __name__ == "__main__":
    main()
