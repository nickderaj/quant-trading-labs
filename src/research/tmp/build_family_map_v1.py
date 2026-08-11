"""Builds `src/risk/configs/family_map_v1.json` (NEXT_PROMPT.md sec 5.1): the
Phase 7 family map (`phase_7_results.json -> family_map`), as a versioned
artifact carrying per product the selected family, the Phase 3 OOS log-score
that selected it, the runner-up family and its score, a BH-significance flag
for the winner (008 Gate CD's own criterion -- see note below for why this is
a flag, not a raw p-value), the fitting window, and `selected_by` provenance.

**Runner-up note.** Phase 3's `ranking` list orders 13 *models*
({garch,gjr} x 6 density families, plus one spliced-EVT), not 6 families --
the top two ranked models are frequently the same density family under the
two different variance processes (e.g. CL: `garch_ged` then `gjr_ged`). The
"runner-up family" recorded here is the first *distinct* density family
below the winner, skipping `spliced_evt` (not a standalone RiskModel family,
per `fit_risk_model`).

**BH-significance note.** `phase_3_results.json` stores a per-product
`best_wins_significantly_bh` boolean (008 Gate CD's own BH-adjusted
significance test of the winner against the field) but not the raw p-value
of that test as a separate field. This artifact records the boolean rather
than fabricating a p-value that was never itself persisted. 008's own
write-up (sec Phase 3) reports the qualitative result this boolean encodes:
only 2/16 products (GC, SI) clear BH-significance, and both are won by the
same family (`ged`).
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")

from risk.families import config_hash

PHASE3_PATH = "src/research/tmp/phase_3_results.json"
PHASE7_PATH = "src/research/tmp/phase_7_results.json"
OUT_PATH = "src/risk/configs/family_map_v1.json"

DEV_START = {"__default__": "2010-06-06", "ES": "2018-01-01", "KE": "2013-12-16"}
DEV_END = "2024-12-31"

_PHASE3_FAMILY_MAP = {
    "normal": "normal",
    "t": "t",
    "ged": "ged",
    "nig": "nig",
    "johnsonsu": "johnsonsu",
    "hansen_skewt": "hansen_skewt",
}


def _collapse(model_name: str) -> str | None:
    suffix = model_name.split("_", 1)[1] if "_" in model_name else model_name
    return _PHASE3_FAMILY_MAP.get(suffix)


def _runner_up(
    ranking: list[str], winning_family: str
) -> tuple[str | None, str | None]:
    for model_name in ranking:
        fam = _collapse(model_name)
        if fam is None or fam == "spliced_evt" or fam == winning_family:
            continue
        return model_name, fam
    return None, None


def main() -> None:
    with open(PHASE7_PATH) as f:
        phase7 = json.load(f)
    family_map = phase7["family_map"]

    with open(PHASE3_PATH) as f:
        phase3 = json.load(f)

    products: dict[str, dict] = {}
    for product, family in sorted(family_map.items()):
        p3 = phase3.get(product)
        entry: dict = {
            "family": family,
            "selected_by": "008 Phase 3",
            "fitting_window": {
                "start": DEV_START.get(product, DEV_START["__default__"]),
                "end": DEV_END,
            },
        }
        if p3 is not None:
            entry["n_obs"] = p3.get("n_obs")
            entry["best_model"] = p3.get("best_model")
            entry["log_score"] = p3.get("mean_log_score", {}).get(p3.get("best_model"))
            entry["best_wins_significantly_bh"] = p3.get("best_wins_significantly_bh")
            runner_up_model, runner_up_family = _runner_up(
                p3.get("ranking", []), family
            )
            entry["runner_up_model"] = runner_up_model
            entry["runner_up_family"] = runner_up_family
            entry["runner_up_log_score"] = (
                p3.get("mean_log_score", {}).get(runner_up_model)
                if runner_up_model
                else None
            )
        else:
            entry["note"] = "not in phase_3_results.json; selected via Phase 2 fallback"
        products[product] = entry

    payload = {
        "version": "v1",
        "created": "2026-08-11",
        "source_notebook": "008 Phase 3 (Phase 2 fallback for products absent "
        "from Phase 3's per-product ranking)",
        "policy": "P1",
        "products": products,
    }
    payload["config_hash"] = config_hash(payload)

    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"written {OUT_PATH} ({len(products)} products)")


if __name__ == "__main__":
    main()
