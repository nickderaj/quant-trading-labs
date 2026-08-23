"""Notebook 022, Phase 1a: map the Binance universe-seed symbols (128, same
seed 018/020 use) onto Hyperliquid's perp `name` field, and write the
resulting intersection + mapping to scratch/022/phase1a_probe.json.

Hyperliquid names coins without the USDT suffix Binance uses, and represents
some low-price tokens with a "k" (x1000) prefix instead of Binance's "1000"
prefix (e.g. Binance "1000SHIBUSDT" == Hyperliquid "kSHIB"). This script
applies exactly that one normalisation rule -- strip "USDT", replace a
leading "1000" with "k" -- and keeps only symbols found in HL's own /info
`meta` response, rather than guessing further with regex (NEXT_PROMPT.md
Candidate 1, Probe P3's own caveat: "a real mapping has to be built and
tested, not regex-guessed"). Anything not found this way is recorded as
unmapped, not silently dropped -- Phase 1b's fetch manifest is the second,
harder check (does Hyperliquid actually have usable history for it).

Usage: uv run python src/research/tmp/run_phase_1a_22_universe_map.py
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
UNIVERSE_SEED_PATH = (
    REPO_ROOT / "src" / "research" / "tmp" / "design_c_v2_universe.json"
)
OUT_PATH = REPO_ROOT / "scratch" / "022" / "phase1a_probe.json"

HL_INFO_URL = "https://api.hyperliquid.xyz/info"


def _post(body: dict) -> dict | list:
    req = urllib.request.Request(
        HL_INFO_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def normalize(binance_symbol: str) -> str:
    base = binance_symbol.removesuffix("USDT")
    if base.startswith("1000"):
        return "k" + base[4:]
    return base


def main() -> None:
    with open(UNIVERSE_SEED_PATH) as f:
        binance_universe: list[str] = json.load(f)

    meta = _post({"type": "meta"})
    assert isinstance(meta, dict)
    hl_entries = {u["name"]: u for u in meta["universe"]}

    mapping: dict[str, str] = {}
    delisted: list[str] = []
    unmapped: list[str] = []
    for sym in binance_universe:
        hl_name = normalize(sym)
        entry = hl_entries.get(hl_name)
        if entry is None:
            unmapped.append(sym)
            continue
        mapping[sym] = hl_name
        if entry.get("isDelisted"):
            delisted.append(sym)

    out = {
        "n_hl_perps_total": len(hl_entries),
        "n_binance_universe_seed": len(binance_universe),
        "n_mapped": len(mapping),
        "n_unmapped": len(unmapped),
        "mapping": mapping,
        "unmapped_binance_symbols": sorted(unmapped),
        "mapped_but_hl_flagged_delisted": sorted(delisted),
        "normalization_rule": "strip trailing USDT; leading '1000' -> 'k'",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(
        f"mapped {len(mapping)}/{len(binance_universe)} symbols "
        f"({len(delisted)} HL-flagged delisted, kept -- may still have usable "
        f"pre-delisting dev-window history); wrote {OUT_PATH}"
    )


if __name__ == "__main__":
    main()
