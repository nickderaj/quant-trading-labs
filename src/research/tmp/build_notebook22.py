"""Notebook 022 builder. A narrative over the phase JSONs -- loads results
and renders them, does not re-run any backtest.

Usage (from repo root): uv run python src/research/tmp/build_notebook22.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TMP = REPO_ROOT / "src" / "research" / "tmp"
OUT_PATH = (
    REPO_ROOT / "src" / "research" / "022_hyperliquid_cex_dex_funding_spread.ipynb"
)


def cid() -> str:
    return uuid.uuid4().hex[:8]


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cid(),
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cid(),
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


cells = []

cells.append(
    md("""\
# Notebook 022 — A CEX/DEX Funding Spread: Hyperliquid vs. Binance

NEXT_PROMPT.md's 2026-08-23 planning pass found a cross-venue funding spread roughly 4x the size of
the one notebook 020 tested (Binance vs. Bybit, both large CEXs, which found nothing worth having),
on a venue class this repo has never touched: an on-chain perpetual exchange (Hyperliquid) against a
CEX (Binance). The planning probe measured +9.41% annualised on BTC over two years of dev-window
data, t≈21, with a structural story for why it should persist — on-chain flow is retail/directional,
a large CEX's is institutional/two-sided, and institutions find a DEX hard to onboard to.

This notebook tests that, on real Hyperliquid data, for the first time in this repo. Full narrative
and numbers: `src/results/022_hyperliquid_cex_dex_funding_spread.md`.
""")
)

cells.append(
    code("""\
import json
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "tmp")

TMP = "tmp"


def load(name):
    with open(f"{TMP}/{name}") as f:
        return json.load(f)


prereg = load("phase_0_22_preregistration.json")
phase2 = load("phase_2_22_results.json")
phase4 = load("phase_4_22_results.json")

print("n_trials (DSR):", prereg["n_trials_itemisation"]["total"])
print("dev window:", prereg["constants"]["hyperliquid_new"]["DEV_START"], "..",
      prereg["constants"]["hyperliquid_new"]["DEV_END"])
""")
)

cells.append(
    md("""\
## Pre-registration (Phase 0)

Sign convention: A=Hyperliquid, B=Binance throughout. `carry = EWMA(hl_funding - binance_funding)`;
carry>0 means Hyperliquid's funding is the more expensive side (the structural prior), and the
w=+1 direction is short Hyperliquid's perp / long Binance's perp — collecting the spread.

Three pre-registered book configurations: `HL_ALWAYSON` (the headline — every liquid symbol held
every bar, direction=sign(carry), no timing), `HL_TIMED_FAST` (bl20's own 15-day-target-hold
thetas, a falsification comparator), `HL_TIMED_SLOW` (30-day target hold). All-in round-turn cost:
23.0bp — cheaper than 020's Bybit book (25.0bp) because Hyperliquid's taker fee undercuts Bybit's.
""")
)

cells.append(
    code("""\
print(json.dumps(prereg["constants"]["hyperliquid_new"], indent=2))
""")
)

cells.append(
    md("""\
## Phase 1 — Universe mapping and data fetch

128 Binance universe-seed symbols mapped onto Hyperliquid's own naming (strip `USDT`, `1000X` ->
`kX`), checked against Hyperliquid's own `/info meta` listing rather than guessed. Phase 1b then
fetched funding history (native hourly, capped at 500 records/call — chunked) and 8h candles for
every mapped symbol, from the public, unauthenticated `api.hyperliquid.xyz/info` endpoint, hard-
guarded to refuse any request reaching past `research.HOLDOUT_START`.
""")
)

cells.append(
    code("""\
with open(f"{TMP}/../../../scratch/022/phase1a_probe.json") as f:
    probe = json.load(f)
print("mapped:", probe["n_mapped"], "/", probe["n_binance_universe_seed"])

with open(f"{TMP}/../../../scratch/022/phase1_manifest.json") as f:
    manifest = json.load(f)
print("fetched ok:", manifest["n_ok"], "/", manifest["n_symbols"])
""")
)

cells.append(
    md("""\
## Phase 2b (unplanned) — a tripwire caught a frozen-feed contamination

Phase 4's *first* run of the headline book produced an implausible signature: net Sharpe 4.29, but
kurtosis (non-excess) 80.4 and skew 2.08 — a smoking gun for a handful of extreme bars, not a smooth
carry return. Tracing the worst/best bars found them clustered in a single week (2025-01-06..11),
dominated by `FTMUSDT`: Fantom's Sonic migration froze its **Binance** perp feed (price flat,
volume=0) while Hyperliquid's own FTM kept trading — so the "spread" on those bars was a real, moving
HL price against a stale Binance mark, not a captured funding spread. This is 018/021's own documented
frozen-feed signature, reused unmodified via `power_lib21.flag_frozen_feed_bars` (import-only).

Scanning all 50 fetched symbols found 3 with material contamination: `FTTUSDT` (100% frozen — FTX
token, dead throughout), `BLZUSDT` (25.8%), `FTMUSDT` (23.9%). A disclosed, mechanical rule — exclude
any symbol frozen on more than 5% of its own dev-window bars, decided before checking whether
exclusion helps or hurts the headline — drops the universe from 50 to 47. `HL_ALWAYSON`'s *lack* of
any timing threshold (every liquid symbol, every bar) is exactly why it is more exposed to this than
018/020's threshold-gated books, where a frozen, unchanging funding rate rarely crosses the entry
threshold in the first place. That is a genuine, reportable property of the always-on construction.
""")
)

cells.append(
    code("""\
with open(f"{TMP}/../../../scratch/022/frozen_feed_exclusions.json") as f:
    screen = json.load(f)
print("excluded:", screen["excluded_symbols"])
print("kept:", screen["n_kept"], "/", screen["n_mapped_universe"])
print(json.dumps({k: v for k, v in screen["per_symbol_frozen_fraction"].items() if v > 0.05},
                  indent=2))
""")
)

cells.append(
    md("""\
## Phase 2 — Does the spread exist? (HD-1) Is the window powered? (HD-3)

HD-1 pools the gross, pre-cost paired return (short Hyperliquid / long Binance, fixed direction)
across every liquid symbol-bar in the mapped universe — not just BTC and ETH, which were seen first
in the planning probe (declared in the trial log). HD-3 computes the minimum detectable annualised
Sharpe closed-form, from this same dev window, before any book is scored.
""")
)

cells.append(
    code("""\
print(json.dumps(phase2["gate_HD1"], indent=2))
print()
print(json.dumps(phase2["gate_HD3"], indent=2))
""")
)

cells.append(
    md("""\
### Per-symbol spread reproduction (Probe P1, full universe)

BTC and ETH's planning-probe numbers (+9.41%/+8.01% annualised, t=21.4/19.5), reproduced here from a
fresh fetch and the full pre-registered dev window, alongside every other liquid symbol in the
mapped universe.
""")
)

cells.append(
    code("""\
btc = phase2["per_symbol_spread"].get("BTCUSDT")
eth = phase2["per_symbol_spread"].get("ETHUSDT")
print("BTC:", json.dumps(btc, indent=2))
print("ETH:", json.dumps(eth, indent=2))
print()
print("BTC price divergence (Probe P2 reproduction):")
print(json.dumps(phase2["btc_price_divergence"], indent=2))
""")
)

cells.append(
    md("""\
## Phase 4/5 — The book grid and gates

`HL_ALWAYSON` at four origin offsets (018/020's own robustness convention for a fixed-parameter,
non-refit strategy), the two timing comparators, and six Phase 5 ablations: cost sensitivity at
0bp/19bp(bare-fee)/40bp(long-tail slippage stress), excluding the top-2 contributing symbols,
and BTC-only / ETH-only decompositions.
""")
)

cells.append(
    code("""\
for o, cell in phase4["cells"]["HL_ALWAYSON_offsets"].items():
    m = cell["metrics"]
    print(f"offset {o}: sharpe_net={m['sharpe_net']:.3f} ci={cell['net_ci_95']} "
          f"beta_ok={cell['beta_ok']} turnover/yr={m['annualized_turnover']:.1f}")
print()
print("HL_TIMED_FAST sharpe_net:", phase4["cells"]["HL_TIMED_FAST"]["metrics"]["sharpe_net"])
print("HL_TIMED_SLOW sharpe_net:", phase4["cells"]["HL_TIMED_SLOW"]["metrics"]["sharpe_net"])
""")
)

cells.append(
    code("""\
print("gate HD-2 (tradeable):", json.dumps(phase4["gate_HD2"], indent=2))
""")
)

cells.append(
    code("""\
print("gate HD-4 (neutral):", json.dumps(phase4["gate_HD4"], indent=2))
""")
)

cells.append(
    code("""\
print("gate HD-5 (not solely episodic):", json.dumps(phase4["gate_HD5"], indent=2))
print()
print("top-2 contributing symbols:", phase4["top2_contributing_symbols"]["symbols"])
""")
)

cells.append(
    code("""\
print("HD-6 falsification check (descriptive):",
      json.dumps(phase4["gate_HD6_falsification_check"], indent=2))
""")
)

cells.append(
    code("""\
print("gate HD-7 (survives 40bp cost stress):", json.dumps(phase4["gate_HD7"], indent=2))
print()
print("gate FUND-HL:", json.dumps(phase4["gate_FUND_HL"], indent=2))
""")
)

cells.append(
    md("""\
## The JELLY force-settlement event (descriptive, not a gate)

On 2025-03-26, Hyperliquid validators voted to force-settle the JELLY perpetual at an
administratively chosen price to protect the protocol's own vault; vault TVL fell from ~$540M to
~$150M. JELLY itself is not Binance-listed and is not in this notebook's universe, so this reports
`HL_ALWAYSON`'s own realized net return through the surrounding window as a platform-risk
characterisation, not a JELLY-specific trade outcome.
""")
)

cells.append(
    code("""\
print(json.dumps(phase4["jelly_event_study"], indent=2))
""")
)

cells.append(
    md("""\
## Holdout access and bottom line
""")
)

cells.append(
    code("""\
print(json.dumps(phase4["holdout_access"], indent=2))
""")
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.13",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print(f"written {OUT_PATH}")
