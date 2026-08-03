"""11b Phase 2: Gate SCR (NEXT_PROMPT.md sec 4.2, sec 2.2).

Does this repo's own 10a sec 4.3 ADF exclusion (`include_in_10b`) earn its
place when applied to the pre-declared trading rule's live-5 universe? Two
books, same trading rule (sec 4.1), same costs, four origin offsets:

- **screen-inclusive** (KEEP the screen): the four ADF-passing live spreads
  (`brent_wti`, `brent_calendar`, `corn_wheat`, `bean_corn` --
  `kc_chicago_wheat` fails ADF at t=-2.41, `phase_2_10a_results.json`).
- **screen-exclusive** (full eligible universe): the same four PLUS
  `kc_chicago_wheat` (one of the external programme's own five live
  spreads, excluded by our screen) and the two named cross-repo-conflict
  spreads `gc_cal_m2m3` (our ADF t=-1.59, their v4 flags it as promising)
  and `es_calendar` (our ADF t=+0.08, their v4 flags it as their strongest
  breadth candidate) -- resolving both conflicts here rather than assuming
  either repo's verdict.

KEEP only if the ADF-passing universe's net Sharpe beats the full universe's
at every offset AND a paired block-bootstrap 95% CI on
(screen-inclusive - screen-exclusive) excludes zero (and is positive --
sec 4.2's "a screen that cannot beat its own absence does not survive").

Writes phase_2_11b_results.json.
"""

import json
import sys

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import commod_lib8 as C8
import numpy as np
import polars as pl
import spread_lib11 as S11

import research

SPREAD_DIR = "src/research/data/market/spreads"
OUT_PATH = "src/research/tmp/phase_2_11b_results.json"
DEV_END = "2024-12-31"
ORIGIN_OFFSETS = [0, 7, 14, 21]
ANNUALIZED_RATE = float(np.sqrt(252))
N_TRIALS_SCR = 8  # 2 universes x 4 offsets

ADF_PASSING = ["brent_wti", "brent_calendar", "corn_wheat", "bean_corn"]
CONFLICT_ADDITIONS = ["kc_chicago_wheat", "gc_cal_m2m3", "es_calendar"]
FULL_ELIGIBLE = ADF_PASSING + CONFLICT_ADDITIONS
STOP_ATR_OVERRIDES = {"brent_calendar": 4.0, "kc_chicago_wheat": 12.0}
REGIME_REQUIREMENTS = {"brent_calendar": "backwardation"}

research.set_seed(0)


def load_frame(name: str, offset: int) -> pl.DataFrame:
    df = pl.read_parquet(f"{SPREAD_DIR}/{name}.parquet")
    df = df.filter(pl.col("date") <= pl.lit(DEV_END).str.to_date()).sort("date")
    if offset > 0:
        dates = df["date"].unique().sort().to_list()
        keep_from = dates[offset] if offset < len(dates) else dates[-1]
        df = df.filter(pl.col("date") >= pl.lit(keep_from))
    return df


def build_book(spreads: list[str], offset: int) -> dict:
    frames = {n: load_frame(n, offset) for n in spreads}
    p = S11.TradingRuleParams()
    params = {n: p for n in spreads}
    return S11.simulate_book(
        frames,
        params,
        {k: v for k, v in STOP_ATR_OVERRIDES.items() if k in spreads},
        {k: v for k, v in REGIME_REQUIREMENTS.items() if k in spreads},
        C8.round_turn_cost_per_contract,
    )


def main() -> None:
    screen_by_offset = {}
    noscreen_by_offset = {}
    for offset in ORIGIN_OFFSETS:
        screen_by_offset[f"offset_{offset}"] = S11.book_metrics(
            build_book(ADF_PASSING, offset)
        )
        noscreen_by_offset[f"offset_{offset}"] = S11.book_metrics(
            build_book(FULL_ELIGIBLE, offset)
        )

    screen_beats_every_offset = all(
        screen_by_offset[f"offset_{o}"]["sharpe"]
        > noscreen_by_offset[f"offset_{o}"]["sharpe"]
        for o in ORIGIN_OFFSETS
    )

    screen_book0 = build_book(ADF_PASSING, 0)
    noscreen_book0 = build_book(FULL_ELIGIBLE, 0)
    treatment_pnl = np.array([t["ret_eq"] for t in screen_book0["trades"]])
    treatment_blocks = S11.trade_blocks(
        np.array([t["exit_date"] for t in screen_book0["trades"]])
    )
    control_pnl = np.array([t["ret_eq"] for t in noscreen_book0["trades"]])
    control_blocks = S11.trade_blocks(
        np.array([t["exit_date"] for t in noscreen_book0["trades"]])
    )
    scr_bootstrap = S11.paired_block_bootstrap(
        control_pnl, control_blocks, treatment_pnl, treatment_blocks
    )
    scr_positive_delta = scr_bootstrap["delta_point"] > 0

    scr_keep = bool(
        screen_beats_every_offset
        and scr_bootstrap["delta_excludes_zero"]
        and scr_positive_delta
    )

    # Per-conflict-spread standalone Sharpe, for the resolution narrative.
    conflict_standalone = {
        name: S11.book_metrics(build_book([name], 0)) for name in CONFLICT_ADDITIONS
    }

    out = {
        "adf_passing_universe": ADF_PASSING,
        "full_eligible_universe": FULL_ELIGIBLE,
        "screen_inclusive_by_offset": screen_by_offset,
        "screen_exclusive_by_offset": noscreen_by_offset,
        "screen_beats_full_every_offset": screen_beats_every_offset,
        "paired_bootstrap": scr_bootstrap,
        "conflict_spread_standalone": conflict_standalone,
        "n_trials": N_TRIALS_SCR,
        "keep_screen": scr_keep,
        "fires": scr_keep,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(
        f"Gate SCR: keep_screen={scr_keep} screen_sharpes={[round(screen_by_offset[f'offset_{o}']['sharpe'], 3) for o in ORIGIN_OFFSETS]} "
        f"full_sharpes={[round(noscreen_by_offset[f'offset_{o}']['sharpe'], 3) for o in ORIGIN_OFFSETS]} "
        f"delta_ci={scr_bootstrap['delta_ci']}"
    )


if __name__ == "__main__":
    main()
