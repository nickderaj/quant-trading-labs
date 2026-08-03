"""11b Phase 1: Gates BF and BF-X (NEXT_PROMPT.md sec 4.2, sec 0.3, sec 0.1
worked example).

**Sign convention, resolved here** (11a Phase 1 flagged, not resolved --
`spread_lib11.carry_ratio`'s docstring): on this repo's own leg1-front
`value` convention, the literal `c_t = -value_t / full_carry_t` formula
evaluates POSITIVE in backwardation and NEGATIVE in contango -- the opposite
of the "+1 at the contango ceiling" description used to state the external
programme's own bucket boundaries (deep backwardation c<-0.5, mild
backwardation -0.5<c<0, above full carry c>1.1). Verified empirically on
`brent_calendar` (n=4079): backwardation rows have c in [+0.01, +11.13]
(mean +0.92), contango rows have c in [-6.94, -0.01] (mean -0.78) -- entirely
consistent with the flagged sign flip and with nothing in between. To apply
their bucket boundaries correctly to OUR sign convention, this notebook uses
`c_corrected = -carry_ratio(value, full_carry)` and applies their boundaries
to `c_corrected` -- equivalently, "mild backwardation" on our data is
`0 < carry_ratio < 0.5`.

**Gate BF**: sign-flip ONLY the entries opened while in the mild-
backwardation bucket (not a whole-spread flip -- sec 0.3's throughput
discipline requires preserving trade count, and flipping only a bucket
still does). Universe: Gate SP's own `calendar` taxonomy group (16
already-ADF-cointegrated spreads, `phase_1_10b_results.json`'s own declared,
non-cherry-picked pooled list -- not re-selected here). Financing rate: our
own cached FRED `DFF`, lagged 1 day, /100 (DFF is quoted in percent).
Storage constants: LOW/MID/HIGH (`spread_lib11.STORAGE_{LOW,MID,HIGH}`), all
three run and reported; MID is the pre-declared headline (their v4 §6.2's
own primary), matching sec 0.3's own gate criterion. `brent_calendar`'s
regime-gate and stop-mult override (sec 4.1) are unchanged -- the flip
operates on TOP of the existing entry rule, changing direction only.

**Gate BF-X**: does the headline (MID storage) book's improvement hold
independently for >=3 eligible calendar spreads (per-spread Sharpe of the
flipped variant vs the unconditional variant, single-spread books)?

Writes phase_1_11b_results.json.
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
GATE_SP_PATH = "src/research/tmp/phase_1_10b_results.json"
FRED_DFF_PATH = "src/research/data/market/fred/DFF.parquet"
OUT_PATH = "src/research/tmp/phase_1_11b_results.json"
DEV_END = "2024-12-31"
ORIGIN_OFFSETS = [0, 7, 14, 21]
ANNUALIZED_RATE = float(np.sqrt(252))
N_TRIALS_BF = 12  # 3 storage constants x 4 offsets
N_TRIALS_BF_X = 0  # per-spread breakdown of BF's own book, pooling convention
MILD_BACKWARDATION_BAND = (0.0, 0.5)  # on carry_ratio (our sign), per docstring above

STOP_ATR_OVERRIDES = {"brent_calendar": 4.0}
REGIME_REQUIREMENTS = {"brent_calendar": "backwardation"}
STORAGE_CONSTANTS = {
    "low": S11.STORAGE_LOW,
    "mid": S11.STORAGE_MID,
    "high": S11.STORAGE_HIGH,
}
HEADLINE_STORAGE = "mid"

research.set_seed(0)


def calendar_universe() -> list[str]:
    with open(GATE_SP_PATH) as f:
        gate_sp = json.load(f)
    return gate_sp["calendar"]["eligible_spreads"]


def load_frame(name: str, offset: int) -> pl.DataFrame:
    df = pl.read_parquet(f"{SPREAD_DIR}/{name}.parquet")
    df = df.filter(pl.col("date") <= pl.lit(DEV_END).str.to_date()).sort("date")
    if offset > 0:
        dates = df["date"].unique().sort().to_list()
        keep_from = dates[offset] if offset < len(dates) else dates[-1]
        df = df.filter(pl.col("date") >= pl.lit(keep_from))
    return df


def dff_frame() -> pl.DataFrame:
    dff = (
        pl.read_parquet(FRED_DFF_PATH)
        .with_columns(pl.col("date").cast(pl.Date))
        .sort("date")
    )
    return dff.with_columns((pl.col("DFF").shift(1) / 100.0).alias("financing_rate"))


def mild_backwardation_mask(
    df: pl.DataFrame, dff: pl.DataFrame, storage: float
) -> np.ndarray:
    joined = df.select(["date", "value", "leg2_price"]).join(
        dff.select(["date", "financing_rate"]), on="date", how="left"
    )
    joined = joined.with_columns(pl.col("financing_rate").fill_null(strategy="forward"))
    value = joined["value"].to_numpy().astype(float)
    leg2 = joined["leg2_price"].to_numpy().astype(float)
    financing = joined["financing_rate"].to_numpy().astype(float)
    fv = S11.compute_carry_fv(leg2, storage, financing)
    c = np.asarray(S11.carry_ratio(value, fv))
    lo, hi = MILD_BACKWARDATION_BAND
    return (c > lo) & (c < hi)


def build_book(
    spreads: list[str], offset: int, storage_key: str | None, dff: pl.DataFrame
) -> dict:
    frames = {n: load_frame(n, offset) for n in spreads}
    p = S11.TradingRuleParams()
    params = {n: p for n in spreads}
    sign_flip_masks = None
    if storage_key is not None:
        storage = STORAGE_CONSTANTS[storage_key]
        sign_flip_masks = {
            n: mild_backwardation_mask(frames[n], dff, storage) for n in spreads
        }
    return S11.simulate_book(
        frames,
        params,
        {k: v for k, v in STOP_ATR_OVERRIDES.items() if k in spreads},
        {k: v for k, v in REGIME_REQUIREMENTS.items() if k in spreads},
        C8.round_turn_cost_per_contract,
        sign_flip_masks=sign_flip_masks,
    )


def main() -> None:
    spreads = calendar_universe()
    dff = dff_frame()

    unconditional_by_offset = {}
    for offset in ORIGIN_OFFSETS:
        book = build_book(spreads, offset, None, dff)
        unconditional_by_offset[f"offset_{offset}"] = S11.book_metrics(book)

    bf_results = {}
    per_storage_offset0_books = {}
    for storage_key in STORAGE_CONSTANTS:
        by_offset = {}
        for offset in ORIGIN_OFFSETS:
            book = build_book(spreads, offset, storage_key, dff)
            by_offset[f"offset_{offset}"] = S11.book_metrics(book)
            if offset == 0:
                per_storage_offset0_books[storage_key] = book
        exceeds_every_offset = all(
            by_offset[f"offset_{o}"]["sharpe"]
            > unconditional_by_offset[f"offset_{o}"]["sharpe"]
            for o in ORIGIN_OFFSETS
        )
        n_bf = by_offset["offset_0"]["n_trades"]
        n_uncond = unconditional_by_offset["offset_0"]["n_trades"]
        trade_count_within_10pct = (
            n_uncond > 0 and abs(n_bf - n_uncond) / n_uncond <= 0.10
        )
        bf_results[storage_key] = {
            "by_offset": by_offset,
            "exceeds_unconditional_every_offset": exceeds_every_offset,
            "n_trades_bf": n_bf,
            "n_trades_unconditional": n_uncond,
            "trade_count_within_10pct": trade_count_within_10pct,
        }

    # Paired bootstrap + DSR on the headline (MID) storage constant, offset 0.
    headline_book = per_storage_offset0_books[HEADLINE_STORAGE]
    uncond_book0 = build_book(spreads, 0, None, dff)
    treatment_pnl = np.array([t["ret_eq"] for t in headline_book["trades"]])
    treatment_blocks = S11.trade_blocks(
        np.array([t["exit_date"] for t in headline_book["trades"]])
    )
    control_pnl = np.array([t["ret_eq"] for t in uncond_book0["trades"]])
    control_blocks = S11.trade_blocks(
        np.array([t["exit_date"] for t in uncond_book0["trades"]])
    )
    bf_bootstrap = S11.paired_block_bootstrap(
        control_pnl, control_blocks, treatment_pnl, treatment_blocks
    )
    headline_sharpe = bf_results[HEADLINE_STORAGE]["by_offset"]["offset_0"]["sharpe"]
    headline_n = bf_results[HEADLINE_STORAGE]["by_offset"]["offset_0"]["n_trades"]
    bf_dsr = (
        research.deflated_sharpe_prob(
            headline_sharpe / ANNUALIZED_RATE, n_trials=N_TRIALS_BF, n_obs=headline_n
        )
        if headline_n > 1
        else float("nan")
    )
    bf_headline = bf_results[HEADLINE_STORAGE]
    bf_fires = bool(
        bf_headline["exceeds_unconditional_every_offset"]
        and bf_bootstrap["delta_excludes_zero"]
        and bf_dsr > 0.95
        and bf_headline["trade_count_within_10pct"]
    )

    # Gate BF-X: per-spread Sharpe, headline storage, single-spread books,
    # unconditional vs flipped, offset 0.
    per_spread_bf_x = {}
    for name in spreads:
        uncond_single = build_book([name], 0, None, dff)
        bf_single = build_book([name], 0, HEADLINE_STORAGE, dff)
        uncond_sharpe = S11.book_metrics(uncond_single)["sharpe"]
        bf_sharpe = S11.book_metrics(bf_single)["sharpe"]
        per_spread_bf_x[name] = {
            "unconditional_sharpe": uncond_sharpe,
            "bf_sharpe": bf_sharpe,
            "bf_improves": bool(
                np.isfinite(bf_sharpe)
                and np.isfinite(uncond_sharpe)
                and bf_sharpe > uncond_sharpe
            ),
        }
    n_spreads_improved = sum(1 for v in per_spread_bf_x.values() if v["bf_improves"])
    bf_x_fires = bool(bf_fires and n_spreads_improved >= 3)

    out = {
        "eligible_universe": spreads,
        "mild_backwardation_band_our_sign": MILD_BACKWARDATION_BAND,
        "sign_convention_note": (
            "carry_ratio's literal formula is positive in backwardation and negative in "
            "contango on this repo's data (opposite of the external repo's stated "
            "description); mild-backwardation bucket boundaries are applied as "
            "0 < carry_ratio < 0.5 to match their -0.5 < c_corrected < 0 under our sign."
        ),
        "unconditional_by_offset": unconditional_by_offset,
        "gate_BF": {
            "headline_storage": HEADLINE_STORAGE,
            "by_storage": bf_results,
            "paired_bootstrap_headline": bf_bootstrap,
            "deflated_sharpe_prob_headline": bf_dsr,
            "n_trials": N_TRIALS_BF,
            "fires": bf_fires,
        },
        "gate_BF_X": {
            "per_spread": per_spread_bf_x,
            "n_spreads_improved": n_spreads_improved,
            "n_trials": N_TRIALS_BF_X,
            "fires": bf_x_fires,
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(
        f"Gate BF: fires={bf_fires} headline={HEADLINE_STORAGE} "
        f"sharpes={[round(bf_results[HEADLINE_STORAGE]['by_offset'][f'offset_{o}']['sharpe'], 3) for o in ORIGIN_OFFSETS]} "
        f"vs unconditional={[round(unconditional_by_offset[f'offset_{o}']['sharpe'], 3) for o in ORIGIN_OFFSETS]} "
        f"dsr={bf_dsr:.4f} trade_count_ok={bf_headline['trade_count_within_10pct']} | "
        f"Gate BF-X: fires={bf_x_fires} n_improved={n_spreads_improved}/{len(spreads)}"
    )


if __name__ == "__main__":
    main()
