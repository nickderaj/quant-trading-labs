"""11b Phase 0: Gates TS and TS-S (NEXT_PROMPT.md sec 4.2).

**Gate TS**: does the discrete-trade, ATR-stopped structured book (11a
Phase 4's `spread_lib11.simulate_book`, pre-declared params, our costs) beat
10b's continuous Gate SP book -- same five live spreads, same costs, same
four origin offsets -- net Sharpe > 0 at every offset AND a paired
block-bootstrap 95% CI on (structured - continuous) excludes zero AND
DSR > 0.95 on the n_trials=4 cumulative count (4 offsets x 1 pre-declared
parameter set, external priors, not swept).

**Gate TS-S**: Gate TS fires AND an otherwise-identical stop-disabled
variant (disable_stop=True, run once at offset 0 as a diagnostic, n_trials=+1)
FAILS the same paired CI -- i.e. removing only the stop, keeping the
discrete-trade packaging, should destroy whatever edge Gate TS found, if the
stop (not the packaging) is the active ingredient.

Origin offset here means: drop the first `offset` unique calendar dates from
EACH spread's own series (both the continuous return series and the
structured simulation's input frame) before running anything on it --
matching notebook 8/10a/10b's own origin-offset convention exactly.

Writes phase_0_11b_results.json.
"""

import json
import sys

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import commod_lib8 as C8
import numpy as np
import polars as pl
import spread_lib11 as S11
from run_phase_1_10b_gate_sp import spread_daily_returns

import research

SPREAD_DIR = "src/research/data/market/spreads"
TAXONOMY_PATH = "src/research/tmp/phase_2_10a_results.json"
OUT_PATH = "src/research/tmp/phase_0_11b_results.json"
DEV_END = "2024-12-31"
ORIGIN_OFFSETS = [0, 7, 14, 21]
ANNUALIZED_RATE = float(np.sqrt(252))
N_TRIALS_TS = 4
N_TRIALS_TS_S = 1

LIVE_SPREADS = [
    "brent_wti",
    "brent_calendar",
    "corn_wheat",
    "bean_corn",
    "kc_chicago_wheat",
]
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


def structured_book_at_offset(offset: int, disable_stop: bool = False) -> dict:
    frames = {n: load_frame(n, offset) for n in LIVE_SPREADS}
    p = S11.TradingRuleParams()
    params = {n: p for n in LIVE_SPREADS}
    return S11.simulate_book(
        frames,
        params,
        STOP_ATR_OVERRIDES,
        REGIME_REQUIREMENTS,
        C8.round_turn_cost_per_contract,
        disable_stop=disable_stop,
    )


def continuous_returns_at_offset(offset: int) -> pl.DataFrame:
    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)["per_spread"]
    rows = []
    for name in LIVE_SPREADS:
        leg_products = taxonomy[name]["leg_products"]
        ret = spread_daily_returns(name, leg_products)
        if ret is None:
            continue
        dates = ret["date"].unique().sort().to_list()
        if offset > 0 and offset < len(dates):
            ret = ret.filter(pl.col("date") >= pl.lit(dates[offset]))
        rows.append(ret.select(["date", "net_return"]))
    pooled = pl.concat(rows).group_by("date").agg(pl.col("net_return").mean()).sort("date")
    return pooled


def sharpe_of(returns: np.ndarray) -> float:
    returns = returns[np.isfinite(returns)]
    if len(returns) == 0 or np.std(returns) == 0:
        return float("nan")
    return float(np.mean(returns) / np.std(returns) * ANNUALIZED_RATE)


def main() -> None:
    structured_by_offset = {}
    continuous_by_offset = {}
    for offset in ORIGIN_OFFSETS:
        s_book = structured_book_at_offset(offset)
        s_metrics = S11.book_metrics(s_book)
        c_ret = continuous_returns_at_offset(offset)
        c_sharpe = sharpe_of(c_ret["net_return"].to_numpy())
        structured_by_offset[f"offset_{offset}"] = s_metrics
        continuous_by_offset[f"offset_{offset}"] = {
            "sharpe": c_sharpe,
            "n": c_ret.height,
        }

    # Offset-0 books for the paired bootstrap (same convention as every
    # prior 10a/10b/11a paired comparison -- the primary book, offsets are
    # a robustness check on the Sharpe/DSR headline, not re-bootstrapped
    # individually).
    s_book0 = structured_book_at_offset(0)
    c_ret0 = continuous_returns_at_offset(0)
    treatment_pnl = np.array([t["ret_eq"] for t in s_book0["trades"]])
    treatment_blocks = S11.trade_blocks(np.array([t["exit_date"] for t in s_book0["trades"]]))
    control_pnl = c_ret0["net_return"].to_numpy()
    control_blocks = S11.trade_blocks(c_ret0["date"].to_numpy())
    ts_bootstrap = S11.paired_block_bootstrap(
        control_pnl, control_blocks, treatment_pnl, treatment_blocks
    )

    ts_sharpes = [structured_by_offset[f"offset_{o}"]["sharpe"] for o in ORIGIN_OFFSETS]
    ts_positive_every_offset = all(sh > 0 for sh in ts_sharpes)
    ts_n = structured_by_offset["offset_0"]["n_trades"]
    ts_dsr = (
        research.deflated_sharpe_prob(
            ts_sharpes[0] / ANNUALIZED_RATE, n_trials=N_TRIALS_TS, n_obs=ts_n
        )
        if ts_n > 1
        else float("nan")
    )
    ts_fires = bool(
        ts_positive_every_offset and ts_bootstrap["delta_excludes_zero"] and ts_dsr > 0.95
    )

    # Gate TS-S: stop-disabled variant, offset 0 only, diagnostic.
    s_book0_nostop = structured_book_at_offset(0, disable_stop=True)
    treatment_pnl_nostop = np.array([t["ret_eq"] for t in s_book0_nostop["trades"]])
    treatment_blocks_nostop = S11.trade_blocks(
        np.array([t["exit_date"] for t in s_book0_nostop["trades"]])
    )
    ts_s_bootstrap = S11.paired_block_bootstrap(
        control_pnl, control_blocks, treatment_pnl_nostop, treatment_blocks_nostop
    )
    ts_s_metrics_nostop = S11.book_metrics(s_book0_nostop)
    ts_s_variant_fails = not ts_s_bootstrap["delta_excludes_zero"]
    ts_s_fires = bool(ts_fires and ts_s_variant_fails)

    out = {
        "gate_TS": {
            "structured_by_offset": structured_by_offset,
            "continuous_by_offset": continuous_by_offset,
            "net_sharpe_positive_every_offset": ts_positive_every_offset,
            "paired_bootstrap": ts_bootstrap,
            "deflated_sharpe_prob": ts_dsr,
            "n_trials": N_TRIALS_TS,
            "fires": ts_fires,
        },
        "gate_TS_S": {
            "stop_disabled_metrics_offset0": ts_s_metrics_nostop,
            "paired_bootstrap_stop_disabled": ts_s_bootstrap,
            "stop_disabled_variant_fails_ci": ts_s_variant_fails,
            "n_trials": N_TRIALS_TS_S,
            "fires": ts_s_fires,
        },
        "live_spreads": LIVE_SPREADS,
        "origin_offsets": ORIGIN_OFFSETS,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(
        f"Gate TS: fires={ts_fires} sharpes={[round(s,3) for s in ts_sharpes]} "
        f"dsr={ts_dsr:.4f} delta_ci={ts_bootstrap['delta_ci']} | "
        f"Gate TS-S: fires={ts_s_fires} stop_disabled_ci_excludes_zero={ts_s_bootstrap['delta_excludes_zero']}"
    )


if __name__ == "__main__":
    main()
