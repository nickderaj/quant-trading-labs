"""10b Phase 1: Gate SP -- the real, costed backtest notebook 9 deferred
(NEXT_PROMPT.md sec 6 Phase 1). Implements exactly the trading rule declared
in 10a's pre-registration (`phase_5_10a_results.json`'s TRADING_RULE) --
nothing here may deviate from what was pre-declared before this script was
written.

**Signal**: 60-day rolling z-score of the spread's own `value` series (same
window as 10a's own descriptive IC probe).
**Position**: position_t = -clip(z_t, -2, 2) / 2, decided using data through
day t, applied to the return realized from day t to day t+1 (no lookahead).
**Return**: dollar P&L per unit spread position, divided by the spread's own
capital basis (sum of |leg price x point value| across legs) -- puts every
spread on a comparable "return on capital employed" scale so pooling into an
equal-weighted book and comparing Sharpe ratios across spreads is meaningful.
**Cost**: ONE round-turn cost per leg, summed across all legs (a spread trade
is N round turns for an N-leg spread, not one -- NEXT_PROMPT.md's own
explicit warning, generalised here beyond the 2-leg case for crack_321 and
crush_soy's 3 legs), charged only on days the position actually changes.
**Roll-window exclusion**: rows flagged `roll_window_flag` are dropped before
computing the signal or accruing any P&L/cost -- applied to the backtest
itself, not only the descriptive screen.
**Universe**: eligible (`include_in_10b == True`, i.e. ADF-cointegrated at
5%) spreads only, per taxonomy group, each group's book equal-weighted
across its own eligible spreads.
**DSR**: n_trials = 8 (2 taxonomy groups x 4 origin offsets), per 10a's
Phase 5 pre-registration -- not re-derived here, only applied.

Writes phase_1_10b_results.json.
"""

import json
import sys

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import commod_lib8 as C
import numpy as np
import polars as pl

import research

SPREAD_DIR = "src/research/data/market/spreads"
TAXONOMY_PATH = "src/research/tmp/phase_2_10a_results.json"
OUT_PATH = "src/research/tmp/phase_1_10b_results.json"
DEV_END = "2024-12-31"
ZSCORE_WINDOW = 60
ORIGIN_OFFSETS = [0, 7, 14, 21]
ANNUALIZED_RATE = float(np.sqrt(252))
N_TRIALS_SP = 8  # 2 taxonomy groups x 4 origin offsets -- from phase_5_10a_results.json

research.set_seed(0)


def point_value(product: str) -> float:
    spec = C.CONTRACT_SPECS[product]
    return spec["tick_value"] / spec["tick"]


def spread_daily_returns(name: str, leg_products: list[str]) -> pl.DataFrame | None:
    """Per-spread net daily return series (net_return_t on date t, cost
    already deducted), plus the raw position, for one spread. Rows on/after
    a roll-window-flagged date are dropped BEFORE the z-score or any return
    is computed -- a contaminated day contributes to neither the signal nor
    the P&L (sec 6 Phase 1's explicit requirement).
    """
    df = pl.read_parquet(f"{SPREAD_DIR}/{name}.parquet")
    df = df.filter(pl.col("date") <= pl.lit(DEV_END).str.to_date()).sort("date")
    df = df.filter(~pl.col("roll_window_flag"))
    if df.height < ZSCORE_WINDOW + 30:
        return None

    value = df["value"]
    roll_mean = value.rolling_mean(window_size=ZSCORE_WINDOW)
    roll_std = value.rolling_std(window_size=ZSCORE_WINDOW)
    z = ((value - roll_mean) / roll_std).clip(-2.0, 2.0)
    position = (-z / 2.0).fill_null(0.0)

    leg_prices = df["leg_prices"].to_list()  # list[list[float]], one row per date
    n_legs = len(leg_products)
    point_values = [point_value(p) for p in leg_products]
    capital_basis = np.array(
        [sum(abs(row[i]) * point_values[i] for i in range(n_legs)) for row in leg_prices]
    )
    round_turn_costs_dollars = sum(C.round_turn_cost_per_contract(p) for p in leg_products)

    v = value.to_numpy()
    pos = position.to_numpy()
    dv = np.diff(v, prepend=np.nan)
    # P&L over [t-1, t] uses the position DECIDED at t-1 (pos shifted by one)
    # applied to the value change realized over that same interval -- no
    # lookahead: pos[t-1] only used z-scores through t-1.
    pos_lag = np.concatenate([[0.0], pos[:-1]])
    dollar_pnl = pos_lag * dv * point_values[0]  # value is quoted in leg1's own price units
    gross_return = dollar_pnl / np.where(capital_basis > 0, capital_basis, np.nan)

    dpos = np.abs(np.diff(pos, prepend=0.0))
    cost_frac = dpos * round_turn_costs_dollars / np.where(capital_basis > 0, capital_basis, np.nan)
    net_return = gross_return - cost_frac

    out = pl.DataFrame({
        "date": df["date"],
        "net_return": net_return,
        "gross_return": gross_return,
        "cost_frac": cost_frac,
        "position": pos,
    })
    return out.filter(pl.col("net_return").is_finite())


def build_book(spread_names: list[str], taxonomy: dict, origin_offset: int) -> pl.DataFrame:
    """Equal-weighted book across all eligible spreads in a taxonomy group,
    at a given origin offset (offset applied by dropping the first
    `origin_offset` unique dates from EACH spread's own return series before
    pooling, matching notebook 8's own origin-offset convention).
    """
    per_spread = []
    for name in spread_names:
        leg_products = taxonomy[name]["leg_products"]
        ret = spread_daily_returns(name, leg_products)
        if ret is None:
            continue
        if origin_offset > 0:
            dates = ret["date"].unique().sort().to_list()
            keep = set(dates[origin_offset:])
            ret = ret.filter(pl.col("date").is_in(list(keep)))
        per_spread.append(ret.select(["date", "net_return"]).rename({"net_return": name}))

    if not per_spread:
        return pl.DataFrame({"date": [], "book_net_return": []})

    book = per_spread[0]
    for f in per_spread[1:]:
        book = book.join(f, on="date", how="full", coalesce=True)
    return_cols = [c for c in book.columns if c != "date"]
    book = book.with_columns(
        pl.mean_horizontal([pl.col(c) for c in return_cols]).alias("book_net_return")
    ).sort("date")
    return book.select(["date", "book_net_return"]).drop_nulls()


def series_metrics(returns: np.ndarray) -> dict:
    r = returns[np.isfinite(returns)]
    if len(r) < 30:
        return {"sharpe": float("nan"), "n": len(r)}
    mean, std = float(np.mean(r)), float(np.std(r))
    sharpe = mean / std * ANNUALIZED_RATE if std > 0 else float("nan")
    equity = np.cumsum(r)
    max_dd = float(np.min(equity - np.maximum.accumulate(equity))) if len(equity) else float("nan")
    return {"sharpe": sharpe, "mean_daily": mean, "std_daily": std, "n": len(r), "max_drawdown_cum": max_dd}


def gate_sp_verdict(group: str, spread_names: list[str], taxonomy: dict) -> dict:
    by_offset = {}
    headline_returns = None
    for offset in ORIGIN_OFFSETS:
        book = build_book(spread_names, taxonomy, offset)
        r = book["book_net_return"].to_numpy()
        m = series_metrics(r)
        by_offset[f"offset_{offset}"] = m
        if offset == 0:
            headline_returns = r

    sharpes = [m["sharpe"] for m in by_offset.values()]
    all_positive = all(np.isfinite(s) and s > 0 for s in sharpes)

    ci_lo, ci_hi = research.block_bootstrap_ci(headline_returns, n_boot=2000, seed=0) if headline_returns is not None and len(headline_returns) > 30 else (None, None)
    ci_excludes_zero = ci_lo is not None and ci_hi is not None and (ci_lo > 0 or ci_hi < 0)

    headline = by_offset["offset_0"]
    dsr = (
        research.deflated_sharpe_prob(headline["sharpe"] / ANNUALIZED_RATE, n_trials=N_TRIALS_SP, n_obs=headline["n"])
        if np.isfinite(headline["sharpe"])
        else float("nan")
    )

    return {
        "taxonomy_group": group,
        "n_eligible_spreads": len(spread_names),
        "eligible_spreads": spread_names,
        "by_offset": by_offset,
        "excess_return_ci_vs_zero": [ci_lo, ci_hi],
        "ci_excludes_zero": ci_excludes_zero,
        "deflated_sharpe_prob": dsr,
        "net_sharpe_positive_at_every_offset": all_positive,
        "fires": bool(all_positive and ci_excludes_zero and np.isfinite(dsr) and dsr > 0.95),
    }


def main():
    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)["per_spread"]
    eligible = {name: v for name, v in taxonomy.items() if v["include_in_10b"]}
    inter_commodity = sorted([n for n, v in eligible.items() if v["taxonomy"] == "inter_commodity"])
    calendar = sorted([n for n, v in eligible.items() if v["taxonomy"] == "calendar"])

    results = {
        "trading_rule": {
            "signal": f"{ZSCORE_WINDOW}-day rolling z-score of spread value, clipped to [-2,2]",
            "position": "position_t = -z_t / 2",
            "cost": "one round-turn cost per leg, summed across legs, charged on |delta position|",
            "roll_window_exclusion": "applied to signal AND P&L, not just the descriptive screen",
            "n_trials": N_TRIALS_SP,
        },
        "inter_commodity": gate_sp_verdict("inter_commodity", inter_commodity, taxonomy),
        "calendar": gate_sp_verdict("calendar", calendar, taxonomy),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"written {OUT_PATH}")
    for g in ("inter_commodity", "calendar"):
        r = results[g]
        print(f"{g}: n={r['n_eligible_spreads']} fires={r['fires']} dsr={r['deflated_sharpe_prob']:.4f} "
              f"ci={r['excess_return_ci_vs_zero']} sharpes={[round(m['sharpe'],3) for m in r['by_offset'].values()]}")


if __name__ == "__main__":
    main()
