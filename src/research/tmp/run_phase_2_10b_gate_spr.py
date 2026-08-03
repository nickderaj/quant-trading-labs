"""10b Phase 2: Gates SPR and SPR-BW (NEXT_PROMPT.md sec 6 Phase 2). Reuses
Phase 1's exact trading rule, cost model, and universe -- inter-commodity,
eligible (ADF-cointegrated) spreads only, per sec 4.2 -- the ONLY change is a
regime gate applied on top of the same z-score signal: position is zeroed
out on any day the regime label is not a "definite" state.

**Regime definition**: PRIMARY = deadband (10a Phase 5's own declared
primary, not raw sign -- see phase_5_10a_results.json's REGIME_DEFINITIONS
rationale). Secondary variants (raw_sign, persistent) are also run and
reported (all three enter the DSR count per 10a's pre-registration; only
deadband is the fire-condition headline). Regime leg = leg1's own curve
(the pre-declared rule from 10a Phase 3/5), except brent_wti's own
"both legs agree" variant (the sec 4.1 named robustness check for Gate
SPR-BW), run once, at offset 0 only, per its declared n_trials=1.

**Gate SPR** (pooled inter-commodity book, deadband primary): fires if the
gated book meets Gate SP's own full criterion AND its net Sharpe exceeds the
unconditional book's net Sharpe at every origin offset AND a block-bootstrap
95% CI on the (gated - unconditional) daily return difference excludes zero.

**Gate SPR-BW**: per-spread (not pooled) comparison at offset 0 -- does the
gated Sharpe exceed the unconditional Sharpe for brent_wti, and independently
for at least 3 other eligible inter-commodity spreads?

Writes phase_2_10b_results.json.
"""

import json
import sys

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import commod_lib8 as C
import numpy as np
import polars as pl
import spread_lib10 as S
from run_phase_1_10b_gate_sp import (
    ANNUALIZED_RATE,
    DEV_END,
    SPREAD_DIR,
    ZSCORE_WINDOW,
    point_value,
    series_metrics,
)

import research

TAXONOMY_PATH = "src/research/tmp/phase_2_10a_results.json"
CURVE_DIR = "src/research/tmp/phase_0_curves"
OUT_PATH = "src/research/tmp/phase_2_10b_results.json"
ORIGIN_OFFSETS = [0, 7, 14, 21]
N_TRIALS_SPR = 12  # 3 regime defs x 4 offsets -- from phase_5_10a_results.json
N_TRIALS_SPR_BW = (
    1  # the brent_wti both-legs-agree variant -- from phase_5_10a_results.json
)

research.set_seed(0)


def load_regime_frame(product: str) -> pl.DataFrame:
    curve = pl.read_parquet(f"{CURVE_DIR}/{product}.parquet")
    curve = curve.filter(pl.col("date") <= pl.lit(DEV_END).str.to_date()).sort("date")
    ts = C.term_structure_state(
        curve.select(["date", "close_f1", "dte_f1", "close_f2", "dte_f2"])
    )
    ts = ts.with_columns(
        S.regime_deadband(ts["roll_slope_annualized"]).alias("state_deadband")
    )
    ts = ts.with_columns(
        S.regime_persistent(ts["term_structure_state"]).alias("state_persistent")
    )
    return ts.select(
        ["date", "term_structure_state", "state_deadband", "state_persistent"]
    )


def spread_returns_gated(
    name: str, leg_products: list[str], regime_col: str, regime_frame: pl.DataFrame
) -> pl.DataFrame | None:
    """Same trading rule as Phase 1's `spread_daily_returns`, but position is
    zeroed on any day where `regime_col` (from `regime_frame`) is "flat"
    (deadband), "unconfirmed" (persistent), or null -- the definite-state
    gate. Everything else (signal, cost, capital basis, roll-window
    exclusion) is byte-identical to Phase 1's unconditional version.
    """
    df = pl.read_parquet(f"{SPREAD_DIR}/{name}.parquet")
    df = df.filter(pl.col("date") <= pl.lit(DEV_END).str.to_date()).sort("date")
    df = df.filter(~pl.col("roll_window_flag"))
    if df.height < ZSCORE_WINDOW + 30:
        return None
    df = df.join(regime_frame, on="date", how="left")

    value = df["value"]
    roll_mean = value.rolling_mean(window_size=ZSCORE_WINDOW)
    roll_std = value.rolling_std(window_size=ZSCORE_WINDOW)
    z = ((value - roll_mean) / roll_std).clip(-2.0, 2.0)
    raw_position = (-z / 2.0).fill_null(0.0)

    no_trade = (
        pl.col(regime_col).is_in(["flat", "unconfirmed"]) | pl.col(regime_col).is_null()
    )
    position = pl.when(no_trade).then(0.0).otherwise(raw_position).alias("position")
    df = df.with_columns(position)

    leg_prices = df["leg_prices"].to_list()
    n_legs = len(leg_products)
    point_values = [point_value(p) for p in leg_products]
    capital_basis = np.array(
        [
            sum(abs(row[i]) * point_values[i] for i in range(n_legs))
            for row in leg_prices
        ]
    )
    round_turn_costs_dollars = sum(
        C.round_turn_cost_per_contract(p) for p in leg_products
    )

    v = value.to_numpy()
    pos = df["position"].to_numpy()
    dv = np.diff(v, prepend=np.nan)
    pos_lag = np.concatenate([[0.0], pos[:-1]])
    dollar_pnl = pos_lag * dv * point_values[0]
    gross_return = dollar_pnl / np.where(capital_basis > 0, capital_basis, np.nan)

    dpos = np.abs(np.diff(pos, prepend=0.0))
    cost_frac = (
        dpos
        * round_turn_costs_dollars
        / np.where(capital_basis > 0, capital_basis, np.nan)
    )
    net_return = gross_return - cost_frac

    out = pl.DataFrame({"date": df["date"], "net_return": net_return})
    return out.filter(pl.col("net_return").is_finite())


def build_book(
    returns_by_spread: dict[str, pl.DataFrame], origin_offset: int
) -> pl.DataFrame:
    per_spread = []
    for name, ret in returns_by_spread.items():
        if ret is None:
            continue
        r = ret
        if origin_offset > 0:
            dates = r["date"].unique().sort().to_list()
            keep = set(dates[origin_offset:])
            r = r.filter(pl.col("date").is_in(list(keep)))
        per_spread.append(r.select(["date", "net_return"]).rename({"net_return": name}))
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


def main():
    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)["per_spread"]
    with open("src/research/tmp/phase_1_10b_results.json") as f:
        sp_results = json.load(f)
    eligible_names = sorted(sp_results["inter_commodity"]["eligible_spreads"])

    regime_frames = {
        name: load_regime_frame(taxonomy[name]["leg_products"][0])
        for name in eligible_names
    }

    definitions = {
        "raw_sign": "term_structure_state",
        "deadband": "state_deadband",
        "persistent": "state_persistent",
    }
    gated_returns = {def_name: {} for def_name in definitions}
    for def_name, col in definitions.items():
        for name in eligible_names:
            gated_returns[def_name][name] = spread_returns_gated(
                name, taxonomy[name]["leg_products"], col, regime_frames[name]
            )

    from run_phase_1_10b_gate_sp import spread_daily_returns

    unconditional_returns = {
        name: spread_daily_returns(name, taxonomy[name]["leg_products"])
        for name in eligible_names
    }

    def per_offset_metrics(
        returns_dict: dict, offset: int
    ) -> tuple[dict, pl.DataFrame]:
        book = build_book(returns_dict, offset)
        r = book["book_net_return"].to_numpy()
        return series_metrics(r), book

    spr_by_definition = {}
    for def_name in definitions:
        by_offset_gated, by_offset_unconditional = {}, {}
        gated_books, unconditional_books = {}, {}
        for offset in ORIGIN_OFFSETS:
            m_g, book_g = per_offset_metrics(gated_returns[def_name], offset)
            m_u, book_u = per_offset_metrics(unconditional_returns, offset)
            by_offset_gated[f"offset_{offset}"] = m_g
            by_offset_unconditional[f"offset_{offset}"] = m_u
            gated_books[offset] = book_g
            unconditional_books[offset] = book_u

        sharpes_g = [m["sharpe"] for m in by_offset_gated.values()]
        sharpes_u = [m["sharpe"] for m in by_offset_unconditional.values()]
        gated_all_positive = all(np.isfinite(s) and s > 0 for s in sharpes_g)
        exceeds_every_offset = all(
            np.isfinite(sg) and np.isfinite(su) and sg > su
            for sg, su in zip(sharpes_g, sharpes_u)
        )

        headline_g = gated_books[0].join(
            unconditional_books[0], on="date", how="inner", suffix="_u"
        )
        diff = (
            headline_g["book_net_return"] - headline_g["book_net_return_u"]
        ).to_numpy()
        diff = diff[np.isfinite(diff)]
        diff_ci = (
            research.block_bootstrap_ci(diff, n_boot=2000, seed=0)
            if len(diff) > 30
            else (None, None)
        )
        diff_ci_excludes_zero = diff_ci[0] is not None and (
            diff_ci[0] > 0 or diff_ci[1] < 0
        )

        gated_headline_returns = gated_books[0]["book_net_return"].to_numpy()
        ci_zero = (
            research.block_bootstrap_ci(gated_headline_returns, n_boot=2000, seed=0)
            if len(gated_headline_returns) > 30
            else (None, None)
        )
        ci_zero_excludes = ci_zero[0] is not None and (ci_zero[0] > 0 or ci_zero[1] < 0)

        n_trials = N_TRIALS_SPR if def_name == "deadband" else None
        dsr = (
            research.deflated_sharpe_prob(
                by_offset_gated["offset_0"]["sharpe"] / ANNUALIZED_RATE,
                n_trials=n_trials,
                n_obs=by_offset_gated["offset_0"]["n"],
            )
            if n_trials and np.isfinite(by_offset_gated["offset_0"]["sharpe"])
            else None
        )

        sp_full_criterion_met = bool(
            gated_all_positive and ci_zero_excludes and dsr is not None and dsr > 0.95
        )
        spr_by_definition[def_name] = {
            "by_offset_gated": by_offset_gated,
            "by_offset_unconditional": by_offset_unconditional,
            "gated_sharpe_positive_every_offset": gated_all_positive,
            "gated_exceeds_unconditional_every_offset": exceeds_every_offset,
            "gated_vs_zero_ci": list(ci_zero),
            "gated_vs_zero_ci_excludes_zero": ci_zero_excludes,
            "gated_minus_unconditional_ci": list(diff_ci),
            "gated_minus_unconditional_ci_excludes_zero": diff_ci_excludes_zero,
            "deflated_sharpe_prob": dsr,
            "is_headline": def_name == "deadband",
            "fires": bool(
                def_name == "deadband"
                and sp_full_criterion_met
                and exceeds_every_offset
                and diff_ci_excludes_zero
            ),
        }

    # Gate SPR-BW: per-spread comparison at offset 0, primary (deadband) definition.
    per_spread_bw = {}
    for name in eligible_names:
        g = gated_returns["deadband"][name]
        u = unconditional_returns[name]
        if g is None or u is None:
            per_spread_bw[name] = {
                "gated_sharpe": None,
                "unconditional_sharpe": None,
                "gated_exceeds": None,
            }
            continue
        mg, mu = (
            series_metrics(g["net_return"].to_numpy()),
            series_metrics(u["net_return"].to_numpy()),
        )
        per_spread_bw[name] = {
            "gated_sharpe": mg["sharpe"],
            "unconditional_sharpe": mu["sharpe"],
            "gated_exceeds": bool(
                np.isfinite(mg["sharpe"])
                and np.isfinite(mu["sharpe"])
                and mg["sharpe"] > mu["sharpe"]
            ),
        }

    n_exceeding = sum(1 for v in per_spread_bw.values() if v["gated_exceeds"])
    brent_wti_exceeds = per_spread_bw.get("brent_wti", {}).get("gated_exceeds")
    others_exceeding = n_exceeding - (1 if brent_wti_exceeds else 0)
    gate_spr_bw_fires = bool(brent_wti_exceeds and others_exceeding >= 3)

    # brent_wti both-legs-agree secondary variant (n_trials=1, offset 0 only).
    bz_regime = load_regime_frame("BZ").rename(
        {
            c: f"bz_{c}"
            for c in ["term_structure_state", "state_deadband", "state_persistent"]
        }
    )
    cl_regime = load_regime_frame("CL").rename(
        {
            c: f"cl_{c}"
            for c in ["term_structure_state", "state_deadband", "state_persistent"]
        }
    )
    both_regime = bz_regime.join(cl_regime, on="date", how="left")
    both_agree_state = (
        pl.when(pl.col("bz_state_deadband") == pl.col("cl_state_deadband"))
        .then(pl.col("bz_state_deadband"))
        .otherwise(pl.lit("disagree"))
        .alias("both_agree_deadband")
    )
    both_regime = both_regime.with_columns(both_agree_state)
    both_regime_for_gate = both_regime.select(["date", "both_agree_deadband"]).rename(
        {"both_agree_deadband": "state_for_gate"}
    )
    both_regime_for_gate = both_regime_for_gate.with_columns(
        pl.when(pl.col("state_for_gate").is_in(["flat", "disagree"]))
        .then(pl.lit("flat"))
        .otherwise(pl.col("state_for_gate"))
        .alias("state_for_gate")
    )
    bw_both_agree_returns = spread_returns_gated(
        "brent_wti",
        taxonomy["brent_wti"]["leg_products"],
        "state_for_gate",
        both_regime_for_gate,
    )
    bw_unconditional_returns = unconditional_returns["brent_wti"]
    m_bw_gated = (
        series_metrics(bw_both_agree_returns["net_return"].to_numpy())
        if bw_both_agree_returns is not None
        else {"sharpe": None}
    )
    m_bw_unconditional = (
        series_metrics(bw_unconditional_returns["net_return"].to_numpy())
        if bw_unconditional_returns is not None
        else {"sharpe": None}
    )

    results = {
        "regime_leg_rule": "leg1 (BZ for brent_wti), fixed per spread -- see phase_5_10a_results.json",
        "primary_regime_definition": "deadband",
        "n_trials_SPR": N_TRIALS_SPR,
        "n_trials_SPR_BW": N_TRIALS_SPR_BW,
        "gate_SPR_by_definition": spr_by_definition,
        "gate_SPR_bw": {
            "per_spread_offset_0": per_spread_bw,
            "n_spreads_gated_exceeds_unconditional": n_exceeding,
            "n_eligible_spreads": len(eligible_names),
            "brent_wti_exceeds": brent_wti_exceeds,
            "other_spreads_exceeding_count": others_exceeding,
            "fires": gate_spr_bw_fires,
            "brent_wti_both_legs_agree_secondary_variant": {
                "gated_sharpe": m_bw_gated["sharpe"],
                "unconditional_sharpe": m_bw_unconditional["sharpe"],
                "gated_exceeds_unconditional": bool(
                    np.isfinite(m_bw_gated["sharpe"])
                    and np.isfinite(m_bw_unconditional["sharpe"])
                    and m_bw_gated["sharpe"] > m_bw_unconditional["sharpe"]
                )
                if m_bw_gated["sharpe"] is not None
                and m_bw_unconditional["sharpe"] is not None
                else None,
            },
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"written {OUT_PATH}")
    for def_name, r in spr_by_definition.items():
        print(
            f"SPR[{def_name}]: fires={r['fires']} dsr={r['deflated_sharpe_prob']} "
            f"exceeds_every_offset={r['gated_exceeds_unconditional_every_offset']}"
        )
    print(
        f"SPR-BW: fires={gate_spr_bw_fires} brent_wti_exceeds={brent_wti_exceeds} others={others_exceeding}/{len(eligible_names) - 1}"
    )
    print(
        f"brent_wti both-legs-agree: gated_sharpe={m_bw_gated['sharpe']} unconditional_sharpe={m_bw_unconditional['sharpe']}"
    )


if __name__ == "__main__":
    main()
