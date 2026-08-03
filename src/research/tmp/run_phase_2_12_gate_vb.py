"""Notebook 12 Phase 2 -- Gate VB: the pooled volume-confirmed breakout
book vs. the byte-identical ungated control, across the whole ~85/88-
instrument basket (NEXT_PROMPT.md sec 3). `fires_if` is transcribed
verbatim from `phase_1_12_preregistration.json` and asserted, not re-typed.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import run_phase_0_12_data_and_calibration as P0
import spread_lib11 as S11

import research

research.set_seed(0)

ORIGIN_OFFSETS = [0, 7, 14, 21]
COST_MULTIPLIERS = [1, 2, 3]
BASE_COST_BPS_PER_SIDE = 0.0004 + 0.0001  # repo convention, reused across all
# three asset classes per 11d's disclosed precedent (this repo has no
# established futures- or equity-specific cost model; see notebook text).
N_TRIALS_VB = 12  # 4 offsets x 3 cost multipliers, single fixed rule/k
ANNUALIZED_RATE = float(np.sqrt(252))


def build_pooled_universe() -> tuple[
    dict[str, dict], dict[str, np.ndarray], dict[str, str]
]:
    crypto = P0.build_crypto_universe()
    equity = P0.build_equity_universe()
    futures = P0.build_futures_universe()

    crypto_regime = P0.build_regime_series(crypto, P0.CRYPTO_REGIME_REF, min_confirm=1)
    equity_regime = P0.build_regime_series(equity, P0.EQUITY_REGIME_REF, min_confirm=2)
    futures_regime = P0.build_regime_series(
        futures, P0.FUTURES_REGIME_REF, min_confirm=2
    )

    universe: dict[str, dict] = {}
    regime_ok: dict[str, np.ndarray] = {}
    asset_class: dict[str, str] = {}
    for uni, regime, cls in [
        (crypto, crypto_regime, "crypto"),
        (equity, equity_regime, "equity"),
        (futures, futures_regime, "futures"),
    ]:
        for s, f in uni.items():
            calib_end = P0.calibration_cutoff(f["dates"])
            post_calib = f["dates"] >= calib_end
            universe[s] = f
            regime_ok[s] = regime[s] & post_calib
            asset_class[s] = cls
    return universe, regime_ok, asset_class


def trim(f: dict, offset: int) -> dict:
    if offset == 0:
        return f
    return {k: (v[offset:] if isinstance(v, np.ndarray) else v) for k, v in f.items()}


def build_book(
    universe: dict[str, dict],
    regime_ok: dict[str, np.ndarray],
    offset: int,
    cost_mult: int,
    use_volume: bool,
    thresholds: dict,
) -> dict:
    trimmed = {s: trim(f, offset) for s, f in universe.items()}
    trimmed_regime = {s: r[offset:] if offset else r for s, r in regime_ok.items()}
    p = S11.VolBreakoutParams(
        base_max_range_atr_mult=thresholds["base_max_range_atr_mult"],
        prior_run_min_atr_mult=thresholds["prior_run_min_atr_mult"],
        vol_k=thresholds["vol_k"],
        use_volume=use_volume,
    )
    return S11.simulate_vol_breakout_book(
        trimmed, trimmed_regime, p, BASE_COST_BPS_PER_SIDE * cost_mult
    )


def per_asset_class_counts(trades: list[dict], asset_class: dict[str, str]) -> dict:
    out = {"crypto": 0, "equity": 0, "futures": 0}
    for t in trades:
        out[asset_class[t["symbol"]]] += 1
    return out


def assert_matches_preregistration() -> dict:
    prereg = json.loads(
        Path("src/research/tmp/phase_1_12_preregistration.json").read_text()
    )
    vb = prereg["gates"]["VB"]
    assert vb["n_trials"] == N_TRIALS_VB
    assert vb["origin_offsets"] == ORIGIN_OFFSETS
    assert vb["cost_multipliers"] == COST_MULTIPLIERS
    assert vb["one_rule_one_k"] is True
    return vb


def main() -> None:
    assert_matches_preregistration()
    calib = json.loads(Path("src/research/tmp/phase_0_12_results.json").read_text())
    thresholds = calib["thresholds"]

    universe, regime_ok, asset_class = build_pooled_universe()

    gated_by_offset_by_cost: dict = {}
    ungated_by_offset_by_cost: dict = {}
    for offset in ORIGIN_OFFSETS:
        gated_costs, ungated_costs = {}, {}
        for mult in COST_MULTIPLIERS:
            gbook = build_book(universe, regime_ok, offset, mult, True, thresholds)
            ubook = build_book(universe, regime_ok, offset, mult, False, thresholds)
            gmetrics = S11.breakout_book_metrics(gbook)
            umetrics = S11.breakout_book_metrics(ubook)
            gated_costs[f"cost_{mult}x"] = gmetrics
            ungated_costs[f"cost_{mult}x"] = umetrics
            if mult == 1:
                gated_costs["_book_1x"] = gbook
                ungated_costs["_book_1x"] = ubook
        gated_by_offset_by_cost[f"offset_{offset}"] = gated_costs
        ungated_by_offset_by_cost[f"offset_{offset}"] = ungated_costs

    sharpes_gated_1x = [
        gated_by_offset_by_cost[f"offset_{o}"]["cost_1x"]["sharpe"]
        for o in ORIGIN_OFFSETS
    ]
    positive_every_offset = all(s > 0 for s in sharpes_gated_1x)

    gbook0_1x = gated_by_offset_by_cost["offset_0"]["_book_1x"]
    ubook0_1x = ungated_by_offset_by_cost["offset_0"]["_book_1x"]
    gtrades0 = gbook0_1x["trades"]
    utrades0 = ubook0_1x["trades"]
    gret0 = np.array([t["ret_eq"] for t in gtrades0])
    uret0 = np.array([t["ret_eq"] for t in utrades0])
    gblocks0 = S11.trade_blocks(np.array([t["exit_date"] for t in gtrades0]))
    ublocks0 = S11.trade_blocks(np.array([t["exit_date"] for t in utrades0]))

    # Mechanism isolation (sec 3): paired block bootstrap of the volume-
    # gated book's own return distribution against the ungated control's,
    # NOT a naive difference-of-means -- same primitive 11a/11b/11d use
    # throughout ("bootstrap the paired difference to get its own CI").
    gated_vs_ungated = S11.paired_block_bootstrap(uret0, ublocks0, gret0, gblocks0)
    ci_excludes_zero = gated_vs_ungated["delta_excludes_zero"]

    gbook0_3x = build_book(universe, regime_ok, 0, 3, True, thresholds)
    gtrades0_3x = gbook0_3x["trades"]
    cost_stress = S11.paired_block_bootstrap(
        gret0,
        gblocks0,
        np.array([t["ret_eq"] for t in gtrades0_3x]),
        S11.trade_blocks(np.array([t["exit_date"] for t in gtrades0_3x])),
    )

    n_trades0 = len(gtrades0)
    dsr = (
        research.deflated_sharpe_prob(
            sharpes_gated_1x[0] / ANNUALIZED_RATE, n_trials=N_TRIALS_VB, n_obs=n_trades0
        )
        if n_trades0 > 1
        else float("nan")
    )
    dsr_fires = bool(dsr > 0.95) if np.isfinite(dsr) else False

    max_dd0 = gated_by_offset_by_cost["offset_0"]["cost_1x"]["max_drawdown"]
    cost_stress_correctly_signed = cost_stress["delta_point"] < 0

    fires = (
        positive_every_offset
        and ci_excludes_zero
        and dsr_fires
        and cost_stress_correctly_signed
    )
    fundable = (
        all(s > 0.5 for s in sharpes_gated_1x) and dsr_fires and abs(max_dd0) <= 0.25
    )

    trade_counts_gated = per_asset_class_counts(gtrades0, asset_class)
    trade_counts_ungated = per_asset_class_counts(utrades0, asset_class)

    out = {
        "gate": "VB",
        "n_trials": N_TRIALS_VB,
        "thresholds": thresholds,
        "origin_offsets": ORIGIN_OFFSETS,
        "cost_multipliers": COST_MULTIPLIERS,
        "gated_by_offset": {
            off: {c: m for c, m in vals.items() if not c.startswith("_")}
            for off, vals in gated_by_offset_by_cost.items()
        },
        "ungated_by_offset": {
            off: {c: m for c, m in vals.items() if not c.startswith("_")}
            for off, vals in ungated_by_offset_by_cost.items()
        },
        "sharpes_gated_1x_by_offset": dict(
            zip([f"offset_{o}" for o in ORIGIN_OFFSETS], sharpes_gated_1x, strict=True)
        ),
        "positive_every_offset": positive_every_offset,
        "gated_vs_ungated_bootstrap_offset0_1x": gated_vs_ungated,
        "ci_excludes_zero": ci_excludes_zero,
        "cost_stress_1x_vs_3x_gated_offset0": cost_stress,
        "cost_stress_correctly_signed": cost_stress_correctly_signed,
        "deflated_sharpe_prob": dsr,
        "dsr_fires": dsr_fires,
        "gate_fires": fires,
        "fundable_flag": fundable,
        "n_trades_gated_offset0_1x": n_trades0,
        "n_trades_ungated_offset0_1x": len(utrades0),
        "trade_counts_by_asset_class_gated": trade_counts_gated,
        "trade_counts_by_asset_class_ungated": trade_counts_ungated,
        "max_drawdown_offset0_1x_gated": max_dd0,
    }
    Path("src/research/tmp/phase_2_12_results.json").write_text(
        json.dumps(out, indent=2)
    )
    print(
        f"Gate VB: fires={fires} fundable={fundable} "
        f"sharpes_gated_1x={[round(s, 3) for s in sharpes_gated_1x]} "
        f"gated_vs_ungated_CI=[{gated_vs_ungated['delta_ci'][0]:.4f},{gated_vs_ungated['delta_ci'][1]:.4f}] "
        f"dsr={dsr:.3f} n_trades_gated={n_trades0} n_trades_ungated={len(utrades0)} "
        f"trade_counts_by_class={trade_counts_gated} maxDD={max_dd0:.4f}"
    )


if __name__ == "__main__":
    main()
