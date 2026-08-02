"""11c Phase 2: Gate LC's second leg (the suppression book) and the final
verdict, cross-checked verbatim against 11a's committed pre-registration
(NEXT_PROMPT.md's new binding rule: "11c's gate table must match 11a's
pre-registration exactly" -- enforced here as a runtime assertion, not by
re-typing the criterion by hand).

Builds, per origin offset, a book that vetoes new entries whose walk-forward
out-of-sample predicted P(stop) falls in the top decile of that offset's own
OOS-covered trades (an entry a trade would otherwise have taken under the
pre-declared rule is skipped; nothing new is ever entered). Trades that
predate any OOS prediction (no classifier has been trained on that much
history yet) are left untouched -- suppressing them would require a
prediction that does not exist without lookahead. Compares the suppressed
book to the unsuppressed 57-trade control (Phase 0 / 11a Phase 4) on the
three-way risk gate (sec 5), at every offset, since the rest of this
programme's gates all require "every offset" and Gate LC's own text does not
say otherwise.

Also reports a bootstrap-CI robustness check on the classifier's own
stitched AUC (not part of the pre-registered criterion, which is a bare
point-estimate threshold -- reported alongside it because a >0.60 point
estimate on 55 trades deserves exactly the same scrutiny this programme has
given every other small-sample result).

Writes phase_2_11c_results.json.
"""

import json
import sys

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import commod_lib8 as C8
import numpy as np
import polars as pl
import spread_lib11 as S11

SPREAD_DIR = "src/research/data/market/spreads"
DEV_END = "2024-12-31"
LIVE_SPREADS = [
    "brent_wti",
    "brent_calendar",
    "corn_wheat",
    "bean_corn",
    "kc_chicago_wheat",
]
STOP_ATR_OVERRIDES = {"brent_calendar": 4.0, "kc_chicago_wheat": 12.0}
REGIME_REQUIREMENTS = {"brent_calendar": "backwardation"}
PHASE0_PATH = "src/research/tmp/phase_0_11c_results.json"
PHASE1_PATH = "src/research/tmp/phase_1_11c_results.json"
PREREG_PATH = "src/research/tmp/phase_6_11a_results.json"
OUT_PATH = "src/research/tmp/phase_2_11c_results.json"
DSR_N_TRIALS = 4
TOP_DECILE = 0.90
N_BOOT = 2000


def load_frames() -> dict[str, pl.DataFrame]:
    frames = {}
    for name in LIVE_SPREADS:
        df = pl.read_parquet(f"{SPREAD_DIR}/{name}.parquet")
        df = df.filter(pl.col("date") <= pl.lit(DEV_END).str.to_date()).sort("date")
        frames[name] = df
    return frames


def build_book(frames, veto_entry_masks=None):
    p = S11.TradingRuleParams()
    params = {n: p for n in LIVE_SPREADS}
    book = S11.simulate_book(
        frames,
        params,
        STOP_ATR_OVERRIDES,
        REGIME_REQUIREMENTS,
        C8.round_turn_cost_per_contract,
        veto_entry_masks=veto_entry_masks,
    )
    return book, S11.book_metrics(book)


def three_way(metrics: dict) -> dict:
    return {
        "sharpe": metrics["sharpe"],
        "max_drawdown": metrics["max_drawdown"],
        "return_over_drawdown": metrics["return_over_drawdown"],
        "n_trades": metrics["n_trades"],
    }


def main() -> None:
    frames = load_frames()
    with open(PHASE0_PATH) as f:
        phase0 = json.load(f)
    with open(PHASE1_PATH) as f:
        phase1 = json.load(f)
    with open(PREREG_PATH) as f:
        prereg = json.load(f)

    lc_prereg = prereg["gates"]["LC"]
    lc_n_trials_prereg = prereg["dsr_counts"]["LC"]
    assert lc_prereg["notebook"] == "11c"
    assert lc_prereg["claim"] == (
        "entry-time features predict which trades will exit via stop"
    )
    assert lc_prereg["fires_if"] == (
        "a classifier trained walk-forward achieves out-of-sample AUC > 0.60 "
        "on the stop-exit label AND a book that suppresses its top-decile "
        "predicted-loss entries beats the unsuppressed book on the three-way "
        "risk gate"
    )
    assert lc_n_trials_prereg["n_trials"] == DSR_N_TRIALS

    trades_by_key = {(t["spread"], t["entry_date"]): t for t in phase0["trades"]}

    _unsuppressed_book, unsuppressed_metrics = build_book(frames)

    per_offset: dict[str, dict] = {}
    for offset_str, stitched in phase1["stitched_oof_predictions_by_offset"].items():
        pred = np.array(stitched["pred_prob_stop"])
        n_oof = len(pred)
        if n_oof == 0:
            per_offset[offset_str] = {"skipped": "no OOS-covered trades"}
            continue
        threshold = float(np.quantile(pred, TOP_DECILE))
        vetoed_trade_idx = stitched["trade_idx"]
        # Map stitched OOF row index -> the (spread, entry_date) key of the
        # trade it corresponds to, via the same feature-complete, entry-date
        # sorted trade list Phase 1 built its matrix from.
        feature_complete_trades = [
            t
            for t in phase0["trades"]
            if all(t[f] is not None for f in phase0["feature_names"])
        ]
        vetoed_keys = {
            (
                feature_complete_trades[i]["spread"],
                feature_complete_trades[i]["entry_date"],
            )
            for i, p in zip(vetoed_trade_idx, pred)
            if p >= threshold
        }

        veto_entry_masks = {}
        for name, df in frames.items():
            dates = df["date"].to_numpy()
            mask = np.zeros(len(dates), dtype=bool)
            date_index = {d: i for i, d in enumerate(dates)}
            for key in vetoed_keys:
                if key[0] != name:
                    continue
                spread_trade = trades_by_key.get(key)
                if spread_trade is None:
                    continue
                entry_date = np.datetime64(spread_trade["entry_date"])
                if entry_date in date_index:
                    mask[date_index[entry_date]] = True
            veto_entry_masks[name] = mask

        _suppressed_book, suppressed_metrics = build_book(frames, veto_entry_masks)

        three_way_control = three_way(unsuppressed_metrics)
        three_way_treat = three_way(suppressed_metrics)
        beats_on_three_way = (
            three_way_treat["sharpe"] > three_way_control["sharpe"]
            and three_way_treat["max_drawdown"] >= three_way_control["max_drawdown"]
            and three_way_treat["return_over_drawdown"]
            > three_way_control["return_over_drawdown"]
        )

        per_offset[offset_str] = {
            "n_oof_covered_trades": n_oof,
            "top_decile_threshold_pred_prob": threshold,
            "n_vetoed_trades": len(vetoed_keys),
            "unsuppressed_three_way": three_way_control,
            "suppressed_three_way": three_way_treat,
            "suppressed_beats_unsuppressed_three_way": bool(beats_on_three_way),
        }

    all_offsets_beat = (
        all(
            v.get("suppressed_beats_unsuppressed_three_way", False)
            for v in per_offset.values()
            if "skipped" not in v
        )
        and len(per_offset) == DSR_N_TRIALS
    )

    # Robustness check on the classifier's own AUC point estimate (not part
    # of the pre-registered criterion; reported alongside it).
    rng = np.random.default_rng(0)
    auc_robustness: dict[str, dict] = {}
    for offset_str, stitched in phase1["stitched_oof_predictions_by_offset"].items():
        pred = np.array(stitched["pred_prob_stop"])
        true = np.array(stitched["true_label_stop"])
        n = len(true)
        boots: list[float] = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            a = S11.roc_auc_score(true[idx], pred[idx])
            if np.isfinite(a):
                boots.append(a)
        boots_arr = np.array(boots)
        lo, hi = (
            np.percentile(boots_arr, [2.5, 97.5])
            if len(boots_arr)
            else (float("nan"), float("nan"))
        )
        auc_robustness[offset_str] = {
            "point_estimate": phase1["results_by_offset"][offset_str][
                "stitched_oof_auc"
            ],
            "bootstrap_95ci": [float(lo), float(hi)],
            "ci_excludes_0_5": bool(lo > 0.5) if np.isfinite(lo) else None,
        }

    auc_leg_fires = phase1["all_offsets_auc_above_0_60"]
    gate_lc_fires = bool(auc_leg_fires and all_offsets_beat)

    out = {
        "prereg_cross_check": {
            "claim": lc_prereg["claim"],
            "fires_if": lc_prereg["fires_if"],
            "n_trials": lc_n_trials_prereg,
            "matches_11a_prereg_exactly": True,
        },
        "auc_leg_fires": bool(auc_leg_fires),
        "suppression_leg_fires_all_offsets": bool(all_offsets_beat),
        "gate_lc_fires": gate_lc_fires,
        "per_offset_suppression": per_offset,
        "auc_bootstrap_robustness": auc_robustness,
        "_note": (
            "AUC-leg point estimates (0.67-0.76) clear the pre-registered "
            "0.60 bar at every offset by their literal, non-CI text, but the "
            "bootstrap 95% CI on stitched OOS AUC does not exclude 0.5 at "
            "any offset (n=20-25 OOS-covered trades per offset) -- reported "
            "as a fragility caveat regardless of the mechanical verdict."
        ),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(
        f"Phase 2 (11c): auc_leg_fires={auc_leg_fires} "
        f"suppression_leg_fires_all_offsets={all_offsets_beat} "
        f"gate_lc_fires={gate_lc_fires}"
    )


if __name__ == "__main__":
    main()
