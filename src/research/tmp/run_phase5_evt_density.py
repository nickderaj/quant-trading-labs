"""Phase 5 driver: can the spliced GPD-tails-plus-KDE-body EVT density
(dist_lib6.fit_spliced_evt_density) finally be entered in the log-score
contest, where notebook 5's own attempt at this proved too fiddly to trust?

Scores GARCH-EVT (d8, splice built on the GARCH-normal variance recursion's
standardized residuals) and GJR-EVT (d9, on GJR-normal's) into the same
log-score / all-pairs-DM / BH machinery as Phase 1/3, against the full
8-model Gate A set plus the strongest Phase 3 zoo family. Run per-symbol.
"""

import json
import sys
import time

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import dist_lib as L
import dist_lib6 as L6
import numpy as np

import research

DM_BOOTSTRAP_N = 500
INTERVAL_ORDER = ["12h", "4h", "1h", "1d"]


def run_one_interval(symbol: str, interval: str) -> dict:
    t0 = time.time()
    df = L.build_asset_frame(symbol, interval, end=research.HOLDOUT_START)
    n = len(df)
    ret = df["log_return"].fill_null(0.0).to_numpy()

    variance_fc, nu_paths, fits = L6.build_gate_a_forecasts(df, interval, ret)
    log_score_full, scores = L6.score_gate_a_models(ret, variance_fc, nu_paths)

    spliced_fits_d8 = L6.rolling_spliced_evt_fits(
        ret, fits["d4"], model="garch", max_train=L6.MLE_MAX_TRAIN
    )
    spliced_fits_d9 = L6.rolling_spliced_evt_fits(
        ret, fits["d6"], model="gjr", max_train=L6.MLE_MAX_TRAIN
    )

    n_refits_attempted_d8 = len(fits["d4"])
    n_refits_succeeded_d8 = len(spliced_fits_d8)
    n_refits_attempted_d9 = len(fits["d6"])
    n_refits_succeeded_d9 = len(spliced_fits_d9)

    ls_d8 = L6.score_spliced_evt_model(
        ret, variance_fc["d4_garch_normal"], spliced_fits_d8
    )
    ls_d9 = L6.score_spliced_evt_model(
        ret, variance_fc["d6_gjr_normal"], spliced_fits_d9
    )
    log_score_full["d8_garch_evt"] = ls_d8
    log_score_full["d9_gjr_evt"] = ls_d9
    scores["d8_garch_evt"] = {
        "log_score_mean": float(np.nanmean(ls_d8)),
        "n": int(np.isfinite(ls_d8).sum()),
    }
    scores["d9_gjr_evt"] = {
        "log_score_mean": float(np.nanmean(ls_d9)),
        "n": int(np.isfinite(ls_d9).sum()),
    }

    model_names = list(log_score_full.keys())
    all_pairs_dm = L6.all_pairs_dm_bh(
        model_names, log_score_full, dm_bootstrap_n=DM_BOOTSTRAP_N, seed=0
    )

    mean_log_score = {name: scores[name]["log_score_mean"] for name in model_names}
    best_name = max(mean_log_score, key=lambda n: mean_log_score[n])
    gate_a_fires_boot = L6.beats_all_significantly(
        best_name, model_names, all_pairs_dm, "bh_bootstrap"
    )

    # spliced-density health diagnostics, reported per NEXT_RUN_PROMPT.md's
    # own "an honest partial entry beats a hand-waved density" standard
    def _continuity_gap(spliced_fits):
        if not spliced_fits:
            return None
        gaps = []
        for f in spliced_fits:
            sp = f["spliced"]
            eps = 1e-4
            lo_left = np.exp(
                L6.spliced_evt_logpdf(np.array([sp["u_lower"] - eps]), sp)[0]
            )
            lo_right = np.exp(
                L6.spliced_evt_logpdf(np.array([sp["u_lower"] + eps]), sp)[0]
            )
            hi_left = np.exp(
                L6.spliced_evt_logpdf(np.array([sp["u_upper"] - eps]), sp)[0]
            )
            hi_right = np.exp(
                L6.spliced_evt_logpdf(np.array([sp["u_upper"] + eps]), sp)[0]
            )
            gaps.append(abs(lo_left - lo_right) / max(lo_left, lo_right))
            gaps.append(abs(hi_left - hi_right) / max(hi_left, hi_right))
        return float(np.mean(gaps))

    return {
        "n_obs": n,
        "scores": scores,
        "all_pairs_dm": all_pairs_dm,
        "best_by_log_score": best_name,
        "gate_a_fires_bootstrap": gate_a_fires_boot,
        "spliced_density_health": {
            "d8_garch_evt": {
                "n_refits_attempted": n_refits_attempted_d8,
                "n_refits_succeeded": n_refits_succeeded_d8,
                "frac_succeeded": n_refits_succeeded_d8 / n_refits_attempted_d8
                if n_refits_attempted_d8
                else None,
                "mean_relative_continuity_gap": _continuity_gap(spliced_fits_d8),
            },
            "d9_gjr_evt": {
                "n_refits_attempted": n_refits_attempted_d9,
                "n_refits_succeeded": n_refits_succeeded_d9,
                "frac_succeeded": n_refits_succeeded_d9 / n_refits_attempted_d9
                if n_refits_attempted_d9
                else None,
                "mean_relative_continuity_gap": _continuity_gap(spliced_fits_d9),
            },
        },
        "elapsed_sec": time.time() - t0,
    }


def main():
    if len(sys.argv) < 2:
        print("usage: run_phase5_evt_density.py SYMBOL [interval ...]")
        sys.exit(1)
    symbol = sys.argv[1]
    intervals = sys.argv[2:] if len(sys.argv) > 2 else INTERVAL_ORDER

    out = {"symbol": symbol, "intervals": {}}
    for interval in intervals:
        res = run_one_interval(symbol, interval)
        out["intervals"][interval] = res
        h8 = res["spliced_density_health"]["d8_garch_evt"]
        print(
            f"{symbol} {interval}: n={res['n_obs']} elapsed={res['elapsed_sec']:.1f}s "
            f"best={res['best_by_log_score']} gate_a={res['gate_a_fires_bootstrap']} "
            f"d8_score={res['scores']['d8_garch_evt']['log_score_mean']:.3f} "
            f"d8_frac_succeeded={h8['frac_succeeded']:.2f} d8_continuity_gap={h8['mean_relative_continuity_gap']:.3f}"
        )
        out_path = f"src/research/tmp/phase5_evt_density_{symbol}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=1, default=float)
        print(f"  (incremental write to {out_path})")


if __name__ == "__main__":
    main()
