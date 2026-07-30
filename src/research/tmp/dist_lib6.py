"""Notebook-6-local machinery: does the tail-risk result generalize, and how
wide is the distribution zoo that could beat it.

Same convention as dist_lib5.py: this module imports dist_lib.py and
dist_lib5.py rather than forking their causal, rolling-refit building blocks
(`import dist_lib as L`, `import dist_lib5 as L5`). Everything here is either
new machinery notebook 5 never needed (the Phase 3 innovation-family
registry, the Phase 4 violation-process PMFs, the Phase 5 spliced EVT
density) or a thin notebook-6-specific composition of dist_lib/dist_lib5's
existing pieces.

Run as a script from the repo root (sys.path.insert(0, "src")), and imported
from the notebook the same way dist_lib5.py is (sys.path.insert(0, "tmp")
from src/research/).
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")
sys.path.insert(0, "src/research/tmp")

import numpy as np  # noqa: E402

import dist_lib as L  # noqa: E402  (path must be set up first)
import dist_lib5 as L5  # noqa: E402

# --------------------------------------------------------------------------
# Shared constants (unchanged from notebook 5's own drivers - reused, not
# re-declared with different values, so cadence stays comparable).
# --------------------------------------------------------------------------

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "BNBUSDT", "XRPUSDT"]
TRANSFER_SYMBOLS = ["ETHUSDT", "SOLUSDT", "DOGEUSDT", "BNBUSDT", "XRPUSDT"]
INTERVALS = ["1h", "4h", "12h", "1d"]
BARS_PER_DAY = {"1h": 24, "4h": 6, "12h": 2, "1d": 1}
MIN_TRAIN_DAYS = 90
CHEAP_REFIT_DAYS = 7
MLE_REFIT_DAYS = 30
MLE_MAX_TRAIN = 500

# The 8-model Phase 1/Gate-A competitor set, unchanged from notebook 5's
# Phase 3 (NEXT_RUN_PROMPT.md's own instruction: "do not add models here").
GATE_A_MODEL_IDS = [
    "d0_trailing_std", "d1_har_rv", "d2_har_log_rv", "d3_range",
    "d4_garch_normal", "d5_garch_t", "d6_gjr_normal", "d7_gjr_t",
]
T_MODEL_IDS = {"d5_garch_t", "d7_gjr_t"}

# Thin-tailed model set for Gate U (ES universality), per NEXT_RUN_PROMPT.md
# section 3's exact list.
THIN_TAILED_MODEL_IDS = {
    "d0_trailing_std", "d1_har_rv", "d2_har_log_rv", "d3_range",
    "d4_garch_normal", "d6_gjr_normal",
}
FAT_TAILED_MODEL_IDS = {"d5_garch_t", "d7_gjr_t"}
EVT_MODEL_IDS = {"d8_garch_evt", "d9_gjr_evt"}


def build_gate_a_forecasts(df, interval: str, ret: np.ndarray) -> tuple[dict, dict]:
    """Rebuild the identical 8-model variance-forecast set notebook 5's
    Phase 3 used, on whatever (symbol, interval) frame is passed in.

    Returns (variance_fc, nu_paths) exactly matching run_phase3_density.py's
    own dict shapes and keys, so downstream scoring/DM/BH code is byte-for-
    byte reusable across notebook 5 and notebook 6.
    """
    import distributions as dist

    bpd = BARS_PER_DAY[interval]
    cheap_refit_every = CHEAP_REFIT_DAYS * bpd
    mle_refit_every = MLE_REFIT_DAYS * bpd
    min_train = MIN_TRAIN_DAYS * bpd
    n = len(df)
    rv = df["rv_target"].to_numpy()

    variance_fc: dict[str, np.ndarray] = {}
    nu_paths: dict[str, np.ndarray] = {}

    trailing_candidates = {f"trailing_{w}": L.rung0_trailing_std(df, w).to_numpy() for w in [8, 24, 96]}

    def _q(fc):
        m = np.isfinite(fc) & (fc > 0) & (rv > 0)
        return np.nanmean(dist.qlike(rv[m], fc[m])) if m.sum() > 10 else np.inf

    _, variance_fc["d0_trailing_std"] = min(trailing_candidates.items(), key=lambda kv: _q(kv[1]))

    har_df = L.make_har_features(df, interval)
    variance_fc["d1_har_rv"] = L.rolling_ols_refit(
        har_df, ["rv_d", "rv_w", "rv_m"], "rv_target", refit_every=cheap_refit_every, min_train=min_train,
    )
    variance_fc["d2_har_log_rv"] = L5.har_log_rv_forecast(df, interval, cheap_refit_every, min_train)

    range_df = L.range_estimator_forecasts(df, window=bpd if bpd > 1 else 24)
    range_candidates = {name: range_df[col].to_numpy() for name, col in
                        [("parkinson", "fc_parkinson"), ("gk", "fc_gk"), ("rs", "fc_rs"), ("yz", "fc_yz")]}
    _, variance_fc["d3_range"] = min(range_candidates.items(), key=lambda kv: _q(kv[1]))

    variance_fc["d4_garch_normal"], fits_d4 = L.rolling_garch_forecast(
        ret, refit_every=mle_refit_every, min_train=min_train, innovation="normal", max_train=MLE_MAX_TRAIN,
    )
    variance_fc["d5_garch_t"], fits_d5 = L.rolling_garch_forecast(
        ret, refit_every=mle_refit_every, min_train=min_train, innovation="t", max_train=MLE_MAX_TRAIN,
    )
    nu_paths["d5_garch_t"] = L.nu_path_from_fits(fits_d5, n, param_index=3)

    variance_fc["d6_gjr_normal"], fits_d6 = L5.rolling_gjr_forecast(
        ret, refit_every=mle_refit_every, min_train=min_train, innovation="normal", max_train=MLE_MAX_TRAIN,
    )
    variance_fc["d7_gjr_t"], fits_d7 = L5.rolling_gjr_forecast(
        ret, refit_every=mle_refit_every, min_train=min_train, innovation="t", max_train=MLE_MAX_TRAIN,
    )
    nu_paths["d7_gjr_t"] = L.nu_path_from_fits(fits_d7, n, param_index=4)

    fits = {"d4": fits_d4, "d5": fits_d5, "d6": fits_d6, "d7": fits_d7}
    return variance_fc, nu_paths, fits


def score_gate_a_models(ret: np.ndarray, variance_fc: dict, nu_paths: dict) -> tuple[dict, dict]:
    """log_score_full (per-model, NaN-padded to len(ret)) and summary scores,
    exactly matching run_phase3_density.py's own scoring loop."""
    n = len(ret)
    log_score_full: dict[str, np.ndarray] = {}
    scores: dict[str, dict] = {}
    for name, fc in variance_fc.items():
        if name in T_MODEL_IDS:
            res = L5.vectorized_t_scores(ret, fc, nu_paths[name])
        else:
            res = L5.vectorized_normal_scores(ret, fc)
        mask, ls = res["mask"], res["log_score"]
        ls_full = np.full(n, np.nan)
        ls_full[mask] = ls
        log_score_full[name] = ls_full
        scores[name] = {"log_score_mean": float(np.nanmean(ls)), "n": int(mask.sum())}
    return log_score_full, scores


def all_pairs_dm_bh(model_names: list[str], log_score_full: dict, dm_bootstrap_n: int = 500, seed: int = 0) -> dict:
    """All-pairs Diebold-Mariano + BH adjustment on log-score loss
    differentials, identical machinery/convention to run_phase3_density.py's
    own inline loop, factored out here so Phase 1/Phase 3 drivers share one
    implementation rather than two drifting copies."""
    import itertools

    import research

    all_pairs_dm = {}
    normal_pvalues, boot_pvalues = {}, {}
    for a, b in itertools.combinations(model_names, 2):
        loss_a, loss_b = -log_score_full[a], -log_score_full[b]
        both = np.isfinite(loss_a) & np.isfinite(loss_b)
        if both.sum() < 30:
            continue
        d = (loss_a - loss_b)[both]
        tstat, p_normal = L.diebold_mariano(loss_a[both], loss_b[both])
        p_boot = research.block_bootstrap_pvalue(d, null_value=0.0, n_boot=dm_bootstrap_n, seed=seed)
        key = f"{a}_vs_{b}"
        all_pairs_dm[key] = {
            "a": a, "b": b, "tstat": tstat, "normal_pvalue": p_normal, "bootstrap_pvalue": p_boot,
            "log_score_a": float(np.nanmean(-loss_a[both])), "log_score_b": float(np.nanmean(-loss_b[both])),
            "n": int(both.sum()),
        }
        normal_pvalues[key] = p_normal
        boot_pvalues[key] = p_boot

    bh_normal = L5.benjamini_hochberg(normal_pvalues, alpha=0.05)
    bh_boot = L5.benjamini_hochberg(boot_pvalues, alpha=0.05)
    for key in all_pairs_dm:
        all_pairs_dm[key]["bh_normal"] = bh_normal[key]
        all_pairs_dm[key]["bh_bootstrap"] = bh_boot[key]
    return all_pairs_dm


def beats_all_significantly(best_name: str, model_names: list[str], all_pairs_dm: dict, bh_field: str) -> bool:
    """Same "does best beat every other competitor, significantly" check as
    run_phase3_density.py's own inline closure, factored out so Phase 1 and
    Phase 3 share the identical Gate A / Gate P decision logic."""
    for other in model_names:
        if other == best_name:
            continue
        key = f"{best_name}_vs_{other}" if f"{best_name}_vs_{other}" in all_pairs_dm else f"{other}_vs_{best_name}"
        entry = all_pairs_dm.get(key)
        if entry is None:
            continue
        a_is_best = entry["a"] == best_name
        bh = entry[bh_field]
        if not bh["significant"]:
            return False
        best_wins_sig = (a_is_best and entry["tstat"] < 0) or (not a_is_best and entry["tstat"] > 0)
        if not best_wins_sig:
            return False
    return True
