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


# --------------------------------------------------------------------------
# Phase 4: the violation process itself - count PMFs (Poisson vs. negative
# binomial, a boundary-nested LR test) and durations (geometric vs. discrete
# Weibull, the same boundary structure). Fit-once, descriptive machinery on
# the OOS evaluation period's own violation sequence (same status as Phase 1
# of notebook 5's Hill estimator: not a rolling forecast input, a diagnostic
# of what the forecasts' own failures look like), not new rolling-refit code.
# --------------------------------------------------------------------------

from scipy import stats as _st  # noqa: E402
from scipy.optimize import minimize as _minimize  # noqa: E402


def violation_blocks_and_durations(hit: np.ndarray, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    """hit: boolean/0-1 violation indicator, one entry per bar (already
    masked to finite/valid bars only by the caller). Returns:
      - counts: violation count in each non-overlapping block of `block_size`
        consecutive bars (the last partial block, if any, is dropped rather
        than padded, so every block has the same exposure).
      - durations: gaps between consecutive violations, in bars, coded as
        (bars since previous violation - 1) so the support starts at 0 -
        duration=0 means two violations landed on directly consecutive bars.
        Only defined from the first violation onward (no duration is coded
        for a hypothetical "time to first violation").
    """
    hit = np.asarray(hit).astype(bool)
    n = len(hit)
    n_blocks = n // block_size
    if n_blocks < 4:
        return np.array([]), np.array([])
    counts = hit[: n_blocks * block_size].reshape(n_blocks, block_size).sum(axis=1)
    hit_idx = np.where(hit)[0]
    if len(hit_idx) < 3:
        durations = np.array([])
    else:
        durations = np.diff(hit_idx) - 1
    return counts, durations


def _poisson_loglik(counts: np.ndarray, mu: float) -> float:
    return float(np.sum(_st.poisson.logpmf(counts, mu)))


def _nb2_negloglik(params: np.ndarray, counts: np.ndarray) -> float:
    log_mu, log_alpha = params
    mu, alpha = np.exp(log_mu), np.exp(log_alpha)
    # NB2: Var = mu + alpha*mu^2, r = 1/alpha, p = r/(r+mu)
    r = 1.0 / max(alpha, 1e-10)
    p = r / (r + mu)
    ll = float(np.sum(_st.nbinom.logpmf(counts, r, p)))
    if not np.isfinite(ll):
        return 1e10
    return -ll


def fit_poisson_counts(counts: np.ndarray) -> dict | None:
    if len(counts) < 4 or np.all(counts == 0):
        return None
    mu = float(np.mean(counts))
    if not np.isfinite(mu) or mu <= 0:
        return None
    return {"mu": mu, "loglik": _poisson_loglik(counts, mu), "n": len(counts)}


def fit_nb_counts(counts: np.ndarray) -> dict | None:
    """MLE (not method-of-moments) over (mu, alpha) in the NB2 dispersion
    parametrization (Var = mu + alpha*mu^2), alpha>=0, so that alpha=0 is
    exactly the Poisson boundary the LR test needs to be meaningful - a
    method-of-moments NB fit (as distributions._fit_nbinom uses, for
    rolling-refit speed reasons that don't apply to this one-shot fit) would
    not give a proper-MLE log-likelihood to compare against Poisson's own.
    """
    if len(counts) < 4 or np.all(counts == 0):
        return None
    mean, var = float(np.mean(counts)), float(np.var(counts))
    if mean <= 0:
        return None
    alpha0 = max((var - mean) / (mean**2), 1e-4) if var > mean else 1e-4
    x0 = np.array([np.log(mean), np.log(alpha0)])
    try:
        res = _minimize(_nb2_negloglik, x0, args=(counts,), method="Nelder-Mead",
                         options={"maxiter": 500, "xatol": 1e-8, "fatol": 1e-8})
    except Exception:
        return None
    if not res.success or not np.all(np.isfinite(res.x)):
        return None
    mu, alpha = float(np.exp(res.x[0])), float(np.exp(res.x[1]))
    ll = -float(res.fun)
    if not np.isfinite(ll):
        return None
    return {"mu": mu, "alpha": alpha, "loglik": ll, "n": len(counts)}


def boundary_lr_test(ll_full: float, ll_null: float) -> tuple[float, float]:
    """Likelihood-ratio test where the null sits on the boundary of the
    full model's parameter space (alpha=0 for NB-vs-Poisson; beta=1 for
    discrete-Weibull-vs-geometric is NOT this case - beta=1 is an interior
    point, so that comparison uses a plain chi2_1 test instead, see
    fit_discrete_weibull's own docstring).

    Self's-Chernoff (1954) boundary result: under the null, the LR statistic
    is distributed as a 50:50 mixture of a point mass at 0 and chi2_1, NOT
    a plain chi2_1 - using chi2_1 alone overstates significance (roughly
    halves the p-value), a classic and specifically-flagged error per
    NEXT_RUN_PROMPT.md section 4 Phase 4. p = 0.5 * chi2_1.sf(LR).
    """
    lr = max(0.0, 2.0 * (ll_full - ll_null))
    p = 0.5 * float(_st.chi2.sf(lr, df=1))
    return lr, p


def _discrete_weibull_negloglik(params: np.ndarray, durations: np.ndarray, fix_beta1: bool) -> float:
    if fix_beta1:
        (logit_q,) = params
        beta = 1.0
    else:
        logit_q, log_beta = params
        beta = np.exp(log_beta)
    q = 1.0 / (1.0 + np.exp(-logit_q))  # logit-transform to keep q in (0,1)
    k = durations.astype(float)
    # P(X=k) = q^(k^beta) - q^((k+1)^beta); compute in log-space via log(q)
    log_q = np.log(q)
    surv_k = np.exp(log_q * (k**beta))
    surv_k1 = np.exp(log_q * ((k + 1.0) ** beta))
    pmf = surv_k - surv_k1
    pmf = np.clip(pmf, 1e-300, None)
    ll = float(np.sum(np.log(pmf)))
    if not np.isfinite(ll):
        return 1e10
    return -ll


def fit_geometric_durations(durations: np.ndarray) -> dict | None:
    """Geometric on {0,1,2,...} (bars-since-last-violation - 1): P(X=k) =
    (1-q)*q^k. This is exactly the discrete-Weibull family at beta=1 fixed
    (its survival function q^k is the beta=1 special case of q^(k^beta)),
    so its own log-likelihood is directly comparable to the general
    discrete-Weibull fit via a plain LR test (beta=1 is an INTERIOR point of
    beta>0's parameter space, not a boundary, so no Chernoff mixture
    correction is needed here - unlike the NB-vs-Poisson alpha=0 case).
    """
    if len(durations) < 10:
        return None
    mean = float(np.mean(durations))
    if not np.isfinite(mean) or mean < 0:
        return None
    q = mean / (mean + 1.0)
    if not (0.0 < q < 1.0):
        return None
    ll = -_discrete_weibull_negloglik(np.array([np.log(q / (1 - q))]), durations, fix_beta1=True)
    return {"q": q, "beta": 1.0, "loglik": float(ll), "n": len(durations)}


def fit_discrete_weibull_durations(durations: np.ndarray) -> dict | None:
    """MLE over (q, beta) for the discrete Weibull (Nakagawa & Osaki 1975):
    survival P(X>k) = q^(k^beta). beta<1 means a FALLING hazard (violations
    cluster - once one has just happened, the next is more likely soon than
    the memoryless geometric null implies); beta>1 means a rising hazard.
    beta=1 nests the geometric exactly (see fit_geometric_durations).
    """
    if len(durations) < 10:
        return None
    geo = fit_geometric_durations(durations)
    if geo is None:
        return None
    x0 = np.array([np.log(geo["q"] / (1 - geo["q"])), 0.0])
    try:
        res = _minimize(_discrete_weibull_negloglik, x0, args=(durations, False), method="Nelder-Mead",
                         options={"maxiter": 1000, "xatol": 1e-8, "fatol": 1e-8})
    except Exception:
        return None
    if not res.success or not np.all(np.isfinite(res.x)):
        return None
    logit_q, log_beta = res.x
    q = 1.0 / (1.0 + np.exp(-logit_q))
    beta = float(np.exp(log_beta))
    ll = -float(res.fun)
    if not np.isfinite(ll) or not (0.0 < q < 1.0):
        return None
    return {"q": q, "beta": beta, "loglik": ll, "n": len(durations)}


# --------------------------------------------------------------------------
# Phase 3 contest: GARCH(1,1) with a Phase-3 zoo innovation family.
#
# Two-stage fit (variance recursion via dist_lib.fit_garch11(innovation=
# "normal") - unchanged, reused per rule 9 - then the zoo family's own
# MLE on the resulting standardized residuals), not a joint MLE over
# (omega,alpha,beta,shape) together. This is a deliberate scoping choice,
# not an oversight: a joint fit needs per-family box constraints/
# reparametrization (NIG's alpha>|beta| in particular) reimplemented in the
# rolling-refit hot loop, where a wrong constraint silently produces a
# non-stationary or ill-defined density every refit; two-stage reuses
# dist_lib's already-tested variance recursion unchanged and asks each zoo
# module's own already-tested `fit` to do only the one job it was built and
# verified (per-family, in tests/test_dist_lib6_*.py) to do well. Documented
# explicitly here and in the write-up so it is not confused with a joint fit.
# --------------------------------------------------------------------------


def fit_garch_zoo_two_stage(r: np.ndarray, family_module) -> dict | None:
    garch_fit = L.fit_garch11(r, innovation="normal")
    if garch_fit is None:
        return None
    omega, alpha, beta = garch_fit["omega"], garch_fit["alpha"], garch_fit["beta"]
    uncond = omega / max(1 - alpha - beta, 1e-6)
    sig2 = L._garch_variance_path(omega, alpha, beta, r, uncond)
    if np.any(sig2 <= 0) or not np.all(np.isfinite(sig2)):
        return None
    z = r / np.sqrt(sig2)
    shape = family_module.fit(z)
    if shape is None:
        return None
    next_sig2 = omega + alpha * r[-1] ** 2 + beta * sig2[-1]
    return {
        "omega": float(omega), "alpha": float(alpha), "beta": float(beta),
        "shape": tuple(float(s) for s in shape), "next_var": float(next_sig2),
        "family": family_module.NAME,
    }


def rolling_garch_forecast_zoo(
    returns: np.ndarray, refit_every: int, min_train: int, family_module, max_train: int = 500,
) -> tuple[np.ndarray, list[dict]]:
    """Structurally identical to dist_lib.rolling_garch_forecast /
    dist_lib5.rolling_gjr_forecast's own refit-then-forward-fill loop,
    generalized to fit_garch_zoo_two_stage's two-stage fit instead of a
    single-family MLE."""
    n = len(returns)
    forecast = np.full(n, np.nan)
    fits = []
    fit = None
    sig2_state = np.nan
    for t in range(n):
        if t >= min_train and t % refit_every == 0:
            start = max(0, t - max_train)
            window = returns[start:t]
            new_fit = fit_garch_zoo_two_stage(window, family_module)
            if new_fit is not None:
                fit = new_fit
                sig2_state = fit["omega"] / max(1 - fit["alpha"] - fit["beta"], 1e-6)
                fits.append({"t": t, **fit})
        if fit is not None:
            forecast[t] = sig2_state
            if t + 1 < n and np.isfinite(returns[t]):
                sig2_state = fit["omega"] + fit["alpha"] * returns[t] ** 2 + fit["beta"] * sig2_state
    return forecast, fits


def zoo_quantile_forecast(variance_forecast: np.ndarray, fits: list[dict], family_module, q: float) -> np.ndarray:
    """VaR forecast at level q for a zoo GARCH model: sigma_t * family_module.ppf(q,
    shape_t), forward-filled step-function shape exactly like score_zoo_model's own
    per-refit-segment convention (used by Phase 6's risk-limit overlay)."""
    n = len(variance_forecast)
    out = np.full(n, np.nan)
    if not fits:
        return out
    for i, f in enumerate(fits):
        start = f["t"]
        end = fits[i + 1]["t"] if i + 1 < len(fits) else n
        v = variance_forecast[start:end]
        mask = np.isfinite(v) & (v > 0)
        sigma = np.sqrt(v[mask])
        z_q = family_module.ppf(q, f["shape"])
        idx = np.arange(start, end)[mask]
        out[idx] = sigma * z_q
    return out


def score_zoo_model(actual: np.ndarray, variance_forecast: np.ndarray, fits: list[dict], family_module) -> np.ndarray:
    """Per-bar log score for a zoo GARCH model: family_module.logpdf(z,
    shape) - log(sigma) (change-of-variables from the unit-variance-
    standardized density to the actual-return scale, same convention as
    dist_lib._garch_negloglik's "- 0.5*log(sig2)" term). Evaluated in
    per-refit-segment batches (shape is a step function of t, forward-filled
    between refits, exactly like dist_lib.nu_path_from_fits) rather than a
    per-bar python loop, since every zoo family's logpdf is vectorized.
    """
    n = len(actual)
    log_score_full = np.full(n, np.nan)
    if not fits:
        return log_score_full
    for i, f in enumerate(fits):
        start = f["t"]
        end = fits[i + 1]["t"] if i + 1 < len(fits) else n
        a = actual[start:end]
        v = variance_forecast[start:end]
        mask = np.isfinite(a) & np.isfinite(v) & (v > 0)
        if not mask.any():
            continue
        sigma = np.sqrt(v[mask])
        z = a[mask] / sigma
        ll = family_module.logpdf(z, f["shape"]) - np.log(sigma)
        idx = np.arange(start, end)[mask]
        log_score_full[idx] = ll
    return log_score_full


# --------------------------------------------------------------------------
# Phase 5: a normalized, splicable semiparametric EVT density - the fix for
# notebook 5's own d8/d9 (GARCH-EVT, GJR-EVT), whose GPD-tails-plus-
# empirical-body density never entered the log-score contest because
# post-hoc normalization ("integrate then rescale the whole thing") proved
# fiddly. The fix here is structural, not iterative: build each of the three
# pieces (lower tail, interior, upper tail) as a SEPARATELY normalized
# density scaled by its own KNOWN weight (k/n for each tail, 1-2k/n for the
# interior) - total mass is then exactly 1 by construction (three weights
# summing to 1, each piece already integrating to 1 over its own support),
# with no post-hoc rescaling of the spliced whole ever needed.
# --------------------------------------------------------------------------

from scipy import integrate as _integrate  # noqa: E402
from scipy import stats as _st2  # noqa: E402


def fit_spliced_evt_density(z: np.ndarray, tail_frac: float = 0.10) -> dict | None:
    """Fit both GPD tails (dist_lib5.fit_gpd_tail) plus a Gaussian-KDE
    interior on the SAME training window's standardized residuals, and
    precompute the interior piece's own normalizing constant (a single
    scalar quad integral over the body range, not an iterative hunt).
    Returns None if either tail fit fails or the KDE degenerates.
    """
    z = z[np.isfinite(z)]
    n = len(z)
    if n < 100:
        return None
    lower_fit = L5.fit_gpd_tail(z, tail_frac=tail_frac, tail="lower")
    upper_fit = L5.fit_gpd_tail(z, tail_frac=tail_frac, tail="upper")
    if lower_fit is None or upper_fit is None:
        return None
    u_lower, u_upper = -lower_fit["u"], upper_fit["u"]  # thresholds in z-space
    if u_lower >= u_upper:
        return None
    k_lower, k_upper = lower_fit["n_exceed"], upper_fit["n_exceed"]
    w_lower, w_upper = k_lower / n, k_upper / n
    w_interior = 1.0 - w_lower - w_upper
    if w_interior <= 1e-6:
        return None

    body = z[(z >= u_lower) & (z <= u_upper)]
    if len(body) < 30 or np.std(body) <= 1e-10:
        return None
    try:
        kde = _st2.gaussian_kde(body, bw_method="silverman")
    except Exception:
        return None
    body_mass, _err = _integrate.quad(lambda x: kde(x)[0], u_lower, u_upper, limit=100)
    if not np.isfinite(body_mass) or body_mass <= 1e-8:
        return None

    return {
        "lower_fit": lower_fit, "upper_fit": upper_fit,
        "u_lower": float(u_lower), "u_upper": float(u_upper),
        "w_lower": float(w_lower), "w_upper": float(w_upper), "w_interior": float(w_interior),
        "kde": kde, "body_mass": float(body_mass), "n": n,
    }


def spliced_evt_logpdf(z: np.ndarray, spliced_fit: dict) -> np.ndarray:
    """Log density of the spliced EVT distribution, vectorized over z. Each
    of the three pieces is a properly normalized density (integrating to 1
    over its own support) times its own known weight - see
    fit_spliced_evt_density's own docstring for why this makes the total
    mass exactly 1 without ever normalizing the spliced whole post-hoc.
    """
    z = np.asarray(z, dtype=float)
    u_lower, u_upper = spliced_fit["u_lower"], spliced_fit["u_upper"]
    w_lower, w_upper, w_interior = spliced_fit["w_lower"], spliced_fit["w_upper"], spliced_fit["w_interior"]
    lower_fit, upper_fit = spliced_fit["lower_fit"], spliced_fit["upper_fit"]
    kde, body_mass = spliced_fit["kde"], spliced_fit["body_mass"]

    out = np.full(z.shape, np.nan)

    lower_mask = z < u_lower
    if lower_mask.any():
        xi, beta = lower_fit["xi"], lower_fit["beta"]
        y = -z[lower_mask] - (-u_lower)  # exceedance in y-space, y = -z, u = -u_lower
        support = 1.0 + xi * y / beta
        valid = support > 0
        dens = np.full(y.shape, 0.0)
        dens[valid] = (1.0 / beta) * support[valid] ** (-1.0 / xi - 1.0)
        with np.errstate(divide="ignore"):
            out[lower_mask] = np.log(w_lower) + np.log(np.where(dens > 0, dens, 1e-300))

    upper_mask = z > u_upper
    if upper_mask.any():
        xi, beta = upper_fit["xi"], upper_fit["beta"]
        y = z[upper_mask] - u_upper
        support = 1.0 + xi * y / beta
        valid = support > 0
        dens = np.full(y.shape, 0.0)
        dens[valid] = (1.0 / beta) * support[valid] ** (-1.0 / xi - 1.0)
        with np.errstate(divide="ignore"):
            out[upper_mask] = np.log(w_upper) + np.log(np.where(dens > 0, dens, 1e-300))

    interior_mask = ~lower_mask & ~upper_mask
    if interior_mask.any():
        kde_vals = kde(z[interior_mask])
        normalized = kde_vals / body_mass
        with np.errstate(divide="ignore"):
            out[interior_mask] = np.log(w_interior) + np.log(np.where(normalized > 0, normalized, 1e-300))

    return out


def rolling_spliced_evt_fits(
    returns: np.ndarray, variance_fits: list[dict], model: str, max_train: int, tail_frac: float = 0.10,
) -> list[dict]:
    """Same refit cadence/training-window convention as
    dist_lib5.rolling_gpd_paths (refit exactly when the underlying variance
    model refits, on the SAME training window), producing a list of
    {"t": ..., "spliced": fit_spliced_evt_density(...)} entries for
    forward-fill scoring via score_spliced_evt_model below."""
    fits_out = []
    for vf in variance_fits:
        t = vf["t"]
        start = max(0, t - max_train)
        window = returns[start:t]
        window = window[np.isfinite(window)]
        if len(window) < 60:
            continue
        sig2 = L5._variance_path_for_fit(vf, window, model)
        if np.any(sig2 <= 0) or not np.all(np.isfinite(sig2)):
            continue
        z = window / np.sqrt(sig2)
        spliced = fit_spliced_evt_density(z, tail_frac=tail_frac)
        if spliced is None:
            continue
        fits_out.append({"t": t, "spliced": spliced})
    return fits_out


def score_spliced_evt_model(actual: np.ndarray, variance_forecast: np.ndarray, spliced_fits: list[dict]) -> np.ndarray:
    """Per-bar log score for the spliced EVT density, same segment-batched
    forward-fill convention as score_zoo_model."""
    n = len(actual)
    log_score_full = np.full(n, np.nan)
    if not spliced_fits:
        return log_score_full
    for i, f in enumerate(spliced_fits):
        start = f["t"]
        end = spliced_fits[i + 1]["t"] if i + 1 < len(spliced_fits) else n
        a = actual[start:end]
        v = variance_forecast[start:end]
        mask = np.isfinite(a) & np.isfinite(v) & (v > 0)
        if not mask.any():
            continue
        sigma = np.sqrt(v[mask])
        z = a[mask] / sigma
        ll = spliced_evt_logpdf(z, f["spliced"]) - np.log(sigma)
        idx = np.arange(start, end)[mask]
        log_score_full[idx] = ll
    return log_score_full
