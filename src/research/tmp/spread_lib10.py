"""Notebook 10a/10b library: term-structure regime definitions, spread taxonomy,
cointegration/stationarity testing, and regime-conditional mean-reversion structure.

Reuses `commod_lib8.term_structure_state` (the F1->F2 roll-slope/regime primitive),
`research_lib9.ols_ar1_diff`/`zscore_ic` (the AR(1)-in-differences mean-reversion test),
and `research.block_bootstrap_ci`/`deflated_sharpe_prob` unmodified. Only genuinely new
machinery for notebooks 10a/10b lives here (NEXT_PROMPT.md sec 5).
"""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Spread taxonomy (NEXT_PROMPT.md sec 4.2). Classification is by leg
# products, read from each spread's own `leg_roles` column, not a hardcoded
# name list -- so a spread with an unexpected leg composition is caught
# rather than silently mis-filed.
# ---------------------------------------------------------------------------


def classify_spread_taxonomy(leg_products: list[str]) -> str:
    """ "calendar" if every leg is the same underlying product (the spread IS
    the term structure itself -- see sec 4.2), else "inter_commodity" (two or
    more distinct underlyings, e.g. crack/crush spreads and cross-product
    pairs). Requires at least 2 legs.
    """
    if len(leg_products) < 2:
        raise ValueError("a spread needs at least 2 legs")
    return "calendar" if len(set(leg_products)) == 1 else "inter_commodity"


# ---------------------------------------------------------------------------
# Cointegration / stationarity precondition (sec 4.3). No statsmodels
# dependency in this repo (research_lib9.py's own docstring), so the ADF
# regression and its critical values are implemented directly here rather
# than left untested (as notebook 9's Phase 4 first-look explicitly did).
# ---------------------------------------------------------------------------

# Asymptotic ADF critical values for the "constant, no trend" case (the
# relevant one for a spread series with a plausibly nonzero, but not
# deterministically trending, mean) -- MacKinnon (2010, "Critical Values for
# Cointegration Tests"), converging to the classic Fuller (1976) table as
# n -> infinity. Every spread series tested here has 2000+ observations, well
# into the range where the asymptotic values are indistinguishable from the
# finite-sample response-surface correction to the precision reported.
ADF_CRITICAL_VALUES = {"1%": -3.43, "5%": -2.86, "10%": -2.57}


def _select_adf_lag(dv: np.ndarray, v_lag: np.ndarray, max_lag: int) -> int:
    """Choose the number of augmenting lagged-difference terms 0..max_lag by
    BIC, the standard automatic lag-selection rule for ADF (Ng-Perron 2001
    discuss AIC vs BIC; BIC is used here as the more parsimonious of the two,
    appropriate given these series show no reason to expect long memory).
    """
    n = len(dv)
    best_lag, best_bic = 0, np.inf
    for lag in range(max_lag + 1):
        if n - lag < 20:
            break
        y = dv[lag:]
        cols = [np.ones_like(y), v_lag[lag:]]
        for i in range(1, lag + 1):
            cols.append(dv[lag - i : n - i])
        X = np.column_stack(cols)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        k = X.shape[1]
        n_eff = len(y)
        sigma2 = float(np.sum(resid**2) / n_eff)
        bic = n_eff * np.log(sigma2) + k * np.log(n_eff)
        if bic < best_bic:
            best_bic, best_lag = bic, lag
    return best_lag


def adf_test(x: np.ndarray, max_lag: int | None = None) -> dict:
    """Augmented Dickey-Fuller test for a unit root, constant-only case:
    delta_v_t = a + b*v_{t-1} + sum_i g_i * delta_v_{t-i} + eps_t.

    H0: b == 0 (unit root, non-stationary -- an uncointegrated spread that
    can drift arbitrarily far, sec 4.3). H1: b < 0 (stationary / mean-
    reverting). The test statistic is b's own OLS t-stat, compared against
    `ADF_CRITICAL_VALUES` (NOT a standard-normal/t critical value -- the
    Dickey-Fuller distribution is skewed left under H0, which is exactly why
    a tabulated critical value, not a t-table lookup, is required).

    Engle-Granger two-step cointegration testing runs exactly this ADF test
    on the residual of a static OLS regression between two legs; every spread
    series here (`value` column) already IS that residual -- `hedge_ratio` is
    precomputed upstream -- so this function doubles as both the plain
    stationarity test (sec 4.3) and the Engle-Granger cointegration test on
    the two legs, with no separate first-stage regression needed.
    """
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 30:
        return {
            "t_stat": np.nan,
            "n_obs": n,
            "n_lags": 0,
            "stationary_5pct": False,
            "stationary_1pct": False,
        }
    if max_lag is None:
        max_lag = min(20, int(12 * (n / 100) ** 0.25))

    v_lag_full = v[:-1]
    dv_full = np.diff(v)
    n_lags = _select_adf_lag(dv_full, v_lag_full, max_lag)

    y = dv_full[n_lags:]
    v_lag = v_lag_full[n_lags:]
    cols = [np.ones_like(y), v_lag]
    for i in range(1, n_lags + 1):
        cols.append(dv_full[n_lags - i : len(dv_full) - i])
    X = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n_eff, k = X.shape
    dof = n_eff - k
    sigma2 = float(resid @ resid) / dof if dof > 0 else np.nan
    xtx_inv = np.linalg.inv(X.T @ X)
    se_b = float(np.sqrt(sigma2 * xtx_inv[1, 1])) if np.isfinite(sigma2) else np.nan
    b = float(coef[1])
    t_stat = b / se_b if se_b and se_b > 0 else np.nan

    return {
        "t_stat": t_stat,
        "b": b,
        "n_obs": n_eff,
        "n_lags": n_lags,
        "critical_values": ADF_CRITICAL_VALUES,
        "stationary_5pct": bool(
            np.isfinite(t_stat) and t_stat < ADF_CRITICAL_VALUES["5%"]
        ),
        "stationary_1pct": bool(
            np.isfinite(t_stat) and t_stat < ADF_CRITICAL_VALUES["1%"]
        ),
    }


# ---------------------------------------------------------------------------
# Term-structure regime definitions (sec 4.1). Three variants, declared in
# advance and capped at three -- every one tried enters the DSR count. The
# raw-sign variant is `commod_lib8.term_structure_state`'s own default label
# and is used unmodified as variant (i); (ii) and (iii) are new here.
# ---------------------------------------------------------------------------

# Deadband threshold for variant (ii): a fixed, pre-declared 2%/year
# annualised roll-slope magnitude, below which the state is "flat" and no
# trade fires. Chosen as a round, pre-declared convention -- roughly the
# scale of a typical financing-rate-only contango (see docs/09's cost-of-
# carry worked example, ~5%/year) -- and NOT swept or tuned against any
# regime-conditional result computed later in this notebook.
REGIME_DEADBAND_ANNUALIZED = 0.02

# Persistence requirement for variant (iii): the raw-sign state must hold for
# this many consecutive trading days before being treated as "confirmed".
REGIME_PERSISTENCE_DAYS = 5


def regime_deadband(
    roll_slope_annualized: pl.Series, deadband: float = REGIME_DEADBAND_ANNUALIZED
) -> pl.Series:
    """Variant (ii): raw sign, but with a magnitude deadband -- |slope| below
    `deadband` is labelled "flat" (no trade fires) rather than assigned to
    whichever side of zero it happens to be on."""
    df = pl.DataFrame({"slope": roll_slope_annualized})
    return df.select(
        pl.when(pl.col("slope").abs() < deadband)
        .then(pl.lit("flat"))
        .when(pl.col("slope") < 0)
        .then(pl.lit("backwardation"))
        .when(pl.col("slope") > 0)
        .then(pl.lit("contango"))
        .otherwise(None)
        .alias("regime")
    )["regime"]


def regime_persistent(
    state: pl.Series, n_days: int = REGIME_PERSISTENCE_DAYS
) -> pl.Series:
    """Variant (iii): the raw-sign state (contango/backwardation/None) is
    only "confirmed" once it has held for `n_days` consecutive observations;
    until then (and on any day where the underlying raw state is null), the
    label is "unconfirmed" -- a deliberately more conservative, lower-
    turnover regime signal than the raw sign alone.
    """
    s = state.to_list()
    n = len(s)
    out = ["unconfirmed"] * n
    run_val, run_len = None, 0
    for i in range(n):
        cur = s[i]
        if cur is not None and cur == run_val:
            run_len += 1
        else:
            run_val, run_len = cur, 1
        if cur is not None and run_len >= n_days:
            out[i] = cur
    return pl.Series("term_structure_state_persistent", out)


# ---------------------------------------------------------------------------
# Rolling leg correlation and regime-conditional structure (sec 5 Phase 2/3).
# ---------------------------------------------------------------------------


def rolling_leg_correlation(
    leg1_returns: np.ndarray, leg2_returns: np.ndarray, window: int = 60
) -> np.ndarray:
    """Rolling Pearson correlation between two legs' log returns -- the
    diagnostic for leg decoupling (sec 7: "this is the chart that directly
    addresses the operator's hypothesis"). NaN wherever fewer than `window`
    finite observations are available in the trailing window.
    """
    r1 = pl.Series("r1", leg1_returns)
    r2 = pl.Series("r2", leg2_returns)
    both_finite = r1.is_finite() & r2.is_finite()
    df = pl.DataFrame({"r1": r1, "r2": r2}).with_columns(
        pl.when(both_finite).then(pl.col("r1")).otherwise(None).alias("r1"),
        pl.when(both_finite).then(pl.col("r2")).otherwise(None).alias("r2"),
    )
    out = df.select(
        pl.rolling_corr(
            pl.col("r1"), pl.col("r2"), window_size=window, min_samples=window
        ).alias("corr")
    )
    return out["corr"].to_numpy()


def regime_conditional_ar1(
    values: np.ndarray, regime_labels: list[str] | np.ndarray
) -> dict:
    """`research_lib9.ols_ar1_diff` run separately within each regime label
    present, plus the pooled (unconditional) fit -- the direct test of sec
    4.1's core question ("does the spread mean-revert harder in one regime
    than another?"), min 60 observations per state to attempt a fit.
    """
    import research_lib9 as R9

    v = np.asarray(values, dtype=float)
    labels = np.asarray(regime_labels, dtype=object)
    mask = np.isfinite(v)
    v, labels = v[mask], labels[mask]

    out: dict = {}
    for state in sorted({s for s in labels.tolist() if s is not None}):
        sel = v[labels == state]
        if len(sel) < 60:
            out[state] = {"n": len(sel), "fit": None}
            continue
        out[state] = {"n": len(sel), "fit": R9.ols_ar1_diff(sel)}
    out["_pooled"] = {"n": len(v), "fit": R9.ols_ar1_diff(v) if len(v) >= 60 else None}
    return out


def regime_conditional_vol(
    returns: np.ndarray, regime_labels: list[str] | np.ndarray
) -> dict:
    """Annualised realised vol of the spread's own return series, split by
    regime label -- the volatility half of sec 5 Phase 3's regime-conditional
    structure question."""
    r = np.asarray(returns, dtype=float)
    labels = np.asarray(regime_labels, dtype=object)
    mask = np.isfinite(r)
    r, labels = r[mask], labels[mask]
    out: dict = {}
    for state in sorted({s for s in labels.tolist() if s is not None}):
        sel = r[labels == state]
        out[state] = {
            "n": len(sel),
            "vol_annualized": float(np.std(sel) * np.sqrt(252))
            if len(sel) >= 20
            else None,
        }
    out["_pooled"] = {
        "n": len(r),
        "vol_annualized": float(np.std(r) * np.sqrt(252)) if len(r) >= 20 else None,
    }
    return out


def regime_state_persistence(state: pl.Series) -> dict:
    """Mean run length per state label and the state-to-state transition
    matrix -- sec 5 Phase 1's regime atlas persistence characterisation.
    Nulls are dropped before computing runs (a null day contributes to
    neither state's run).
    """
    s = [x for x in state.to_list() if x is not None]
    if not s:
        return {"mean_run_length": {}, "transition_matrix": {}}
    runs: dict[str, list[int]] = {}
    cur_val, cur_len = s[0], 1
    for x in s[1:]:
        if x == cur_val:
            cur_len += 1
        else:
            runs.setdefault(cur_val, []).append(cur_len)
            cur_val, cur_len = x, 1
    runs.setdefault(cur_val, []).append(cur_len)
    mean_run = {k: float(np.mean(v)) for k, v in runs.items()}

    states = sorted(set(s))
    trans = {a: {b: 0 for b in states} for a in states}
    for a, b in itertools.pairwise(s):
        trans[a][b] += 1
    trans_frac: dict[str, dict[str, float]] = {}
    for a in states:
        total = sum(trans[a].values())
        trans_frac[a] = {b: (trans[a][b] / total if total > 0 else 0.0) for b in states}
    return {"mean_run_length": mean_run, "transition_matrix": trans_frac}


# ---------------------------------------------------------------------------
# COT positioning (sec 5 Phase 4, optional). This repo's cache holds exactly
# one CFTC series (067651 = CL light sweet crude, NYMEX) -- see
# docs/09-market-data-and-microstructure.md's own "Hedging pressure and the
# COT report" pitfall. CL-only, never extrapolated into a panel claim.
# ---------------------------------------------------------------------------


def cot_net_noncomm_fraction(cot: pl.DataFrame) -> pl.DataFrame:
    """Weekly net non-commercial position as a fraction of open interest,
    lagged to its public release convention (as-of Tuesday, released Friday
    -- lag the report_date by >=3 calendar days before ever joining it onto a
    daily signal, so no daily bar ever sees a COT print before it was public).
    """
    out = cot.select(
        [
            pl.col("report_date").cast(pl.Date).alias("report_date"),
            pl.col("noncomm_positions_long_all").alias("noncomm_long"),
            pl.col("noncomm_positions_short_all").alias("noncomm_short"),
            pl.col("open_interest_all").alias("open_interest"),
        ]
    ).sort("report_date")
    out = out.with_columns(
        (
            (pl.col("noncomm_long") - pl.col("noncomm_short"))
            / pl.col("open_interest").clip(lower_bound=1)
        ).alias("net_noncomm_frac")
    )
    out = out.with_columns(
        (pl.col("report_date") + pl.duration(days=3)).alias("public_date")
    )
    return out.select(
        ["report_date", "public_date", "net_noncomm_frac", "open_interest"]
    )
