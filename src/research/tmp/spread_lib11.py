"""Notebook 11a library: ported evaluation machinery from the external
`~/Documents/ultron/apps/trading-labs` programme, reimplemented against this
repo's own data, cost model and statistics (NEXT_PROMPT.md sec 3, "do not
import their code"). Reuses `spread_lib10.adf_test`, `research_lib9.ols_ar1_diff`
and `research.block_bootstrap_ci`/`deflated_sharpe_prob` unmodified; only
genuinely new machinery for notebook 11 lives here.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import polars as pl

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import research_lib9 as R9
import spread_lib10 as S10


def approx_adf_pvalue(t_stat: float) -> float:
    """Approximate one-sided ADF p-value via linear interpolation/
    extrapolation against `spread_lib10.ADF_CRITICAL_VALUES` (1%/5%/10%
    only -- `adf_test` reports no p-value directly). This is a crude
    monotone approximation, not the exact Dickey-Fuller CDF, adequate only
    for the reentry gate's threshold comparison against `adf_pmax`, not for
    any reported inferential claim.
    """
    cv = S10.ADF_CRITICAL_VALUES
    xs = [cv["1%"], cv["5%"], cv["10%"]]
    ys = [0.01, 0.05, 0.10]
    if t_stat <= xs[0]:
        # extrapolate linearly below 1%, floored at a small positive value
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return float(max(1e-4, ys[0] + slope * (t_stat - xs[0])))
    if t_stat >= xs[2]:
        slope = (ys[2] - ys[1]) / (xs[2] - xs[1])
        return float(min(1.0, ys[2] + slope * (t_stat - xs[2])))
    return float(np.interp(t_stat, xs, ys))


# ---------------------------------------------------------------------------
# Phase 1 -- ported primitives (NEXT_PROMPT.md sec 3 Phase 1)
# ---------------------------------------------------------------------------


def compute_zscore(spread: np.ndarray, lookback: int) -> np.ndarray:
    """Rolling z-score of `spread` over `lookback` bars, shift(1): bar t's
    z-score uses only bars strictly before t (their no-lookahead guarantee).
    """
    s = pl.Series(np.asarray(spread, dtype=float))
    mean = s.rolling_mean(window_size=lookback)
    std = s.rolling_std(window_size=lookback)
    z = ((s - mean) / std).shift(1)
    return z.to_numpy()


def compute_atr_series(spread: np.ndarray, window: int = 14) -> np.ndarray:
    """Rolling std of the spread's own day-over-day change, shift(1) --
    named "atr" only because that is what the external programme's config
    calls it. This is NOT a true ATR: spreads have no OHLC, so there is no
    daily range to average. It approximates the same thing an ATR is used
    for here (a stop-distance / position-sizing volatility scale) via the
    std of daily changes over `window` bars, which is the same quantity
    `filters.compute_filter_masks`'s `vol` column uses upstream.
    """
    s = pl.Series(np.asarray(spread, dtype=float))
    atr = s.diff().rolling_std(window_size=window).shift(1)
    return atr.to_numpy()


class FixedFractionalSizing:
    """`qty = floor(equity * risk_pct / (max(atr, min_atr) * stop_atr_mult))`,
    capped at `floor(max_leverage * equity / |price|)`.
    """

    @staticmethod
    def quantity(
        equity: float,
        atr: float,
        price: float,
        risk_pct: float,
        stop_atr_mult: float,
        min_atr: float,
        max_leverage: float,
    ) -> int:
        if not np.isfinite(atr) or not np.isfinite(price) or price == 0 or equity <= 0:
            return 0
        atr_eff = max(atr, min_atr)
        risk_qty = np.floor(equity * risk_pct / (atr_eff * stop_atr_mult))
        leverage_qty = np.floor(max_leverage * equity / abs(price))
        return int(max(0.0, min(risk_qty, leverage_qty)))


STORAGE_LOW = 0.30
STORAGE_MID = 0.45
STORAGE_HIGH = 0.60
FINANCING_RATE_APPROX = 0.05


def compute_carry_fv(
    leg2_price: float, storage_per_month: float, financing_rate: float
) -> float:
    """Theoretical full-carry fair value of a calendar spread's deviation:
    `-(storage_per_month + financing_rate * leg2_price / 12)`. Negative,
    reflecting that storage + financing cost always pushes the deferred leg
    above the front leg under full carry.
    """
    return -(storage_per_month + financing_rate * leg2_price / 12.0)


def carry_ratio(
    value: np.ndarray | float, full_carry: np.ndarray | float
) -> np.ndarray | float:
    """`c_t = -value_t / full_carry_t`, as specified in NEXT_PROMPT.md sec 3
    Phase 1.

    NOTE (flagged, not silently corrected): under this repo's leg1=front
    convention (`value = leg1_price - leg2_price`, verified to match theirs
    -- backwardation is value > 0, contango is value < 0, corroborated
    empirically against `brent_calendar`'s own `ts_regime` column), this
    literal formula evaluates to approximately -1 at the deep-contango
    "full carry" boundary (value ~= full_carry, both negative) and to a
    positive number in backwardation -- the OPPOSITE of the "+1 at the
    contango ceiling, negative in backwardation" description in
    NEXT_PROMPT.md sec 3 Phase 1. The formula is implemented literally as
    specified; the sign direction is left for 11b (the first notebook to
    actually consume `carry_ratio` for Gate BF) to resolve against the
    external repo's live carry-ratio output on a shared date, since 11a
    makes no gate verdicts and this discrepancy does not affect anything
    computed here.
    """
    scalar_input = np.isscalar(value) and np.isscalar(full_carry)
    value_arr = np.asarray(value, dtype=float)
    full_carry_arr = np.asarray(full_carry, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = -value_arr / full_carry_arr
    return float(ratio) if scalar_input else ratio


def label_ts_regime(spread_value: np.ndarray, flat_band: float) -> np.ndarray:
    """Their calendar-spread regime label: "backwardation" if
    spread_value > flat_band, "contango" if spread_value < -flat_band, else
    "flat". Matches this repo's own leg1-front sign convention (verified
    against `brent_calendar`'s pre-existing `ts_regime` column: value > 0
    rows are all labelled "backwardation" there, value < 0 rows "contango").
    """
    spread_value = np.asarray(spread_value, dtype=float)
    out = np.full(spread_value.shape, "flat", dtype=object)
    out[spread_value > flat_band] = "backwardation"
    out[spread_value < -flat_band] = "contango"
    out[~np.isfinite(spread_value)] = "unknown"
    return out


def variance_ratio(x: np.ndarray, q: int, window: int | None = None) -> dict:
    """Lo-MacKinlay variance ratio: VR(q) = Var(q-period returns) /
    (q * Var(1-period returns)), on the last `window` observations of `x`
    (or all of `x` if window is None). x is a level series (e.g. spread
    value); 1-period returns are its first difference.

    Under the random-walk null, VR(q) = 1. The homoscedastic-null z-stat
    (Lo & MacKinlay 1988 eq. 10) is
    z = (VR(q) - 1) / sqrt(2 * (2q - 1) * (q - 1) / (3 * q * n)),
    asymptotically N(0, 1); a one-sided test against mean reversion rejects
    the random walk when z < -1.645 (VR(q) significantly below 1).
    """
    x = np.asarray(x, dtype=float)
    if window is not None:
        x = x[-window:]
    x = x[np.isfinite(x)]
    n = len(x) - 1
    if n < 2 * q:
        return {"vr": float("nan"), "z_stat": float("nan"), "n": n}
    r1 = np.diff(x)
    var1 = np.var(r1, ddof=1)
    rq = x[q:] - x[:-q]
    varq = np.var(rq, ddof=1) / q
    vr = float(varq / var1) if var1 > 0 else float("nan")
    se = np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * n))
    z_stat = float((vr - 1) / se) if se > 0 and np.isfinite(vr) else float("nan")
    return {"vr": vr, "z_stat": z_stat, "n": int(n)}


def hurst_exponent(
    x: np.ndarray, min_lag: int = 2, max_lag: int | None = None
) -> float:
    """Hurst exponent via the variance-of-lagged-differences method: for a
    range of lags k, Var(x[t] - x[t-k]) ~ k^(2H) for a self-affine series,
    so H is (half) the slope of log(Var(diff_k)) regressed on log(k).
    H < 0.5 indicates mean reversion, H = 0.5 a random walk, H > 0.5
    trending/persistent.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if max_lag is None:
        max_lag = max(min_lag + 1, min(100, n // 4))
    lags = range(min_lag, max_lag)
    tau = []
    used_lags = []
    for lag in lags:
        diffs = x[lag:] - x[:-lag]
        if len(diffs) < 2:
            continue
        var = np.var(diffs)
        if var > 0:
            tau.append(var)
            used_lags.append(lag)
    if len(used_lags) < 2:
        return float("nan")
    log_lags = np.log(used_lags)
    log_tau = np.log(tau)
    slope, _ = np.polyfit(log_lags, log_tau, 1)
    return float(slope / 2.0)


def rolling_half_life(x: np.ndarray, window: int) -> np.ndarray:
    """Half-life (days) of `x`'s own AR(1)-in-differences mean reversion
    (`research_lib9.ols_ar1_diff`), computed on non-overlapping chunks of
    `window` consecutive observations rather than bar-by-bar (the OLS fit
    is unstable on windows much smaller than a few multiples of the true
    half-life, so a rolling *chunk* estimate, not a rolling single-bar
    estimate, is the honest granularity). Returns one value per chunk;
    NaN where the chunk's beta is not in (-1, 0) (not mean-reverting) or
    the chunk is too short to fit.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n_chunks = len(x) // window
    out = np.full(n_chunks, np.nan)
    for i in range(n_chunks):
        chunk = x[i * window : (i + 1) * window]
        if len(chunk) < 10:
            continue
        fit = R9.ols_ar1_diff(chunk)
        hl = fit.get("half_life_days")
        out[i] = hl if hl is not None else np.nan
    return out


def rolling_adf_stat(x: np.ndarray, window: int) -> np.ndarray:
    """ADF t-statistic of `x` on the same non-overlapping chunking as
    `rolling_half_life`, reusing `spread_lib10.adf_test` unmodified.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n_chunks = len(x) // window
    out = np.full(n_chunks, np.nan)
    for i in range(n_chunks):
        chunk = x[i * window : (i + 1) * window]
        if len(chunk) < 60:
            continue
        out[i] = S10.adf_test(chunk)["t_stat"]
    return out


def rolling_stability(
    x: np.ndarray, n_subperiods: int = 4, hl_band: tuple[float, float] = (3.0, 60.0)
) -> dict:
    """Fraction of `n_subperiods` contiguous, equal-length sub-periods of
    `x` whose own AR(1) half-life falls inside `hl_band`, alongside the
    full-sample half-life. Used by the new screen (Phase 3): full-sample
    half-life in-band AND at least 3 of 4 sub-periods in-band.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    window = len(x) // n_subperiods
    sub_hls = rolling_half_life(x, window) if window >= 10 else np.array([])
    in_band = np.array(
        [np.isfinite(hl) and hl_band[0] <= hl <= hl_band[1] for hl in sub_hls]
    )
    full_fit = R9.ols_ar1_diff(x) if len(x) >= 30 else None
    full_hl = full_fit.get("half_life_days") if full_fit else None
    full_in_band = full_hl is not None and hl_band[0] <= full_hl <= hl_band[1]
    return {
        "full_sample_half_life": full_hl,
        "full_sample_in_band": bool(full_in_band),
        "n_subperiods": len(sub_hls),
        "n_subperiods_in_band": int(in_band.sum()),
        "sub_half_lives": [float(h) if np.isfinite(h) else None for h in sub_hls],
        "stable": bool(full_in_band and in_band.sum() >= 3),
    }


# ---------------------------------------------------------------------------
# Phase 2 -- their evaluation harness (NEXT_PROMPT.md sec 3 Phase 2)
# ---------------------------------------------------------------------------


def pnl_atr(realized_pnl: float, quantity: float, atr_at_entry: float) -> float:
    """Per-trade edge normalized by entry volatility, path-independent."""
    if quantity == 0 or not np.isfinite(atr_at_entry) or atr_at_entry == 0:
        return float("nan")
    return realized_pnl / (quantity * atr_at_entry)


def ret_eq(realized_pnl: float, equity_at_open: float) -> float:
    """Fixed-notional trade return: realized_pnl / equity strictly BEFORE
    the trade's opening day (guards the lookahead bug their v4 integration
    audit found -- a same-day post-fill equity snapshot). Summed
    arithmetically across trades this gives a fixed-notional return series
    invariant to compounding.
    """
    if equity_at_open == 0:
        return float("nan")
    return realized_pnl / equity_at_open


def trade_blocks(dates: np.ndarray, freq: str = "1q") -> np.ndarray:
    """Assign each trade's date to a contiguous calendar block (quarter by
    default) for the paired block bootstrap -- an integer block id per
    trade, monotone in time.
    """
    d = pl.Series(np.asarray(dates))
    return d.dt.truncate(freq).to_numpy()


def paired_block_bootstrap(
    control_pnl: np.ndarray,
    control_blocks: np.ndarray,
    treatment_pnl: np.ndarray,
    treatment_blocks: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Paired block bootstrap for the (treatment - control) delta.

    Groups both control's and treatment's trades into contiguous calendar
    blocks (via `trade_blocks`), resamples the SHARED set of block ids with
    replacement, and evaluates control and treatment on the SAME resampled
    blocks each draw -- so the shared price path that produced both books
    cancels in the delta and only the configuration disagreement between
    them remains. Returns the delta's point estimate and CI, plus each
    side's own bootstrap distribution (for the noise-floor report).
    """
    all_blocks = np.unique(np.concatenate([control_blocks, treatment_blocks]))
    rng = np.random.default_rng(seed)
    n_blocks = len(all_blocks)
    control_sum_by_block = {
        b: control_pnl[control_blocks == b].sum() for b in all_blocks
    }
    treatment_sum_by_block = {
        b: treatment_pnl[treatment_blocks == b].sum() for b in all_blocks
    }
    control_boot = np.empty(n_boot)
    treatment_boot = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(all_blocks, size=n_blocks, replace=True)
        control_boot[i] = sum(control_sum_by_block[b] for b in sample)
        treatment_boot[i] = sum(treatment_sum_by_block[b] for b in sample)
    delta_boot = treatment_boot - control_boot
    alpha = (1 - ci) / 2
    delta_lo, delta_hi = np.quantile(delta_boot, [alpha, 1 - alpha])
    control_lo, control_hi = np.quantile(control_boot, [alpha, 1 - alpha])
    return {
        "control_point": float(control_pnl.sum()),
        "treatment_point": float(treatment_pnl.sum()),
        "delta_point": float(treatment_pnl.sum() - control_pnl.sum()),
        "delta_ci": [float(delta_lo), float(delta_hi)],
        "delta_excludes_zero": bool(delta_lo > 0 or delta_hi < 0),
        "control_ci": [float(control_lo), float(control_hi)],
        "n_blocks": int(n_blocks),
    }


def noise_floor(
    ret_eq_values: np.ndarray,
    blocks: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap the control book's own fixed-notional return (sum of
    `ret_eq` per trade, matching `book_metrics`'s `fixed_notional_return`)
    alone, block-resampled the same way as `paired_block_bootstrap`, to get
    its own interval width -- the "noise floor" stating what any comparison
    against this control can possibly resolve. Reported the way their own
    v4 report it: point and CI as returns (e.g. +0.851 = +85.1%), half-width
    in percentage POINTS of return (their "+-41.9pp"), not a percentage of
    the point estimate (which blows up near a zero point).
    """
    all_blocks = np.unique(blocks)
    rng = np.random.default_rng(seed)
    n_blocks = len(all_blocks)
    sum_by_block = {b: ret_eq_values[blocks == b].sum() for b in all_blocks}
    boot = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(all_blocks, size=n_blocks, replace=True)
        boot[i] = sum(sum_by_block[b] for b in sample)
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    point = float(ret_eq_values.sum())
    return {
        "point_return": point,
        "ci_return": [float(lo), float(hi)],
        "half_width_pp": float((hi - lo) / 2 * 100),
    }


# ---------------------------------------------------------------------------
# Phase 4 -- the trading rule and backtest engine (NEXT_PROMPT.md sec 3
# Phase 4, sec 4.1). A single-position-per-spread, daily-bar, ATR-sized,
# stopped mean-reversion book. Reimplemented from the external repo's
# strategy.py/filters.py/config.yaml *specification*, not its code.
# ---------------------------------------------------------------------------

POINT_VALUE = {
    "CL": 1000.0,
    "BZ": 1000.0,
    "NG": 10000.0,
    "HO": 42000.0,
    "RB": 42000.0,
    "GC": 100.0,
    "SI": 5000.0,
    "PL": 50.0,
    "PA": 100.0,
    "ZC": 5000.0,
    "ZW": 5000.0,
    "KE": 5000.0,
    "ZS": 5000.0,
    "ZL": 60000.0,
    "ZM": 100.0,
    "ES": 50.0,
}


@dataclass
class TradingRuleParams:
    entry_threshold: float = 2.0
    exit_threshold: float = 0.75
    lookback: int = 60
    stop_atr_mult: float = 6.0
    risk_pct: float = 0.03
    atr_window: int = 14
    min_atr: float = 0.10
    max_leverage: float = 5.0
    cooldown_days: int = 10
    episode_window_days: int = 30
    max_per_episode: int = 1
    half_life_max: float = 45.0
    adf_pmax: float = 0.10
    vol_pctile: float = 0.75
    vol_regime_pctile: float = 0.90
    liquidity_pctile: float = 0.10
    vol_window: int = 20
    percentile_window: int = 252
    max_single_name_pct: float = 12.0
    max_gross_exposure_pct: float = 100.0
    daily_drawdown_limit_pct: float = 3.0
    overnight_concentration_max_pct: float = 30.0


def _suppression_masks(
    value: np.ndarray, roll_window_flag: np.ndarray, p: TradingRuleParams
) -> dict:
    """Vectorized reproduction of `filters.compute_filter_masks`: liquidity
    is always all-False here (no per-spread volume series in this repo's
    spread parquets, same fail-open behaviour their own code documents for
    a missing volume column).
    """
    s = pl.Series(np.asarray(value, dtype=float))
    vol = s.diff().rolling_std(
        window_size=p.vol_window, min_samples=max(1, p.vol_window // 2)
    )
    vol_pct = vol.rolling_map(
        lambda w: (
            float((w.rank()[-1] - 1) / (len(w) - 1)) if len(w) > 1 else float("nan")
        ),
        window_size=p.percentile_window,
        min_samples=max(1, p.percentile_window // 4),
    )
    vol_pct_np = vol_pct.to_numpy()
    roll_flag = np.asarray(roll_window_flag, dtype=bool)
    roll_vol_suppress = roll_flag & (vol_pct_np > p.vol_pctile)
    vol_regime_suppress = vol_pct_np > p.vol_regime_pctile
    any_suppress = roll_vol_suppress | vol_regime_suppress
    return {"any_suppress": any_suppress, "vol_pct": vol_pct_np}


def simulate_single_spread(
    df: pl.DataFrame,
    p: TradingRuleParams,
    cost_per_contract: float,
    stop_atr_mult_override: float | None = None,
    require_ts_regime: str | None = None,
    start_equity: float = 1_000_000.0,
    sign_flip: bool = False,
    disable_stop: bool = False,
) -> dict:
    """Simulate the pre-registered trading rule on one spread's own
    equity slice (single-position-per-spread; the shared-portfolio-equity
    sizing and cross-spread risk caps are applied by `simulate_book`, which
    calls this once per spread and rescales). Returns raw per-spread trades
    and a daily equity-delta series (in dollars), computed as if this spread
    alone owned `start_equity`.

    `sign_flip`: reverse entry direction (Gate BF, sec 4.2) -- long the
    spread when z > entry_threshold instead of < -entry_threshold, and vice
    versa. `disable_stop`: never check the stop breach (Gate TS-S).
    """
    stop_mult = (
        stop_atr_mult_override
        if stop_atr_mult_override is not None
        else p.stop_atr_mult
    )
    dates = df["date"].to_numpy()
    value = df["value"].to_numpy().astype(float)
    leg1 = df["leg1_price"].to_numpy().astype(float)
    leg2 = df["leg2_price"].to_numpy().astype(float)
    leg_products = [r["product"] for r in df["leg_roles"][0]]
    point_value = POINT_VALUE[leg_products[0]]
    roll_flag = df["roll_window_flag"].to_numpy()
    ts_regime = df["ts_regime"].to_numpy()

    z = compute_zscore(value, p.lookback)
    atr = compute_atr_series(value, p.atr_window)
    suppress = _suppression_masks(value, roll_flag, p)["any_suppress"]

    equity = start_equity
    pos = None
    trades: list[dict] = []
    equity_curve = np.full(len(df), np.nan)
    equity_curve[0] = equity
    last_stop_date = None
    stop_dates_in_episode: list = []

    for i in range(1, len(df)):
        value_i, value_prev = value[i], value[i - 1]
        z_i, atr_i = z[i], atr[i]

        if pos is not None:
            pnl_mtm = (
                pos["direction"] * pos["qty"] * point_value * (value_i - value_prev)
            )
            equity += pnl_mtm
            adverse = (value_i - pos["entry_value"]) * (-pos["direction"])
            stopped = (not disable_stop) and adverse >= pos["stop_distance"]
            exit_reason = None
            if stopped:
                exit_reason = "stop"
            elif np.isfinite(z_i) and abs(z_i) <= p.exit_threshold:
                exit_reason = "zscore"
            if exit_reason is not None:
                cost = pos["qty"] * cost_per_contract
                equity -= cost
                realized_pnl = (
                    pos["direction"]
                    * pos["qty"]
                    * point_value
                    * (value_i - pos["entry_value"])
                    - cost
                )
                trades.append(
                    {
                        "spread": None,
                        "entry_date": pos["entry_date"],
                        "exit_date": dates[i],
                        "direction": pos["direction"],
                        "qty": pos["qty"],
                        "entry_value": pos["entry_value"],
                        "exit_value": value_i,
                        "entry_z": pos["entry_z"],
                        "entry_atr": pos["entry_atr"],
                        "entry_equity": pos["entry_equity"],
                        "equity_at_open": pos["entry_equity"],
                        "realized_pnl": realized_pnl,
                        "exit_reason": exit_reason,
                        "mae_atr": pos["mae"] / pos["entry_atr"]
                        if pos["entry_atr"]
                        else float("nan"),
                        "mfe_atr": pos["mfe"] / pos["entry_atr"]
                        if pos["entry_atr"]
                        else float("nan"),
                        "pnl_atr": pnl_atr(
                            realized_pnl, pos["qty"], pos["entry_atr"] * point_value
                        ),
                        "ret_eq": ret_eq(realized_pnl, pos["entry_equity"]),
                        "n_days_held": (dates[i] - pos["entry_date"])
                        .astype("timedelta64[D]")
                        .astype(int),
                    }
                )
                if exit_reason == "stop":
                    last_stop_date = dates[i]
                    if (
                        stop_dates_in_episode
                        and (dates[i] - stop_dates_in_episode[0])
                        .astype("timedelta64[D]")
                        .astype(int)
                        > p.episode_window_days
                    ):
                        stop_dates_in_episode = []
                    stop_dates_in_episode.append(dates[i])
                pos = None
            else:
                favorable = (value_i - pos["entry_value"]) * pos["direction"]
                pos["mae"] = min(pos["mae"], favorable)
                pos["mfe"] = max(pos["mfe"], favorable)

        if pos is None and np.isfinite(z_i) and np.isfinite(atr_i) and atr_i > 0:
            regime_ok = True
            if require_ts_regime is not None:
                regime_ok = ts_regime[i] == require_ts_regime
            reentry_ok = True
            if last_stop_date is not None:
                days_since_stop = (
                    (dates[i] - last_stop_date).astype("timedelta64[D]").astype(int)
                )
                cooldown_cleared = days_since_stop >= p.cooldown_days
                gated_bypass = False
                if not cooldown_cleared and i > p.lookback * 2:
                    window = value[max(0, i - p.lookback * 2) : i]
                    window = window[np.isfinite(window)]
                    if len(window) >= 60:
                        fit = R9.ols_ar1_diff(window)
                        hl = fit.get("half_life_days")
                        adf_p = approx_adf_pvalue(S10.adf_test(window)["t_stat"])
                        gated_bypass = (
                            hl is not None
                            and hl <= p.half_life_max
                            and adf_p <= p.adf_pmax
                        )
                episode_count = len(stop_dates_in_episode)
                reentry_ok = (
                    cooldown_cleared or gated_bypass
                ) and episode_count <= p.max_per_episode

            enter = (
                (not suppress[i])
                and regime_ok
                and reentry_ok
                and abs(z_i) > p.entry_threshold
            )
            if enter:
                raw_direction = -1 if z_i > 0 else 1
                direction = -raw_direction if sign_flip else raw_direction
                notional_per_contract = max(abs(leg1[i]), abs(leg2[i])) * point_value
                atr_dollars = atr_i * point_value
                min_atr_dollars = p.min_atr * point_value
                qty = FixedFractionalSizing.quantity(
                    equity=equity,
                    atr=atr_dollars,
                    price=notional_per_contract,
                    risk_pct=p.risk_pct,
                    stop_atr_mult=stop_mult,
                    min_atr=min_atr_dollars,
                    max_leverage=p.max_leverage,
                )
                single_name_cap = np.floor(
                    p.max_single_name_pct / 100.0 * equity / notional_per_contract
                )
                qty = int(min(qty, single_name_cap))
                if qty > 0:
                    pos = {
                        "direction": direction,
                        "qty": qty,
                        "entry_value": value_i,
                        "entry_z": float(z_i),
                        "entry_atr": float(atr_i),
                        "entry_equity": equity,
                        "entry_date": dates[i],
                        "stop_distance": stop_mult * atr_i,
                        "mae": 0.0,
                        "mfe": 0.0,
                    }
        equity_curve[i] = equity

    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "dates": dates,
        "point_value": point_value,
        "leg_products": leg_products,
    }


def cost_per_contract_for_spread(leg_products: list[str], round_turn_cost_fn) -> float:
    """Sum of each leg's own round-turn cost per contract -- a spread trade
    is two round turns (NEXT_PROMPT.md sec 3.1)."""
    return sum(round_turn_cost_fn(prod) for prod in leg_products)


START_EQUITY = 1_000_000.0
"""Not specified anywhere in NEXT_PROMPT.md or the priors in sec 4.1
(no starting-capital figure is given); chosen as a documented assumption
sized so that risk_pct=3% position sizing and the reported live PnL
magnitudes (tens of thousands of dollars per trade) sit in a plausible
institutional range. Because every return metric here is expressed as a
ratio (fixed-notional %, equity-path %, Sharpe), the choice of level is
immaterial to any of Phase 4's reported comparisons except the dollar
figures, which are reported alongside the ratios and labelled with this
assumption."""


def simulate_book(
    spread_frames: dict[str, pl.DataFrame],
    params: dict[str, TradingRuleParams],
    stop_atr_mult_overrides: dict[str, float],
    require_ts_regime: dict[str, str],
    round_turn_cost_fn,
    start_equity: float = START_EQUITY,
    sign_flip_spreads: set[str] | None = None,
    disable_stop: bool = False,
) -> dict:
    """Run `simulate_single_spread` independently per spread (each sized
    against its own copy of `start_equity` -- see `START_EQUITY`'s
    docstring for the documented joint-sizing simplification this implies),
    then pool trades and dollar-PnL into one portfolio book.

    Portfolio equity(t) = start_equity + sum over spreads of
    (that spread's own equity_curve(t) - start_equity), forward-filled
    across each spread's own calendar and dates before a spread's history
    begins (contributes zero). This is dollar-additive pooling of five
    independently-sized P&L streams into one book, not joint risk-engine
    sizing against a single shared equity path -- the true multi-strategy
    risk caps (max_gross_exposure_pct, daily_drawdown_limit_pct,
    overnight_concentration_max_pct) are NOT enforced at the portfolio
    level for this reason and are a disclosed scope simplification.
    """
    sign_flip_spreads = sign_flip_spreads or set()
    all_dates_sorted = sorted(
        set().union(*[set(df["date"].to_numpy()) for df in spread_frames.values()])
    )
    all_dates = np.array(all_dates_sorted)
    per_spread_results = {}
    all_trades = []
    for name, df in spread_frames.items():
        p = params[name]
        leg_products = [r["product"] for r in df["leg_roles"][0]]
        cost = cost_per_contract_for_spread(leg_products, round_turn_cost_fn)
        res = simulate_single_spread(
            df,
            p,
            cost_per_contract=cost,
            stop_atr_mult_override=stop_atr_mult_overrides.get(name),
            require_ts_regime=require_ts_regime.get(name),
            start_equity=start_equity,
            sign_flip=name in sign_flip_spreads,
            disable_stop=disable_stop,
        )
        for t in res["trades"]:
            t["spread"] = name
        per_spread_results[name] = res
        all_trades.extend(res["trades"])

    portfolio_delta = np.zeros(len(all_dates))
    date_index = {d: i for i, d in enumerate(all_dates)}
    for name, res in per_spread_results.items():
        ec = res["equity_curve"]
        prev = start_equity
        for j, d in enumerate(res["dates"]):
            cur = ec[j]
            if np.isfinite(cur):
                portfolio_delta[date_index[d]] += (
                    cur - prev if np.isfinite(prev) else 0.0
                )
                prev = cur
    portfolio_equity = start_equity + np.cumsum(portfolio_delta)

    all_trades.sort(key=lambda t: t["exit_date"])
    return {
        "trades": all_trades,
        "portfolio_equity": portfolio_equity,
        "dates": all_dates,
        "per_spread": per_spread_results,
        "start_equity": start_equity,
    }


def book_metrics(book: dict) -> dict:
    """Sharpe, max drawdown, fixed-notional and equity-path return for a
    `simulate_book` result -- the three-way risk gate's required numbers,
    reported together (NEXT_PROMPT.md sec 5)."""
    equity = book["portfolio_equity"]
    start_equity = book["start_equity"]
    daily_ret = np.diff(equity) / equity[:-1]
    daily_ret = daily_ret[np.isfinite(daily_ret)]
    sharpe = (
        float(np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252))
        if np.std(daily_ret) > 0
        else float("nan")
    )
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    max_dd = float(np.min(drawdown))
    fixed_notional_return = float(sum(t["ret_eq"] for t in book["trades"]))
    equity_path_return = float(equity[-1] / start_equity - 1.0)
    n_trades = len(book["trades"])
    n_stop = sum(1 for t in book["trades"] if t["exit_reason"] == "stop")
    n_zscore = n_trades - n_stop
    return {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "fixed_notional_return": fixed_notional_return,
        "equity_path_return": equity_path_return,
        "return_over_drawdown": float(equity_path_return / abs(max_dd))
        if max_dd != 0
        else float("nan"),
        "n_trades": n_trades,
        "n_stop_exits": n_stop,
        "n_zscore_exits": n_zscore,
        "final_equity": float(equity[-1]),
    }
