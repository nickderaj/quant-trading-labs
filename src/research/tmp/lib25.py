"""Shared core for notebook 025 -- pricing bespoke commodity derivatives.

A teaching notebook, not a "does this model beat that model" study: no option
prices exist anywhere in this repo, so every premium below is a *model* price
computed from realised volatility (lib24's machinery), never a market quote.
See NEXT_PROMPT.md for the full argument. This module is the single door
(`inputs_as_of`) every pricer and every case study walks through for market
inputs, plus the closed-form/path-engine/structure primitives shared by all
of them.

Binding constraints from 023/024, reused rather than re-derived here:
 - the forward curve is Nelson-Siegel (`lib23.bench_nelson_siegel`), never
   Gabillon (023 Phase 9: Gabillon lost the held-out-maturity gate);
 - volatility inputs come from realised vol (`lib24.mad_vol`), never from a
   curve fit (023 Phase 7: curve-implied sigma_F collapses to ~3.5% while
   realised vol is 27-36% everywhere);
 - panel hygiene (`lib24.load_panel`, whole-contract exclusion) is reused
   verbatim, never reimplemented.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats as sstats

_TMP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TMP_DIR.parents[2]
for _p in (str(_TMP_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lib23
import lib24

import distributions as _distributions

DATA_DIR = "src/research/data/market/databento"
FRED_DIR = "src/research/data/market/fred"
FX_DIR = "src/research/data/market/yfinance/daily"
DAYS_PER_YEAR = 365.25

OptType = Literal["call", "put"]
BarrierKind = Literal["do", "di", "uo", "ui"]


# --------------------------------------------------------------------------- #
# Market inputs
# --------------------------------------------------------------------------- #


def forward_curve(product: str, date: pd.Timestamp) -> pd.DataFrame:
    """The observed (tau, F) points for `product` as of `date`, after lib24
    hygiene (whole-contract exclusion for tickers that ever printed a
    non-positive close), plus a Nelson-Siegel interpolation.

    Returns columns: tau, F, contract_id, ticker, source ('observed' for the
    actual settle that day, 'interpolated' rows are not added here -- callers
    needing an interpolated point use `inputs_as_of`'s `F` callable).
    """
    panel = lib24.load_panel(product)
    day = panel[panel["date"] == pd.Timestamp(date)].copy()
    day = day[day["dte"] > 0]
    day["tau"] = day["dte"] / DAYS_PER_YEAR
    day["log_close"] = np.log(day["close"])
    day = day.sort_values("tau").reset_index(drop=True)
    return day[["tau", "close", "contract_id", "ticker", "log_close"]].rename(
        columns={"close": "F"}
    )


def _ns_forward_fn(curve: pd.DataFrame) -> Callable[[float | np.ndarray], np.ndarray]:
    """Wrap `lib23.bench_nelson_siegel` as a tau -> F callable. Falls back to
    flat extrapolation from the nearest observed point if there are too few
    curve points to fit (bench_nelson_siegel needs >= 5)."""
    if len(curve) < 5:
        if len(curve) == 0:
            raise ValueError("empty curve: cannot build forward function")
        taus = curve["tau"].to_numpy()
        fs = curve["F"].to_numpy()

        def flat(tau: float | np.ndarray) -> np.ndarray:
            tau_arr = np.atleast_1d(np.asarray(tau, dtype=float))
            idx = np.argmin(np.abs(taus[:, None] - tau_arr[None, :]), axis=0)
            return fs[idx]

        return flat

    day_df = curve.rename(columns={"tau": "tau", "log_close": "log_close"})

    def fn(tau: float | np.ndarray) -> np.ndarray:
        tau_arr = np.atleast_1d(np.asarray(tau, dtype=float))
        tau_arr = np.clip(tau_arr, 1e-6, None)
        out = lib23.bench_nelson_siegel(day_df, tau_arr)
        if out is None:
            taus = curve["tau"].to_numpy()
            fs = curve["F"].to_numpy()
            idx = np.argmin(np.abs(taus[:, None] - tau_arr[None, :]), axis=0)
            return fs[idx]
        return np.asarray(out, dtype=float)

    return fn


def vol_term_structure(
    product: str,
    asof: pd.Timestamp,
    window: int = 756,
    by_state: bool = False,
) -> dict:
    """Fit sigma(tau) = sigma_1 * tau^(-k) to trailing realised (MAD) vol by
    dte bucket, using only data strictly before `asof` (causal). Mirrors
    024 Phase 3 (`lib24.samuelson_slope`) but restricted to a trailing window
    and exposed as a callable. `window` is in trading-panel calendar days.

    Returns {'sigma_1', 'k', 'r2', 'fn': callable(tau)->sigma, 'buckets': DataFrame}
    plus, if by_state, {'contango': {...}, 'backwardation': {...}} sub-dicts
    of the same shape (024 D4).
    """
    panel = lib24.load_panel(product)
    train = panel[panel["date"] < pd.Timestamp(asof)]
    cutoff = pd.Timestamp(asof) - pd.Timedelta(days=window)
    train = train[train["date"] >= cutoff]
    train = lib24.usable_returns(train)

    def _fit(sub: pd.DataFrame) -> dict:
        fit = lib24.samuelson_slope(sub, lib24.mad_vol, n_boot=0)
        sigma_1 = (
            float(np.exp(fit["intercept"]))
            if np.isfinite(fit["intercept"])
            else float("nan")
        )
        k = float(-fit["slope"]) if np.isfinite(fit["slope"]) else float("nan")
        r2 = fit["r2"]

        def fn(tau: float | np.ndarray, sigma_1=sigma_1, k=k) -> np.ndarray:
            tau_arr = np.clip(np.atleast_1d(np.asarray(tau, dtype=float)), 1e-4, None)
            tau_days = tau_arr * DAYS_PER_YEAR
            if not (np.isfinite(sigma_1) and np.isfinite(k)):
                return np.full_like(tau_days, np.nan)
            return sigma_1 * tau_days ** (-k)

        buckets = lib24.vol_by_bucket(sub, lib24.mad_vol)
        return {"sigma_1": sigma_1, "k": k, "r2": r2, "fn": fn, "buckets": buckets}

    out = _fit(train)
    if by_state:
        states = lib24.curve_state(panel)
        train_states = train.merge(states, on="date", how="left")
        for state in ("contango", "backwardation"):
            sub = train_states[train_states["curve_state"] == state].drop(
                columns=["curve_state", "basis"]
            )
            out[state] = _fit(sub) if len(sub) > 100 else None
    return out


def discount_curve(asof: pd.Timestamp) -> Callable[[float | np.ndarray], np.ndarray]:
    """tau -> discount factor DF = exp(-r * tau), from DGS2 (percent, forward-
    filled over non-trading days) as of the most recent observation <= asof.
    """
    dgs2_df = pd.read_parquet(f"{FRED_DIR}/DGS2.parquet")
    dgs2_df = dgs2_df.sort_values("date")
    dgs2_df = dgs2_df[dgs2_df["date"] <= pd.Timestamp(asof)]
    dgs2 = dgs2_df.set_index("date")["DGS2"].ffill()
    if len(dgs2) == 0:
        raise ValueError(f"no DGS2 observation on or before {asof}")
    r = float(dgs2.iloc[-1]) / 100.0

    def df(tau: float | np.ndarray) -> np.ndarray:
        tau_arr = np.asarray(tau, dtype=float)
        return np.exp(-r * tau_arr)

    return df


def fx_series(pair: str = "6E") -> pd.Series:
    """Daily close series for an FX future, indexed by date."""
    fx = pd.read_parquet(f"{FX_DIR}/{pair}=F.parquet")
    fx = fx.sort_values("timestamp")
    s = fx.set_index("timestamp")["close"]
    s.index.name = "date"
    return s


def corr_rolling(prod_a: str, prod_b: str, window: int = 126) -> pd.Series:
    """Rolling `window`-day correlation of front-month log returns between two
    products (or a product and an FX pair via `fx_log_returns`), indexed by
    date, causal (uses only data up to and including that date)."""
    ra = _front_month_returns(prod_a)
    rb = _front_month_returns(prod_b)
    joined = pd.DataFrame({"a": ra, "b": rb}).dropna()
    return joined["a"].rolling(window).corr(joined["b"])


def _front_month_returns(product: str) -> pd.Series:
    if product.startswith("6") and len(product) <= 3:
        s = fx_series(product)
        r = np.log(s).diff()
        r.index = pd.to_datetime(r.index)
        return r
    panel = lib24.load_panel(product)
    front = panel.loc[panel.groupby("date")["dte"].idxmin()].sort_values("date")
    r = front.set_index("date")["r"]
    return r


def inputs_as_of(product: str, date: pd.Timestamp) -> dict:
    """The single door every pricer and every case study walks through.

    {'F': callable(tau)->F, 'sigma': callable(tau)->sigma, 'r': callable(tau)->rate,
     'curve': DataFrame, 'state': 'contango'|'backwardation'}
    """
    date = pd.Timestamp(date)
    curve = forward_curve(product, date)
    F_fn = _ns_forward_fn(curve)
    vol = vol_term_structure(product, date)
    sigma_fn = vol["fn"]
    df_fn = discount_curve(date)

    def r_fn(tau: float | np.ndarray) -> np.ndarray:
        tau_arr = np.asarray(tau, dtype=float)
        df = df_fn(tau_arr)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(tau_arr > 0, -np.log(df) / np.maximum(tau_arr, 1e-9), 0.0)

    if len(curve) >= 2:
        c = curve.sort_values("tau")
        state = "backwardation" if c["F"].iloc[0] > c["F"].iloc[1] else "contango"
    else:
        state = "unknown"

    return {
        "F": F_fn,
        "sigma": sigma_fn,
        "r": r_fn,
        "df": df_fn,
        "curve": curve,
        "state": state,
        "vol_fit": {"sigma_1": vol["sigma_1"], "k": vol["k"], "r2": vol["r2"]},
    }


# --------------------------------------------------------------------------- #
# Closed form
# --------------------------------------------------------------------------- #


def _d1_d2(F: float, K: float, T: float, sigma: float) -> tuple[float, float]:
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black76(
    F: float, K: float, T: float, sigma: float, df: float, opt: OptType
) -> float:
    """Black-76: the underlying is a *futures* price. No dividend/carry term --
    zero drift under the futures measure; discounting applied once, at the
    front. Requires F, K, sigma, T > 0 (undefined for negative F/K -- see
    `bachelier` for the model that admits them)."""
    if T <= 0:
        intrinsic = max(F - K, 0.0) if opt == "call" else max(K - F, 0.0)
        return df * intrinsic
    if sigma <= 0:
        intrinsic = max(F - K, 0.0) if opt == "call" else max(K - F, 0.0)
        return df * intrinsic
    d1, d2 = _d1_d2(F, K, T, sigma)
    if opt == "call":
        return df * (F * sstats.norm.cdf(d1) - K * sstats.norm.cdf(d2))
    return df * (K * sstats.norm.cdf(-d2) - F * sstats.norm.cdf(-d1))


def black76_greeks(
    F: float, K: float, T: float, sigma: float, df: float, opt: OptType
) -> dict:
    if T <= 0 or sigma <= 0:
        return {
            "delta": float("nan"),
            "gamma": float("nan"),
            "vega": float("nan"),
            "theta": float("nan"),
        }
    d1, d2 = _d1_d2(F, K, T, sigma)
    pdf1 = sstats.norm.pdf(d1)
    sign = 1.0 if opt == "call" else -1.0
    delta = sign * df * sstats.norm.cdf(sign * d1)
    gamma = df * pdf1 / (F * sigma * np.sqrt(T))
    vega = df * F * pdf1 * np.sqrt(T)
    r = -np.log(df) / T if T > 0 else 0.0
    theta = -(df * F * pdf1 * sigma) / (2 * np.sqrt(T)) + sign * r * df * (
        K * sstats.norm.cdf(-sign * d2)
        if opt == "put"
        else -F * sstats.norm.cdf(sign * d1)
    )
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
    }


def bachelier(
    F: float, K: float, T: float, sigma_n: float, df: float, opt: OptType
) -> float:
    """Normal (Bachelier) model: F_T = F + sigma_n * sqrt(T) * Z. Admits
    negative F and negative K -- this is the model that does not assign
    probability zero to a negative settlement (case study E)."""
    if T <= 0 or sigma_n <= 0:
        intrinsic = max(F - K, 0.0) if opt == "call" else max(K - F, 0.0)
        return df * intrinsic
    s = sigma_n * np.sqrt(T)
    d = (F - K) / s
    if opt == "call":
        return df * ((F - K) * sstats.norm.cdf(d) + s * sstats.norm.pdf(d))
    return df * ((K - F) * sstats.norm.cdf(-d) + s * sstats.norm.pdf(d))


def displaced_black(
    F: float, K: float, T: float, sigma: float, df: float, opt: OptType, shift: float
) -> float:
    """Displaced diffusion: shift F and K by `shift` (alpha) and apply
    Black-76 to (F+shift, K+shift), with sigma scaled so the displaced
    process's lognormal vol matches. As shift -> 0 this is Black-76; for
    large shift relative to F it converges to Bachelier."""
    if shift <= 0:
        return black76(F, K, T, sigma, df, opt)
    Fs, Ks = F + shift, K + shift
    if Fs <= 0 or Ks <= 0:
        # displaced measure has gone non-positive: fall back to Bachelier
        # with an equivalent normal vol as the degenerate limit.
        sigma_n = sigma * F if F > 0 else sigma * shift
        return bachelier(F, K, T, sigma_n, df, opt)
    return black76(Fs, Ks, T, sigma, df, opt)


def implied_vol(
    price: float,
    F: float,
    K: float,
    T: float,
    df: float,
    opt: OptType,
    model: str = "black76",
) -> float:
    """Invert `black76` (or `bachelier`) for sigma via bisection. Round-trips
    to high precision by construction (a scalar monotone root-find)."""
    pricer: Callable[[float], float]
    if model == "black76":
        pricer = lambda s: black76(F, K, T, s, df, opt)
        lo, hi = 1e-6, 10.0
    elif model == "bachelier":
        pricer = lambda s: bachelier(F, K, T, s, df, opt)
        lo, hi = 1e-6, abs(F) * 5 + 100.0
    else:
        raise ValueError(f"unknown model {model!r}")

    f_lo, f_hi = pricer(lo) - price, pricer(hi) - price
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = pricer(mid) - price
        if abs(f_mid) < 1e-12:
            return float(mid)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return float(0.5 * (lo + hi))


# --------------------------------------------------------------------------- #
# Path engines
# --------------------------------------------------------------------------- #


def _normals(n_steps: int, n_paths: int, seed: int, antithetic: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if antithetic:
        half = (n_paths + 1) // 2
        z = rng.standard_normal((half, n_steps))
        z = np.vstack([z, -z])[:n_paths]
    else:
        z = rng.standard_normal((n_paths, n_steps))
    return z


def simulate_gbm(
    F0: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int,
    antithetic: bool = True,
) -> np.ndarray:
    """Driftless GBM paths for a futures price (Q-measure: zero drift).
    Returns array shape (n_paths, n_steps+1), column 0 = F0."""
    dt = T / n_steps
    z = _normals(n_steps, n_paths, seed, antithetic)
    increments = (-0.5 * sigma**2 * dt) + sigma * np.sqrt(dt) * z
    log_paths = np.log(F0) + np.cumsum(increments, axis=1)
    paths = np.exp(log_paths)
    return np.hstack([np.full((n_paths, 1), F0), paths])


def simulate_bachelier(
    F0: float,
    sigma_n: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int,
    antithetic: bool = True,
) -> np.ndarray:
    """Driftless arithmetic Brownian motion paths -- admits negative values."""
    dt = T / n_steps
    z = _normals(n_steps, n_paths, seed, antithetic)
    increments = sigma_n * np.sqrt(dt) * z
    paths = F0 + np.cumsum(increments, axis=1)
    return np.hstack([np.full((n_paths, 1), F0), paths])


def simulate_correlated(
    F0s: list[float],
    sigmas: list[float],
    rho: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """Two correlated driftless GBM paths. Returns shape (2, n_paths, n_steps+1)."""
    if len(F0s) != 2 or len(sigmas) != 2:
        raise ValueError("simulate_correlated prices exactly two correlated assets")
    dt = T / n_steps
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal((n_paths, n_steps))
    z_ind = rng.standard_normal((n_paths, n_steps))
    z2 = rho * z1 + np.sqrt(max(1 - rho**2, 0.0)) * z_ind
    out = []
    for F0, sigma, z in zip(F0s, sigmas, (z1, z2), strict=True):
        increments = (-0.5 * sigma**2 * dt) + sigma * np.sqrt(dt) * z
        log_paths = np.log(F0) + np.cumsum(increments, axis=1)
        paths = np.exp(log_paths)
        out.append(np.hstack([np.full((n_paths, 1), F0), paths]))
    return np.stack(out, axis=0)


def bootstrap_paths(
    product: str,
    T: float,
    n_paths: int,
    block_days: int,
    seed: int,
    asof: pd.Timestamp | None = None,
) -> np.ndarray:
    """Block-resample historical curve (front-month) log-return changes and
    cumulate into synthetic paths. No model beyond stationarity. Never chains
    a return across contract_id (024's rule): blocks are drawn from within a
    single contract's own return history, never spanning a roll.

    Returns shape (n_paths, n_steps+1) of simulated futures levels starting
    at the front-month's last observed close on or before `asof`.
    """
    panel = lib24.load_panel(product)
    if asof is not None:
        panel = panel[panel["date"] <= pd.Timestamp(asof)]
    panel = lib24.usable_returns(panel)
    n_steps = max(round(T * DAYS_PER_YEAR), 1)

    by_contract = {
        cid: g.sort_values("date")["r"].to_numpy()
        for cid, g in panel.groupby("contract_id")
        if g["r"].notna().sum() >= block_days
    }
    contracts = [cid for cid, r in by_contract.items() if len(r) >= block_days]
    if not contracts:
        raise ValueError(
            f"no contract for {product} has >= {block_days} usable returns"
        )

    front = panel.sort_values("date").iloc[-1]
    F0 = float(front["close"])

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_steps / block_days))
    all_returns = np.empty((n_paths, n_blocks * block_days))
    for p in range(n_paths):
        chunks = []
        for _ in range(n_blocks):
            cid = contracts[rng.integers(len(contracts))]
            series = by_contract[cid]
            start = rng.integers(0, len(series) - block_days + 1)
            chunks.append(series[start : start + block_days])
        all_returns[p] = np.concatenate(chunks)
    all_returns = all_returns[:, :n_steps]
    log_paths = np.log(F0) + np.cumsum(all_returns, axis=1)
    paths = np.exp(log_paths)
    return np.hstack([np.full((n_paths, 1), F0), paths])


def sobol_normals(n_paths: int, n_steps: int, seed: int) -> np.ndarray:
    """Sobol quasi-random normal draws, shape (n_paths, n_steps)."""
    sampler = sstats.qmc.Sobol(d=n_steps, seed=seed)
    m = int(np.ceil(np.log2(max(n_paths, 2))))
    u = sampler.random_base2(m=m)[:n_paths]
    u = np.clip(u, 1e-10, 1 - 1e-10)
    return sstats.norm.ppf(u)


# --------------------------------------------------------------------------- #
# Structures
# --------------------------------------------------------------------------- #


def collar(
    F: float, K_put: float, K_call: float, T: float, sigma: float, df: float
) -> dict:
    """Long put at K_put, short call at K_call. Net premium is a cost when
    positive, a credit when negative."""
    put = black76(F, K_put, T, sigma, df, "put")
    call = black76(F, K_call, T, sigma, df, "call")
    return {
        "premium": put - call,
        "legs": [
            {"type": "put", "strike": K_put, "qty": 1, "price": put},
            {"type": "call", "strike": K_call, "qty": -1, "price": call},
        ],
    }


def zero_cost_collar(F: float, K_put: float, T: float, sigma: float, df: float) -> dict:
    """Solve for the call strike K_call >= F that makes the collar's net
    premium zero, given a chosen put strike."""
    put = black76(F, K_put, T, sigma, df, "put")

    def f(K_call: float) -> float:
        return black76(F, K_call, T, sigma, df, "call") - put

    lo, hi = F, F * 20
    f_lo = f(lo)
    if f_lo <= 0:
        K_call = lo
    else:
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            f_mid = f(mid)
            if abs(f_mid) < 1e-10:
                break
            if f_mid > 0:
                lo = mid
            else:
                hi = mid
        K_call = 0.5 * (lo + hi)
    call = black76(F, K_call, T, sigma, df, "call")
    return {
        "premium": put - call,
        "K_call": K_call,
        "legs": [
            {"type": "put", "strike": K_put, "qty": 1, "price": put},
            {"type": "call", "strike": K_call, "qty": -1, "price": call},
        ],
    }


def participating_forward(
    F: float, K: float, participation: float, T: float, sigma: float, df: float
) -> dict:
    """A forward at K, with the client keeping `participation` fraction of
    the upside above K (buying a fraction of a call to fund the rest)."""
    put = black76(F, K, T, sigma, df, "put")
    call = black76(F, K, T, sigma, df, "call")
    # zero cost: long forward (put - call = 0 synthetic), long participation*call, sized
    # so total premium is zero: solve participation implicitly if not given via strike;
    # here participation is given, premium is whatever it costs.
    premium = participation * call - put
    return {
        "premium": premium,
        "participation": participation,
        "legs": [
            {"type": "forward", "strike": K, "qty": 1},
            {"type": "call", "strike": K, "qty": participation, "price": call},
            {"type": "put", "strike": K, "qty": -1, "price": put},
        ],
    }


def price_structure(legs: list[dict], inputs: dict) -> dict:
    """Generic summer: legs is a list of dicts each with a 'price' key (and
    optional 'qty', default 1). Returns {'premium': sum(qty*price), 'legs': legs}."""
    total = sum(leg.get("qty", 1) * leg["price"] for leg in legs)
    return {"premium": float(total), "legs": legs}


def realised_payoff(structure: dict, path: np.ndarray, schedule: dict) -> float:
    """Arithmetic realised payoff of a structure given one recorded/simulated
    path. `schedule` carries whatever the structure's legs need:
      - 'terminal': path[-1] used for vanilla/collar payoffs
      - 'barrier': {'level': B, 'kind': 'do'|'di'|'uo'|'ui'} -- monitored on
        every point in `path`
      - 'avg_idx': indices into `path` used for an Asian average

    Handles barrier monitoring, averaging schedules and autocall observations
    via the keys present in `schedule` and each leg's 'type'.
    """
    barrier = schedule.get("barrier")
    alive = True
    if barrier is not None:
        level, kind = barrier["level"], barrier["kind"]
        if kind in ("do", "di"):
            hit = np.any(path <= level)
        else:
            hit = np.any(path >= level)
        if kind in ("do", "uo"):
            alive = not hit
        else:  # knock-in
            alive = bool(hit)

    avg_idx = schedule.get("avg_idx")
    terminal = float(np.mean(path[avg_idx])) if avg_idx is not None else float(path[-1])

    total = 0.0
    for leg in structure["legs"]:
        qty = leg.get("qty", 1)
        typ = leg["type"]
        K = leg.get("strike")
        if typ == "forward":
            total += qty * (terminal - K)
        elif typ == "call":
            payoff = max(terminal - K, 0.0) if alive else 0.0
            total += qty * payoff
        elif typ == "put":
            payoff = max(K - terminal, 0.0) if alive else 0.0
            total += qty * payoff
    return float(total)


# --------------------------------------------------------------------------- #
# Scoring (case study E)
# --------------------------------------------------------------------------- #


def kupiec_pof(exceedances: np.ndarray, alpha: float) -> dict:
    """Wraps `distributions.kupiec_test`: unconditional-coverage
    (proportion-of-failures) test of a left-tail quantile forecast."""
    exceedances = np.asarray(exceedances, dtype=bool)
    lr, pvalue = _distributions.kupiec_test(exceedances, alpha)
    return {
        "n": len(exceedances),
        "n_exceed": int(exceedances.sum()),
        "observed_rate": float(exceedances.mean())
        if len(exceedances)
        else float("nan"),
        "expected_rate": float(alpha),
        "lr_stat": lr,
        "pvalue": pvalue,
        "reject_5pct": bool(pvalue < 0.05),
    }


def qlike(forecast_var: np.ndarray, realised_var: np.ndarray) -> float:
    """Wraps `distributions.qlike`, mean over the sample."""
    losses = _distributions.qlike(
        np.asarray(realised_var, dtype=float), np.asarray(forecast_var, dtype=float)
    )
    return float(np.nanmean(losses))
