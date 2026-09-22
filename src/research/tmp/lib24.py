"""Shared logic for notebook 024 -- the Samuelson effect across 16 futures markets.

Descriptive study: does realised volatility rise as a contract approaches expiry?
Uses a MAD-scaled robust volatility estimator throughout because the far-dated end
of every panel in this repo contains junk prints (stale settlements, sign flips on
illiquid stubs) that destroy a plain standard deviation. See NEXT_PROMPT.md section
2 for the full argument and the smoke-test numbers this module is expected to
reproduce.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

DATA_DIR = "src/research/data/market/databento"

PRODUCTS: list[str] = [
    "CL",
    "BZ",
    "NG",
    "HO",
    "RB",
    "GC",
    "SI",
    "PL",
    "PA",
    "ZC",
    "ZS",
    "ZW",
    "KE",
    "ZL",
    "ZM",
    "ES",
]

SECTOR: dict[str, str] = {
    "CL": "energy",
    "BZ": "energy",
    "NG": "energy",
    "HO": "energy",
    "RB": "energy",
    "GC": "metals",
    "SI": "metals",
    "PL": "metals",
    "PA": "metals",
    "ZC": "ags",
    "ZS": "ags",
    "ZW": "ags",
    "KE": "ags",
    "ZL": "ags",
    "ZM": "ags",
    "ES": "control",
}

FULL_NAME: dict[str, str] = {
    "CL": "WTI crude",
    "BZ": "Brent crude",
    "NG": "Natural gas",
    "HO": "Heating oil",
    "RB": "RBOB gasoline",
    "GC": "Gold",
    "SI": "Silver",
    "PL": "Platinum",
    "PA": "Palladium",
    "ZC": "Corn",
    "ZS": "Soybeans",
    "ZW": "Chicago wheat",
    "KE": "KC wheat",
    "ZL": "Soybean oil",
    "ZM": "Soybean meal",
    "ES": "E-mini S&P 500",
}

DTE_BINS: list[int] = [
    0,
    30,
    60,
    90,
    120,
    180,
    270,
    365,
    547,
    730,
    1095,
    1460,
    2190,
    3650,
]

MAD_CONST = 1.4826

# The single real structural-limitation event: CL202005 settled at -2.67 on
# 2020-04-20 (negative WTI). Excluded from every fit (ln undefined) but never
# silently dropped from the record -- see the write-up.
NEGATIVE_PRICE_EVENT = {
    "ticker": "CL202005",
    "date": pd.Timestamp("2020-04-20"),
    "close": -2.67,
}

CRISIS_WINDOWS: dict[str, tuple[str, str]] = {
    "oil_collapse_2014_15": ("2014-06-01", "2015-12-31"),
    "covid_2020": ("2020-02-01", "2020-06-30"),
    "ukraine_2022": ("2022-01-01", "2022-12-31"),
    "calm_2017": ("2017-01-01", "2017-12-31"),
}


# --------------------------------------------------------------------------- #
# Phase 1 -- panel construction
# --------------------------------------------------------------------------- #


def load_panel(product: str) -> pd.DataFrame:
    """OHLCV joined to contracts, close>0, sorted, with r, dte, gap, log_close.

    Returns every row that survives the close>0 filter; further filtering
    (gap, volume) is the caller's job via usable_returns so the counts stay
    visible in the Phase 1 report.
    """
    ohlcv = pd.read_parquet(f"{DATA_DIR}/ohlcv/{product}.parquet")
    contracts = pd.read_parquet(f"{DATA_DIR}/contracts.parquet")
    contracts = contracts[contracts["product"] == product][
        ["contract_id", "ticker", "expiry", "contract_month"]
    ]
    df = ohlcv.merge(contracts, on="contract_id", suffixes=("", "_c"))
    df["ticker"] = df["ticker"].where(df["ticker"].notna(), df["ticker_c"])
    df = df.drop(columns=["ticker_c"])
    df["date"] = pd.to_datetime(df["date"])
    df["expiry"] = pd.to_datetime(df["expiry"])

    is_event = (df["ticker"] == NEGATIVE_PRICE_EVENT["ticker"]) & (
        df["date"] == NEGATIVE_PRICE_EVENT["date"]
    )
    df = df[~is_event]

    df = df[df["close"] > 0].copy()
    df = df.sort_values(["contract_id", "date"]).reset_index(drop=True)
    df["log_close"] = np.log(df["close"])
    df["r"] = df.groupby("contract_id")["log_close"].diff()
    df["gap"] = df.groupby("contract_id")["date"].diff().dt.days
    df["dte"] = (df["expiry"] - df["date"]).dt.days
    df["product"] = product
    return df


def usable_returns(
    panel: pd.DataFrame, max_gap: int = 1, min_volume: int = 0
) -> pd.DataFrame:
    """Filter to returns usable for volatility estimation.

    Drops the first (NaN-return) row of each contract, keeps only returns
    whose calendar gap to the previous observation is <= max_gap days, and
    applies the volume floor on the *current* day's volume.
    """
    df = panel.dropna(subset=["r", "gap"])
    df = df[df["gap"] <= max_gap]
    if min_volume > 0:
        df = df[df["volume"] > min_volume]
    return df


# --------------------------------------------------------------------------- #
# Volatility estimators
# --------------------------------------------------------------------------- #


def mad_vol(r: np.ndarray, periods: int = 252) -> float:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("nan")
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    return float(MAD_CONST * mad * np.sqrt(periods))


def std_vol(r: np.ndarray, periods: int = 252) -> float:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("nan")
    return float(np.std(r, ddof=1) * np.sqrt(periods))


def winsor_vol(r: np.ndarray, pct: float = 0.01, periods: int = 252) -> float:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("nan")
    lo, hi = np.quantile(r, [pct, 1 - pct])
    clipped = np.clip(r, lo, hi)
    return float(np.std(clipped, ddof=1) * np.sqrt(periods))


ESTIMATORS = {"mad": mad_vol, "std": std_vol, "winsor": winsor_vol}


# --------------------------------------------------------------------------- #
# Bucketed profiles
# --------------------------------------------------------------------------- #


def vol_by_bucket(
    df: pd.DataFrame,
    estimator,
    bins: list[int] | None = None,
    min_obs: int = 50,
    min_contracts: int = 1,
) -> pd.DataFrame:
    """One row per dte bucket: dte_lo, dte_hi, dte_mid, vol, n_obs, n_contracts.

    `min_contracts` guards against a bucket backed by a single contract's
    history masquerading as a "maturity effect" estimate -- observed in
    several products' longest bucket (e.g. one soybean-oil contract supplying
    every observation 6-10 years out). `min_obs` alone does not catch this,
    since one contract's multi-year history alone easily clears any obs floor.
    """
    bins = bins if bins is not None else DTE_BINS
    d = df.copy()
    d["bucket"] = pd.cut(d["dte"], bins=bins, right=False)
    rows = []
    for key, g in d.groupby("bucket", observed=True):
        if len(g) < min_obs or g["contract_id"].nunique() < min_contracts:
            continue
        interval = cast(pd.Interval, key)
        rows.append(
            {
                "dte_lo": float(interval.left),
                "dte_hi": float(interval.right),
                "dte_mid": float((interval.left + interval.right) / 2),
                "vol": estimator(g["r"].to_numpy()),
                "n_obs": len(g),
                "n_contracts": int(g["contract_id"].nunique()),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("dte_mid").reset_index(drop=True)
    return out


def samuelson_slope(
    df: pd.DataFrame,
    estimator,
    n_boot: int = 1000,
    seed: int = 24,
    bins: list[int] | None = None,
    min_obs: int = 50,
    min_contracts: int = 5,
) -> dict:
    """OLS of log(bucket vol) on log(bucket dte_mid), weighted by sqrt(n_obs).

    A negative slope means vol rises as maturity shortens: the Samuelson effect.
    The bootstrap CI resamples at the CONTRACT level, since returns within one
    contract are serially dependent and a row-level bootstrap understates the
    true sampling uncertainty. `min_contracts` (default 5) excludes buckets
    backed by too few distinct contracts from the fit -- several products have
    their longest-maturity bucket supplied by exactly one contract, which is
    not a maturity-effect estimate at all, just that one contract's history.
    """
    bucket = vol_by_bucket(
        df, estimator, bins=bins, min_obs=min_obs, min_contracts=min_contracts
    )
    if len(bucket) < 2:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "r2": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n_boot": 0,
        }

    def fit(b: pd.DataFrame) -> tuple[float, float, float]:
        x = np.log(b["dte_mid"].to_numpy())
        y = np.log(b["vol"].to_numpy())
        w = np.sqrt(b["n_obs"].to_numpy())
        X = np.column_stack([np.ones_like(x), x])
        Wm = np.diag(w)
        coef, *_ = np.linalg.lstsq(Wm @ X, Wm @ y, rcond=None)
        pred = X @ coef
        ss_res = np.sum(w * (y - pred) ** 2)
        ss_tot = np.sum(w * (y - np.average(y, weights=w)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        return float(coef[1]), float(coef[0]), r2

    slope, intercept, r2 = fit(bucket)

    # Contract-level bootstrap, vectorised: precompute each contract's (dte, r)
    # arrays once, then resample and bin with np.searchsorted per draw instead
    # of a pandas concat + groupby per draw (the naive version is O(n_boot *
    # n_contracts) in pandas overhead and is unusably slow for CL/NG-sized
    # panels at n_boot=300+).
    bins_arr = np.asarray(bins if bins is not None else DTE_BINS, dtype=float)
    contracts = df["contract_id"].unique()
    boot_slopes = []
    boot_min_obs = max(5, min_obs // 5)
    boot_min_contracts = max(2, min_contracts // 2)
    if len(contracts) >= 2:
        by_contract = {
            cid: (g["dte"].to_numpy(), g["r"].to_numpy())
            for cid, g in df.groupby("contract_id")
        }
        rng = np.random.default_rng(seed)
        n_bin = len(bins_arr) - 1
        for _ in range(n_boot):
            sampled = rng.choice(contracts, size=len(contracts), replace=True)
            dte_all = np.concatenate([by_contract[c][0] for c in sampled])
            r_all = np.concatenate([by_contract[c][1] for c in sampled])
            # Draw-slot index per row, to count how many distinct resample
            # slots (not raw contract ids, which can repeat under
            # with-replacement sampling) contribute to each bin.
            slot_all = np.concatenate(
                [np.full(len(by_contract[c][0]), i) for i, c in enumerate(sampled)]
            )
            idx = np.searchsorted(bins_arr, dte_all, side="right") - 1
            valid = (idx >= 0) & (idx < n_bin)
            idx, r_valid, slot_valid = idx[valid], r_all[valid], slot_all[valid]

            xs, ys, ws = [], [], []
            for b_idx in range(n_bin):
                mask = idx == b_idx
                n = int(mask.sum())
                if (
                    n < boot_min_obs
                    or len(np.unique(slot_valid[mask])) < boot_min_contracts
                ):
                    continue
                vol = estimator(r_valid[mask])
                if not (np.isfinite(vol) and vol > 0):
                    continue
                xs.append(np.log((bins_arr[b_idx] + bins_arr[b_idx + 1]) / 2))
                ys.append(np.log(vol))
                ws.append(np.sqrt(n))
            if len(xs) < 2:
                continue
            x, y, w = np.array(xs), np.array(ys), np.array(ws)
            X = np.column_stack([np.ones_like(x), x])
            coef, *_ = np.linalg.lstsq(np.diag(w) @ X, np.diag(w) @ y, rcond=None)
            s = float(coef[1])
            if np.isfinite(s):
                boot_slopes.append(s)

    if boot_slopes:
        ci_lo, ci_hi = np.quantile(boot_slopes, [0.025, 0.975])
    else:
        ci_lo, ci_hi = float("nan"), float("nan")

    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "n_boot": len(boot_slopes),
    }


# --------------------------------------------------------------------------- #
# Time dimension
# --------------------------------------------------------------------------- #


def rolling_maturity_vol(
    panel: pd.DataFrame, window: int = 63, buckets: list[tuple[int, int]] | None = None
) -> pd.DataFrame:
    """Date x bucket table of trailing robust vol. Causal: the value at date t
    uses only returns at or before t (a trailing, backward-looking window over
    calendar dates present in the panel)."""
    buckets = buckets or [(0, 60), (60, 180), (180, 365), (365, 730), (730, 3650)]
    df = panel.dropna(subset=["r"]).copy()
    dates = np.sort(df["date"].unique())

    daily = {}
    for lo, hi in buckets:
        mask = (df["dte"] >= lo) & (df["dte"] < hi)
        sub = df[mask]
        s = sub.groupby("date")["r"].apply(lambda x: x.to_numpy())
        s = s.reindex(dates)
        daily[(lo, hi)] = s

    rows = []
    for i, dt in enumerate(dates):
        lo_i = max(0, i - window + 1)
        row = {"date": dt}
        for (lo, hi), series in daily.items():
            window_returns = series.iloc[lo_i : i + 1].dropna()
            if len(window_returns) == 0:
                pooled = np.array([])
            else:
                pooled = np.concatenate([np.asarray(a) for a in window_returns])
            row[f"vol_{lo}_{hi}"] = (
                mad_vol(pooled) if len(pooled) >= 20 else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Moments and moneyness
# --------------------------------------------------------------------------- #


def realised_moments_by_bucket(
    df: pd.DataFrame, bins: list[int] | None = None, min_obs: int = 50
) -> pd.DataFrame:
    """skew, excess kurtosis, and the 5%/95% quantile ratio, per dte bucket."""
    from scipy import stats as sstats

    bins = bins if bins is not None else DTE_BINS
    d = df.copy()
    d["bucket"] = pd.cut(d["dte"], bins=bins, right=False)
    rows = []
    for key, g in d.groupby("bucket", observed=True):
        r = g["r"].dropna().to_numpy()
        if len(r) < min_obs:
            continue
        interval = cast(pd.Interval, key)
        q05, q95 = np.quantile(r, [0.05, 0.95])
        rows.append(
            {
                "dte_lo": float(interval.left),
                "dte_hi": float(interval.right),
                "dte_mid": float((interval.left + interval.right) / 2),
                "skew": float(sstats.skew(r)),
                "excess_kurtosis": float(sstats.kurtosis(r)),
                "q95_q05_ratio": float(q95 / q05) if q05 != 0 else float("nan"),
                "n_obs": len(r),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("dte_mid").reset_index(drop=True)
    return out


def log_moneyness(panel: pd.DataFrame) -> pd.DataFrame:
    """Per contract-day: m = log(close / front_close_same_date), the realised
    analogue of a moneyness axis. Front = the shortest-dte contract that day."""
    d = panel.dropna(subset=["dte"]).copy()
    front_close = d.loc[d.groupby("date")["dte"].idxmin(), ["date", "close"]]
    front_close = front_close.rename(columns={"close": "front_close"})
    d = d.merge(front_close, on="date", how="left")
    d["log_moneyness"] = np.log(d["close"] / d["front_close"])
    return d


def curve_state(panel: pd.DataFrame) -> pd.DataFrame:
    """Per date: contango (+1) or backwardation (-1) from the sign of the
    front-to-second basis (front_close - second_close). Positive basis
    (front > second) is backwardation; negative is contango."""
    d = panel.dropna(subset=["dte"]).copy()
    ranked = d.sort_values(["date", "dte"]).groupby("date").head(2)
    counts = ranked.groupby("date").size()
    valid_dates = counts[counts == 2].index
    ranked = ranked[ranked["date"].isin(valid_dates)]
    rows = []
    for date, g in ranked.groupby("date"):
        g = g.sort_values("dte")
        front, second = g["close"].iloc[0], g["close"].iloc[1]
        state = "backwardation" if front > second else "contango"
        rows.append(
            {"date": date, "curve_state": state, "basis": float(front - second)}
        )
    return pd.DataFrame(rows)
