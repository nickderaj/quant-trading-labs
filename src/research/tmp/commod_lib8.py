"""Notebook 8 library: futures-specific construction, hygiene, costs, seasonality/
curve features, dependence/copula tools, and the risk-engine API.

Only genuinely new machinery lives here. Walk-forward, bootstrap, portfolio, and
distributional machinery is reused from research.py / distributions.py / dist_lib5.py /
dist_lib6.py / alpha_lib7.py — see NEXT_PROMPT.md section 2.
"""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
from scipy import stats as st

# ---------------------------------------------------------------------------
# Product metadata: sectors and a stable colour palette (NEXT_PROMPT.md sec 6).
# ---------------------------------------------------------------------------

PRODUCTS = [
    "CL",
    "BZ",
    "NG",
    "HO",
    "RB",  # energy
    "GC",
    "SI",
    "PL",
    "PA",  # metals
    "ZC",
    "ZW",
    "KE",
    "ZS",
    "ZL",
    "ZM",  # grains
    "ES",  # control (non-commodity)
]

SECTOR = {
    "CL": "energy",
    "BZ": "energy",
    "NG": "energy",
    "HO": "energy",
    "RB": "energy",
    "GC": "metals",
    "SI": "metals",
    "PL": "metals",
    "PA": "metals",
    "ZC": "grains",
    "ZW": "grains",
    "KE": "grains",
    "ZS": "grains",
    "ZL": "grains",
    "ZM": "grains",
    "ES": "control",
}

# Physically-delivered products must be rolled before first notice.
PHYSICALLY_DELIVERED = {
    "CL",
    "NG",
    "HO",
    "RB",
    "GC",
    "SI",
    "PL",
    "PA",
    "ZC",
    "ZW",
    "KE",
    "ZS",
    "ZL",
    "ZM",
}

# Sector base hues (matplotlib/altair hex), with per-product shade variation applied
# by `product_color`. Defined once here and reused across every figure in the
# notebook so colour is consistent for a given product everywhere it appears.
_SECTOR_BASE = {
    "energy": "#d1495b",  # warm red
    "metals": "#8886d1",  # violet-grey
    "grains": "#2a9d8f",  # teal-green
    "control": "#6c757d",  # neutral grey (ES: not a commodity)
}

_SECTOR_SHADES = {
    "energy": ["#8c1f30", "#b3324a", "#d1495b", "#e2727f", "#ef9ba3"],
    "metals": ["#4b3f96", "#6a5cb0", "#8886d1", "#a8a6dd", "#c7c6ea"],
    "grains": ["#1b6358", "#217e70", "#2a9d8f", "#5cb8ac", "#8ecec5", "#b6e0d9"],
    "control": ["#495057"],
}


def product_color(product: str) -> str:
    """Stable hex colour for a product, grouped by sector shade family."""
    sector = SECTOR[product]
    members = sorted([p for p in PRODUCTS if SECTOR[p] == sector])
    idx = members.index(product)
    shades = _SECTOR_SHADES[sector]
    return shades[idx % len(shades)]


def sector_base_color(sector: str) -> str:
    return _SECTOR_BASE[sector]


# ---------------------------------------------------------------------------
# Phase 0.1 -- duplicate-tree check
# ---------------------------------------------------------------------------


def check_duplicate_tree(root_a: str, root_b: str) -> dict:
    """Compare two candidate data roots. Returns a dict describing which exist and,
    if both exist, whether a sample of files match by size+hash.
    """
    from pathlib import Path

    a, b = Path(root_a), Path(root_b)
    result: dict = {
        "root_a": str(a),
        "root_b": str(b),
        "a_exists": a.exists(),
        "b_exists": b.exists(),
    }
    if not (result["a_exists"] and result["b_exists"]):
        result["verdict"] = "only one tree present; duplicate check inapplicable"
        return result

    import hashlib

    files_a = sorted(a.rglob("*.parquet"))
    mismatches = []
    checked = 0
    for fa in files_a[:25]:
        fb = b / fa.relative_to(a)
        if not fb.exists():
            mismatches.append(str(fa))
            continue
        ha = hashlib.md5(fa.read_bytes()).hexdigest()
        hb = hashlib.md5(fb.read_bytes()).hexdigest()
        checked += 1
        if ha != hb:
            mismatches.append(str(fa))
    result["files_checked"] = checked
    result["mismatches"] = mismatches
    result["verdict"] = "identical (sampled)" if not mismatches else "DIFFERS"
    return result


# ---------------------------------------------------------------------------
# Phase 0.2 -- price hygiene filter
# ---------------------------------------------------------------------------


def flag_contaminated_rows(
    df: pl.DataFrame,
    deviation_threshold: float = 0.3,
    volume_materiality: int = 50_000,
    contract_junk_frac: float = 0.5,
    min_days_for_persistence: int = 10,
) -> pl.DataFrame:
    """Flag spread/differential-settlement rows contaminating outright OHLCV data.

    Volume alone cannot separate the real cases from the fake ones: CL's genuine
    2020-04-20 negative settle traded on 8.4% of that day's total CL volume, and
    NG's mislabeled spread-differential contracts (e.g. NG202507 on 2025-05-23)
    traded on a *comparable* 9.9% of that day's NG volume -- an absolute or
    relative volume cutoff flags both or neither.

    The signal that does separate them is **persistence**. NG202507 prints a
    near-zero/negative close on 97% of the ~575 days it appears in the data --
    it is not an outright contract having one bad day, it is a differential
    series mislabeled as one, for its entire life. CL's contract 752 (CL202005)
    deviates like this on 0.6% of its ~343 days -- one genuine event in an
    otherwise normally-trading contract.

    Two-tier rule, evaluated relative to each date's highest-volume contract of
    the same product (the anchor -- almost always a genuine, liquid outright):

    1. **Contract-level.** For any contract_id with >= `min_days_for_persistence`
       printed days, if the fraction of its days deviating from the anchor by
       more than `deviation_threshold` exceeds `contract_junk_frac`, every row
       of that contract is flagged -- it is not an outright series at all.
    2. **Row-level.** For contracts that pass (1) -- including any with too
       little history to judge persistence -- a single day is still flagged if
       it deviates from the anchor by more than `deviation_threshold` AND its
       volume is below `volume_materiality`. This is the safety net for a
       genuine one-off glitch on an otherwise-clean, low-volume contract; a
       move on `volume_materiality`-or-more contracts is price discovery, not a
       data error (CL 2020-04-20's 102,083-contract volume clears this easily).

    Requires columns: product, date, contract_id, close, volume.
    """
    anchor = (
        df.sort(["product", "date", "volume"], descending=[False, False, True])
        .group_by(["product", "date"], maintain_order=True)
        .first()
        .select(["product", "date", pl.col("close").alias("_anchor_close")])
    )
    out = df.join(anchor, on=["product", "date"], how="left")
    out = out.with_columns(
        (
            (pl.col("close") - pl.col("_anchor_close")).abs()
            / pl.col("_anchor_close").abs().clip(lower_bound=1e-9)
        ).alias("_deviation")
    )
    out = out.with_columns(
        (pl.col("_deviation") > deviation_threshold).alias("_bad_day")
    )

    contract_stats = out.group_by(["product", "contract_id"]).agg(
        pl.col("_bad_day").mean().alias("_frac_bad"), pl.len().alias("_n_days")
    )
    out = out.join(contract_stats, on=["product", "contract_id"], how="left")

    is_junk_contract = (pl.col("_n_days") >= min_days_for_persistence) & (
        pl.col("_frac_bad") > contract_junk_frac
    )
    is_row_glitch = pl.col("_bad_day") & (pl.col("volume") < volume_materiality)
    out = out.with_columns((is_junk_contract | is_row_glitch).alias("contaminated"))
    return out.drop(["_anchor_close", "_deviation", "_bad_day", "_frac_bad", "_n_days"])


def apply_hygiene_filter(df: pl.DataFrame, **kwargs) -> pl.DataFrame:
    """Flag and drop contaminated rows, returning the clean frame."""
    flagged = flag_contaminated_rows(df, **kwargs)
    return flagged.filter(~pl.col("contaminated")).drop("contaminated")


# ---------------------------------------------------------------------------
# Phase 0.3 -- liquidity screen
# ---------------------------------------------------------------------------


def liquidity_screen(
    df: pl.DataFrame, min_volume: int = 50, min_active_contracts: int = 2
) -> pl.DataFrame:
    """Drop rows below a minimum volume, then drop (product, date) groups with
    fewer than `min_active_contracts` surviving contracts (front-month selection
    needs at least one alternative to confirm the curve isn't a single stale quote).
    """
    screened = df.filter(pl.col("volume") >= min_volume)
    counts = screened.group_by(["product", "date"]).agg(pl.len().alias("_n_active"))
    screened = screened.join(counts, on=["product", "date"], how="left")
    screened = screened.filter(pl.col("_n_active") >= min_active_contracts).drop(
        "_n_active"
    )
    return screened


# ---------------------------------------------------------------------------
# Phase 0.4 -- roll calendar and continuous series construction
# ---------------------------------------------------------------------------


def build_roll_schedule(
    roll_calendar: pl.DataFrame,
    product: str,
    roll_days_before: int = 5,
    valid_contract_months: set[str] | None = None,
) -> pl.DataFrame:
    """Per contract_month for `product`, the roll date = last_trade_date minus
    `roll_days_before` *business* days (approximated with calendar-day offset then
    snapped backward off weekends -- daily bars only, no exchange calendar available).

    first_notice_date is used instead of last_trade_date when present (physically
    delivered contracts must be off the book before first notice); roll_calendar's
    first_notice_date is ~94% null, so last_trade_date is the practical anchor.

    `valid_contract_months`, when given, restricts the schedule to contract
    months that actually exist in contracts.parquet. This matters:
    roll_calendar.parquet lists an entry for every calendar month for every
    product, but seasonal/quarterly-cycle products (the grains: ZC/ZW/KE/ZS/
    ZL/ZM trade Mar/May/Jul/Sep/Dec-style cycles, not monthly) were never
    actually listed in most of those months. Without this filter, the roll
    schedule "rolls into" a month with zero real contracts and the F1 series
    goes null for every date until the next *real* month's roll_date -- on
    ZC this silently dropped 60% of trading days before it was caught.
    """
    cal = roll_calendar.filter(pl.col("product") == product).sort("expiry")
    if valid_contract_months is not None:
        cal = cal.filter(pl.col("contract_month").is_in(list(valid_contract_months)))
    anchor = pl.coalesce([pl.col("first_notice_date"), pl.col("last_trade_date")])
    cal = cal.with_columns(anchor.alias("_anchor"))

    def _offset_business_days(date_col: pl.Expr, n: int) -> pl.Expr:
        d = date_col.cast(pl.Date)
        shifted = d.dt.offset_by(f"-{n}d")
        dow = shifted.dt.weekday()  # 1=Mon .. 7=Sun
        shifted = (
            pl.when(dow == 6)
            .then(shifted.dt.offset_by("-1d"))
            .when(dow == 7)
            .then(shifted.dt.offset_by("-2d"))
            .otherwise(shifted)
        )
        return shifted

    cal = cal.with_columns(
        _offset_business_days(pl.col("_anchor"), roll_days_before).alias("roll_date")
    )
    return cal.select(
        [
            "product",
            "contract_month",
            "expiry",
            "first_notice_date",
            "last_trade_date",
            "_anchor",
            "roll_date",
        ]
    ).rename({"_anchor": "anchor_date"})


def liquid_contract_months(
    ohlcv: pl.DataFrame,
    contracts: pl.DataFrame,
    product: str,
    min_total_volume: int = 5000,
) -> set[str]:
    """Contract months for `product` that were genuinely traded, not just
    nominally listed. Some products list a ticker for every calendar month
    but only trade actively on a seasonal/quarterly cycle -- PL and PA trade
    real size in Jan/Apr/Jul/Oct (total life volume in the hundreds of
    thousands) while the "in-between" months print a handful of trades on a
    handful of days (total life volume in the tens to low hundreds) before
    going silent. Rolling the F1 series into one of those near-dead months
    (which `build_roll_schedule`'s contracts.parquet-membership check alone
    does not exclude, since they ARE nominally listed) creates a hole in the
    front-month series for that whole month -- this is what caught it on PL
    (57% of F1 rows null before this filter).
    """
    prod_contracts = contracts.filter(pl.col("product") == product).select(
        ["contract_id", "contract_month"]
    )
    j = ohlcv.filter(pl.col("product") == product).join(
        prod_contracts, on="contract_id", how="inner"
    )
    agg = j.group_by("contract_month").agg(pl.col("volume").sum().alias("total_vol"))
    return set(
        agg.filter(pl.col("total_vol") >= min_total_volume)["contract_month"].to_list()
    )


def build_continuous_series(
    ohlcv: pl.DataFrame,
    contracts: pl.DataFrame,
    roll_calendar: pl.DataFrame,
    product: str,
    roll_days_before: int = 5,
    n_legs: int = 3,
    min_contract_volume: int = 5000,
) -> pl.DataFrame:
    """Build F1..F{n_legs} continuous price series with days-to-expiry, plus three
    return conventions on F1:

    - `log_return_unadj`: log(close_t / close_{t-1}) computed WITHIN a single
      contract only; NaN/null across a roll boundary (the correct series for
      backtests -- costs and P&L are charged on the actual traded contract).
    - `log_return_backadj`: return on the back-adjusted (Panama/difference) price,
      which splices a constant additive offset at each roll so the *level* is
      continuous for charting/technical signals. The offset equals the gap between
      the old and new front-month contract's price on the roll date.
    - `log_return_ratioadj`: return on the ratio-adjusted price (multiplicative
      splice) -- an alternative continuous convention, better behaved for
      log-return-based statistics when price levels vary a lot over the sample.

    Roll rule: at the roll date for a contract month (last_trade_date - N business
    days, or first_notice_date - N if notice is populated), F1 switches to the next
    contract_month in expiry order. This must happen strictly *before* first notice
    for physically-delivered products, which is why the offset is subtracted from
    last_trade_date/first_notice rather than added.

    `min_contract_volume` (via `liquid_contract_months`) drops nominally-listed
    but never-really-traded months from the roll sequence (sec above); set to 0
    to disable and use raw contracts.parquet membership only.
    """
    prod_ohlcv = ohlcv.filter(pl.col("product") == product).sort(
        ["contract_id", "date"]
    )
    prod_contracts = contracts.filter(pl.col("product") == product).select(
        ["contract_id", "contract_month", "expiry"]
    )
    valid_months = set(prod_contracts["contract_month"].to_list())
    if min_contract_volume > 0:
        valid_months &= liquid_contract_months(
            ohlcv, contracts, product, min_contract_volume
        )
    schedule = build_roll_schedule(
        roll_calendar, product, roll_days_before, valid_contract_months=valid_months
    ).sort("expiry")

    px = prod_ohlcv.join(prod_contracts, on="contract_id", how="inner")
    px = px.join(
        schedule.select(["contract_month", "roll_date", "anchor_date"]),
        on="contract_month",
        how="left",
    )
    px = px.filter(pl.col("roll_date").is_not_null()).sort(["expiry", "date"])

    sched_sorted = schedule.filter(pl.col("roll_date").is_not_null()).sort("expiry")
    months = sched_sorted["contract_month"].to_list()
    roll_dates = sched_sorted["roll_date"].to_numpy().astype("datetime64[D]")

    dates_list = px.select("date").unique().sort("date")["date"].to_list()
    dates_arr = np.array(dates_list, dtype="datetime64[D]")

    # For each date d, the active F1 contract index is the count of contracts
    # whose roll_date has already passed (roll_date <= d) -- i.e. the book has
    # rolled into a contract on its roll date, not the day after. `roll_dates` is
    # sorted ascending (expiry order); side="right" counts ties as "already rolled".
    f1_idx = np.searchsorted(roll_dates, dates_arr, side="right")

    px_lookup = px.select(["date", "contract_month", "close", "expiry"])

    legs = []
    n_months = len(months)
    for leg in range(1, n_legs + 1):
        leg_idx = f1_idx + (leg - 1)
        valid = leg_idx < n_months
        sel_dates = dates_arr[valid].astype("datetime64[ms]")
        sel_months = [months[i] for i in leg_idx[valid]]
        leg_map = pl.DataFrame(
            {"date": sel_dates, "contract_month": sel_months}
        ).with_columns(pl.col("date").cast(pl.Date))
        leg_px = leg_map.join(px_lookup, on=["date", "contract_month"], how="left")
        leg_px = leg_px.with_columns(
            (pl.col("expiry") - pl.col("date")).dt.total_days().alias(f"dte_f{leg}")
        ).rename({"close": f"close_f{leg}", "contract_month": f"contract_month_f{leg}"})
        legs.append(
            leg_px.select(
                ["date", f"close_f{leg}", f"dte_f{leg}", f"contract_month_f{leg}"]
            )
        )

    curve = legs[0]
    for leg_df in legs[1:]:
        curve = curve.join(leg_df, on="date", how="left")

    curve = curve.sort("date")
    curve = _add_return_conventions(curve)
    return curve


def _add_return_conventions(curve: pl.DataFrame) -> pl.DataFrame:
    """Given a curve frame with close_f1 and contract_month_f1, add the three
    return conventions defined in `build_continuous_series`'s docstring.
    """
    curve = curve.with_columns(
        pl.col("contract_month_f1")
        .ne(pl.col("contract_month_f1").shift(1))
        .fill_null(False)
        .alias("is_roll")
    )
    curve = curve.with_columns(
        pl.when(pl.col("is_roll") | pl.col("close_f1").shift(1).is_null())
        .then(None)
        .otherwise((pl.col("close_f1") / pl.col("close_f1").shift(1)).log())
        .alias("log_return_unadj")
    )

    close = curve["close_f1"].to_numpy()
    is_roll = curve["is_roll"].to_numpy()
    n = len(close)
    offset = np.zeros(n)
    cum_offset = 0.0
    for i in range(1, n):
        if is_roll[i] and not np.isnan(close[i]) and not np.isnan(close[i - 1]):
            # gap between the new front month's price and what the old front
            # month would have implied -- roll the cumulative offset forward
            cum_offset += close[i] - close[i - 1]
        offset[i] = cum_offset
    backadj = close - offset
    curve = curve.with_columns(pl.Series("close_backadj", backadj))
    curve = curve.with_columns(
        (pl.col("close_backadj") / pl.col("close_backadj").shift(1))
        .log()
        .alias("log_return_backadj")
    )

    ratio = np.ones(n)
    cum_ratio = 1.0
    for i in range(1, n):
        if (
            is_roll[i]
            and not np.isnan(close[i])
            and not np.isnan(close[i - 1])
            and close[i - 1] != 0
        ):
            cum_ratio *= close[i - 1] / close[i]
        ratio[i] = cum_ratio
    ratioadj = close * ratio
    curve = curve.with_columns(pl.Series("close_ratioadj", ratioadj))
    curve = curve.with_columns(
        (pl.col("close_ratioadj") / pl.col("close_ratioadj").shift(1))
        .log()
        .alias("log_return_ratioadj")
    )
    return curve


# ---------------------------------------------------------------------------
# Phase 0.5 -- reconciliation helpers
# ---------------------------------------------------------------------------


def reconcile_curves(
    built: pl.DataFrame, reference: pl.DataFrame, tick_size: float, leg: int = 1
) -> dict:
    """Compare a built F{leg} close series against a ready-made research/*_curve
    close_f{leg} series on overlapping dates. Pass condition: >=99% of overlapping
    dates agree to within one tick.
    """
    col_built = f"close_f{leg}"
    col_ref = f"close_f{leg}"
    joined = built.select(["date", col_built]).join(
        reference.select(["date", col_ref]).rename({col_ref: "_ref"}),
        on="date",
        how="inner",
    )
    joined = joined.drop_nulls()
    if joined.height == 0:
        return {"n_overlap": 0, "pct_within_tick": None, "pass": False}
    diff = (joined[col_built] - joined["_ref"]).abs()
    within = (diff <= tick_size + 1e-9).sum()
    pct = within / joined.height
    return {
        "n_overlap": joined.height,
        "pct_within_tick": float(pct),
        "pass": bool(pct >= 0.99),
    }


def reconcile_vol(
    built_vol: pl.DataFrame, metrics_vol: pl.DataFrame, tolerance: float = 0.15
) -> dict:
    """Compare computed realised vol against metrics/*.parquet's realised_vol_20d.
    `built_vol`/`metrics_vol` must share columns [date, vol]. Tolerance is relative.
    """
    joined = (
        built_vol.rename({"vol": "_built"})
        .join(metrics_vol.rename({"vol": "_ref"}), on="date", how="inner")
        .drop_nulls()
    )
    if joined.height == 0:
        return {"n_overlap": 0, "pct_within_tolerance": None, "pass": False}
    rel_diff = (joined["_built"] - joined["_ref"]).abs() / joined["_ref"].abs().clip(
        1e-9
    )
    within = (rel_diff <= tolerance).sum()
    pct = within / joined.height
    return {
        "n_overlap": joined.height,
        "pct_within_tolerance": float(pct),
        "pass": bool(pct >= 0.90),
    }


def reconcile_returns_yfinance(
    built_returns: pl.DataFrame, yf_returns: pl.DataFrame
) -> dict:
    """Correlate our continuous log returns against a yfinance continuous-futures
    proxy's daily log returns. Pass condition: correlation > 0.98.
    """
    joined = (
        built_returns.rename({"ret": "_built"})
        .join(yf_returns.rename({"ret": "_yf"}), on="date", how="inner")
        .unique(subset=["date"])
        .drop_nulls()
    )
    joined = joined.filter(pl.col("_built").is_finite() & pl.col("_yf").is_finite())
    if joined.height < 30:
        return {"n_overlap": joined.height, "corr": None, "pass": False}
    corr = float(
        np.corrcoef(joined["_built"].to_numpy(), joined["_yf"].to_numpy())[0, 1]
    )
    return {"n_overlap": joined.height, "corr": corr, "pass": bool(corr > 0.98)}


# ---------------------------------------------------------------------------
# Phase 0.7 -- stale-bar audit
# ---------------------------------------------------------------------------


def stale_bar_runs(close: np.ndarray) -> dict:
    """Count runs of consecutive identical close prices. Returns run-length stats."""
    close = np.asarray(close, dtype=float)
    if len(close) < 2:
        return {"n_runs": 0, "max_run": 0, "n_stale_days": 0}
    same = close[1:] == close[:-1]
    run_lengths = []
    cur = 0
    for s in same:
        if s:
            cur += 1
        else:
            if cur > 0:
                run_lengths.append(cur + 1)
            cur = 0
    if cur > 0:
        run_lengths.append(cur + 1)
    return {
        "n_runs": len(run_lengths),
        "max_run": int(max(run_lengths)) if run_lengths else 0,
        "n_stale_days": int(sum(r - 1 for r in run_lengths)),
    }


# ---------------------------------------------------------------------------
# Section 7 -- futures cost model
# ---------------------------------------------------------------------------

# Verified against CME/ICE exchange contract specs.
CONTRACT_SPECS = {
    "CL": {"tick": 0.01, "tick_value": 10.0, "commission_per_contract": 2.50},
    "BZ": {"tick": 0.01, "tick_value": 10.0, "commission_per_contract": 2.50},
    "NG": {"tick": 0.001, "tick_value": 10.0, "commission_per_contract": 2.50},
    "HO": {"tick": 0.0001, "tick_value": 4.20, "commission_per_contract": 2.50},
    "RB": {"tick": 0.0001, "tick_value": 4.20, "commission_per_contract": 2.50},
    "GC": {"tick": 0.10, "tick_value": 10.0, "commission_per_contract": 2.50},
    "SI": {"tick": 0.005, "tick_value": 25.0, "commission_per_contract": 2.50},
    "PL": {"tick": 0.10, "tick_value": 5.0, "commission_per_contract": 2.50},
    "PA": {"tick": 0.10, "tick_value": 10.0, "commission_per_contract": 2.50},
    "ZC": {"tick": 0.0025, "tick_value": 12.50, "commission_per_contract": 2.50},
    "ZW": {"tick": 0.0025, "tick_value": 12.50, "commission_per_contract": 2.50},
    "KE": {"tick": 0.0025, "tick_value": 12.50, "commission_per_contract": 2.50},
    "ZS": {"tick": 0.0025, "tick_value": 12.50, "commission_per_contract": 2.50},
    "ZL": {"tick": 0.0001, "tick_value": 6.0, "commission_per_contract": 2.50},
    "ZM": {"tick": 0.10, "tick_value": 10.0, "commission_per_contract": 2.50},
    "ES": {"tick": 0.25, "tick_value": 12.50, "commission_per_contract": 2.50},
}

# Thin products get a wider slippage multiplier than the 1-tick baseline.
THIN_PRODUCTS = {"PA", "PL", "KE"}


def round_turn_cost_per_contract(product: str, tick_multiplier: float = 1.0) -> float:
    """Round-turn commission + fees + `tick_multiplier` ticks of slippage, in
    dollars per contract. Thin products (PA/PL/KE) get 2x the slippage multiplier
    to reflect wider realistic spreads.
    """
    spec = CONTRACT_SPECS[product]
    mult = tick_multiplier * (2.0 if product in THIN_PRODUCTS else 1.0)
    slippage = spec["tick_value"] * mult
    return spec["commission_per_contract"] + slippage


def cost_per_unit_notional(
    product: str, price: float, tick_multiplier: float = 1.0
) -> float:
    """Round-turn cost expressed as a fraction of contract notional, for use as an
    add_portfolio_costs-style `taker_fee`-equivalent. `price` is the current futures
    price (used to approximate notional via tick_value/tick as the point value).
    """
    spec = CONTRACT_SPECS[product]
    point_value = spec["tick_value"] / spec["tick"]
    notional = abs(price) * point_value
    cost = round_turn_cost_per_contract(product, tick_multiplier)
    return cost / notional if notional > 0 else 0.0


# ---------------------------------------------------------------------------
# Phase 1 -- tail atlas machinery not already covered by dist_lib5/6.
# ---------------------------------------------------------------------------


def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation of x at lags 1..max_lag (lag-0 excluded)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = np.sum(x**2)
    out = np.empty(max_lag)
    for lag in range(1, max_lag + 1):
        out[lag - 1] = np.sum(x[:-lag] * x[lag:]) / denom if denom > 0 else np.nan
    return out


def ljung_box_test(x: np.ndarray, lags: int) -> dict:
    """Ljung-Box portmanteau test for autocorrelation up to `lags`. No
    statsmodels dependency in this repo (sec 2), so implemented directly from
    the textbook formula: Q = n(n+2) * sum_{k=1}^{lags} rho_k^2 / (n-k),
    Q ~ chi2(lags) under H0 of no autocorrelation.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    rho = acf(x, lags)
    k = np.arange(1, lags + 1)
    q_stat = n * (n + 2) * np.sum(rho**2 / (n - k))
    p_value = float(st.chi2.sf(q_stat, df=lags))
    return {"Q": float(q_stat), "p_value": p_value, "lags": lags, "n": n}


def leverage_correlation(
    returns: np.ndarray, vol_next: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> dict:
    """corr(r_t, sigma_{t+1}) with a bootstrap CI. Negative in equities (the
    leverage effect: vol rises when price falls); commodities are hypothesised
    to show the opposite sign (sec 3.1's "inverse leverage effect") because a
    price spike signals scarcity, not distress.
    """
    r, s = np.asarray(returns, dtype=float), np.asarray(vol_next, dtype=float)
    mask = np.isfinite(r) & np.isfinite(s)
    r, s = r[mask], s[mask]
    if len(r) < 30:
        return {"corr": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n": len(r)}
    corr = float(np.corrcoef(r, s)[0, 1])
    rng = np.random.default_rng(seed)
    n = len(r)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = np.corrcoef(r[idx], s[idx])[0, 1]
    ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return {"corr": corr, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": n}


def samuelson_effect(
    vol: np.ndarray, dte: np.ndarray, bin_edges: list[int] | None = None
) -> dict:
    """Mean realised vol bucketed by days-to-expiry. Rising vol as dte -> 0 is
    the Samuelson effect: a contract's price becomes more sensitive to news as
    delivery approaches and time to arbitrage the news away shrinks.
    """
    if bin_edges is None:
        bin_edges = [0, 5, 10, 20, 30, 45, 60, 90, 120, 999]
    v, d = np.asarray(vol, dtype=float), np.asarray(dte, dtype=float)
    mask = np.isfinite(v) & np.isfinite(d) & (d >= 0)
    v, d = v[mask], d[mask]
    out = []
    for lo, hi in itertools.pairwise(bin_edges):
        in_bin = (d >= lo) & (d < hi)
        if in_bin.sum() < 5:
            continue
        out.append(
            {
                "dte_lo": lo,
                "dte_hi": hi,
                "mean_vol": float(np.mean(v[in_bin])),
                "n": int(in_bin.sum()),
            }
        )
    return {"buckets": out}


def month_of_year_stats(dates: list, returns: np.ndarray) -> dict:
    """Mean return and vol by calendar month (NG heating season, grain
    growing/harvest seasonality, sec 3.1)."""
    months = np.array([d.month for d in dates])
    r = np.asarray(returns, dtype=float)
    out = []
    for m in range(1, 13):
        sel = r[months == m]
        sel = sel[np.isfinite(sel)]
        if len(sel) < 5:
            continue
        out.append(
            {
                "month": m,
                "mean_return": float(np.mean(sel)),
                "vol": float(np.std(sel)),
                "n": len(sel),
            }
        )
    return {"months": out}


def day_of_week_stats(dates: list, returns: np.ndarray) -> dict:
    weekdays = np.array([d.weekday() for d in dates])
    r = np.asarray(returns, dtype=float)
    out = []
    for wd in range(5):
        sel = r[weekdays == wd]
        sel = sel[np.isfinite(sel)]
        if len(sel) < 5:
            continue
        out.append(
            {
                "weekday": wd,
                "mean_return": float(np.mean(sel)),
                "vol": float(np.std(sel)),
                "n": len(sel),
            }
        )
    return {"weekdays": out}


# Named events for tail-atlas annotation (sec 4 Phase 1). Windows are
# inclusive [start, end] in ISO date strings.
NAMED_EVENTS: list[dict[str, str | list[str]]] = [
    {
        "name": "2011 Libya supply shock",
        "start": "2011-02-15",
        "end": "2011-04-30",
        "products": ["CL", "BZ"],
    },
    {
        "name": "2014-15 OPEC collapse",
        "start": "2014-11-01",
        "end": "2015-03-01",
        "products": ["CL", "BZ", "RB", "HO"],
    },
    {
        "name": "2020-04-20 negative WTI",
        "start": "2020-04-15",
        "end": "2020-04-25",
        "products": ["CL"],
    },
    {
        "name": "2021-02 Uri freeze",
        "start": "2021-02-10",
        "end": "2021-02-20",
        "products": ["NG", "CL"],
    },
    {
        "name": "2022-02 Ukraine invasion",
        "start": "2022-02-20",
        "end": "2022-04-01",
        "products": ["NG", "BZ", "CL", "ZW", "ZC"],
    },
    {
        "name": "2022 nickel-style squeeze era (energy crunch)",
        "start": "2022-08-01",
        "end": "2022-09-15",
        "products": ["NG"],
    },
    {
        "name": "2023-24 normalisation",
        "start": "2023-01-01",
        "end": "2024-06-30",
        "products": PRODUCTS,
    },
]


def events_in_window(product: str, start: str, end: str) -> list[dict]:
    """Named events overlapping [start, end] that name `product` (or apply to
    all products)."""
    from datetime import date as _date

    s = _date.fromisoformat(start)
    e = _date.fromisoformat(end)
    out = []
    for ev in NAMED_EVENTS:
        if product not in ev["products"]:
            continue
        ev_start, ev_end = str(ev["start"]), str(ev["end"])
        ev_s, ev_e = _date.fromisoformat(ev_start), _date.fromisoformat(ev_end)
        if ev_s <= e and ev_e >= s:
            out.append(ev)
    return out


# ---------------------------------------------------------------------------
# Phase 2 -- unconditional density selection. The four `densities/*.py`
# family modules (ged, hansen_skewt, johnsonsu, nig) and dist_lib6's spliced-
# EVT fit are shape-only: they assume the input is already standardized
# (mean ~0, var ~1) and expose logpdf/ppf/es on that standardized scale, with
# no cdf. `numerical_pit` gives every such family a shared, vectorized way to
# get PIT values (and therefore a KS calibration test) from nothing but its
# logpdf, via one cumulative-trapezoid integration per fit.
# ---------------------------------------------------------------------------


def numerical_cdf_grid(
    logpdf_fn, grid_lo: float = -20.0, grid_hi: float = 20.0, n_grid: int = 4001
) -> dict:
    """Build a numerical CDF over `grid_lo..grid_hi` from a logpdf callable
    (z: np.ndarray) -> np.ndarray, via cumulative trapezoidal integration.
    Returns {"grid": z-grid, "cdf": F(z)}; F is normalised to end at 1.0 so a
    logpdf that isn't perfectly normalised over the truncated grid still
    yields a valid PIT.
    """
    grid = np.linspace(grid_lo, grid_hi, n_grid)
    pdf = np.exp(logpdf_fn(grid))
    pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
    cdf = np.concatenate([[0.0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * np.diff(grid))])
    total = cdf[-1]
    if total > 0:
        cdf = cdf / total
    return {"grid": grid, "cdf": cdf}


def numerical_pit(
    logpdf_fn, z_values: np.ndarray, grid_lo: float = -20.0, grid_hi: float = 20.0
) -> np.ndarray:
    """PIT values for `z_values` under a shape-only family's logpdf, via
    `numerical_cdf_grid` + linear interpolation."""
    g = numerical_cdf_grid(logpdf_fn, grid_lo, grid_hi)
    z = np.clip(z_values, grid_lo, grid_hi)
    return np.interp(z, g["grid"], g["cdf"])


def numerical_ppf(
    logpdf_fn, u: np.ndarray, grid_lo: float = -20.0, grid_hi: float = 20.0
) -> np.ndarray:
    """Inverse-CDF (quantile function) for `u` under a shape-only family's
    logpdf, via one `numerical_cdf_grid` build + interpolation -- the
    `numerical_pit` inverse. Exists because nig.ppf has no closed form and
    root-finds its CDF per point (~50ms/point measured); a 20,000-point
    Monte Carlo draw for portfolio simulation would take ~15+ minutes per
    asset called that way. This is O(n_grid) once, then O(len(u)), regardless
    of how many draws are needed.
    """
    g = numerical_cdf_grid(logpdf_fn, grid_lo, grid_hi)
    # cdf must be strictly increasing for interp's x-array; de-duplicate flat
    # (near-zero-density) regions by keeping only strictly increasing points.
    cdf, grid = g["cdf"], g["grid"]
    keep = np.concatenate([[True], np.diff(cdf) > 0])
    return np.interp(np.clip(u, cdf[keep][0], cdf[keep][-1]), cdf[keep], grid[keep])


# ---------------------------------------------------------------------------
# Phase 3 -- GJR + zoo-family two-stage fitting. dist_lib6.fit_garch_zoo_two_stage
# / rolling_garch_forecast_zoo already do this for plain GARCH(1,1); GJR's own
# leverage recursion (dist_lib5._gjr_variance_path) needs the same two-stage
# composition (fit the variance process under a normal first stage, then fit
# an arbitrary innovation family's shape to the standardized residuals as a
# second stage) that dist_lib6 never needed since notebook 6 was GARCH-only.
# ---------------------------------------------------------------------------


def fit_gjr_zoo_two_stage(r: np.ndarray, family_module) -> dict | None:
    """GJR(1,1,1) variance process (normal-innovation first stage, via
    dist_lib5.fit_gjr11) + an arbitrary innovation family's shape fit to the
    standardized residuals (second stage) -- the GJR analogue of
    dist_lib6.fit_garch_zoo_two_stage.
    """
    import dist_lib5 as L5

    gjr_fit = L5.fit_gjr11(r, innovation="normal")
    if gjr_fit is None:
        return None
    omega, alpha, gamma, beta = (
        gjr_fit["omega"],
        gjr_fit["alpha"],
        gjr_fit["gamma"],
        gjr_fit["beta"],
    )
    uncond = omega / max(1 - alpha - gamma / 2.0 - beta, 1e-6)
    sig2 = L5._gjr_variance_path(omega, alpha, gamma, beta, r, uncond)
    if np.any(sig2 <= 0) or not np.all(np.isfinite(sig2)):
        return None
    z = r / np.sqrt(sig2)
    shape = family_module.fit(z)
    if shape is None:
        return None
    last_shock = r[-1] ** 2
    lev_last = gamma * last_shock if r[-1] < 0.0 else 0.0
    next_sig2 = omega + alpha * last_shock + lev_last + beta * sig2[-1]
    return {
        "omega": float(omega),
        "alpha": float(alpha),
        "gamma": float(gamma),
        "beta": float(beta),
        "shape": tuple(float(s) for s in shape),
        "next_var": float(next_sig2),
        "family": family_module.NAME,
    }


def rolling_gjr_forecast_zoo(
    returns: np.ndarray,
    refit_every: int,
    min_train: int,
    family_module,
    max_train: int = 1500,
) -> tuple[np.ndarray, list[dict]]:
    """Refit-then-forward-fill rolling variance forecast for the GJR+zoo
    two-stage fit, structurally identical to
    dist_lib6.rolling_garch_forecast_zoo but propagating the GJR leverage
    recursion between refits instead of the plain-GARCH one."""
    n = len(returns)
    forecast = np.full(n, np.nan)
    fits: list[dict] = []
    fit = None
    sig2_state = np.nan
    for t in range(n):
        if t >= min_train and t % refit_every == 0:
            start = max(0, t - max_train)
            window = returns[start:t]
            new_fit = fit_gjr_zoo_two_stage(window, family_module)
            if new_fit is not None:
                fit = new_fit
                sig2_state = fit["omega"] / max(
                    1 - fit["alpha"] - fit["gamma"] / 2.0 - fit["beta"], 1e-6
                )
                fits.append({"t": t, **fit})
        if fit is not None:
            forecast[t] = sig2_state
            if t + 1 < n and np.isfinite(returns[t]):
                shock = returns[t] ** 2
                lev = fit["gamma"] * shock if returns[t] < 0.0 else 0.0
                sig2_state = (
                    fit["omega"] + fit["alpha"] * shock + lev + fit["beta"] * sig2_state
                )
    return forecast, fits


def zoo_es_forecast_upper(
    variance_forecast: np.ndarray,
    fits: list[dict],
    family_module,
    q: float,
    n_points: int = 30,
) -> np.ndarray:
    """Upper-tail ES at exceedance probability q for a zoo family:
    sigma_t * (1/q) * integral_{1-q}^{1} ppf(u) du, via numerical
    integration of the family's own ppf (its `es(q, shape)` is defined as
    the average of the BOTTOM q-fraction only -- see e.g. densities/ged.py --
    so it cannot be reused directly for the top tail on a family that may be
    skewed). One integral per refit segment (not per bar), then broadcast
    like `dist_lib6.zoo_quantile_forecast`'s own segment-batched convention.

    n_points defaults low (30, not the usual few-thousand-point grid) because
    nig.ppf has no closed form and root-finds its CDF numerically per point
    (~50ms/point measured) -- 1000 points x 6 refits x 2 variance processes x
    2 ES levels was ~20 minutes per product on NIG alone. 30 points is a
    coarser mean-of-the-tail estimate, adequate for an ES *forecast* (as
    opposed to a metric requiring tight numerical precision).
    """
    n = len(variance_forecast)
    out = np.full(n, np.nan)
    if not fits:
        return out
    u = np.linspace(1 - q, 1 - 1e-6, n_points)
    for i, f in enumerate(fits):
        start = f["t"]
        end = fits[i + 1]["t"] if i + 1 < len(fits) else n
        v = variance_forecast[start:end]
        mask = np.isfinite(v) & (v > 0)
        sigma = np.sqrt(v[mask])
        upper_es_z = float(np.mean(family_module.ppf(u, f["shape"])))
        idx = np.arange(start, end)[mask]
        out[idx] = sigma * upper_es_z
    return out


def spliced_evt_var_es_forecast(
    variance_forecast: np.ndarray, spliced_fits: list[dict], q: float
) -> dict:
    """VaR/ES forecast at exceedance level q from a rolling spliced-EVT fit
    path (dist_lib6.rolling_spliced_evt_fits' output). Reuses each fit's own
    GPD tail sub-fit (dist_lib5.gpd_var_es) -- valid because notebook 8 only
    ever needs quantile levels {0.5%, 1%, 2.5%, 5%} (both tails), all inside
    the spliced fit's own tail_frac=0.10 region, so the GPD tail piece alone
    (not the KDE interior) is exactly what governs these levels.
    """
    import dist_lib5 as L5

    sigma = np.sqrt(np.where(variance_forecast > 0, variance_forecast, np.nan))
    n = len(sigma)
    var_out, es_out = np.full(n, np.nan), np.full(n, np.nan)
    if not spliced_fits:
        return {"var": var_out, "es": es_out}
    tail = "lower" if q < 0.5 else "upper"
    exceed_q = q if q < 0.5 else 1.0 - q
    for i, f in enumerate(spliced_fits):
        start = f["t"]
        end = spliced_fits[i + 1]["t"] if i + 1 < len(spliced_fits) else n
        tail_fit = f["spliced"][f"{tail}_fit"]
        z_q, es_q = L5.gpd_var_es(tail_fit, exceed_q)
        if not (np.isfinite(z_q) and np.isfinite(es_q)):
            continue
        seg = np.arange(start, end)
        sign = -1.0 if tail == "lower" else 1.0
        var_out[seg] = sign * sigma[seg] * z_q
        es_out[seg] = sign * sigma[seg] * es_q
    return {"var": var_out, "es": es_out}


# ---------------------------------------------------------------------------
# Phase 4 -- inventory/seasonal/macro conditioning.
# ---------------------------------------------------------------------------

# Delivery-cycle seasonal windows, by product. NG's heating season is
# Nov-Mar demand-driven; grains follow the US growing calendar
# (planting/growing/harvest); energy/metals outside NG have no comparably
# strong calendar-driven demand cycle and are left unclassified ("na").
_SEASONAL_WINDOWS = {
    "NG": {"heating_season": [11, 12, 1, 2, 3]},
    "ZC": {"planting": [4, 5], "growing": [6, 7, 8], "harvest": [9, 10, 11]},
    "ZW": {
        "planting": [9, 10],
        "growing": [3, 4, 5],
        "harvest": [6, 7],
    },  # winter wheat cycle
    "ZS": {"planting": [4, 5], "growing": [6, 7, 8], "harvest": [9, 10]},
    "ZL": {"planting": [4, 5], "growing": [6, 7, 8], "harvest": [9, 10]},
    "ZM": {"planting": [4, 5], "growing": [6, 7, 8], "harvest": [9, 10]},
    "KE": {"planting": [9, 10], "growing": [3, 4, 5], "harvest": [6, 7]},
}


def term_structure_state(curve: pl.DataFrame) -> pl.DataFrame:
    """Annualised F1->F2 roll-yield slope and a backwardation/contango state
    label, from a Phase 0 curve frame (needs close_f1/dte_f1/close_f2/dte_f2).
    Backwardation (state="backwardation", slope<0): F2 cheaper than F1, the
    low-inventory / high-convenience-yield state theory predicts should carry
    higher vol and a fatter right tail. Contango (slope>0): F2 richer than F1.
    """
    out = curve.with_columns(
        (
            (pl.col("close_f2") / pl.col("close_f1")).log()
            / (pl.col("dte_f2") - pl.col("dte_f1")).clip(lower_bound=1)
            * 365.0
        ).alias("roll_slope_annualized")
    )
    out = out.with_columns(
        pl.when(pl.col("roll_slope_annualized") < 0)
        .then(pl.lit("backwardation"))
        .when(pl.col("roll_slope_annualized") > 0)
        .then(pl.lit("contango"))
        .otherwise(None)
        .alias("term_structure_state")
    )
    return out.select(["date", "roll_slope_annualized", "term_structure_state"])


def seasonal_state(dates: list, product: str) -> list[str]:
    """Per-date seasonal label for `product` (sec _SEASONAL_WINDOWS above).
    Products without a defined calendar cycle get "na" for every date."""
    windows = _SEASONAL_WINDOWS.get(product)
    if windows is None:
        return ["na"] * len(dates)
    month_to_state = {}
    for state, months in windows.items():
        for m in months:
            month_to_state[m] = state
    return [month_to_state.get(d.month, "off_season") for d in dates]


def macro_regime(
    fred_frames: dict[str, pl.DataFrame], dates: pl.Series, lag_days: int = 1
) -> pl.DataFrame:
    """VIX tercile, T10Y2Y sign, and DFF level, each lagged `lag_days`
    business days before being joined onto `dates` (no lookahead: FRED is
    published with a lag and this repo's own discipline requires lagging it
    at least one business day regardless, sec 1.9/8).

    `fred_frames` = {"VIXCLS": df, "T10Y2Y": df, "DFF": df}, each with columns
    [date, <SERIES>] as loaded directly from data/market/fred/*.parquet.
    """
    base = pl.DataFrame({"date": dates}).with_columns(pl.col("date").cast(pl.Date))

    vix = fred_frames["VIXCLS"].with_columns(pl.col("date").cast(pl.Date)).sort("date")
    vix = vix.with_columns(pl.col("VIXCLS").shift(lag_days).alias("vix_lagged"))
    terciles = vix["vix_lagged"].drop_nulls().to_numpy()
    t1, t2 = (
        np.percentile(terciles, [33.3, 66.7])
        if len(terciles) > 10
        else (np.nan, np.nan)
    )
    vix = vix.with_columns(
        pl.when(pl.col("vix_lagged") < t1)
        .then(pl.lit("low_vol"))
        .when(pl.col("vix_lagged") < t2)
        .then(pl.lit("mid_vol"))
        .otherwise(pl.lit("high_vol"))
        .alias("vix_regime")
    )

    t10y2y = (
        fred_frames["T10Y2Y"].with_columns(pl.col("date").cast(pl.Date)).sort("date")
    )
    t10y2y = t10y2y.with_columns(
        pl.col("T10Y2Y").shift(lag_days).alias("t10y2y_lagged")
    )
    t10y2y = t10y2y.with_columns(
        pl.when(pl.col("t10y2y_lagged") < 0)
        .then(pl.lit("inverted"))
        .otherwise(pl.lit("normal"))
        .alias("yield_curve_regime")
    )

    dff = fred_frames["DFF"].with_columns(pl.col("date").cast(pl.Date)).sort("date")
    dff = dff.with_columns(pl.col("DFF").shift(lag_days).alias("dff_lagged"))
    dff_terciles = dff["dff_lagged"].drop_nulls().to_numpy()
    d1, d2 = (
        np.percentile(dff_terciles, [33.3, 66.7])
        if len(dff_terciles) > 10
        else (np.nan, np.nan)
    )
    dff = dff.with_columns(
        pl.when(pl.col("dff_lagged") < d1)
        .then(pl.lit("low_rate"))
        .when(pl.col("dff_lagged") < d2)
        .then(pl.lit("mid_rate"))
        .otherwise(pl.lit("high_rate"))
        .alias("dff_regime")
    )

    out = base.join(vix.select(["date", "vix_regime"]), on="date", how="left")
    out = out.join(t10y2y.select(["date", "yield_curve_regime"]), on="date", how="left")
    out = out.join(dff.select(["date", "dff_regime"]), on="date", how="left")
    return out


def kupiec_by_state(
    hits: np.ndarray, states: list[str] | np.ndarray, expected_rate: float
) -> dict:
    """Kupiec unconditional-coverage test run separately within each state
    label, plus the pooled (unconditional) test, for a direct
    state-conditioned-vs-unconditional coverage comparison (Gate CI)."""
    import distributions as dist

    hits = np.asarray(hits)
    states = np.asarray(states)
    out: dict = {}
    for state in sorted(set(states.tolist())):
        mask = states == state
        h = hits[mask]
        if len(h) < 20:
            out[state] = {"n": len(h), "kupiec_p": None, "observed_rate": None}
            continue
        _, p = dist.kupiec_test(h, expected_rate)
        out[state] = {
            "n": len(h),
            "kupiec_p": float(p),
            "observed_rate": float(np.mean(h)),
        }
    _, p_all = dist.kupiec_test(hits, expected_rate)
    out["_pooled"] = {
        "n": len(hits),
        "kupiec_p": float(p_all),
        "observed_rate": float(np.mean(hits)),
    }
    return out


# ---------------------------------------------------------------------------
# Phase 7 -- the risk engine. Family selection is driven by Phase 2/3's own
# fitted results (stored as JSON, read here -- never hardcoded), per-product
# density parameters are fit fresh on the full return series for a "current"
# VaR/ES estimate, and portfolio-level risk is assembled via three explicit
# dependence assumptions (empirical bootstrap, Gaussian copula, t-copula) so
# the Gaussian copula's tail-dependence understatement (sec 3.4) is a
# side-by-side comparison, not an assertion.
# ---------------------------------------------------------------------------


class RiskModel:
    """A fitted per-product density, with VaR/ES/simulate/stress. `kind` is
    "loc_scale" (normal/t: params already on the return scale) or
    "standardized" (ged/nig/johnsonsu/hansen_skewt/spliced_evt: shape fit on
    (r-mean)/std, so VaR/ES need the mean/std Jacobian applied explicitly).
    """

    def __init__(
        self,
        product: str,
        family: str,
        kind: str,
        mean: float,
        std: float,
        params: tuple,
    ):
        self.product = product
        self.family = family
        self.kind = kind
        self.mean = mean
        self.std = std
        self.params = params

    def _lower_q(self, alpha: float) -> float:
        import distributions as dist

        if self.kind == "loc_scale":
            d = dist.frozen_dist(self.family, self.params)
            return float(d.ppf(alpha))
        import densities

        mod = densities.REGISTRY[self.family]
        return float(self.mean + self.std * mod.ppf(alpha, self.params))

    def _lower_es(self, alpha: float) -> float:

        if self.family == "normal":
            from scipy import stats as st

            z = st.norm.ppf(alpha)
            return float(self.mean - self.std * st.norm.pdf(z) / alpha)
        if self.family == "t":
            from scipy import stats as st

            df = self.params[0]
            z = st.t.ppf(alpha, df=df)
            es_z = -st.t.pdf(z, df=df) * (df + z**2) / (df - 1) / alpha
            return float(self.mean + self.std * es_z)
        import densities

        mod = densities.REGISTRY[self.family]
        return float(self.mean + self.std * mod.es(alpha, self.params))

    def var(self, alpha: float, horizon: int = 1) -> float:
        """Value at Risk: the alpha-quantile loss, positive horizon scaled by
        sqrt(horizon) (the standard, and limited, iid scaling -- documented
        here rather than silently assumed)."""
        q = self._lower_q(alpha)
        return float(-q * np.sqrt(horizon))

    def es(self, alpha: float, horizon: int = 1) -> float:
        """Expected Shortfall: the average loss beyond VaR at level alpha."""
        e = self._lower_es(alpha)
        return float(-e * np.sqrt(horizon))

    def var_conditional(self, alpha: float, sigma_t: float, horizon: int = 1) -> float:
        """VaR using a caller-supplied *current* volatility `sigma_t`
        (e.g. from `ewma_vol`) instead of the model's static full-sample std
        -- the shape (skew/kurtosis) stays as fitted, only the scale is
        time-varying. This is the "conditioning" half of `fit_risk_model`'s
        spec: a full GARCH refit per day is what Phase 3 already validated;
        this is the cheap, still-causal middle ground the risk *engine* uses
        so its own VaR isn't frozen at one full-sample volatility level for
        years at a time.
        """
        q = self._lower_q_at_scale(alpha, sigma_t)
        return float(-q * np.sqrt(horizon))

    def es_conditional(self, alpha: float, sigma_t: float, horizon: int = 1) -> float:
        e = self._lower_es_at_scale(alpha, sigma_t)
        return float(-e * np.sqrt(horizon))

    def _lower_q_at_scale(self, alpha: float, sigma_t: float) -> float:
        base = self._lower_q(alpha)
        return (
            float(self.mean + (base - self.mean) * (sigma_t / self.std))
            if self.std > 0
            else base
        )

    def _lower_es_at_scale(self, alpha: float, sigma_t: float) -> float:
        base = self._lower_es(alpha)
        return (
            float(self.mean + (base - self.mean) * (sigma_t / self.std))
            if self.std > 0
            else base
        )

    def simulate(self, n: int, seed: int = 0) -> np.ndarray:
        """Draw n i.i.d. returns from the fitted marginal (used to build
        portfolio-level Monte Carlo scenarios and copula transforms)."""
        rng = np.random.default_rng(seed)
        u = rng.uniform(1e-6, 1 - 1e-6, n)
        return self.ppf_from_u(u)

    def ppf_from_u(self, u: np.ndarray) -> np.ndarray:
        import distributions as dist

        if self.kind == "loc_scale":
            d = dist.frozen_dist(self.family, self.params)
            return np.asarray(d.ppf(u))
        import densities

        mod = densities.REGISTRY[self.family]
        # nig.ppf has no closed form and root-finds per point (~50ms/point) --
        # far too slow for a Monte Carlo draw of thousands of points, so a
        # numerical grid-inversion is used for every standardized family here
        # (fast, and consistent regardless of which family is fitted).
        z = numerical_ppf(lambda zz: mod.logpdf(zz, self.params), u)
        return self.mean + self.std * z

    def cdf_from_x(self, x: np.ndarray) -> np.ndarray:
        import distributions as dist

        if self.kind == "loc_scale":
            d = dist.frozen_dist(self.family, self.params)
            return np.asarray(d.cdf(x))
        import densities

        mod = densities.REGISTRY[self.family]
        z = (x - self.mean) / self.std
        return numerical_pit(lambda zz: mod.logpdf(zz, self.params), z)

    def stress(self, scenario_returns: np.ndarray) -> dict:
        """Replay a named historical event's realized return path against
        this model: cumulative P&L and the worst single-day loss, both on the
        return scale (a $1 notional). `scenario_returns` is a log-return
        path (e.g. the product's own returns during a Phase 1 named event
        window)."""
        scenario_returns = scenario_returns[np.isfinite(scenario_returns)]
        if len(scenario_returns) == 0:
            return {"cum_return": None, "worst_day": None, "n_days": 0}
        return {
            "cum_return": float(np.sum(scenario_returns)),
            "worst_day": float(np.min(scenario_returns)),
            "n_days": len(scenario_returns),
        }


def ewma_vol(
    returns: np.ndarray, lam: float = 0.94, seed_window: int = 20
) -> np.ndarray:
    """Causal RiskMetrics-style EWMA volatility path: sigma2_t = lam *
    sigma2_{t-1} + (1-lam) * r_{t-1}^2, seeded by the sample variance of the
    first `seed_window` observations. sigma_t (the output) is only ever a
    function of r_0..r_{t-1} -- never r_t itself -- so it is safe to use as
    "today's" conditioning volatility for a VaR forecast made before the
    close.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    sigma2 = np.full(n, np.nan)
    if n < seed_window + 1:
        return np.sqrt(np.maximum(sigma2, 0))
    sigma2[seed_window] = float(np.var(r[:seed_window]))
    for t in range(seed_window + 1, n):
        sigma2[t] = lam * sigma2[t - 1] + (1 - lam) * r[t - 1] ** 2
    return np.sqrt(np.maximum(sigma2, 0))


def fit_risk_model(returns: np.ndarray, product: str, family: str) -> RiskModel | None:
    """Fit `family` fresh on the full `returns` series. `family` is expected
    to come from the caller's own family-selection JSON (Phase 2/3 output) --
    this function does not choose a family itself, per sec 4 Phase 7's
    "driven by Phase 2/3 results, not hardcoded" requirement.
    """
    import distributions as dist

    returns = returns[np.isfinite(returns)]
    if len(returns) < 100:
        return None
    if family in ("normal", "t"):
        fit_fn = dist._fit_normal if family == "normal" else dist._fit_t
        params = fit_fn(returns)
        if params is None:
            return None
        return RiskModel(
            product,
            family,
            "loc_scale",
            mean=float(np.mean(returns)),
            std=float(np.std(returns)),
            params=params,
        )
    import densities

    if family == "spliced_evt":
        return None  # not supported as a standalone RiskModel (needs a variance process); use Phase 3's own path instead
    mod = densities.REGISTRY[family]
    mean, std = float(np.mean(returns)), float(np.std(returns))
    z = (returns - mean) / std
    shape = mod.fit(z)
    if shape is None:
        return None
    return RiskModel(product, family, "standardized", mean=mean, std=std, params=shape)


def empirical_lower_tail_dependence(
    u_i: np.ndarray, u_j: np.ndarray, q: float = 0.1
) -> float:
    """Empirical lower-tail dependence coefficient at threshold q:
    P(U_j <= q | U_i <= q), from rank-transformed (pseudo-uniform) marginals.
    """
    mask_i = u_i <= q
    if mask_i.sum() == 0:
        return float("nan")
    return float(np.mean(u_j[mask_i] <= q))


def to_pseudo_uniform(x: np.ndarray) -> np.ndarray:
    """Rank-transform to pseudo-uniform marginals (empirical CDF ranks in
    (0,1)), the standard first step before any copula fit or tail-dependence
    estimate."""
    order = np.argsort(np.argsort(x))
    return (order + 0.5) / len(x)


def portfolio_risk(
    models: dict[str, RiskModel],
    weights: dict[str, float],
    dependence: str = "empirical",
    historical_returns: dict[str, np.ndarray] | None = None,
    n_sims: int = 20000,
    t_df: float = 5.0,
    seed: int = 0,
) -> dict:
    """Portfolio-level VaR/ES via Monte Carlo, under one of three explicit
    dependence assumptions:

    - "empirical": bootstrap-resample historical JOINT return vectors
      (requires `historical_returns`, aligned by date/index across products)
      -- captures whatever dependence (incl. tail dependence) is actually in
      the data.
    - "gaussian": simulate a Gaussian copula from the empirical (pseudo-
      uniform) correlation matrix, transform each margin through its own
      RiskModel.ppf_from_u -- has ZERO asymptotic tail dependence by
      construction, regardless of the correlation used.
    - "t": simulate a Student-t copula (same correlation, `t_df` degrees of
      freedom) -- has positive tail dependence, growing as t_df falls.

    Reports portfolio VaR/ES at 1%/5%, plus empirical lower-tail dependence
    per pair for both the real (empirical-dependence) simulation and the
    Gaussian-copula simulation, so any understatement (sec 3.4) is a
    reported, quantified comparison, not an assertion.
    """
    products = list(models.keys())
    n_assets = len(products)
    rng = np.random.default_rng(seed)
    w = np.array([weights[p] for p in products])

    if dependence == "empirical":
        if historical_returns is None:
            raise ValueError("empirical dependence requires historical_returns")
        mat = np.column_stack([historical_returns[p] for p in products])
        mat = mat[np.all(np.isfinite(mat), axis=1)]
        idx = rng.integers(0, len(mat), n_sims)
        sims = mat[idx]
    else:
        if historical_returns is not None:
            mat = np.column_stack([historical_returns[p] for p in products])
            mat = mat[np.all(np.isfinite(mat), axis=1)]
            pseudo = np.column_stack(
                [to_pseudo_uniform(mat[:, i]) for i in range(n_assets)]
            )
            corr = np.corrcoef(pseudo, rowvar=False)
        else:
            corr = np.eye(n_assets)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        chol = np.linalg.cholesky(corr + 1e-8 * np.eye(n_assets))
        if dependence == "gaussian":
            z = rng.standard_normal((n_sims, n_assets)) @ chol.T
            u = _norm_cdf(z)
        elif dependence == "t":
            g = rng.standard_normal((n_sims, n_assets)) @ chol.T
            chi2 = rng.chisquare(t_df, n_sims)
            t_draws = g / np.sqrt(chi2 / t_df)[:, None]
            u = _t_cdf(t_draws, t_df)
        else:
            raise ValueError(f"unknown dependence: {dependence!r}")
        sims = np.column_stack(
            [models[p].ppf_from_u(u[:, i]) for i, p in enumerate(products)]
        )

    port_ret = sims @ w
    var_01 = float(-np.percentile(port_ret, 1))
    var_05 = float(-np.percentile(port_ret, 5))
    es_01 = float(-np.mean(port_ret[port_ret <= np.percentile(port_ret, 1)]))
    es_05 = float(-np.mean(port_ret[port_ret <= np.percentile(port_ret, 5)]))

    tail_dep = {}
    pseudo_sims = np.column_stack(
        [to_pseudo_uniform(sims[:, i]) for i in range(n_assets)]
    )
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            key = f"{products[i]}_{products[j]}"
            tail_dep[key] = empirical_lower_tail_dependence(
                pseudo_sims[:, i], pseudo_sims[:, j]
            )

    return {
        "dependence": dependence,
        "n_sims": n_sims,
        "var_01": var_01,
        "var_05": var_05,
        "es_01": es_01,
        "es_05": es_05,
        "lower_tail_dependence": tail_dep,
    }


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    from scipy import stats as st

    return st.norm.cdf(z)


def _t_cdf(x: np.ndarray, df: float) -> np.ndarray:
    from scipy import stats as st

    return st.t.cdf(x, df=df)


# ---------------------------------------------------------------------------
# Phase 5 -- futures cost model applied to a cross-sectional weights panel.
# research.add_portfolio_costs charges a single scalar fee fraction against
# total turnover; futures costs are per-contract and product-specific (sec
# 7), so a multi-product book needs a per-bar, per-symbol cost that varies
# by which products actually traded that bar, not one constant.
# ---------------------------------------------------------------------------


def portfolio_costs_futures(
    weights: pl.DataFrame,
    prices: pl.DataFrame,
    datetime_col: str = "datetime",
    symbol_col: str = "symbol",
    tick_multiplier: float = 1.0,
) -> pl.DataFrame:
    """Per-bar total cost fraction of portfolio equity, from each symbol's
    own |weight change| times its own `cost_per_unit_notional` (round-turn
    commission+slippage / contract notional, at that bar's price) -- weight
    is already a fraction of portfolio equity, and cost_per_unit_notional is
    already a fraction of one contract's notional, so their product is
    directly a fraction of portfolio equity lost to that symbol's turnover
    that bar; summed across symbols, that is the bar's total cost fraction.

    Returns one row per `datetime_col` with a `cost_frac` column, ready to
    combine with `research.portfolio_trade_frame`'s trade_log_return the same
    way `research.add_portfolio_costs` does, but with a per-bar-varying fee
    instead of one constant.
    """
    w = weights.sort([symbol_col, datetime_col])
    w = w.with_columns(
        pl.col("weight")
        .diff()
        .fill_null(pl.col("weight"))
        .abs()
        .over(symbol_col)
        .alias("abs_dweight")
    )
    joined = w.join(
        prices.select([datetime_col, symbol_col, "close"]),
        on=[datetime_col, symbol_col],
        how="left",
    )
    joined = joined.drop_nulls(subset=["close"])

    rows = joined.select([symbol_col, "close"]).to_dicts()
    cost_fracs = [
        cost_per_unit_notional(r[symbol_col], r["close"], tick_multiplier)
        if r["close"] and r["close"] != 0
        else 0.0
        for r in rows
    ]
    joined = joined.with_columns(pl.Series("cost_frac_per_unit", cost_fracs))
    joined = joined.with_columns(
        (pl.col("abs_dweight") * pl.col("cost_frac_per_unit")).alias(
            "cost_contribution"
        )
    )
    return (
        joined.group_by(datetime_col)
        .agg(pl.col("cost_contribution").sum().alias("cost_frac"))
        .sort(datetime_col)
    )


def add_portfolio_costs_futures(
    trade_frame: pl.DataFrame, costs: pl.DataFrame, datetime_col: str = "datetime"
) -> pl.DataFrame:
    """`research.add_portfolio_costs`'s cost math, but with a per-bar cost
    fraction (from `portfolio_costs_futures`) instead of one constant fee."""
    out = trade_frame.join(costs, on=datetime_col, how="left").with_columns(
        pl.col("cost_frac").fill_null(0.0)
    )
    out = out.with_columns((1 - pl.col("cost_frac")).log().alias("cost_log_return"))
    out = out.with_columns(
        (pl.col("trade_log_return") + pl.col("cost_log_return")).alias(
            "trade_log_return_net"
        )
    )
    out = out.with_columns(
        pl.col("trade_log_return_net").cum_sum().alias("equity_curve_net")
    )
    out = out.with_columns(
        (pl.col("equity_curve_net") - pl.col("equity_curve_net").cum_max()).alias(
            "drawdown_log_return_net"
        )
    )
    return out


def futures_portfolio_metrics(
    trade_frame: pl.DataFrame,
    costs: pl.DataFrame,
    annualized_rate: float,
    datetime_col: str = "datetime",
    label: str = "portfolio",
) -> dict:
    """`research.portfolio_metrics`'s summary, but net-of-cost via
    `add_portfolio_costs_futures` (per-bar, per-product costs) instead of
    `research.add_portfolio_costs`'s single scalar fee. Reuses
    `research._series_metrics`/`research.cost_summary` for the actual metric
    math so gross/net/turnover reporting stays byte-identical in convention
    to every other notebook's portfolio summary.
    """
    import research

    if len(trade_frame) == 0:
        return {"label": label, "no_bars": 0}
    metrics = research._series_metrics(
        trade_frame["trade_log_return"], annualized_rate, label
    )
    costed = add_portfolio_costs_futures(trade_frame, costs, datetime_col)
    net_metrics = research._series_metrics(
        costed["trade_log_return_net"], annualized_rate, f"{label}_net"
    )
    metrics["sharpe_net"] = net_metrics["sharpe"]
    metrics["total_log_return_net"] = net_metrics["total_log_return"]
    metrics["compound_return_net"] = net_metrics["compound_return"]
    metrics["max_drawdown_net"] = net_metrics["max_drawdown"]
    metrics.update(research.cost_summary(costed, annualized_rate))
    # tripwire (sec 7): net Sharpe must never exceed gross -- a broken cost
    # accounting bug, not a real result, if it ever does. Skipped when either
    # side is NaN (degenerate/empty book), which is not a cost-accounting
    # question at all.
    both_finite = np.isfinite(metrics["sharpe_net"]) and np.isfinite(metrics["sharpe"])
    assert not both_finite or metrics["sharpe_net"] <= metrics["sharpe"] + 1e-9, (
        f"{label}: net Sharpe ({metrics['sharpe_net']}) exceeds gross ({metrics['sharpe']}) -- cost accounting is broken"
    )
    return metrics
