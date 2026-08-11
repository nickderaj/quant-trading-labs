"""The data contract: hygiene, liquidity, roll/continuous-series construction,
and return-convention machinery, ported verbatim from
``src/research/tmp/commod_lib8.py`` (see ``docs/10-risk-engine.md``).

Every function below is behaviour-identical to its ``commod_lib8.py`` original
(NEXT_PROMPT.md sec 2, ground rule 2) -- only module-level imports and type
annotations were added during the move. ``commod_lib8.py`` re-imports these
names and re-exports them so notebook 008 and its tests keep passing
unchanged against the promoted code.

``build_risk_inputs``/``assert_risk_inputs`` (below the ported section) are
new: the data contract notebook 008 discovered but never enforced at a call
boundary. See NEXT_PROMPT.md sec 4.
"""

from __future__ import annotations

import numpy as np
import polars as pl

PROVENANCE_ATTR = "__risk_hygiene_provenance__"
PROVENANCE_VALUE = "build_risk_inputs.v1"

MIN_RISK_OBSERVATIONS = 100
MAX_OBSERVED_STALE_RUN = 3
REALIZED_VOL_WINDOW = 20


class RiskInputError(ValueError):
    """Raised by ``assert_risk_inputs`` when a frame fails the data contract."""


# ---------------------------------------------------------------------------
# Ported verbatim from commod_lib8.py lines 156-602 (as of 7641ee4).
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


def apply_hygiene_filter(df: pl.DataFrame, **kwargs: object) -> pl.DataFrame:
    """Flag and drop contaminated rows, returning the clean frame."""
    flagged = flag_contaminated_rows(df, **kwargs)  # type: ignore[arg-type]
    return flagged.filter(~pl.col("contaminated")).drop("contaminated")


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


def build_continuous_series_ohlcv(
    ohlcv: pl.DataFrame,
    contracts: pl.DataFrame,
    roll_calendar: pl.DataFrame,
    product: str,
    roll_days_before: int = 5,
    min_contract_volume: int = 5000,
) -> pl.DataFrame:
    """F1-only continuous OHLCV series -- `build_continuous_series` above
    drops open/high/low/volume entirely (it exists for spread/return work,
    which only ever needed `close_f{leg}`). A volume-gated breakout needs
    genuine bars, so this reuses the identical roll schedule and
    `searchsorted` F1-selection logic and joins the full OHLCV row instead
    of `close` alone.

    `volume` is raw, per-traded-contract and therefore discontinuous at
    rolls by construction (front-month volume ramps down into expiry and
    the new F1 starts mid-life) -- callers must treat it as relative to its
    own trailing, within-contract history only, never as an absolute
    level. `is_roll` (True on the first bar of a new front-month contract)
    is provided so callers can suppress the roll-date volume spike rather
    than mistake it for a breakout confirmation.
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
        schedule.select(["contract_month", "roll_date"]),
        on="contract_month",
        how="left",
    )
    px = px.filter(pl.col("roll_date").is_not_null()).sort(["expiry", "date"])

    sched_sorted = schedule.filter(pl.col("roll_date").is_not_null()).sort("expiry")
    months = sched_sorted["contract_month"].to_list()
    roll_dates = sched_sorted["roll_date"].to_numpy().astype("datetime64[D]")

    dates_list = px.select("date").unique().sort("date")["date"].to_list()
    dates_arr = np.array(dates_list, dtype="datetime64[D]")
    f1_idx = np.searchsorted(roll_dates, dates_arr, side="right")

    px_lookup = px.select(
        ["date", "contract_month", "open", "high", "low", "close", "volume"]
    )
    n_months = len(months)
    valid = f1_idx < n_months
    sel_dates = dates_arr[valid].astype("datetime64[ms]")
    sel_months = [months[i] for i in f1_idx[valid]]
    leg_map = pl.DataFrame(
        {"date": sel_dates, "contract_month": sel_months}
    ).with_columns(pl.col("date").cast(pl.Date))
    curve = leg_map.join(px_lookup, on=["date", "contract_month"], how="left")
    curve = curve.sort("date")
    curve = curve.with_columns(
        pl.col("contract_month")
        .ne(pl.col("contract_month").shift(1))
        .fill_null(False)
        .alias("is_roll")
    )

    # Back-adjust open/high/low/close by the same additive Panama offset
    # (identical construction to `_add_return_conventions`'s `close_backadj`)
    # so the breakout rule sees a gap-free price level -- raw per-contract
    # `close` jumps at every roll and would manufacture spurious breakouts.
    # `volume` is left raw: it is a traded quantity, not a price, and
    # additive adjustment would be meaningless for it.
    close = curve["close"].to_numpy()
    is_roll = curve["is_roll"].to_numpy()
    n = len(close)
    offset = np.zeros(n)
    cum_offset = 0.0
    for i in range(1, n):
        if is_roll[i] and not np.isnan(close[i]) and not np.isnan(close[i - 1]):
            cum_offset += close[i] - close[i - 1]
        offset[i] = cum_offset
    for col in ["open", "high", "low", "close"]:
        curve = curve.with_columns(
            (pl.col(col) - pl.Series(offset)).alias(f"{col}_backadj")
        )
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
# The data contract (NEXT_PROMPT.md sec 4) -- new code, not a port.
# ---------------------------------------------------------------------------


def build_risk_inputs(
    ohlcv: pl.DataFrame,
    contracts: pl.DataFrame,
    roll_calendar: pl.DataFrame,
    product: str,
    roll_days_before: int = 5,
    n_legs: int = 3,
    min_contract_volume: int = 5000,
    hygiene_kwargs: dict[str, float | int] | None = None,
    liquidity_kwargs: dict[str, int] | None = None,
) -> pl.DataFrame:
    """The full input chain the risk engine's certified numbers depend on:
    hygiene filter -> liquidity screen -> liquid contract months -> continuous
    series -> return conventions (NEXT_PROMPT.md sec 4).

    Returns a frame carrying **only** `log_return_ratioadj` under the name
    `log_return`; `log_return_unadj` and `log_return_backadj` remain available
    under their own explicit names (never as the default) so a caller who
    genuinely needs the unadjusted or back-adjusted convention can still get
    it, but cannot get it by accident. The returned frame is stamped with a
    provenance attribute (`PROVENANCE_ATTR`/`PROVENANCE_VALUE`) that
    `assert_risk_inputs` checks for -- a column named `log_return` alone does
    not prove it came from this function, since a caller can rename a column.
    """
    hygiene_kwargs = hygiene_kwargs or {}
    liquidity_kwargs = liquidity_kwargs or {}

    clean = apply_hygiene_filter(ohlcv, **hygiene_kwargs)  # type: ignore[arg-type]
    screened = liquidity_screen(clean, **liquidity_kwargs)  # type: ignore[arg-type]
    curve = build_continuous_series(
        screened,
        contracts,
        roll_calendar,
        product,
        roll_days_before=roll_days_before,
        n_legs=n_legs,
        min_contract_volume=min_contract_volume,
    )
    curve = curve.with_columns(pl.col("log_return_ratioadj").alias("log_return"))
    setattr(curve, PROVENANCE_ATTR, PROVENANCE_VALUE)
    return curve


def assert_risk_inputs(frame: pl.DataFrame) -> None:
    """Precondition checker `fit_risk_model` calls before fitting. Rejects,
    with a named error, a frame that does not satisfy the data contract
    (NEXT_PROMPT.md sec 4):

    - a return column that is not provably `log_return_ratioadj` (checked via
      the provenance attribute `build_risk_inputs` stamps, not by column name
      alone -- a caller can rename a column);
    - any `|log_return| > 0.5` on a day not already excluded by the hygiene
      filter (a 65% single-day move is either a real, nameable event or a
      splice artifact; a human should look before it enters a VaR fit);
    - any run of more than 3 consecutive identical closes (008's stale-bar
      audit found a worst-case run of 3 on real, hygiene-passed data -- CL's
      own clean curve has one -- so 3 is the observed ceiling and is
      accepted; anything longer is new and unexamined);
    - fewer than 100 finite observations;
    - any `realized_vol <= 1e-12` over a rolling window (the frozen-bar rule
      that corrupted notebook 003 twice: a stale multi-day block with zero
      variance, not an individual quiet day -- a single exactly-zero return
      is a normal, unremarkable event on real data and is not itself
      rejected).
    """
    if not getattr(frame, PROVENANCE_ATTR, None) == PROVENANCE_VALUE:
        raise RiskInputError(
            "frame is not provably the output of build_risk_inputs() -- "
            f"missing/incorrect provenance attribute {PROVENANCE_ATTR!r}. "
            "A production caller must construct inputs via build_risk_inputs(); "
            "renaming a column to `log_return` is not sufficient."
        )
    if "log_return" not in frame.columns:
        raise RiskInputError("frame is missing the `log_return` column")

    ret = frame["log_return"].to_numpy()
    finite = np.isfinite(ret)
    if finite.sum() < MIN_RISK_OBSERVATIONS:
        raise RiskInputError(
            f"only {int(finite.sum())} finite log_return observations, "
            f"below the minimum of {MIN_RISK_OBSERVATIONS}"
        )

    large = finite & (np.abs(ret) > 0.5)
    if large.any():
        raise RiskInputError(
            f"{int(large.sum())} row(s) have |log_return| > 0.5 -- either a "
            "real, nameable event or a splice artifact; a human must review "
            "before this enters a VaR fit"
        )

    if "close_f1" in frame.columns:
        close = frame["close_f1"].to_numpy()
        max_run = _max_consecutive_equal_run(close)
        if max_run > MAX_OBSERVED_STALE_RUN:
            raise RiskInputError(
                f"found a run of {max_run} consecutive identical close_f1 "
                f"prices -- the observed ceiling from 008's stale-bar audit "
                f"is {MAX_OBSERVED_STALE_RUN}; anything longer is new and "
                "unexamined"
            )

    frozen = _rolling_realized_vol(ret) <= 1e-12
    if frozen.any():
        raise RiskInputError(
            f"{int(frozen.sum())} row(s) sit inside a {REALIZED_VOL_WINDOW}-"
            "observation window with realized_vol <= 1e-12 -- the frozen-bar "
            "rule that corrupted notebook 003 twice (a stale, non-trading "
            "data block), not a single quiet day"
        )


def _max_consecutive_equal_run(x: np.ndarray) -> int:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 1 if len(x) == 1 else 0
    same = x[1:] == x[:-1]
    max_run = 1
    cur = 1
    for s in same:
        if s:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 1
    return max_run


def _rolling_realized_vol(
    ret: np.ndarray, window: int = REALIZED_VOL_WINDOW
) -> np.ndarray:
    """Trailing-window realized vol (population std) ending at each index,
    NaN wherever fewer than `window` observations precede it or the window
    contains a non-finite value. NaN never compares `<= 1e-12`, so an
    under-filled or gappy window is never flagged as frozen -- only a
    genuinely complete, genuinely flat window is."""
    ret = np.asarray(ret, dtype=float)
    n = len(ret)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        w = ret[i - window + 1 : i + 1]
        if np.all(np.isfinite(w)):
            out[i] = float(np.std(w))
    return out
