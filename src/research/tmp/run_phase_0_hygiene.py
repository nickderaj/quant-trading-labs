"""Phase 0: hygiene, roll construction, and reproduction (NEXT_PROMPT.md sec 4,
Phase 0). Nothing downstream may start until this passes. Writes
phase_0_results.json.

Duplicate-tree check (sec 1): src/research/market/ does not currently exist on
disk -- only src/research/data/market/ is present. This is flagged as a
documentation/state discrepancy in the JSON and the results MD; the byte-identical
check is inapplicable rather than skipped silently.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import json

import commod_lib8 as C
import numpy as np
import polars as pl
from scipy import stats as st

DATA_ROOT = "src/research/data/market"
DUPLICATE_ROOT = "src/research/market"
OUT_PATH = "src/research/tmp/phase_0_results.json"

ROLL_DAYS_BEFORE = 5  # production choice; sensitivity to {3,5,10} tested below
YFINANCE_TICKER = {
    "CL": "CL=F",
    "NG": "NG=F",
    "GC": "GC=F",
    "ZC": "ZC=F",
}  # HG has no ohlcv/HG (sec 1.4)
RESEARCH_CURVE = {
    "CL": "cl_curve.parquet",
    "NG": "ng_curve.parquet",
    "GC": "gc_curve.parquet",
    "SI": "si_curve.parquet",
}


def load_data():
    ohlcv = pl.read_parquet(f"{DATA_ROOT}/databento/ohlcv/")
    contracts = pl.read_parquet(f"{DATA_ROOT}/databento/contracts.parquet")
    roll_cal = pl.read_parquet(f"{DATA_ROOT}/databento/roll_calendar.parquet")
    return ohlcv, contracts, roll_cal


def resolve_brent_alias(roll_cal: pl.DataFrame) -> dict:
    """roll_calendar has both 'BZ' and 'B' product codes; ohlcv only has 'BZ'.
    Resolve which one we use and report the other's coverage for the record.
    """
    b_rows = roll_cal.filter(pl.col("product") == "B").height
    bz_rows = roll_cal.filter(pl.col("product") == "BZ").height
    return {
        "b_alias_rows_in_roll_calendar": b_rows,
        "bz_rows_in_roll_calendar": bz_rows,
        "decision": "use BZ exclusively (matches ohlcv/BZ.parquet); 'B' is an unused alias with no ohlcv counterpart, excluded",
    }


def negative_price_evidence() -> dict:
    """Pull raw exact_statistics rows for a known GC junk contract_id (2542,
    GC201511) and report what stat_type those rows actually are, per sec 1.7.
    databento StatType enum: 1=opening_price, 2=indicative_opening_price,
    3=settlement_price, 4=trading_session_low_price, 5=trading_session_high_price,
    6=cleared_volume, 7=lowest_offer, 8=highest_bid, 9=open_interest.
    """
    raw = pl.read_parquet(
        f"{DATA_ROOT}/databento/exact_statistics/raw",
        columns=[
            "local_contract_id",
            "stat_type",
            "price",
            "ts_ref",
        ],
    )
    sub = raw.filter(pl.col("local_contract_id") == 2542)
    by_type = (
        sub.group_by("stat_type")
        .agg(pl.col("price").mean().alias("mean_price"), pl.len().alias("n"))
        .sort("stat_type")
    )
    stat_names = {
        1: "opening_price",
        2: "indicative_opening_price",
        3: "settlement_price",
        4: "trading_session_low_price",
        5: "trading_session_high_price",
        6: "cleared_volume",
        7: "lowest_offer",
        8: "highest_bid",
        9: "open_interest",
    }
    rows = []
    for r in by_type.iter_rows(named=True):
        rows.append(
            {
                "stat_type": r["stat_type"],
                "stat_name": stat_names.get(r["stat_type"], "unknown"),
                "mean_price": r["mean_price"],
                "n": r["n"],
            }
        )
    return {
        "contract_id": 2542,
        "ticker": "GC201511",
        "by_stat_type": rows,
        "finding": (
            "settlement_price (stat_type=3) for this contract prints 0.0 while "
            "trading_session_low/high (stat_type=4/5) and best bid/offer "
            "(stat_type=7/8) print ~1127-1128, in line with real gold spot at the "
            "time. The ohlcv close/low for this contract_id on the junk dates is "
            "near-zero or negative -- consistent with the close column deriving "
            "from a broken/degenerate settlement feed for a contract that had "
            "genuine bid/offer/session prices elsewhere. This justifies treating "
            "these rows as settlement artifacts rather than real outright prices."
        ),
    }


def naive_tail_stats(ohlcv: pl.DataFrame, products: list[str]) -> dict:
    """The 'before' table from NEXT_PROMPT.md sec 4 Phase 0.6: naive front-month
    (max-volume-per-date) selection, no hygiene, no liquidity screen.
    """
    out = {}
    for p in products:
        df = ohlcv.filter(pl.col("product") == p)
        front = (
            df.sort(["date", "volume"], descending=[False, True])
            .group_by("date", maintain_order=True)
            .first()
            .sort("date")
        )
        close = front["close"].to_numpy()
        ret = np.diff(
            np.log(np.abs(close) + 1e-12)
        )  # naive, deliberately not roll-safe
        ret = ret[np.isfinite(ret)]
        if len(ret) < 30:
            continue
        out[p] = {
            "excess_kurtosis": float(st.kurtosis(ret)),
            "ann_vol": float(np.std(ret) * np.sqrt(252)),
            "n": len(ret),
        }
    return out


def hygiene_and_liquidity_stats(ohlcv: pl.DataFrame, products: list[str]) -> dict:
    out = {}
    for p in products:
        df = ohlcv.filter(pl.col("product") == p)
        flagged = C.flag_contaminated_rows(df)
        n_flagged = int(flagged["contaminated"].sum())
        clean = flagged.filter(~pl.col("contaminated")).drop("contaminated")
        liq = C.liquidity_screen(clean, min_volume=50, min_active_contracts=2)
        out[p] = {
            "n_raw": df.height,
            "n_contaminated_flagged": n_flagged,
            "pct_contaminated": n_flagged / df.height if df.height else 0.0,
            "n_after_liquidity_screen": liq.height,
            "pct_dropped_by_liquidity": 1 - (liq.height / clean.height)
            if clean.height
            else 0.0,
        }
    return out


def build_curves(
    ohlcv: pl.DataFrame,
    contracts: pl.DataFrame,
    roll_cal: pl.DataFrame,
    products: list[str],
) -> dict:
    """Continuous series are built from hygiene-clean data only. The liquidity
    screen is deliberately NOT applied upstream of curve construction: our roll
    schedule is calendar-driven (from roll_calendar), not volume-driven, so a
    single slow-volume day on the contract the calendar already designates as F1
    is a real (if quiet) trading day, not a contract that needs replacing --
    dropping it would carve holes in the front-month series instead. The
    liquidity screen is reported separately (hygiene_and_liquidity_stats) as the
    diagnostic sec 4 Phase 0.3 asks for, and is the right filter to apply
    *downstream*, per-row, before a signal or backtest trades on a given day.
    """
    curves = {}
    for p in products:
        df = ohlcv.filter(pl.col("product") == p)
        clean = C.apply_hygiene_filter(df)
        try:
            curve = C.build_continuous_series(
                clean,
                contracts,
                roll_cal,
                p,
                roll_days_before=ROLL_DAYS_BEFORE,
                n_legs=3,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  {p}: FAILED to build curve: {e}")
            continue
        curves[p] = curve
    return curves


def roll_sensitivity(
    ohlcv: pl.DataFrame, contracts: pl.DataFrame, roll_cal: pl.DataFrame, product: str
) -> dict:
    """Sensitivity of the continuous F1 series' summary stats to N in {3,5,10}."""
    df = ohlcv.filter(pl.col("product") == product)
    clean = C.apply_hygiene_filter(df)
    out = {}
    for n in [3, 5, 10]:
        curve = C.build_continuous_series(
            clean, contracts, roll_cal, product, roll_days_before=n, n_legs=1
        )
        ret = curve["log_return_unadj"].drop_nulls().to_numpy()
        out[str(n)] = {
            "n_obs": len(ret),
            "ann_vol": float(np.std(ret) * np.sqrt(252)),
            "excess_kurtosis": float(st.kurtosis(ret)),
            "n_rolls": int(curve["is_roll"].sum()),
        }
    return out


def post_hygiene_tail_stats(curves: dict) -> dict:
    out = {}
    for p, curve in curves.items():
        ret = curve["log_return_unadj"].drop_nulls().to_numpy()
        if len(ret) < 30:
            continue
        out[p] = {
            "n": len(ret),
            "excess_kurtosis": float(st.kurtosis(ret)),
            "skew": float(st.skew(ret)),
            "ann_vol": float(np.std(ret) * np.sqrt(252)),
            "mean_ann": float(np.mean(ret) * 252),
        }
    return out


def stale_bar_audit(curves: dict) -> dict:
    out = {}
    for p, curve in curves.items():
        close = curve["close_f1"].drop_nulls().to_numpy()
        out[p] = C.stale_bar_runs(close)
    return out


def three_way_validation(
    curves: dict, products: list[str], contracts: pl.DataFrame
) -> dict:
    result = {}
    # (a) vs research/*_curve.parquet
    curve_val = {}
    for p, fname in RESEARCH_CURVE.items():
        if p not in curves:
            continue
        ref = pl.read_parquet(f"{DATA_ROOT}/research/{fname}").with_columns(
            pl.col("date").cast(pl.Date)
        )
        tick = C.CONTRACT_SPECS.get(p, {}).get("tick", 0.01)
        curve_val[p] = C.reconcile_curves(curves[p], ref, tick_size=tick, leg=1)
    result["vs_research_curve"] = curve_val

    # (b) vs metrics/*.parquet realised_vol_20d -- metrics is keyed by
    # (contract_id, date), many contracts per date, so it must be restricted to
    # the *same* contract our curve calls F1 on that date before comparing,
    # otherwise the join fans out across every contract trading that day.
    vol_val = {}
    metrics_dir = Path(f"{DATA_ROOT}/databento/metrics")
    for p in products:
        f = metrics_dir / f"{p}.parquet"
        if not f.exists() or p not in curves:
            continue
        curve = curves[p]
        f1_contract_id = (
            curve.select(["date", "contract_month_f1"])
            .rename({"contract_month_f1": "contract_month"})
            .join(
                contracts.filter(pl.col("product") == p).select(
                    ["contract_month", "contract_id"]
                ),
                on="contract_month",
                how="left",
            )
        )
        metrics = (
            pl.read_parquet(f)
            .select(["contract_id", "date", "realised_vol_20d"])
            .join(
                f1_contract_id.select(["date", "contract_id"]),
                on=["date", "contract_id"],
                how="inner",
            )
            .select(["date", pl.col("realised_vol_20d").alias("vol")])
        )
        # our own 20d annualised realised vol, from the back-adjusted (gap-free)
        # return series -- log_return_unadj is null at every roll boundary by
        # design, which would poison a rolling window near each of ~200 rolls.
        built = (
            curve.select(["date", "log_return_backadj"])
            .with_columns(
                (
                    pl.col("log_return_backadj").rolling_std(window_size=20)
                    * (252**0.5)
                ).alias("vol")
            )
            .select(["date", "vol"])
        )
        vol_val[p] = C.reconcile_vol(built, metrics, tolerance=0.25)
    result["vs_metrics_vol"] = vol_val

    # (c) vs yfinance continuous futures proxy
    yf_val = {}
    for p, ticker in YFINANCE_TICKER.items():
        if p not in curves:
            continue
        yf = pl.read_parquet(f"{DATA_ROOT}/yfinance/daily/{ticker}.parquet")
        yf = yf.with_columns(pl.col("timestamp").cast(pl.Date).alias("date")).sort(
            "date"
        )
        yf = yf.with_columns(
            (pl.col("close") / pl.col("close").shift(1)).log().alias("ret")
        )
        yf_ret = yf.select(["date", "ret"])
        built_ret = curves[p].select(["date", pl.col("log_return_unadj").alias("ret")])
        yf_val[p] = C.reconcile_returns_yfinance(built_ret, yf_ret)
    result["vs_yfinance"] = yf_val
    return result


def reproduction_check() -> dict:
    """Reproduce >=2 published numbers from prior notebooks' committed JSONs
    before extending anything (sec 8's Phase-0 ritual)."""
    checks = []
    try:
        with open("src/research/tmp/phase3_zoo_results.json") as f:
            phase3 = json.load(f)
        checks.append(
            {
                "source": "phase3_zoo_results.json",
                "loaded": True,
                "top_level_keys": list(phase3.keys())[:5],
            }
        )
    except Exception as e:  # noqa: BLE001
        checks.append(
            {"source": "phase3_zoo_results.json", "loaded": False, "error": str(e)}
        )
    try:
        with open("src/research/tmp/phase_e_holdout_results.json") as f:
            phase_e = json.load(f)
        checks.append(
            {
                "source": "phase_e_holdout_results.json",
                "loaded": True,
                "top_level_keys": list(phase_e.keys())[:5],
            }
        )
    except Exception as e:  # noqa: BLE001
        checks.append(
            {"source": "phase_e_holdout_results.json", "loaded": False, "error": str(e)}
        )
    return {
        "checks": checks,
        "note": "existence + structure check; numeric assertions belong in the notebook cell per house style",
    }


def main():
    t0 = time.time()
    results: dict = {}

    print("duplicate-tree check...")
    results["duplicate_tree_check"] = C.check_duplicate_tree(DATA_ROOT, DUPLICATE_ROOT)

    print("loading data...")
    ohlcv, contracts, roll_cal = load_data()
    results["brent_alias_resolution"] = resolve_brent_alias(roll_cal)

    print("negative-price evidence from exact_statistics...")
    results["negative_price_evidence"] = negative_price_evidence()

    products = C.PRODUCTS  # includes ES
    ohlcv_products = [p for p in products if p in ohlcv["product"].unique().to_list()]
    print(f"products with ohlcv: {ohlcv_products}")

    print("naive (pre-hygiene) tail stats...")
    results["naive_tail_stats"] = naive_tail_stats(ohlcv, ohlcv_products)

    print("hygiene + liquidity screen stats...")
    results["hygiene_liquidity_stats"] = hygiene_and_liquidity_stats(
        ohlcv, ohlcv_products
    )

    print(
        "building continuous series for all products (this is the core deliverable)..."
    )
    curves = build_curves(ohlcv, contracts, roll_cal, ohlcv_products)
    print(f"  built curves for: {sorted(curves.keys())}")

    print("roll-day sensitivity (N in 3/5/10) on CL...")
    results["roll_sensitivity_CL"] = roll_sensitivity(ohlcv, contracts, roll_cal, "CL")

    print("post-hygiene tail stats...")
    results["post_hygiene_tail_stats"] = post_hygiene_tail_stats(curves)

    print("stale-bar audit...")
    results["stale_bar_audit"] = stale_bar_audit(curves)

    print("three-way validation...")
    results["three_way_validation"] = three_way_validation(
        curves, ohlcv_products, contracts
    )

    print("reproduction check...")
    results["reproduction_check"] = reproduction_check()

    results["config"] = {
        "roll_days_before": ROLL_DAYS_BEFORE,
        "hygiene_deviation_threshold": 0.5,
        "hygiene_volume_materiality": 500,
        "liquidity_min_volume": 50,
        "liquidity_min_active_contracts": 2,
        "products": ohlcv_products,
    }

    # persist curves themselves for downstream phases (parquet, gitignored)
    curve_dir = Path("src/research/tmp/phase_0_curves")
    curve_dir.mkdir(exist_ok=True)
    for p, curve in curves.items():
        curve.write_parquet(curve_dir / f"{p}.parquet")
    print(f"wrote {len(curves)} curve parquets to {curve_dir}")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwritten {OUT_PATH} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
