import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, "src")
import polars as pl

import data
import features
import research

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "DOTUSDT",
    "AVAXUSDT",
    "MATICUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "ATOMUSDT",
    "UNIUSDT",
    "ETCUSDT",
    "XLMUSDT",
    "ALGOUSDT",
    "VETUSDT",
    "FILUSDT",
    "TRXUSDT",
    "EOSUSDT",
    "AAVEUSDT",
    "SANDUSDT",
    "MANAUSDT",
    "AXSUSDT",
    "THETAUSDT",
    "NEARUSDT",
    "FTMUSDT",
    "LUNAUSDT",
    "FTTUSDT",
]
START = datetime(2021, 7, 1, tzinfo=UTC)
END = research.HOLDOUT_START  # 2025-07-01, holdout excluded
CACHE_DIR = "src/research/cache"
DOWNLOAD_DIR = "src/research/tmp"
INTERVALS = ["4h", "12h", "1d"]
BARS_PER_DAY = {"4h": 6, "12h": 2, "1d": 1}

CONFIG_LOG_PATH = Path("src/research/tmp/config_log.jsonl")


# Per-feature Newey-West lag: roughly the feature's own lookback window.
def nw_lag_for(col: str, bars_per_day: int) -> int:
    for suffix in ("_z20",):
        if col.endswith(suffix):
            return 20
    if col.endswith("_z60"):
        return 60
    if col.startswith(("momentum_", "mean_reversion_")):
        return int(col.rsplit("_", 1)[1])
    if col.startswith("realized_vol_"):
        return int(col.rsplit("_", 1)[1])
    if col.startswith("vol_of_vol_"):
        return int(col.rsplit("_", 1)[1])
    if col == "vol_regime":
        return 96
    if col in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
        return max(bars_per_day, 1)
    if col in ("taker_buy_ratio", "order_flow_imbalance", "avg_trade_size", "count"):
        return 1
    if col in ("funding_rate",):
        return 1
    return 1


def load_funding_by_symbol() -> dict[str, pl.DataFrame]:
    out = {}
    for sym in SYMBOLS:
        try:
            out[sym] = data.download_funding_rate_range(
                sym, START, END, cache_dir=CACHE_DIR
            )
        except ValueError:
            continue
    return out


def main():
    funding_by_symbol = load_funding_by_symbol()
    all_rows = []

    for interval in INTERVALS:
        print(f"=== {interval} ===", flush=True)
        panel = research.load_universe_panel(
            SYMBOLS,
            interval,
            START,
            END,
            min_cross_section=10,
            download_dir=DOWNLOAD_DIR,
            cache_dir=CACHE_DIR,
        )
        featured = features.apply_per_symbol(
            panel,
            lambda df: (
                features.add_all_raw_features(df)
                .pipe(
                    lambda d: (
                        features.add_funding_rate_feature(
                            d, funding_by_symbol[d["symbol"][0]]
                        )
                        if d["symbol"][0] in funding_by_symbol
                        else d
                    )
                )
                .with_columns(features.forward_return())
            ),
        )

        base_cols = set(panel.columns)
        raw_feature_cols = [
            c
            for c in featured.columns
            if c not in base_cols and not c.startswith("fwd_return_")
        ]
        print(
            f"{len(raw_feature_cols)} candidate features: {raw_feature_cols}",
            flush=True,
        )

        bars_per_day = BARS_PER_DAY[interval]
        for col in raw_feature_cols:
            lag = nw_lag_for(col, bars_per_day)
            ic_df = research.cross_sectional_ic(
                featured, col, "fwd_return_1", min_symbols=10
            )
            stats = research.cross_sectional_ic_stats(ic_df, nw_lag=lag)
            stability = research.ic_stability(ic_df)
            per_year = stability["per_year_ic"]
            per_year_signs = (
                per_year["mean_ic"].sign().to_list() if len(per_year) else []
            )
            sign_consistent = (
                len({s for s in per_year_signs if s != 0}) <= 1
                and len(per_year_signs) > 0
            )
            tripwire = (
                abs(stats["mean_ic"]) > 0.10
                if stats["mean_ic"] == stats["mean_ic"]
                else False
            )

            row = {
                "feature": col,
                "interval": interval,
                "nw_lag": lag,
                "mean_ic": stats["mean_ic"],
                "nw_tstat": stats["nw_tstat"],
                "n_periods": stats["n_periods"],
                "frac_positive_months": stability["frac_positive_months"],
                "per_year_ic": {
                    str(y): v
                    for y, v in zip(
                        per_year["year"].to_list(), per_year["mean_ic"].to_list()
                    )
                }
                if len(per_year)
                else {},
                "sign_consistent_across_years": sign_consistent,
                "lookahead_tripwire": tripwire,
            }
            all_rows.append(row)
            with open(CONFIG_LOG_PATH, "a") as f:
                f.write(json.dumps(row, default=str) + "\n")

            if tripwire:
                print(
                    f"!!! LOOKAHEAD TRIPWIRE: {col} {interval} mean_ic={stats['mean_ic']:.4f} !!!",
                    flush=True,
                )

    df = (
        pl.DataFrame(all_rows)
        .with_columns(pl.col("nw_tstat").abs().alias("abs_tstat"))
        .sort("abs_tstat", descending=True)
    )
    df.write_parquet("src/research/tmp/screening_results.parquet")
    print(
        df.select(
            "feature",
            "interval",
            "mean_ic",
            "nw_tstat",
            "frac_positive_months",
            "sign_consistent_across_years",
        ).head(40)
    )

    survivors = df.filter(
        (pl.col("abs_tstat") > 3) & pl.col("sign_consistent_across_years")
    )
    print(
        f"\n{len(survivors)} / {len(df)} configs survive |t|>3 AND sign-consistent-across-years:"
    )
    print(
        survivors.select(
            "feature", "interval", "mean_ic", "nw_tstat", "frac_positive_months"
        )
    )


if __name__ == "__main__":
    main()
