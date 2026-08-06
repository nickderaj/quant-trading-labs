"""Notebook 13, Design D -- cross-sectional attention over the crypto
correlation graph (NEXT_PROMPT.md sec4.D). The claim under test: crypto
returns are driven by correlated neighbours rather than by their own
history, and adding temporal mixing makes it worse.

Universe: symbols with COMPLETE daily coverage across the full dev window
(2021-07-01 to 2025-06-30) -- a fixed node count is required to batch-train
one graph attention model across all rebalance dates; LUNA/FTT (post-
collapse/delisting) and MATIC (POL rebrand tail gap, sec2) are excluded by
this criterion, not hand-picked. Disclosed structural weakening on top of
the source design's own 66-vs-30 gap.

Graph: rolling 30-day causal Pearson correlation (exec_lib13.
rolling_causal_corr_graph), rebuilt at every rebalance from trailing data
only (sec7 trap3). Two model variants share every weight shape except the
GRU time-mixing branch (exec_lib13.GraphAttentionPredictor).

Order of operations matches the design's own instruction: cross-sectional
IC (vs 003's survival filter) is computed and reported BEFORE either book's
Sharpe is treated as a result.

Writes phase_D_13_results.json.
"""

import json
import sys
from datetime import UTC, datetime
from typing import Any

import numpy as np
import polars as pl
import torch
from torch import nn

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import alpha_lib7 as A
import exec_lib13 as E

import research

ANNUALIZED_RATE = float(np.sqrt(252))
N_TRIALS_POOLED = 18
DEV_START = datetime(2021, 7, 1, tzinfo=UTC)
DEV_END = datetime(2025, 6, 30, tzinfo=UTC)
CORR_LOOKBACK = 30
CORR_THRESHOLD = 0.3
TRAIN_DAYS = 365 * 2
TEST_DAYS = 180
TOP_K = 5
N_EPOCHS_PER_FOLD = 80
HIDDEN_DIM = 16
TAKER_FEE = 0.0004
SLIPPAGE = 0.0001

SYMBOLS = [
    s + "USDT"
    for s in [
        "AAVE",
        "ADA",
        "ALGO",
        "ATOM",
        "AVAX",
        "AXS",
        "BNB",
        "BTC",
        "DOGE",
        "DOT",
        "EOS",
        "ETC",
        "ETH",
        "FIL",
        "FTM",
        "FTT",
        "LINK",
        "LTC",
        "LUNA",
        "MANA",
        "MATIC",
        "NEAR",
        "SAND",
        "SOL",
        "THETA",
        "TRX",
        "UNI",
        "VET",
        "XLM",
        "XRP",
    ]
]


def load_panel() -> pl.DataFrame:
    panel = research.load_universe_panel(
        SYMBOLS,
        "1d",
        DEV_START,
        DEV_END,
        download_dir="/tmp/claude-1000/-home-nick-Documents-quant-trading-labs/47229a3d-77fb-43df-b666-7397be0c6d9a/scratchpad",
        cache_dir="src/research/cache",
    )
    return panel.sort(["symbol", "datetime"])


def restrict_to_complete_symbols(panel: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    n_dates = panel["datetime"].n_unique()
    counts = panel.group_by("symbol").agg(pl.len().alias("n"))
    complete = counts.filter(pl.col("n") >= int(n_dates * 0.98))["symbol"].to_list()
    return panel.filter(pl.col("symbol").is_in(complete)), sorted(complete)


def build_features(panel: pl.DataFrame) -> pl.DataFrame:
    panel = panel.with_columns(
        (pl.col("close").log() - pl.col("close").log().shift(1))
        .over("symbol")
        .alias("ret")
    )
    for lag in range(1, 6):
        panel = panel.with_columns(
            pl.col("ret").shift(lag).over("symbol").alias(f"lag_{lag}")
        )
    panel = panel.with_columns(pl.col("ret").shift(-1).over("symbol").alias("fwd_ret"))
    return panel.drop_nulls(subset=[f"lag_{i}" for i in range(1, 6)] + ["fwd_ret"])


def wide_matrices(panel: pl.DataFrame, symbols: list[str], feature_cols: list[str]):
    """(dates, {feature: date -> row vector across symbols}) plus fwd_ret wide."""
    wide = {}
    for col in feature_cols + ["fwd_ret"]:
        wide[col] = (
            panel.pivot(index="datetime", on="symbol", values=col)
            .sort("datetime")
            .select(["datetime"] + symbols)
        )
    dates = wide[feature_cols[0]]["datetime"].to_list()
    return dates, wide


def train_and_predict(
    panel: pl.DataFrame,
    symbols: list[str],
    feature_cols: list[str],
    use_time_mixing: bool,
    seed: int = 0,
) -> pl.DataFrame:
    research.set_seed(seed)
    dates, wide = wide_matrices(panel, symbols, feature_cols)
    graphs = rolling_graph(panel, symbols)
    n = len(dates)
    n_features = len(feature_cols)

    splits = research.walk_forward_splits(n, TRAIN_DAYS, TEST_DAYS, mode="rolling")
    model = E.GraphAttentionPredictor(
        n_features,
        hidden_dim=HIDDEN_DIM,
        use_time_mixing=use_time_mixing,
        time_window=5,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    rows = []
    for train_idx, test_idx in splits:
        for epoch in range(N_EPOCHS_PER_FOLD):
            t = int(np.random.default_rng(epoch).choice(train_idx))
            date = dates[t]
            if date not in graphs:
                continue
            x, y, adj = build_step_tensors(
                wide, feature_cols, symbols, t, use_time_mixing
            )
            if x is None:
                continue
            optimizer.zero_grad()
            pred = model(x, adj)
            mask = torch.isfinite(y)
            if mask.sum() < 3:
                continue
            loss = loss_fn(pred[mask], y[mask])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for t in test_idx:
                date = dates[int(t)]
                if date not in graphs:
                    continue
                x, y, adj = build_step_tensors(
                    wide, feature_cols, symbols, int(t), use_time_mixing
                )
                if x is None:
                    continue
                pred = model(x, adj).numpy()
                for i, sym in enumerate(symbols):
                    rows.append(
                        {
                            "datetime": date,
                            "symbol": sym,
                            "pred": float(pred[i]),
                            "fwd_ret": float(y[i]) if np.isfinite(y[i]) else None,
                        }
                    )
        model.train()

    return pl.DataFrame(rows).drop_nulls(subset=["fwd_ret"])


def rolling_graph(panel: pl.DataFrame, symbols: list[str]) -> dict:
    ret_panel = panel.select("datetime", "symbol", "ret").drop_nulls()
    return E.rolling_causal_corr_graph(
        ret_panel, lookback=CORR_LOOKBACK, threshold=CORR_THRESHOLD, return_col="ret"
    )


_GRAPH_CACHE: dict[str, Any] = {}


def build_step_tensors(wide, feature_cols, symbols, t, use_time_mixing):
    date = wide[feature_cols[0]]["datetime"][t]
    graphs = _GRAPH_CACHE["graphs"]
    if date not in graphs:
        return None, None, None
    adj = torch.tensor(graphs[date], dtype=torch.float32)
    y_row = wide["fwd_ret"].row(t)[1:]
    y = torch.tensor(
        [v if v is not None else float("nan") for v in y_row], dtype=torch.float32
    )

    if use_time_mixing:
        window = 5
        if t < window:
            return None, None, None
        feats = []
        for lag_t in range(t - window + 1, t + 1):
            row = []
            for fc in feature_cols:
                vals = wide[fc].row(lag_t)[1:]
                row.append([v if v is not None else 0.0 for v in vals])
            feats.append(np.array(row).T)  # (n_symbols, n_features)
        x = torch.tensor(
            np.stack(feats, axis=1), dtype=torch.float32
        )  # (n_symbols, window, n_features)
    else:
        row = []
        for fc in feature_cols:
            vals = wide[fc].row(t)[1:]
            row.append([v if v is not None else 0.0 for v in vals])
        x = torch.tensor(
            np.array(row).T, dtype=torch.float32
        )  # (n_symbols, n_features)
    return x, y, adj


def gate_ic(preds: pl.DataFrame) -> dict:
    ic_stats = research.cross_sectional_ic_stats(
        research.cross_sectional_ic(
            preds, pred_col="pred", target_col="fwd_ret", datetime_col="datetime"
        ),
        nw_lag=5,
    )
    passes_003_filter = bool(abs(ic_stats.get("nw_tstat", 0)) > 3)
    leak_flag = bool(abs(ic_stats.get("mean_ic", 0)) > 0.10)
    return {
        **ic_stats,
        "passes_003_survival_filter": passes_003_filter,
        "leak_suspected": leak_flag,
    }


def build_books(preds: pl.DataFrame) -> dict:
    long_only = (
        preds.sort(["datetime", "pred"], descending=[False, True])
        .group_by("datetime", maintain_order=True)
        .head(TOP_K)
        .with_columns((pl.lit(1.0 / TOP_K)).alias("weight"))
        .select("datetime", "symbol", "weight")
    )
    dn = research.dollar_neutral_weights(
        preds,
        pred_col="pred",
        datetime_col="datetime",
        top_frac=TOP_K / preds["symbol"].n_unique(),
    )
    return {"long_only": long_only, "dollar_neutral": dn}


def book_metrics(weights: pl.DataFrame, returns: pl.DataFrame, label: str) -> dict:
    trade_frame = research.portfolio_trade_frame(
        weights, returns, target_col="fwd_ret", datetime_col="datetime"
    )
    turnover_df = research.portfolio_turnover(weights)
    joined = trade_frame.join(turnover_df, on="datetime", how="left").fill_null(0.0)
    joined = joined.with_columns(
        (
            pl.col("trade_log_return") - (TAKER_FEE + SLIPPAGE) * pl.col("turnover")
        ).alias("trade_log_return_net")
    )
    gross = joined["trade_log_return"].to_numpy()
    netr = joined["trade_log_return_net"].to_numpy()

    def m(x, lbl):
        std, mean = float(np.std(x)), float(np.mean(x))
        return {
            "label": lbl,
            "sharpe": (mean / std) * ANNUALIZED_RATE if std > 0 else 0.0,
            "n_bars": len(x),
        }

    ci_lo, ci_hi = (
        research.block_bootstrap_ci(netr, seed=0) if len(netr) > 30 else (None, None)
    )
    return {
        "gross": m(gross, f"{label}_gross"),
        "net": m(netr, f"{label}_net"),
        "ci_95": [ci_lo, ci_hi],
        "ci_excludes_zero": bool(
            ci_lo is not None and ci_hi is not None and (ci_lo > 0 or ci_hi < 0)
        ),
        "mean_turnover": float(np.mean(turnover_df["turnover"].to_numpy())),
    }


def main():
    print("Loading panel...", flush=True)
    panel = load_panel()
    panel, symbols = restrict_to_complete_symbols(panel)
    print(f"Complete-coverage universe: {len(symbols)} symbols", flush=True)
    panel = build_features(panel)
    feature_cols = [f"lag_{i}" for i in range(1, 6)]

    print("Building causal correlation graphs...", flush=True)
    graphs = rolling_graph(panel, symbols)
    _GRAPH_CACHE["graphs"] = graphs
    degree_stats = E.node_degree_stats(graphs)
    print("Node degree stats:", degree_stats, flush=True)

    print("Training no-time-mixing variant...", flush=True)
    preds_no_mix = train_and_predict(
        panel, symbols, feature_cols, use_time_mixing=False
    )
    print("Training time-mixing variant...", flush=True)
    preds_mix = train_and_predict(panel, symbols, feature_cols, use_time_mixing=True)

    ic_no_mix = gate_ic(preds_no_mix)
    ic_mix = gate_ic(preds_mix)

    ic_diff_series = preds_no_mix.select(
        "datetime", "symbol", pl.col("pred").alias("pred_no_mix")
    ).join(
        preds_mix.select(
            "datetime", "symbol", pl.col("pred").alias("pred_mix"), "fwd_ret"
        ),
        on=["datetime", "symbol"],
    )
    ic_no_mix_by_date = research.cross_sectional_ic(
        ic_diff_series.rename({"pred_no_mix": "pred"}), "pred", "fwd_ret", "datetime"
    )
    ic_mix_by_date = research.cross_sectional_ic(
        ic_diff_series.rename({"pred_mix": "pred"}), "pred", "fwd_ret", "datetime"
    )
    joined_ic = ic_no_mix_by_date.join(ic_mix_by_date, on="datetime", suffix="_mix")
    ic_col = next(
        c
        for c in joined_ic.columns
        if c not in ("datetime",) and not c.endswith("_mix")
    )
    ic_col_mix = ic_col + "_mix"
    diff = (joined_ic[ic_col] - joined_ic[ic_col_mix]).drop_nulls().to_numpy()
    tm_ci = (
        research.block_bootstrap_ci(diff, seed=0) if len(diff) > 30 else (None, None)
    )
    gate_tm_fires = bool(tm_ci[0] is not None and tm_ci[0] > 0)

    books_no_mix = build_books(preds_no_mix)
    returns_no_mix = preds_no_mix.select("datetime", "symbol", "fwd_ret")
    long_metrics = book_metrics(books_no_mix["long_only"], returns_no_mix, "long_only")
    dn_metrics = book_metrics(
        books_no_mix["dollar_neutral"], returns_no_mix, "dollar_neutral"
    )

    basket = research.equal_weight_basket_returns(
        returns_no_mix, target_col="fwd_ret", datetime_col="datetime"
    )
    dn_trade = research.portfolio_trade_frame(
        books_no_mix["dollar_neutral"],
        returns_no_mix,
        target_col="fwd_ret",
        datetime_col="datetime",
    )
    joined_beta = dn_trade.join(
        basket.rename({"trade_log_return": "basket_return"}), on="datetime", how="inner"
    ).drop_nulls()
    if joined_beta.height > 30:
        beta = float(
            np.polyfit(
                joined_beta["basket_return"].to_numpy(),
                joined_beta["trade_log_return"].to_numpy(),
                1,
            )[0]
        )
    else:
        beta = None

    throttled = A.throttle_weights(books_no_mix["dollar_neutral"], k=3)
    throttled_metrics = book_metrics(
        throttled, returns_no_mix, "dollar_neutral_throttled"
    )

    dsr_dn = (
        research.deflated_sharpe_prob(
            dn_metrics["net"]["sharpe"] / ANNUALIZED_RATE,
            N_TRIALS_POOLED,
            dn_metrics["net"]["n_bars"],
        )
        if dn_metrics["net"]["n_bars"] > 1
        else float("nan")
    )

    gate_xs_fires = bool(
        dn_metrics["net"]["sharpe"] > 1.5
        and dn_metrics["ci_excludes_zero"]
        and dsr_dn >= 0.95
        and ic_no_mix["passes_003_survival_filter"]
        and beta is not None
        and abs(beta) < 0.2
    )

    out = {
        "gate": "XS",
        "n_trials_pooled": N_TRIALS_POOLED,
        "universe": symbols,
        "universe_size": len(symbols),
        "node_degree": degree_stats,
        "ic_sanity_check": {"no_time_mixing": ic_no_mix, "time_mixing": ic_mix},
        "ic_003_context": {
            "mean_reversion_ic": 0.042,
            "realized_vol_ic": -0.038,
            "source": "notebook 003",
        },
        "gate_TM": {"ci_diff_no_mix_minus_mix": tm_ci, "fires": gate_tm_fires},
        "books": {
            "long_only": long_metrics,
            "dollar_neutral": dn_metrics,
            "dollar_neutral_throttled": throttled_metrics,
        },
        "dollar_neutral_beta_to_basket": beta,
        "dsr_dollar_neutral": dsr_dn,
        "gate_XS_fires": gate_xs_fires,
    }
    with open("src/research/tmp/phase_D_13_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
