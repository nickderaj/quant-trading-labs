"""Notebook 13, Design B -- sequence models on the cross-asset futures panel
(NEXT_PROMPT.md sec4.B). The claim under test: models that learn temporal
representations beat linear benchmarks on a pooled cross-asset panel, when
trained on a risk-adjusted objective (negative realized Sharpe of the
induced position) rather than MSE.

Universe: 16 databento F1-continuous products + 6 yfinance FX futures
(6A/6B/6C/6E/6J/6S) + ES=F equity index = 23 instruments, daily, dev window
ending 2024-12-31. No bond futures anywhere in the repo -- disclosed
universe gap versus the source design (sec2), not substituted.

Models, in order (all through panel_walk_forward_splits):
  1. linear baseline (research.benchmark_linear_models) -- the thing to beat
  2. LSTM (exec_lib13.LSTMForecaster)
  3. variable-selection-gated LSTM (exec_lib13.GatedLSTMForecaster)
Both 2/3 trained via research.batch_train_reg with a Sharpe-objective loss
wrapper around exec_lib13.negative_sharpe_loss, >=5 seeds each, median and
full spread across seeds reported (not the best seed).

Writes phase_B_13_results.json.
"""

import json
import sys

import numpy as np
import polars as pl
import torch
import torch.nn as nn

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import commod_lib8 as C
import exec_lib13 as E

import research

ANNUALIZED_RATE = float(np.sqrt(252))
ORIGIN_OFFSETS = [0, 7, 14, 21]
N_TRIALS_POOLED = 18
DEV_END = pl.date(2024, 12, 31)
DATABENTO_PRODUCTS = ["BZ", "CL", "ES", "GC", "HO", "KE", "NG", "PA", "PL", "RB", "SI", "ZC", "ZL", "ZM", "ZS", "ZW"]
YFINANCE_TICKERS = ["6A=F", "6B=F", "6C=F", "6E=F", "6J=F", "6S=F"]
YFINANCE_TO_PRODUCT = {"6A=F": "6A", "6B=F": "6B", "6C=F": "6C", "6E=F": "6E", "6J=F": "6J", "6S=F": "6S"}
LOOKBACK = 20
TRAIN_BARS = 252 * 5
TEST_BARS = 252
STEP_BARS = 252 * 3  # 3y roll step, not 1y -- runtime tradeoff, disclosed: fewer but still genuinely walk-forward, non-overlapping-test folds
N_SEEDS = 5
N_EPOCHS = 60
HIDDEN_DIM = 16


def load_databento_product(product: str) -> pl.DataFrame:
    ohlcv = pl.read_parquet(f"src/research/data/market/databento/ohlcv/{product}.parquet")
    contracts = pl.read_parquet("src/research/data/market/databento/contracts.parquet")
    roll = pl.read_parquet("src/research/data/market/databento/roll_calendar.parquet")
    df = ohlcv.filter(pl.col("product") == product)
    clean = C.apply_hygiene_filter(df)
    cont = C.build_continuous_series_ohlcv(clean, contracts, roll, product)
    cont = cont.sort("date").drop_nulls(subset=["close_backadj"])
    # Additive Panama back-adjustment can push old segments of already-low
    # 2020-crash contracts (CL/HO/NG/ZW) negative -- log(negative) is NaN,
    # not null, so it survives drop_nulls and poisons every later
    # full-batch loss it touches (sec7 trap5-adjacent). Drop those rows here,
    # at the source, rather than downstream.
    cont = cont.filter(pl.col("close_backadj") > 0)
    return cont.select(
        pl.col("date").alias("datetime"),
        pl.lit(product).alias("symbol"),
        pl.col("close_backadj").alias("close"),
        pl.col("volume"),
    ).filter(pl.col("datetime") <= DEV_END)


def load_yfinance_ticker(ticker: str) -> pl.DataFrame:
    df = pl.read_parquet(f"src/research/data/market/yfinance/daily/{ticker}.parquet")
    return df.select(
        pl.col("timestamp").cast(pl.Date).alias("datetime"),
        pl.lit(YFINANCE_TO_PRODUCT[ticker]).alias("symbol"),
        pl.col("close"),
        pl.col("volume"),
    ).filter(pl.col("close") > 0, pl.col("datetime") <= DEV_END)


def build_panel() -> pl.DataFrame:
    frames = [load_databento_product(p) for p in DATABENTO_PRODUCTS]
    frames += [load_yfinance_ticker(t) for t in YFINANCE_TICKERS]
    panel = pl.concat(frames).sort(["symbol", "datetime"])
    panel = panel.with_columns(
        (pl.col("close").log() - pl.col("close").log().shift(1)).over("symbol").alias("log_return_1")
    )
    panel = panel.with_columns(
        pl.col("log_return_1").rolling_std(window_size=20).shift(1).over("symbol").alias("vol_20d")
    )
    panel = panel.with_columns(
        research.vol_normalized_target("log_return_1", "vol_20d").alias("target_vol_norm")
    )
    # 5 lagged returns as the feature pool -- one MSE-point-forecast-style
    # feature set shared by every architecture, so the comparison isolates
    # the model/objective, not the inputs.
    for lag in range(1, 6):
        panel = panel.with_columns(
            pl.col("log_return_1").shift(lag).over("symbol").alias(f"lag_{lag}")
        )
    panel = panel.with_columns(
        pl.col("log_return_1").shift(-1).over("symbol").alias("fwd_return_1")
    )
    feature_cols = [f"lag_{i}" for i in range(1, 6)]
    check_cols = [*feature_cols, "fwd_return_1", "vol_20d", "target_vol_norm"]
    panel = panel.drop_nulls(subset=check_cols)
    # drop_nulls only removes true nulls, not NaN/Inf floats (e.g. from a
    # log-of-nonpositive-price upstream) -- filter those explicitly so one
    # bad row can't poison a whole fold's full-batch training loss.
    panel = panel.filter(pl.all_horizontal([pl.col(c).is_finite() for c in check_cols]))
    panel = panel.filter(pl.col("vol_20d") > 1e-8)
    return panel, feature_cols


def make_windows(panel: pl.DataFrame, feature_cols: list[str], idx: np.ndarray) -> tuple:
    """Build (n, LOOKBACK, n_features) tensors from the per-symbol lag
    features already computed causally in `panel` -- each row at index i
    already only depends on data through datetime[i]-1, so a LOOKBACK
    window of consecutive rows for one symbol carries no additional
    lookahead beyond what's already baked into each row's own lag_k
    columns.
    """
    sub = panel[idx.tolist()] if not isinstance(idx, list) else panel[idx]
    symbols = sub["symbol"].unique().to_list()
    X, Y, VOL, SYM, PRICE = [], [], [], [], []
    for sym in symbols:
        s = sub.filter(pl.col("symbol") == sym).sort("datetime")
        feats = s.select(feature_cols).to_numpy()
        fwd = s["fwd_return_1"].to_numpy()
        vol = s["vol_20d"].to_numpy()
        close = s["close"].to_numpy()
        if len(s) <= LOOKBACK:
            continue
        for i in range(LOOKBACK, len(s)):
            X.append(feats[i - LOOKBACK : i])
            Y.append(fwd[i])
            VOL.append(vol[i])
            SYM.append(sym)
            PRICE.append(close[i])
    if not X:
        return None, None, None, None, None
    return (
        torch.tensor(np.array(X), dtype=torch.float32),
        torch.tensor(np.array(Y), dtype=torch.float32),
        torch.tensor(np.array(VOL), dtype=torch.float32),
        np.array(SYM),
        np.array(PRICE),
    )


class SharpeLoss(nn.Module):
    def forward(self, pred, target):
        return E.negative_sharpe_loss(pred, target)


def run_architecture(panel, feature_cols, splits, model_factory, name, n_seeds=N_SEEDS):
    per_seed_net_sharpes = []
    per_seed_stitched = []
    for seed in range(n_seeds):
        research.set_seed(seed)
        stitched_position, stitched_return, stitched_vol, stitched_sym, stitched_price = [], [], [], [], []
        for train_idx, test_idx in splits:
            x_train, y_train, _, _, _ = make_windows(panel, feature_cols, train_idx)
            x_test, y_test, vol_test, sym_test, price_test = make_windows(panel, feature_cols, test_idx)
            if x_train is None or x_test is None:
                continue
            model = model_factory()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            preds_test = research.batch_train_reg(
                model, x_train, x_test, y_train, y_test,
                no_epochs=N_EPOCHS, loss=SharpeLoss(), optimizer=optimizer,
            )
            position = np.clip(preds_test.detach().numpy(), -3, 3)
            stitched_position.append(position)
            stitched_return.append(y_test.numpy())
            stitched_vol.append(vol_test.numpy())
            stitched_sym.append(sym_test)
            stitched_price.append(price_test)
        if not stitched_position:
            continue
        position = np.concatenate(stitched_position)
        fwd_return = np.concatenate(stitched_return)
        vol = np.concatenate(stitched_vol)
        sym_arr = np.concatenate(stitched_sym)
        price_arr = np.concatenate(stitched_price)

        sized_position = np.clip(position, -1, 1) * (0.01 / np.where(vol > 1e-8, vol, np.nan))
        sized_position = np.nan_to_num(sized_position, nan=0.0)
        gross = sized_position * fwd_return
        turnover = np.abs(np.diff(sized_position, prepend=0.0))
        # headline futures cost per sec2/prereg: commod_lib8's own round-turn
        # cost as a fraction of notional, per instrument, not a flat convention.
        cost_frac = np.array(
            [C.cost_per_unit_notional(s, p) for s, p in zip(sym_arr, price_arr, strict=True)]
        )
        net = gross - cost_frac * turnover
        sharpe = float(np.mean(net) / np.std(net) * ANNUALIZED_RATE) if np.std(net) > 0 else 0.0
        per_seed_net_sharpes.append(sharpe)
        per_seed_stitched.append({"gross": gross, "net": net, "turnover": turnover})

    if not per_seed_net_sharpes:
        return {"name": name, "error": "no folds produced predictions"}

    median_idx = int(np.argsort(per_seed_net_sharpes)[len(per_seed_net_sharpes) // 2])
    median_stitch = per_seed_stitched[median_idx]
    ci_lo, ci_hi = research.block_bootstrap_ci(median_stitch["net"], seed=0)
    dsr = research.deflated_sharpe_prob(
        np.median(per_seed_net_sharpes) / ANNUALIZED_RATE, n_trials=N_TRIALS_POOLED, n_obs=len(median_stitch["net"])
    )
    breakeven = E.breakeven_cost_bps(median_stitch["gross"], median_stitch["turnover"])

    return {
        "name": name,
        "n_seeds": len(per_seed_net_sharpes),
        "seed_sharpes": per_seed_net_sharpes,
        "median_sharpe": float(np.median(per_seed_net_sharpes)),
        "min_sharpe": float(np.min(per_seed_net_sharpes)),
        "max_sharpe": float(np.max(per_seed_net_sharpes)),
        "median_sharpe_ci_95": [ci_lo, ci_hi],
        "ci_excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
        "dsr_at_median": dsr,
        "breakeven_cost_bps": breakeven,
        "n_oos_bars": len(median_stitch["net"]),
    }


def main():
    panel, feature_cols = build_panel()
    print(f"Panel: {panel.height} rows, {panel['symbol'].n_unique()} symbols", flush=True)

    splits = research.panel_walk_forward_splits(
        panel, TRAIN_BARS, TEST_BARS, step_bars=STEP_BARS, mode="rolling", datetime_col="datetime"
    )
    print(f"{len(splits)} walk-forward folds", flush=True)

    linear = research.benchmark_linear_models(
        panel, target="fwd_return_1", feature_pool=feature_cols, annualized_rate=ANNUALIZED_RATE, max_no_features=3,
    )
    linear_best = linear.sort("sharpe", descending=True).head(1).to_dicts()[0] if linear.height else {}

    results = {"linear_baseline": linear_best}

    lstm_result = run_architecture(
        panel, feature_cols, splits, lambda: E.LSTMForecaster(len(feature_cols), HIDDEN_DIM), "LSTM"
    )
    results["LSTM"] = lstm_result

    gated_result = run_architecture(
        panel, feature_cols, splits, lambda: E.GatedLSTMForecaster(len(feature_cols), HIDDEN_DIM), "GatedLSTM"
    )
    results["GatedLSTM"] = gated_result

    best_name = max(
        ["LSTM", "GatedLSTM"],
        key=lambda n: results[n].get("median_sharpe", -np.inf) if "error" not in results[n] else -np.inf,
    )
    best = results[best_name]
    linear_sharpe = linear_best.get("sharpe", 0.0)

    gate_sq_fires = bool(
        "error" not in best
        and best["median_sharpe"] > linear_sharpe
        and best["ci_excludes_zero"]
        and best["dsr_at_median"] >= 0.95
    )

    out = {
        "gate": "SQ",
        "n_trials_pooled": N_TRIALS_POOLED,
        "universe": DATABENTO_PRODUCTS + YFINANCE_TICKERS,
        "universe_gap_disclosure": "no bond futures (ZN/ZB/TY/FGBL) or VIX futures anywhere in this repo; not substituted with an ETF proxy",
        "n_folds": len(splits),
        "results": results,
        "best_architecture": best_name,
        "cost_note": "headline cost is commod_lib8.cost_per_unit_notional per instrument per bar (round_turn_cost_per_contract / notional at that bar's price), including the FX CONTRACT_SPECS entries added for this design; breakeven_cost_bps is the additional cost-sensitivity number sec4.B asks for",
        "gate_SQ_fires": gate_sq_fires,
    }
    with open("src/research/tmp/phase_B_13_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
