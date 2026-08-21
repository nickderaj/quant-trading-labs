# quant-trading-labs

A research programme in quantitative trading: a sequence of notebooks testing ideas against
real market data, pre-registered and honestly reported — most came back null. One finding held
up, and got turned into production software: a risk engine.

## Live: Risk Engine Dashboard

[![Risk engine operator dashboard](docs/img/risk-dashboard.png)](https://quant-trading-labs.vercel.app/)

**[Open the live dashboard →](https://quant-trading-labs.vercel.app/)**

`src/risk/` covers 16 daily commodity and equity-index futures. For each product it fits a
calibrated return distribution, computes Value-at-Risk and Expected Shortfall at multiple
horizons, and continuously monitors whether that calibration is still holding as new data comes
in — with a data-cleaning contract, a versioned model selection per product, a calibration
monitor, and a refresh pipeline behind it.

Read [docs/10-risk-engine.md](docs/10-risk-engine.md) for the full write-up: how the data is
cleaned, how the model per product was chosen, how the monitor works, and what the engine has
and hasn't been validated for.

The dashboard is static HTML, CSS, and vanilla JavaScript, with no server. Regenerate it with
`uv run python src/research/tmp/render_risk_dashboard.py` (writes `index.html` at the repo root
from the current `src/risk/data/` ingest; run `risk.ingest.refresh()` first if that's stale).

## Research log

| Research | Results | Summary |
|---|---|---|
| [001a](src/research/001a_mean_reversion_single_asset_ML.ipynb) · [001b](src/research/001b_trend_following_single_asset_ML.ipynb) | [Simple Linear](src/results/001_simple_linear.md) | Tested whether a simple linear model trading BTC's return sign was profitable, after fixing three fee/backtest bugs; results untrustworthy (overfitting/constant bets). |
| [002](src/research/002_walk_forward_multi_asset.ipynb) | [Walk-Forward Multi-Asset](src/results/002_walk_forward_multi_asset.md) | Rigorously walk-forward validated the linear approach across 5 years and 6 symbols with proper baselines and deflated Sharpe; found no validated edge. |
| [003](src/research/003_cross_sectional_ic.ipynb) | [Cross-Sectional IC](src/results/003_cross_sectional_ic.md) | Built a 30-symbol cross-sectional IC-screening pipeline with real transaction costs; signal was gross-profitable but costs and a holdout run erased any edge. |
| [004](src/research/004_distributional_models.ipynb) | [Distributional Models](src/results/004_distributional_models.md) | Tested whether distributional models could forecast BTC volatility or identify informative regimes instead of predicting direction; found no point-forecast winner and regimes predict risk, not return. |
| [005](src/research/005_tail_risk_evt.ipynb) | [Tail Risk (EVT)](src/results/005_tail_risk_evt.md) | Tested which distributional models best calibrate BTC's conditional tail risk (GARCH-t, GJR-GARCH, EVT); found statistically certified tail-calibration winners but no application cleared the bar for trading. |
| [006](src/research/006_distribution_zoo.ipynb) | [Distribution Zoo](src/results/006_distribution_zoo.md) | Tests whether notebook 5's BTC-only tail-risk findings generalize across more crypto symbols, intervals, and distribution families. |
| [007](src/research/007_alpha_generation.ipynb) | [Alpha Generation](src/results/007_alpha_generation.md) | Tests whether cutting transaction costs on a known-profitable crypto signal, plus carry and tail-factor signals, can produce tradeable alpha. |
| [008](src/research/008_commodity_tails_and_risk.ipynb) | [Commodity Tails and Risk](src/results/008_commodity_tails_and_risk.md) | Tests whether crypto's fat-tail and risk-calibration findings, plus carry/momentum strategies, replicate in commodity futures. |
| [009](src/research/009_external_research_review.ipynb) | [External Research Review](src/results/009_external_research_review.md) | Surveys external research literature to diagnose why eight notebooks found no tradeable crypto/commodity alpha. |
| [010a](src/research/010a_term_structure_regimes_and_spreads.ipynb) | [Term Structure Regimes and Spreads](src/results/010a_term_structure_regimes_and_spreads.md) | Builds a term-structure regime atlas and spread taxonomy to pre-register a future commodity spread mean-reversion backtest. |
| [010b](src/research/010b_spread_strategies.ipynb) | [Spread Strategies](src/results/010b_spread_strategies.md) | Tested five commodity spread-trading gates (mean-reversion, regime-gating, vol-scaled carry, blended momentum) for cost-surviving alpha; all five came back null. |
| [011a](src/research/011a_methodology_transfer_and_reproduction.ipynb) | [Methodology Transfer and Reproduction](src/results/011a_methodology_transfer_and_reproduction.md) | Reproduced and validated an external spread-trading programme's half-life and control-book results on this repo's own data before pre-registering gates for follow-on notebooks. |
| [011b](src/research/011b_spread_mechanism_gates.ipynb) | [Spread Mechanism Gates](src/results/011b_spread_mechanism_gates.md) | Tested whether an external programme's discrete-trade spread mechanisms (stops, sign-flips, screens, reentry sweeps) beat this repo's simple continuous benchmark; none did. |
| [011c](src/research/011c_entry_time_loss_classifier.ipynb) | [Entry-Time Loss Classifier](src/results/011c_entry_time_loss_classifier.md) | Tested a walk-forward classifier predicting which spread trades would stop out from entry-time features; it did not reliably beat chance or improve the trading book. |
| [011d](src/research/011d_momentum_breakout_transfer.ipynb) | [Momentum Breakout Transfer](src/results/011d_momentum_breakout_transfer.md) | Tested a transferred breakout trading rule on crypto perpetuals and commodity-equities; crypto was inconclusive due to small sample, equities failed outright. |
| [012](src/research/012_volume_confirmed_breakout.ipynb) | [Volume-Confirmed Breakout](src/results/012_volume_confirmed_breakout.md) | Tested whether gating a symmetric bull-flag breakout with a volume-confirmation filter improves an 88-instrument, three-asset-class pooled backtest; the filter failed to pay for itself. |
| [013](src/research/013_four_outside_designs_rebuilt_and_scored.ipynb) | [Four Outside Designs, Rebuilt and Scored](src/results/013_four_outside_designs_rebuilt_and_scored.md) | Rebuilt four externally-published trading strategies faithfully on this repo's own data and costs to see if any reproduce; none did. |
| [014](src/research/014_market_regime_engine_and_accuracy.ipynb) | [Market Regime Engine and Accuracy](src/results/014_market_regime_engine_and_accuracy.md) | Ported the live production market-regime engine and scored its historical label accuracy against two independent ground-truth sources; the port was faithful but accuracy and crisis-lag checks failed. |
| [015](src/research/015_trend_ceiling_and_independent_validation.ipynb) | [Trend Ceiling and Independent Validation](src/results/015_trend_ceiling_and_independent_validation.md) | Re-tested 014's two ambiguous regime dimensions against fully independent targets and ran a ceiling test on directional trend predictability; both came back as settled nulls with a quantified detection bound. |
| [016](src/research/016_polynomial_power_features.ipynb) | [Polynomial Power Features](src/results/016_polynomial_power_features.md) | Swapped notebook 1's single linear feature for 3-feature polynomial power combinations of the same lag, walk-forward validated; no combo found a real edge, and turnover (not signal quality) drove which config lost the least. |
| [017](src/research/017_deflated_sharpe_correction.ipynb) | [Deflated Sharpe Correction](src/results/017_deflated_sharpe_correction.md) | Diagnosed and tried to repair a real defect in the Deflated Sharpe estimator (it mis-scales for correlated trial families, confirmed by Monte Carlo across 756 grid cells); no candidate repair was simultaneously well-calibrated and powerful enough to adopt, so `research.py` is unchanged and every historical DSR value in the repo stands as before. |
| [018](src/research/018_funding_basis_trade.ipynb) | [Funding Basis Trade](src/results/018_funding_basis_trade.md) | Tested the crypto perpetual funding basis trade (long spot, short perp, delta-neutral) that 009 shortlisted but couldn't confirm; the carry mechanism is real and significant and the hedge is genuinely delta-neutral, but net Sharpe's bootstrap CI and deflated Sharpe don't clear the tradeable-alpha bar, so the holdout stays unspent. |
| [019](src/research/019_dsr_correlation_switch.ipynb) | [DSR Correlation Switch](src/results/019_dsr_correlation_switch.md) | Tested whether routing 017's Deflated Sharpe repair through a cheap inter-trial correlation estimate could separate a fixable power loss from an unfixable one; the switch mechanism and its prediction machinery both check out cleanly, but the claimed N≥12 validated-regime boundary missed one out-of-sample confirmation cell by a modest margin, so `research.py` stays unchanged. |
