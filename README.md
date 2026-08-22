# quant-trading-labs

A research programme in quantitative trading: a sequence of notebooks testing ideas against real
market data, pre-registered and honestly reported — most came back null. One finding held up, and got
turned into production software: a risk engine.

Each notebook fixes what would count as a result **before** running the test, charges realistic
transaction costs everywhere, corrects for how many configurations were tried, and reports what
happened either way. Every write-up in [`src/results/`](src/results/) states the numbers behind its
verdict, including the bugs found along the way.

## Live: Risk Engine Dashboard

[![Risk engine operator dashboard](docs/img/risk-dashboard.png)](https://quant-trading-labs.vercel.app/)

**[Open the live dashboard →](https://quant-trading-labs.vercel.app/)**

`src/risk/` covers 16 daily commodity and equity-index futures. For each product it fits a
calibrated return distribution, computes Value-at-Risk and Expected Shortfall at multiple horizons,
and continuously monitors whether that calibration is still holding as new data comes in — with a
data-cleaning contract, a versioned model selection per product, a calibration monitor, and a refresh
pipeline behind it.

Read [docs/10-risk-engine.md](docs/10-risk-engine.md) for the full write-up: how the data is
cleaned, how the model per product was chosen, how the monitor works, and what the engine has
and hasn't been validated for.

The dashboard is static HTML, CSS, and vanilla JavaScript, with no server. Regenerate it with
`uv run python src/research/tmp/render_risk_dashboard.py` (writes `index.html` at the repo root
from the current `src/risk/data/` ingest; run `risk.ingest.refresh()` first if that's stale).

## What the programme found

**No tradeable edge.** Across more than thirty pre-registered tests — this repo's own constructions
and four independently published outside designs, on crypto perpetuals, commodity futures and
equities — nothing cleared the bar set at the outset. The closest was a crypto funding basis trade
(notebooks 018, 020): a real, statistically significant, genuinely market-neutral carry mechanism
that still could not be shown to beat its own benchmark once costs and multiple testing were charged
honestly.

**Risk modelling is a different story.** Crypto and commodity returns are extremely fat-tailed;
models that ignore that don't merely score worse on an abstract metric, they **measurably
underestimate how bad the worst days get, at every interval checked**. That finding replicated across
asset classes and holdouts, and is what the production risk engine is built on.

**Costs and turnover explain most of the nulls.** Signals are repeatedly real before costs and gone
after them. Where a fix was tried, controlling turnover was necessary but never sufficient.

## Research log

| Research | Results | What it tested, and what happened |
|---|---|---|
| [001a](src/research/001a_mean_reversion_single_asset_ML.ipynb) · [001b](src/research/001b_trend_following_single_asset_ML.ipynb) | [Simple Linear Models](src/results/001_simple_linear.md) | Fitted a straight line to Bitcoin's next return. Found and fixed three real bugs in the fee and backtest code; the resulting "profits" turned out to be a constant directional bet and the best of 756 overfit configurations. |
| [002](src/research/002_walk_forward_multi_asset.ipynb) | [Walk-Forward Validation](src/results/002_walk_forward_multi_asset.md) | Put that approach through five years of walk-forward testing across six symbols, with proper baselines and a correction for search width. No validated edge — and shifting the fold grid by three weeks flips the Sharpe's sign twice. |
| [003](src/research/003_cross_sectional_ic.ipynb) | [Cross-Sectional Screening](src/results/003_cross_sectional_ic.md) | Built a 30-symbol screening pipeline with real transaction costs. The signal is gross-profitable at every interval; costs erase it, the one surviving configuration fails an origin shift, and the holdout came back negative. |
| [004](src/research/004_distributional_models.ipynb) | [Volatility and Regime Models](src/results/004_distributional_models.md) | Stopped predicting direction and asked whether volatility is forecastable and whether regimes carry information. No point-forecast winner; regimes predict risk, not return — replicated on every transfer symbol. |
| [005](src/research/005_tail_risk_evt.ipynb) | [Tail Risk and Extreme Value Theory](src/results/005_tail_risk_evt.md) | Rescored the same models on tail calibration instead of point accuracy. Found a certified density winner at three of four intervals, and the programme's strongest single result: every model ignoring fat tails understates its own worst-1% loss, everywhere. |
| [006](src/research/006_distribution_zoo.ipynb) | [Does the Tail Result Generalise?](src/results/006_distribution_zoo.md) | Tested those claims across more symbols, more quantile levels and four new distribution families. They survive in bounded rather than universal form — and a wider search beats the incumbent model cross-sectionally. |
| [007](src/research/007_alpha_generation.ipynb) | [Is Cost the Thing Blocking Every Edge?](src/results/007_alpha_generation.md) | Cut turnover 71% on a known gross-profitable signal, gated it on predicted tail risk, and tested carry and tail-shape factors. All four null — and the hypothesis that cost alone was the blocker does not survive its own most direct test. |
| [008](src/research/008_commodity_tails_and_risk.ipynb) | [Do the Findings Hold in Commodities?](src/results/008_commodity_tails_and_risk.md) | Rebuilt everything on 16 commodity futures after extensive data-hygiene work (four bugs that would have invalidated the lot). The risk findings replicate cleanly; carry and momentum fail, in a market where the literature expected them to work. |
| [009](src/research/009_external_research_review.ipynb) | [Why Eight Notebooks Found Nothing](src/results/009_external_research_review.md) | Surveyed outside research to adjudicate five competing explanations. Market efficiency and missing infrastructure are well-supported; the cost model being too pessimistic is contradicted outright. |
| [010a](src/research/010a_term_structure_regimes_and_spreads.ipynb) | [Term-Structure Regimes and Spreads](src/results/010a_term_structure_regimes_and_spreads.md) | Built the descriptive groundwork for a spread backtest: a regime atlas, a taxonomy of 30 spreads, and the cointegration check the prior probe skipped — which cut the tradeable universe from 11 to 7. |
| [010b](src/research/010b_spread_strategies.ipynb) | [Five Spread and Sizing Strategies](src/results/010b_spread_strategies.md) | Backtested all five. Spread mean reversion genuinely survives cost but not the search-width correction; volatility-scaled carry clears Sharpe and deflation decisively and fails on drawdown alone. |
| [011a](src/research/011a_methodology_transfer_and_reproduction.ipynb) | [Reproducing an Outside Programme](src/results/011a_methodology_transfer_and_reproduction.md) | Independently verified a second research programme's spread work. Its half-life measurements corroborate exactly; its headline Sharpe does not reproduce at all — but its trade *shape* does, on a book 170× smaller. |
| [011b](src/research/011b_spread_mechanism_gates.ipynb) | [Do Their Mechanisms Actually Help?](src/results/011b_spread_mechanism_gates.md) | Tested seven of that programme's mechanisms against a paired control. All null — and the structured, stopped packaging is significantly *worse* than a plain continuous position. |
| [011c](src/research/011c_entry_time_loss_classifier.ipynb) | [Predicting Which Trades Stop Out](src/results/011c_entry_time_loss_classifier.md) | A walk-forward classifier on 15 entry-time features. It clears its accuracy bar on the point estimate and cannot be distinguished from chance on a bootstrap check; the practical payoff fails outright. |
| [011d](src/research/011d_momentum_breakout_transfer.ipynb) | [A Breakout Rule on Two Asset Classes](src/results/011d_momentum_breakout_transfer.md) | Positive Sharpe everywhere on crypto but far too few trades to conclude anything; net-negative on commodity equities. The gap points at thresholds implicitly calibrated to crypto's volatility. |
| [012](src/research/012_volume_confirmed_breakout.ipynb) | [Does Breakout Volume Confirm the Move?](src/results/012_volume_confirmed_breakout.md) | The best-powered book in the programme — 88 instruments, three asset classes, an honest 12 trials. The volume filter's point estimate moves the **wrong** way against its own control. |
| [013](src/research/013_four_outside_designs_rebuilt_and_scored.ipynb) | [Four Published Designs, Rebuilt](src/results/013_four_outside_designs_rebuilt_and_scored.md) | Rebuilt four externally published strategies faithfully, including the mechanism each credits for its edge. None reproduces. One fails in the sharpest way possible: a significant signal pointing the wrong direction. |
| [014](src/research/014_market_regime_engine_and_accuracy.ipynb) | [The Regime Engine, Scored](src/results/014_market_regime_engine_and_accuracy.md) | Ported the live production regime engine and measured its accuracy for the first time. The port is byte-identical and structurally sound; the labels fail every accuracy check against two independent sources. |
| [015](src/research/015_trend_ceiling_and_independent_validation.ipynb) | [The Trend Ceiling](src/results/015_trend_ceiling_and_independent_validation.md) | Re-tested the two ambiguous regime dimensions against provably disjoint targets, and ran a model-capacity ceiling test on trend predictability. Both settled as nulls — with a quantified detection bound. |
| [016](src/research/016_polynomial_power_features.ipynb) | [Polynomial Power Features](src/results/016_polynomial_power_features.md) | Replaced one feature with three powers of it. No hidden signal — and gross Sharpe turns out uncorrelated with net Sharpe, because turnover, not fit quality, decided which configuration lost least. |
| [017](src/research/017_deflated_sharpe_correction.ipynb) | [Repairing the Deflation Estimator](src/results/017_deflated_sharpe_correction.md) | Confirmed by simulation across 756 grid cells that the estimator mis-scales for correlated trial families. No candidate repair was both well-calibrated and powerful enough to adopt, so every historical value stands as recorded. |
| [018](src/research/018_funding_basis_trade.ipynb) | [The Funding Basis Trade](src/results/018_funding_basis_trade.md) | The first structurally non-directional trade in the programme. The carry mechanism is real and significant, the hedge is genuinely delta-neutral — and the tradeability bar is still not cleared. |
| [019](src/research/019_dsr_correlation_switch.ipynb) | [A Correlation-Triggered Deflation Switch](src/results/019_dsr_correlation_switch.md) | Routed the repair through a cheap correlation estimate. The switch mechanism and its prediction machinery both check out cleanly; the claimed validated regime missed one out-of-sample cell by a modest margin. |
| [020](src/research/020_basis_refinement_and_cross_venue.ipynb) | [Basis Refinement and Cross-Venue Spread](src/results/020_basis_refinement_and_cross_venue.md) | A diversification floor and slower carry lift net Sharpe from 0.58 to 3.89 and clear deflation decisively — but not the paired comparison against the original book. A cross-venue funding spread loses to the plain single-venue trade. |
| [021](src/research/021_rc3_power_and_data_quality.ipynb) | [Power or Data Quality?](src/results/021_rc3_power_and_data_quality.md) | Asked whether that paired comparison was blocked by frozen-feed artefacts or by sample size. A mechanical exclusion rule nominally clears the interval and then fails its own placebo control: the answer is power — roughly 27.6 years of history would be needed, against the 3.50 available. |

## Repository layout

| Path | What's in it |
|---|---|
| [`src/research/`](src/research/) | The notebooks, in order. |
| [`src/results/`](src/results/) | One write-up per notebook — the numbers, the verdicts, and the bugs. |
| [`src/risk/`](src/risk/) | The production risk engine: ingest, model selection, calibration monitor, serving. |
| [`src/regime/`](src/regime/) | The ported market-regime engine (scored in notebook 014). |
| [`src/`](src/) | Shared machinery: walk-forward validation, distributions and scoring rules, feature engineering, data loading. |
| [`docs/`](docs/README.md) | Concepts defined from scratch, with worked examples from this repo's own numbers. |
| [`tests/`](tests/) | The test suite. |

## Running things

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run pytest tests/ -q
```
