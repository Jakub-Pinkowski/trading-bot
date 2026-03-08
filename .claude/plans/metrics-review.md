# Metrics Review

Covers all metrics produced by `per_trade_metrics.py` and `summary_metrics.py`, with observations
from the live CSV output and the aggregation logic in `strategy_analyzer.py`.

---

## Legend

- **Fine** — correct, interval-independent, no issues
- **Debatable** — valid concept but has caveats worth knowing
- **Broken/Misleading** — produces wrong or incomparable values in current form
- **Missing** — not computed but would be useful
- **Fixed** — was broken/missing, now resolved

---

## Per-Trade Metrics (`per_trade_metrics.py`)

| Metric                                                         | Status    | Notes                                                                                                                                                                                                                                                             |
|----------------------------------------------------------------|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `entry_time`, `exit_time`, `entry_price`, `exit_price`, `side` | Fine      | Raw trade data, no issues                                                                                                                                                                                                                                         |
| `duration` (timedelta)                                         | Fine      | Correct per-trade. Runner converts to `duration_bars` using interval. Hours derivable as `duration.total_seconds() / 3600`                                                                                                                                        |
| `net_pnl`                                                      | Fine      | Gross P&L minus fixed $4 commission. Commission is flat regardless of contract size or slippage — acceptable simplification                                                                                                                                       |
| `return_percentage_of_contract`                                | Fine      | `net_pnl / (entry_price × multiplier) × 100`. Consistent within a symbol. Problematic when aggregated across symbols with different contract values — a 0.1% move on ZS ($22k contract) ≠ 0.1% on CL ($70k contract) in capital terms                             |
| `return_percentage_of_margin`                                  | Debatable | Correct concept — normalizes by actual capital at risk. But margin ratios are fixed class-level estimates from Jan 2026, applied to all historical periods. Margin requirements change with volatility. Computed per-trade; not surfaced in summary (intentional) |
| `margin_requirement`                                           | Debatable | Derived from fixed asset-class ratios. Acceptable for relative comparison, misleading as an absolute dollar figure for historical trades                                                                                                                          |
| `commission`                                                   | Debatable | Flat $4 per trade regardless of contract size. A $4 commission on a $70k CL trade (0.006%) vs a $4 commission on a $5k MZC trade (0.08%) creates inconsistent cost modelling across symbols                                                                       |

---

## Summary Metrics (`summary_metrics.py`)

### Basic Trade Statistics

| Metric                                | Status    | Notes                                                                                                                                                                                                                  |
|---------------------------------------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `total_trades`                        | Fine      |                                                                                                                                                                                                                        |
| `winning_trades` / `losing_trades`    | Fine      | Counts, not percentages — fine. Break-even trades (0% return) classified as losses — intentional, documented in code                                                                                                   |
| `win_rate`                            | Fine      | Interval-independent ratio                                                                                                                                                                                             |
| `average_trade_duration_bars`         | **Fixed** | Was `average_trade_duration_hours` — broken for cross-interval comparison. Now computed as `duration / interval_hours` per trade in runner, averaged in summary. A 4-bar trade is a 4-bar trade regardless of interval |
| `win_loss_ratio`                      | **Fixed** | Was missing. Now `avg_win / abs(avg_loss)`. Returns `INFINITY_REPLACEMENT` when no losses. Complements `win_rate` — high win rate + low W/L ratio signals a marginal strategy                                          |
| `max_consecutive_wins`                | **Fixed** | Was missing. Characterises strategy type (trend vs mean-reversion)                                                                                                                                                     |
| `max_consecutive_losses`              | **Fixed** | Was missing. Determines minimum survival capital — more direct than max_drawdown                                                                                                                                       |
| `total_wins_percentage_of_contract`   | Debatable | Useful as an intermediate for profit_factor, but grows with trade count as a standalone metric                                                                                                                         |
| `total_losses_percentage_of_contract` | Debatable | Same as above                                                                                                                                                                                                          |

### Return Metrics

| Metric                                        | Status    | Notes                                                                                                                                                                                                                                        |
|-----------------------------------------------|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `total_return_percentage_of_contract`         | Debatable | Still accumulates with trade frequency — a 15m strategy produces ~4× more trades than 4h over the same period. Kept as-is since `expectancy_per_bar` and `avg_return_per_symbol_pct_contract` (aggregation) serve as normalised alternatives |
| `average_trade_return_percentage_of_contract` | **Fixed** | Was `total_return / total_trades` (via single-element list). Now correctly `safe_average(self.returns)`. Numerically identical but semantically correct. This is the classic **expectancy** per trade                                        |
| `average_win_percentage_of_contract`          | Fine      | Per-trade average, interval-independent                                                                                                                                                                                                      |
| `average_loss_percentage_of_contract`         | Fine      | Per-trade average, interval-independent                                                                                                                                                                                                      |
| `profit_factor`                               | Fine      | `total_wins% / abs(total_losses%)`. Interval-independent ratio. Infinity replaced with `INFINITY_REPLACEMENT` (9999.99) — safe for aggregation                                                                                               |
| `expectancy_per_bar`                          | **Fixed** | Was missing. `avg_trade_return / avg_duration_bars`. Normalises for both trade size and time held. Most cross-interval-comparable return metric in the set                                                                                   |

### Risk Metrics

| Metric                        | Status    | Notes                                                                                                                                                                                                                                                                                       |
|-------------------------------|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `maximum_drawdown_percentage` | Debatable | Calculated on cumulative trade-return sequence, not on a time series. Cross-interval comparison unreliable (more trades = more opportunities for drawdown). Within a single (symbol, interval) run it is meaningful                                                                         |
| `sharpe_ratio`                | **Fixed** | Was using `ddof=0` (population std dev). Now `ddof=1` (sample std dev). Still uses per-trade returns rather than time-period returns — a known approximation documented in code. Not comparable across intervals                                                                            |
| `sortino_ratio`               | **Fixed** | Denominator now divides by N (all trades), not N_negative. Fixes inflation when loss rate is low. Same time-unit approximation caveat as Sharpe                                                                                                                                             |
| `calmar_ratio`                | **Fixed** | Was using cumulative total return (grew with trade count). Now annualised via `total_return × (8760 / dataset_total_hours)`. Falls back to cumulative return if `dataset_total_hours` unavailable, with docstring warning. `dataset_total_hours` is always passed from runner since Phase 2 |
| `value_at_risk`               | **Fixed** | `MIN_RETURNS_FOR_VAR` raised from 5 to 30. Tail boundary uses `floor()` to avoid overshooting. With N=30 at 95% CL: `floor(0.05×30) = 1` — still thin, but the minimum threshold ensures basic statistical meaning. Not time-normalised; not comparable across intervals                    |
| `expected_shortfall`          | **Fixed** | Same MIN threshold fix as VaR. Averages the `tail_count` worst returns, consistent with VaR boundary                                                                                                                                                                                        |
| `ulcer_index`                 | Debatable | Measures depth × duration of drawdowns in trade-sequence space. Same cross-interval incompatibility as max drawdown. Within a single run it is a meaningful penaliser for strategies that stay underwater for many consecutive trades                                                       |
| `time_in_market_percentage`   | **Fixed** | Was missing. `sum(duration_hours) / dataset_total_hours × 100`. Returns 0 if `dataset_total_hours` not available. Useful for opportunity cost and margin usage analysis                                                                                                                     |

### Missing / Not Yet Added

These were not part of the original review but emerged as useful candidates:

| Metric                                | Formula                                 | Priority | Rationale                                                                                                                                                                      |
|---------------------------------------|-----------------------------------------|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `largest_win_percentage_of_contract`  | `max(returns)`                          | Medium   | Detects outlier-driven strategies. If one trade accounts for the majority of total return, the strategy is fragile. Zero implementation complexity                             |
| `largest_loss_percentage_of_contract` | `min(returns)`                          | Medium   | Pair with largest_win. Tells you the worst single trade — relevant for position sizing and account survival                                                                    |
| `return_skewness`                     | `mean(((r - μ) / σ)³)` over all returns | Medium   | Characterises distribution shape. Negative skew = rare large losses hidden behind frequent small wins (dangerous). Positive skew = occasional large wins. Pure numpy, no scipy |

---

## Aggregation (`strategy_analyzer.py`)

| Issue                                                                              | Status    | Detail                                                                                                                                                                                                          |
|------------------------------------------------------------------------------------|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `average_trade_duration_hours` simple-averaged across combinations                 | **Fixed** | Replaced by `average_trade_duration_bars` which is trade-weighted across combinations in the weighted branch                                                                                                    |
| `total_return_percentage_of_contract` summed across combinations                   | **Fixed** | Renamed to `total_return_all_symbols_pct_contract` (intentionally sums — represents total capital deployed). Added `avg_return_per_symbol_pct_contract` as the symbol-count-neutral alternative                 |
| `sharpe_ratio` / `sortino_ratio` / `calmar_ratio` averaged across combinations     | Debatable | Now trade-weighted with `MIN_TRADES_FOR_RATIO = 30` filter to reduce noise. Documented limitation: ideally recomputed from the combined trade stream. Out of scope without storing per-trade records in parquet |
| Simple (non-weighted) averaging treats a 10-trade combo equal to a 100-trade combo | Debatable | Min-trades filter helps but weighting problem remains at the boundary. Acceptable given the alternative requires per-trade storage                                                                              |

---

## Known Non-Goals

- Funding rate modelling for crypto perps
- Time-series-based Sharpe (requires a daily P&L series, not per-trade returns)
- Recomputing Sharpe/Sortino/Calmar from the combined trade stream in aggregation (requires storing raw trade records in
  parquet — large schema change)
- Dynamic margin requirements (real SPAN/TIMS margin)
- Max drawdown duration in number of trades (useful but adds implementation complexity)
- System Quality Number (another ratio on top of existing ones, marginal value)