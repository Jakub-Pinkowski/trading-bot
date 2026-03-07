# Metrics Improvements Plan

Based on `metrics-review.md`. Covers bug fixes, normalization improvements, aggregation
correctness, new metrics, and an instrument abstraction layer for future Binance crypto support.

---

## Context: How Interval Flows Through the Stack

Currently the `interval` string (e.g., `'4h'`) is known at the runner level
(`runner.py`) but is never passed into `SummaryMetrics` or stored on individual trade
dicts. Several fixes below require it. The cleanest approach is to pass `interval` into
`run_single_test` and attach interval-derived values to each trade dict before
`SummaryMetrics` is called, rather than modifying `SummaryMetrics` to accept an
`interval` param.

```
runner.py
  → compute interval_hours from INTERVAL_HOURS constant
  → compute dataset_total_hours from df timestamps (df.iloc[-1] entry_time − df.iloc[0] entry_time)
  → add duration_bars = duration_hours / interval_hours to each trade dict
  → pass dataset_total_hours into SummaryMetrics (new constructor param)
```

New constant needed in `summary_metrics.py` (or a shared `constants.py`):

```python
INTERVAL_HOURS = {
    '1m': 1 / 60, '5m': 5 / 60, '15m': 0.25, '30m': 0.5,
    '1h': 1.0, '2h': 2.0, '4h': 4.0, '1d': 24.0, '1w': 168.0
}
```

---

## Phase 1 — Bug Fixes (High Priority) ✓ COMPLETE

### 1.1 Fix Sortino denominator ✓

**File**: `app/backtesting/metrics/summary_metrics.py`
**Problem**: downside variance is divided by `len(negative_returns)` (via `safe_average`)
instead of `N` (all trades). This inflates Sortino when loss rate is low.

```python
# Current (wrong)
downside_variance = safe_average([r ** 2 for r in negative_returns])

# Fix
n = len(self.returns)
downside_variance = sum(r ** 2 for r in negative_returns) / n
```

### 1.2 Fix `ddof` in Sharpe (and Sortino after 1.1) ✓

**File**: `summary_metrics.py`
**Problem**: `np.std(self.returns, ddof=0)` uses population std dev. For a sample of
trades it should be `ddof=1`.

```python
std_dev = np.std(self.returns, ddof=1)
```

### 1.3 Fix `average_trade_return` computation ✓

**File**: `summary_metrics.py`
**Problem**: `safe_average([self.total_return_contract], self.total_trades)` passes a
single-element list with an explicit count — correct numerically but misleading.

```python
# Fix: use the already-cached returns list directly
average_trade_return_percentage_of_contract = safe_average(self.returns)
```

### 1.4 Add `duration_bars` per trade and `average_trade_duration_bars` in summary ✓

**File**: `app/backtesting/testing/runner.py` (compute per trade),
`summary_metrics.py` (aggregate)

In `run_single_test`, after calculating `trades_with_metrics_list`:

```python
interval_hours = INTERVAL_HOURS.get(interval, 1.0)
for trade in trades_with_metrics_list:
    trade['duration_bars'] = round(trade['duration_hours'] / interval_hours, 2)
```

In `SummaryMetrics.calculate_all_metrics` and `_initialize_calculations`:

```python
self.duration_bars = [t.get('duration_bars', 0) for t in self.trades]
average_trade_duration_bars = safe_average(self.duration_bars)
```

Add `average_trade_duration_bars` to the returned dict and to the columns list in
`reporting.py`.

**Keep `average_trade_duration_hours`** in the output too — it is still useful for
absolute time analysis. Just stop relying on it for cross-interval comparisons.

### 1.5 Raise `MIN_RETURNS_FOR_VAR` ✓

**File**: `summary_metrics.py`
**Problem**: `MIN_RETURNS_FOR_VAR = 5` means VaR and ES are computed from 5 trades,
which gives the worst single trade as VaR, not a 95th-percentile estimate.

```python
MIN_RETURNS_FOR_VAR = 30  # Minimum for statistically meaningful 5th-percentile estimate
```

---

## Phase 2 — Normalization & New Metrics (Medium Priority)

### 2.1 Fix Calmar ratio

**File**: `summary_metrics.py`
**Problem**: Current implementation divides *cumulative* total return by max drawdown.
Both grow with trade count/dataset length — not comparable across intervals or lengths.
**Fix**: Divide *annualised* return by max drawdown.

`SummaryMetrics` needs `dataset_total_hours` (passed from runner):

```python
# In __init__
def __init__(self, trades, dataset_total_hours=None):
    ...
    self.dataset_total_hours = dataset_total_hours


# In _calculate_calmar_ratio
def _calculate_calmar_ratio(self):
    if not self._has_trades() or self.maximum_drawdown_percentage == 0:
        return INFINITY_REPLACEMENT if self.maximum_drawdown_percentage == 0 else 0

    if self.dataset_total_hours and self.dataset_total_hours > 0:
        annualisation_factor = 8760 / self.dataset_total_hours  # hours in a year
        annualised_return = self.total_return_contract * annualisation_factor
    else:
        # Fallback: label as non-annualised so it is not misleadingly named
        annualised_return = self.total_return_contract

    return safe_divide(annualised_return, self.maximum_drawdown_percentage)
```

### 2.2 Surface `return_percentage_of_margin` in summary output

**File**: `summary_metrics.py`, `reporting.py`
**Problem**: `return_percentage_of_margin` is computed per trade but never aggregated.

Add to `_initialize_calculations`:

```python
self.margin_returns = [t.get('return_percentage_of_margin', 0) for t in self.trades]
```

Add to `calculate_all_metrics` return dict:

```python
'total_return_percentage_of_margin': round(sum(self.margin_returns), 2),
'average_trade_return_percentage_of_margin': round(safe_average(self.margin_returns), 2),
```

Add both columns to `reporting.py` columns list and `strategy_analyzer.py` aggregation.

### 2.3 Add `win_loss_ratio`

**Formula**: `avg_win / abs(avg_loss)`. Complements win_rate — a high win rate with a
poor W/L ratio signals a marginal strategy.

```python
def _calculate_win_loss_ratio(self):
    avg_win = self._calculate_average_win_percentage_of_contract()
    avg_loss = self._calculate_average_loss_percentage_of_contract()
    if avg_loss == 0:
        return INFINITY_REPLACEMENT
    return round(abs(safe_divide(avg_win, avg_loss)), 2)
```

### 2.4 Add `max_consecutive_losses` and `max_consecutive_wins`

Determines minimum survival capital and characterises strategy type (trend/mean-reversion).

```python
def _calculate_consecutive_streaks(self):
    max_wins = max_losses = cur_wins = cur_losses = 0
    for r in self.returns:
        if r > 0:
            cur_wins += 1
            cur_losses = 0
        else:
            cur_losses += 1
            cur_wins = 0
        max_wins = max(max_wins, cur_wins)
        max_losses = max(max_losses, cur_losses)
    return max_wins, max_losses
```

### 2.5 Add `expectancy_per_bar`

**Formula**: `average_trade_return / average_trade_duration_bars`
Normalises return for both trade size and time held. Cross-interval comparison is
meaningful once both components are normalised.

Requires Phase 1.4 to be complete (`duration_bars` on each trade).

### 2.6 Add `time_in_market_percentage`

**Formula**: `sum(duration_hours) / dataset_total_hours * 100`
Requires `dataset_total_hours` from the runner (same as Calmar fix in 2.1).

---

## Phase 3 — Aggregation Fixes (Medium Priority)

**File**: `app/backtesting/analysis/strategy_analyzer.py`

### 3.1 `average_trade_duration_bars` aggregation

Average `average_trade_duration_bars` (not hours) in `_aggregate_strategies`. Add it to
both the `weighted` and non-weighted branches.

### 3.2 `total_return_percentage_of_contract` in aggregation

Currently **summed** across symbols. A strategy on 4 symbols gets 4× the return of the
same strategy on 1 symbol. Options:

- **Option A (recommended)**: Keep the sum but rename to
  `total_return_all_symbols_pct_contract` so intent is explicit. Add a separate
  `avg_return_per_symbol_pct_contract = total / symbol_count` which is symbol-count
  neutral.
- **Option B**: Replace sum with mean. Simpler but loses the "total capital deployed"
  perspective.

### 3.3 Sharpe/Sortino/Calmar aggregation

Current approach: trade-weighted average of per-run ratios. This is an approximation.
Ideal: recalculate from the combined trade stream. This is architecturally complex today
because the raw trade lists are not stored in the parquet output — only summary metrics
are.

**Pragmatic fix for now**: document the limitation in comments and keep trade-weighted
averaging, but exclude any combinations below `MIN_TRADES_FOR_RATIO` (e.g., 30) from
the weighted average to reduce noise.

**Future option**: Store per-trade records in a separate parquet (or JSON sidecar) so
ratios can be recomputed from scratch. Out of scope for this branch.

---

## Phase 4 — Instrument Abstraction for Crypto (Architecture)

### Motivation

Binance crypto perpetuals differ from CME futures in:

| Aspect              | Futures (CME)                       | Crypto Perp (Binance)                 |
|---------------------|-------------------------------------|---------------------------------------|
| Commission          | Flat $4/trade                       | % of notional (e.g., 0.04% taker fee) |
| Margin              | Exchange-set, asset-class ratio     | Leverage-based (e.g., 10×)            |
| Contract multiplier | Defined in `futures_config`         | Defined in `crypto_config` (new)      |
| Rollover            | Front-month switch required         | Perpetuals, no rollover               |
| Trading hours       | Session-based (affects time-in-mkt) | 24/7                                  |
| Delivery            | Physical or cash, delivery risk     | Cash-settled (funding rate instead)   |

The goal is to keep `SummaryMetrics` 100% instrument-agnostic (it only operates on the
per-trade dict), and push all instrument-specific logic into `per_trade_metrics.py`.

### 4.1 Add `InstrumentType` enum

**File**: `app/backtesting/metrics/per_trade_metrics.py` (or a new
`app/backtesting/metrics/instrument.py`)

```python
from enum import Enum


class InstrumentType(Enum):
    FUTURES = 'futures'
    CRYPTO_PERP = 'crypto_perp'
```

### 4.2 Abstract commission calculation

```python
FUTURES_COMMISSION_FLAT = 4.0  # USD per trade
CRYPTO_TAKER_FEE_RATE = 0.0004  # 0.04% of notional (Binance default)


def _calculate_commission(instrument_type, notional_value):
    if instrument_type == InstrumentType.FUTURES:
        return FUTURES_COMMISSION_FLAT
    elif instrument_type == InstrumentType.CRYPTO_PERP:
        return notional_value * CRYPTO_TAKER_FEE_RATE
```

### 4.3 Abstract contract specs lookup

Currently `per_trade_metrics.py` calls `get_contract_multiplier(symbol)` from
`futures_config`. For crypto, specs come from a new `crypto_config` module.

```python
def _get_contract_multiplier(symbol, instrument_type):
    if instrument_type == InstrumentType.FUTURES:
        return get_contract_multiplier(symbol)  # existing futures_config call
    elif instrument_type == InstrumentType.CRYPTO_PERP:
        return get_crypto_contract_multiplier(symbol)  # new crypto_config call
```

### 4.4 Add `crypto_config/` module (stubbed for now)

Mirror `futures_config/` structure:

```
crypto_config/
  __init__.py          # get_crypto_contract_multiplier(symbol)
  contracts.py         # BTC: multiplier=1, ETH: multiplier=1 (USDT-margined perps)
  symbol_groups.py     # Correlated groups (BTC/MBTC, ETH/METH)
  margin_ratios.py     # Per-symbol leverage defaults or fixed rates
```

For Binance USDT-margined perps, the contract notional is typically
`price × contract_size`. For BTC/USDT perp, 1 contract = 0.001 BTC (on some configs) or
the notional may be denominated directly in USDT. This needs to be confirmed from Binance
API specs when implementing.

### 4.5 Update `calculate_trade_metrics` signature

```python
def calculate_trade_metrics(trade, symbol, instrument_type=InstrumentType.FUTURES):
```

All existing callers continue to work unchanged (default = FUTURES). Future Binance
runner passes `instrument_type=InstrumentType.CRYPTO_PERP`.

### 4.6 Metrics that stay futures-only vs universal

| Metric                          | Futures | Crypto Perp | Notes                                    |
|---------------------------------|---------|-------------|------------------------------------------|
| `margin_requirement`            | Yes     | Yes*        | *Crypto margin = notional / leverage     |
| `return_percentage_of_margin`   | Yes     | Yes         | Universal once margin is abstracted      |
| `return_percentage_of_contract` | Yes     | Yes         | Universal                                |
| `duration_bars`                 | Yes     | Yes         | Universal                                |
| `time_in_market_percentage`     | Yes     | Yes         | 24/7 for crypto (interpret differently)  |
| `commission`                    | Flat $4 | % notional  | Handled by abstraction                   |
| Rollover handling               | Yes     | No          | Perps have no rollover                   |
| Funding rate cost               | No      | Future      | Out of scope; document as known omission |

---

## Implementation Order

```
Phase 1 (bugs, no architecture change): ✓ COMPLETE — 2540 tests passing
  1.3 → fix average_trade_return computation          ✓
  1.2 → fix ddof in Sharpe                           ✓
  1.1 → fix Sortino denominator                      ✓
  1.5 → raise MIN_RETURNS_FOR_VAR to 30              ✓
  1.4 → add duration_bars (runner + summary_metrics) ✓

Phase 2 (new metrics, needs Phase 1.4):
  2.1 → Calmar annualisation                        [30 min, needs dataset_total_hours]
  2.2 → surface margin returns in summary            [30 min]
  2.3 → win_loss_ratio                               [15 min]
  2.4 → consecutive streaks                          [20 min]
  2.5 → expectancy_per_bar                           [15 min, needs 1.4]
  2.6 → time_in_market_percentage                    [15 min, needs 2.1 infrastructure]

Phase 3 (aggregation):
  3.1 → aggregate duration_bars                      [15 min]
  3.2 → rename/fix total_return aggregation          [20 min]
  3.3 → document Sharpe/Sortino/Calmar limitation    [10 min, comment only]

Phase 4 (crypto abstraction, separate branch):
  4.1-4.3 → InstrumentType + abstracted helpers      [2h]
  4.4     → crypto_config stub                       [1h]
  4.5     → update calculate_trade_metrics signature [30 min]
```

---

## Files Touched

| File                                            | Changes                                                              |
|-------------------------------------------------|----------------------------------------------------------------------|
| `app/backtesting/metrics/per_trade_metrics.py`  | Phase 4: InstrumentType, abstracted commission/multiplier            |
| `app/backtesting/metrics/summary_metrics.py`    | Phase 1+2: all formula fixes, new metrics, dataset_total_hours param |
| `app/backtesting/testing/runner.py`             | Phase 1.4: compute duration_bars, pass dataset_total_hours           |
| `app/backtesting/testing/reporting.py`          | New columns for duration_bars, margin returns, new metrics           |
| `app/backtesting/analysis/strategy_analyzer.py` | Phase 3: aggregation fixes                                           |
| `crypto_config/` (new)                          | Phase 4: contract specs for Binance perps                            |
| `tests/backtesting/metrics/`                    | Updated tests for all formula changes                                |

---

## Known Non-Goals (Out of Scope)

- Funding rate modelling for crypto perps
- Recomputing Sharpe/Sortino from combined trade stream in aggregation (requires storing
  raw trade records in parquet — large schema change)
- Time-series-based Sharpe (requires a daily P&L series, not per-trade returns)
- Dynamic margin requirements (real SPAN/TIMS margin for futures backtesting)
