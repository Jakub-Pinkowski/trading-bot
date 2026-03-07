# Crypto Backtesting Readiness Assessment

**Summary: ~85% ready.** All indicator/strategy logic is generic. The gaps are in contract specs config, P&L margin
logic, and data ingestion symbol mapping.

---

## What Works Out of the Box

| Component                             | Status  | Notes                                                                                      |
|---------------------------------------|---------|--------------------------------------------------------------------------------------------|
| OHLCV data format                     | ✅ Ready | Expects `open`, `high`, `low`, `close` with DatetimeIndex — identical to crypto            |
| File path structure                   | ✅ Ready | `data/historical_data/{month}/{symbol}/{symbol}_{interval}.parquet` — works for any symbol |
| All indicators (RSI, EMA, MACD, etc.) | ✅ Ready | Pure OHLC math, no futures coupling                                                        |
| All strategies                        | ✅ Ready | No instrument-type assumptions                                                             |
| Parallel execution framework          | ✅ Ready | `MassTester` + `ProcessPoolExecutor` is instrument-agnostic                                |
| Summary metrics aggregation           | ✅ Ready | Uses `return_percentage_of_contract`, not futures-specific                                 |
| Contract switch handling              | ✅ Ready | Already disabled via `rollover=False` in `mass_backtest.py`                                |

---

## What's Broken or Missing

### 1. Symbol Specs — REQUIRED (`futures_config/symbol_specs.py`)

Every symbol must exist in `SYMBOL_SPECS`. If it doesn't, the system crashes with `ValueError`.

The dict drives three things:

- `multiplier` → used in P&L: `gross_pnl = pnl_points * multiplier`
- `tick_size` → used in slippage calculation in `position_manager.py`
- `margin` → used in return-on-margin metric

**Fix:** Add crypto symbols. For spot crypto, use:

```python
'BTC': {
    'category': 'Crypto',
    'exchange': 'BINANCE',  # or whichever exchange the data comes from
    'multiplier': 1.0,  # 1 coin per contract
    'tick_size': 0.01,  # USD quoting precision
    'margin': 0.0  # No margin in spot (see P&L note below)
}
```

Note: CME crypto futures (`BTC`, `ETH`, `MBT`, `MET`) already exist in `SYMBOL_SPECS` — those are ready.

---

### 2. Margin in P&L Metrics — REQUIRED (`app/backtesting/metrics/per_trade_metrics.py`)

Current logic (`_estimate_margin`) estimates margin as a ratio of contract value based on category. With `margin=0.0`
for spot, `return_percentage_of_margin` will be `inf` or crash.

**Two options:**

**Option A — Use contract value as margin (preferred for spot)**

```python
def _estimate_margin(symbol, entry_price, contract_multiplier):
    from futures_config.helpers import get_margin_requirement
    stored_margin = get_margin_requirement(symbol)
    if stored_margin == 0.0:
        # Spot: margin = full contract value
        return entry_price * contract_multiplier
    return stored_margin
```

**Option B — Store approximate margin in SYMBOL_SPECS**
Set `margin` to something meaningful (e.g., 10% of contract value at a representative price). Less clean but no code
change needed.

---

### 3. Data Ingestion / Symbol Mapping — REQUIRED (if fetching from TradingView)

`fetch_data.py` uses `tvDatafeed` with TradingView symbol notation. Spot crypto symbols differ:

- Futures: `'ES1!'`, `'ZC1!'`
- Spot crypto: `'BTCUSD'`, `'ETHUSD'`, `'BINANCE:BTCUSDT'`

**If you're providing candle data directly (your scenario):** This is a non-issue — just drop properly formatted
`.parquet` files into `data/historical_data/{month}/{symbol}/`.

---

### 4. Commission Model — LOW PRIORITY (`app/backtesting/metrics/per_trade_metrics.py`)

Fixed at `COMMISSION_PER_TRADE = 4.0` USD. This is fine as a starting approximation for crypto but doesn't reflect
percentage-based exchange fees (e.g., Binance's 0.1% taker fee on a $100k BTC trade = $100, not $4).

**If accuracy matters:** Make commission configurable per symbol or as a percentage of contract value.

---

## Files to Change

| File                                           | Priority | Change                                                   |
|------------------------------------------------|----------|----------------------------------------------------------|
| `futures_config/symbol_specs.py`               | Critical | Add crypto spot symbols                                  |
| `app/backtesting/metrics/per_trade_metrics.py` | Critical | Handle `margin=0` for spot (Option A above)              |
| `mass_backtest.py`                             | Required | Update `SYMBOLS_TO_TEST` to crypto symbols               |
| `app/backtesting/metrics/per_trade_metrics.py` | Optional | Percentage-based commission                              |
| `futures_config/symbol_groups.py`              | Optional | Add crypto groups (used in `analyze_strategies.py` only) |

**No changes needed:** All strategy files, all indicator files, all data loading/validation, orchestrator, parallel
executor, summary metrics.

---

## Minimal Steps to Get Running

1. Drop candle `.parquet` files into `data/historical_data/1!/{SYMBOL}/{SYMBOL}_{interval}.parquet`
2. Add the symbol to `futures_config/symbol_specs.py` with `multiplier=1.0`, appropriate `tick_size`, `margin=0.0`
3. Fix `_estimate_margin()` in `per_trade_metrics.py` to fall back to contract value when `margin=0`
4. Set `SYMBOLS_TO_TEST = ['BTC', ...]` and `rollovers=[False]` in `mass_backtest.py`
5. Run `python mass_backtest.py`

---

## Key Risk: Margin-Based Return Metrics

The system's primary performance metric is `return_percentage_of_margin`. For futures this makes sense (you control a
large notional with small margin). For spot crypto, if you set margin = contract value, this metric becomes equivalent
to `return_percentage_of_contract` — which is fine, but the strategy rankings won't be comparable across futures and
spot results without awareness of this difference.