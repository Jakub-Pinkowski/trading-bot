# Backtesting Architecture

## Overview

The backtesting system is designed to efficiently test multiple trading strategies across different time periods,
symbols, and intervals using parallel processing and intelligent caching. The architecture prioritizes reliability,
performance, and accurate trade simulation.

## System Flow

### 1. Mass Tester Initialization

**Entry Point**: `MassTester.__init__()` in `app/backtesting/mass_testing.py`

```
Initialize MassTester
├── Load switch dates from YAML
│   └── Contract rollover dates for futures trading
├── Store test parameters
│   ├── tested_months: List of months to backtest
│   ├── symbols: List of symbols (e.g., 'ZS', 'GC', 'CL')
│   └── intervals: List of timeframes (e.g., '1h', '4h', '1d')
└── Initialize results storage
```

**Adding Strategies**:

```
MassTester.add_*_tests()
├── Create parameter grid (all combinations)
│   └── Example: rsi_periods × lower_thresholds × upper_thresholds
├── Generate strategy instances
│   └── Use strategy_factory.create_strategy()
└── Store in self.strategies list
```

### 2. Test Execution Flow

**Main Execution**: `MassTester.run_tests()`

```
Phase 1: Preparation
├── Load existing results from parquet
│   └── Create set of (month, symbol, interval, strategy) tuples for O(1) lookup
├── Generate all test combinations
│   └── Cartesian product of: months × symbols × intervals × strategies
├── Preprocess switch dates
│   └── Convert to pandas datetime for each symbol
├── Cache file paths
│   └── Build filepath patterns for each (month, symbol, interval)
└── Filter already-run tests (if skip_existing=True)

Phase 2: Parallel Execution
├── Create ProcessPoolExecutor with max_workers
├── Submit all test combinations to worker pool
│   └── Each worker runs _run_single_test()
├── Monitor progress
│   ├── Print progress every 100 tests
│   ├── Save intermediate results every 1000 tests
│   └── Run garbage collection periodically
└── Handle worker exceptions gracefully
    └── Log and continue (don't crash entire run)

Phase 3: Results Aggregation
├── Collect results from completed futures
├── Convert to DataFrame
│   ├── Validate metrics types
│   └── Handle missing/invalid values
├── Save to parquet (append mode)
└── Save caches (DataFrame and Indicator)
```

### 3. Single Test Execution (Worker Process)

**Worker Function**: `MassTester._run_single_test()`

```
For Each Test Combination:
├── 1. Load DataFrame
│   ├── Check DataFrame cache first
│   │   └── get_cached_dataframe(filepath)
│   ├── If not cached, load from parquet
│   └── Validate DataFrame
│       ├── Check required columns (open, high, low, close)
│       ├── Check for excessive NaN values (>10%)
│       ├── Verify index is sorted
│       └── Check for duplicate timestamps
│
├── 2. Run Strategy
│   └── strategy_instance.run(df, switch_dates)
│       ├── 2a. Add Indicators
│       │   ├── Strategy calls indicator functions
│       │   │   └── calculate_rsi(), calculate_ema(), etc.
│       │   ├── Each indicator checks cache first
│       │   │   └── Hash input series + parameters
│       │   │   └── Return cached value if available
│       │   └── Add indicator columns to DataFrame
│       │
│       ├── 2b. Generate Signals
│       │   ├── Apply strategy logic to indicators
│       │   ├── Use helper methods:
│       │   │   ├── _detect_crossover() for line crosses
│       │   │   └── _detect_threshold_cross() for threshold breaks
│       │   └── Add 'signal' column to DataFrame
│       │       ├── 1 = Long entry signal
│       │       ├── -1 = Short entry signal
│       │       └── 0 = No action
│       │
│       └── 2c. Extract Trades
│           ├── Iterate through DataFrame row by row
│           ├── Skip first INDICATOR_WARMUP_PERIOD candles (100)
│           ├── Handle trailing stops (if enabled)
│           ├── Handle contract switches (futures rollover)
│           ├── Execute queued signals from previous bar
│           │   └── See "Signal Queuing" section below
│           └── Return list of trades
│
├── 3. Calculate Metrics
│   ├── Per-trade metrics (for each trade)
│   │   └── calculate_trade_metrics(trade, symbol)
│   │       ├── Load contract specs (multiplier, margin)
│   │       ├── Calculate P&L in points and dollars
│   │       ├── Calculate percentage returns
│   │       └── Add commission costs
│   │
│   └── Summary metrics (aggregate)
│       └── SummaryMetrics.calculate_all_metrics()
│           ├── Basic: total_trades, win_rate
│           ├── Returns: total/average returns (% of contract and margin)
│           ├── Risk: profit_factor, max_drawdown
│           └── Advanced: Sharpe, Sortino, Calmar ratios, VaR, ES
│
└── 4. Return Result
    └── Dictionary with:
        ├── month, symbol, interval, strategy
        ├── metrics (dictionary of all calculated metrics)
        ├── timestamp (ISO format)
        └── verbose_output (if verbose=True)
```

### 4. Results Aggregation and Storage

**Post-Processing**: After all workers complete

```
Results Processing:
├── Collect all worker results
├── Convert to DataFrame
│   ├── Pre-allocate arrays for efficiency
│   ├── Validate metric types (must be numeric)
│   └── Handle inf/NaN values (replace with 0)
│
└── Save to Parquet
    ├── Filename: mass_test_results_all.parquet
    ├── Use save_to_parquet() with file locking
    └── Append to existing results (unique entries)

Cache Management:
├── Save DataFrame cache to disk
│   └── Prevents reloading same data in future runs
└── Save Indicator cache to disk
    └── Prevents recalculating same indicators
```

## Key Design Decisions

### Why Queue Signals?

**Problem**: In real trading, you can't execute at the exact moment a signal is generated.

**Solution**: Queue signals for next-bar execution.

```
Bar N (Signal Generated):
├── Close price: 100
├── Strategy detects crossover
└── Signal queued: BUY

Bar N+1 (Signal Executed):
├── Open price: 101
└── Position opened at 101 (not 100)
```

**Implementation**:

- Signals are detected based on bar close data
- Signal stored in `self.queued_signal`
- Next bar, signal executed at open price
- This simulates realistic order execution delay


## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Main Process                             │
│                                                                 │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐ │
│  │ Load Existing│      │   Generate   │      │  Pre-process │ │
│  │   Results    │ ───> │ Test Combos  │ ───> │ Switch Dates │ │
│  └──────────────┘      └──────────────┘      └──────────────┘ │
│         │                                            │          │
│         └────────────────────┬───────────────────────┘          │
│                              ▼                                  │
│                  ┌───────────────────────┐                      │
│                  │ ProcessPoolExecutor   │                      │
│                  │   (max_workers CPUs)  │                      │
│                  └───────────────────────┘                      │
│                              │                                  │
└──────────────────────────────┼──────────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
    ┌──────────┐        ┌──────────┐        ┌──────────┐
    │ Worker 1 │        │ Worker 2 │        │ Worker N │
    └──────────┘        └──────────┘        └──────────┘
           │                   │                   │
           │                   │                   │
    Each Worker Executes:                         
           │                                       
           ├─> Load DataFrame (with cache)         
           │                                       
           ├─> Add Indicators (with cache)         
           │   ├─> Check indicator_cache           
           │   ├─> Calculate if not cached         
           │   └─> Store in cache                  
           │                                       
           ├─> Generate Signals                    
           │   ├─> Apply strategy logic            
           │   └─> Queue signals for next bar      
           │                                       
           ├─> Extract Trades                      
           │   ├─> Skip warm-up period (100 bars) 
           │   ├─> Execute queued signals          
           │   ├─> Handle trailing stops           
           │   └─> Handle contract switches        
           │                                       
           ├─> Calculate Metrics                   
           │   ├─> Per-trade metrics               
           │   └─> Summary metrics                 
           │                                       
           └─> Return Result                       
                   │                               
                   └───────────┬──────────┐        
                               │          │        
                               ▼          ▼        
                    ┌────────────────────────┐     
                    │   Results Collection   │     
                    │    (Main Process)      │     
                    └────────────────────────┘     
                               │                   
                               ▼                   
                    ┌────────────────────────┐     
                    │ Convert to DataFrame   │     
                    │  Validate Metrics      │     
                    │  Save to Parquet       │     
                    │  Save Caches           │     
                    └────────────────────────┘     
```

## Strategy Execution Pipeline

```
DataFrame (OHLC + Volume)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ Step 1: Add Indicators                                        │
│                                                               │
│  For each indicator function:                                 │
│    ├─> Generate cache key: (name, data_hash, params)         │
│    ├─> Check indicator_cache                                  │
│    │   └─> HIT: Return cached series (instant)               │
│    │   └─> MISS: Calculate indicator                         │
│    │           ├─> Calculate using pandas operations          │
│    │           ├─> Store in cache                             │
│    │           └─> Return calculated series                   │
│    └─> Add column to DataFrame                                │
│                                                               │
│  Result: DataFrame with new columns:                          │
│    └─> df['rsi'], df['ema_short'], df['macd'], etc.         │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ Step 2: Generate Signals                                      │
│                                                               │
│  Strategy logic applied to indicators:                        │
│                                                               │
│  Example (RSI Strategy):                                      │
│    ├─> Detect RSI crossing below lower threshold (30)        │
│    │   └─> df['signal'] = 1 (BUY)                            │
│    │                                                          │
│    └─> Detect RSI crossing above upper threshold (70)        │
│        └─> df['signal'] = -1 (SELL)                          │
│                                                               │
│  Result: DataFrame with 'signal' column:                      │
│    └─> 1 = Long entry, -1 = Short entry, 0 = No action      │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ Step 3: Extract Trades (Row-by-Row Iteration)                │
│                                                               │
│  For each row in DataFrame:                                   │
│    │                                                          │
│    ├─> Check candle count                                    │
│    │   └─> Skip if count <= INDICATOR_WARMUP_PERIOD (100)   │
│    │                                                          │
│    ├─> Handle Trailing Stop (if enabled)                     │
│    │   ├─> Update trailing_stop based on high/low            │
│    │   └─> Close position if stop triggered                  │
│    │                                                          │
│    ├─> Handle Contract Switch (futures rollover)             │
│    │   ├─> Check if current_time >= next_switch              │
│    │   ├─> Close position at prev bar's open                 │
│    │   └─> Reopen on new contract (if rollover=True)         │
│    │                                                          │
│    ├─> Execute Queued Signal (from previous bar)             │
│    │   ├─> If queued_signal == 1 and position != 1:         │
│    │   │   ├─> Close current position (if any)               │
│    │   │   └─> Open LONG at current bar's open               │
│    │   ├─> If queued_signal == -1 and position != -1:       │
│    │   │   ├─> Close current position (if any)               │
│    │   │   └─> Open SHORT at current bar's open              │
│    │   └─> Reset queued_signal to None                       │
│    │                                                          │
│    └─> Queue New Signal (for next bar)                       │
│        └─> If signal != 0: queued_signal = signal            │
│                                                               │
│  Result: List of trades                                       │
│    └─> [{entry_time, entry_price, exit_time, exit_price,    │
│           side, switch}, ...]                                 │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ Step 4: Calculate Metrics                                     │
│                                                               │
│  Per-Trade Metrics:                                           │
│    ├─> Load contract specifications                           │
│    ├─> Calculate points_gained                                │
│    ├─> Calculate dollar_return = points × multiplier         │
│    ├─> Calculate return_pct_of_contract                       │
│    ├─> Calculate return_pct_of_margin                         │
│    └─> Add commission costs                                   │
│                                                               │
│  Summary Metrics:                                             │
│    ├─> Basic: total_trades, wins, losses, win_rate           │
│    ├─> Returns: total_return, avg_return                      │
│    ├─> Risk: profit_factor, max_drawdown                      │
│    └─> Advanced: Sharpe, Sortino, Calmar, VaR, ES            │
│                                                               │
│  Result: Metrics dictionary                                   │
│    └─> {total_trades: 42, win_rate: 0.57, ...}              │
└───────────────────────────────────────────────────────────────┘
```

## Signal Queuing Deep Dive

### Why Signals Are Queued

In real-world trading, you cannot execute a trade at the exact moment a technical condition is met. There is always a
delay between:

1. Detecting the signal (e.g., at bar close)
2. Placing the order
3. Order execution (usually at next bar open)

### Implementation Timeline

```
Bar N (19:00 - 20:00):
├─> CLOSE: 100.50
├─> RSI crosses below 30 (at close)
├─> Signal detected: BUY
└─> Signal queued: self.queued_signal = 1

Bar N+1 (20:00 - 21:00):
├─> OPEN: 101.00 ← Order executes here
├─> Execute queued signal:
│   └─> Open LONG position at 101.00
└─> Reset: self.queued_signal = None
```

### Code Flow

```python
# In _extract_trades() loop:

for idx, row in df.iterrows():
    signal = row['signal']  # 0, 1, or -1
    price_open = row['open']

    # Step 1: Execute queued signal from previous bar
    if self.queued_signal is not None:
        if self.queued_signal == 1 and self.position != 1:
            # Close current position if any
            if self.position is not None:
                self._close_position(idx, price_open)
            # Open LONG position
            self._open_new_position(1, idx, price_open)
        elif self.queued_signal == -1 and self.position != -1:
            # Close current position if any
            if self.position is not None:
                self._close_position(idx, price_open)
            # Open SHORT position
            self._open_new_position(-1, idx, price_open)

        self.queued_signal = None  # Reset after execution

    # Step 2: Queue new signal for next bar
    if signal != 0:
        self.queued_signal = signal
```

### Example Scenario

```
Scenario: RSI Strategy, lower=30, upper=70

Bar 100: close=100.00, RSI=32
  └─> No signal (RSI > 30)

Bar 101: close=99.50, RSI=28 ← RSI crosses below 30
  └─> Signal detected: BUY
  └─> Queued: self.queued_signal = 1

Bar 102: open=99.80 ← Gap up from previous close
  └─> Execute queued signal
  └─> Open LONG at 99.80 (not 99.50!)
  └─> This is realistic - markets gap

Bar 110: close=105.00, RSI=72 ← RSI crosses above 70
  └─> Signal detected: SELL
  └─> Queued: self.queued_signal = -1

Bar 111: open=104.50 ← Gap down
  └─> Execute queued signal
  └─> Close LONG at 104.50
  └─> Open SHORT at 104.50
```

## File Structure

```
app/backtesting/
├── mass_testing.py                # Main orchestration
├── strategy_factory.py            # Strategy creation and validation
├── per_trade_metrics.py           # Individual trade calculations
├── summary_metrics.py             # Aggregate statistics
├── strategies/
│   ├── base_strategy.py          # Base class with trade extraction
│   ├── rsi.py                    # RSI strategy implementation
│   ├── ema_crossover.py          # EMA crossover strategy
│   ├── macd.py                   # MACD strategy
│   ├── bollinger_bands.py        # Bollinger Bands strategy
│   └── ichimoku_cloud.py         # Ichimoku strategy
├── indicators/
│   └── indicators.py             # All indicator calculations
└── cache/
    ├── cache_base.py             # Base cache class with LRU
    ├── dataframe_cache.py        # DataFrame caching
    └── indicators_cache.py       # Indicator caching
```

## Configuration Constants

```python
# In mass_testing.py
MIN_ROWS_FOR_BACKTEST = 150  # Minimum DataFrame rows

# In base_strategy.py
INDICATOR_WARMUP_PERIOD = 100  # Candles to skip

# In cache_base.py
DEFAULT_CACHE_MAX_SIZE = 1000  # Max cache items
DEFAULT_CACHE_MAX_AGE = 86400  # Cache expiration (seconds)
DEFAULT_CACHE_LOCK_TIMEOUT = 60  # File lock timeout
DEFAULT_CACHE_RETRY_ATTEMPTS = 3  # Save retry attempts

# In indicators_cache.py
MAX_SIZE = 500  # Indicator cache size
MAX_AGE = 2592000  # 30 days

# In dataframe_cache.py
MAX_SIZE = 50  # DataFrame cache size
MAX_AGE = 604800  # 7 days
```

## Error Handling

1. **Parameter Validation**: In strategy_factory
2. **DataFrame Validation**: Before strategy execution
3. **Metrics Validation**: After calculation
4. **Type Validation**: Before saving to parquet

### Logging Levels

- **ERROR**: Critical failures (missing required columns, file not found)
- **WARNING**: Data quality issues (NaN values, small datasets)
- **INFO**: Normal operation (no trades generated, test completion)

## Testing Strategy

### Unit Tests

- Individual strategy logic
- Indicator calculations
- Cache operations
- Metric calculations

### Integration Tests

- Strategy + indicators + trade extraction
- Cache persistence
- Multiprocessing with real executor

### Test Coverage Areas

- Signal generation accuracy
- Trade extraction timing
- Slippage application
- Contract switch handling
- Trailing stop logic
- Cache hit/miss scenarios
- Concurrent cache access

## Future Enhancements

### Potential Optimizations

1. **Distributed Computing**: Extend to multiple machines
2. **Database Backend**: Replace parquet with PostgreSQL
3. **Real-time Monitoring**: Web dashboard for progress
4. **Adaptive Caching**: Dynamic cache sizes based on memory
5. **Strategy Compilation**: JIT compilation for faster execution

### Scalability Considerations

- Current system scales to ~100k tests per run
- Bottleneck: Memory for result storage
- Solution: Streaming writes to database
- Target: 1M+ tests per run

## Conclusion

The backtesting architecture uses parallel processing with intelligent caching to efficiently test thousands of strategy
variants while maintaining realistic signal execution and comprehensive validation for production trading decisions.
