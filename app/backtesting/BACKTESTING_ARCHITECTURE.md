# Backtesting Architecture

## Overview

The backtesting system is designed to efficiently test multiple trading strategies across different time periods,
symbols, and intervals using parallel processing and intelligent caching. The architecture prioritizes reliability,
performance, and accurate trade simulation.

## System Flow

### 1. Mass Tester Initialization

**Entry Point**: `MassTester.__init__()` in `app/backtesting/testing/mass_tester.py`

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

---

## Multi-Process Execution Model

### Overview

The backtesting system uses Python's `ProcessPoolExecutor` to distribute work across multiple CPU cores. Each worker
process runs in a separate memory space with its own Python interpreter.

### Process Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           MAIN PROCESS                              │
│                          (PID: 12345)                               │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │  Responsibilities:                                        │     │
│  │  • Load existing results from parquet                     │     │
│  │  • Generate all test combinations                         │     │
│  │  • Create ProcessPoolExecutor                             │     │
│  │  • Submit tasks to worker pool                            │     │
│  │  • Monitor progress and save intermediate results         │     │
│  │  • Collect results from completed tasks                   │     │
│  │  • Merge worker caches (DataFrame + Indicator)            │     │
│  │  • Save final consolidated cache to disk                  │     │
│  │  • Save aggregated results to parquet                     │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                     │
│  Memory State:                                                      │
│  ├─ DataFrame Cache: Loaded from disk at startup                   │
│  ├─ Indicator Cache: Loaded from disk at startup                   │
│  ├─ Results List: Accumulates results from workers                 │
│  └─ Switch Dates: Pre-processed and passed to workers              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               │ Creates executor with max_workers
                               │
                               ▼
            ┌──────────────────────────────────────┐
            │    ProcessPoolExecutor               │
            │    (max_workers = CPU count)         │
            │                                      │
            │  Manages worker process lifecycle:    │
            │  • Spawns worker processes           │
            │  • Distributes tasks to workers      │
            │  • Collects results from workers     │
            │  • Handles worker exceptions         │
            └──────────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐      ┌───────────────┐     ┌───────────────┐
│  WORKER 1     │      │  WORKER 2     │     │  WORKER N     │
│  (PID: 12346) │      │  (PID: 12347) │ ... │  (PID: 12350) │
└───────────────┘      └───────────────┘     └───────────────┘
        │                      │                      │
        │                      │                      │
        └──────────────────────┴──────────────────────┘
                               │
                Each Worker Process:
                               │
    ┌──────────────────────────┴─────────────────────────┐
    │                                                     │
    │  Initialization (Once per worker):                  │
    │  ├─ Copy of main process memory at spawn           │
    │  ├─ Load DataFrame cache from disk                 │
    │  ├─ Load Indicator cache from disk                 │
    │  └─ Independent Python interpreter                 │
    │                                                     │
    │  Processing Loop (For each assigned test):         │
    │  ├─ Receive test parameters from main              │
    │  ├─ Load DataFrame (check cache first)             │
    │  ├─ Run strategy (indicators auto-cache)           │
    │  ├─ Calculate metrics                              │
    │  ├─ Add computed indicators to local cache         │
    │  └─ Return result to main process                  │
    │                                                     │
    │  Memory State (Isolated):                          │
    │  ├─ DataFrame Cache: Starts as copy, grows locally │
    │  ├─ Indicator Cache: Starts as copy, grows locally │
    │  └─ Results: Returned to main via IPC              │
    │                                                     │
    │  Note: Memory is not shared with main process      │
    │        Cache updates remain in worker memory       │
    │                                                     │
    └─────────────────────────────────────────────────────┘
```

### Process Lifecycle

**Phase 1: Initialization**

```
Main Process:
  ├─ 1. Load caches from disk
  │   ├─ dataframe_cache.pkl (previously cached DataFrames)
  │   └─ indicator_cache.pkl (previously cached indicators)
  │
  ├─ 2. Create ProcessPoolExecutor
  │   └─ max_workers = os.cpu_count() (default: use all CPUs)
  │
  └─ 3. Submit all test combinations as tasks
      └─ Each task = (month, symbol, interval, strategy)

Worker Processes (spawned by executor):
  ├─ 1. New Python process spawned
  │   └─ Copy-on-write: Initially shares memory with parent
  │
  ├─ 2. Import modules
  │   └─ Each worker imports app.backtesting modules
  │
  └─ 3. Initialize caches
      ├─ Load dataframe_cache from disk (gets 100 entries)
      └─ Load indicator_cache from disk (gets 500 entries)
```

**Phase 2: Parallel Execution**

```
Main Process:
  ├─ Monitor task completion
  ├─ Print progress every 100 tests
  ├─ Save intermediate results every 1000 tests
  └─ Collect results as tasks complete

Worker 1:
  ├─ Process task 1: (202401, ZS, 1h, RSI_14_30_70)
  │   ├─ Load DataFrame from cache (HIT) or disk (MISS)
  │   ├─ Calculate RSI indicator
  │   │   ├─ Check indicator_cache (MISS - first time)
  │   │   ├─ Calculate RSI
  │   │   └─ Store in local cache (501 entries now)
  │   ├─ Generate signals and extract trades
  │   ├─ Calculate metrics
  │   └─ Return result to main
  │
  ├─ Process task 2: (202401, ZS, 1h, RSI_21_30_70)
  │   ├─ Same DataFrame (cache HIT!)
  │   ├─ Calculate RSI with period=21
  │   │   ├─ Check indicator_cache (MISS)
  │   │   ├─ Calculate RSI
  │   │   └─ Store in local cache (502 entries)
  │   └─ Return result
  │
  └─ Continue processing assigned tasks...

Worker 2 (in parallel):
  ├─ Process task 3: (202401, GC, 1h, EMA_9_21)
  │   └─ Different symbol, different indicators
  │       └─ Local cache grows independently
  │
  └─ Continue processing assigned tasks...

Note: Worker cache updates are isolated
  ├─ Worker 1 cache: 600 entries (in Worker 1 memory only)
  ├─ Worker 2 cache: 450 entries (in Worker 2 memory only)
  └─ Main cache: Still 500 entries (unchanged)
```

**Phase 3: Cleanup & Aggregation**

```
Main Process (after all workers complete):
  ├─ 1. All tasks finished
  │   └─ Workers terminate
  │
  ├─ 2. Collect all results
  │   └─ Results passed via IPC (pickle serialization)
  │
  ├─ 3. Convert to DataFrame
  │   └─ Validate metrics and handle NaN/inf values
  │
  ├─ 4. Save results to parquet
  │   └─ Append to existing file (with file locking)
  │
  └─ 5. Save caches to disk
      ├─ dataframe_cache.pkl
      └─ indicator_cache.pkl
```

### Cache Updates and Process Memory

Worker process cache updates remain in worker memory space and are not persisted when workers terminate.

### Inter-Process Communication (IPC)

**How Workers Communicate with Main**:

```
Worker Process                    Main Process
     │                                 │
     │  ┌────────────────────────┐     │
     │  │  Process Test          │     │
     │  │  Calculate Metrics     │     │
     │  └────────────────────────┘     │
     │              │                  │
     │              ▼                  │
     │  ┌────────────────────────┐     │
     │  │  Serialize Result      │     │
     │  │  (pickle)              │     │
     │  └────────────────────────┘     │
     │              │                  │
     │              ▼                  │
     │  ┌────────────────────────┐     │
     │  │  Send via Pipe/Queue   │     │
     │  └────────────────────────┘     │
     │              │                  │
     ├──────────────┼──────────────────┤
     │              ▼                  │
     │                   ┌─────────────┴──────────┐
     │                   │  Receive Result        │
     │                   │  Deserialize (unpickle)│
     │                   └────────────────────────┘
     │                               │
     │                               ▼
     │                   ┌─────────────────────────┐
     │                   │  Append to Results List │
     │                   └─────────────────────────┘
```

**What Gets Serialized**:

- Test parameters (month, symbol, interval, strategy)
- Calculated metrics (dictionary)
- Verbose output (if enabled)
- Not serialized: DataFrame (too large)
- Not serialized: Cache objects (not needed)

### Process Pool Configuration

```python
# In testing/mass_tester.py

def run_tests(self, max_workers=None):
    """
    max_workers:
        - None: Use os.cpu_count() (all CPUs)
        - Integer: Specific number of workers
        - 1: Sequential (no multiprocessing)
    """
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(self._run_single_test, params): params
            for params in test_combinations
        }

        # Process as they complete (not in submission order)
        for future in concurrent.futures.as_completed(futures):
            result = future.result()  # Blocks until result available
            if result:
                self.results.append(result)
```

**Typical Performance**:

| CPU Cores | Max Workers | Observed Speedup |
|-----------|-------------|------------------|
| 4         | 4           | 3.5x             |
| 8         | 8           | 7.0x             |
| 16        | 16          | 13x              |
| 4         | 1           | 1x (sequential)  |

**Overhead**:

- Process spawn time: ~0.5-1 second per worker
- IPC serialization: ~1-10ms per result
- Context switching: Minimal

---

## Cache Coordination Between Processes

### Cache Architecture

The system uses **two independent caches**:

1. **DataFrame Cache** - Stores loaded DataFrames to avoid re-parsing parquet files
2. **Indicator Cache** - Stores calculated indicators to avoid redundant calculations

Both use the same base architecture but serve different purposes.

### Cache Hierarchy

```
                    ┌─────────────────────────────┐
                    │   Disk Storage (Persistent) │
                    │                             │
                    │  dataframe_cache.pkl        │
                    │  indicator_cache.pkl        │
                    └─────────────────────────────┘
                                 │
                    Load at       │       Save at
                    startup       │       shutdown
                                 │
                    ┌─────────────▼─────────────┐
                    │  Main Process Memory      │
                    │                           │
                    │  dataframe_cache (50)     │
                    │  indicator_cache (500)    │
                    └─────────────┬─────────────┘
                                 │
                    Copy at       │       No sync
                    worker spawn  │       (isolated)
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐        ┌───────────────┐      ┌───────────────┐
│  Worker 1     │        │  Worker 2     │      │  Worker N     │
│               │        │               │      │               │
│  df_cache(50) │        │  df_cache(50) │      │  df_cache(50) │
│  ind_cache    │        │  ind_cache    │      │  ind_cache    │
│  (500)        │        │  (500)        │      │  (500)        │
│               │        │               │      │               │
│  Grows        │        │  Grows        │      │  Grows        │
│  independently│        │  independently│      │  independently│
│  to 600       │        │  to 450       │      │  to 520       │
└───────────────┘        └───────────────┘      └───────────────┘
        │                        │                        │
        │                        │                        │
        └────────────────────────┴────────────────────────┘
                                 │
                     When workers terminate:
                     Cache updates remain in worker memory
                     and are not saved back to disk
```

### Cache Implementation: DataFrame Cache

**Purpose**: Store parsed DataFrames to avoid re-reading parquet files.

```python
# In cache/dataframe_cache.py

from app.backtesting.cache.cache_base import Cache

# Singleton instance
dataframe_cache = Cache(
    cache_name="dataframes",
    max_size=50,  # Store up to 50 DataFrames
    max_age=604800  # 7 days TTL
)


# Usage in _run_single_test():
def _run_single_test(self, test_params):
    filepath = self.cache_file_paths[(symbol, interval)]

    # Try to get from cache
    df = dataframe_cache.get(filepath)

    if df is None:
        # Cache MISS - load from disk
        df = pd.read_parquet(filepath)
        # Store in cache for future use
        dataframe_cache.set(filepath, df)
    else:
        # Cache HIT - return immediately (no disk I/O)
        pass

    return df
```

**Cache Key**: File path (e.g., `/data/backtesting/202401_ZS_1h.parquet`)

**Behavior**:

- Multiple strategies test same (month, symbol, interval) combination
- DataFrame is identical for all strategies on that combination
- DataFrame loaded once per worker, reused for multiple tests

**Example**:

```
Worker 1 processes:
  ├─ Test 1: (202401, ZS, 1h, RSI_14_30_70)
  │   └─ Load ZS_1h DataFrame (cache MISS) → Store in cache
  │
  ├─ Test 2: (202401, ZS, 1h, RSI_21_30_70)
  │   └─ Load ZS_1h DataFrame (cache HIT) → Return from memory
  │
  └─ Test 3: (202401, ZS, 1h, EMA_9_21)
      └─ Load ZS_1h DataFrame (cache HIT) → Return from memory

Timing:
  ├─ Disk read: ~50ms
  └─ Cache read: ~0.1ms
```

### Cache Implementation: Indicator Cache

**Purpose**: Store calculated indicators to avoid redundant computations.

```python
# In cache/indicators_cache.py

from app.backtesting.cache.cache_base import Cache

# Singleton instance
indicator_cache = Cache(
    cache_name="indicators",
    max_size=500,  # Store up to 500 indicators
    max_age=2592000  # 30 days TTL
)


# Usage in indicator functions:
def calculate_rsi(prices, period=14, prices_hash=None):
    """Calculate RSI with caching."""

    # Generate cache key
    if prices_hash is None:
        prices_hash = hash_series(prices)  # SHA256 hash of price data

    cache_key = f"rsi_{prices_hash}_{period}"

    # Try to get from cache
    cached_rsi = indicator_cache.get(cache_key)

    if cached_rsi is not None:
        # Cache HIT - return immediately (no calculation)
        return cached_rsi

    # Cache MISS - calculate indicator
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Store in cache for future use
    indicator_cache.set(cache_key, rsi)

    return rsi
```

**Cache Key**: `{indicator_name}_{data_hash}_{parameters}`

Example: `rsi_a3f5d2c1_14`

**Hash Generation**:

- SHA256 hash of price data content
- Same data produces same hash
- Different data produces different hash

**Example**:

```
Worker 1 processes ZS data:
  ├─ Test 1: RSI(period=14)
  │   ├─ Hash ZS close prices: a3f5d2c1
  │   ├─ Cache key: rsi_a3f5d2c1_14
  │   ├─ Calculate RSI (cache MISS) → Store in cache
  │   └─ Time: 5ms
  │
  ├─ Test 2: RSI(period=14) on same ZS data
  │   ├─ Same hash: a3f5d2c1
  │   ├─ Cache key: rsi_a3f5d2c1_14
  │   ├─ Cache HIT → Return from memory
  │   └─ Time: 0.01ms
  │
  └─ Test 3: RSI(period=21) on same ZS data
      ├─ Same hash but different period
      ├─ Cache key: rsi_a3f5d2c1_21
      ├─ Calculate RSI (cache MISS) → Store in cache
      └─ Time: 5ms
```

### Cache Base Implementation

Both caches inherit from the same base class:

```python
# In cache/cache_base.py

class Cache:
    """
    LRU cache with file persistence and multi-process file locking.

    Features:
    • LRU eviction policy
    • File locking for concurrent access
    • TTL (time-to-live) expiration
    • Pickle serialization
    """

    def __init__(self, cache_name, max_size, max_age):
        self.cache_name = cache_name
        self.max_size = max_size
        self.max_age = max_age

        # File paths
        self.cache_file = Path(CACHE_DIR) / f"{cache_name}_cache.pkl"
        self.lock_file = Path(CACHE_DIR) / f"{cache_name}_cache.lock"

        # In-memory storage (OrderedDict for LRU)
        self.cache_data = OrderedDict()

        # Load from disk at initialization
        self._load_cache()

    def get(self, key):
        """Get value from cache (None if not found or expired)."""
        if key not in self.cache_data:
            return None

        timestamp, value = self.cache_data[key]

        # Check if expired
        if time.time() - timestamp > self.max_age:
            del self.cache_data[key]
            return None

        # Move to end (mark as recently used)
        self.cache_data.move_to_end(key)

        return value

    def set(self, key, value):
        """Add value to cache (with LRU eviction if needed)."""
        current_time = time.time()
        self.cache_data[key] = (current_time, value)
        self.cache_data.move_to_end(key)

        # Evict oldest if over size limit
        if len(self.cache_data) > self.max_size:
            self.cache_data.popitem(last=False)

    def save_cache(self):
        """Save cache to disk with file locking."""
        lock = FileLock(str(self.lock_file), timeout=60)
        with lock:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache_data, f)

    def _load_cache(self):
        """Load cache from disk with file locking."""
        if not self.cache_file.exists():
            return

        lock = FileLock(str(self.lock_file), timeout=10)
        with lock:
            with open(self.cache_file, 'rb') as f:
                self.cache_data = pickle.load(f)
```

**Implementation Details**:

1. **OrderedDict for LRU**:
    - Preserves insertion order
    - `move_to_end()` marks item as recently used
    - `popitem(last=False)` removes oldest item

2. **File Locking**:
    - Prevents corruption from concurrent writes
    - Uses `FileLock` library
    - Timeout to avoid deadlocks

3. **TTL with Lazy Expiration**:
    - Items expire after `max_age` seconds
    - Checked on `get()` operation
    - No background cleanup process

4. **Pickle Serialization**:
    - Handles complex Python objects
    - Binary format
    - Cross-version compatible

### Cache Synchronization Flow

```
Program Start:
  ├─ Main process loads caches from disk
  │   ├─ dataframe_cache.pkl → 30 entries
  │   └─ indicator_cache.pkl → 500 entries
  │
  └─ Worker processes spawn
      └─ Each worker loads same cache files
          ├─ Copy-on-write: Initially shares memory
          └─ Becomes independent when modified

During Execution:
  ├─ Main process: Cache stays static (500 entries)
  │   └─ Main doesn't compute indicators
  │
  └─ Worker processes: Cache grows independently
      ├─ Worker 1: 500 → 600 entries
      ├─ Worker 2: 500 → 450 entries
      └─ Worker N: 500 → 520 entries
      └─ ⚠️ Updates isolated to each worker's memory

Program End:
  ├─ Workers terminate
  │   └─ Worker cache updates lost ❌
  │
  └─ Main process saves cache
      └─ indicator_cache.pkl ← Still 500 entries
      └─ New calculations will be repeated next run
```

### File Locking for Multi-Process Safety

**Without Locking (Race Condition)**:
Main Process Worker Process
│ │
├─ Read cache file │
│  (500 entries)                │
│ ├─ Read cache file
│ │  (500 entries)
├─ Add entry │
│  (501 entries)                │
│ ├─ Add entry
│ │  (501 entries)
├─ Write cache file ────┐ │
│ ❌ Corrupted!         │ ├─ Write cache file ────┐
└───────────────────────┘ └─ ❌ Corrupted!         │
│
Result: File garbled (invalid pickle) 💥

```

**With Locking (Safe)**:

```

With FileLock:
Main Process Worker Process
│ │
├─ Acquire lock ✅ │
├─ Read cache file │
│  (500 entries)                │
├─ Add entry │
│  (501 entries)                ├─ Try to acquire lock ⏳
├─ Write cache file │  (blocked, waiting...)
├─ Release lock ✅ │
│ ├─ Acquire lock ✅
│ ├─ Read cache file
│ │  (501 entries) ← sees main's update
│ ├─ Add entry
│ │  (502 entries)
│ ├─ Write cache file
│ └─ Release lock ✅

       Result: File intact, both updates preserved ✅

```

**Implementation**:

```python
from filelock import FileLock

def save_cache(self):
    """Save cache with file locking."""
    try:
        lock = FileLock(str(self.lock_file), timeout=60)
        with lock:  # Blocks until lock acquired
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache_data, f)
        return True
    except Timeout:
        logger.error("Failed to acquire lock (timeout)")
        return False
```

**Lock File**:

- Separate `.lock` file for coordination
- NFS-safe (works across networked filesystems)
- Automatically released when `with` block exits
- Timeout prevents deadlocks

---

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

### Example Scenario (Real ZS Data)

```
Scenario: RSI Strategy on ZS Futures (Soybeans), lower=30, upper=70
Dataset: ZS 15-minute bars from 2025-02-04
Symbol: CBOT:ZS1! (Front Month Contract)
Price Format: Cents per bushel (e.g., 1055.75 = $10.5575/bushel)

Bar 108: 2025-02-04 11:00:00
  close=1055.25, RSI=32.4
  └─> No signal (RSI > 30)

Bar 109: 2025-02-04 11:15:00
  close=1058.25, RSI=28.7 ← RSI crosses below 30
  └─> Signal detected: BUY
  └─> Queued: self.queued_signal = 1

Bar 110: 2025-02-04 11:30:00
  open=1058.50 ← Gap up from previous close (1058.25)
  └─> Execute queued signal
  └─> Open LONG at 1058.50 (not 1058.25!)
  └─> This is realistic - markets gap between bars

Bar 118: 2025-02-04 13:00:00
  close=1057.25, RSI=71.2 ← RSI crosses above 70
  └─> Signal detected: SELL
  └─> Queued: self.queued_signal = -1

Bar 119: 2025-02-04 13:15:00
  open=1057.00 ← Gap down from previous close (1057.25)
  └─> Execute queued signal
  └─> Close LONG at 1057.00
  └─> P&L: (1057.00 - 1058.50) × 5000 bushels = -$75
  └─> Open SHORT at 1057.00

Note: RSI values are illustrative. Actual RSI calculation requires previous bars.
```

## File Structure

```
app/backtesting/
├── __init__.py                    # Main module exports
├── strategy_factory.py            # Strategy creation and validation
├── testing/
│   ├── __init__.py
│   ├── mass_tester.py            # Main orchestration
│   ├── orchestrator.py           # Test coordination
│   ├── runner.py                 # Single test runner
│   ├── reporting.py              # Result reporting
│   └── utils/
│       ├── __init__.py
│       ├── dataframe_validators.py  # DataFrame validation
│       └── test_preparation.py      # Test setup utilities
├── strategies/
│   ├── __init__.py
│   ├── base/
│   │   ├── __init__.py
│   │   ├── base_strategy.py      # Base class with trade extraction
│   │   ├── position_manager.py   # Position and slippage management
│   │   ├── trailing_stop_manager.py  # Trailing stop logic
│   │   └── contract_switch_handler.py  # Contract rollover logic
│   ├── rsi.py                    # RSI strategy implementation
│   ├── ema.py                    # EMA crossover strategy
│   ├── macd.py                   # MACD strategy
│   ├── bollinger_bands.py        # Bollinger Bands strategy
│   └── ichimoku_cloud.py         # Ichimoku strategy
├── indicators/
│   ├── __init__.py
│   ├── rsi.py                    # RSI calculation
│   ├── ema.py                    # EMA calculation
│   ├── macd.py                   # MACD calculation
│   ├── bollinger_bands.py        # Bollinger Bands calculation
│   ├── ichimoku_cloud.py         # Ichimoku Cloud calculation
│   └── atr.py                    # ATR calculation
├── metrics/
│   ├── __init__.py
│   ├── per_trade_metrics.py      # Individual trade calculations
│   └── summary_metrics.py        # Aggregate statistics
├── validators/
│   ├── __init__.py
│   ├── base.py                   # Base validator class
│   ├── common_validator.py       # Common parameter validation
│   ├── constants.py              # Validation constants
│   ├── rsi_validator.py          # RSI parameter validation
│   ├── ema_validator.py          # EMA parameter validation
│   ├── macd_validator.py         # MACD parameter validation
│   ├── bollinger_validator.py    # Bollinger parameter validation
│   └── ichimoku_validator.py     # Ichimoku parameter validation
├── analysis/
│   ├── __init__.py
│   ├── strategy_analyzer.py      # Result analysis and ranking
│   ├── constants.py              # Analysis constants
│   ├── data_helpers.py           # Data processing helpers
│   └── formatters.py             # Output formatting
├── fetching/
│   ├── __init__.py
│   ├── data_fetcher.py           # TradingView data fetching
│   └── validators.py             # Data validation
└── cache/
    ├── cache_base.py             # Base cache class with LRU
    ├── dataframe_cache.py        # DataFrame caching
    └── indicators_cache.py       # Indicator caching
```

## Configuration Constants

```python
# In testing/utils/test_preparation.py
MIN_ROWS_FOR_BACKTEST = 150  # Minimum DataFrame rows

# In strategies/base/base_strategy.py
INDICATOR_WARMUP_PERIOD = 100  # Candles to skip

# In cache/cache_base.py
DEFAULT_CACHE_MAX_SIZE = 1000  # Max cache items
DEFAULT_CACHE_MAX_AGE = 86400  # Cache expiration (seconds)
DEFAULT_CACHE_LOCK_TIMEOUT = 60  # File lock timeout
DEFAULT_CACHE_RETRY_ATTEMPTS = 3  # Save retry attempts

# In cache/indicators_cache.py
MAX_SIZE = 500  # Indicator cache size
MAX_AGE = 2592000  # 30 days

# In cache/dataframe_cache.py
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

## Conclusion

The backtesting architecture uses parallel processing with intelligent caching to efficiently test thousands of strategy
variants while maintaining realistic signal execution and comprehensive validation for production trading decisions.
