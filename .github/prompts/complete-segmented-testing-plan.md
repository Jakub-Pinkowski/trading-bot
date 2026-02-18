# Complete Segmented Testing Implementation Plan

**Last Updated**: February 18, 2026
**Status**: Phase 1 Complete (Segmentation Infrastructure) - Production Ready

---

## Executive Summary

### The Complete Workflow

```
1. INCREMENTAL BATCH TESTING (accumulate over time)
   → Run batches of 10k-20k strategies at a time
   → Run on FULL PERIODS only (no segments yet)
   → Each batch appends (with deduplication) to: mass_test_results_all.parquet
   → skip_existing=True prevents re-running completed tests
   → Repeat until satisfied with strategy space coverage

2. FILTER TO TOP 1% (once enough coverage is accumulated)
   → Rank by profit_factor/sharpe/win_rate
   → Keep only strategies with >X trades
   → Run when: enough unique strategies tested (e.g. 50k+)

3. SEGMENTED VALIDATION (Multiple Scenarios, Separate Files)
   → Scenario A: 5 periods → 4 train, 1 test
     Saves to: mass_test_results_segment_5_periods_4_train_1_test.parquet
   → Scenario B: 4 periods → 3 train, 1 test
     Saves to: mass_test_results_segment_4_periods_3_train_1_test.parquet
   → Scenario C: 3 periods → 2 train, 1 test
     Saves to: mass_test_results_segment_3_periods_2_train_1_test.parquet

4. FINAL SELECTION
   → Compare degradation across all scenarios
   → Pick strategies that validate consistently
   → Export for live trading
```

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Phase 1: Core Segmentation (COMPLETE)](#phase-1-core-segmentation-complete)
3. [Phase 2: Incremental Batch Testing & Filtering](#phase-2-incremental-batch-testing--filtering)
4. [Phase 3: Multi-Scenario Validation](#phase-3-multi-scenario-validation)
5. [File Organization](#file-organization)
6. [Implementation Details](#implementation-details)

---

## Architecture Overview

### Two-File Separation of Concerns

**Why separate `gap_detector.py` and `period_splitter.py`?**

```python
from app.backtesting.testing.segmentation import detect_periods, split_all_periods
```

**Rationale**: Two distinct problems, each useful independently:

1. **`detect_periods()`** → Solves: "Where are the data gaps?"
    - For ≥15m: Always returns 1 period (continuous)
    - For <15m: Returns 2-4 periods (gaps exist)
    - **Use alone**: Data quality checks, regime analysis

2. **`split_all_periods()`** → Solves: "How do I split for train/test?"
    - Takes periods, creates equal-row segments
    - Never crosses period boundaries
    - **Use alone**: Split continuous data for validation

**Example**:

```python
# Just periods (no segmentation)
periods = detect_periods(df, '5m')
# Analyze each period separately, no train/test split needed

# Periods + segments (train/test workflow)
periods = detect_periods(df, '5m')
segments = split_all_periods(periods, segments_per_period=4)
# Use segments 1-3 for training, segment 4 for validation
```

### Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: INCREMENTAL BATCH TESTING (run repeatedly over time)   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Historical Data (per symbol/interval)                         │
│         ↓                                                       │
│  detect_periods(df, interval)                                  │
│         ↓                                                       │
│  Periods (continuous blocks)                                   │
│    - For ≥15m: 1 period (always continuous)                   │
│    - For <15m: 2-4 periods (gaps exist)                       │
│         ↓                                                       │
│  Test ONE BATCH (10k-20k strategies) on FULL PERIODS          │
│    - No segments yet                                           │
│    - skip_existing=True skips already-tested strategies        │
│    - Low resource usage (4-6 workers on laptop)               │
│         ↓                                                       │
│  Append to: mass_test_results_all.parquet (deduplicated)      │
│    - Columns: month, symbol, interval, strategy, metrics      │
│         ↓                                                       │
│  ← Repeat with next batch until coverage is sufficient →      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: FILTER TO TOP 1% (when coverage is sufficient)        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Check progress: unique strategies tested, coverage by type   │
│         ↓                                                       │
│  Load: mass_test_results_all.parquet (pure full-period data)  │
│         ↓                                                       │
│  Rank by: profit_factor / sharpe_ratio / win_rate             │
│         ↓                                                       │
│  Filter: min_trades ≥ 20                                       │
│         ↓                                                       │
│  Select: Top 1% of tested strategies                          │
│         ↓                                                       │
│  Save: strategy_rankings_top_1_percentage.csv                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: MULTI-SCENARIO VALIDATION (10-20k Strategies)         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For each top strategy:                                        │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ SCENARIO A: 5 segments → 4 train, 1 test               │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │  split_all_periods(periods, segments_per_period=5)      │ │
│  │  segment_filter=[1,2,3,4] → Train                       │ │
│  │  segment_filter=[5]       → Test (OOS)                  │ │
│  │  Save: mass_test_results_segment_5_periods_4_train_1_test.parquet │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ SCENARIO B: 4 segments → 3 train, 1 test               │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │  split_all_periods(periods, segments_per_period=4)      │ │
│  │  segment_filter=[1,2,3] → Train                         │ │
│  │  segment_filter=[4]     → Test (OOS)                    │ │
│  │  Save: mass_test_results_segment_4_periods_3_train_1_test.parquet │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ SCENARIO C: 3 segments → 2 train, 1 test               │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │  split_all_periods(periods, segments_per_period=3)      │ │
│  │  segment_filter=[1,2] → Train                           │ │
│  │  segment_filter=[3]   → Test (OOS)                      │ │
│  │  Save: mass_test_results_segment_3_periods_2_train_1_test.parquet │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: FINAL SELECTION                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For each strategy:                                            │
│    - Compare train vs OOS in Scenario A                       │
│    - Compare train vs OOS in Scenario B                       │
│    - Compare train vs OOS in Scenario C                       │
│         ↓                                                       │
│  Calculate degradation % across all scenarios                  │
│         ↓                                                       │
│  Keep only: degradation < 30% in ALL scenarios                │
│         ↓                                                       │
│  Export: final_robust_strategies.csv                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Core Segmentation (COMPLETE) ✅

### What's Already Built

```python
from app.backtesting.testing.segmentation import detect_periods, split_all_periods
```

**Files**:

- ✅ `app/backtesting/testing/segmentation/gap_detector.py` (70 tests)
- ✅ `app/backtesting/testing/segmentation/period_splitter.py` (75 tests)
- ✅ `app/backtesting/testing/mass_tester.py` (accepts segments parameter)
- ✅ `app/backtesting/testing/orchestrator.py` (segment filtering)
- ✅ `app/backtesting/testing/runner.py` (segment slicing)
- ✅ `app/backtesting/testing/reporting.py` (segment_id/period_id columns)

**Current Capabilities**:

```python
# Detect periods
periods = detect_periods(df, interval='5m')
# Returns: 1 period for ≥15m, 2-4 periods for <15m

# Split into segments
segments = split_all_periods(periods, segments_per_period=4)

# Run MassTester with segments
tester = MassTester(['1!'], ['ZC'], ['5m'], segments=segments)
tester.add_rsi_tests(...)
results = tester.run_tests(segment_filter=[1, 2, 3])  # Train on first 3
```

**Status**: Production-ready, all 2628 tests passing ✅

---

## New Support Modules (Phases 2-4)

### Overview

Three new modules contain reusable logic for the workflow, keeping scripts thin:

```
app/backtesting/testing/
├── selection.py    # Strategy filtering and ranking
├── validation.py   # Scenario execution and degradation calculation
└── analysis.py     # Cross-scenario comparison
```

### Module Responsibilities

#### `selection.py` - Strategy Filtering & Ranking

**Purpose**: Reusable functions for filtering and ranking strategies from backtest results

**Key Functions**:

```python
from app.backtesting.testing.selection import (
    rank_strategies,  # Rank by metric (profit_factor, sharpe_ratio, etc.)
    select_top_strategies  # Select top N or top %
)

# Example usage:
ranked = rank_strategies(results_df, metric='profit_factor', min_trades=20)
top_strategies = select_top_strategies(ranked, top_percentage=0.01)
```

**Used by**: Phase 2 scripts (filtering top 1%)

#### `validation.py` - Scenario Execution & Validation

**Purpose**: Functions for running segmentation scenarios and calculating degradation

**Key Functions**:

```python
from app.backtesting.testing.validation import (
    SCENARIO_CONFIGS,  # List of 3 pre-defined scenario dicts
    prepare_segments_for_scenario,  # Load data → detect periods → split into segments
    run_scenario,  # Execute train + test phases, returns segment ID lists
    calculate_scenario_degradation,  # Calculate train vs OOS degradation per strategy
    validate_strategies  # Apply pass/fail criteria
)

# Actual signatures:
# prepare_segments_for_scenario(tested_month, reference_symbol, reference_interval,
#                               segments_per_period, train_count)
#   → (segments, train_segment_ids, test_segment_ids)

# run_scenario(scenario_config, strategy_adder_func, tested_months, symbols, intervals,
#              reference_symbol, max_workers, skip_existing, verbose)
#   → (train_segment_ids, test_segment_ids)

# Example usage:
for scenario in SCENARIO_CONFIGS:
    train_segment_ids, test_segment_ids = run_scenario(
        scenario_config=scenario,
        strategy_adder_func=lambda t: add_top_strategies(t, top_strategy_names),
        tested_months=TESTED_MONTHS,
        symbols=SYMBOLS,
        intervals=INTERVALS,
        reference_symbol='ZC',
        max_workers=4,
        skip_existing=True,
        verbose=False
    )
```

**Used by**: Phase 3 scripts (multi-scenario validation)

#### `analysis.py` - Cross-Scenario Comparison

**Purpose**: Functions for comparing results across multiple scenarios

**Key Functions**:

```python
from app.backtesting.testing.analysis import (
    compare_scenarios,  # Compare train vs OOS across scenarios
    find_robust_strategies,  # Find strategies passing all scenarios
    generate_validation_report  # Create comprehensive summary
)

# Actual signatures:
# compare_scenarios(scenario_results, metric)          ← metric is MANDATORY, no default
#   → DataFrame with columns: strategy, train_metric, test_metric,
#                             degradation_percentage, scenario_name
# find_robust_strategies(comparison_df, max_degradation_percentage=30.0) → List[str]
# generate_validation_report(comparison_df, output_dir, max_degradation_percentage=30.0)

# Example usage:
comparison_df = compare_scenarios(scenario_results_list, metric='profit_factor')
robust = find_robust_strategies(comparison_df, max_degradation_percentage=30.0)
generate_validation_report(comparison_df, output_dir=BACKTESTING_DIR)
```

**Used by**: Phase 4 scripts (final selection)

### Benefits of This Architecture

**✅ Separation of Concerns**:

- **Library code** (`app/backtesting/testing/`) = Reusable, testable functions
- **Workflow scripts** (`scripts/`) = Thin orchestration, configuration

**✅ Testability**:

- Each module can be unit tested independently
- Scripts become simple integration tests

**✅ Reusability**:

- Functions can be used in different workflows
- Easy to create custom validation scenarios

**✅ Maintainability**:

- Business logic centralized in modules
- Scripts are just configuration + orchestration

---

## Phase 2: Incremental Batch Testing & Filtering

### Goal

Accumulate full-period backtest results over time by running manageable batches
(10k-20k strategies each), then filter to top 1% once sufficient coverage is reached.

### Why Incremental?

Running 1-2M strategies in a single session would overwhelm a laptop. Instead:

- Each batch run is independent and safe to interrupt
- `skip_existing=True` ensures no strategy is ever tested twice
- All Phase 2 results accumulate into one file: **`mass_test_results_all.parquet`**

> **Important**: `mass_test_results_all.parquet` is the **pure full-period file**.
> Phase 3 segmented results go to separate per-scenario files and never touch this file.
> This keeps the file clean and means Phase 2 filtering never needs a `segment_id` filter.
>
> `load_existing_results()` in `test_preparation.py` is hardcoded to
> `mass_test_results_all.parquet`. Phase 3 requires a code change to `save_results()`
> and `load_existing_results()` to accept a configurable output path (see Phase 3 notes).

### Implementation

#### 2.1 Batch Testing Script

**Script**: `mass_backtest_full_periods.py` (NEW)

Run this script repeatedly with different parameter subsets. Each run appends new
results and skips anything already tested.

```python
"""
Step 1 (run repeatedly): Test one batch of strategies on full periods.
Appends to: data/backtesting/mass_test_results_all.parquet

Change the parameter ranges between runs to cover different strategy subsets.
skip_existing=True ensures no duplicate work.
"""

from app.backtesting.testing import MassTester

# ==================== Configuration ====================

TESTED_MONTHS = ['1!', '2!']
SYMBOLS = ['ZC', 'ZS', 'CL', 'GC', 'ES']  # Start with fewer symbols
INTERVALS = ['15m', '1h', '4h']
MAX_WORKERS = 4  # Laptop-safe: leave cores free for the OS

# --- Batch definition: change these ranges between runs ---
# Example batch 1: RSI short periods
RSI_PERIODS = range(5, 20)  # 15 values
RSI_LOWER = range(20, 40)  # 20 values
RSI_UPPER = range(60, 80)  # 20 values
TRAILING_STOPS = [None, 2]
SLIPPAGE_TICKS = [1, 2]
# Batch size: 15 × 20 × 20 × 2 × 2 = 24,000 strategies

# Example batch 2 (next run): RSI longer periods
# RSI_PERIODS = range(20, 50)
# ...

# ==================== Run Batch ====================

tester = MassTester(
    tested_months=TESTED_MONTHS,
    symbols=SYMBOLS,
    intervals=INTERVALS,
    segments=None  # Full periods only
)

tester.add_rsi_tests(
    rsi_periods=RSI_PERIODS,
    lower_thresholds=RSI_LOWER,
    upper_thresholds=RSI_UPPER,
    rollovers=[False],
    trailing_stops=TRAILING_STOPS,
    slippage_ticks_list=SLIPPAGE_TICKS
)

results = tester.run_tests(
    verbose=False,
    max_workers=MAX_WORKERS,
    skip_existing=True  # Safe to re-run: skips everything already in the parquet
)
```

**Suggested batch split by strategy type**:

| Run | Strategy        | Parameter Subset                | ~Strategies |
|-----|-----------------|---------------------------------|-------------|
| 1   | RSI             | periods 5–20, thresholds narrow | 15k         |
| 2   | RSI             | periods 20–50, thresholds wide  | 18k         |
| 3   | EMA Crossover   | short 5–15, long 20–50          | 12k         |
| 4   | EMA Crossover   | short 15–30, long 50–100        | 15k         |
| 5   | Bollinger Bands | all periods                     | 10k         |
| 6   | MACD            | all combinations                | 14k         |
| N   | ...             | ...                             | ...         |

#### 2.2 Progress Check

Before running filtering, check how many unique strategies have been accumulated.
Run this anytime to see current status:

```python
"""
Check accumulation progress before running filter_top_strategies.py.
"""

import pandas as pd
from config import DATA_DIR

BACKTESTING_DIR = DATA_DIR / "backtesting"

df = pd.read_parquet(
    f'{BACKTESTING_DIR}/mass_test_results_all.parquet',
    columns=['strategy', 'symbol', 'interval']
)

print(f"Unique strategies tested: {df['strategy'].nunique():,}")
print(f"Total rows (strategy × symbol × interval): {len(df):,}")
print(f"\nBreakdown by strategy type:")
df['strategy_type'] = df['strategy'].str.split('_').str[0]
print(df.groupby('strategy_type')['strategy'].nunique().to_string())
```

Run filtering (Phase 2.3) when you have enough unique strategies (suggested: 30k+
across at least 2 strategy types).

#### 2.3 Strategy Filtering & Ranking

**Script**: `filter_top_strategies.py` (NEW)

```python
"""
Step 2: Load full-period results, rank, and select top 1%.
Saves rankings to: strategy_rankings_top_1_percentage.csv

Uses selection.py module for reusable ranking logic.
"""

import pandas as pd
from app.backtesting.testing.selection import rank_strategies, select_top_strategies
from config import DATA_DIR

BACKTESTING_DIR = DATA_DIR / "backtesting"

# ==================== Configuration ====================

RANKING_METRIC = 'profit_factor'  # or 'sharpe_ratio', 'win_rate'
MIN_TRADES = 20
TOP_PERCENTAGE = 0.01  # Top 1%

# ==================== Load Results ====================

# mass_test_results_all.parquet contains only Phase 2 (full-period) results
results_df = pd.read_parquet(f'{BACKTESTING_DIR}/mass_test_results_all.parquet')

print(f"Total unique strategies tested: {results_df['strategy'].nunique():,}")

# ==================== Rank & Select Top 1% ====================

ranked = rank_strategies(
    results_df,
    metric=RANKING_METRIC,
    min_trades=MIN_TRADES,
    ascending=False
)

top_strategies = select_top_strategies(
    ranked,
    top_percentage=TOP_PERCENTAGE
)

print(f"\nTop {TOP_PERCENTAGE * 100:.1f}%: {len(top_strategies):,} strategies")

# ==================== Save Rankings ====================

top_strategies.to_csv(
    f'{BACKTESTING_DIR}/strategy_rankings_top_1_percentage.csv',
    index=False
)

print(f"\n✓ Top {len(top_strategies):,} strategies saved!")
print(f"Best strategy: {top_strategies.iloc[0]['strategy']}")
print(f"  {RANKING_METRIC}: {top_strategies.iloc[0][RANKING_METRIC]:.2f}")
print(f"  Total trades: {top_strategies.iloc[0]['total_trades']:.0f}")
```

---

## Phase 3: Multi-Scenario Validation

### Goal

Test top 1% (~10-20k strategies) across multiple segmentation scenarios to find consistently robust strategies.

### Why Multiple Scenarios?

Different market conditions might favor different train/test splits:

- **5 segments** (4 train, 1 test): 80/20 split, smaller OOS sample
- **4 segments** (3 train, 1 test): 75/25 split, balanced
- **3 segments** (2 train, 1 test): 67/33 split, larger OOS sample

**Robust strategies should validate across ALL scenarios.**

### Required Code Changes Before Running Phase 3

#### 1. Configurable results path (`skip_existing` correctness)

Phase 3 saves results to **separate per-scenario parquet files**, not to
`mass_test_results_all.parquet`. Without this change, `skip_existing=True` will
check the wrong file and re-run everything on every invocation (no crash, just wasted work).

- **`reporting.py`**: Add an optional `output_path` parameter to `save_results()`.
  Default stays `mass_test_results_all.parquet` so Phase 2 is unchanged.
- **`test_preparation.py`**: Add an optional `results_path` parameter to
  `load_existing_results()`. Default stays `mass_test_results_all.parquet`.
- **`orchestrator.py`**: Thread `results_path` through `run_tests()`.
- **`validation.py`**: `run_scenario()` derives the path from the scenario name
  and passes it to the orchestrator:

```python
# Derived automatically inside run_scenario():
results_path = BACKTESTING_DIR / f"mass_test_results_segment_{scenario_config['name']}.parquet"
```

> **Ordering**: implement this change before running Phase 3 with `skip_existing=True`.
> Running Phase 3 without it will work (results still save correctly to the scenario
> file) but will re-run all tests on every invocation instead of skipping done work.

#### 2. Filtering strategies by name (`add_top_strategies`)

Phase 3 tests only the **top 1% strategies**, not the full parameter space.
`MassTester.add_*_tests()` currently generates all parameter combinations — there is
no built-in mechanism to restrict to a subset of strategy names.

**Required**: Add a `filter_strategies(strategy_names)` method to `MassTester`:

```python
def filter_strategies(self, strategy_names):
    """Keep only strategies whose names are in strategy_names."""
    name_set = set(strategy_names)
    self.strategies = [(name, inst) for name, inst in self.strategies
                       if name in name_set]
```

Then `add_top_strategies` in the Phase 3 script becomes:

```python
def add_top_strategies(tester, top_strategy_names):
    # Add the same parameter space used in Phase 2
    tester.add_rsi_tests(rsi_periods=range(5, 50), ...)
    tester.add_ema_tests(...)
    # ...
    # Trim to only the top strategies
    tester.filter_strategies(top_strategy_names)
```

> **Note**: `add_*_tests` builds strategy objects for the full parameter space before
> filtering. For very large spaces (>500k combinations), generating objects upfront may
> be slow. If this becomes a bottleneck, add a `strategy_name_filter` set parameter to
> each `add_*_tests` method to filter during generation instead.

### Implementation

#### 3.1 Segmentation Scenario Runner

**Script**: `run_segmented_validation.py` (NEW)

```python
"""
Step 3: Test top 1% strategies across multiple segmentation scenarios.
Each scenario saves to its own parquet file.

Uses validation.py module for reusable scenario execution logic.

Prerequisites:
  - reporting.py / test_preparation.py / orchestrator.py must have the
    configurable results_path change implemented (for skip_existing to work)
  - MassTester must have filter_strategies() method implemented
"""

import pandas as pd
from app.backtesting.testing.validation import SCENARIO_CONFIGS, run_scenario
from config import DATA_DIR

BACKTESTING_DIR = DATA_DIR / "backtesting"

# ==================== Configuration ====================

TESTED_MONTHS = ['1!', '2!']
SYMBOLS = ['ZC', 'ZS', 'CL', 'GC', 'ES']
INTERVALS = ['15m', '1h', '4h']
REFERENCE_SYMBOL = 'ZC'  # Used to detect period structure (segment date ranges)
MAX_WORKERS = 4

# ==================== Load Top Strategies ====================

top_strategies_df = pd.read_csv(
    f'{BACKTESTING_DIR}/strategy_rankings_top_1_percentage.csv'
)
top_strategy_names = top_strategies_df['strategy'].tolist()
print(f"Testing {len(top_strategy_names):,} top strategies across {len(SCENARIO_CONFIGS)} scenarios")


# ==================== Strategy Adder ====================

def add_top_strategies(tester, strategy_names):
    """
    Add the same full parameter space used in Phase 2, then trim to
    only the top strategies. Requires MassTester.filter_strategies().
    """
    tester.add_rsi_tests(
        rsi_periods=range(5, 50),
        lower_thresholds=range(20, 45),
        upper_thresholds=range(55, 80),
        rollovers=[False],
        trailing_stops=[None, 2],
        slippage_ticks_list=[1, 2]
    )
    # Add other strategy types here to cover the full Phase 2 parameter space
    # tester.add_ema_tests(...)
    tester.filter_strategies(strategy_names)


# ==================== Run Each Scenario ====================

for scenario in SCENARIO_CONFIGS:
    print(f"\n{'=' * 80}")
    print(f"Running Scenario: {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"  Saves to: mass_test_results_segment_{scenario['name']}.parquet")
    print(f"{'=' * 80}")

    train_segment_ids, test_segment_ids = run_scenario(
        scenario_config=scenario,
        strategy_adder_func=lambda t: add_top_strategies(t, top_strategy_names),
        tested_months=TESTED_MONTHS,
        symbols=SYMBOLS,
        intervals=INTERVALS,
        reference_symbol=REFERENCE_SYMBOL,
        max_workers=MAX_WORKERS,
        skip_existing=True,
        verbose=False
    )

    print(f"✓ Scenario {scenario['name']} complete!")
    print(f"  Train segment IDs: {train_segment_ids}")
    print(f"  Test segment IDs:  {test_segment_ids}")
```

> **Key Design**: Each scenario's results are isolated in its own parquet file.
> `mass_test_results_all.parquet` stays pure — it only ever contains full-period
> Phase 2 results. Segment IDs in scenario files are independent and do not
> conflict across scenarios.

#### 3.2 Cross-Scenario Analysis

**Script**: `analyze_validation_results.py` (NEW)

```python
"""
Step 4: Compare train vs OOS across all scenarios.
Identify strategies that validate consistently.

Uses analysis.py module for cross-scenario comparison logic.
"""

import pandas as pd
from app.backtesting.testing.analysis import (
    compare_scenarios,
    find_robust_strategies,
    generate_validation_report
)
from app.backtesting.testing.validation import SCENARIO_CONFIGS
from config import DATA_DIR

BACKTESTING_DIR = DATA_DIR / "backtesting"

# ==================== Configuration ====================

MAX_DEGRADATION_PERCENTAGE = 30.0  # Maximum acceptable degradation
METRIC = 'profit_factor'

# ==================== Load Per-Scenario Results ====================

# Each scenario has its own parquet file — no segment_id filtering needed
# train_segment_ids and test_segment_ids come from the scenario config
# (re-derived here from SCENARIO_CONFIGS for analysis, not re-running)
scenario_results = []

for scenario in SCENARIO_CONFIGS:
    path = BACKTESTING_DIR / f"mass_test_results_segment_{scenario['name']}.parquet"
    scenario_df = pd.read_parquet(path)

    # Derive segment IDs from the scenario config
    # (same logic as _compute_segment_filters in validation.py)
    segments_per_period = scenario['segments_per_period']
    train_count = scenario['train_count']

    # For single-period data (≥15m): segment IDs are 1..segments_per_period
    # train = first train_count, test = remainder
    train_segment_ids = list(range(1, train_count + 1))
    test_segment_ids = list(range(train_count + 1, segments_per_period + 1))

    scenario_results.append({
        'name': scenario['name'],
        'df': scenario_df,
        'train_segments': train_segment_ids,
        'test_segments': test_segment_ids
    })

    print(f"Loaded {scenario['name']}: {len(scenario_df):,} rows")

# ==================== Compare Across Scenarios ====================

comparison_df = compare_scenarios(
    scenario_results,
    metric=METRIC
)

robust_strategies = find_robust_strategies(
    comparison_df,
    max_degradation_percentage=MAX_DEGRADATION_PERCENTAGE
)

print(f"\n{'=' * 80}")
print(f"VALIDATION SUMMARY")
print(f"{'=' * 80}")
print(f"Total strategies tested: {comparison_df['strategy'].nunique()}")
print(f"Passed all {len(SCENARIO_CONFIGS)} scenarios: {len(robust_strategies)}")
print(f"Pass rate: {len(robust_strategies) / comparison_df['strategy'].nunique() * 100:.1f}%")

# ==================== Export Results ====================

generate_validation_report(
    comparison_df,
    output_dir=BACKTESTING_DIR,
    max_degradation_percentage=MAX_DEGRADATION_PERCENTAGE
)

print(f"\n✓ Results exported:")
print(f"  - validation_summary_all_scenarios.csv (all strategies)")
print(f"  - final_robust_strategies.csv (passed all scenarios)")
```

> **Note on segment ID derivation in analysis**: The script above derives
> `train_segment_ids` and `test_segment_ids` from the scenario config for
> single-period data (≥15m intervals, always 1 period). For multi-period data,
> the IDs span periods (e.g., periods 1-3 × 5 segments = IDs 1-15). In that case,
> save `train_segment_ids` and `test_segment_ids` returned by `run_scenario()` to
> a JSON sidecar file alongside the parquet, and load them here instead of
> re-deriving.

---

## File Organization

### Data Files Structure

```
data/backtesting/
├── mass_test_results_all.parquet                              # Phase 2: full-period results ONLY
├── strategy_rankings_top_1_percentage.csv                     # Top 1% ranked strategies
├── mass_test_results_segment_5_periods_4_train_1_test.parquet # Phase 3 Scenario A
├── mass_test_results_segment_4_periods_3_train_1_test.parquet # Phase 3 Scenario B
├── mass_test_results_segment_3_periods_2_train_1_test.parquet # Phase 3 Scenario C
├── validation_summary_all_scenarios.csv
└── final_robust_strategies.csv
```

### Data File Convention

Each phase has its own file. No `segment_id` filtering is needed at read time:

| File                                                         | Contents                 | Phase   |
|--------------------------------------------------------------|--------------------------|---------|
| `mass_test_results_all.parquet`                              | Full-period results only | Phase 2 |
| `mass_test_results_segment_5_periods_4_train_1_test.parquet` | Scenario A rows          | Phase 3 |
| `mass_test_results_segment_4_periods_3_train_1_test.parquet` | Scenario B rows          | Phase 3 |
| `mass_test_results_segment_3_periods_2_train_1_test.parquet` | Scenario C rows          | Phase 3 |

> **Segment IDs within Phase 3 files**: Segment IDs are scenario-specific.
> Scenario A uses IDs 1–5 per period, Scenario B uses 1–4, Scenario C uses 1–3.
> Because each scenario has its own file, there is no ID collision.

---

## Implementation Details

### Handling Different Intervals

```python
# Example: ZC symbol across intervals

# 5m data
df_5m = load_data('ZC', '5m')
periods_5m = detect_periods(df_5m, '5m')
# Returns: 3 periods (has gaps)

# 1h data
df_1h = load_data('ZC', '1h')
periods_1h = detect_periods(df_1h, '1h')
# Returns: 1 period (continuous)

# Both can be segmented the same way
segments_5m = split_all_periods(periods_5m, segments_per_period=4)
# Creates: 12 segments (3 periods × 4 segments each)

segments_1h = split_all_periods(periods_1h, segments_per_period=4)
# Creates: 4 segments (1 period × 4 segments)

# The workflow is identical regardless of period count!
```

### Segment ID Management

**Question**: How do segment IDs work across different scenarios?

**Answer**: Segment IDs are **globally unique** within each scenario run, and
each scenario saves to its **own file**, so there is no cross-contamination.

```python
# Scenario A: 5 segments per period
# Period 1: segments [1, 2, 3, 4, 5]
# Period 2: segments [6, 7, 8, 9, 10]
# Period 3: segments [11, 12, 13, 14, 15]
# Saved to: mass_test_results_segment_5_periods_4_train_1_test.parquet

# Scenario B: 4 segments per period
# Period 1: segments [1, 2, 3, 4]
# Period 2: segments [5, 6, 7, 8]
# Period 3: segments [9, 10, 11, 12]
# Saved to: mass_test_results_segment_4_periods_3_train_1_test.parquet

# Segment ID 3 means something different in each scenario,
# but since they are in separate files there is no ambiguity.
```

### Computational Estimates

**Phase 2: Incremental Batch Testing**

- Batch size: 10k-20k strategies
- Symbols: 5-10 (start small, expand)
- Intervals: 3-5
- Months: 2-3
- Rows per batch: ~20k × 5 × 3 × 2 = 600k
- Estimated time per batch (4 workers, laptop): 20-60 minutes
- Repeat across N sessions until coverage is sufficient

**Phase 3: Segmented Validation (top 1% of tested strategies)**

- Strategies: depends on Phase 2 coverage (e.g., 300-500 if 30k-50k tested)
- Scenarios: 3 (each to its own file)
- Segments per scenario: ~4 avg
- Estimated time (4 workers, laptop): 1-3 hours per scenario

**Total**: Spread over multiple sessions at your own pace

---

## Usage Example

### Complete End-to-End Workflow

```python
# ==================== STEP 1: Incremental Batch Testing ====================

from app.backtesting.testing import MassTester

# Run once per session with a different parameter subset each time
tester = MassTester(['1!', '2!'], SYMBOLS, INTERVALS, segments=None)
tester.add_rsi_tests(
    rsi_periods=range(5, 20),  # Change this range each batch
    lower_thresholds=range(20, 40),
    upper_thresholds=range(60, 80),
    rollovers=[False],
    trailing_stops=[None, 2],
    slippage_ticks_list=[1, 2]
)
# skip_existing=True: safe to re-run, never duplicates work
results = tester.run_tests(verbose=False, max_workers=4, skip_existing=True)

# Appends to: mass_test_results_all.parquet (pure full-period data)

# ==================== STEP 2: Filter Top 1% (when ready) ====================

import pandas as pd
from app.backtesting.testing.selection import rank_strategies, select_top_strategies

# mass_test_results_all.parquet is pure Phase 2 data — load directly, no filter needed
results_df = pd.read_parquet('data/backtesting/mass_test_results_all.parquet')

ranked = rank_strategies(results_df, metric='profit_factor', min_trades=20, ascending=False)
top_strategies = select_top_strategies(ranked, top_percentage=0.01)
top_strategies.to_csv('data/backtesting/strategy_rankings_top_1_percentage.csv')

# ==================== STEP 3: Run Segmentation Scenarios ====================

from app.backtesting.testing.validation import SCENARIO_CONFIGS, run_scenario

top_strategy_names = top_strategies['strategy'].tolist()


def add_top_strategies(tester, strategy_names):
    # Add same parameter space as Phase 2, then trim to the top strategies
    tester.add_rsi_tests(rsi_periods=range(5, 50), ...)
    # tester.add_ema_tests(...)
    tester.filter_strategies(strategy_names)  # Requires MassTester.filter_strategies()


for scenario in SCENARIO_CONFIGS:
    # run_scenario saves results to a per-scenario parquet file:
    #   mass_test_results_segment_{scenario['name']}.parquet
    train_segment_ids, test_segment_ids = run_scenario(
        scenario_config=scenario,
        strategy_adder_func=lambda t: add_top_strategies(t, top_strategy_names),
        tested_months=['1!', '2!'],
        symbols=SYMBOLS,
        intervals=INTERVALS,
        reference_symbol='ZC',
        max_workers=4,
        skip_existing=True,
        verbose=False
    )

# ==================== STEP 4: Analyze & Select ====================

from app.backtesting.testing.analysis import compare_scenarios, find_robust_strategies, generate_validation_report

# Segment IDs: valid for ≥15m (1 period). For <15m multi-period data, load
# train/test IDs from a sidecar JSON saved during run_scenario instead.
scenario_results = []
for scenario in SCENARIO_CONFIGS:
    path = f'data/backtesting/mass_test_results_segment_{scenario["name"]}.parquet'
    scenario_df = pd.read_parquet(path)
    scenario_results.append({
        'name': scenario['name'],
        'df': scenario_df,
        'train_segments': list(range(1, scenario['train_count'] + 1)),
        'test_segments': list(range(scenario['train_count'] + 1, scenario['segments_per_period'] + 1))
    })

comparison_df = compare_scenarios(scenario_results, metric='profit_factor')
robust_strategies = find_robust_strategies(comparison_df, max_degradation_percentage=30.0)

print(f"Robust strategies: {len(robust_strategies)}")
generate_validation_report(comparison_df, output_dir='data/backtesting/')
```

---

## Complete Folder Structure (After All Phases)

### Project Structure

```
trading-bot/
│
├── app/
│   └── backtesting/
│       └── testing/
│           ├── segmentation/                           # ✅ COMPLETE (Phase 1)
│           │   ├── __init__.py
│           │   ├── gap_detector.py
│           │   └── period_splitter.py
│           │
│           ├── selection.py                            # ✅ COMPLETE (Phase 2)
│           ├── validation.py                           # ✅ COMPLETE (Phase 3)
│           ├── analysis.py                             # ✅ COMPLETE (Phase 4)
│           │
│           ├── mass_tester.py                          # ✅ COMPLETE (needs filter_strategies() for Phase 3)
│           ├── orchestrator.py                         # ✅ COMPLETE (needs results_path param for Phase 3)
│           ├── runner.py                               # ✅ COMPLETE
│           ├── reporting.py                            # ✅ COMPLETE (needs output_path param for Phase 3)
│           │
│           └── utils/
│               └── test_preparation.py                 # ✅ COMPLETE (needs results_path param for Phase 3)
│
├── data/
│   └── backtesting/
│       ├── mass_test_results_all.parquet               # Phase 2 ONLY: pure full-period results
│       ├── mass_test_results_segment_5_periods_4_train_1_test.parquet  # Phase 3 Scenario A
│       ├── mass_test_results_segment_4_periods_3_train_1_test.parquet  # Phase 3 Scenario B
│       ├── mass_test_results_segment_3_periods_2_train_1_test.parquet  # Phase 3 Scenario C
│       ├── strategy_rankings_top_1_percentage.csv      # Top 1% ranked strategies
│       ├── validation_summary_all_scenarios.csv
│       └── final_robust_strategies.csv
│
├── scripts/                                             # 🔲 NEW FOLDER
│   ├── phase2_batch_testing/
│   │   ├── mass_backtest_full_periods.py               # Run repeatedly per batch
│   │   └── filter_top_strategies.py
│   │
│   ├── phase3_validation/
│   │   └── run_segmented_validation.py
│   │
│   └── phase4_analysis/
│       └── analyze_validation_results.py
│
├── tests/
│   └── backtesting/
│       └── testing/
│           └── segmentation/                            # ✅ 145 tests passing
│               ├── test_gap_detector.py                 # 70 tests
│               ├── test_period_splitter.py              # 75 tests
│               ├── test_integration.py
│               └── test_scenarios.py
│
└── .github/
    └── prompts/                                         # ✅ Documentation
        ├── complete-segmented-testing-plan.md           # This document
        ├── segmentation_architecture_rationale.md       # Why separate files
        ├── segment_fixes_summary.md                     # Recent improvements
        └── segmentation_review_and_improvements.md      # Code review notes
```

### Scripts Overview

```
scripts/
├── phase2_batch_testing/
│   ├── mass_backtest_full_periods.py   # Run repeatedly per batch
│   └── filter_top_strategies.py        # Run once, when coverage is sufficient
├── phase3_validation/
│   └── run_segmented_validation.py
└── phase4_analysis/
    └── analyze_validation_results.py
```

### Data Files

Phase 2 and Phase 3 results live in **separate files**:

```
mass_test_results_all.parquet
  └── Full-period results only (Phase 2, accumulates over time)

mass_test_results_segment_5_periods_4_train_1_test.parquet
  └── Scenario A: 4 train segments + 1 test segment per period

mass_test_results_segment_4_periods_3_train_1_test.parquet
  └── Scenario B: 3 train segments + 1 test segment per period

mass_test_results_segment_3_periods_2_train_1_test.parquet
  └── Scenario C: 2 train segments + 1 test segment per period
```

**Analysis outputs**:

```
strategy_rankings_top_1_percentage.csv
validation_summary_all_scenarios.csv
final_robust_strategies.csv
```

---

## Summary

### The Complete Plan

1. ✅ **Phase 1 Complete**: Segmentation infrastructure production-ready
2. **Phase 2**: Accumulate full-period results via repeated batch runs → Filter to top 1% when ready
3. **Phase 3**: Validate top 1% across 3 segmentation scenarios (each to its own file)
4. **Phase 4**: Compare cross-scenario, select robust strategies

### Key Decisions Made

✅ **Separate parquet files**: `mass_test_results_all.parquet` (Phase 2 only) + per-scenario files (Phase 3)
✅ **No segment_id filter needed** when reading Phase 2 or Phase 3 files — each file is pure
✅ **Incremental batches** of 10k-20k strategies, 4 workers, safe to run on a laptop
✅ **Top 1% of tested strategies** for segmented validation (not a fixed count)
✅ **3 scenarios**: 5p/4train/1test, 4p/3train/1test, 3p/2train/1test
✅ **Pass criteria**: <30% degradation in ALL scenarios
✅ **Configurable `results_path`**: `save_results()`, `load_existing_results()`, `orchestrator.run_tests()`,
`MassTester.run_tests()` all accept an optional path; `run_scenario()` derives it from the scenario name automatically
✅ **`filter_strategies()`**: `MassTester` has a `filter_strategies(strategy_names)` method to trim to top 1% before
Phase 3 runs

### What's Ready to Use Now

```python
# Already working:
from app.backtesting.testing.segmentation import detect_periods, split_all_periods

periods = detect_periods(df, '5m')
segments = split_all_periods(periods, segments_per_period=4)

tester = MassTester(['1!'], ['ZC'], ['5m'], segments=segments)
results = tester.run_tests(segment_filter=[1, 2, 3])
```

### What Needs to Be Built

**Code Changes in `app/backtesting/testing/`**:

- [x] `mass_tester.py` - Add `filter_strategies(strategy_names)` method (trim strategy list to a name set) ✅
- [x] `reporting.py` - Add optional `output_path` param to `save_results()` (default: current hardcoded path) ✅
- [x] `test_preparation.py` - Add optional `results_path` param to `load_existing_results()` ✅
- [x] `orchestrator.py` - Thread `results_path` through `run_tests()` ✅
- [x] `validation.py` - In `run_scenario()`, derive and pass the per-scenario `results_path` ✅

**New Modules in `app/backtesting/testing/`**:

- [x] `selection.py` - Strategy filtering and ranking functions ✅
- [x] `validation.py` - Scenario execution and degradation calculation ✅
- [x] `analysis.py` - Cross-scenario comparison and reporting ✅

**New Scripts in `scripts/`**:

- [ ] `mass_backtest_full_periods.py` - Batch runner (run repeatedly per subset)
- [ ] `filter_top_strategies.py` - Rank and select top 1% (run when ready)
- [ ] `run_segmented_validation.py` - Multi-scenario testing
- [ ] `analyze_validation_results.py` - Cross-scenario analysis

---

**Document Version**: 5.1
**Last Updated**: February 18, 2026
**Status**: Phase 1 Complete, Phases 2-4 Designed (Incremental Batch, Separate Files)
