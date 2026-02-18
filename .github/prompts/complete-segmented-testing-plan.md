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

3. SEGMENTED VALIDATION (Multiple Scenarios)
   → Scenario A: 5 periods → 4 train, 1 test
   → Scenario B: 4 periods → 3 train, 1 test
   → Scenario C: 3 periods → 2 train, 1 test
   → Each appends to mass_test_results_all.parquet (same file, different segment_id)

4. FINAL SELECTION
   → Compare degradation across all scenarios
   → Pick strategies that validate consistently
   → Export for live trading
```

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Phase 1: Core Segmentation (COMPLETE)](#phase-1-core-segmentation-complete)
3. [Phase 2: Massive Testing & Filtering](#phase-2-massive-testing--filtering)
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
│    - segment_id=None for all full-period results              │
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
│  Load full-period rows: mass_test_results_all.parquet         │
│    (filter: segment_id IS NULL)                               │
│         ↓                                                       │
│  Rank by: profit_factor / sharpe_ratio / win_rate             │
│         ↓                                                       │
│  Filter: min_trades ≥ 20                                       │
│         ↓                                                       │
│  Select: Top 1% of tested strategies                          │
│         ↓                                                       │
│  Save: strategy_rankings_top_1pct.csv                         │
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
│  │  Save: mass_test_results_seg_5p_4train_1test.parquet   │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ SCENARIO B: 4 segments → 3 train, 1 test               │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │  split_all_periods(periods, segments_per_period=4)      │ │
│  │  segment_filter=[1,2,3] → Train                         │ │
│  │  segment_filter=[4]     → Test (OOS)                    │ │
│  │  Save: mass_test_results_seg_4p_3train_1test.parquet   │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ SCENARIO C: 3 segments → 2 train, 1 test               │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │  split_all_periods(periods, segments_per_period=3)      │ │
│  │  segment_filter=[1,2] → Train                           │ │
│  │  segment_filter=[3]   → Test (OOS)                      │ │
│  │  Save: mass_test_results_seg_3p_2train_1test.parquet   │ │
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
    select_top_strategies,  # Select top N or top %
    filter_by_metrics  # Multi-criteria filtering
)

# Example usage:
ranked = rank_strategies(results_df, metric='profit_factor', min_trades=20)
top_1pct = select_top_strategies(ranked, top_pct=0.01)
```

**Used by**: Phase 2 scripts (filtering top 1%)

#### `validation.py` - Scenario Execution & Validation

**Purpose**: Functions for running segmentation scenarios and calculating degradation

**Key Functions**:

```python
from app.backtesting.testing.validation import (
    prepare_segments_for_scenario,  # Create segments for a scenario
    run_scenario,  # Execute train + test phases
    calculate_scenario_degradation,  # Calculate train vs OOS degradation
    validate_strategies  # Apply pass/fail criteria
)

# Example usage:
segments = prepare_segments_for_scenario(load_func, symbols, intervals, 5)
train_results, test_results = run_scenario(scenario, strategy_adder, ...)
degradation_df = calculate_scenario_degradation(results_df, train_segs, test_segs)
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

# Example usage:
comparison = compare_scenarios(scenario_results_list)
robust = find_robust_strategies(comparison, max_degradation=30.0)
report = generate_validation_report(comparison, output_path='summary.csv')
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
- All results accumulate into one file: **`mass_test_results_all.parquet`**
- Phase 3 (segmented) results also go to the same file, distinguished by `segment_id`

> **Important**: `load_existing_results()` in `test_preparation.py` is hardcoded to
> `mass_test_results_all.parquet`. This is the single source of truth for both Phase 2
> and Phase 3 results. Do NOT use a different filename or `skip_existing` will not
> detect previously tested strategies.

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
    columns=['strategy', 'symbol', 'interval', 'segment_id']
)

# Full-period rows only (Phase 2 results)
full_period = df[df['segment_id'].isna()]

print(f"Unique strategies tested: {full_period['strategy'].nunique():,}")
print(f"Total rows (strategy × symbol × interval): {len(full_period):,}")
print(f"\nBreakdown by strategy type:")
full_period['strategy_type'] = full_period['strategy'].str.split('_').str[0]
print(full_period.groupby('strategy_type')['strategy'].nunique().to_string())
```

Run filtering (Phase 2.3) when you have enough unique strategies (suggested: 30k+
across at least 2 strategy types).

#### 2.3 Strategy Filtering & Ranking

**Script**: `filter_top_strategies.py` (NEW)

```python
"""
Step 2: Load full-period results, rank, and select top 1%.
Saves rankings to: strategy_rankings_top_1pct.csv

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

# ==================== Load Full-Period Results Only ====================

# Must filter segment_id IS NULL to exclude any segmented (Phase 3) results
# that may already be in the same file
all_df = pd.read_parquet(f'{BACKTESTING_DIR}/mass_test_results_all.parquet')
results_df = all_df[all_df['segment_id'].isna()].copy()

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
    f'{BACKTESTING_DIR}/strategy_rankings_top_1pct.csv',
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

### Implementation

#### 3.1 Segmentation Scenario Runner

**Script**: `run_segmented_validation.py` (NEW)

```python
"""
Step 3: Test top 1% strategies across multiple segmentation scenarios.

Uses validation.py module for reusable scenario execution logic.
"""

import pandas as pd
from app.backtesting.testing.validation import (
    DEFAULT_SCENARIOS,
    prepare_segments_for_scenario,
    run_scenario,
    extract_scenario_results
)
from config import DATA_DIR

BACKTESTING_DIR = DATA_DIR / "backtesting"

# ==================== Load Top Strategies ====================

top_strategies_df = pd.read_csv(
    f'{BACKTESTING_DIR}/strategy_rankings_top_1pct.csv'
)

top_strategy_names = top_strategies_df['strategy'].tolist()
print(f"Testing {len(top_strategy_names):,} top strategies")

# ==================== Run Each Scenario ====================

for scenario in DEFAULT_SCENARIOS:
    print(f"\n{'=' * 80}")
    print(f"Running Scenario: {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"{'=' * 80}")

    # Use validation.py module to run scenario
    train_results, test_results = run_scenario(
        scenario=scenario,
        strategy_adder_func=lambda t: add_top_strategies(t, top_strategy_names),
        data_loader_func=load_data,
        tested_months=TESTED_MONTHS,
        symbols=SYMBOLS,
        intervals=INTERVALS,
        max_workers=16,
        skip_existing=True,
        verbose=False
    )

    # Extract and save scenario-specific results
    all_segments = scenario['train_segments'] + scenario['test_segments']
    extract_scenario_results(scenario['name'], all_segments)

    print(f"✓ Scenario {scenario['name']} complete!")
```

**Key Insight**: All scenarios save to the **same parquet file**:

- `mass_test_results_segmented.parquet`
- Use **segment_id** to identify which scenario/phase
- Use **custom metadata column** to tag scenarios

**Better Approach**: Separate files per scenario for clarity:

```python
# At the end of each scenario loop:

# Load the main results file
all_results = pd.read_parquet(
    f'{BACKTESTING_DIR}/mass_test_results_segmented.parquet'
)

# Filter to this scenario's segments
scenario_segments = scenario['train_segments'] + scenario['test_segments']
scenario_results = all_results[
    all_results['segment_id'].isin(scenario_segments)
]

# Save to scenario-specific file
scenario_results.to_parquet(
    f'{BACKTESTING_DIR}/mass_test_results_seg_{scenario["name"]}.parquet',
    index=False
)

print(f"  Saved to: mass_test_results_seg_{scenario['name']}.parquet")
```

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
from app.backtesting.testing.validation import DEFAULT_SCENARIOS
from config import DATA_DIR

BACKTESTING_DIR = DATA_DIR / "backtesting"

# ==================== Configuration ====================

MAX_DEGRADATION_PCT = 30.0  # Maximum acceptable degradation
METRIC = 'profit_factor'

# ==================== Load Scenario Results ====================

scenario_results = []

for scenario in DEFAULT_SCENARIOS:
    df = pd.read_parquet(
        f'{BACKTESTING_DIR}/mass_test_results_seg_{scenario["name"]}.parquet'
    )
    scenario_results.append({
        'name': scenario['name'],
        'df': df,
        'train_segments': scenario['train_segments'],
        'test_segments': scenario['test_segments']
    })

# ==================== Compare Across Scenarios ====================

# Use analysis.py module
comparison_df = compare_scenarios(
    scenario_results,
    metric=METRIC
)

# Find strategies passing all scenarios
robust_strategies = find_robust_strategies(
    comparison_df,
    max_degradation_pct=MAX_DEGRADATION_PCT
)

print(f"\n{'=' * 80}")
print(f"VALIDATION SUMMARY")
print(f"{'=' * 80}")
print(f"Total strategies tested: {comparison_df['strategy'].nunique()}")
print(f"Passed all {len(DEFAULT_SCENARIOS)} scenarios: {len(robust_strategies)}")
print(f"Pass rate: {len(robust_strategies) / comparison_df['strategy'].nunique() * 100:.1f}%")

# ==================== Export Results ====================

# Generate comprehensive validation report
generate_validation_report(
    comparison_df,
    output_dir=BACKTESTING_DIR,
    max_degradation_pct=MAX_DEGRADATION_PCT
)

print(f"\n✓ Results exported:")
print(f"  - validation_summary_all_scenarios.csv (all strategies)")
print(f"  - final_robust_strategies.csv (passed all scenarios)")
```

---

## File Organization

### Data Files Structure

```
data/backtesting/
├── mass_test_results_full_periods.parquet
├── strategy_rankings_top_1pct.csv
├── mass_test_results_segmented.parquet
├── mass_test_results_seg_5p_4train_1test.parquet
├── mass_test_results_seg_4p_3train_1test.parquet
├── mass_test_results_seg_3p_2train_1test.parquet
├── validation_summary_all_scenarios.csv
└── final_robust_strategies.csv
```

### Naming Convention

**Pattern**: `mass_test_results_seg_{segments_per_period}p_{train_count}train_{test_count}test.parquet`

**Examples**:

- `mass_test_results_seg_5p_4train_1test.parquet` → 5 segments: [1,2,3,4] train, [5] test
- `mass_test_results_seg_4p_3train_1test.parquet` → 4 segments: [1,2,3] train, [4] test
- `mass_test_results_seg_3p_2train_1test.parquet` → 3 segments: [1,2] train, [3] test

**Why separate files?**

- ✅ Easy to identify which scenario
- ✅ Can analyze scenarios independently
- ✅ Simpler querying (no need to filter by segment_id every time)
- ✅ Can delete individual scenarios if re-running

**Alternative**: Single file with scenario column

```python
# Add scenario identifier
df['scenario'] = '5p_4train_1test'
```

- ❌ Larger file size
- ❌ Need to filter every query
- ✅ Single file to manage

**Recommendation**: **Separate files per scenario** for clarity and flexibility.

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

**Answer**: Segment IDs are **globally unique** within each scenario run.

```python
# Scenario A: 5 segments per period
# Period 1: segments [1, 2, 3, 4, 5]
# Period 2: segments [6, 7, 8, 9, 10]
# Period 3: segments [11, 12, 13, 14, 15]

# Scenario B: 4 segments per period
# Period 1: segments [1, 2, 3, 4]
# Period 2: segments [5, 6, 7, 8]
# Period 3: segments [9, 10, 11, 12]

# They're NOT comparable across scenarios!
# Use separate files to avoid confusion.
```

### Computational Estimates

**Phase 2: Massive Testing (1-2M strategies)**

- Strategies: 2,000,000
- Symbols: 20
- Intervals: 5
- Months: 3
- Total tests: 2M × 20 × 5 × 3 = 600M tests
- Estimated time (16 cores): 3-7 days

**Phase 3: Segmented Validation (10-20k strategies)**

- Strategies: 15,000 (top 1%)
- Scenarios: 3
- Segments per scenario: ~4 avg
- Total tests: 15k × 20 × 5 × 3 × 3 × 4 = 54M tests
- Estimated time (16 cores): 12-24 hours per scenario

**Total**: ~4-8 days for complete workflow

---

## Usage Example

### Complete End-to-End Workflow

```python
# ==================== STEP 1: Massive Testing ====================

from app.backtesting.testing import MassTester

tester = MassTester(['1!', '2!', '3!'], SYMBOLS, INTERVALS, segments=None)
tester.add_rsi_tests(...)  # Add 2M strategies
results = tester.run_tests(max_workers=16, skip_existing=True)

# Saves to: mass_test_results_full_periods.parquet

# ==================== STEP 2: Filter Top 1% ====================

import pandas as pd

df = pd.read_parquet('data/backtesting/mass_test_results_full_periods.parquet')
filtered = df[df['total_trades'] >= 20]
aggregated = filtered.groupby('strategy')['profit_factor'].mean().sort_values(ascending=False)
top_1pct = aggregated.head(int(len(aggregated) * 0.01))
top_1pct.to_csv('data/backtesting/strategy_rankings_top_1pct.csv')

# ==================== STEP 3: Run Segmentation Scenarios ====================

from app.backtesting.testing.segmentation import detect_periods, split_all_periods

scenarios = [
    {'name': '5p_4train_1test', 'segs': 5, 'train': [1, 2, 3, 4], 'test': [5]},
    {'name': '4p_3train_1test', 'segs': 4, 'train': [1, 2, 3], 'test': [4]},
    {'name': '3p_2train_1test', 'segs': 3, 'train': [1, 2], 'test': [3]}
]

for scenario in scenarios:
    # Prepare segments
    df = load_data('ZC', '5m')
    periods = detect_periods(df, '5m')
    segments = split_all_periods(periods, segments_per_period=scenario['segs'])

    # Test top strategies
    tester = MassTester(['1!'], ['ZC'], ['5m'], segments=segments)
    add_top_strategies(tester, top_1pct.index)

    # Train
    tester.run_tests(segment_filter=scenario['train'])

    # Test (OOS)
    tester.run_tests(segment_filter=scenario['test'])

    # Extract and save scenario results
    save_scenario_results(scenario['name'])

# ==================== STEP 4: Analyze & Select ====================

# Compare degradation across all scenarios
summary = []
for scenario in scenarios:
    df = pd.read_parquet(f'data/backtesting/mass_test_results_seg_{scenario["name"]}.parquet')
    train = df[df['segment_id'].isin(scenario['train'])].groupby('strategy')['profit_factor'].mean()
    test = df[df['segment_id'].isin(scenario['test'])].groupby('strategy')['profit_factor'].mean()

    for strategy in train.index:
        degradation = ((train[strategy] - test[strategy]) / train[strategy]) * 100
        summary.append({'strategy': strategy, 'scenario': scenario['name'],
                        'degradation': degradation, 'passed': degradation < 30})

summary_df = pd.DataFrame(summary)

# Find strategies that passed ALL scenarios
robust = summary_df.groupby('strategy')['passed'].sum()
robust = robust[robust == len(scenarios)].index

print(f"Robust strategies: {len(robust)}")
robust_df = summary_df[summary_df['strategy'].isin(robust)]
robust_df.to_csv('data/backtesting/final_robust_strategies.csv')
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
│           ├── selection.py                            # 🆕 NEW (Phases 2-4)
│           ├── validation.py                           # 🆕 NEW (Phases 2-4)
│           ├── analysis.py                             # 🆕 NEW (Phases 2-4)
│           │
│           ├── mass_tester.py                          # ✅ COMPLETE
│           ├── orchestrator.py                         # ✅ COMPLETE
│           ├── runner.py                               # ✅ COMPLETE
│           ├── reporting.py                            # ✅ COMPLETE
│           │
│           └── utils/
│               └── test_preparation.py                 # ✅ COMPLETE
│
├── data/
│   └── backtesting/
│       ├── mass_test_results_full_periods.parquet      # 1-2M strategies on full periods
│       ├── strategy_rankings_top_1pct.csv              # Top 1% ranked strategies
│       ├── mass_test_results_segmented.parquet         # Intermediate file (all scenarios)
│       ├── mass_test_results_seg_5p_4train_1test.parquet
│       ├── mass_test_results_seg_4p_3train_1test.parquet
│       ├── mass_test_results_seg_3p_2train_1test.parquet
│       ├── validation_summary_all_scenarios.csv
│       └── final_robust_strategies.csv
│
├── scripts/                                             # 🔲 NEW FOLDER
│   ├── phase2_massive_testing/
│   │   ├── mass_backtest_full_periods.py
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
├── phase2_massive_testing/
│   ├── mass_backtest_full_periods.py
│   └── filter_top_strategies.py
├── phase3_validation/
│   └── run_segmented_validation.py
└── phase4_analysis/
    └── analyze_validation_results.py
```

### Data Files Naming Convention

**Full Period Results**:

```
mass_test_results_full_periods.parquet
```

**Segmented Results**:

```
mass_test_results_seg_{X}p_{Y}train_{Z}test.parquet

Examples:
├── mass_test_results_seg_5p_4train_1test.parquet  # 5 segs: [1-4] train, [5] test
├── mass_test_results_seg_4p_3train_1test.parquet  # 4 segs: [1-3] train, [4] test
└── mass_test_results_seg_3p_2train_1test.parquet  # 3 segs: [1-2] train, [3] test
```

**Analysis Results**:

```
validation_summary_all_scenarios.csv
final_robust_strategies.csv
```

---

## Summary

### The Complete Plan

1. ✅ **Phase 1 Complete**: Segmentation infrastructure production-ready
2. **Phase 2**: Test 1-2M strategies on full periods → Filter to top 1%
3. **Phase 3**: Validate top 1% across 3 segmentation scenarios
4. **Phase 4**: Compare cross-scenario, select robust strategies

### Key Decisions Made

✅ **Separate parquet files per scenario** for clarity  
✅ **Naming convention**: `mass_test_results_seg_{X}p_{Y}train_{Z}test.parquet`  
✅ **Top 1% threshold**: ~10-20k strategies for segmented validation  
✅ **3 scenarios**: 5p/4train/1test, 4p/3train/1test, 3p/2train/1test  
✅ **Pass criteria**: <30% degradation in ALL scenarios

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

**New Modules in `app/backtesting/testing/`**:

- [ ] `selection.py` - Strategy filtering and ranking functions
- [ ] `validation.py` - Scenario execution and degradation calculation
- [ ] `analysis.py` - Cross-scenario comparison and reporting

**New Scripts in `scripts/`**:

- [ ] `mass_backtest_full_periods.py` - Run 1-2M strategies
- [ ] `filter_top_strategies.py` - Rank and select top 1%
- [ ] `run_segmented_validation.py` - Multi-scenario testing
- [ ] `analyze_validation_results.py` - Cross-scenario analysis

**Estimated Implementation Time**: 1-2 weeks for modules + scripts + testing

---

**Document Version**: 3.0  
**Last Updated**: February 17, 2026  
**Status**: Phase 1 Complete, Phases 2-4 Designed








