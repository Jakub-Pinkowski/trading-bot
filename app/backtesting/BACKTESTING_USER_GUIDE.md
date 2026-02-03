# Backtesting User Guide

**Last Updated:** January 20, 2026  
**For Version:** 1.0  
**Module:** `app/backtesting/`

---

## Table of Contents

1. [Introduction](#introduction)
2. [Running Mass Backtests](#running-mass-backtests)
3. [Analyzing Results](#analyzing-results)
4. [Interpreting Metrics](#interpreting-metrics)
5. [Common Pitfalls](#common-pitfalls)
6. [Troubleshooting](#troubleshooting)

---

## Introduction

This guide will help you run mass backtests, analyze results, and interpret performance metrics for the trading
strategies in this system.

### What You'll Learn

- ✅ How to configure and run mass backtests
- ✅ How to analyze backtest results
- ✅ How to interpret performance metrics
- ✅ Common mistakes and how to avoid them

### Available Strategies

The system includes 5 built-in strategies:

- **Bollinger Bands**: Price bands based on volatility
- **EMA**: Exponential Moving Average crossover
- **Ichimoku Cloud**: Japanese cloud charting
- **MACD**: Moving Average Convergence Divergence
- **RSI**: Relative Strength Index (oversold/overbought)

### Prerequisites

- Python 3.8+
- Historical data in `data/historical_data/`
- Contract switch dates in `data/historical_data/contract_switch_dates.yaml`

---

## Running Mass Backtests

Mass backtesting allows you to test multiple strategies across different symbols, intervals, and parameter combinations
in parallel.

### Basic Usage

```python
from app.backtesting import MassTester

# Initialize the mass tester
tester = MassTester(
    tested_months=['1!'],  # Test on front month contract
    symbols=['ZS', 'CL', 'GC'],  # Symbols to test
    intervals=['15m', '1h', '4h']  # Timeframes to test
)

# Add RSI tests
tester.add_rsi_tests(
    rsi_periods=[14, 21],
    lower_thresholds=[25, 30],
    upper_thresholds=[70, 75],
    rollovers=[False],
    trailing_stops=[None, 1, 2],
  slippage_ticks_list=[2]
)

# Run all tests
tester.run_tests(verbose=True, max_workers=4, skip_existing=True)
```

### Symbol Groups

Group symbols by asset class for easier configuration:

```python
# Classification of futures
grains = ['ZC', 'ZW', 'ZS', 'ZL']
softs = ['SB', 'KC', 'CC']
energy = ['CL', 'NG']
metal = ['GC', 'SI', 'HG', 'PL']
crypto = ['BTC', 'ETH']
index = ['YM', 'ZB']
forex = ['6E', '6J', '6B', '6A', '6C', '6S']

# Use groups in MassTester
tester = MassTester(
    tested_months=['1!'],
    symbols=grains + softs + energy + metal,
    intervals=['4h']
)
```

### Adding Strategy Tests

#### Bollinger Bands Strategy

```python
tester.add_bollinger_bands_tests(
    periods=[20, 25],
    num_stds=[2.0, 2.5],
    rollovers=[False],
    trailing_stops=[None, 1, 2, 3],
  slippage_ticks_list=[2]
)
```

#### EMA Crossover Strategy

```python
tester.add_ema_crossover_tests(
    ema_shorts=[9, 12, 15],
    ema_longs=[21, 26, 30],
    rollovers=[False],
    trailing_stops=[None, 1, 2, 3],
  slippage_ticks_list=[2]
)
```

#### Ichimoku Cloud Strategy

```python
tester.add_ichimoku_cloud_tests(
    tenkan_periods=[9, 12],
    kijun_periods=[26, 30],
    senkou_span_b_periods=[52, 60],
    displacements=[26, 30],
    rollovers=[False],
    trailing_stops=[None, 1, 2, 3],
  slippage_ticks_list=[2]
)
```

#### MACD Strategy

```python
tester.add_macd_tests(
    fast_periods=[12, 15],
    slow_periods=[26, 30],
    signal_periods=[9, 12],
    rollovers=[False],
    trailing_stops=[None, 1, 2, 3],
  slippage_ticks_list=[2]
)
```

#### RSI Strategy

```python
tester.add_rsi_tests(
    rsi_periods=[13, 14, 21],
    lower_thresholds=[20, 30, 40],
    upper_thresholds=[60, 70, 80],
    rollovers=[False],
    trailing_stops=[None, 1, 2, 3],
  slippage_ticks_list=[2]
)
```

### Complete Example

```python
from app.backtesting import MassTester

# Define symbol groups
grains = ['ZC', 'ZW', 'ZS', 'ZL']
softs = ['SB', 'KC', 'CC']
energy = ['CL', 'NG']
metal = ['GC', 'SI', 'HG', 'PL']


def main():
    # Initialize the mass tester
    tester = MassTester(
        tested_months=['1!'],
        symbols=grains + softs + energy + metal,
        intervals=['4h']
    )

    # Bollinger Bands measure volatility
    tester.add_bollinger_bands_tests(
        periods=[20],
        num_stds=[2.0],
        rollovers=[False],
        trailing_stops=[None, 1, 2, 3],
        slippage_ticks_list=[2]
    )

    # EMA Crossover uses two moving averages
    tester.add_ema_crossover_tests(
        ema_shorts=[9, 12],
        ema_longs=[21, 26],
        rollovers=[False],
        trailing_stops=[None, 1, 2, 3],
        slippage_ticks_list=[2]
    )

    # RSI is a momentum oscillator
    tester.add_rsi_tests(
        rsi_periods=[13],
        lower_thresholds=[20, 30, 40],
        upper_thresholds=[60, 70, 80],
        rollovers=[False],
        trailing_stops=[None, 1, 2, 3],
        slippage_ticks_list=[2]
    )

    # Run all tests
    tester.run_tests(verbose=True, max_workers=2, skip_existing=True)


if __name__ == '__main__':
    main()
```

### Run Tests Parameters

**verbose** (bool): Print progress messages

- Default: `True`
- Set to `False` for silent operation

**max_workers** (int): Number of parallel workers

- Default: `None` (uses system default)
- Recommended: 2-4 for most systems
- Higher values may cause memory issues

**skip_existing** (bool): Skip already completed tests

- Default: `True`
- Always use `True` to avoid re-running tests
- System automatically saves progress every 1000 tests

### Performance Tips

1. **Start with one strategy**: Test one strategy type first to validate setup
2. **Use reasonable parameter ranges**: Avoid testing too many combinations
3. **Test incrementally**: Start with 1-2 symbols, then expand
4. **Monitor memory**: Reduce max_workers if memory issues occur
5. **Use skip_existing=True**: Always enable to preserve progress

---

## Analyzing Results

### Using StrategyAnalyzer

The `StrategyAnalyzer` class provides methods to analyze and rank strategies from your mass backtest results.

```python
from app.backtesting import StrategyAnalyzer

# Initialize analyzer
analyzer = StrategyAnalyzer()

# Get top strategies by profit factor
top_profit_factor = analyzer.get_top_strategies(
    metric='profit_factor',
    min_avg_trades_per_combination=20,
    limit=30,
    aggregate=True,
    interval='4h',
    weighted=True,
  min_slippage_ticks=2,
    min_symbol_count=3
)

print(top_profit_factor)
```

### get_top_strategies() Parameters

**metric** (str): Metric to rank strategies by

- Options: `'profit_factor'`, `'average_trade_return_percentage_of_margin'`, `'total_return_percentage_of_margin'`,
  `'sharpe_ratio'`, `'win_rate'`
- Most common: `'profit_factor'` or `'average_trade_return_percentage_of_margin'`

**min_avg_trades_per_combination** (int): Minimum average trades required

- Filters out strategies with too few trades
- Default: `10`
- Higher values = more statistically significant results
- Example: `20` ensures strategies have good sample size

**limit** (int): Number of top strategies to return

- Default: `10`
- Common values: `10`, `20`, `30`, `50`

**aggregate** (bool): Aggregate across symbols/months

- `True`: Combine results across all symbols and months
- `False`: Show individual results
- Recommended: `True` for overall performance

**interval** (str): Filter by specific timeframe

- Options: `'5m'`, `'15m'`, `'30m'`, `'1h'`, `'2h'`, `'4h'`, `'1d'`
- Optional: Leave `None` to include all intervals
- Example: `'4h'` shows only 4-hour strategies

**weighted** (bool): Use weighted aggregation

- `True`: Weight by number of trades (recommended)
- `False`: Simple average
- Weighted gives more importance to strategies with more trades

**min_slippage_ticks** (int): Minimum slippage level in ticks

- Filters to only realistic slippage values
- Example: `2` includes only strategies tested with 2 or more ticks of slippage
- Ensures realistic transaction costs

**min_symbol_count** (int): Minimum number of symbols strategy must work on

- Filters strategies that only work on one or two symbols
- Example: `3` requires strategy to work on at least 3 different symbols
- Helps avoid overfitting to specific instruments

### Complete Analysis Example

```python
from app.backtesting import StrategyAnalyzer


def main():
    print("Analyzing strategy results...")

    # Initialize analyzer
    analyzer = StrategyAnalyzer()

    # Get top strategies by profit factor (4-hour timeframe, weighted)
    top_profit_factor = analyzer.get_top_strategies(
        metric='profit_factor',
        min_avg_trades_per_combination=20,
        limit=30,
        aggregate=True,
        interval='4h',
        weighted=True,
        min_slippage_ticks=2,
        min_symbol_count=3
    )

    print("\nTop 30 Strategies by Profit Factor:")
    print(top_profit_factor)

    # Get top strategies by average trade return
    top_avg_return = analyzer.get_top_strategies(
        metric='average_trade_return_percentage_of_margin',
        min_avg_trades_per_combination=10,
        limit=30,
        aggregate=True,
        interval='4h',
        weighted=True,
        min_slippage_ticks=2,
        min_symbol_count=3
    )

    print("\nTop 30 Strategies by Average Trade Return:")
    print(top_avg_return)

    # Get top strategies across all timeframes
    top_overall = analyzer.get_top_strategies(
        metric='sharpe_ratio',
        min_avg_trades_per_combination=15,
        limit=50,
        aggregate=True,
        weighted=True,
        min_slippage_ticks=2,
        min_symbol_count=2
    )

    print("\nTop 50 Strategies Overall by Sharpe Ratio:")
    print(top_overall)


if __name__ == "__main__":
    main()
```

## Interpreting Metrics

### Key Performance Metrics

#### 1. Total Return (%)

```
What: Total profit/loss as percentage of margin
Range: Can be any value (negative = loss, positive = profit)
Good: > 10% per month
Great: > 20% per month

Interpretation:
- This is your raw performance
- Does NOT account for risk
- Higher is better, but check risk metrics too
```

#### 2. Win Rate (%)

```
What: Percentage of winning trades
Range: 0-100%
Good: > 50%
Great: > 60%

Interpretation:
- 50%+ means more winners than losers
- Low win rate + high returns = large winners, many small losers
- High win rate + low returns = many small winners, large losers
```

#### 3. Sharpe Ratio

```
What: Risk-adjusted return (return per unit of risk)
Range: Can be negative to positive
Good: > 1.0
Great: > 2.0

Interpretation:
- Sharpe < 0: Losing strategy
- Sharpe 0-1: Poor risk-adjusted returns
- Sharpe 1-2: Good risk-adjusted returns
- Sharpe > 2: Excellent risk-adjusted returns
```

#### 4. Maximum Drawdown (%)

```
What: Largest peak-to-trough decline
Range: 0-100% (always shown as positive)
Good: < 20%
Great: < 10%

Interpretation:
- Shows worst-case loss scenario
- If 30%, you lost 30% from highest point
- Key for position sizing
- Lower is better
```

#### 5. Calmar Ratio

```
What: Return / Max Drawdown
Range: Can be any value
Good: > 1.0
Great: > 3.0

Interpretation:
- Measures return vs risk
- Calmar 3.0 = 3% return for every 1% max drawdown
- Higher is better
```

#### 6. Profit Factor

```
What: Gross profit / Gross loss
Range: 0 to infinity
Good: > 1.5
Great: > 2.0

Interpretation:
- PF 1.5 = $1.50 profit for every $1.00 loss
- PF < 1.0 = Losing strategy
- PF 1.0-1.5 = Barely profitable
- PF > 2.0 = Strong profitability
```

### Metric Combinations to Watch

#### High Win Rate + Low Profit Factor

```
Example: Win Rate 70%, Profit Factor 1.2

Problem: Many small wins, few large losses
Risk: One bad trade wipes out many wins
Action: Improve stop loss or target sizing
```

#### Low Win Rate + High Profit Factor

```
Example: Win Rate 40%, Profit Factor 2.5

Good: Large winners compensate for losses
Risk: Psychologically difficult (many losses)
Action: This is fine - trend-following style
```

#### High Returns + High Drawdown

```
Example: Return 50%, Max DD 40%

Problem: Unsustainable volatility
Risk: Large drawdowns are hard to recover from
Action: Consider reducing position size
```

#### Good Sharpe + Low Calmar

```
Example: Sharpe 2.0, Calmar 0.8

Problem: Returns good vs volatility, bad vs drawdown
Interpretation: Consistent small profits with large occasional loss
Action: Improve worst-case risk management
```

### Red Flags

🚩 **Total Trades < 10**: Not statistically significant
🚩 **Sharpe < 0.5**: Poor risk-adjusted returns
🚩 **Max DD > 50%**: Unacceptable risk
🚩 **Profit Factor < 1.2**: Barely profitable
🚩 **Win Rate < 30%**: Too many losers (unless huge winners)
🚩 **Win Rate > 90%**: Likely curve-fitting or look-ahead bias

---

## Additional Resources

### Documentation

- **Architecture**: `BACKTESTING_ARCHITECTURE.md` - System design overview
- **Examples**: `BACKTESTING_EXAMPLES.md` - Real-world examples with data
- **Analysis**: `.github/prompts/BACKTESTING_ANALYSIS.md` - Code quality review

### Code References

- **Mass Testing**: `app/backtesting/mass_testing.py`
- **Strategies**: `app/backtesting/strategies/`
- **Indicators**: `app/backtesting/indicators/`

### Data Files

- **Results**: `data/backtesting/mass_test_results_all.parquet`
- **Historical Data**: `data/historical_data/{symbol}/{symbol}_{interval}.parquet`
- **Switch Dates**: `data/historical_data/contract_switch_dates.yaml`

---

**Happy Backtesting! 🚀📈**

For questions or issues, review the source code or check the troubleshooting section above.
