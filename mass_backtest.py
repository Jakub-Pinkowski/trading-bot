from app.backtesting.mass_testing import MassTester

# Initialize the mass tester with multiple symbols and timeframes
tester = MassTester(
    tested_months=["1!"],  # Front month contracts
    symbols=["ZW", "ZC", "ZS"],  # Wheat, Corn, Soybeans
    intervals=["1h", "4h"]  # 1-hour and 4-hour timeframes
)

# Add RSI strategy tests with various parameter combinations
tester.add_rsi_tests(
    rsi_periods=[7, 14, 21],  # Test different RSI periods
    lower_thresholds=[20, 30],  # Test different oversold thresholds
    upper_thresholds=[70, 80],  # Test different overbought thresholds
    rollovers=[False, True],  # Test with and without a rollover
    trailing_stops=[None, 1.0, 2.0]  # Test with different trailing stops
)

# Add EMA Crossover strategy tests
tester.add_ema_crossover_tests(
    ema_shorts=[5, 9, 13],  # Test different short EMA periods
    ema_longs=[21, 34],  # Test different long EMA periods
    rollovers=[False],  # Only test without rollover
    trailing_stops=[None, 1.5]  # Test with and without a trailing stop
)

# Run all tests
print("Running backtests for all parameter combinations...")
results = tester.run_tests(verbose=True, save_results=True)

# Get top strategies by profit factor with at least 10 trades
print("\n===== TOP STRATEGIES BY PROFIT FACTOR =====")
top_by_profit = tester.get_top_strategies(metric="profit_factor", min_trades=10)
print(top_by_profit.head(10))

# Get top strategies by win rate
print("\n===== TOP STRATEGIES BY WIN RATE =====")
top_by_winrate = tester.get_top_strategies(metric="win_rate", min_trades=10)
print(top_by_winrate.head(10))

# Compare strategies by symbol
print("\n===== STRATEGY COMPARISON BY SYMBOL =====")
by_symbol = tester.compare_strategies(group_by=["strategy", "symbol"])
print(by_symbol.head(20))

# Compare strategies by timeframe
print("\n===== STRATEGY COMPARISON BY TIMEFRAME =====")
by_interval = tester.compare_strategies(group_by=["strategy", "interval"])
print(by_interval.head(20))

print(f"\nResults have been saved")
