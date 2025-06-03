from app.backtesting.mass_testing import MassTester


def main():
    # Initialize the mass tester with multiple symbols and timeframes
    tester = MassTester(
        tested_months=['1!'],
        symbols=['ZW', 'ZC', 'ZS', 'ZL'],
        intervals=['5m', '15m', '30m']
    )

    # Add RSI strategy tests with various parameter combinations
    tester.add_rsi_tests(
        rsi_periods=[7, 14],
        lower_thresholds=[20, 30],
        upper_thresholds=[70, 80],
        rollovers=[False, True],
        trailing_stops=[None, 1.0, 1.5, 2]
    )

    # Add EMA Crossover strategy tests
    tester.add_ema_crossover_tests(
        ema_shorts=[5, 9],
        ema_longs=[21, 34],
        rollovers=[False, True],
        trailing_stops=[None, 1.0, 1.5, 2]
    )

    # Run all tests in parallel
    print('Running backtests for all parameter combinations in parallel...')
    # max_workers=None will use the number of processors on the machine
    results = tester.run_tests(verbose=False, save_results=True, max_workers=None)

    # Get top strategies by profit factor
    print('\n===== TOP STRATEGIES BY PROFIT FACTOR =====')
    top_by_profit = tester.get_top_strategies(metric='profit_factor', min_trades=10)
    print(top_by_profit.head(10))

    # Get top strategies by total return percentage
    print('\n===== TOP STRATEGIES BY TOTAL TRADE RETURN PERCENTAGE =====')
    top_by_total_return_percentage = tester.get_top_strategies(metric='total_return_percentage_of_margin',
                                                               min_trades=10)
    print(top_by_total_return_percentage.head(10))

    # Get top strategies by average return percentage
    print('\n===== TOP STRATEGIES BY AVERAGE TRADE RETURN PERCENTAGE OF MARGIN =====')
    top_by_trade_return_percentage = tester.get_top_strategies(metric='average_trade_return_percentage_of_margin',
                                                               min_trades=10)
    print(top_by_trade_return_percentage.head(10))

    # Compare strategies by symbol
    print('\n===== STRATEGY COMPARISON BY SYMBOL =====')
    by_symbol = tester.compare_strategies(group_by=['strategy', 'symbol'])
    print(by_symbol.head(20))

    # Compare strategies by timeframe
    print('\n===== STRATEGY COMPARISON BY TIMEFRAME =====')
    by_interval = tester.compare_strategies(group_by=['strategy', 'interval'])
    print(by_interval.head(20))

    print(f'\nResults have been saved')


if __name__ == '__main__':
    main()
