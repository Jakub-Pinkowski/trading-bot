from app.backtesting.mass_testing import MassTester


def main():
    # Initialize the mass tester with multiple symbols and timeframes
    tester = MassTester(
        tested_months=['1!'],
        symbols=['ZW', 'ZC', 'ZS', 'ZL'],
        intervals=['5m', '15m']
    )

    # Add RSI strategy tests with various parameter combinations
    tester.add_rsi_tests(
        rsi_periods=[7, 14],
        lower_thresholds=[20, 30],
        upper_thresholds=[70, 80],
        rollovers=[False, True],
        trailing_stops=[None, 2]
    )

    # Add EMA Crossover strategy tests
    tester.add_ema_crossover_tests(
        ema_shorts=[5, 9],
        ema_longs=[21, 34],
        rollovers=[False, True],
        trailing_stops=[None, 2]
    )

    # Run all tests in parallel
    print('Running backtests for all parameter combinations in parallel...')
    # max_workers=None will use the number of processors on the machine
    results = tester.run_tests(verbose=False, save_results=True, max_workers=None)

    print(f'\nResults have been saved')

    # Compare strategies by symbol
    print('\n===== STRATEGY COMPARISON BY SYMBOL =====')
    by_symbol = tester.compare_strategies(group_by=['strategy', 'symbol'])
    print(by_symbol.head(20))


if __name__ == '__main__':
    main()
