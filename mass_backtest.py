from app.backtesting.mass_testing import MassTester


def main():
    # Initialize the mass tester with multiple symbols and timeframes
    tester = MassTester(
        tested_months=['1!'],
        symbols=['ZC', 'ZW', 'ZS', 'ZL'],
        intervals=['1h', '30m']
    )

    # Add RSI strategy tests with various parameter combinations
    tester.add_rsi_tests(
        rsi_periods=[10, 14],
        lower_thresholds=[30],
        upper_thresholds=[70],
        rollovers=[False],
        trailing_stops=[None, 2]
    )

    # Add EMA Crossover strategy tests
    tester.add_ema_crossover_tests(
        ema_shorts=[9],
        ema_longs=[21],
        rollovers=[False, True],
        trailing_stops=[None, 2]
    )

    # Run all tests in parallel
    print('Running backtests for all parameter combinations in parallel...')
    # max_workers=None will use the number of processors on the machine
    results = tester.run_tests(verbose=True, save_results=True, max_workers=None)

    print(f'\nResults have been saved')


if __name__ == '__main__':
    main()
