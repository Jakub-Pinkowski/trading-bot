import time

from app.backtesting.mass_testing import MassTester


def main():
    start_time = time.time()

    # Initialize the mass tester with multiple symbols and timeframes
    tester = MassTester(
        tested_months=['1!'],
        symbols=['ZC', 'ZW', 'ZS', 'ZL'],
        intervals=['5m', '15m', '30m', '1h', '2h', '4h']
    )

    # Add RSI strategy tests with various parameter combinations
    tester.add_rsi_tests(
        rsi_periods=[56],
        lower_thresholds=[20, 25, 30, 35],
        upper_thresholds=[65, 70, 75, 80],
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

    # Run all tests
    tester.run_tests(verbose=False, max_workers=None)

    end_time = time.time()
    total_time = end_time - start_time

    # Format total_time as MM:SS
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    print(f'Total execution time: {minutes:02}:{seconds:02} (MM:SS) | {total_time:.2f} seconds')


if __name__ == '__main__':
    main()
