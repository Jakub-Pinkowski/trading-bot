import time

from app.backtesting.mass_testing import MassTester


def main():
    start_time = time.time()

    # Initialize the mass tester with multiple symbols and timeframes
    tester = MassTester(
        tested_months=['1!'],
        symbols=['ZC', 'ZW', 'ZS', 'ZL'],
        intervals=['5m', '15m', '30m', '1h', '2h', '4h', '1d']
    )

    # RSI is a momentum oscillator that measures the speed and change of price movements
    tester.add_rsi_tests(
        rsi_periods=[7, 8, 14, 15, 16, 21],
        lower_thresholds=[20, 25, 30, 35],
        upper_thresholds=[65, 70, 75, 80],
        rollovers=[False],
        trailing_stops=[None, 1, 2],
        slippages=[0, 0.05, 0.1, 0.15]
    )
    #
    # # EMA Crossover uses two moving averages to identify trend changes
    # tester.add_ema_crossover_tests(
    #     ema_shorts=[5, 8, 9, 10, 12],
    #     ema_longs=[20, 21, 25, 30, 50],
    #     rollovers=[False],
    #     trailing_stops=[None, 1, 2],
    #     slippages=[0, 0.05, 0.1]
    # )
    #
    # # Bollinger Bands measure volatility and relative price levels
    # tester.add_bollinger_bands_tests(
    #     periods=[10, 20, 22, 30, 40],
    #     num_stds=[1.5, 2, 2.5, 3],
    #     rollovers=[False],
    #     trailing_stops=[None, 1, 2],
    #     slippages=[0, 0.05, 0.1]
    # )

    # MACD identifies changes in momentum, direction, and strength
    # tester.add_macd_tests(
    #     fast_periods=[8, 10, 12, 15],
    #     slow_periods=[21, 26, 30, 35],
    #     signal_periods=[5, 7, 9, 12],
    #     rollovers=[False],
    #     trailing_stops=[None,0.5, 1],
    #     slippages=[0, 0.05, 0.1, 0.15, 0.2]
    # )

    # Run all tests
    # Set skip_existing=False to force re-running of all tests
    tester.run_tests(verbose=False, max_workers=None)

    end_time = time.time()
    total_time = end_time - start_time

    # Format total_time as MM:SS
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    print(f'Total execution time: {minutes:02}:{seconds:02} (MM:SS) | {total_time:.2f} seconds')


if __name__ == '__main__':
    main()
