from app.backtesting.mass_testing import MassTester


def main():
    # Initialize the mass tester with multiple symbols and timeframes
    tester = MassTester(
        tested_months=['1!'],
        symbols=['ZC', 'ZS', 'ZL', 'ZW'],
        intervals=['4h'],
    )

    # RSI is a momentum oscillator that measures the speed and change of price movements
    tester.add_rsi_tests(
        rsi_periods=[14],
        lower_thresholds=[30],
        upper_thresholds=[70],
        rollovers=[False],
        trailing_stops=[None],
        slippages=[0]
    )

    # EMA Crossover uses two moving averages to identify trend changes
    # tester.add_ema_crossover_tests(
    #     ema_shorts=[5, 6, 8, 9, 10, 11, 12, 13, 15],
    #     ema_longs=[18, 20, 21, 22, 23, 24, 26, 30, 35, 50],
    #     rollovers=[False],
    #     trailing_stops=[None, 1, 1.5, 2, 2.5, 3],
    #     slippages=[0, 0.05, 0.1]
    # )

    # Bollinger Bands measure volatility and relative price levels
    # tester.add_bollinger_bands_tests(
    #     periods=[10, 14, 18, 20, 22, 25],
    #     num_stds=[1, 1.5, 2, 2.5, 3],
    #     rollovers=[False],
    #     trailing_stops=[None, 1, 1.5, 2, 2.5, 3],
    #     slippages=[0, 0.05, 0.1]
    # )

    # MACD identifies changes in momentum, direction, and strength
    # tester.add_macd_tests(
    #     fast_periods=[5, 8, 10, 12, 13],
    #     slow_periods=[18, 21, 26, 30, 35],
    #     signal_periods=[5, 7, 9, 12, 15],
    #     rollovers=[False],
    #     trailing_stops=[None, 1, 1.5, 2, 2.5, 3],
    #     slippages=[0, 0.05, 0.1]
    # )

    # Ichimoku Cloud is a comprehensive indicator that provides information on support/resistance, trend direction, and momentum
    # tester.add_ichimoku_cloud_tests(
    #     tenkan_periods=[5, 7, 9, 12, 15],
    #     kijun_periods=[18, 22, 26, 30, 40],
    #     senkou_span_b_periods=[30, 42, 52, 60, 80],
    #     displacements=[10, 22, 26, 30, 40],
    #     rollovers=[False],
    #     trailing_stops=[None, 1, 1.5, 2, 2.5, 3],
    #     slippages=[0, 0.05, 0.1]
    # )

    # Run all tests
    # Set skip_existing=False to force re-running of all tests
    tester.run_tests(verbose=False, max_workers=None)


if __name__ == '__main__':
    main()
