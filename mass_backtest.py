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
        rsi_periods=[13, 14, 15],
        lower_thresholds=[20, 25, 30],
        upper_thresholds=[60, 65, 70],
        rollovers=[False],
        trailing_stops=[None, 1, 1.5, 2, 2.5, 3],
        slippages=[0, 0.05, 0.1]
    )
    #
    # EMA Crossover uses two moving averages to identify trend changes
    tester.add_ema_crossover_tests(
        ema_shorts=[9, 10, 11, 12],
        ema_longs=[21, 22, 23, 24],
        rollovers=[False],
        trailing_stops=[None, 1, 1.5, 2, 2.5, 3],
        slippages=[0, 0.05, 0.1]
    )

    # Bollinger Bands measure volatility and relative price levels
    tester.add_bollinger_bands_tests(
        periods=[20],
        num_stds=[2],
        rollovers=[False],
        trailing_stops=[None, 1, 1.5, 2, 2.5, 3],
        slippages=[0, 0.05, 0.1]
    )

    # MACD identifies changes in momentum, direction, and strength
    tester.add_macd_tests(
        fast_periods=[12],
        slow_periods=[26],
        signal_periods=[9],
        rollovers=[False],
        trailing_stops=[None, 1, 1.5, 2, 2.5, 3],
        slippages=[0, 0.05, 0.1]
    )

    # Ichimoku Cloud is a comprehensive indicator that provides information on support/resistance, trend direction, and momentum
    tester.add_ichimoku_cloud_tests(
        tenkan_periods=[9],
        kijun_periods=[26],
        senkou_span_b_periods=[52],
        displacements=[26],
        rollovers=[False],
        trailing_stops=[None, 1, 1.5, 2, 2.5, 3],
        slippages=[0, 0.05, 0.1]
    )

    # Run all tests
    # Set skip_existing=False to force re-running of all tests
    tester.run_tests(verbose=False, max_workers=None)


if __name__ == '__main__':
    main()
