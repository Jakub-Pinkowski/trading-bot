from app.backtesting.mass_testing import MassTester


def main():
    # Initialize the mass tester with multiple symbols and timeframes
    tester = MassTester(
        tested_months=['1!'],
        symbols=['ZW'],
        intervals=['1h']
    )

    # RSI is a momentum oscillator that measures the speed and change of price movements
    # tester.add_rsi_tests(
    #     rsi_periods=[14],
    #     lower_thresholds=[30],
    #     upper_thresholds=[70],
    #     rollovers=[False],
    #     trailing_stops=[None],
    #     slippages=[0]
    # )
    #
    # # EMA Crossover uses two moving averages to identify trend changes
    # tester.add_ema_crossover_tests(
    #     ema_shorts=[9],
    #     ema_longs=[21],
    #     rollovers=[False],
    #     trailing_stops=[None],
    #     slippages=[0]
    # )

    # # Bollinger Bands measure volatility and relative price levels
    # tester.add_bollinger_bands_tests(
    #     periods=[20],
    #     num_stds=[2],
    #     rollovers=[False],
    #     trailing_stops=[None],
    #     slippages=[0]
    # )

    # MACD identifies changes in momentum, direction, and strength
    # tester.add_macd_tests(
    #     fast_periods=[12],
    #     slow_periods=[26],
    #     signal_periods=[9],
    #     rollovers=[False],
    #     trailing_stops=[None],
    #     slippages=[0]
    # )

    # Ichimoku Cloud is a comprehensive indicator that provides information on support/resistance, trend direction, and momentum
    tester.add_ichimoku_cloud_tests(
        tenkan_periods=[9],
        kijun_periods=[26],
        senkou_span_b_periods=[52],
        displacements=[26],
        rollovers=[False],
        trailing_stops=[None],
        slippages=[0]
    )



    # Run all tests
    # Set skip_existing=False to force re-running of all tests
    tester.run_tests(verbose=False, max_workers=None)


if __name__ == '__main__':
    main()
