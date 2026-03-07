from app.backtesting import MassTester

# ==================== Configuration ====================

# Normal Grains only (ZC=Corn, ZW=Wheat, ZS=Soybeans, ZL=Soybean Oil)
# Excludes mini (XC, XW, XK) and micro (MZC, MZW, MZS, MZL) contracts
SYMBOLS_TO_TEST = ['ZC', 'ZW', 'ZS', 'ZL']

# Common parameters shared across all strategy tests
COMMON_PARAMS = dict(
    rollovers=[False],
    trailing_stops=[None, 1, 2, 3],
    slippage_ticks_list=[3]
)


# ==================== Main ====================

def main():
    # Initialize the mass tester with multiple symbols and timeframes
    tester = MassTester(
        tested_months=['1!'],
        symbols=SYMBOLS_TO_TEST,
        intervals=['15m', '30m', '1h', '4h'],
    )

    # Bollinger Bands measure volatility and relative price levels
    # tester.add_bollinger_bands_tests(
    #     periods=[15, 20, 25, 30],
    #     number_of_standard_deviations_list=[1.5, 2.0, 2.5, 3.0],
    #     **COMMON_PARAMS
    # )

    # EMA Crossover uses two moving averages to identify trend changes
    # tester.add_ema_crossover_tests(
    #     short_ema_periods=[5, 8, 12, 15],
    #     long_ema_periods=[20, 25, 35, 40],
    #     **COMMON_PARAMS
    # )

    # Ichimoku Cloud is a comprehensive indicator that provides information on support/resistance, trend direction, and momentum
    # tester.add_ichimoku_cloud_tests(
    #     tenkan_periods=[7, 9, 12, 18],
    #     kijun_periods=[20, 24, 26, 30],
    #     senkou_span_b_periods=[44, 52, 60, 80],
    #     displacements=[10, 18, 22, 30],
    #     **COMMON_PARAMS
    # )

    # MACD identifies changes in momentum, direction, and strength
    # tester.add_macd_tests(
    #     fast_periods=[8, 10, 12, 16],
    #     slow_periods=[22, 26, 30, 38],
    #     signal_periods=[7, 9, 11, 15],
    #     **COMMON_PARAMS
    # )

    # RSI is a momentum oscillator that measures the speed and change of price movements
    tester.add_rsi_tests(
        rsi_periods=[11, 12, 13, 14, 15, 16],
        lower_thresholds=[20, 25, 30, 35, 40],
        upper_thresholds=[60, 65, 70, 75, 80],
        **COMMON_PARAMS
    )

    # Run all tests
    # Set skip_existing=False to force re-running of all tests
    tester.run_tests(verbose=False, max_workers=8, skip_existing=True)


if __name__ == '__main__':
    main()
