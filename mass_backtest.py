from app.backtesting import MassTester
from futures_config import CATEGORIES

# ==================== Configuration ====================

# Select which categories to test (see CATEGORIES.keys() for all available)
# Available categories: Grains, Softs, Energy, Metals, Crypto, Index, Forex
CATEGORIES_TO_TEST = ['Grains']

# Build symbol list from selected categories
SYMBOLS_TO_TEST = []
for category in CATEGORIES_TO_TEST:
    SYMBOLS_TO_TEST.extend(CATEGORIES[category])

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
    #     periods=[15, 20, 25, 30, 35],
    #     num_stds=[1.5, 2.0, 2.5, 3.0, 3.5],
    #     **COMMON_PARAMS
    # )

    # EMA Crossover uses two moving averages to identify trend changes
    # tester.add_ema_crossover_tests(
    #     ema_shorts=[5, 8, 10, 12, 15],
    #     ema_longs=[20, 25, 30, 35, 40],
    #     **COMMON_PARAMS
    # )

    # Ichimoku Cloud is a comprehensive indicator that provides information on support/resistance, trend direction, and momentum
    # tester.add_ichimoku_cloud_tests(
    #     tenkan_periods=[7, 9, 12, 15, 18],
    #     kijun_periods=[20, 24, 26, 28, 30],
    #     senkou_span_b_periods=[44, 52, 60, 70, 80],
    #     displacements=[10, 18, 22, 26, 30],
    #     **COMMON_PARAMS
    # )

    # MACD identifies changes in momentum, direction, and strength
    # tester.add_macd_tests(
    #     fast_periods=[8, 10, 12, 14, 16],
    #     slow_periods=[22, 26, 30, 34, 38],
    #     signal_periods=[7, 9, 11, 13, 15],
    #     **COMMON_PARAMS
    # )

    # RSI is a momentum oscillator that measures the speed and change of price movements
    tester.add_rsi_tests(
        rsi_periods=[13],
        lower_thresholds=[20, 30, 40],
        upper_thresholds=[60, 70, 80],
        **COMMON_PARAMS
    )

    # Run all tests
    # Set skip_existing=False to force re-running of all tests
    tester.run_tests(verbose=False, max_workers=8, skip_existing=True)


if __name__ == '__main__':
    main()
