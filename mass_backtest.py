from app.backtesting import MassTester

# Classification of futures
grains = [
    'ZC',  # Corn
    'ZW',  # Wheat
    'ZS',  # Soybean
    'ZL',  # Soybean Oil
]

softs = [
    'SB',  # Sugar
    'KC',  # Coffee
    'CC',  # Cocoa
]

energy = [
    'CL',  # Crude Oil
    'NG',  # Natural Gas
]

metal = [
    'GC',  # Gold
    'SI',  # Silver
    'HG',  # Copper
    'PL',  # Platinum
]

crypto = [
    'BTC',  # Bitcoin
    'ETH',  # Ethereum
]

index = [
    # 'ES',  # S&P-500 NOTE: Currently not available
    # 'NQ',  # NASDAQ-100 NOTE: Currently not available
    'YM',  # DOW
    # 'RTY',  # RUSSELL 2000 NOTE: Currently not available
    'ZB',  # Treasury Bond
]

forex = [
    '6E',  # EURO FX
    '6J',  # Japanese Yen
    '6B',  # British Pound
    '6A',  # Australian Dollar
    '6C',  # Canadian Dollar
    '6S',  # Swiss Franc
]


def main():
    # Initialize the mass tester with multiple symbols and timeframes
    tester = MassTester(
        tested_months=['1!'],
        symbols=grains + softs + energy + metal,
        intervals=['4h'],
    )

    # RSI is a momentum oscillator that measures the speed and change of price movements
    tester.add_rsi_tests(
        rsi_periods=[13],
        lower_thresholds=[20, 30, 40],
        upper_thresholds=[60, 70, 80],
        rollovers=[False],
        trailing_stops=[None, 1, 2, 3],
        slippages=[0.05]
    )

    # EMA Crossover uses two moving averages to identify trend changes
    # tester.add_ema_crossover_tests(
    #     ema_shorts=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    #     ema_longs=[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    #     rollovers=[False],
    #     trailing_stops=[None, 1, 2, 3],
    #     slippages=[0.05]
    # )

    # Bollinger Bands measure volatility and relative price levels
    # tester.add_bollinger_bands_tests(
    #     periods=[10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    #     num_stds=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3],
    #     rollovers=[False],
    #     trailing_stops=[None, 1, 2, 3],
    #     slippages=[0.05]
    # )

    # MACD identifies changes in momentum, direction, and strength
    # tester.add_macd_tests(
    #     fast_periods=[8, 9, 10, 11, 12, 13, 14, 15, 16],
    #     slow_periods=[21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    #     signal_periods=[6, 7, 8, 9, 10, 11, 12, 13],
    #     rollovers=[False],
    #     trailing_stops=[None, 1, 2, 3],
    #     slippages=[0.05]
    # )

    # Ichimoku Cloud is a comprehensive indicator that provides information on support/resistance, trend direction, and momentum
    # tester.add_ichimoku_cloud_tests(
    #     tenkan_periods=[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    #     kijun_periods=[18, 19, 20, 21, 22, 26, 27, 28, 29, 30],
    #     senkou_span_b_periods=[42, 43, 44, 45, 52, 60, 80],
    #     displacements=[10, 22, 26, 30],
    #     rollovers=[False],
    #     trailing_stops=[None, 1, 2, 3],
    #     slippages=[0.05]
    # )

    # Run all tests
    # Set skip_existing=False to force re-running of all tests
    tester.run_tests(verbose=False, max_workers=8)


if __name__ == '__main__':
    main()
