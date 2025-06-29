from app.backtesting.mass_testing import MassTester

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
        symbols=grains + energy + metal,
        intervals=['4h'],
    )

    # RSI is a momentum oscillator that measures the speed and change of price movements
    tester.add_rsi_tests(
        rsi_periods=[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        lower_thresholds=[20, 25, 30, 35, 40],
        upper_thresholds=[60, 65, 70, 75, 80],
        rollovers=[False],
        trailing_stops=[None, 1, 2, 3],
        slippages=[0.05]
    )

    # EMA Crossover uses two moving averages to identify trend changes
    tester.add_ema_crossover_tests(
        ema_shorts=[5, 9, 12],
        ema_longs=[18, 21, 26],
        rollovers=[False],
        trailing_stops=[None, 1, 2, 3],
        slippages=[0.05]
    )

    # Bollinger Bands measure volatility and relative price levels
    tester.add_bollinger_bands_tests(
        periods=[10, 14, 20, 25],
        num_stds=[1.5, 2, 2.5, 3],
        rollovers=[False],
        trailing_stops=[None, 1, 2, 3],
        slippages=[0.05]
    )

    # MACD identifies changes in momentum, direction, and strength
    tester.add_macd_tests(
        fast_periods=[8, 12, 13, 15],
        slow_periods=[21, 26, 30, 35],
        signal_periods=[7, 9, 12],
        rollovers=[False],
        trailing_stops=[None, 1, 2, 3],
        slippages=[0.05]
    )

    # Ichimoku Cloud is a comprehensive indicator that provides information on support/resistance, trend direction, and momentum
    tester.add_ichimoku_cloud_tests(
        tenkan_periods=[7, 9, 12, 15],
        kijun_periods=[18, 22, 26, 30],
        senkou_span_b_periods=[42, 52, 60, 80],
        displacements=[10, 22, 26, 30],
        rollovers=[False],
        trailing_stops=[None, 1, 2, 3],
        slippages=[0.05]
    )

    # Run all tests
    # Set skip_existing=False to force re-running of all tests
    tester.run_tests(verbose=False, max_workers=None)


if __name__ == '__main__':
    main()
