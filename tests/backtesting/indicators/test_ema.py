import numpy as np
import pandas as pd
import pytest

from app.backtesting.indicators import calculate_ema


def test_calculate_ema_basic_case():
    """Test EMA calculation on a sample data"""
    prices = pd.Series([10, 12, 14, 16, 18, 20])
    period = 3
    ema = calculate_ema(prices, period=period)
    expected = prices.ewm(span=period, adjust=False).mean()
    pd.testing.assert_series_equal(ema, expected)


def test_calculate_ema_empty_series():
    """Test EMA calculation with an empty Series"""
    prices = pd.Series(dtype=float)
    ema = calculate_ema(prices)
    expected = pd.Series(dtype=float)
    pd.testing.assert_series_equal(ema, expected)


def test_calculate_ema_single_value():
    """Test EMA calculation on a single value"""
    prices = pd.Series([10])
    ema = calculate_ema(prices, period=3)
    expected = prices.ewm(span=3, adjust=False).mean()
    pd.testing.assert_series_equal(ema, expected)


def test_calculate_ema_invalid_period():
    """Test EMA calculation with an invalid period"""
    prices = pd.Series([10, 12, 14])
    with pytest.raises(ValueError):
        calculate_ema(prices, period=0)


def test_calculate_ema_non_numeric_data():
    """Test EMA calculation with non-numeric data"""
    prices = pd.Series(["a", "b", "c"])
    with pytest.raises(pd.errors.DataError):
        calculate_ema(prices)


def test_calculate_ema_with_custom_period():
    """Test EMA calculation with different custom periods"""
    prices = pd.Series([10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])

    # Test with period=5
    ema5 = calculate_ema(prices, period=5)
    expected5 = prices.ewm(span=5, adjust=False).mean()
    pd.testing.assert_series_equal(ema5, expected5)

    # Test with period=7
    ema7 = calculate_ema(prices, period=7)
    expected7 = prices.ewm(span=7, adjust=False).mean()
    pd.testing.assert_series_equal(ema7, expected7)


def test_calculate_ema_with_increasing_prices():
    """Test EMA calculation with consistently increasing prices"""
    prices = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    period = 5
    ema = calculate_ema(prices, period=period)

    # With consistently increasing prices, EMA should lag behind the actual price
    # For increasing prices, EMA should be less than the current price after the initial period
    for i in range(period, len(prices)):
        assert ema.iloc[i] < prices.iloc[i]


def test_calculate_ema_with_decreasing_prices():
    """Test EMA calculation with consistently decreasing prices"""
    prices = pd.Series([20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10])
    period = 5
    ema = calculate_ema(prices, period=period)

    # With consistently decreasing prices, EMA should lag behind the actual price
    # For decreasing prices, EMA should be greater than the current price after the initial period
    for i in range(period, len(prices)):
        assert ema.iloc[i] > prices.iloc[i]


def test_calculate_ema_with_alternating_prices():
    """Test EMA calculation with alternating increasing and decreasing prices"""
    prices = pd.Series([10, 12, 10, 12, 10, 12, 10, 12, 10, 12])
    period = 3
    ema = calculate_ema(prices, period=period)

    # EMA should smooth out the alternating prices
    expected = prices.ewm(span=period, adjust=False).mean()
    pd.testing.assert_series_equal(ema, expected)


def test_calculate_ema_with_negative_prices():
    """Test EMA calculation with negative price values"""
    prices = pd.Series([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
    period = 4
    ema = calculate_ema(prices, period=period)

    # EMA should work correctly with negative values
    expected = prices.ewm(span=period, adjust=False).mean()
    pd.testing.assert_series_equal(ema, expected)


def test_calculate_ema_with_known_values():
    """Test EMA calculation with known expected values"""
    # Example from a financial calculation
    prices = pd.Series([22.27, 22.19, 22.08, 22.17, 22.18, 22.13, 22.23, 22.43, 22.24, 22.29])
    period = 3
    ema = calculate_ema(prices, period=period)

    # Get the actual values from the pandas ewm implementation
    expected = prices.ewm(span=period, adjust=False).mean()

    # Check that the values match exactly
    pd.testing.assert_series_equal(ema, expected)


def test_calculate_ema_crossover_signal():
    """Test EMA crossover signals (important for trading strategies)"""
    # Create a price series that will generate a crossover
    # First downtrend, then uptrend to create a crossover
    prices = pd.Series([
                           100, 98, 96, 94, 92, 90, 88, 86, 84, 82, 80,
                           # Uptrend starts here to create crossover
                           82, 84, 86, 88, 90, 92, 94, 96, 98, 100
                       ])

    # Calculate fast and slow EMAs
    fast_ema = calculate_ema(prices, period=5)  # Short-term EMA
    slow_ema = calculate_ema(prices, period=10)  # Long-term EMA

    # Find crossover points
    # A buy signal occurs when fast EMA crosses above slow EMA
    crossovers = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))

    # Verify that we have at least one crossover (buy signal)
    assert crossovers.sum() >= 1

    # Find the index of the first crossover
    if crossovers.any():
        crossover_idx = crossovers[crossovers].index[0]

        # Verify that before the crossover, fast EMA was below slow EMA
        assert fast_ema.loc[crossover_idx - 1] <= slow_ema.loc[crossover_idx - 1]

        # Verify that at the crossover, fast EMA is above slow EMA
        assert fast_ema.loc[crossover_idx] > slow_ema.loc[crossover_idx]


def test_calculate_ema_with_price_gaps():
    """Test EMA calculation with price gaps (missing data)"""
    # Create a series with NaN values to simulate missing data
    prices = pd.Series([
                           50, 51, 52, np.nan, 54, 55, np.nan, np.nan, 58,
                           59, 60, 61, 62, np.nan, 64, 65, 66, 67, 68
                       ])

    # First, let's handle the NaN values by forward filling
    filled_prices = prices.ffill()

    # Calculate EMA on the filled prices
    ema = calculate_ema(filled_prices, period=5)

    # Check that the EMA calculation completes successfully
    assert not ema.isna().all()

    # Check that the last few values are calculated correctly
    assert not np.isnan(ema.iloc[-1])

    # Verify that EMA follows the general trend of the prices
    # For an upward trend, the last EMA value should be higher than the first valid EMA value
    first_valid_idx = ema.first_valid_index()
    if first_valid_idx is not None:
        assert ema.iloc[-1] > ema.iloc[first_valid_idx]

    # Calculate another EMA with a different period for comparison
    ema_longer = calculate_ema(filled_prices, period=10)

    # The shorter period EMA should be more responsive to recent price changes
    # In an uptrend, the shorter period EMA should be higher at the end
    assert ema.iloc[-1] > ema_longer.iloc[-1]


def test_calculate_ema_with_market_crash():
    """Test EMA calculation during a market crash scenario"""
    # Simulate a stable market followed by a sharp decline (crash)
    prices = pd.Series([
                           100, 101, 102, 103, 102, 101, 100, 99, 100, 101,
                           # Sharp decline starts here
                           95, 90, 85, 80, 75, 70, 65, 60, 55, 50
                       ])

    # Calculate EMAs with different periods
    short_ema = calculate_ema(prices, period=5)
    long_ema = calculate_ema(prices, period=10)

    # During a crash, shorter-period EMAs should fall faster than longer-period EMAs
    # Check the rate of decline in the crash period
    assert (short_ema.iloc[-1] / short_ema.iloc[-10]) < (long_ema.iloc[-1] / long_ema.iloc[-10])

    # Short EMA should be below long EMA during a sustained downtrend
    assert short_ema.iloc[-1] < long_ema.iloc[-1]

    # Both EMAs should be below the initial price level
    assert short_ema.iloc[-1] < prices.iloc[0]
    assert long_ema.iloc[-1] < prices.iloc[0]


def test_calculate_ema_with_multiple_periods():
    """Test multiple EMAs with different periods (common in trading strategies)"""
    # Create a realistic price series
    prices = pd.Series([
                           100, 102, 104, 103, 105, 107, 106, 108, 110, 109,
                           111, 113, 112, 114, 116, 115, 117, 119, 118, 120
                       ])

    # Calculate EMAs with different periods commonly used in trading
    ema9 = calculate_ema(prices, period=9)
    ema21 = calculate_ema(prices, period=21)
    ema50 = calculate_ema(prices, period=50)

    # In an uptrend, shorter EMAs should be above longer EMAs
    # But we don't have enough data for EMA50, so we'll just check EMA9 vs EMA21

    # Verify that EMA9 responds more quickly to recent price changes than EMA21
    # In the last part of our uptrend, EMA9 should be higher
    assert ema9.iloc[-1] > ema21.iloc[-1]

    # Calculate the correlation between the EMAs
    correlation = ema9.corr(ema21)

    # EMAs should be highly correlated even with different periods
    assert correlation > 0.9

    # Verify that EMA9 is more volatile (has higher standard deviation)
    assert ema9.std() > ema21.std()


def test_calculate_ema_with_sideways_market():
    """Test EMA calculation in a sideways (range-bound) market"""
    # Simulate a sideways market with prices oscillating in a range
    prices = pd.Series([
                           100, 102, 98, 103, 97, 104, 96, 105, 95, 104,
                           96, 103, 97, 102, 98, 101, 99, 100, 100, 101
                       ])

    # Calculate EMAs with different periods
    short_ema = calculate_ema(prices, period=5)
    long_ema = calculate_ema(prices, period=10)

    # In a sideways market, EMAs should converge and stay close to the middle of the range
    # Calculate the average price
    avg_price = prices.mean()

    # Both EMAs should be close to the average price in a sideways market
    assert abs(short_ema.iloc[-1] - avg_price) < 5
    assert abs(long_ema.iloc[-1] - avg_price) < 5

    # EMAs should be close to each other
    assert abs(short_ema.iloc[-1] - long_ema.iloc[-1]) < 3

    # The standard deviation of the EMAs should be less than that of the prices
    # (EMAs smooth out the volatility)
    assert short_ema.std() < prices.std()
    assert long_ema.std() < prices.std()