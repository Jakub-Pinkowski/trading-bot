import numpy as np
import pandas as pd
import pytest

from app.backtesting.indicators import calculate_rsi, calculate_ema


def test_calculate_rsi_with_valid_prices():
    """Test RSI calculation with valid price data"""
    prices = pd.Series([44, 47, 45, 50, 55, 60, 63, 62, 64, 69, 70, 75, 80, 85, 88])
    rsi = calculate_rsi(prices)
    assert rsi.isna().sum() == 14  # Initial undefined values
    assert all(rsi[14:].between(0, 100))  # RSI should be between 0-100 after the period


def test_calculate_rsi_with_not_enough_data():
    """Test RSI calculation with price data less than the default period"""
    prices = pd.Series([44, 47, 45])
    rsi = calculate_rsi(prices)
    assert rsi.isna().all()  # All values should be NaN


def test_calculate_rsi_with_custom_period():
    """Test RSI calculation with a custom period"""
    prices = pd.Series([44, 47, 45, 50, 55, 60, 63, 62, 64, 69, 70, 75, 80])
    rsi = calculate_rsi(prices, period=10)
    assert rsi.isna().sum() == 10  # Undefined values equal to the custom period
    assert all(rsi[10:].between(0, 100))  # RSI should be between 0-100 after the period


def test_calculate_rsi_with_constant_prices():
    """Test RSI calculation when prices remain constant"""
    prices = pd.Series([50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50])
    rsi = calculate_rsi(prices)
    # The entire array is NaN because we have fewer prices than a period
    assert rsi.isna().sum() == 13  # Initial undefined values


def test_calculate_rsi_handles_empty_prices():
    """Test RSI calculation with empty price data"""
    prices = pd.Series(dtype='float64')
    rsi = calculate_rsi(prices)
    assert rsi.empty  # RSI should be empty as well


def test_calculate_rsi_with_increasing_prices():
    """Test RSI calculation with consistently increasing prices"""
    prices = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
    rsi = calculate_rsi(prices)
    # With consistently increasing, prices, RSI should be high (close to 100)
    assert rsi.isna().sum() == 14  # Initial undefined values
    assert all(rsi[14:] > 70)  # RSI should be high for consistently increasing prices


def test_calculate_rsi_with_decreasing_prices():
    """Test RSI calculation with consistently decreasing prices"""
    prices = pd.Series([30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10])
    rsi = calculate_rsi(prices)
    # With consistently decreasing prices, RSI should be low (close to 0)
    assert rsi.isna().sum() == 14  # Initial undefined values
    assert all(rsi[14:] < 30)  # RSI should be low for consistently decreasing prices


def test_calculate_rsi_with_alternating_prices():
    """Test RSI calculation with alternating increasing and decreasing prices"""
    prices = pd.Series([10, 12, 10, 12, 10, 12, 10, 12, 10, 12, 10, 12, 10, 12, 10, 12, 10, 12, 10, 12])
    rsi = calculate_rsi(prices)
    assert rsi.isna().sum() == 14  # Initial undefined values
    # With alternating prices, RSI should be around 50
    assert all(rsi[14:].between(40, 60))


def test_calculate_rsi_with_negative_prices():
    """Test RSI calculation with negative price values"""
    prices = pd.Series([-10, -8, -9, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    rsi = calculate_rsi(prices)
    assert rsi.isna().sum() == 14  # Initial undefined values
    assert all(rsi[14:].between(0, 100))  # RSI should still be between 0-100


def test_calculate_rsi_with_invalid_period():
    """Test RSI calculation with invalid period values"""
    prices = pd.Series([10, 12, 14, 16, 18, 20])
    # Test with zero periods
    with pytest.raises(ValueError):
        calculate_rsi(prices, period=0)

    # Test with a negative period
    with pytest.raises(ValueError):
        calculate_rsi(prices, period=-5)


def test_calculate_ema_basic_case():
    """Test EMA calculation on a sample data"""
    prices = pd.Series([10, 12, 14, 16, 18, 20])
    period = 3
    ema = calculate_ema(prices, period=period)
    expected = prices.ewm(span=period, adjust=False).mean()
    expected[:period - 1] = np.nan
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
    expected = pd.Series([np.nan])
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
    expected5[:4] = np.nan
    pd.testing.assert_series_equal(ema5, expected5)

    # Test with period=7
    ema7 = calculate_ema(prices, period=7)
    expected7 = prices.ewm(span=7, adjust=False).mean()
    expected7[:6] = np.nan
    pd.testing.assert_series_equal(ema7, expected7)


def test_calculate_ema_with_increasing_prices():
    """Test EMA calculation with consistently increasing prices"""
    prices = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    period = 5
    ema = calculate_ema(prices, period=period)

    # With consistently increasing prices, EMA should lag behind the actual price
    assert ema.isna().sum() == period - 1  # Initial undefined values
    # For increasing prices, EMA should be less than the current price
    for i in range(period, len(prices)):
        assert ema.iloc[i] < prices.iloc[i]


def test_calculate_ema_with_decreasing_prices():
    """Test EMA calculation with consistently decreasing prices"""
    prices = pd.Series([20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10])
    period = 5
    ema = calculate_ema(prices, period=period)

    # With consistently decreasing prices, EMA should lag behind the actual price
    assert ema.isna().sum() == period - 1  # Initial undefined values
    # For decreasing prices, EMA should be greater than the current price
    for i in range(period, len(prices)):
        assert ema.iloc[i] > prices.iloc[i]


def test_calculate_ema_with_alternating_prices():
    """Test EMA calculation with alternating increasing and decreasing prices"""
    prices = pd.Series([10, 12, 10, 12, 10, 12, 10, 12, 10, 12])
    period = 3
    ema = calculate_ema(prices, period=period)

    assert ema.isna().sum() == period - 1  # Initial undefined values
    # EMA should smooth out the alternating prices
    expected = prices.ewm(span=period, adjust=False).mean()
    expected[:period - 1] = np.nan
    pd.testing.assert_series_equal(ema, expected)


def test_calculate_ema_with_negative_prices():
    """Test EMA calculation with negative price values"""
    prices = pd.Series([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
    period = 4
    ema = calculate_ema(prices, period=period)

    assert ema.isna().sum() == period - 1  # Initial undefined values
    # EMA should work correctly with negative values
    expected = prices.ewm(span=period, adjust=False).mean()
    expected[:period - 1] = np.nan
    pd.testing.assert_series_equal(ema, expected)


def test_calculate_ema_with_known_values():
    """Test EMA calculation with known expected values"""
    # Example from a financial calculation
    prices = pd.Series([22.27, 22.19, 22.08, 22.17, 22.18, 22.13, 22.23, 22.43, 22.24, 22.29])
    period = 3
    ema = calculate_ema(prices, period=period)

    # The first two values should be NaN
    assert np.isnan(ema.iloc[0])
    assert np.isnan(ema.iloc[1])

    # Get the actual values from the pandas ewm implementation
    expected = prices.ewm(span=period, adjust=False).mean()
    expected[:period - 1] = np.nan

    # Check that the values match exactly
    pd.testing.assert_series_equal(ema, expected)
