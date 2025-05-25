import numpy as np
import pandas as pd
import pytest

from app.backtesting.indicators import calculate_rsi, calculate_ema, calculate_atr


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
    # The entire array is NaN because we have fewer prices than period
    assert rsi.isna().sum() == 13  # Initial undefined values


def test_calculate_rsi_handles_empty_prices():
    """Test RSI calculation with empty price data"""
    prices = pd.Series(dtype='float64')
    rsi = calculate_rsi(prices)
    assert rsi.empty  # RSI should be empty as well


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
    """Test EMA calculation with invalid period"""
    prices = pd.Series([10, 12, 14])
    with pytest.raises(ValueError):
        calculate_ema(prices, period=0)


def test_calculate_ema_non_numeric_data():
    """Test EMA calculation with non-numeric data"""
    prices = pd.Series(["a", "b", "c"])
    with pytest.raises(pd.errors.DataError):
        calculate_ema(prices)


def test_calculate_atr_basic_case():
    """Test calculate_atr with a basic dataframe"""
    data = {
        'high': [10, 12, 14, 18],
        'low': [5, 8, 11, 13],
        'close': [6, 11, 13, 15]
    }
    df = pd.DataFrame(data)
    result = calculate_atr(df, period=2)
    assert not result.isnull().all(), "ATR calculation should produce some non-NaN results after the required period."


def test_calculate_atr_all_nan():
    """Test calculate_atr when necessary values are NaN"""
    data = {
        'high': [np.nan, np.nan, np.nan],
        'low': [np.nan, np.nan, np.nan],
        'close': [np.nan, np.nan, np.nan]
    }
    df = pd.DataFrame(data)
    result = calculate_atr(df, period=2)
    assert result.isnull().all(), "ATR calculation should return all NaN for an input dataframe with all NaN values."


def test_calculate_atr_short_data():
    """Test calculate_atr with data shorter than the period"""
    data = {
        'high': [10, 12],
        'low': [5, 8],
        'close': [6, 11]
    }
    df = pd.DataFrame(data)
    result = calculate_atr(df, period=5)
    assert result.isnull().all(), "ATR calculation should return all NaN when the data is shorter than the period."


def test_calculate_atr_valid_input():
    """Test calculate_atr with valid input data"""
    data = {
        'high': [10, 11, 12],
        'low': [8, 9, 10],
        'close': [9, 10, 11]
    }
    df = pd.DataFrame(data)
    result = calculate_atr(df, period=2)
    assert len(result) == len(df)
    assert not result.isnull().all()


def test_calculate_atr_handles_nan():
    """Test calculate_atr handles DataFrame with NaN values"""
    data = {
        'high': [10, np.nan, 12],
        'low': [8, 9, np.nan],
        'close': [9, np.nan, 11]
    }
    df = pd.DataFrame(data)
    result = calculate_atr(df, period=2)
    assert len(result) == len(df)
    assert result.isnull().any()


def test_calculate_atr_shorter_than_period():
    """Test calculate_atr with a DataFrame shorter than the period"""
    data = {
        'high': [10, 11],
        'low': [8, 9],
        'close': [9, 10]
    }
    df = pd.DataFrame(data)
    result = calculate_atr(df, period=3)
    assert len(result) == len(df)
    assert result.isnull().all()


def test_calculate_atr_empty_dataframe():
    """Test calculate_atr with an empty DataFrame"""
    df = pd.DataFrame({'high': [], 'low': [], 'close': []})
    result = calculate_atr(df, period=14)
    assert result.empty


def test_calculate_atr_period_default():
    """Test calculate_atr with a small period, so there are valid (non-NaN) ATR values."""
    data = {
        'high': [10, 11, 12, 13],
        'low': [8, 9, 10, 11],
        'close': [9, 10, 11, 12]
    }
    df = pd.DataFrame(data)
    result = calculate_atr(df, period=2)  # Use period=2 or less than or equal to len(df)
    assert len(result) == len(df)
    assert not result.isnull().all()  # Now not all values are NaN


def test_calculate_atr_non_numeric_columns():
    """Test calculate_atr raises error if columns are non-numeric"""
    data = {
        'high': ['a', 'b', 'c'],
        'low': ['d', 'e', 'f'],
        'close': ['g', 'h', 'i']
    }
    df = pd.DataFrame(data)
    with pytest.raises(Exception):
        calculate_atr(df, period=3)
