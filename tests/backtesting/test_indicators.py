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
    """Test EMA calculation with an invalid period"""
    prices = pd.Series([10, 12, 14])
    with pytest.raises(ValueError):
        calculate_ema(prices, period=0)


def test_calculate_ema_non_numeric_data():
    """Test EMA calculation with non-numeric data"""
    prices = pd.Series(["a", "b", "c"])
    with pytest.raises(pd.errors.DataError):
        calculate_ema(prices)