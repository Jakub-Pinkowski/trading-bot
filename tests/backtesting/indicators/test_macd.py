import numpy as np
import pandas as pd
import pytest

from app.backtesting.indicators import calculate_macd


def test_calculate_macd_with_valid_prices():
    """Test MACD calculation with valid price data"""
    prices = pd.Series([
                           44, 47, 45, 50, 55, 60, 63, 62, 64, 69, 70, 75, 80, 85, 88, 90, 92, 95, 98, 100,
                           102, 105, 108, 110, 112, 115, 118, 120, 122, 125
                       ])
    macd = calculate_macd(prices)

    # Check that the result is a DataFrame with the expected columns
    assert isinstance(macd, pd.DataFrame)
    assert all(col in macd.columns for col in ['macd_line', 'signal_line', 'histogram'])

    # Check that the first slow_period-1 values are NaN
    assert macd.iloc[:25].isna().all().all()

    # Check that values after the initial period are not NaN
    assert not macd.iloc[26:].isna().all().all()


def test_calculate_macd_with_not_enough_data():
    """Test MACD calculation with price data less than the default slow period"""
    prices = pd.Series([44, 47, 45, 50, 55])
    macd = calculate_macd(prices)

    # All values should be NaN
    assert macd.isna().all().all()


def test_calculate_macd_with_custom_parameters():
    """Test MACD calculation with custom parameters"""
    prices = pd.Series([44, 47, 45, 50, 55, 60, 63, 62, 64, 69, 70, 75, 80, 85, 88, 90, 92, 95, 98, 100])

    # Test with custom fast, slow, and signal periods
    macd = calculate_macd(prices, fast_period=5, slow_period=10, signal_period=3)

    # Check that the first slow_period-1 values are NaN
    assert macd.iloc[:9].isna().all().all()

    # Check that values after the initial period are not NaN
    assert not macd.iloc[10:].isna().all().all()


def test_calculate_macd_parameter_sensitivity():
    """Test MACD sensitivity to different parameter values"""
    prices = pd.Series([
                           44, 47, 45, 50, 55, 60, 63, 62, 64, 69, 70, 75, 80, 85, 88, 90, 92, 95, 98, 100,
                           102, 105, 108, 110, 112, 115, 118, 120, 122, 125, 128, 130, 132, 135, 138, 140
                       ])

    # Calculate MACD with default parameters
    macd_default = calculate_macd(prices)

    # Calculate MACD with faster response parameters
    macd_fast = calculate_macd(prices, fast_period=6, slow_period=13, signal_period=4)

    # Calculate MACD with slower response parameters
    macd_slow = calculate_macd(prices, fast_period=24, slow_period=52, signal_period=18)

    # Skip NaN values for comparison
    valid_idx = ~macd_default.isna().all(axis=1) & ~macd_fast.isna().all(axis=1) & ~macd_slow.isna().all(axis=1)

    if valid_idx.any():
        # Fast parameters should respond more quickly to price changes
        # This means the MACD line should have larger absolute values
        assert abs(macd_fast.loc[valid_idx, 'macd_line']).mean() > abs(macd_slow.loc[valid_idx, 'macd_line']).mean()

        # Fast parameters should also generate more crossovers (signal line crosses MACD line)
        fast_crossovers = ((macd_fast.loc[valid_idx, 'macd_line'] > macd_fast.loc[valid_idx, 'signal_line']) !=
                           (macd_fast.loc[valid_idx, 'macd_line'].shift(1) > macd_fast.loc[
                               valid_idx, 'signal_line'].shift(1))).sum()

        slow_crossovers = ((macd_slow.loc[valid_idx, 'macd_line'] > macd_slow.loc[valid_idx, 'signal_line']) !=
                           (macd_slow.loc[valid_idx, 'macd_line'].shift(1) > macd_slow.loc[
                               valid_idx, 'signal_line'].shift(1))).sum()

        # This assertion might not always hold for all price series, but it's a reasonable expectation
        # that faster parameters lead to more crossovers
        assert fast_crossovers >= slow_crossovers


def test_calculate_macd_with_uptrend():
    """Test MACD calculation with consistently increasing prices"""
    prices = pd.Series(np.linspace(100, 200, 50))  # Linear uptrend
    macd = calculate_macd(prices)

    # In an uptrend, MACD line should be above signal line
    valid_idx = ~macd.isna().all(axis=1)
    if valid_idx.any():
        # Check if MACD line is above signal line in the latter part of the data
        assert (macd.loc[valid_idx, 'macd_line'].iloc[-5:] > macd.loc[valid_idx, 'signal_line'].iloc[-5:]).all()

        # Histogram should be positive
        assert (macd.loc[valid_idx, 'histogram'].iloc[-5:] > 0).all()


def test_calculate_macd_with_downtrend():
    """Test MACD calculation with consistently decreasing prices"""
    prices = pd.Series(np.linspace(200, 100, 50))  # Linear downtrend
    macd = calculate_macd(prices)

    # In a downtrend, MACD line should be below signal line
    valid_idx = ~macd.isna().all(axis=1)
    if valid_idx.any():
        # Check if MACD line is below signal line in the latter part of the data
        assert (macd.loc[valid_idx, 'macd_line'].iloc[-5:] < macd.loc[valid_idx, 'signal_line'].iloc[-5:]).all()

        # Histogram should be negative
        assert (macd.loc[valid_idx, 'histogram'].iloc[-5:] < 0).all()


def test_calculate_macd_with_sideways_market():
    """Test MACD calculation with sideways market (oscillating prices)"""
    # Create oscillating prices
    x = np.linspace(0, 4 * np.pi, 100)
    prices = pd.Series(np.sin(x) * 10 + 100)

    macd = calculate_macd(prices)

    # In a sideways market, MACD and signal lines should cross multiple times
    valid_idx = ~macd.isna().all(axis=1)
    if valid_idx.any():
        # Count crossovers (MACD line crosses signal line)
        crossovers = ((macd.loc[valid_idx, 'macd_line'] > macd.loc[valid_idx, 'signal_line']) !=
                      (macd.loc[valid_idx, 'macd_line'].shift(1) > macd.loc[valid_idx, 'signal_line'].shift(1)))

        # There should be multiple crossovers in an oscillating market
        assert crossovers.sum() >= 2


def test_calculate_macd_with_trend_reversal():
    """Test MACD calculation with trend reversal"""
    # Create a price series with an uptrend followed by a downtrend
    uptrend = np.linspace(100, 200, 30)
    downtrend = np.linspace(200, 100, 30)
    prices = pd.Series(np.concatenate([uptrend, downtrend]))

    macd = calculate_macd(prices)

    # Find valid indices
    valid_idx = ~macd.isna().all(axis=1)

    if valid_idx.any():
        # Check that MACD line changes direction after the trend reversal
        # Find the index where the trend reverses
        trend_reversal_idx = 30  # This is where the price starts decreasing

        # Get valid indices after the initial NaN period
        valid_data_idx = np.where(valid_idx)[0]

        if len(valid_data_idx) > 0 and valid_data_idx[-1] > trend_reversal_idx:
            # Check that MACD line is decreasing during the downtrend
            downtrend_macd = macd.loc[valid_idx, 'macd_line'].iloc[trend_reversal_idx:]
            assert (
                               downtrend_macd.diff().dropna() < 0).sum() > len(downtrend_macd) * 0.7  # At least 70% of points should be decreasing

        # There should be at least one crossover (MACD line crosses signal line)
        crossovers = ((macd.loc[valid_idx, 'macd_line'] > macd.loc[valid_idx, 'signal_line']) !=
                      (macd.loc[valid_idx, 'macd_line'].shift(1) > macd.loc[valid_idx, 'signal_line'].shift(1)))

        assert crossovers.sum() >= 1


def test_calculate_macd_with_empty_prices():
    """Test MACD calculation with empty price data"""
    prices = pd.Series(dtype='float64')
    macd = calculate_macd(prices)

    # Result should be an empty DataFrame with the expected columns
    assert isinstance(macd, pd.DataFrame)
    assert all(col in macd.columns for col in ['macd_line', 'signal_line', 'histogram'])
    assert macd.empty


def test_calculate_macd_calculation_correctness():
    """Test MACD calculation correctness by manually calculating values"""
    # Create a simple price series
    prices = pd.Series([
                           100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120,
                           122, 124, 126, 128, 130, 132, 134, 136, 138, 140,
                           142, 144, 146, 148, 150, 152, 154, 156, 158, 160
                       ])

    # Calculate MACD with specific parameters for easier verification
    fast_period = 3
    slow_period = 6
    signal_period = 2
    macd = calculate_macd(prices, fast_period, slow_period, signal_period)

    # Manually calculate the expected values
    fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
    expected_macd_line = fast_ema - slow_ema
    expected_signal_line = expected_macd_line.ewm(span=signal_period, adjust=False).mean()
    expected_histogram = expected_macd_line - expected_signal_line

    # The first slow_period-1 values should be NaN
    assert macd.iloc[:slow_period - 1].isna().all().all()

    # Compare the calculated values with the expected values
    # Skip the first slow_period-1 values which are NaN
    pd.testing.assert_series_equal(
        macd.loc[slow_period - 1:, 'macd_line'],
        expected_macd_line.loc[slow_period - 1:],
        check_names=False
    )

    pd.testing.assert_series_equal(
        macd.loc[slow_period - 1:, 'signal_line'],
        expected_signal_line.loc[slow_period - 1:],
        check_names=False
    )

    pd.testing.assert_series_equal(
        macd.loc[slow_period - 1:, 'histogram'],
        expected_histogram.loc[slow_period - 1:],
        check_names=False
    )
