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


def test_calculate_macd_with_nan_values():
    """Test MACD calculation with NaN values in the input data"""
    # Create a price series with NaN values
    prices = pd.Series([
        100, 102, np.nan, 106, 108, 110, np.nan, np.nan, 116, 118, 120,
        122, 124, np.nan, 128, 130, 132, 134, 136, 138, 140,
        142, 144, 146, 148, 150, np.nan, 154, 156, 158, 160
    ])

    # Forward fill NaN values to create a clean series for comparison
    clean_prices = prices.ffill()

    # Calculate MACD for both series
    macd_with_nans = calculate_macd(prices)
    macd_clean = calculate_macd(clean_prices)

    # Check that the result is a DataFrame with the expected columns
    assert isinstance(macd_with_nans, pd.DataFrame)
    assert all(col in macd_with_nans.columns for col in ['macd_line', 'signal_line', 'histogram'])

    # The NaN values should be forward filled in the calculation
    # So the results should be similar to calculating on the clean series
    # We'll check a few points after the NaN values

    # Find valid indices (after the initial NaN period)
    valid_idx = ~macd_with_nans.isna().all(axis=1) & ~macd_clean.isna().all(axis=1)

    if valid_idx.any():
        # The values should be close but not necessarily identical due to the forward filling
        # We'll use a relative tolerance for the comparison
        np.testing.assert_allclose(
            macd_with_nans.loc[valid_idx, 'macd_line'].values,
            macd_clean.loc[valid_idx, 'macd_line'].values,
            rtol=0.1  # 10% relative tolerance
        )


def test_calculate_macd_with_market_crash():
    """Test MACD calculation during a market crash scenario"""
    # Create a price series with a stable period followed by a sharp decline (crash)
    stable_period = np.linspace(100, 105, 30)  # Stable prices around 100-105
    crash_period = np.linspace(105, 50, 20)  # Sharp decline from 105 to 50
    prices = pd.Series(np.concatenate([stable_period, crash_period]))

    macd = calculate_macd(prices)

    # Find valid indices (after the initial NaN period)
    valid_idx = ~macd.isna().all(axis=1)

    if valid_idx.any():
        # During a crash, the MACD line should fall below the signal line
        # and the histogram should become increasingly negative

        # Get the last 10 points of the valid data (during the crash)
        crash_macd = macd.loc[valid_idx].iloc[-10:]

        # MACD line should be below signal line during crash
        assert (crash_macd['macd_line'] < crash_macd['signal_line']).all()

        # Histogram should be negative during crash
        assert (crash_macd['histogram'] < 0).all()

        # MACD line should be decreasing during crash
        assert (crash_macd['macd_line'].diff().dropna() < 0).sum() >= len(crash_macd) * 0.7


def test_calculate_macd_with_market_bubble():
    """Test MACD calculation during a market bubble scenario"""
    # Create a price series with a normal growth followed by exponential growth (bubble)
    normal_growth = np.linspace(100, 120, 30)  # Normal linear growth
    bubble_growth = np.array([120, 125, 132, 142, 155, 172, 195, 225, 265, 315])  # Exponential growth
    prices = pd.Series(np.concatenate([normal_growth, bubble_growth]))

    macd = calculate_macd(prices)

    # Find valid indices (after the initial NaN period)
    valid_idx = ~macd.isna().all(axis=1)

    if valid_idx.any():
        # During a bubble, the MACD line should rise above the signal line
        # and the histogram should become increasingly positive

        # Get the last 5 points of the valid data (during the bubble)
        bubble_macd = macd.loc[valid_idx].iloc[-5:]

        # MACD line should be above signal line during bubble
        assert (bubble_macd['macd_line'] > bubble_macd['signal_line']).all()

        # Histogram should be positive during bubble
        assert (bubble_macd['histogram'] > 0).all()

        # MACD line should be increasing during bubble
        assert (bubble_macd['macd_line'].diff().dropna() > 0).all()

        # The rate of increase should accelerate (second derivative positive)
        macd_line_diff = bubble_macd['macd_line'].diff().dropna()
        assert (macd_line_diff.diff().dropna() > 0).sum() >= len(macd_line_diff.diff().dropna()) * 0.5


def test_calculate_macd_divergence():
    """Test MACD divergence detection"""
    # For this test, we'll skip the complex divergence detection and focus on a simpler approach
    # We'll create a price series where price is making higher highs but momentum is weakening

    # Create a price series with a clear divergence pattern
    # First part: strong uptrend with increasing momentum
    prices = [100, 102, 105, 109, 114, 120, 127, 135, 144, 154]

    # Second part: continued uptrend but with weakening momentum
    # Price continues to rise but at a slower rate
    prices.extend([160, 164, 167, 169, 170, 171, 172, 173, 174, 175])

    prices = pd.Series(prices)

    # Calculate MACD with parameters that will highlight the divergence
    macd = calculate_macd(prices, fast_period=3, slow_period=6, signal_period=2)

    # Find valid indices (after the initial NaN period)
    valid_idx = ~macd.isna().all(axis=1)

    if valid_idx.any():
        # Split the data into two segments: uptrend and divergence
        mid_point = len(prices) // 2

        # Get the first and second half of the valid MACD data
        first_half_macd = macd.loc[valid_idx, 'macd_line'].iloc[:mid_point]
        second_half_macd = macd.loc[valid_idx, 'macd_line'].iloc[mid_point:]

        # Get the first and second half of the price data
        first_half_price = prices.iloc[:mid_point]
        second_half_price = prices.iloc[mid_point:]

        # Check for divergence conditions:
        # 1. Price is still rising in the second half
        price_still_rising = second_half_price.iloc[-1] > second_half_price.iloc[0]

        # 2. MACD slope is decreasing in the second half compared to the first half
        # Calculate the average rate of change for each half
        if len(first_half_macd) >= 2 and len(second_half_macd) >= 2:
            first_half_slope = (first_half_macd.iloc[-1] - first_half_macd.iloc[0]) / len(first_half_macd)
            second_half_slope = (second_half_macd.iloc[-1] - second_half_macd.iloc[0]) / len(second_half_macd)

            momentum_weakening = second_half_slope < first_half_slope

            # Assert that we have a divergence
            assert price_still_rising and momentum_weakening, "Failed to detect bearish divergence"
        else:
            # If we don't have enough data points, skip the test
            pytest.skip("Not enough valid data points to test for divergence")
    else:
        # If we don't have any valid MACD values, skip the test
        pytest.skip("No valid MACD values to test for divergence")


def test_calculate_macd_zero_line_crossovers():
    """Test MACD zero line crossovers"""
    # Create a price series that will generate MACD zero line crossovers
    prices = pd.Series([
        100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120,
        118, 116, 114, 112, 110, 108, 106, 104, 102, 100,
        98, 96, 94, 92, 90, 88, 86, 84, 82, 80,
        82, 84, 86, 88, 90, 92, 94, 96, 98, 100
    ])

    macd = calculate_macd(prices, fast_period=5, slow_period=10, signal_period=3)

    # Find valid indices (after the initial NaN period)
    valid_idx = ~macd.isna().all(axis=1)

    if valid_idx.any():
        # Get the MACD line for analysis
        macd_line = macd.loc[valid_idx, 'macd_line']

        # Count zero line crossovers (MACD line crosses the zero line)
        zero_crossovers = ((macd_line > 0) != (macd_line.shift(1) > 0)).sum()

        # There should be multiple zero line crossovers in this price series
        assert zero_crossovers >= 2

        # Check if zero crossovers correspond to trend changes
        # When MACD crosses above zero, it's bullish; below zero, it's bearish
        bullish_crossovers = ((macd_line > 0) & (macd_line.shift(1) <= 0)).sum()
        bearish_crossovers = ((macd_line < 0) & (macd_line.shift(1) >= 0)).sum()

        # Both bullish and bearish crossovers should occur
        assert bullish_crossovers >= 1
        assert bearish_crossovers >= 1
