import numpy as np
import pandas as pd

from app.backtesting.indicators import calculate_bollinger_bands


def test_calculate_bollinger_bands_with_valid_prices():
    """Test Bollinger Bands calculation with valid price data"""
    prices = pd.Series([
                           44, 47, 45, 50, 55, 60, 63, 62, 64, 69, 70, 75, 80, 85, 88, 90, 92, 95, 98, 100,
                           102, 105, 108, 110, 112, 115, 118, 120, 122, 125
                       ])
    bb = calculate_bollinger_bands(prices)

    # Check that the result is a DataFrame with the expected columns
    assert isinstance(bb, pd.DataFrame)
    assert all(col in bb.columns for col in ['middle_band', 'upper_band', 'lower_band'])

    # Check that the first period-1 values are NaN
    assert bb.iloc[:19].isna().all().all()

    # Check that values after the initial period are not NaN
    assert not bb.iloc[20:].isna().all().all()

    # Check that an upper band is always greater than a middle band
    valid_idx = ~bb.isna().all(axis=1)
    assert (bb.loc[valid_idx, 'upper_band'] > bb.loc[valid_idx, 'middle_band']).all()

    # Check that a lower band is always less than a middle band
    assert (bb.loc[valid_idx, 'lower_band'] < bb.loc[valid_idx, 'middle_band']).all()


def test_calculate_bollinger_bands_with_not_enough_data():
    """Test Bollinger Bands calculation with price data less than the default period"""
    prices = pd.Series([44, 47, 45, 50, 55])
    bb = calculate_bollinger_bands(prices)

    # All values should be NaN
    assert bb.isna().all().all()


def test_calculate_bollinger_bands_with_custom_period():
    """Test Bollinger Bands calculation with a custom period"""
    prices = pd.Series([44, 47, 45, 50, 55, 60, 63, 62, 64, 69, 70, 75, 80, 85, 88, 90])

    # Test with a custom period
    bb = calculate_bollinger_bands(prices, period=10)

    # Check that the first period-1 values are NaN
    assert bb.iloc[:9].isna().all().all()

    # Check that values after the initial period are not NaN
    assert not bb.iloc[10:].isna().all().all()


def test_calculate_bollinger_bands_with_different_std_values():
    """Test Bollinger Bands calculation with different standard deviation values"""
    prices = pd.Series([
                           44, 47, 45, 50, 55, 60, 63, 62, 64, 69, 70, 75, 80, 85, 88, 90, 92, 95, 98, 100,
                           102, 105, 108, 110, 112, 115, 118, 120, 122, 125
                       ])

    # Calculate Bollinger Bands with different standard deviation values
    bb_1std = calculate_bollinger_bands(prices, num_std=1)
    bb_2std = calculate_bollinger_bands(prices, num_std=2)  # Default
    bb_3std = calculate_bollinger_bands(prices, num_std=3)

    # Find valid indices (where all calculations have non-NaN values)
    valid_idx = ~bb_1std.isna().all(axis=1) & ~bb_2std.isna().all(axis=1) & ~bb_3std.isna().all(axis=1)

    if valid_idx.any():
        # Middle band should be the same for all standard deviation values
        pd.testing.assert_series_equal(
            bb_1std.loc[valid_idx, 'middle_band'],
            bb_2std.loc[valid_idx, 'middle_band']
        )

        pd.testing.assert_series_equal(
            bb_2std.loc[valid_idx, 'middle_band'],
            bb_3std.loc[valid_idx, 'middle_band']
        )

        # Upper bands should widen as standard deviation increases
        assert (bb_3std.loc[valid_idx, 'upper_band'] > bb_2std.loc[valid_idx, 'upper_band']).all()
        assert (bb_2std.loc[valid_idx, 'upper_band'] > bb_1std.loc[valid_idx, 'upper_band']).all()

        # Lower bands should widen as standard deviation increases
        assert (bb_3std.loc[valid_idx, 'lower_band'] < bb_2std.loc[valid_idx, 'lower_band']).all()
        assert (bb_2std.loc[valid_idx, 'lower_band'] < bb_1std.loc[valid_idx, 'lower_band']).all()

        # The width of the bands (upper - lower) should increase with standard deviation
        width_1std = bb_1std.loc[valid_idx, 'upper_band'] - bb_1std.loc[valid_idx, 'lower_band']
        width_2std = bb_2std.loc[valid_idx, 'upper_band'] - bb_2std.loc[valid_idx, 'lower_band']
        width_3std = bb_3std.loc[valid_idx, 'upper_band'] - bb_3std.loc[valid_idx, 'lower_band']

        assert (width_3std > width_2std).all()
        assert (width_2std > width_1std).all()


def test_calculate_bollinger_bands_with_constant_prices():
    """Test Bollinger Bands calculation with constant prices"""
    prices = pd.Series([100] * 30)
    bb = calculate_bollinger_bands(prices)

    # Find valid indices
    valid_idx = ~bb.isna().all(axis=1)

    if valid_idx.any():
        # For constant prices, standard deviation is 0, so upper and lower bands should equal middle band
        pd.testing.assert_series_equal(
            bb.loc[valid_idx, 'upper_band'],
            bb.loc[valid_idx, 'middle_band'],
            check_names=False
        )

        pd.testing.assert_series_equal(
            bb.loc[valid_idx, 'lower_band'],
            bb.loc[valid_idx, 'middle_band'],
            check_names=False
        )


def test_calculate_bollinger_bands_with_uptrend():
    """Test Bollinger Bands calculation with consistently increasing prices"""
    prices = pd.Series(np.linspace(100, 200, 50))  # Linear uptrend
    bb = calculate_bollinger_bands(prices)

    # Find valid indices
    valid_idx = ~bb.isna().all(axis=1)

    if valid_idx.any():
        # In an uptrend, the middle band should be increasing
        assert (bb.loc[valid_idx, 'middle_band'].diff().dropna() > 0).all()

        # Upper and lower bands should also be increasing
        assert (bb.loc[valid_idx, 'upper_band'].diff().dropna() > 0).all()
        assert (bb.loc[valid_idx, 'lower_band'].diff().dropna() > 0).all()


def test_calculate_bollinger_bands_with_downtrend():
    """Test Bollinger Bands calculation with consistently decreasing prices"""
    prices = pd.Series(np.linspace(200, 100, 50))  # Linear downtrend
    bb = calculate_bollinger_bands(prices)

    # Find valid indices
    valid_idx = ~bb.isna().all(axis=1)

    if valid_idx.any():
        # In a downtrend, the middle band should be decreasing
        assert (bb.loc[valid_idx, 'middle_band'].diff().dropna() < 0).all()

        # Upper and lower bands should also be decreasing
        assert (bb.loc[valid_idx, 'upper_band'].diff().dropna() < 0).all()
        assert (bb.loc[valid_idx, 'lower_band'].diff().dropna() < 0).all()


def test_calculate_bollinger_bands_with_volatility_change():
    """Test Bollinger Bands calculation with changing volatility"""
    # Create a price series with low volatility followed by high volatility
    low_vol = np.linspace(100, 110, 25) + np.random.normal(0, 1, 25)
    high_vol = np.linspace(110, 120, 25) + np.random.normal(0, 5, 25)
    prices = pd.Series(np.concatenate([low_vol, high_vol]))

    bb = calculate_bollinger_bands(prices)

    # Find valid indices
    valid_idx = ~bb.isna().all(axis=1)

    if valid_idx.any() and len(valid_idx) >= 30:
        # Calculate band width for low volatility and high volatility periods
        low_vol_width = bb.loc[valid_idx, 'upper_band'].iloc[0:10] - bb.loc[valid_idx, 'lower_band'].iloc[0:10]
        high_vol_width = bb.loc[valid_idx, 'upper_band'].iloc[-10:] - bb.loc[valid_idx, 'lower_band'].iloc[-10:]

        # Band width should be greater during high volatility
        assert high_vol_width.mean() > low_vol_width.mean()


def test_calculate_bollinger_bands_with_empty_prices():
    """Test Bollinger Bands calculation with empty price data"""
    prices = pd.Series(dtype='float64')
    bb = calculate_bollinger_bands(prices)

    # Result should be an empty DataFrame with the expected columns
    assert isinstance(bb, pd.DataFrame)
    assert all(col in bb.columns for col in ['middle_band', 'upper_band', 'lower_band'])
    assert bb.empty


def test_calculate_bollinger_bands_calculation_correctness():
    """Test Bollinger Bands calculation correctness by manually calculating values"""
    # Create a simple price series
    prices = pd.Series([
                           100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120,
                           122, 124, 126, 128, 130, 132, 134, 136, 138, 140
                       ])

    # Calculate Bollinger Bands with specific parameters for easier verification
    period = 5
    num_std = 2
    bb = calculate_bollinger_bands(prices, period=period, num_std=num_std)

    # Manually calculate the expected values
    expected_middle_band = prices.rolling(window=period).mean()
    expected_std = prices.rolling(window=period).std(ddof=0)
    expected_upper_band = expected_middle_band + (expected_std * num_std)
    expected_lower_band = expected_middle_band - (expected_std * num_std)

    # Compare the calculated values with the expected values
    # Skip the first period-1 values which are NaN
    pd.testing.assert_series_equal(
        bb.loc[period - 1:, 'middle_band'],
        expected_middle_band.loc[period - 1:],
        check_names=False
    )

    pd.testing.assert_series_equal(
        bb.loc[period - 1:, 'upper_band'],
        expected_upper_band.loc[period - 1:],
        check_names=False
    )

    pd.testing.assert_series_equal(
        bb.loc[period - 1:, 'lower_band'],
        expected_lower_band.loc[period - 1:],
        check_names=False
    )


def test_calculate_bollinger_bands_price_relationship():
    """Test relationship between prices and Bollinger Bands"""
    # Create a price series with some volatility
    np.random.seed(42)  # For reproducibility
    base = np.linspace(100, 150, 50)
    noise = np.random.normal(0, 10, 50)
    prices = pd.Series(base + noise)

    bb = calculate_bollinger_bands(prices)

    # Find valid indices
    valid_idx = ~bb.isna().all(axis=1)

    if valid_idx.any():
        # Count how many times prices are outside the bands
        outside_upper = (prices.loc[valid_idx] > bb.loc[valid_idx, 'upper_band']).sum()
        outside_lower = (prices.loc[valid_idx] < bb.loc[valid_idx, 'lower_band']).sum()
        total_valid = valid_idx.sum()

        # With 2 standard deviations, approximately 5% of prices should be outside the bands
        # (2.5% above an upper band, 2.5% below a lower band)
        assert outside_upper / total_valid < 0.1  # Allow some flexibility
        assert outside_lower / total_valid < 0.1  # Allow some flexibility


def test_calculate_bollinger_bands_with_nan_values():
    """Test Bollinger Bands calculation with NaN values in the input data"""
    # Create a price series with NaN values
    prices = pd.Series([
                           100, 102, np.nan, 106, 108, 110, np.nan, np.nan, 116, 118, 120,
                           122, 124, np.nan, 128, 130, 132, 134, 136, 138, 140,
                           142, 144, 146, 148, 150, np.nan, 154, 156, 158, 160
                       ])

    # Forward fill NaN values to create a clean series for comparison
    clean_prices = prices.ffill()

    # Calculate Bollinger Bands for both series
    bb_with_nans = calculate_bollinger_bands(prices)
    bb_clean = calculate_bollinger_bands(clean_prices)

    # Check that the result is a DataFrame with the expected columns
    assert isinstance(bb_with_nans, pd.DataFrame)
    assert all(col in bb_with_nans.columns for col in ['middle_band', 'upper_band', 'lower_band'])

    # The NaN values should be forward filled in the calculation,
    # So the results should be similar to calculating on the clean series

    # Find valid indices (after the initial NaN period)
    valid_idx = ~bb_with_nans.isna().all(axis=1) & ~bb_clean.isna().all(axis=1)

    if valid_idx.any():
        # The values should be close but not necessarily identical due to the forward filling
        # We'll use a relative tolerance for the comparison
        np.testing.assert_allclose(
            bb_with_nans.loc[valid_idx, 'middle_band'].values,
            bb_clean.loc[valid_idx, 'middle_band'].values,
            rtol=0.1  # 10% relative tolerance
        )


def test_calculate_bollinger_bands_with_market_crash():
    """Test Bollinger Bands calculation during a market crash scenario"""
    # Create a price series with a stable period followed by a sharp decline (crash)
    stable_period = np.linspace(100, 105, 30)  # Stable prices around 100-105
    crash_period = np.linspace(105, 50, 20)  # Sharp decline from 105 to 50
    prices = pd.Series(np.concatenate([stable_period, crash_period]))

    bb = calculate_bollinger_bands(prices)

    # Find valid indices (after the initial NaN period)
    valid_idx = ~bb.isna().all(axis=1)

    if valid_idx.any():
        # During a crash, the bands should generally move downward
        # and the width might increase due to higher volatility

        # Get the last 10 points of the valid data (during the crash)
        crash_bb = bb.loc[valid_idx].iloc[-10:]

        # Middle band should be decreasing during crash
        # We'll check that the overall trend is downward
        assert crash_bb['middle_band'].iloc[-1] < crash_bb['middle_band'].iloc[0]

        # Upper and lower bands should also be decreasing overall
        # Note: During the initial phase of a crash, the upper band might temporarily increase
        # due to increased volatility before the downward trend takes over
        assert crash_bb['upper_band'].iloc[-1] < crash_bb['upper_band'].iloc[0]
        assert crash_bb['lower_band'].iloc[-1] < crash_bb['lower_band'].iloc[0]

        # Check that most of the points are decreasing
        # At least 70% of the points should be decreasing
        middle_band_decreasing = (crash_bb['middle_band'].diff().dropna() < 0).mean() >= 0.7
        lower_band_decreasing = (crash_bb['lower_band'].diff().dropna() < 0).mean() >= 0.7

        assert middle_band_decreasing, "Middle band should be decreasing for most points during crash"
        assert lower_band_decreasing, "Lower band should be decreasing for most points during crash"

        # Check if prices are near or below the lower band during crash
        # This is a common occurrence during market crashes
        crash_prices = prices.iloc[-10:]
        lower_band = crash_bb['lower_band']

        # Calculate how close prices are to the lower band
        # During a crash, prices often approach or break below the lower band
        price_to_lower_ratio = (crash_prices / lower_band).mean()

        # Prices should be close to or below the lower band (ratio close to or less than 1)
        assert price_to_lower_ratio <= 1.1  # Allow a reasonable margin


def test_calculate_bollinger_bands_with_market_bubble():
    """Test Bollinger Bands calculation during a market bubble scenario"""
    # Create a price series with a normal growth followed by exponential growth (bubble)
    normal_growth = np.linspace(100, 120, 30)  # Normal linear growth
    bubble_growth = np.array([120, 125, 132, 142, 155, 172, 195, 225, 265, 315])  # Exponential growth
    prices = pd.Series(np.concatenate([normal_growth, bubble_growth]))

    bb = calculate_bollinger_bands(prices)

    # Find valid indices (after the initial NaN period)
    valid_idx = ~bb.isna().all(axis=1)

    if valid_idx.any():
        # During a bubble, the bands should move upward rapidly
        # and the width might increase due to higher volatility

        # Get the last 5 points of the valid data (during the bubble)
        bubble_bb = bb.loc[valid_idx].iloc[-5:]

        # Middle band should be increasing rapidly during bubble
        middle_band_diff = bubble_bb['middle_band'].diff().dropna()
        assert (middle_band_diff > 0).all()

        # The rate of increase should accelerate (second derivative positive)
        assert (middle_band_diff.diff().dropna() > 0).sum() >= len(middle_band_diff.diff().dropna()) * 0.5

        # Check if prices are near or above the upper band during bubble
        # This is a common occurrence during market bubbles
        bubble_prices = prices.iloc[-5:]
        upper_band = bubble_bb['upper_band']

        # Calculate how close prices are to the upper band
        # During a bubble, prices often approach or break above the upper band
        price_to_upper_ratio = (bubble_prices / upper_band).mean()

        # Prices should be close to or above the upper band (ratio close to or greater than 1)
        assert price_to_upper_ratio >= 0.95  # Allow a small margin


def test_calculate_bollinger_bands_squeeze():
    """Test Bollinger Bands squeeze (when bands narrow significantly)"""
    # Create a price series with low volatility (to create a squeeze)
    # followed by higher volatility (expansion after squeeze)
    squeeze_period = np.linspace(100, 102, 30) + np.random.normal(0, 0.1, 30)  # Very low volatility
    expansion_period = np.linspace(102, 120, 20) + np.random.normal(0, 5, 20)  # Higher volatility
    prices = pd.Series(np.concatenate([squeeze_period, expansion_period]))

    bb = calculate_bollinger_bands(prices)

    # Find valid indices (after the initial NaN period)
    valid_idx = ~bb.isna().all(axis=1)

    if valid_idx.any():
        # Calculate Bollinger Band width (upper - lower)
        bb_width = bb.loc[valid_idx, 'upper_band'] - bb.loc[valid_idx, 'lower_band']

        # During squeeze, the width should be very small
        squeeze_width = bb_width.iloc[5:25]  # Middle of the squeeze period

        # During expansion, the width should increase
        expansion_width = bb_width.iloc[-10:]  # End of the expansion period

        # The width during expansion should be significantly larger than during squeeze
        assert expansion_width.mean() > squeeze_width.mean() * 2

        # The minimum width during squeeze should be very small
        # compared to the maximum width during expansion
        assert squeeze_width.min() < expansion_width.max() * 0.3


def test_calculate_bollinger_bands_width_as_volatility_indicator():
    """Test Bollinger Bands width as a volatility indicator"""
    # Create a price series with varying volatility
    np.random.seed(42)  # For reproducibility

    # Low-volatility period
    low_vol_prices = np.linspace(100, 110, 30) + np.random.normal(0, 1, 30)

    # Medium volatility period
    med_vol_prices = np.linspace(110, 120, 30) + np.random.normal(0, 3, 30)

    # High-volatility period
    high_vol_prices = np.linspace(120, 130, 30) + np.random.normal(0, 7, 30)

    # Combine all periods
    prices = pd.Series(np.concatenate([low_vol_prices, med_vol_prices, high_vol_prices]))

    bb = calculate_bollinger_bands(prices)

    # Find valid indices (after the initial NaN period)
    valid_idx = ~bb.isna().all(axis=1)

    if valid_idx.any() and len(bb.loc[valid_idx]) >= 60:
        # Calculate Bollinger Band width (upper - lower)
        bb_width = bb.loc[valid_idx, 'upper_band'] - bb.loc[valid_idx, 'lower_band']

        # Calculate the average width for each volatility period
        low_vol_width = bb_width.iloc[0:20].mean()
        med_vol_width = bb_width.iloc[30:50].mean()
        high_vol_width = bb_width.iloc[-20:].mean()

        # Bandwidth should increase with volatility
        assert low_vol_width < med_vol_width
        assert med_vol_width < high_vol_width

        # Calculate the correlation between rolling volatility and bandwidth
        # First, calculate rolling standard deviation of prices as a measure of volatility
        rolling_vol = prices.rolling(window=20).std(ddof=0)

        # Calculate the correlation between rolling volatility and bandwidth
        # (only for valid indices where both are defined)
        corr_indices = valid_idx & ~rolling_vol.isna()
        if corr_indices.sum() > 10:  # Ensure we have enough points for correlation
            correlation = np.corrcoef(rolling_vol.loc[corr_indices], bb_width.loc[corr_indices])[0, 1]

            # There should be a strong positive correlation between volatility and bandwidth
            assert correlation > 0.7
