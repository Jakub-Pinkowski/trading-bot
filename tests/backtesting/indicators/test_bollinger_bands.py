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

    # Check that an upper band is always greater than middle band
    valid_idx = ~bb.isna().all(axis=1)
    assert (bb.loc[valid_idx, 'upper_band'] > bb.loc[valid_idx, 'middle_band']).all()

    # Check that lower band is always less than middle band
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
    expected_std = prices.rolling(window=period).std()
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
        # (2.5% above upper band, 2.5% below lower band)
        assert outside_upper / total_valid < 0.1  # Allow some flexibility
        assert outside_lower / total_valid < 0.1  # Allow some flexibility
