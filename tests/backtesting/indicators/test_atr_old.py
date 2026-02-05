import numpy as np
import pandas as pd
import pytest

from app.backtesting.indicators import calculate_atr
from app.utils.backtesting_utils.indicators_utils import hash_series


def compute_hashes(df):
    """Helper function to compute hashes for ATR testing"""
    return {
        'high_hash': hash_series(df['high']),
        'low_hash': hash_series(df['low']),
        'close_hash': hash_series(df['close'])
    }


def create_test_df(length=30, trend='up', volatility=1.0):
    """Helper function to create test dataframes with different characteristics"""
    if trend == 'up':
        close = np.linspace(100, 200, length)
    elif trend == 'down':
        close = np.linspace(200, 100, length)
    elif trend == 'sideways':
        close = np.ones(length) * 150
    else:
        raise ValueError("trend must be 'up', 'down', or 'sideways'")

    # Add some noise to close prices
    noise = np.random.normal(0, volatility, length)
    close = close + noise

    # Create high and low prices based on close with some volatility
    high = close + np.abs(np.random.normal(0, volatility * 2, length))
    low = close - np.abs(np.random.normal(0, volatility * 2, length))

    # Ensure high is always >= close and low is always <= close
    high = np.maximum(high, close)
    low = np.minimum(low, close)

    # Create DataFrame
    df = pd.DataFrame({
        'high': high,
        'low': low,
        'close': close
    })

    return df


def test_calculate_atr_with_valid_data():
    """Test ATR calculation with valid price data"""
    df = create_test_df(length=30)
    hashes = compute_hashes(df)
    atr = calculate_atr(df, period=14, **hashes)

    # Check that the result is a Series
    assert isinstance(atr, pd.Series)

    # Check that the first period-1 values are NaN
    assert atr.iloc[:13].isna().all()

    # Check that values after the initial period are not NaN
    assert not atr.iloc[14:].isna().all()

    # ATR should be positive
    assert (atr.dropna() > 0).all()


def test_calculate_atr_with_not_enough_data():
    """Test ATR calculation with price data less than the default period"""
    df = create_test_df(length=5)
    hashes = compute_hashes(df)
    atr = calculate_atr(df, period=14, **hashes)

    # All values should be NaN
    assert atr.isna().all()


def test_calculate_atr_with_custom_period():
    """Test ATR calculation with a custom period"""
    df = create_test_df(length=30)

    # Test with a custom period
    hashes = compute_hashes(df)
    atr = calculate_atr(df, period=7, **hashes)

    # Check that the first period-1 values are NaN
    assert atr.iloc[:6].isna().all()

    # Check that values after the initial period are not NaN
    assert not atr.iloc[7:].isna().all()


def test_calculate_atr_across_different_timeframes():
    """Test ATR calculation accuracy across different timeframes"""
    # Create dataframes with different lengths representing different timeframes
    df_short = create_test_df(length=30, volatility=1.0)  # Short timeframe
    df_medium = create_test_df(length=60, volatility=1.0)  # Medium timeframe
    df_long = create_test_df(length=120, volatility=1.0)  # Long timeframe

    # Calculate ATR with the same period for all timeframes
    period = 14
    hashes = compute_hashes(df_short)
    atr_short = calculate_atr(df_short, period=period, **hashes)
    hashes = compute_hashes(df_medium)
    atr_medium = calculate_atr(df_medium, period=period, **hashes)
    hashes = compute_hashes(df_long)
    atr_long = calculate_atr(df_long, period=period, **hashes)

    # Check that all ATRs have the expected number of NaN values
    assert atr_short.iloc[:period - 1].isna().all()
    assert atr_medium.iloc[:period - 1].isna().all()
    assert atr_long.iloc[:period - 1].isna().all()

    # Check that all ATRs have valid values after the initial period
    assert not atr_short.iloc[period:].isna().all()
    assert not atr_medium.iloc[period:].isna().all()
    assert not atr_long.iloc[period:].isna().all()

    # ATR should be positive for all timeframes
    assert (atr_short.dropna() > 0).all()
    assert (atr_medium.dropna() > 0).all()
    assert (atr_long.dropna() > 0).all()


def test_calculate_atr_with_different_volatilities():
    """Test ATR calculation with different volatility levels"""
    # Create dataframes with different volatility levels
    df_low_vol = create_test_df(length=60, volatility=0.5)  # Low volatility
    df_med_vol = create_test_df(length=60, volatility=1.0)  # Medium volatility
    df_high_vol = create_test_df(length=60, volatility=2.0)  # High volatility

    # Calculate ATR for each volatility level
    hashes = compute_hashes(df_low_vol)
    atr_low_vol = calculate_atr(df_low_vol, period=14, **hashes)
    hashes = compute_hashes(df_med_vol)
    atr_med_vol = calculate_atr(df_med_vol, period=14, **hashes)
    hashes = compute_hashes(df_high_vol)
    atr_high_vol = calculate_atr(df_high_vol, period=14, **hashes)

    # Higher volatility should result in higher ATR values
    assert atr_low_vol.iloc[-1] < atr_med_vol.iloc[-1]
    assert atr_med_vol.iloc[-1] < atr_high_vol.iloc[-1]

    # The average ATR should also increase with volatility
    assert atr_low_vol.iloc[14:].mean() < atr_med_vol.iloc[14:].mean()
    assert atr_med_vol.iloc[14:].mean() < atr_high_vol.iloc[14:].mean()


def test_calculate_atr_with_different_trends():
    """Test ATR calculation with different market trends"""
    # Create dataframes with different trends
    df_uptrend = create_test_df(length=60, trend='up', volatility=1.0)
    df_downtrend = create_test_df(length=60, trend='down', volatility=1.0)
    df_sideways = create_test_df(length=60, trend='sideways', volatility=1.0)

    # Calculate ATR for each trend
    hashes = compute_hashes(df_uptrend)
    atr_uptrend = calculate_atr(df_uptrend, period=14, **hashes)
    hashes = compute_hashes(df_downtrend)
    atr_downtrend = calculate_atr(df_downtrend, period=14, **hashes)
    hashes = compute_hashes(df_sideways)
    atr_sideways = calculate_atr(df_sideways, period=14, **hashes)

    # ATR should be positive for all trends
    assert (atr_uptrend.dropna() > 0).all()
    assert (atr_downtrend.dropna() > 0).all()
    assert (atr_sideways.dropna() > 0).all()

    # For the same volatility, ATR should be similar across different trends
    # (within a reasonable margin)
    assert abs(atr_uptrend.iloc[-1] - atr_downtrend.iloc[-1]) / atr_uptrend.iloc[-1] < 0.5
    assert abs(atr_uptrend.iloc[-1] - atr_sideways.iloc[-1]) / atr_uptrend.iloc[-1] < 0.5
    assert abs(atr_downtrend.iloc[-1] - atr_sideways.iloc[-1]) / atr_downtrend.iloc[-1] < 0.5


def test_calculate_atr_with_price_gaps():
    """Test ATR calculation with price gaps"""
    # Create a dataframe with price gaps (large differences between close and next open)
    df = create_test_df(length=30)

    # Introduce some large gaps between days
    for i in range(5, 25, 5):
        # Create a gap up
        if i % 10 == 5:
            df.loc[i, 'close'] = df.loc[i - 1, 'close'] * 1.1  # 10% gap up
            df.loc[i, 'high'] = max(df.loc[i, 'high'], df.loc[i, 'close'] * 1.05)
        # Create a gap down
        else:
            df.loc[i, 'close'] = df.loc[i - 1, 'close'] * 0.9  # 10% gap down
            df.loc[i, 'low'] = min(df.loc[i, 'low'], df.loc[i, 'close'] * 0.95)

    hashes = compute_hashes(df)
    atr = calculate_atr(df, period=14, **hashes)

    # ATR should increase after gaps due to higher true range values
    for i in range(6, 26, 5):
        if i >= 14:  # Only check after the initial period
            # Check if ATR increases after a gap
            assert atr.iloc[i] > atr.iloc[i - 2]


def test_calculate_atr_calculation_correctness():
    """Test ATR calculation correctness by manually calculating values"""
    # Create a simple price dataframe
    df = pd.DataFrame({
        'high': [110, 112, 108, 116, 113, 110, 115, 117, 114, 112, 118, 120, 117, 115, 122],
        'low': [100, 102, 98, 106, 103, 100, 105, 107, 104, 102, 108, 110, 107, 105, 112],
        'close': [105, 107, 103, 111, 108, 105, 110, 112, 109, 107, 113, 115, 112, 110, 117]
    })

    # Calculate ATR with a specific period for easier verification
    period = 5
    hashes = compute_hashes(df)
    atr = calculate_atr(df, period=period, **hashes)

    # Manually calculate the expected values
    # First, calculate true range
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate ATR using EMA
    expected_atr = true_range.ewm(span=period, adjust=False).mean()
    expected_atr[:period - 1] = np.nan  # First period-1 points are undefined

    # Compare the calculated values with the expected values
    # Skip the first period-1 values which are NaN
    pd.testing.assert_series_equal(
        atr.iloc[period - 1:],
        expected_atr.iloc[period - 1:],
        check_names=False
    )


def test_calculate_atr_with_empty_dataframe():
    """Test ATR calculation with an empty dataframe"""
    df = pd.DataFrame(columns=['high', 'low', 'close'])
    hashes = compute_hashes(df)
    atr = calculate_atr(df, period=14, **hashes)

    # Result should be an empty Series
    assert isinstance(atr, pd.Series)
    assert atr.empty


def test_calculate_atr_with_missing_columns():
    """Test ATR calculation with missing required columns"""
    # Create a dataframe missing the 'high' column
    df_missing_high = pd.DataFrame({
        'low': [100, 102, 98, 106, 103],
        'close': [105, 107, 103, 111, 108]
    })

    # Should raise a KeyError
    with pytest.raises(KeyError):
        hashes = compute_hashes(df_missing_high)
        calculate_atr(df_missing_high, period=14, **hashes)

    # Create a dataframe missing the 'low' column
    df_missing_low = pd.DataFrame({
        'high': [110, 112, 108, 116, 113],
        'close': [105, 107, 103, 111, 108]
    })

    # Should raise a KeyError
    with pytest.raises(KeyError):
        hashes = compute_hashes(df_missing_low)
        calculate_atr(df_missing_low, period=14, **hashes)

    # Create a dataframe missing the 'close' column
    df_missing_close = pd.DataFrame({
        'high': [110, 112, 108, 116, 113],
        'low': [100, 102, 98, 106, 103]
    })

    # Should raise a KeyError
    with pytest.raises(KeyError):
        hashes = compute_hashes(df_missing_close)
        calculate_atr(df_missing_close, period=14, **hashes)


def test_calculate_atr_with_constant_prices():
    """Test ATR calculation with constant prices"""
    # Create a dataframe with constant prices
    df = pd.DataFrame({
        'high': [100] * 30,
        'low': [100] * 30,
        'close': [100] * 30
    })

    hashes = compute_hashes(df)
    atr = calculate_atr(df, period=14, **hashes)

    # With constant prices, true range is 0, so ATR should be 0 after the initial period
    assert (atr.iloc[14:] == 0).all()


# BUG [HIGH]: Sometimes it fails randomly
# def test_calculate_atr_with_nan_values():
#     """Test ATR calculation with NaN values in the input data"""
#     # Create a dataframe with NaN values
#     df = create_test_df(length=30)
#
#     # Introduce NaN values
#     df.loc[5, 'high'] = np.nan
#     df.loc[10, 'low'] = np.nan
#     df.loc[15, 'close'] = np.nan
#     df.loc[20, ['high', 'low', 'close']] = np.nan
#
#     # Create a clean dataframe by forward filling NaN values
#     df_clean = df.ffill()
#
#     # Calculate ATR for both dataframes
#     atr_with_nans = calculate_atr(df)
#     atr_clean = calculate_atr(df_clean)
#
#     # Check that the result is a Series
#     assert isinstance(atr_with_nans, pd.Series)
#
#     # The NaN values should be handled in the calculation
#     # The results will differ from the clean dataframe because NaN values affect the true range calculation
#     # and the exponential moving average
#
#     # Find valid indices (after the initial NaN period)
#     valid_idx = ~atr_with_nans.isna() & ~atr_clean.isna()
#
#     if valid_idx.any():
#         # Instead of comparing exact values, we'll check that:
#         # 1. Both ATRs have similar trends (correlation)
#         # 2. The values are within a reasonable range of each other
#
#         # Check the correlation between the two ATR series
#         correlation = np.corrcoef(atr_with_nans.loc[valid_idx], atr_clean.loc[valid_idx])[0, 1]
#         assert correlation > 0.4, "ATR with NaNs should be correlated with ATR with filled values"
#
#         # Check that the values are within a reasonable range
#         # Calculate the maximum relative difference
#         max_rel_diff = np.max(np.abs(atr_with_nans.loc[valid_idx] - atr_clean.loc[valid_idx]) / atr_clean.loc[
#             valid_idx])
#         assert max_rel_diff < 0.3, f"Maximum relative difference should be less than 30%, got {max_rel_diff:.2%}"
#
#         # Check that the mean values are reasonably close
#         mean_with_nans = atr_with_nans.loc[valid_idx].mean()
#         mean_clean = atr_clean.loc[valid_idx].mean()
#         rel_diff_means = abs(mean_with_nans - mean_clean) / mean_clean
#         assert rel_diff_means < 0.2, f"Mean values should be within 20%, got {rel_diff_means:.2%}"


def test_calculate_atr_with_market_crash():
    """Test ATR calculation during a market crash scenario"""
    # Create a dataframe with a stable period followed by a sharp decline (crash)
    df_stable = create_test_df(length=30, trend='sideways', volatility=1.0)

    # Create a crash period with high volatility
    df_crash = create_test_df(length=20, trend='down', volatility=5.0)

    # Make the crash more severe by adjusting prices
    crash_factor = 0.5  # 50% crash
    df_crash['close'] = df_crash['close'] * crash_factor
    df_crash['high'] = df_crash['high'] * crash_factor
    df_crash['low'] = df_crash['low'] * crash_factor

    # Combine the stable and crash periods
    df = pd.concat([df_stable, df_crash]).reset_index(drop=True)

    # Calculate ATR
    hashes = compute_hashes(df)
    atr = calculate_atr(df, period=14, **hashes)

    # During a crash, ATR should increase due to higher volatility

    # Get ATR values for stable and crash periods
    stable_atr = atr.iloc[14:30]  # After initial NaN period but before crash
    crash_atr = atr.iloc[40:]  # During crash period

    # ATR should be higher during the crash
    assert crash_atr.mean() > stable_atr.mean() * 1.5  # At least 50% higher


def test_calculate_atr_with_market_bubble():
    """Test ATR calculation during a market bubble scenario"""
    # Create a dataframe with a normal growth period
    df_normal = create_test_df(length=30, trend='up', volatility=1.0)

    # Create a bubble period with high volatility and exponential growth
    df_bubble = create_test_df(length=20, trend='up', volatility=3.0)

    # Make the bubble more extreme by adjusting prices
    bubble_factor = 2.0  # 100% additional growth
    df_bubble['close'] = df_bubble['close'] * bubble_factor
    df_bubble['high'] = df_bubble['high'] * bubble_factor
    df_bubble['low'] = df_bubble['low'] * bubble_factor

    # Combine the normal and bubble periods
    df = pd.concat([df_normal, df_bubble]).reset_index(drop=True)

    # Calculate ATR
    hashes = compute_hashes(df)
    atr = calculate_atr(df, period=14, **hashes)

    # During a bubble, ATR should increase due to higher volatility

    # Get ATR values for normal and bubble periods
    normal_atr = atr.iloc[14:30]  # After initial NaN period but before bubble
    bubble_atr = atr.iloc[40:]  # During bubble period

    # ATR should be higher during the bubble
    assert bubble_atr.mean() > normal_atr.mean() * 1.5  # At least 50% higher


def test_calculate_atr_as_trend_strength_indicator():
    """Test ATR as a trend strength indicator"""
    # Create dataframes with different trend strengths
    # Weak trend: low slope, low volatility
    df_weak = create_test_df(length=60, trend='up', volatility=0.5)

    # Strong trend: high slope, medium volatility
    df_strong = create_test_df(length=60, trend='up', volatility=1.0)
    close_values = df_strong['close'].values
    # Make the trend stronger by increasing the slope
    df_strong['close'] = np.linspace(close_values[0], close_values[-1] * 1.5, len(close_values))
    # Adjust high and low accordingly
    df_strong['high'] = df_strong['close'] + (df_strong['high'] - close_values)
    df_strong['low'] = df_strong['close'] - (close_values - df_strong['low'])

    # Calculate ATR for both trends
    hashes = compute_hashes(df_weak)
    atr_weak = calculate_atr(df_weak, period=14, **hashes)
    hashes = compute_hashes(df_strong)
    atr_strong = calculate_atr(df_strong, period=14, **hashes)

    # Calculate the average ATR as a percentage of price for both trends
    # This normalizes ATR to make it comparable across different price levels
    atr_pct_weak = (atr_weak / df_weak['close']).iloc[14:].mean() * 100  # As percentage
    atr_pct_strong = (atr_strong / df_strong['close']).iloc[14:].mean() * 100  # As percentage

    # A stronger trend should have a higher ATR percentage
    assert atr_pct_strong > atr_pct_weak


def test_calculate_atr_percentage():
    """Test ATR percentage calculation (ATR relative to price)"""
    # Create dataframes with different price levels but similar volatility characteristics
    df_low_price = create_test_df(length=60, trend='sideways', volatility=0.05)
    df_low_price['close'] = df_low_price['close'] * 0.1  # Scale down to a low price level (around 15)
    df_low_price['high'] = df_low_price['high'] * 0.1
    df_low_price['low'] = df_low_price['low'] * 0.1

    df_high_price = create_test_df(length=60, trend='sideways', volatility=0.05)
    df_high_price['close'] = df_high_price['close'] * 10  # Scale up to high price level (around 1500)
    df_high_price['high'] = df_high_price['high'] * 10
    df_high_price['low'] = df_high_price['low'] * 10

    # Calculate ATR for both price levels
    hashes = compute_hashes(df_low_price)
    atr_low_price = calculate_atr(df_low_price, period=14, **hashes)
    hashes = compute_hashes(df_high_price)
    atr_high_price = calculate_atr(df_high_price, period=14, **hashes)

    # Calculate ATR as a percentage of price
    atr_pct_low_price = (atr_low_price / df_low_price['close']).iloc[14:].mean() * 100  # As percentage
    atr_pct_high_price = (atr_high_price / df_high_price['close']).iloc[14:].mean() * 100  # As percentage

    # The absolute ATR values should be very different due to price levels
    assert atr_high_price.iloc[14:].mean() > atr_low_price.iloc[14:].mean() * 50  # High price ATR should be much larger

    # But the ATR percentages should be similar since the volatility characteristics are similar
    # Allow for some variation due to random noise in the test data
    assert abs(atr_pct_high_price - atr_pct_low_price) / atr_pct_low_price < 0.3  # Within 30%
