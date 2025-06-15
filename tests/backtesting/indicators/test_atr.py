import numpy as np
import pandas as pd
import pytest

from app.backtesting.indicators import calculate_atr


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
    atr = calculate_atr(df)

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
    atr = calculate_atr(df)

    # All values should be NaN
    assert atr.isna().all()


def test_calculate_atr_with_custom_period():
    """Test ATR calculation with a custom period"""
    df = create_test_df(length=30)

    # Test with a custom period
    atr = calculate_atr(df, period=7)

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
    atr_short = calculate_atr(df_short, period=period)
    atr_medium = calculate_atr(df_medium, period=period)
    atr_long = calculate_atr(df_long, period=period)

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
    atr_low_vol = calculate_atr(df_low_vol)
    atr_med_vol = calculate_atr(df_med_vol)
    atr_high_vol = calculate_atr(df_high_vol)

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
    atr_uptrend = calculate_atr(df_uptrend)
    atr_downtrend = calculate_atr(df_downtrend)
    atr_sideways = calculate_atr(df_sideways)

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

    atr = calculate_atr(df)

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
    atr = calculate_atr(df, period=period)

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
    atr = calculate_atr(df)

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
        calculate_atr(df_missing_high)

    # Create a dataframe missing the 'low' column
    df_missing_low = pd.DataFrame({
        'high': [110, 112, 108, 116, 113],
        'close': [105, 107, 103, 111, 108]
    })

    # Should raise a KeyError
    with pytest.raises(KeyError):
        calculate_atr(df_missing_low)

    # Create a dataframe missing the 'close' column
    df_missing_close = pd.DataFrame({
        'high': [110, 112, 108, 116, 113],
        'low': [100, 102, 98, 106, 103]
    })

    # Should raise a KeyError
    with pytest.raises(KeyError):
        calculate_atr(df_missing_close)


def test_calculate_atr_with_constant_prices():
    """Test ATR calculation with constant prices"""
    # Create a dataframe with constant prices
    df = pd.DataFrame({
        'high': [100] * 30,
        'low': [100] * 30,
        'close': [100] * 30
    })

    atr = calculate_atr(df)

    # With constant prices, true range is 0, so ATR should be 0 after the initial period
    assert (atr.iloc[14:] == 0).all()
