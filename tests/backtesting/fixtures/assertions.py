"""
Custom assertion functions for backtesting tests.

Provides cross-module assertion for validating indicator structure and values.
Module-specific assertions are in their respective test_utils files:
- indicators/indicator_test_utils.py
- strategies/strategy_test_utils.py
"""
import numpy as np
import pandas as pd


# ==================== Indicator Validation ====================

def assert_valid_indicator(series, name, min_val=None, max_val=None, allow_nan=True):
    """
    Assert indicator series contains valid values within expected range.

    Validates that an indicator series (RSI, EMA, MACD, etc.) has proper structure,
    contains values within specified bounds, and handles NaN values appropriately.

    Used across both indicator and strategy tests.

    Args:
        series: Pandas Series containing indicator values
        name: Indicator name for error messages (e.g., 'RSI', 'EMA(9)')
        min_val: Minimum valid value (None = no minimum check)
        max_val: Maximum valid value (None = no maximum check)
        allow_nan: If True, allow NaN values (common for initial periods)

    Raises:
        AssertionError: If any validation check fails

    Examples:
        assert_valid_indicator(rsi, 'RSI', min_val=0, max_val=100)
        assert_valid_indicator(ema, 'EMA(9)', min_val=0)
    """
    # Check series type
    assert isinstance(series, pd.Series), f"{name} must be a pandas Series"

    # Check series is not empty
    assert len(series) > 0, f"{name} series is empty"

    # Get valid (non-NaN) values for range checking
    valid_values = series.dropna()

    # Check if we have any valid values
    if not allow_nan:
        assert len(valid_values) == len(series), f"{name} contains NaN values"
    else:
        assert len(valid_values) > 0, f"{name} contains only NaN values"

    # Check minimum value
    if min_val is not None:
        min_actual = valid_values.min()
        assert min_actual >= min_val, \
            f"{name} minimum value {min_actual} is below expected minimum {min_val}"

    # Check maximum value
    if max_val is not None:
        max_actual = valid_values.max()
        assert max_actual <= max_val, \
            f"{name} maximum value {max_actual} is above expected maximum {max_val}"

    # Check for infinite values
    assert not np.isinf(valid_values).any(), f"{name} contains infinite values"


def assert_indicator_varies(series, name, min_std=0.1):
    """
    Assert indicator shows variation (not constant).

    Validates that an indicator actually responds to price changes rather than
    remaining constant. Useful for detecting calculation errors.

    Used across both indicator and strategy tests.

    Args:
        series: Pandas Series containing indicator values
        name: Indicator name for error messages
        min_std: Minimum standard deviation expected

    Raises:
        AssertionError: If indicator is too constant
    """
    valid_values = series.dropna()
    assert len(valid_values) > 1, f"{name} has insufficient valid values"

    std = valid_values.std()
    assert std >= min_std, \
        f"{name} has insufficient variation (std={std:.4f}, min={min_std})"


def assert_band_relationships(df, upper_col, middle_col, lower_col, indicator_name='Bands'):
    """
    Assert proper band relationships (upper >= middle >= lower).

    Common for indicators with bands (Bollinger Bands, Ichimoku, etc.)
    Used across both indicator and strategy tests.

    Args:
        df: DataFrame with band columns
        upper_col: Name of upper band column
        middle_col: Name of middle band column
        lower_col: Name of lower band column
        indicator_name: Name for error messages

    Example:
        bb = calculate_bollinger_bands(data)
        assert_band_relationships(bb, 'upper_band', 'middle_band', 'lower_band', 'BB')
    """
    valid_df = df.dropna()

    assert (valid_df[upper_col] >= valid_df[middle_col]).all(), \
        f"{indicator_name}: {upper_col} must be >= {middle_col}"
    assert (valid_df[middle_col] >= valid_df[lower_col]).all(), \
        f"{indicator_name}: {middle_col} must be >= {lower_col}"
