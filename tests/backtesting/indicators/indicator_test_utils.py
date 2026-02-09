"""
Shared utilities for indicator testing.

Provides reusable functions for common test patterns across all indicator tests:
- Cache validation
- Result comparison
- Edge case testing
"""
import numpy as np
import pandas as pd

from app.backtesting.cache.indicators_cache import indicator_cache


# ==================== Indicator Validation ====================

def assert_indicator_varies(series, name, min_std=0.1):
    """
    Assert indicator shows variation (not constant).

    Validates that an indicator actually responds to price changes rather than
    remaining constant. Useful for detecting calculation errors.

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


# ==================== Cache Testing Utilities ====================

def assert_cache_hit_on_second_call(first_result, second_result, result_type='series'):
    """
    Assert that two calculation results are identical (validates cached result matches original).

    This function only validates that the results are identical - it does NOT check
    cache hit/miss counters. Use assert_cache_was_hit() separately to validate counters.

    Args:
        first_result: Result from first calculation
        second_result: Result from second calculation (should be cached)
        result_type: Type of result ('series', 'dataframe', or 'dict')

    Example:
        # First calculation
        rsi_1 = calculate_rsi(data, period=14)
        misses_before = indicator_cache.misses

        # Second calculation (should hit cache)
        rsi_2 = calculate_rsi(data, period=14)

        # Validate cache hit and result identity
        assert_cache_was_hit(misses_before)  # Check cache counters
        assert_cache_hit_on_second_call(rsi_1, rsi_2, 'series')  # Check results match
    """
    # Results must be identical
    if result_type == 'series':
        assert isinstance(first_result, pd.Series), "First result must be pandas Series"
        assert isinstance(second_result, pd.Series), "Second result must be pandas Series"
        assert len(first_result) == len(second_result), "Results should have same length"
        np.testing.assert_array_equal(
            first_result.values,
            second_result.values,
            err_msg="Cached result should match exactly"
        )
    elif result_type == 'dataframe':
        assert isinstance(first_result, pd.DataFrame), "First result must be pandas DataFrame"
        assert isinstance(second_result, pd.DataFrame), "Second result must be pandas DataFrame"
        pd.testing.assert_frame_equal(
            first_result,
            second_result,
            check_exact=True,
            obj="Cached result"
        )
    elif result_type == 'dict':
        assert isinstance(first_result, dict), "First result must be dict"
        assert isinstance(second_result, dict), "Second result must be dict"
        assert first_result.keys() == second_result.keys(), "Dict keys should match"
        for key in first_result.keys():
            if isinstance(first_result[key], pd.Series):
                np.testing.assert_array_equal(
                    first_result[key].values,
                    second_result[key].values,
                    err_msg=f"Cached {key} should match exactly"
                )
            else:
                assert first_result[key] == second_result[key], f"Cached {key} should match"
    else:
        raise ValueError(f"Unknown result_type: {result_type}")


def setup_cache_test():
    """
    Prepare cache for testing by clearing data and resetting stats.

    Returns:
        Initial miss count (should be 0 after reset)

    Example:
        initial_misses = setup_cache_test()
        result = calculate_indicator(...)
        assert indicator_cache.misses == initial_misses + 1
    """
    indicator_cache.cache_data.clear()
    indicator_cache.reset_stats()
    return indicator_cache.misses


def assert_different_params_use_different_cache(result1, result2):
    """
    Assert that different parameters produce different results (not cached together).

    Args:
        result1: Result with first parameter set
        result2: Result with second parameter set

    Example:
        rsi_14 = calculate_rsi(data, period=14)
        rsi_21 = calculate_rsi(data, period=21)
        assert_different_params_use_different_cache(rsi_14, rsi_21)
    """
    if isinstance(result1, pd.Series) and isinstance(result2, pd.Series):
        assert not result1.equals(result2), \
            "Different parameters should produce different results"
    elif isinstance(result1, pd.DataFrame) and isinstance(result2, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(result1, result2)
            raise AssertionError("Different parameters should produce different results")
        except AssertionError:
            # Expected when dataframes are different
            return
    elif isinstance(result1, dict) and isinstance(result2, dict):
        # Check at least one part differs
        differs = False
        for key in result1.keys():
            if isinstance(result1[key], pd.Series):
                if not result1[key].equals(result2[key]):
                    differs = True
                    break
            else:
                if result1[key] != result2[key]:
                    differs = True
                    break
        assert differs, "Different parameters should produce different results"
    else:
        raise ValueError(f"Unsupported result types: {type(result1)} and {type(result2)}")


# ==================== Structure Validation Utilities ====================

def assert_indicator_structure(
    result, expected_length, expected_type='series',
    expected_columns=None, indicator_name='Indicator'
):
    """
    Validate basic structure of indicator result.

    Args:
        result: Indicator calculation result
        expected_length: Expected length of result
        expected_type: 'series', 'dataframe', or 'dict'
        expected_columns: For dataframes, list of expected column names
        indicator_name: Name for error messages

    Example:
        bb = calculate_bollinger_bands(data, period=20)
        assert_indicator_structure(
            bb, len(data), 'dataframe',
            ['middle_band', 'upper_band', 'lower_band'],
            'Bollinger Bands'
        )
    """
    if expected_type == 'series':
        assert isinstance(result, pd.Series), \
            f"{indicator_name} must return pandas Series, got {type(result)}"
        assert len(result) == expected_length, \
            f"{indicator_name} length must equal input length ({expected_length}), got {len(result)}"
    elif expected_type == 'dataframe':
        assert isinstance(result, pd.DataFrame), \
            f"{indicator_name} must return pandas DataFrame, got {type(result)}"
        assert len(result) == expected_length, \
            f"{indicator_name} length must equal input length ({expected_length}), got {len(result)}"
        if expected_columns:
            assert list(result.columns) == expected_columns, \
                f"{indicator_name} columns should be {expected_columns}, got {list(result.columns)}"
    elif expected_type == 'dict':
        assert isinstance(result, dict), \
            f"{indicator_name} must return dict, got {type(result)}"
        if expected_columns:
            assert set(result.keys()) == set(expected_columns), \
                f"{indicator_name} keys should be {expected_columns}, got {list(result.keys())}"
    else:
        raise ValueError(f"Unknown expected_type: {expected_type}")


# ==================== Value Validation Utilities ====================

def assert_values_in_range(
    result, min_val=None, max_val=None, column=None,
    indicator_name='Indicator', check_valid_only=True
):
    """
    Assert indicator values are within expected range.

    Args:
        result: Indicator result (Series or DataFrame)
        min_val: Minimum allowed value (None = no minimum)
        max_val: Maximum allowed value (None = no maximum)
        column: For DataFrames, which column to check (None = check all)
        indicator_name: Name for error messages
        check_valid_only: If True, only check non-NaN values

    Example:
        rsi = calculate_rsi(data, period=14)
        assert_values_in_range(rsi, 0, 100, indicator_name='RSI')
    """
    if isinstance(result, pd.Series):
        values = result.dropna() if check_valid_only else result
        if min_val is not None:
            assert (values >= min_val).all(), \
                f"{indicator_name} has values < {min_val}: min={values.min()}"
        if max_val is not None:
            assert (values <= max_val).all(), \
                f"{indicator_name} has values > {max_val}: max={values.max()}"
    elif isinstance(result, pd.DataFrame):
        columns_to_check = [column] if column else result.columns
        for col in columns_to_check:
            values = result[col].dropna() if check_valid_only else result[col]
            if min_val is not None:
                assert (values >= min_val).all(), \
                    f"{indicator_name}[{col}] has values < {min_val}: min={values.min()}"
            if max_val is not None:
                assert (values <= max_val).all(), \
                    f"{indicator_name}[{col}] has values > {max_val}: max={values.max()}"
    else:
        raise ValueError(f"Result must be Series or DataFrame, got {type(result)}")


# ==================== Relationship Validation Utilities ====================

def assert_band_relationships(df, upper_col, middle_col, lower_col, indicator_name='Bands'):
    """
    Assert proper band relationships (upper >= middle >= lower).

    Common for indicators with bands (Bollinger Bands, Ichimoku, etc.)

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


# ==================== Comparison Utilities ====================

def assert_longer_period_smoother(short_result, long_result, indicator_name='Indicator'):
    """
    Assert that longer period produces smoother (less volatile) results.

    Compares the volatility of changes between two results with different periods.

    Args:
        short_result: Result with shorter period
        long_result: Result with longer period
        indicator_name: Name for error messages

    Example:
        ema_9 = calculate_ema(data, period=9)
        ema_50 = calculate_ema(data, period=50)
        assert_longer_period_smoother(ema_9, ema_50, 'EMA')
    """
    if isinstance(short_result, pd.Series) and isinstance(long_result, pd.Series):
        # Compare volatility of changes
        short_changes = short_result.diff().dropna()
        long_changes = long_result.diff().dropna()

        assert short_changes.std() > long_changes.std(), \
            f"{indicator_name}: Shorter period should be more volatile " \
            f"(short std={short_changes.std():.6f}, long std={long_changes.std():.6f})"
    else:
        raise ValueError("Results must be pandas Series")


# ==================== Edge Case Testing Utilities ====================

def assert_insufficient_data_returns_nan(result, input_length, indicator_name='Indicator'):
    """
    Assert indicator returns all NaN when data < period.

    Common edge case: When insufficient data for calculation, indicators
    should return NaN values but maintain correct length.

    Args:
        result: Indicator result (Series or DataFrame)
        input_length: Length of input data
        indicator_name: Name for error messages

    Example:
        rsi = calculate_rsi(minimal_data, period=14)
        assert_insufficient_data_returns_nan(rsi, len(minimal_data), 'RSI')
    """
    if isinstance(result, pd.Series):
        assert result.isna().all(), \
            f"{indicator_name}: All values should be NaN when data < period"
        assert len(result) == input_length, \
            f"{indicator_name}: Length should match input ({input_length})"
    elif isinstance(result, pd.DataFrame):
        for col in result.columns:
            assert result[col].isna().all(), \
                f"{indicator_name}[{col}]: All values should be NaN when data < period"
        assert len(result) == input_length, \
            f"{indicator_name}: Length should match input ({input_length})"
    elif isinstance(result, dict):
        for key, series in result.items():
            # Allow for some non-NaN values in edge cases
            nan_ratio = series.isna().sum() / len(series)
            assert nan_ratio >= 0.95 or series.notna().sum() <= 1, \
                f"{indicator_name}[{key}]: Should be mostly NaN when data < period"
    else:
        raise ValueError(f"Result must be Series, DataFrame, or dict, got {type(result)}")


def assert_empty_series_returns_empty(result, result_type='series', indicator_name='Indicator'):
    """
    Assert indicator returns empty result for empty input.

    Validates that indicators handle empty input gracefully by returning
    empty results of the appropriate type.

    Args:
        result: Indicator result
        result_type: 'series', 'dataframe', or 'dict'
        indicator_name: Name for error messages

    Example:
        rsi = calculate_rsi(empty_series, period=14)
        assert_empty_series_returns_empty(rsi, 'series', 'RSI')

        macd = calculate_macd(empty_series)
        assert_empty_series_returns_empty(macd, 'dataframe', 'MACD')

        ichimoku = calculate_ichimoku(empty_h, empty_l, empty_c)
        assert_empty_series_returns_empty(ichimoku, 'dict', 'Ichimoku')
    """
    if result_type == 'series':
        assert len(result) == 0, f"{indicator_name}: Empty input should return empty Series"
        assert isinstance(result, pd.Series), f"{indicator_name}: Result should be a Series"
    elif result_type == 'dataframe':
        assert len(result) == 0, f"{indicator_name}: Empty input should return empty DataFrame"
        assert isinstance(result, pd.DataFrame), f"{indicator_name}: Result should be a DataFrame"
    elif result_type == 'dict':
        for key, series in result.items():
            assert len(series) == 0, \
                f"{indicator_name}[{key}]: Empty input should return empty component"
            assert isinstance(series, pd.Series), \
                f"{indicator_name}[{key}]: Component should be a Series"
    else:
        raise ValueError(f"Unknown result_type: {result_type}. Must be 'series', 'dataframe', or 'dict'")


# ==================== Complete Cache Test Patterns ====================


def assert_cache_distinguishes_different_data(result1, result2, indicator_name='Indicator'):
    """
    Test that cache distinguishes different data series.

    Validates that different datasets produce different results.

    Args:
        result1: Result from first dataset
        result2: Result from second dataset
        indicator_name: Name for error messages

    Example:
        rsi_zs = _calculate_rsi(zs_data['close'], period=14)
        rsi_cl = _calculate_rsi(cl_data['close'], period=14)
        assert_cache_distinguishes_different_data(rsi_zs, rsi_cl, 'RSI')
    """
    # Different data should produce different results
    if isinstance(result1, pd.Series) and isinstance(result2, pd.Series):
        assert not result1.equals(result2), \
            f"{indicator_name}: Different data should produce different results"
    elif isinstance(result1, pd.DataFrame) and isinstance(result2, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(result1, result2)
            raise AssertionError(f"{indicator_name}: Different data should produce different results")
        except AssertionError:
            # Expected when dataframes are different
            return
    elif isinstance(result1, dict) and isinstance(result2, dict):
        # Check at least one part differs
        differs = False
        for key in result1.keys():
            if not result1[key].equals(result2[key]):
                differs = True
                break
        assert differs, f"{indicator_name}: Different data should produce different results"


# ==================== Hash Parameter Contract Tests ====================

def assert_hash_parameter_required(calculate_func, prices, required_params, indicator_name='Indicator'):
    """
    Test that indicator function requires hash parameter(s).

    This is a contract test to ensure hash optimization is enforced.
    Without hash parameters, the function should raise TypeError.

    Args:
        calculate_func: The indicator calculation function to test
        prices: Price series or tuple of price series for multi-input indicators
        required_params: Dict of required parameters (excluding hash parameters)
        indicator_name: Name for error messages

    Example - Single hash parameter:
        assert_hash_parameter_required(
            calculate_func=calculate_rsi,
            prices=price_series,
            required_params={'period': 14},
            indicator_name='RSI'
        )

    Example - Multiple hash parameters (Ichimoku):
        assert_hash_parameter_required(
            calculate_func=calculate_ichimoku_cloud,
            prices=(high_series, low_series, close_series),
            required_params={
                'tenkan_period': 9,
                'kijun_period': 26,
                'senkou_b_period': 52
            },
            indicator_name='Ichimoku'
        )
    """

    # Try to call without hash parameter(s)
    try:
        if isinstance(prices, tuple):
            # Multiple price series (e.g., Ichimoku with high, low, close)
            calculate_func(*prices, **required_params)
        else:
            # Single price series
            calculate_func(prices, **required_params)

        # If we get here, the function didn't require hash - test fails
        raise AssertionError(
            f"{indicator_name}: Function should require hash parameter(s) but accepted call without them"
        )

    except TypeError as e:
        # Expected - function requires hash parameter
        error_msg = str(e)
        assert 'hash' in error_msg.lower() or 'missing' in error_msg.lower(), \
            f"{indicator_name}: TypeError should mention missing hash parameter, got: {error_msg}"


def assert_hash_parameter_required_even_with_cache(
    calculate_func,
    calculate_with_hash_func,
    prices,
    required_params,
    indicator_name='Indicator'
):
    """
    Test that hash parameter is required even when result might be cached.

    This ensures the API contract is enforced regardless of cache state.

    Args:
        calculate_func: The indicator calculation function (raw, without hash)
        calculate_with_hash_func: Helper function that calculates with hash
        prices: Price series or tuple of price series
        required_params: Dict of required parameters (excluding hash parameters)
        indicator_name: Name for error messages

    Example:
        assert_hash_parameter_required_even_with_cache(
            calculate_func=calculate_rsi,
            calculate_with_hash_func=lambda: _calculate_rsi(prices, period=14),
            prices=price_series,
            required_params={'period': 14},
            indicator_name='RSI'
        )
    """

    # First calculate normally to potentially cache it
    calculate_with_hash_func()

    # Should still fail without hash parameter even if cached
    try:
        if isinstance(prices, tuple):
            calculate_func(*prices, **required_params)
        else:
            calculate_func(prices, **required_params)

        raise AssertionError(
            f"{indicator_name}: Function should require hash parameter even with cached result"
        )

    except TypeError as e:
        # Expected - hash is always required
        error_msg = str(e)
        assert 'hash' in error_msg.lower() or 'missing' in error_msg.lower(), \
            f"{indicator_name}: TypeError should mention missing hash parameter"


# ==================== Data Modification for Testing ====================

def inject_price_spike(df, index, spike_pct, direction='up'):
    """
    Inject artificial price spike at specific index.

    Useful for testing indicator behavior during extreme movements.
    Modifies high/low/close prices at specified bar.

    Args:
        df: DataFrame with OHLCV data
        index: Index position or timestamp where spike occurs
        spike_pct: Percentage size of spike
        direction: 'up' for spike up, 'down' for spike down

    Returns:
        Modified DataFrame with price spike

    Example:
        # Inject 5% spike upward at bar 100
        modified_df = inject_price_spike(df.copy(), 100, 5.0, 'up')

        # Test ATR response to spike
        atr = calculate_atr(modified_df, period=14)
        assert atr.iloc[100] > atr.iloc[99]
    """
    df = df.copy()

    if isinstance(index, (pd.Timestamp, str)):
        index = df.index.get_loc(index)

    base_close = df.iloc[index]['close']
    spike_amount = base_close * (spike_pct / 100)

    if direction == 'up':
        df.iloc[index, df.columns.get_loc('high')] = base_close + spike_amount
        df.iloc[index, df.columns.get_loc('close')] = base_close + (spike_amount * 0.5)
    else:
        df.iloc[index, df.columns.get_loc('low')] = base_close - spike_amount
        df.iloc[index, df.columns.get_loc('close')] = base_close - (spike_amount * 0.5)

    return df


def inject_gap(df, index, gap_pct, direction='up'):
    """
    Inject price gap between bars.

    Creates gap between previous close and next open. Useful for testing
    gap-related logic and ATR true range calculation.

    Args:
        df: DataFrame with OHLCV data
        index: Index where gap occurs (gap is between index-1 and index)
        gap_pct: Percentage size of gap
        direction: 'up' for gap up, 'down' for gap down

    Returns:
        Modified DataFrame with price gap

    Example:
        # Create 3% gap up at bar 50
        modified_df = inject_gap(df.copy(), 50, 3.0, 'up')

        # Test ATR captures the gap
        atr = calculate_atr(modified_df, period=14)
        assert atr.iloc[50] > atr.iloc[49]
    """
    df = df.copy()

    if isinstance(index, (pd.Timestamp, str)):
        index = df.index.get_loc(index)

    if index == 0:
        raise ValueError("Cannot inject gap at first bar")

    prev_close = df.iloc[index - 1]['close']
    gap_amount = prev_close * (gap_pct / 100)

    if direction == 'up':
        new_low = prev_close + gap_amount
        # Set low to gap up from previous close
        # This ensures true range captures the gap
        df.iloc[index, df.columns.get_loc('low')] = new_low
        # Adjust high and close proportionally above the new low
        df.iloc[index, df.columns.get_loc('high')] = new_low + 2.0
        df.iloc[index, df.columns.get_loc('close')] = new_low + 1.0
        df.iloc[index, df.columns.get_loc('open')] = new_low + 0.5
    else:
        new_high = prev_close - gap_amount
        # Set high to gap down from previous close
        df.iloc[index, df.columns.get_loc('high')] = new_high
        # Adjust low and close proportionally below the new high
        df.iloc[index, df.columns.get_loc('low')] = new_high - 2.0
        df.iloc[index, df.columns.get_loc('close')] = new_high - 1.0
        df.iloc[index, df.columns.get_loc('open')] = new_high - 0.5

    return df
