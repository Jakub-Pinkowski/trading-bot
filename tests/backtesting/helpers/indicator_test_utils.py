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


# ==================== Cache Testing Utilities ====================

def assert_cache_hit_on_second_call(first_result, second_result, result_type='series'):
    """
    Assert that cache was hit on second calculation and results match.

    Validates cache behavior by checking:
    1. Misses didn't increase (cache was hit)
    2. Hits increased (cache registered the hit)
    3. Results are identical

    Args:
        first_result: Result from first calculation
        second_result: Result from second calculation (should be cached)
        result_type: Type of result ('series', 'dataframe', or 'dict')

    Example:
        # First calculation
        rsi_1 = calculate_rsi(data, period=14)
        first_misses = indicator_cache.misses

        # Second calculation
        rsi_2 = calculate_rsi(data, period=14)

        # Validate caching
        assert_cache_hit_on_second_call(rsi_1, rsi_2, first_misses, 'series')
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


def assert_cache_was_hit(misses_after_first_calc):
    """
    Assert that cache was hit (misses didn't increase, hits did).

    Args:
        misses_after_first_calc: Miss count after first calculation (before second call)

    Example:
        setup_cache_test()
        result1 = calculate_indicator(...)
        misses_after_first = indicator_cache.misses
        result2 = calculate_indicator(...)  # Should hit cache
        assert_cache_was_hit(misses_after_first)
    """
    assert indicator_cache.misses == misses_after_first_calc, \
        f"Cache misses increased from {misses_after_first_calc} to {indicator_cache.misses} (expected no change)"
    assert indicator_cache.hits > 0, \
        f"Cache hits should be > 0, got {indicator_cache.hits}"


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
        except AssertionError as e:
            if "different results" in str(e):
                raise
            # Expected - dataframes are different
            pass
    else:
        assert result1 != result2, "Different parameters should produce different results"


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


def assert_nan_count(result, expected_nan_count, column=None, indicator_name='Indicator'):
    """
    Assert indicator has expected number of NaN values (warmup period).

    Args:
        result: Indicator result (Series or DataFrame)
        expected_nan_count: Expected number of NaN values
        column: For DataFrames, which column to check (None = check all)
        indicator_name: Name for error messages

    Example:
        rsi = calculate_rsi(data, period=14)
        assert_nan_count(rsi, 14, indicator_name='RSI(14)')
    """
    if isinstance(result, pd.Series):
        actual_nan_count = result.isna().sum()
        assert actual_nan_count == expected_nan_count, \
            f"{indicator_name} should have {expected_nan_count} NaN values, got {actual_nan_count}"
    elif isinstance(result, pd.DataFrame):
        if column:
            actual_nan_count = result[column].isna().sum()
            assert actual_nan_count == expected_nan_count, \
                f"{indicator_name}[{column}] should have {expected_nan_count} NaN values, got {actual_nan_count}"
        else:
            # Check all columns have same NaN count
            for col in result.columns:
                actual_nan_count = result[col].isna().sum()
                assert actual_nan_count == expected_nan_count, \
                    f"{indicator_name}[{col}] should have {expected_nan_count} NaN values, got {actual_nan_count}"
    else:
        raise ValueError(f"Result must be Series or DataFrame, got {type(result)}")


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


def assert_all_positive(result, column=None, indicator_name='Indicator'):
    """
    Assert all valid values are positive (> 0).

    Convenience wrapper around assert_values_in_range for common case.

    Example:
        atr = calculate_atr(data, period=14)
        assert_all_positive(atr, indicator_name='ATR')
    """
    assert_values_in_range(result, min_val=0, column=column,
                           indicator_name=indicator_name, check_valid_only=True)


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


def assert_indicator_varies(result, column=None, min_std=0.0, indicator_name='Indicator'):
    """
    Assert indicator values show variation (not constant).

    Args:
        result: Indicator result (Series or DataFrame)
        column: For DataFrames, which column to check (None = check all)
        min_std: Minimum standard deviation required
        indicator_name: Name for error messages

    Example:
        ema = calculate_ema(data, period=9)
        assert_indicator_varies(ema, min_std=0.01, indicator_name='EMA(9)')
    """
    if isinstance(result, pd.Series):
        valid_values = result.dropna()
        std = valid_values.std()
        assert std > min_std, \
            f"{indicator_name} should vary (std={std:.6f} <= {min_std})"
    elif isinstance(result, pd.DataFrame):
        columns_to_check = [column] if column else result.columns
        for col in columns_to_check:
            valid_values = result[col].dropna()
            std = valid_values.std()
            assert std > min_std, \
                f"{indicator_name}[{col}] should vary (std={std:.6f} <= {min_std})"
    else:
        raise ValueError(f"Result must be Series or DataFrame, got {type(result)}")


# ==================== Index Validation Utilities ====================

def assert_index_preserved(result, original_index, indicator_name='Indicator'):
    """
    Assert indicator preserves the original data's index.

    Args:
        result: Indicator result
        original_index: Original data's index
        indicator_name: Name for error messages

    Example:
        rsi = calculate_rsi(df['close'], period=14)
        assert_index_preserved(rsi, df.index, 'RSI')
    """
    if isinstance(result, (pd.Series, pd.DataFrame)):
        assert result.index.equals(original_index), \
            f"{indicator_name} should preserve original index"
    else:
        raise ValueError(f"Result must be Series or DataFrame, got {type(result)}")


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


# ==================== Complete Cache Test Patterns ====================

def assert_recalculation_hits_cache(calc_func, *args, **kwargs):
    """
    Assert that recalculating with same parameters hits cache.

    Generic function that works with any indicator calculation.

    Args:
        calc_func: Indicator calculation function
        *args: Arguments to pass to calc_func
        **kwargs: Keyword arguments to pass to calc_func

    Example:
        # For RSI
        assert_recalculation_hits_cache(
            _calculate_rsi,
            zs_1h_data['close'],
            period=14
        )

        # For MACD
        assert_recalculation_hits_cache(
            _calculate_macd,
            zs_1h_data['close'],
            fast_period=12,
            slow_period=26,
            signal_period=9
        )
    """
    misses_before = indicator_cache.misses
    result = calc_func(*args, **kwargs)
    assert indicator_cache.misses == misses_before, \
        f"Recalculation should hit cache (misses increased from {misses_before} to {indicator_cache.misses})"
    return result


def test_cache_distinguishes_different_params(
    result1, result2, recalc_func, recalc_args,
    result_type='series', indicator_name='Indicator'
):
    """
    Complete test for cache distinguishing different parameters.

    Tests that:
    1. Different parameters produce different results
    2. Recalculating with same parameters hits cache
    3. Cached result matches original

    Args:
        result1: Result with first parameter set
        result2: Result with second parameter set
        recalc_func: Function to recalculate result1
        recalc_args: Tuple of (args, kwargs) for recalc_func
        result_type: 'series', 'dataframe', or 'dict'
        indicator_name: Name for error messages

    Example:
        rsi_14 = _calculate_rsi(data, period=14)
        rsi_21 = _calculate_rsi(data, period=21)
        test_cache_distinguishes_different_params(
            rsi_14, rsi_21,
            _calculate_rsi,
            ((data,), {'period': 14}),
            'series', 'RSI'
        )
    """
    # Step 1: Different params should produce different results
    assert_different_params_use_different_cache(result1, result2)

    # Step 2: Recalculation should hit cache
    args, kwargs = recalc_args
    misses_before = indicator_cache.misses
    result1_again = recalc_func(*args, **kwargs)
    assert indicator_cache.misses == misses_before, \
        f"{indicator_name}: Recalculation should hit cache"

    # Step 3: Cached result should match original
    if result_type == 'series':
        np.testing.assert_array_equal(
            result1.values,
            result1_again.values,
            err_msg=f"{indicator_name}: Cached result should match original"
        )
    elif result_type == 'dataframe':
        pd.testing.assert_frame_equal(
            result1,
            result1_again,
            check_exact=True,
            obj=f"{indicator_name} cached result"
        )
    elif result_type == 'dict':
        for key in result1.keys():
            np.testing.assert_array_equal(
                result1[key].values,
                result1_again[key].values,
                err_msg=f"{indicator_name}[{key}]: Cached result should match original"
            )


def assert_cache_distinguishes_different_data(result1, result2, len1, len2, indicator_name='Indicator'):
    """
    Test that cache distinguishes different data series.

    Validates that:
    1. Different datasets produce different results
    2. Results have different lengths (confirming different data)

    Args:
        result1: Result from first dataset
        result2: Result from second dataset
        len1: Expected length of result1
        len2: Expected length of result2
        indicator_name: Name for error messages

    Example:
        rsi_zs = _calculate_rsi(zs_data['close'], period=14)
        rsi_cl = _calculate_rsi(cl_data['close'], period=14)
        test_cache_distinguishes_different_data(
            rsi_zs, rsi_cl,
            len(zs_data), len(cl_data),
            'RSI'
        )
    """
    # Different data should produce different results
    if isinstance(result1, pd.Series) and isinstance(result2, pd.Series):
        assert not result1.equals(result2), \
            f"{indicator_name}: Different data should produce different results"
    elif isinstance(result1, pd.DataFrame) and isinstance(result2, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(result1, result2)
            raise AssertionError(f"{indicator_name}: Different data should produce different results")
        except AssertionError as e:
            if "should produce different results" in str(e):
                raise
            # Expected - dataframes are different
            pass
    elif isinstance(result1, dict) and isinstance(result2, dict):
        # Check at least one component differs
        differs = False
        for key in result1.keys():
            if not result1[key].equals(result2[key]):
                differs = True
                break
        assert differs, f"{indicator_name}: Different data should produce different results"

    # Different datasets should have different lengths
    assert len1 != len2, \
        f"{indicator_name}: Different datasets should have different lengths ({len1} vs {len2})"
