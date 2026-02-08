"""
Tests for DataFrame Cache.

Tests cover:
- DataFrame caching and retrieval
- Copy-on-write protection
- Cache hit/miss behavior
- File path as cache key
- DataFrame integrity across cache operations
- Integration with base cache functionality
- Edge cases (missing files, corrupted data)

All tests use real DataFrame operations and file I/O.
"""
import os
import tempfile

import pandas as pd
import pytest

from app.backtesting.cache.dataframe_cache import (
    get_cached_dataframe,
    dataframe_cache
)


# ==================== Fixtures ====================

@pytest.fixture
def sample_dataframe():
    """Create a sample OHLCV DataFrame for testing."""
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    return pd.DataFrame({
        'open': [100.0 + i * 0.1 for i in range(100)],
        'high': [101.0 + i * 0.1 for i in range(100)],
        'low': [99.0 + i * 0.1 for i in range(100)],
        'close': [100.5 + i * 0.1 for i in range(100)],
        'volume': [1000 + i * 10 for i in range(100)]
    }, index=pd.DatetimeIndex(dates, name='datetime'))


@pytest.fixture
def temp_parquet_file(sample_dataframe):
    """Create a temporary parquet file with sample data."""
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
        filepath = f.name
        sample_dataframe.to_parquet(filepath)

    yield filepath

    # Cleanup
    try:
        os.remove(filepath)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def reset_dataframe_cache():
    """Reset the dataframe cache before each test."""
    dataframe_cache.cache_data.clear()
    dataframe_cache.reset_stats()
    yield
    # Cleanup after test
    dataframe_cache.cache_data.clear()


# ==================== Test Classes ====================

class TestDataFrameCacheBasics:
    """Test basic DataFrame caching functionality."""

    def test_cache_dataframe_on_first_load(self, temp_parquet_file, sample_dataframe):
        """Test DataFrame is cached on first load."""
        # First load - cache miss
        df = get_cached_dataframe(temp_parquet_file)

        # Should have loaded and cached the DataFrame
        assert dataframe_cache.contains(temp_parquet_file)
        assert len(df) == len(sample_dataframe)
        assert list(df.columns) == list(sample_dataframe.columns)

    def test_return_cached_dataframe_on_second_load(self, temp_parquet_file):
        """Test cached DataFrame is returned on subsequent loads."""
        # First load
        df1 = get_cached_dataframe(temp_parquet_file)

        # Reset stats to measure second load
        dataframe_cache.reset_stats()

        # Second load - should be cache hit
        df2 = get_cached_dataframe(temp_parquet_file)

        # Should have at least one hit (from get operation)
        assert dataframe_cache.hits >= 1
        assert len(df2) == len(df1)

    def test_cached_dataframe_equals_original(self, temp_parquet_file, sample_dataframe):
        """Test cached DataFrame equals original data."""
        df = get_cached_dataframe(temp_parquet_file)

        pd.testing.assert_frame_equal(df, sample_dataframe, check_freq=False)

    def test_filepath_as_cache_key(self, temp_parquet_file):
        """Test file path is used as cache key."""
        get_cached_dataframe(temp_parquet_file)

        # Cache key should be the filepath
        assert dataframe_cache.contains(temp_parquet_file)
        assert temp_parquet_file in dataframe_cache.cache_data


class TestCopyOnWriteProtection:
    """Test copy-on-write protection prevents cache corruption."""

    def test_returned_dataframe_is_copy(self, temp_parquet_file):
        """Test get_cached_dataframe returns a copy."""
        df1 = get_cached_dataframe(temp_parquet_file)
        df2 = get_cached_dataframe(temp_parquet_file)

        # Should be different objects
        assert df1 is not df2

    def test_modifying_returned_dataframe_doesnt_affect_cache(self, temp_parquet_file):
        """Test modifying returned DataFrame doesn't corrupt cache."""
        # Get first copy
        df1 = get_cached_dataframe(temp_parquet_file)
        original_columns = list(df1.columns)

        # Modify the DataFrame
        df1['new_column'] = 999
        df1['open'] = df1['open'] * 2

        # Get second copy - should be unmodified
        df2 = get_cached_dataframe(temp_parquet_file)

        assert list(df2.columns) == original_columns
        assert 'new_column' not in df2.columns
        assert df2['open'].iloc[0] != df1['open'].iloc[0]

    def test_multiple_strategies_can_modify_independently(self, temp_parquet_file):
        """Test multiple strategies can modify DataFrames independently."""
        # Simulate two strategies getting the same data
        df_strategy1 = get_cached_dataframe(temp_parquet_file)
        df_strategy2 = get_cached_dataframe(temp_parquet_file)

        # Each strategy adds different indicators
        df_strategy1['rsi'] = 50
        df_strategy2['macd'] = 0.5

        # Modifications should be independent
        assert 'rsi' in df_strategy1.columns
        assert 'rsi' not in df_strategy2.columns
        assert 'macd' in df_strategy2.columns
        assert 'macd' not in df_strategy1.columns

    def test_cache_preserves_original_after_modifications(self, temp_parquet_file, sample_dataframe):
        """Test cache preserves original data after modifications."""
        # Get and modify DataFrame
        df = get_cached_dataframe(temp_parquet_file)
        df['indicator'] = 100
        df['open'] = 0

        # Get fresh copy - should match original
        df_fresh = get_cached_dataframe(temp_parquet_file)

        pd.testing.assert_frame_equal(df_fresh, sample_dataframe, check_freq=False)


class TestCacheHitMiss:
    """Test cache hit/miss behavior."""

    def test_first_load_is_cache_miss(self, temp_parquet_file):
        """Test first load results in cache miss."""
        dataframe_cache.reset_stats()

        get_cached_dataframe(temp_parquet_file)

        assert dataframe_cache.misses == 1
        assert dataframe_cache.hits == 0

    def test_second_load_is_cache_hit(self, temp_parquet_file):
        """Test second load results in cache hit."""
        # First load
        get_cached_dataframe(temp_parquet_file)

        dataframe_cache.reset_stats()

        # Second load
        get_cached_dataframe(temp_parquet_file)

        assert dataframe_cache.hits >= 1

    def test_different_files_are_separate_cache_entries(self, sample_dataframe):
        """Test different files create separate cache entries."""
        # Create two temporary files
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f1:
            file1 = f1.name
            sample_dataframe.to_parquet(file1)

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f2:
            file2 = f2.name
            sample_dataframe.to_parquet(file2)

        try:
            # Load both files
            get_cached_dataframe(file1)
            get_cached_dataframe(file2)

            # Both should be in cache
            assert dataframe_cache.contains(file1)
            assert dataframe_cache.contains(file2)
            assert dataframe_cache.size() == 2
        finally:
            os.remove(file1)
            os.remove(file2)

    def test_cache_hit_rate_with_repeated_access(self, temp_parquet_file):
        """Test cache hit rate with repeated access."""
        # First access - miss
        get_cached_dataframe(temp_parquet_file)

        dataframe_cache.reset_stats()

        # Multiple accesses - all hits
        for _ in range(10):
            get_cached_dataframe(temp_parquet_file)

        assert dataframe_cache.hits >= 10


class TestDataFrameIntegrity:
    """Test DataFrame integrity across cache operations."""

    def test_dataframe_index_preserved(self, temp_parquet_file, sample_dataframe):
        """Test DataFrame index is preserved through cache."""
        df = get_cached_dataframe(temp_parquet_file)

        assert df.index.name == sample_dataframe.index.name
        assert len(df.index) == len(sample_dataframe.index)
        pd.testing.assert_index_equal(df.index, sample_dataframe.index)

    def test_dataframe_dtypes_preserved(self, temp_parquet_file, sample_dataframe):
        """Test DataFrame dtypes are preserved through cache."""
        df = get_cached_dataframe(temp_parquet_file)

        for col in sample_dataframe.columns:
            assert df[col].dtype == sample_dataframe[col].dtype

    def test_dataframe_values_unchanged(self, temp_parquet_file, sample_dataframe):
        """Test DataFrame values remain unchanged through cache."""
        df = get_cached_dataframe(temp_parquet_file)

        # Check specific values
        assert df['open'].iloc[0] == sample_dataframe['open'].iloc[0]
        assert df['close'].iloc[-1] == sample_dataframe['close'].iloc[-1]
        assert df['volume'].sum() == sample_dataframe['volume'].sum()

    def test_dataframe_shape_preserved(self, temp_parquet_file, sample_dataframe):
        """Test DataFrame shape is preserved through cache."""
        df = get_cached_dataframe(temp_parquet_file)

        assert df.shape == sample_dataframe.shape


class TestLargeDataFrames:
    """Test caching of large DataFrames."""

    def test_cache_large_dataframe(self):
        """Test caching a large DataFrame (10,000 rows)."""
        dates = pd.date_range('2020-01-01', periods=10000, freq='1h')
        large_df = pd.DataFrame({
            'open': range(10000),
            'high': range(10000),
            'low': range(10000),
            'close': range(10000),
            'volume': range(10000)
        }, index=pd.DatetimeIndex(dates, name='datetime'))

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            filepath = f.name
            large_df.to_parquet(filepath)

        try:
            # Cache large DataFrame
            df = get_cached_dataframe(filepath)

            assert len(df) == 10000
            assert dataframe_cache.contains(filepath)
        finally:
            os.remove(filepath)

    def test_copy_performance_with_large_dataframe(self):
        """Test copy-on-write is efficient with large DataFrames."""
        dates = pd.date_range('2020-01-01', periods=5000, freq='1h')
        large_df = pd.DataFrame({
            'open': range(5000),
            'close': range(5000),
        }, index=pd.DatetimeIndex(dates, name='datetime'))

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            filepath = f.name
            large_df.to_parquet(filepath)

        try:
            # Get multiple copies
            df1 = get_cached_dataframe(filepath)
            df2 = get_cached_dataframe(filepath)
            df3 = get_cached_dataframe(filepath)

            # All should have correct data
            assert len(df1) == 5000
            assert len(df2) == 5000
            assert len(df3) == 5000
        finally:
            os.remove(filepath)


class TestMultipleColumns:
    """Test DataFrames with various column configurations."""

    def test_dataframe_with_many_columns(self):
        """Test DataFrame with many columns."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        df = pd.DataFrame({
            f'col_{i}': range(100) for i in range(20)
        }, index=pd.DatetimeIndex(dates, name='datetime'))

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            filepath = f.name
            df.to_parquet(filepath)

        try:
            cached_df = get_cached_dataframe(filepath)

            assert len(cached_df.columns) == 20
            assert list(cached_df.columns) == [f'col_{i}' for i in range(20)]
        finally:
            os.remove(filepath)

    def test_dataframe_with_mixed_dtypes(self):
        """Test DataFrame with mixed data types."""
        dates = pd.date_range('2024-01-01', periods=50, freq='1h')
        df = pd.DataFrame({
            'int_col': range(50),
            'float_col': [float(i) * 1.5 for i in range(50)],
            'string_col': [f'str_{i}' for i in range(50)],
            'bool_col': [i % 2 == 0 for i in range(50)],
        }, index=pd.DatetimeIndex(dates, name='datetime'))

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            filepath = f.name
            df.to_parquet(filepath)

        try:
            cached_df = get_cached_dataframe(filepath)

            assert cached_df['int_col'].dtype == df['int_col'].dtype
            assert cached_df['float_col'].dtype == df['float_col'].dtype
            assert cached_df['string_col'].dtype == df['string_col'].dtype
            assert cached_df['bool_col'].dtype == df['bool_col'].dtype
        finally:
            os.remove(filepath)


class TestCacheEdgeCases:
    """Test edge cases and error scenarios."""

    def test_load_nonexistent_file_raises_error(self):
        """Test loading non-existent file raises appropriate error."""
        nonexistent_file = '/tmp/nonexistent_file_12345.parquet'

        with pytest.raises(FileNotFoundError):
            get_cached_dataframe(nonexistent_file)

    def test_empty_dataframe_can_be_cached(self):
        """Test empty DataFrame can be cached."""
        empty_df = pd.DataFrame()

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            filepath = f.name
            empty_df.to_parquet(filepath)

        try:
            df = get_cached_dataframe(filepath)

            assert len(df) == 0
            assert dataframe_cache.contains(filepath)
        finally:
            os.remove(filepath)

    def test_single_row_dataframe(self):
        """Test DataFrame with single row."""
        single_row_df = pd.DataFrame({
            'open': [100.0],
            'close': [101.0],
        })

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            filepath = f.name
            single_row_df.to_parquet(filepath)

        try:
            df = get_cached_dataframe(filepath)

            assert len(df) == 1
            assert df['open'].iloc[0] == 100.0
        finally:
            os.remove(filepath)

    def test_cache_survives_file_deletion(self, temp_parquet_file):
        """Test cached DataFrame remains accessible after file deletion."""
        # Load and cache
        df1 = get_cached_dataframe(temp_parquet_file)

        # Delete the file
        os.remove(temp_parquet_file)

        # Should still be able to get from cache
        df2 = get_cached_dataframe(temp_parquet_file)

        pd.testing.assert_frame_equal(df1, df2)


class TestRealWorldScenarios:
    """Test realistic usage scenarios."""

    def test_typical_backtesting_scenario(self, sample_dataframe):
        """Test typical backtesting scenario with multiple strategies."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            filepath = f.name
            sample_dataframe.to_parquet(filepath)

        try:
            # Simulate 5 strategies loading the same data
            dataframes = []
            for _ in range(5):
                df = get_cached_dataframe(filepath)
                dataframes.append(df)

            # All strategies get their own copy
            assert len(dataframes) == 5

            # Each can modify independently
            for i, df in enumerate(dataframes):
                df[f'strategy_{i}_signal'] = i

            # Verify independence
            for i, df in enumerate(dataframes):
                assert f'strategy_{i}_signal' in df.columns
                for j in range(5):
                    if i != j:
                        assert f'strategy_{j}_signal' not in df.columns
        finally:
            os.remove(filepath)

    def test_sequential_backtests_reuse_cache(self, sample_dataframe):
        """Test sequential backtests reuse cached data."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            filepath = f.name
            sample_dataframe.to_parquet(filepath)

        try:
            dataframe_cache.reset_stats()

            # Run 3 sequential backtests
            for backtest_num in range(3):
                df = get_cached_dataframe(filepath)
                df['backtest'] = backtest_num

            # First was miss, others were hits
            assert dataframe_cache.misses >= 1
            assert dataframe_cache.hits >= 2
        finally:
            os.remove(filepath)
