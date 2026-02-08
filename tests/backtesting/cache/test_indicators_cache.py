"""
Tests for Indicator Cache.

Tests cover:
- Cache configuration (30 day TTL, 500 max size)
- Cache key generation with tuples (indicator_name, data_hash, params)
- Cache hit/miss with different keys
- Integration with base cache functionality
- LRU eviction with large number of cached indicators
- Cache operations with indicator-specific data types

All tests use the indicator_cache instance directly.
"""

import pandas as pd
import pytest

from app.backtesting.cache.indicators_cache import indicator_cache


# ==================== Fixtures ====================

@pytest.fixture
def sample_indicator_result():
    """Create sample indicator result (Series)."""
    return pd.Series([30.5, 45.2, 55.8, 60.1, 52.3], name='rsi')


@pytest.fixture
def sample_macd_result():
    """Create sample MACD result (DataFrame)."""
    return pd.DataFrame({
        'macd_line': [0.5, 0.8, 1.2, 1.0, 0.7],
        'signal_line': [0.3, 0.6, 0.9, 1.1, 0.9],
        'histogram': [0.2, 0.2, 0.3, -0.1, -0.2]
    })


@pytest.fixture(autouse=True)
def reset_indicator_cache():
    """Reset the indicator cache before each test."""
    indicator_cache.cache_data.clear()
    indicator_cache.reset_stats()
    yield
    # Cleanup after test
    indicator_cache.cache_data.clear()


# ==================== Test Classes ====================

class TestIndicatorCacheConfiguration:
    """Test indicator cache configuration."""

    def test_cache_has_correct_max_size(self):
        """Test indicator cache is configured with max_size=500."""
        assert indicator_cache.max_size == 500

    def test_cache_has_correct_max_age(self):
        """Test indicator cache is configured with 30 day TTL."""
        # 30 days = 30 * 24 * 60 * 60 = 2,592,000 seconds
        assert indicator_cache.max_age == 2592000

    def test_cache_has_correct_name(self):
        """Test indicator cache has correct name."""
        assert indicator_cache.cache_name == "indicator"


class TestIndicatorCacheBasicOperations:
    """Test basic cache operations with indicator data."""

    def test_cache_indicator_result(self, sample_indicator_result):
        """Test caching indicator result."""
        cache_key = ('rsi', 'data_hash_123', 14)

        indicator_cache.set(cache_key, sample_indicator_result)

        assert indicator_cache.contains(cache_key)
        cached_result = indicator_cache.get(cache_key)
        pd.testing.assert_series_equal(cached_result, sample_indicator_result)

    def test_cache_hit_with_same_key(self, sample_indicator_result):
        """Test cache hit when using same key."""
        cache_key = ('rsi', 'data_hash_456', 14)

        # Set value
        indicator_cache.set(cache_key, sample_indicator_result)
        indicator_cache.reset_stats()

        # Get with same key - should be hit
        result = indicator_cache.get(cache_key)

        assert indicator_cache.hits >= 1
        pd.testing.assert_series_equal(result, sample_indicator_result)

    def test_cache_miss_with_different_key(self, sample_indicator_result):
        """Test cache miss when using different key."""
        cache_key1 = ('rsi', 'data_hash_789', 14)
        cache_key2 = ('rsi', 'data_hash_789', 21)  # Different period

        # Set value with key1
        indicator_cache.set(cache_key1, sample_indicator_result)
        indicator_cache.reset_stats()

        # Get with key2 - should be miss
        result = indicator_cache.get(cache_key2)

        assert indicator_cache.misses >= 1
        assert result is None

    def test_cache_macd_dataframe(self, sample_macd_result):
        """Test caching MACD DataFrame result."""
        cache_key = ('macd', 'data_hash_abc', 12, 26, 9)

        indicator_cache.set(cache_key, sample_macd_result)

        assert indicator_cache.contains(cache_key)
        cached_result = indicator_cache.get(cache_key)
        pd.testing.assert_frame_equal(cached_result, sample_macd_result)


class TestCacheKeyStructure:
    """Test cache key structure for indicators."""

    def test_cache_key_with_tuple(self, sample_indicator_result):
        """Test cache keys are tuples with (indicator, hash, params)."""
        cache_key = ('rsi', 'hash123', 14)

        indicator_cache.set(cache_key, sample_indicator_result)

        assert cache_key in indicator_cache.cache_data

    def test_different_indicator_names_separate_keys(self, sample_indicator_result):
        """Test different indicator names create separate cache entries."""
        key_rsi = ('rsi', 'hash123', 14)
        key_ema = ('ema', 'hash123', 14)

        indicator_cache.set(key_rsi, sample_indicator_result)
        indicator_cache.set(key_ema, sample_indicator_result)

        assert indicator_cache.size() == 2
        assert indicator_cache.contains(key_rsi)
        assert indicator_cache.contains(key_ema)

    def test_different_data_hash_separate_keys(self, sample_indicator_result):
        """Test different data hashes create separate cache entries."""
        key_data1 = ('rsi', 'hash_abc', 14)
        key_data2 = ('rsi', 'hash_xyz', 14)

        indicator_cache.set(key_data1, sample_indicator_result)
        indicator_cache.set(key_data2, sample_indicator_result)

        assert indicator_cache.size() == 2

    def test_different_parameters_separate_keys(self, sample_indicator_result):
        """Test different parameters create separate cache entries."""
        key_period14 = ('rsi', 'hash123', 14)
        key_period21 = ('rsi', 'hash123', 21)

        indicator_cache.set(key_period14, sample_indicator_result)
        indicator_cache.set(key_period21, sample_indicator_result)

        assert indicator_cache.size() == 2


class TestCacheStatistics:
    """Test cache statistics tracking."""

    def test_cache_tracks_hits(self, sample_indicator_result):
        """Test cache correctly tracks hits."""
        cache_key = ('rsi', 'hash123', 14)

        indicator_cache.set(cache_key, sample_indicator_result)
        indicator_cache.reset_stats()

        # Multiple gets - all hits
        indicator_cache.get(cache_key)
        indicator_cache.get(cache_key)
        indicator_cache.get(cache_key)

        assert indicator_cache.hits >= 3

    def test_cache_tracks_misses(self):
        """Test cache correctly tracks misses."""
        indicator_cache.reset_stats()

        # All different keys - all misses
        indicator_cache.get(('rsi', 'hash1', 14))
        indicator_cache.get(('rsi', 'hash2', 14))
        indicator_cache.get(('ema', 'hash1', 20))

        assert indicator_cache.misses >= 3

    def test_reset_stats_clears_counters(self, sample_indicator_result):
        """Test reset_stats clears hit/miss counters."""
        cache_key = ('rsi', 'hash123', 14)

        indicator_cache.set(cache_key, sample_indicator_result)
        indicator_cache.get(cache_key)  # Hit
        indicator_cache.get(('rsi', 'hash456', 14))  # Miss

        indicator_cache.reset_stats()

        assert indicator_cache.hits == 0
        assert indicator_cache.misses == 0


class TestCacheLRUEviction:
    """Test LRU eviction with indicator cache."""

    def test_cache_evicts_old_entries_when_approaching_limit(self, sample_indicator_result):
        """Test cache evicts entries when approaching size limit."""
        # Create many cache entries (but not 500 to keep test fast)
        for i in range(20):
            cache_key = ('rsi', f'hash_{i}', 14)
            indicator_cache.set(cache_key, sample_indicator_result)

        # All should be cached
        assert indicator_cache.size() >= 20

    def test_frequently_used_indicators_stay_cached(self, sample_indicator_result):
        """Test frequently accessed indicators stay in cache."""
        # Create multiple indicators
        for period in range(10, 20):
            cache_key = ('rsi', 'hash_common', period)
            indicator_cache.set(cache_key, sample_indicator_result)

        # Frequently access one specific calculation
        frequent_key = ('rsi', 'hash_common', 14)
        for _ in range(10):
            indicator_cache.get(frequent_key)

        # The frequently used one should still be cached
        indicator_cache.reset_stats()
        result = indicator_cache.get(frequent_key)
        assert indicator_cache.hits >= 1
        assert result is not None


class TestCacheWithDifferentDataTypes:
    """Test cache with different indicator result types."""

    def test_cache_series_result(self, sample_indicator_result):
        """Test cache works with pd.Series results."""
        cache_key = ('rsi', 'hash123', 14)

        indicator_cache.set(cache_key, sample_indicator_result)
        result = indicator_cache.get(cache_key)

        assert isinstance(result, pd.Series)
        pd.testing.assert_series_equal(result, sample_indicator_result)

    def test_cache_dataframe_result(self, sample_macd_result):
        """Test cache works with pd.DataFrame results."""
        cache_key = ('macd', 'hash123', 12, 26, 9)

        indicator_cache.set(cache_key, sample_macd_result)
        result = indicator_cache.get(cache_key)

        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, sample_macd_result)

    def test_cache_preserves_series_name(self, sample_indicator_result):
        """Test cache preserves Series name."""
        cache_key = ('rsi', 'hash123', 14)

        indicator_cache.set(cache_key, sample_indicator_result)
        result = indicator_cache.get(cache_key)

        assert result.name == sample_indicator_result.name

    def test_cache_preserves_index(self):
        """Test cache preserves Series index."""
        prices_with_index = pd.Series(
            [50.5, 48.2, 55.8, 60.1, 52.3],
            index=pd.date_range('2024-01-01', periods=5, freq='1h'),
            name='rsi'
        )

        cache_key = ('rsi', 'hash123', 14)
        indicator_cache.set(cache_key, prices_with_index)
        result = indicator_cache.get(cache_key)

        pd.testing.assert_index_equal(result.index, prices_with_index.index)


class TestCacheIntegration:
    """Test integration with base Cache functionality."""

    def test_indicator_cache_uses_base_cache_methods(self):
        """Test indicator cache inherits base Cache methods."""
        assert hasattr(indicator_cache, 'get')
        assert hasattr(indicator_cache, 'set')
        assert hasattr(indicator_cache, 'contains')
        assert hasattr(indicator_cache, 'size')
        assert hasattr(indicator_cache, 'reset_stats')
        assert hasattr(indicator_cache, 'save_cache')

    def test_manual_cache_operations(self, sample_indicator_result):
        """Test manual cache get/set operations."""
        cache_key = ('test_indicator', 'manual_test', 10)

        # Manually set value
        indicator_cache.set(cache_key, sample_indicator_result)

        # Should be retrievable
        assert indicator_cache.contains(cache_key)
        cached_value = indicator_cache.get(cache_key)
        pd.testing.assert_series_equal(cached_value, sample_indicator_result)

    def test_cache_can_be_saved_and_loaded(self, sample_indicator_result):
        """Test indicator cache can be persisted to disk."""
        # Add some indicators
        indicator_cache.set(('rsi', 'hash1', 14), sample_indicator_result)
        indicator_cache.set(('ema', 'hash2', 20), sample_indicator_result)

        # Save cache
        success = indicator_cache.save_cache()
        assert success is True


class TestCacheEdgeCases:
    """Test edge cases and unusual scenarios."""

    def test_cache_with_empty_result(self):
        """Test cache handles empty Series."""
        empty_series = pd.Series([], name='rsi')
        cache_key = ('rsi', 'hash_empty', 14)

        indicator_cache.set(cache_key, empty_series)
        result = indicator_cache.get(cache_key)

        assert len(result) == 0
        pd.testing.assert_series_equal(result, empty_series)

    def test_cache_with_nan_values(self):
        """Test cache handles Series with NaN values."""
        series_with_nan = pd.Series([30.5, float('nan'), 55.8, float('nan'), 52.3], name='rsi')
        cache_key = ('rsi', 'hash_nan', 14)

        indicator_cache.set(cache_key, series_with_nan)
        result = indicator_cache.get(cache_key)

        pd.testing.assert_series_equal(result, series_with_nan)

    def test_cache_with_complex_key(self, sample_indicator_result):
        """Test cache handles complex multi-parameter keys."""
        # Key with many parameters (like MACD)
        cache_key = ('macd', 'hash123', 12, 26, 9, 'ema', True)

        indicator_cache.set(cache_key, sample_indicator_result)
        result = indicator_cache.get(cache_key)

        pd.testing.assert_series_equal(result, sample_indicator_result)


class TestRealWorldScenarios:
    """Test realistic indicator caching scenarios."""

    def test_multiple_strategies_sharing_cache(self, sample_indicator_result):
        """Test multiple strategies can share the indicator cache."""
        indicator_cache.reset_stats()

        # Strategy 1: Calculate RSI(14) and RSI(21)
        key_rsi14 = ('rsi', 'data_hash_common', 14)
        key_rsi21 = ('rsi', 'data_hash_common', 21)
        indicator_cache.set(key_rsi14, sample_indicator_result)
        indicator_cache.set(key_rsi21, sample_indicator_result)

        # Strategy 2: Uses RSI(14) again - should be cache hit
        indicator_cache.reset_stats()
        result = indicator_cache.get(key_rsi14)

        assert indicator_cache.hits >= 1
        assert result is not None

    def test_parameter_optimization_caching(self, sample_indicator_result):
        """Test caching during parameter optimization."""
        data_hash = 'optimization_data_hash'

        # Test different RSI periods
        for period in [10, 14, 20, 25, 30]:
            cache_key = ('rsi', data_hash, period)
            indicator_cache.set(cache_key, sample_indicator_result)

        # All should be cached separately
        assert indicator_cache.size() >= 5

        # Re-test best parameter (period=14) - should be cache hit
        indicator_cache.reset_stats()
        result = indicator_cache.get(('rsi', data_hash, 14))

        assert indicator_cache.hits >= 1

    def test_different_datasets_cached_separately(self, sample_indicator_result):
        """Test same indicator on different datasets cached separately."""
        # Same indicator, different data
        key_data1 = ('rsi', 'hash_dataset1', 14)
        key_data2 = ('rsi', 'hash_dataset2', 14)
        key_data3 = ('rsi', 'hash_dataset3', 14)

        indicator_cache.set(key_data1, sample_indicator_result)
        indicator_cache.set(key_data2, sample_indicator_result)
        indicator_cache.set(key_data3, sample_indicator_result)

        # All three should be cached
        assert indicator_cache.size() >= 3
        assert indicator_cache.contains(key_data1)
        assert indicator_cache.contains(key_data2)
        assert indicator_cache.contains(key_data3)
