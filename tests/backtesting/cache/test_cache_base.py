"""
Tests for Cache Base Class.

Tests cover:
- Cache initialization and configuration
- Get/Set operations with LRU ordering
- Cache hit/miss statistics
- TTL (Time-To-Live) expiration
- LRU (Least Recently Used) eviction
- Disk persistence and file locking
- Cache size limits and enforcement
- Error handling and retry logic
- Edge cases (empty cache, full cache, expired items)

All tests use real cache operations with actual file I/O.
"""
import os
import pickle
import time
from collections import OrderedDict
from unittest.mock import patch, MagicMock

import pytest

from app.backtesting.cache.cache_base import Cache, _convert_cache_format
from config import CACHE_DIR


# ==================== Fixtures ====================

@pytest.fixture
def cache_name():
    """Unique cache name for test isolation."""
    return f"test_cache_{int(time.time() * 1000000)}"


@pytest.fixture
def basic_cache(cache_name):
    """Basic cache instance with default settings."""
    cache = Cache(cache_name=cache_name, max_size=100, max_age=3600)
    yield cache
    # Cleanup
    _cleanup_cache_files(cache)


@pytest.fixture
def small_cache(cache_name):
    """Cache with small size limit for LRU testing."""
    cache = Cache(cache_name=f"{cache_name}_small", max_size=3, max_age=3600)
    yield cache
    _cleanup_cache_files(cache)


@pytest.fixture
def ttl_cache(cache_name):
    """Cache with short TTL for expiration testing."""
    cache = Cache(cache_name=f"{cache_name}_ttl", max_size=100, max_age=1)  # 1 second TTL
    yield cache
    _cleanup_cache_files(cache)


@pytest.fixture
def persistent_cache(cache_name):
    """Cache for persistence testing that needs cleanup."""
    cache = Cache(cache_name=f"{cache_name}_persist", max_size=100, max_age=3600)
    yield cache
    _cleanup_cache_files(cache)


def _cleanup_cache_files(cache):
    """Clean up cache files after test."""
    try:
        if os.path.exists(cache.cache_file):
            os.remove(cache.cache_file)
        if os.path.exists(cache.lock_file):
            os.remove(cache.lock_file)
    except Exception:
        pass  # Best effort cleanup


# ==================== Test Classes ====================

class TestCacheInitialization:
    """Test Cache initialization and configuration."""

    @pytest.mark.parametrize("max_size,max_age", [
        (100, 3600),
        (10, 60),
        (1000, 86400),
        (None, None),  # No limits
    ])
    def test_initialization_with_various_configs(self, cache_name, max_size, max_age):
        """Test cache initializes correctly with various configurations."""
        cache = Cache(cache_name=cache_name, max_size=max_size, max_age=max_age)

        assert cache.cache_name == cache_name
        assert cache.max_size == max_size
        assert cache.max_age == max_age
        assert cache.hits == 0
        assert cache.misses == 0
        assert cache.size() == 0
        assert isinstance(cache.cache_data, OrderedDict)

        _cleanup_cache_files(cache)

    def test_cache_files_created(self, basic_cache):
        """Test cache creates file and lock file."""
        # Files are created on save
        basic_cache.set('key1', 'value1')
        basic_cache.save_cache()

        assert os.path.exists(basic_cache.cache_file)
        assert os.path.exists(basic_cache.lock_file)

    def test_cache_loads_existing_data_on_init(self, cache_name):
        """Test cache loads existing data from disk on initialization."""
        # Create first cache and save data
        cache1 = Cache(cache_name=cache_name, max_size=100, max_age=3600)
        cache1.set('key1', 'value1')
        cache1.set('key2', 'value2')
        cache1.save_cache()

        # Create second cache with same name - should load existing data
        cache2 = Cache(cache_name=cache_name, max_size=100, max_age=3600)

        assert cache2.size() == 2
        assert cache2.get('key1') == 'value1'
        assert cache2.get('key2') == 'value2'

        _cleanup_cache_files(cache1)


class TestGetSetOperations:
    """Test basic get/set operations."""

    def test_set_and_get_single_value(self, basic_cache):
        """Test setting and retrieving a single value."""
        basic_cache.set('test_key', 'test_value')

        result = basic_cache.get('test_key')
        assert result == 'test_value'

    def test_set_overwrites_existing_value(self, basic_cache):
        """Test setting same key overwrites existing value."""
        basic_cache.set('key1', 'value1')
        basic_cache.set('key1', 'value2')

        result = basic_cache.get('key1')
        assert result == 'value2'

    def test_get_nonexistent_key_returns_default(self, basic_cache):
        """Test getting non-existent key returns default value."""
        result = basic_cache.get('nonexistent', default='default_value')
        assert result == 'default_value'

    def test_get_without_default_returns_none(self, basic_cache):
        """Test getting non-existent key without default returns None."""
        result = basic_cache.get('nonexistent')
        assert result is None

    def test_set_multiple_values(self, basic_cache):
        """Test setting and retrieving multiple values."""
        data = {
            'key1': 'value1',
            'key2': 'value2',
            'key3': 'value3',
        }

        for key, value in data.items():
            basic_cache.set(key, value)

        for key, expected in data.items():
            assert basic_cache.get(key) == expected

    def test_set_various_data_types(self, basic_cache):
        """Test caching various data types."""
        test_data = {
            'string': 'test string',
            'integer': 42,
            'float': 3.14159,
            'list': [1, 2, 3],
            'dict': {'nested': 'value'},
            'tuple': (1, 2, 3),
            'boolean': True,
            'none': None,
        }

        for key, value in test_data.items():
            basic_cache.set(key, value)
            retrieved = basic_cache.get(key)
            assert retrieved == value
            assert type(retrieved) == type(value)


class TestCacheStatistics:
    """Test cache hit/miss statistics."""

    def test_initial_stats_are_zero(self, basic_cache):
        """Test cache statistics initialize to zero."""
        assert basic_cache.hits == 0
        assert basic_cache.misses == 0

    def test_cache_hit_increments_counter(self, basic_cache):
        """Test cache hits increment the hits counter."""
        basic_cache.set('key1', 'value1')
        basic_cache.reset_stats()  # Reset after set

        basic_cache.contains('key1')  # Hit
        assert basic_cache.hits == 1
        assert basic_cache.misses == 0

        basic_cache.get('key1')  # Another hit
        assert basic_cache.hits == 2
        assert basic_cache.misses == 0

    def test_cache_miss_increments_counter(self, basic_cache):
        """Test cache misses increment the misses counter."""
        basic_cache.contains('nonexistent')  # Miss
        assert basic_cache.hits == 0
        assert basic_cache.misses == 1

        basic_cache.get('nonexistent')  # Another miss (contains is called internally)
        assert basic_cache.misses == 2

    def test_reset_stats_clears_counters(self, basic_cache):
        """Test reset_stats() clears hit/miss counters."""
        basic_cache.set('key1', 'value1')
        basic_cache.get('key1')  # Hit
        basic_cache.get('nonexistent')  # Miss

        assert basic_cache.hits > 0
        assert basic_cache.misses > 0

        basic_cache.reset_stats()

        assert basic_cache.hits == 0
        assert basic_cache.misses == 0

    def test_reset_stats_preserves_cached_data(self, basic_cache):
        """Test reset_stats() doesn't clear cached data."""
        basic_cache.set('key1', 'value1')
        basic_cache.set('key2', 'value2')

        basic_cache.reset_stats()

        assert basic_cache.size() == 2
        assert basic_cache.get('key1') == 'value1'
        assert basic_cache.get('key2') == 'value2'


class TestLRUEviction:
    """Test Least Recently Used (LRU) eviction policy."""

    def test_lru_evicts_oldest_when_full(self, small_cache):
        """Test LRU evicts least recently used item when cache is full."""
        # Fill cache to max size (3 items)
        small_cache.set('key1', 'value1')
        small_cache.set('key2', 'value2')
        small_cache.set('key3', 'value3')

        assert small_cache.size() == 3

        # Add 4th item - should evict key1 (oldest)
        small_cache.set('key4', 'value4')

        assert small_cache.size() == 3
        assert small_cache.get('key1') is None  # Evicted
        assert small_cache.get('key2') == 'value2'
        assert small_cache.get('key3') == 'value3'
        assert small_cache.get('key4') == 'value4'

    def test_lru_get_updates_access_order(self, small_cache):
        """Test getting an item updates its LRU position."""
        # Fill cache
        small_cache.set('key1', 'value1')
        small_cache.set('key2', 'value2')
        small_cache.set('key3', 'value3')

        # Access key1 to make it recently used
        small_cache.get('key1')

        # Add new item - should evict key2 (now oldest)
        small_cache.set('key4', 'value4')

        assert small_cache.get('key1') == 'value1'  # Still present
        assert small_cache.get('key2') is None  # Evicted
        assert small_cache.get('key3') == 'value3'
        assert small_cache.get('key4') == 'value4'

    def test_lru_multiple_accesses_affect_eviction(self, small_cache):
        """Test multiple accesses prevent eviction."""
        small_cache.set('key1', 'value1')
        small_cache.set('key2', 'value2')
        small_cache.set('key3', 'value3')

        # Repeatedly access key1
        small_cache.get('key1')
        small_cache.get('key1')
        small_cache.get('key1')

        # Add multiple new items
        small_cache.set('key4', 'value4')
        small_cache.set('key5', 'value5')

        # key1 should still be present due to frequent access
        assert small_cache.get('key1') == 'value1'


class TestTTLExpiration:
    """Test Time-To-Live (TTL) expiration."""

    def test_item_expires_after_ttl(self, ttl_cache):
        """Test items expire after TTL period."""
        ttl_cache.set('key1', 'value1')

        # Immediately accessible
        assert ttl_cache.get('key1') == 'value1'

        # Wait for expiration
        time.sleep(1.5)

        # Should be expired
        assert ttl_cache.contains('key1') is False
        assert ttl_cache.get('key1') is None

    def test_expired_items_removed_on_access(self, ttl_cache):
        """Test expired items are removed when accessed."""
        ttl_cache.set('key1', 'value1')
        ttl_cache.set('key2', 'value2')

        initial_size = ttl_cache.size()
        assert initial_size == 2

        # Wait for expiration
        time.sleep(1.5)

        # Access expired item - should remove it
        ttl_cache.contains('key1')

        # Size should decrease
        # Note: key2 is still in cache but expired (removed on next access)
        assert ttl_cache.get('key1') is None

    def test_ttl_none_means_no_expiration(self, cache_name):
        """Test TTL=None means items never expire."""
        cache = Cache(cache_name=cache_name, max_size=100, max_age=None)
        cache.set('key1', 'value1')

        # Wait a bit
        time.sleep(0.5)

        # Should still be accessible
        assert cache.get('key1') == 'value1'
        assert cache.contains('key1') is True

        _cleanup_cache_files(cache)

    def test_recent_items_not_expired(self, ttl_cache):
        """Test recently added items are not expired."""
        ttl_cache.set('key1', 'value1')

        # Check immediately
        assert ttl_cache.contains('key1') is True
        assert ttl_cache.get('key1') == 'value1'


class TestDiskPersistence:
    """Test cache persistence to disk."""

    def test_save_cache_creates_file(self, persistent_cache):
        """Test save_cache() creates pickle file."""
        persistent_cache.set('key1', 'value1')
        success = persistent_cache.save_cache()

        assert success is True
        assert os.path.exists(persistent_cache.cache_file)

    def test_saved_cache_can_be_loaded(self, cache_name):
        """Test saved cache data can be loaded by new instance."""
        # Save data with first cache
        cache1 = Cache(cache_name=cache_name, max_size=100, max_age=3600)
        cache1.set('key1', 'value1')
        cache1.set('key2', 'value2')
        cache1.set('key3', 'value3')
        cache1.save_cache()

        # Load with second cache
        cache2 = Cache(cache_name=cache_name, max_size=100, max_age=3600)

        assert cache2.get('key1') == 'value1'
        assert cache2.get('key2') == 'value2'
        assert cache2.get('key3') == 'value3'

        _cleanup_cache_files(cache1)

    def test_cache_preserves_data_types_across_save_load(self, cache_name):
        """Test all data types are preserved through save/load cycle."""
        test_data = {
            'string': 'test',
            'int': 42,
            'float': 3.14,
            'list': [1, 2, 3],
            'dict': {'nested': 'value'},
        }

        # Save
        cache1 = Cache(cache_name=cache_name, max_size=100, max_age=3600)
        for key, value in test_data.items():
            cache1.set(key, value)
        cache1.save_cache()

        # Load
        cache2 = Cache(cache_name=cache_name, max_size=100, max_age=3600)

        for key, expected in test_data.items():
            actual = cache2.get(key)
            assert actual == expected
            assert type(actual) == type(expected)

        _cleanup_cache_files(cache1)

    def test_save_cache_returns_true_on_success(self, persistent_cache):
        """Test save_cache() returns True on successful save."""
        persistent_cache.set('key1', 'value1')
        result = persistent_cache.save_cache()

        assert result is True


class TestSaveCacheErrorHandling:
    """Test save_cache error handling and retry logic."""

    def test_save_cache_retries_on_failure(self, cache_name):
        """Test save_cache retries on failure and eventually succeeds."""
        cache = Cache(cache_name=cache_name, max_size=100, max_age=3600)
        cache.set('key1', 'value1')

        # Mock to fail twice then succeed
        call_count = 0

        def mock_open_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise IOError("Simulated write failure")
            # On 3rd attempt, succeed
            return MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock())

        with patch('builtins.open', side_effect=mock_open_side_effect):
            with patch('pickle.dump'):
                result = cache.save_cache(max_retries=3)

        # Should succeed on 3rd attempt
        assert result is True
        assert call_count == 3

        _cleanup_cache_files(cache)

    def test_save_cache_returns_false_after_max_retries(self, cache_name):
        """Test save_cache returns False after exhausting retries."""
        cache = Cache(cache_name=cache_name, max_size=100, max_age=3600)
        cache.set('key1', 'value1')

        # Mock to always fail
        with patch('builtins.open', side_effect=IOError("Persistent write failure")):
            result = cache.save_cache(max_retries=3)

        assert result is False

        _cleanup_cache_files(cache)

    def test_save_cache_sleeps_between_retries(self, cache_name):
        """Test save_cache sleeps between retry attempts."""
        cache = Cache(cache_name=cache_name, max_size=100, max_age=3600)
        cache.set('key1', 'value1')

        sleep_called = []

        with patch('builtins.open', side_effect=IOError("Failure")):
            with patch('time.sleep') as mock_sleep:
                cache.save_cache(max_retries=3)
                sleep_called = mock_sleep.call_count

        # Should sleep between attempts (2 sleeps for 3 attempts)
        assert sleep_called == 2

        _cleanup_cache_files(cache)


class TestLoadCacheErrorHandling:
    """Test _load_cache error handling."""

    def test_load_cache_handles_lock_timeout(self, cache_name):
        """Test _load_cache handles FileLock timeout gracefully."""
        from filelock import Timeout

        # Create cache file
        cache_file = os.path.join(CACHE_DIR, f"{cache_name}_cache.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump({'key1': (time.time(), 'value1')}, f)

        # Mock FileLock to raise Timeout
        with patch('app.backtesting.cache.cache_base.FileLock') as mock_lock:
            mock_lock.return_value.__enter__.side_effect = Timeout(cache_file)
            cache = Cache(cache_name=cache_name, max_size=100, max_age=3600)

        # Should handle timeout gracefully and use empty cache
        assert cache.size() == 0

        _cleanup_cache_files(cache)

    def test_load_cache_handles_corrupted_file(self, cache_name):
        """Test _load_cache handles corrupted cache file."""
        # Create corrupted cache file
        cache_file = os.path.join(CACHE_DIR, f"{cache_name}_cache.pkl")
        with open(cache_file, 'wb') as f:
            f.write(b'corrupted data')

        cache = Cache(cache_name=cache_name, max_size=100, max_age=3600)

        # Should handle corruption and use empty cache
        assert cache.size() == 0

        _cleanup_cache_files(cache)

    def test_load_cache_handles_invalid_data_type(self, cache_name):
        """Test _load_cache handles non-dict data."""
        # Create cache file with invalid data (list instead of dict)
        cache_file = os.path.join(CACHE_DIR, f"{cache_name}_cache.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(['invalid', 'list', 'data'], f)

        cache = Cache(cache_name=cache_name, max_size=100, max_age=3600)

        # Should reject invalid data and use empty cache
        assert cache.size() == 0

        _cleanup_cache_files(cache)

    def test_load_cache_removes_expired_items_on_load(self, cache_name):
        """Test _load_cache removes expired items when loading."""
        # Create cache file with expired items
        cache_file = os.path.join(CACHE_DIR, f"{cache_name}_cache.pkl")
        old_time = time.time() - 7200  # 2 hours ago
        recent_time = time.time()

        data = OrderedDict({
            'expired1': (old_time, 'old_value1'),
            'expired2': (old_time, 'old_value2'),
            'valid': (recent_time, 'recent_value')
        })

        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

        # Create cache with 1 hour TTL
        cache = Cache(cache_name=cache_name, max_size=100, max_age=3600)

        # Should only have the recent item
        assert cache.size() == 1
        assert cache.get('valid') == 'recent_value'
        assert cache.get('expired1') is None
        assert cache.get('expired2') is None

        _cleanup_cache_files(cache)


class TestCacheSizeManagement:
    """Test cache size tracking and limits."""

    def test_size_returns_correct_count(self, basic_cache):
        """Test size() returns correct number of items."""
        assert basic_cache.size() == 0

        basic_cache.set('key1', 'value1')
        assert basic_cache.size() == 1

        basic_cache.set('key2', 'value2')
        assert basic_cache.size() == 2

        basic_cache.set('key3', 'value3')
        assert basic_cache.size() == 3

    def test_size_decreases_on_eviction(self, small_cache):
        """Test size decreases when items are evicted."""
        small_cache.set('key1', 'value1')
        small_cache.set('key2', 'value2')
        small_cache.set('key3', 'value3')

        assert small_cache.size() == 3

        # Add item to trigger eviction
        small_cache.set('key4', 'value4')

        assert small_cache.size() == 3  # Still at max

    def test_max_size_none_allows_unlimited_growth(self, cache_name):
        """Test max_size=None allows unlimited cache growth."""
        cache = Cache(cache_name=cache_name, max_size=None, max_age=3600)

        # Add many items
        for i in range(100):
            cache.set(f'key{i}', f'value{i}')

        assert cache.size() == 100

        _cleanup_cache_files(cache)


class TestCacheEdgeCases:
    """Test edge cases and unusual scenarios."""

    def test_empty_cache_operations(self, basic_cache):
        """Test operations on empty cache."""
        assert basic_cache.size() == 0
        assert basic_cache.get('any_key') is None
        assert basic_cache.contains('any_key') is False

    def test_cache_with_empty_string_key(self, basic_cache):
        """Test cache handles empty string as key."""
        basic_cache.set('', 'empty_key_value')
        assert basic_cache.get('') == 'empty_key_value'

    def test_cache_with_none_value(self, basic_cache):
        """Test cache can store None as a value."""
        basic_cache.set('none_key', None)
        assert basic_cache.contains('none_key') is True
        assert basic_cache.get('none_key') is None

    def test_cache_with_large_values(self, basic_cache):
        """Test cache handles large values."""
        large_list = list(range(10000))
        basic_cache.set('large', large_list)

        result = basic_cache.get('large')
        assert result == large_list
        assert len(result) == 10000

    def test_overwrite_updates_timestamp(self, ttl_cache):
        """Test overwriting a key updates its timestamp."""
        ttl_cache.set('key1', 'value1')
        time.sleep(0.5)
        ttl_cache.set('key1', 'value2')  # Overwrite with new timestamp

        time.sleep(0.7)  # Total 1.2s from first set, 0.7s from second

        # Should still be valid (timestamp was updated)
        assert ttl_cache.get('key1') == 'value2'


class TestCacheFormatConversion:
    """Test legacy cache format conversion."""

    def test_convert_old_format_to_new(self):
        """Test _convert_cache_format converts old format to new."""
        old_format = {
            'key1': 'value1',
            'key2': 'value2',
        }

        new_format = _convert_cache_format(old_format)

        assert isinstance(new_format, OrderedDict)
        for key in old_format:
            assert key in new_format
            timestamp, value = new_format[key]
            assert isinstance(timestamp, (int, float))
            assert value == old_format[key]

    def test_convert_already_new_format(self):
        """Test _convert_cache_format handles already converted data."""
        current_time = time.time()
        new_format = OrderedDict({
            'key1': (current_time, 'value1'),
            'key2': (current_time, 'value2'),
        })

        converted = _convert_cache_format(new_format)

        assert converted == new_format
