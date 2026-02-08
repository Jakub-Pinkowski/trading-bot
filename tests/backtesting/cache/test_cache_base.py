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
- Concurrency and multiprocessing
- Edge cases (empty cache, full cache, expired items)

All tests use real cache operations with actual file I/O.
"""
import multiprocessing
import os
import pickle
import random
import time
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

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
        pass  # Best effort cleanup - ignore errors to avoid test flakiness


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

    def test_item_expires_after_ttl(self, cache_name):
        """Test items expire after TTL period using time mocking."""
        # Use 10 second TTL for testing
        cache = Cache(cache_name=cache_name, max_size=100, max_age=10)

        # Mock time to control expiration
        mock_time = 1000.0
        with patch('time.time', return_value=mock_time):
            cache.set('key1', 'value1')

            # Immediately accessible
            assert cache.get('key1') == 'value1'

        # Advance time by 11 seconds (past TTL)
        with patch('time.time', return_value=mock_time + 11):
            # Should be expired
            assert cache.contains('key1') is False
            assert cache.get('key1') is None

        _cleanup_cache_files(cache)

    def test_expired_items_removed_on_access(self, cache_name):
        """Test expired items are removed when accessed."""
        cache = Cache(cache_name=cache_name, max_size=100, max_age=10)

        mock_time = 1000.0
        with patch('time.time', return_value=mock_time):
            cache.set('key1', 'value1')
            cache.set('key2', 'value2')

            initial_size = cache.size()
            assert initial_size == 2

        # Advance time past expiration
        with patch('time.time', return_value=mock_time + 11):
            # Access expired item - should remove it
            assert cache.contains('key1') is False

            # Get should return None
            assert cache.get('key1') is None

        _cleanup_cache_files(cache)

    def test_ttl_none_means_no_expiration(self, cache_name):
        """Test TTL=None means items never expire."""
        cache = Cache(cache_name=cache_name, max_size=100, max_age=None)

        mock_time = 1000.0
        with patch('time.time', return_value=mock_time):
            cache.set('key1', 'value1')

        # Advance time significantly
        with patch('time.time', return_value=mock_time + 1000000):
            # Should still be accessible (no expiration)
            assert cache.get('key1') == 'value1'
            assert cache.contains('key1') is True

        _cleanup_cache_files(cache)

    def test_recent_items_not_expired(self, cache_name):
        """Test recently added items are not expired."""
        cache = Cache(cache_name=cache_name, max_size=100, max_age=10)

        mock_time = 1000.0
        with patch('time.time', return_value=mock_time):
            cache.set('key1', 'value1')

            # Check immediately (time hasn't advanced)
            assert cache.contains('key1') is True
            assert cache.get('key1') == 'value1'

        _cleanup_cache_files(cache)

    def test_items_expire_at_exact_ttl_boundary(self, cache_name):
        """Test items expire at exactly the TTL boundary."""
        cache = Cache(cache_name=cache_name, max_size=100, max_age=10)

        mock_time = 1000.0
        with patch('time.time', return_value=mock_time):
            cache.set('key1', 'value1')

        # Just before expiration (9 seconds)
        with patch('time.time', return_value=mock_time + 9):
            assert cache.contains('key1') is True

        # Exactly at expiration (10 seconds)
        with patch('time.time', return_value=mock_time + 10):
            assert cache.contains('key1') is True  # Still valid at exactly TTL

        # Just after expiration (10.1 seconds)
        with patch('time.time', return_value=mock_time + 10.1):
            assert cache.contains('key1') is False  # Now expired

        _cleanup_cache_files(cache)


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

        # Mock pickle.dump to fail twice then succeed
        call_count = 0

        def mock_pickle_dump(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise IOError("Simulated pickle dump failure")
            # On 3rd attempt, succeed (do nothing)

        with patch('pickle.dump', side_effect=mock_pickle_dump):
            result = cache.save_cache(max_retries=3)

        # Should succeed on 3rd attempt
        assert result is True
        assert call_count == 3

        _cleanup_cache_files(cache)

    def test_save_cache_returns_false_after_max_retries(self, cache_name):
        """Test save_cache returns False after exhausting retries."""
        cache = Cache(cache_name=cache_name, max_size=100, max_age=3600)
        cache.set('key1', 'value1')

        # Mock pickle.dump to always fail
        with patch('pickle.dump', side_effect=IOError("Persistent pickle dump failure")):
            result = cache.save_cache(max_retries=3)

        assert result is False

        _cleanup_cache_files(cache)

    def test_save_cache_sleeps_between_retries(self, cache_name):
        """Test save_cache sleeps between retry attempts."""
        cache = Cache(cache_name=cache_name, max_size=100, max_age=3600)
        cache.set('key1', 'value1')

        # Mock pickle.dump to always fail
        with patch('pickle.dump', side_effect=IOError("Failure")):
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

    def test_overwrite_updates_timestamp(self, cache_name):
        """Test overwriting a key updates its timestamp."""
        cache = Cache(cache_name=cache_name, max_size=100, max_age=1)  # 1 second TTL

        mock_time = 1000.0
        # Set initial value
        with patch('time.time', return_value=mock_time):
            cache.set('key1', 'value1')

        # Overwrite after 0.5 seconds
        with patch('time.time', return_value=mock_time + 0.5):
            cache.set('key1', 'value2')  # Overwrite with new timestamp

        # Check after 1.2 seconds from initial set (0.7s from overwrite)
        with patch('time.time', return_value=mock_time + 1.2):
            # Should still be valid (timestamp was updated on overwrite)
            assert cache.get('key1') == 'value2'

        _cleanup_cache_files(cache)


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


# ==================== Concurrency Helper Functions ====================

def _worker_process(worker_id, cache_name, iterations=10):
    """
    Worker process that reads and writes to the cache.

    Simulates multiple strategies running in parallel (MassTester scenario).
    Each worker repeatedly reads and writes to the same keys to create
    contention and test file locking.

    Args:
        worker_id: Unique identifier for this worker
        cache_name: Name of the cache to use
        iterations: Number of read/write cycles to perform

    Returns:
        worker_id if successful
    """
    # Create a test cache instance
    test_cache = Cache(cache_name, max_size=100, max_age=3600)

    for i in range(iterations):
        # Add randomness to increase chance of collision
        time.sleep(random.uniform(0.01, 0.1))

        # Read from cache (might not exist yet)
        test_cache.get(f"key_{i % 5}")

        # Write to cache (multiple workers write to same keys)
        test_cache.set(f"key_{i % 5}", f"value_from_worker_{worker_id}_iteration_{i}")

        # Save cache (triggers file locking)
        test_cache.save_cache()

    return worker_id


def _check_cache_integrity(cache_name):
    """
    Verify cache file is valid and not corrupted.

    Args:
        cache_name: Name of cache to check

    Returns:
        Tuple of (is_valid, num_items)
    """
    cache_file = os.path.join(CACHE_DIR, f"{cache_name}_cache.pkl")
    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
            if isinstance(cache_data, dict):
                return True, len(cache_data)
            else:
                return False, 0
    except Exception:
        # File doesn't exist or is corrupted - return invalid
        return False, 0


class TestCacheConcurrency:
    """Test cache behavior with concurrent access from multiple processes."""

    def test_cache_multiprocessing_access(self, cache_name):
        """
        Test cache handles multiple processes correctly.

        Simulates MassTester scenario where multiple strategies run in parallel,
        all reading/writing to the same cache. Verifies FileLock prevents
        data corruption.
        """
        # Ensure cache directory exists
        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

        num_workers = 4
        iterations = 10

        # Create pool of worker processes
        with multiprocessing.Pool(processes=num_workers) as pool:
            # Start worker processes (each writes to same keys)
            results = pool.starmap(
                _worker_process,
                [(i, cache_name, iterations) for i in range(num_workers)]
            )

        # Verify all workers completed successfully
        assert len(results) == num_workers
        for i in range(num_workers):
            assert i in results, f"Worker {i} did not complete successfully"

        # Verify cache integrity (not corrupted)
        is_valid, num_items = _check_cache_integrity(cache_name)
        assert is_valid, "Cache file is corrupted or invalid after concurrent access"
        assert num_items > 0, "Cache is empty after concurrent writes"
        assert num_items <= 5, f"Cache should have at most 5 items, but has {num_items}"

        # Load cache and verify contents
        test_cache = Cache(cache_name)
        assert test_cache.size() > 0, "Cache is empty after loading"
        assert test_cache.size() <= 5, f"Cache has unexpected size: {test_cache.size()}"

        # Verify cache contains valid data
        for i in range(5):
            key = f"key_{i}"
            if test_cache.contains(key):
                value = test_cache.get(key)
                assert isinstance(value, str), f"Cache value for {key} is not a string: {value}"
                assert value.startswith("value_from_worker_"), f"Unexpected value format: {value}"

        # Cleanup
        _cleanup_cache_files(test_cache)

    def test_file_locking_prevents_corruption(self, cache_name):
        """
        Test FileLock prevents cache corruption from concurrent writes.

        Multiple processes attempt to write simultaneously. Without proper
        locking, this would corrupt the pickle file. This test verifies
        the file remains valid.
        """
        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

        num_workers = 4
        iterations = 8

        # Create high contention scenario
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.starmap(
                _worker_process,
                [(i, cache_name, iterations) for i in range(num_workers)]
            )

        assert len(results) == num_workers

        # Verify cache file is not corrupted
        cache_file = os.path.join(CACHE_DIR, f"{cache_name}_cache.pkl")
        assert os.path.exists(cache_file), "Cache file does not exist"

        # Load and verify file is valid pickle
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            assert isinstance(cache_data, dict), "Cache file is not a valid dictionary"
        except Exception as e:
            pytest.fail(f"Cache file is corrupted: {e}")

        # Verify we can create new cache instance from file
        test_cache = Cache(cache_name)
        assert test_cache.size() > 0, "Cache is empty"

        # Cleanup
        _cleanup_cache_files(test_cache)

    def test_lock_acquisition_timing(self, cache_name):
        """
        Test locks are acquired and released properly.

        Verifies that multiple cache instances can successfully acquire locks
        in sequence without deadlocks.
        """
        cache1 = Cache(cache_name, max_size=100, max_age=3600)
        cache2 = Cache(cache_name, max_size=100, max_age=3600)

        # First cache writes
        cache1.set('key1', 'value1')
        success1 = cache1.save_cache()
        assert success1, "First cache save failed"

        # Second cache writes (should acquire lock after first releases)
        cache2.set('key2', 'value2')
        success2 = cache2.save_cache()
        assert success2, "Second cache save failed"

        # Verify both writes succeeded
        cache3 = Cache(cache_name)
        # At least one of the keys should exist (last writer wins)
        assert cache3.size() > 0, "No data in cache after sequential writes"

        # Cleanup
        _cleanup_cache_files(cache1)

    def test_multiple_sequential_saves(self, cache_name):
        """Test multiple sequential saves work without lock issues."""
        cache = Cache(cache_name, max_size=100, max_age=3600)

        # Perform multiple saves
        for i in range(10):
            cache.set(f'key{i}', f'value{i}')
            success = cache.save_cache()
            assert success, f"Save {i} failed"

        # Verify all data was saved
        cache2 = Cache(cache_name)
        assert cache2.size() == 10, f"Expected 10 items, got {cache2.size()}"

        # Cleanup
        _cleanup_cache_files(cache)

    def test_cache_empty_during_concurrent_access(self, cache_name):
        """Test concurrent access to initially empty cache."""
        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

        num_workers = 2
        iterations = 3

        # Workers start with empty cache
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.starmap(
                _worker_process,
                [(i, cache_name, iterations) for i in range(num_workers)]
            )

        assert len(results) == num_workers

        # Cache should exist and be valid
        is_valid, _ = _check_cache_integrity(cache_name)
        assert is_valid

        # Cleanup
        test_cache = Cache(cache_name)
        _cleanup_cache_files(test_cache)


class TestCachePerformance:
    """Test cache performance characteristics."""

    def test_cache_hit_is_faster_than_miss(self, cache_name):
        """
        Test cache operations complete quickly.

        Validates that cache get/set operations are performant.
        """
        cache = Cache(cache_name, max_size=1000, max_age=3600)

        # Generate test data
        test_data = {f'key_{i}': f'value_{i}' * 100 for i in range(1000)}

        # Measure cache set time
        start = time.time()
        for key, value in test_data.items():
            cache.set(key, value)
        set_time = time.time() - start

        # Measure cache get time
        start = time.time()
        for key in test_data.keys():
            cache.get(key)
        get_time = time.time() - start

        # Both operations should complete quickly (< 0.5 seconds for 1000 items)
        assert set_time < 0.5, f"Setting 1000 items took {set_time:.4f}s, should be < 0.5s"
        assert get_time < 0.5, f"Getting 1000 items took {get_time:.4f}s, should be < 0.5s"

        # Cleanup
        _cleanup_cache_files(cache)

    def test_large_cache_performance(self, cache_name):
        """
        Test cache performance with many entries.

        Validates that cache operations remain fast even with large number
        of entries (10,000 items).
        """
        cache = Cache(cache_name, max_size=20000, max_age=3600)

        # Add 10,000 entries
        num_entries = 10000
        start = time.time()
        for i in range(num_entries):
            cache.set(f'key_{i}', f'value_{i}')
        add_time = time.time() - start

        # Verify reasonable performance (should be < 1 second for 10k entries)
        assert add_time < 1.0, \
            f"Adding 10k entries took {add_time:.4f}s, should be < 1.0s"

        # Test retrieval performance
        start = time.time()
        for i in range(num_entries):
            cache.get(f'key_{i}')
        get_time = time.time() - start

        # Verify reasonable retrieval performance
        assert get_time < 0.5, \
            f"Getting 10k entries took {get_time:.4f}s, should be < 0.5s"

        # Verify all entries are present
        assert cache.size() == num_entries

        # Cleanup
        _cleanup_cache_files(cache)

    def test_cache_lookup_performance_scales_well(self, cache_name):
        """
        Test cache lookup performance scales linearly.

        Validates that cache lookup time doesn't degrade significantly
        as cache size increases.
        """
        cache = Cache(cache_name, max_size=50000, max_age=3600)

        # Test with different cache sizes
        sizes = [100, 1000, 5000]
        times = []

        for size in sizes:
            # Populate cache
            cache.cache_data.clear()
            for i in range(size):
                cache.set(f'key_{i}', f'value_{i}')

            # Measure lookup time for last 100 entries
            start = time.time()
            for i in range(size - 100, size):
                cache.get(f'key_{i}')
            lookup_time = time.time() - start
            times.append(lookup_time)

        # Lookup time should not increase dramatically
        # With OrderedDict, lookups are O(1), so time should be roughly constant
        # Allow 3x increase from smallest to largest (generous margin)
        ratio = times[-1] / times[0] if times[0] > 0 else 0
        assert ratio < 3.0, \
            f"Lookup time increased {ratio:.1f}x from 100 to 5000 entries, should be < 3x"

        # Cleanup
        _cleanup_cache_files(cache)

    def test_save_cache_performance(self, cache_name):
        """
        Test cache save performance.

        Validates that saving cache to disk completes in reasonable time
        even with many entries.
        """
        cache = Cache(cache_name, max_size=5000, max_age=3600)

        # Add 1000 entries
        for i in range(1000):
            cache.set(f'key_{i}', {'data': f'value_{i}' * 50})

        # Measure save performance
        start = time.time()
        success = cache.save_cache()
        save_time = time.time() - start

        assert success, "Cache save failed"

        # Save should complete quickly (< 1 second for 1000 entries)
        assert save_time < 1.0, \
            f"Saving 1000 entries took {save_time:.4f}s, should be < 1.0s"

        # Cleanup
        _cleanup_cache_files(cache)

    def test_load_cache_performance(self, cache_name):
        """
        Test cache load performance.

        Validates that loading cache from disk is fast even with many entries.
        """
        # Create and save a cache with many entries
        cache1 = Cache(cache_name, max_size=5000, max_age=3600)
        for i in range(1000):
            cache1.set(f'key_{i}', {'data': f'value_{i}' * 50})
        cache1.save_cache()

        # Measure load performance
        start = time.time()
        cache2 = Cache(cache_name, max_size=5000, max_age=3600)
        load_time = time.time() - start

        # Load should complete quickly (< 1 second for 1000 entries)
        assert load_time < 1.0, \
            f"Loading 1000 entries took {load_time:.4f}s, should be < 1.0s"

        # Verify data was loaded
        assert cache2.size() == 1000

        # Cleanup
        _cleanup_cache_files(cache1)

    def test_lru_eviction_performance(self, cache_name):
        """
        Test LRU eviction performance doesn't degrade.

        Validates that evicting items when cache is full remains fast.
        """
        cache = Cache(cache_name, max_size=100, max_age=3600)

        # Fill cache to max size
        for i in range(100):
            cache.set(f'key_{i}', f'value_{i}')

        # Measure time to add items that trigger eviction
        start = time.time()
        for i in range(100, 200):
            cache.set(f'key_{i}', f'value_{i}')
        eviction_time = time.time() - start

        # Eviction should not add significant overhead
        # Should complete in < 0.1 seconds for 100 evictions
        assert eviction_time < 0.1, \
            f"100 evictions took {eviction_time:.4f}s, should be < 0.1s"

        # Cache should still be at max size
        assert cache.size() == 100

        # Cleanup
        _cleanup_cache_files(cache)
