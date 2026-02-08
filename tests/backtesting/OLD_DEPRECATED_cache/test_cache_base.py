import os
import pickle
import time
from unittest.mock import patch, mock_open, MagicMock

import pytest
from filelock import Timeout

from app.backtesting.cache.cache_base import Cache
from config import CACHE_DIR


@pytest.fixture
def temp_cache_file(tmp_path):
    """Create a temporary cache file for testing."""
    cache_file = tmp_path / "test_cache.pkl"
    return str(cache_file)


@pytest.fixture
def mock_cache_dir(monkeypatch):
    """Mock the CACHE_DIR to use a temporary directory."""
    temp_dir = os.path.join(os.path.dirname(__file__), "temp_cache")
    os.makedirs(temp_dir, exist_ok=True)
    monkeypatch.setattr("app.backtesting.cache.cache_base.CACHE_DIR", temp_dir)
    yield temp_dir
    # Clean up
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)


def test_cache_initialization():
    """Test that the cache initializes correctly."""
    cache = Cache("test")
    assert cache.cache_name == "test"
    assert cache.cache_file == os.path.join(CACHE_DIR, "test_cache.pkl")
    assert cache.lock_file == os.path.join(CACHE_DIR, "test_cache.lock")
    assert isinstance(cache.cache_data, dict)
    assert len(cache.cache_data) == 0
    assert cache.max_size == 1000
    assert cache.max_age == 86400


def test_cache_set_and_get():
    """Test setting and getting values from the cache."""
    cache = Cache("test")

    # Test setting and getting a value
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"

    # Test getting a non-existent key
    assert cache.get("non_existent_key") is None

    # Test getting a non-existent key with a default value
    assert cache.get("non_existent_key", "default") == "default"


def test_cache_contains():
    """Test checking if a key exists in the cache."""
    cache = Cache("test")

    # Key doesn't exist initially
    assert not cache.contains("key1")

    # Add the key
    cache.set("key1", "value1")
    assert cache.contains("key1")


def test_cache_clear():
    """Test clearing the cache."""
    cache = Cache("test")

    # Add some items to the cache
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    assert cache.size() == 2

    # Clear the cache
    cache.cache_data.clear()
    assert cache.size() == 0
    assert not cache.contains("key1")
    assert not cache.contains("key2")


def test_cache_size():
    """Test getting the size of the cache."""
    cache = Cache("test")

    # Empty cache
    assert cache.size() == 0

    # Add items
    cache.set("key1", "value1")
    assert cache.size() == 1

    cache.set("key2", "value2")
    assert cache.size() == 2

    # Overwrite an existing key
    cache.set("key1", "new_value")
    assert cache.size() == 2


def test_load_cache_file_exists(mock_cache_dir):
    """Test loading a cache file that exists."""
    # Create a cache file with some data
    cache_data = {"key1": "value1", "key2": "value2"}
    cache_file = os.path.join(mock_cache_dir, "test_cache.pkl")

    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

    # Mock the FileLock to verify it's used
    with patch('app.backtesting.cache.cache_base.FileLock', autospec=True) as mock_filelock:
        # Set up the mock to return a context manager
        mock_lock_instance = MagicMock()
        mock_filelock.return_value = mock_lock_instance
        mock_lock_instance.__enter__.return_value = mock_lock_instance
        mock_lock_instance.__exit__.return_value = None

        # Initialize cache, which should load the file
        cache = Cache("test")

        # Verify FileLock was called with the correct lock file
        mock_filelock.assert_called_once_with(os.path.join(mock_cache_dir, "test_cache.lock"), timeout=60)
        # Verify the lock's __enter__ method was called (context manager was used)
        mock_lock_instance.__enter__.assert_called_once()

    # Verify the data was loaded
    # The cache now stores (timestamp, value) tuples, so we can't directly compare
    assert len(cache.cache_data) == 2
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"


def test_load_cache_file_does_not_exist():
    """Test loading a cache file that doesn't exist."""
    # Use a unique name to ensure the file doesn't exist
    cache = Cache("nonexistent_test")

    # Verify the cache is empty
    assert cache.cache_data == {}
    assert cache.size() == 0


def test_load_cache_invalid_data(mock_cache_dir):
    """Test loading a cache file with invalid data."""
    # Create a cache file with invalid data (not a dictionary)
    cache_file = os.path.join(mock_cache_dir, "test_cache.pkl")

    with open(cache_file, 'wb') as f:
        pickle.dump("not a dictionary", f)

    # Initialize cache, which should handle the invalid data
    cache = Cache("test")

    # Verify the cache is empty
    assert cache.cache_data == {}
    assert cache.size() == 0


def test_load_cache_exception_handling():
    """Test handling exceptions when loading the cache."""
    # Mock os.path.exists to return True so that the code will try to acquire the lock
    with patch('os.path.exists', return_value=True):
        # Mock the FileLock to verify it's used even when there's an exception
        with patch('app.backtesting.cache.cache_base.FileLock', autospec=True) as mock_filelock:
            # Set up the mock to return a context manager
            mock_lock_instance = MagicMock()
            mock_filelock.return_value = mock_lock_instance
            mock_lock_instance.__enter__.return_value = mock_lock_instance
            mock_lock_instance.__exit__.return_value = None

            with patch("builtins.open", mock_open()) as mock_file:
                mock_file.side_effect = Exception("Test exception")

                # Initialize cache, which should handle the exception
                cache = Cache("test")

                # Verify FileLock was called
                mock_filelock.assert_called_once()
                # Verify the lock's __enter__ method was called (context manager was used)
                mock_lock_instance.__enter__.assert_called_once()

            # Verify the cache is empty
            assert cache.cache_data == {}
            assert cache.size() == 0


def test_load_cache_timeout_handling():
    """Test handling Timeout exception when loading the cache."""
    # Mock os.path.exists to return True so that the code will try to acquire the lock
    with patch('os.path.exists', return_value=True):
        # Mock the FileLock to raise Timeout exception
        with patch('app.backtesting.cache.cache_base.FileLock', autospec=True) as mock_filelock:
            # Set up the mock to raise Timeout when entering the context manager
            mock_lock_instance = MagicMock()
            mock_filelock.return_value = mock_lock_instance
            mock_lock_instance.__enter__.side_effect = Timeout("Lock acquisition timeout")

            # Initialize cache, which should handle the Timeout exception gracefully
            cache = Cache("test_timeout")

            # Verify FileLock was called
            mock_filelock.assert_called_once()
            # Verify the lock's __enter__ method was called (context manager was attempted)
            mock_lock_instance.__enter__.assert_called_once()

        # Verify the cache is empty (should proceed without cache after timeout)
        assert cache.cache_data == {}
        assert cache.size() == 0


def test_load_cache_timeout_vs_general_exception():
    """Test that Timeout exceptions are handled differently from general exceptions."""
    # Test Timeout exception
    with patch('os.path.exists', return_value=True):
        with patch('app.backtesting.cache.cache_base.FileLock', autospec=True) as mock_filelock:
            mock_lock_instance = MagicMock()
            mock_filelock.return_value = mock_lock_instance
            mock_lock_instance.__enter__.side_effect = Timeout("Lock timeout")

            # This should log a warning (not an error) for Timeout
            cache_timeout = Cache("test_timeout")
            assert cache_timeout.size() == 0

    # Test general exception
    with patch('os.path.exists', return_value=True):
        with patch('app.backtesting.cache.cache_base.FileLock', autospec=True) as mock_filelock:
            mock_lock_instance = MagicMock()
            mock_filelock.return_value = mock_lock_instance
            mock_lock_instance.__enter__.return_value = mock_lock_instance
            mock_lock_instance.__exit__.return_value = None

            with patch("builtins.open", mock_open()) as mock_file:
                mock_file.side_effect = IOError("File read error")

                # This should log an error for general exceptions
                cache_error = Cache("test_error")
                assert cache_error.size() == 0


def test_save_cache_timeout_handling():
    """Test handling Timeout exception when saving the cache."""
    cache = Cache("test_save_timeout")
    cache.set("key1", "value1")

    # Mock FileLock to raise Timeout on save
    with patch('app.backtesting.cache.cache_base.FileLock', autospec=True) as mock_filelock:
        mock_lock_instance = MagicMock()
        mock_filelock.return_value = mock_lock_instance
        mock_lock_instance.__enter__.side_effect = Timeout("Lock acquisition timeout")

        # Mock sleep to speed up test
        with patch('time.sleep'):
            result = cache.save_cache()

        # Should return False after all retries fail
        assert result is False

        # Verify FileLock was called 3 times (default retry attempts)
        assert mock_filelock.call_count == 3

    # Verify cache data is still intact
    assert cache.get("key1") == "value1"


def test_save_cache(mock_cache_dir):
    """Test saving the cache to disk."""
    cache = Cache("test")

    # Add some data to the cache
    cache.set("key1", "value1")
    cache.set("key2", "value2")

    # Mock the FileLock to verify it's used
    with patch('app.backtesting.cache.cache_base.FileLock', autospec=True) as mock_filelock:
        # Set up the mock to return a context manager
        mock_lock_instance = MagicMock()
        mock_filelock.return_value = mock_lock_instance
        mock_lock_instance.__enter__.return_value = mock_lock_instance
        mock_lock_instance.__exit__.return_value = None

        # Save the cache
        cache.save_cache()

        # Verify FileLock was called with the correct lock file
        mock_filelock.assert_called_once_with(os.path.join(mock_cache_dir, "test_cache.lock"), timeout=60)
        # Verify the lock's __enter__ method was called (context manager was used)
        mock_lock_instance.__enter__.assert_called_once()

    # Verify the file was created
    cache_file = os.path.join(mock_cache_dir, "test_cache.pkl")
    assert os.path.exists(cache_file)

    # Load the file and verify the data
    with open(cache_file, 'rb') as f:
        loaded_data = pickle.load(f)

    # The saved data now contains timestamps, so we need to extract the values
    assert len(loaded_data) == 2
    assert "key1" in loaded_data
    assert "key2" in loaded_data
    assert isinstance(loaded_data["key1"], tuple)
    assert isinstance(loaded_data["key2"], tuple)
    assert loaded_data["key1"][1] == "value1"
    assert loaded_data["key2"][1] == "value2"


def test_save_cache_exception_handling():
    """Test handling exceptions when saving the cache with retry mechanism."""
    cache = Cache("test")
    cache.set("key1", "value1")

    # Mock the FileLock to verify it's used even when there's an exception
    with patch('app.backtesting.cache.cache_base.FileLock', autospec=True) as mock_filelock:
        # Set up the mock to return a context manager
        mock_lock_instance = MagicMock()
        mock_filelock.return_value = mock_lock_instance
        mock_lock_instance.__enter__.return_value = mock_lock_instance
        mock_lock_instance.__exit__.return_value = None

        with patch("builtins.open", mock_open()) as mock_file:
            mock_file.side_effect = Exception("Test exception")

            # This should not raise an exception, but should return False after 3 retries
            with patch('time.sleep'):  # Mock sleep to speed up test
                result = cache.save_cache()

            assert result is False, "save_cache should return False after all retries fail"

            # Verify FileLock was called 3 times (max_retries default = 3)
            assert mock_filelock.call_count == 3, f"Expected 3 retry attempts, got {mock_filelock.call_count}"

            # Verify the lock's __enter__ method was called 3 times
            assert mock_lock_instance.__enter__.call_count == 3

        # The cache data should still be intact
        assert cache.get("key1") == "value1"


def test_save_cache_retry_success():
    """Test that save_cache succeeds after a retry."""
    cache = Cache("test")
    cache.set("key1", "value1")

    # Mock FileLock
    with patch('app.backtesting.cache.cache_base.FileLock', autospec=True) as mock_filelock:
        mock_lock_instance = MagicMock()
        mock_filelock.return_value = mock_lock_instance
        mock_lock_instance.__enter__.return_value = mock_lock_instance
        mock_lock_instance.__exit__.return_value = None

        # Mock open to fail once, then succeed
        with patch("builtins.open", mock_open()) as mock_file:
            # First call fails, second call succeeds
            mock_file.side_effect = [Exception("Temporary failure"), mock_open()()]

            # Mock sleep to speed up test
            with patch('time.sleep'):
                result = cache.save_cache()

            # Should succeed on second attempt
            assert result is True, "save_cache should return True after successful retry"

            # Verify FileLock was called twice (failed once, succeeded on retry)
            assert mock_filelock.call_count == 2, f"Expected 2 attempts, got {mock_filelock.call_count}"


def test_cache_with_complex_data():
    """Test the cache with complex data types."""
    cache = Cache("test")

    # Test with a list
    cache.set("list", [1, 2, 3])
    assert cache.get("list") == [1, 2, 3]

    # Test with a dictionary
    cache.set("dict", {"a": 1, "b": 2})
    assert cache.get("dict") == {"a": 1, "b": 2}

    # Test with a nested structure
    complex_data = {
        "name": "test",
        "values": [1, 2, 3],
        "metadata": {
            "created": "2023-01-01",
            "tags": ["tag1", "tag2"]
        }
    }
    cache.set("complex", complex_data)
    assert cache.get("complex") == complex_data


def test_cache_performance_with_large_dataset():
    """Test cache performance with a large number of items."""
    cache = Cache("test_performance", max_size=20000)  # Use a larger max_size for this test

    # Add a large number of items to the cache
    num_items = 10000
    start_time = time.time()

    for i in range(num_items):
        cache.set(f"key_{i}", f"value_{i}")

    set_time = time.time() - start_time

    # Verify all items were added
    assert cache.size() == num_items

    # Test retrieval performance
    start_time = time.time()

    for i in range(num_items):
        value = cache.get(f"key_{i}")
        assert value == f"value_{i}"

    get_time = time.time() - start_time

    # Test contains performance
    start_time = time.time()

    for i in range(num_items):
        assert cache.contains(f"key_{i}")

    contains_time = time.time() - start_time

    # Log performance metrics (these are not strict assertions, just informational)
    # Performance metrics are measured but not printed to keep test output clean

    # Ensure operations complete in a reasonable time
    # These thresholds are very generous and should pass on any modern system
    # They're mainly to catch catastrophic performance issues
    assert set_time < 2.0, "Setting items took too long"
    assert get_time < 2.0, "Getting items took too long"
    assert contains_time < 2.0, "Checking contains took too long"


def test_cache_file_corruption(mock_cache_dir):
    """Test handling of corrupted cache files."""
    # Create a cache file path
    cache_file = os.path.join(mock_cache_dir, "corrupted_cache.pkl")

    # Write corrupted data to the file
    with open(cache_file, 'wb') as f:
        f.write(b'This is not a valid pickle file')

    # Initialize cache, which should handle the corrupted file
    cache = Cache("corrupted")

    # Verify the cache is empty
    assert cache.cache_data == {}
    assert cache.size() == 0

    # Test with a partially corrupted pickle file (truncated)
    with open(cache_file, 'wb') as f:
        # Start with a valid pickle header but then truncate it
        pickle.dump({"key1": "value1"}, f)
        f.truncate(10)  # Truncate to first 10 bytes

    # Initialize cache again, which should handle the corrupted file
    cache = Cache("corrupted")

    # Verify the cache is empty
    assert cache.cache_data == {}
    assert cache.size() == 0


def test_cache_with_different_key_types():
    """Test the cache with different types of keys."""
    cache = Cache("test_keys")

    # Test with string keys
    cache.set("string_key", "string_value")
    assert cache.get("string_key") == "string_value"

    # Test with integer keys
    cache.set(42, "integer_value")
    assert cache.get(42) == "integer_value"

    # Test with float keys
    cache.set(3.14, "float_value")
    assert cache.get(3.14) == "float_value"

    # Test with tuple keys (hashable)
    cache.set((1, 2, 3), "tuple_value")
    assert cache.get((1, 2, 3)) == "tuple_value"

    # Test with complex tuple keys
    complex_key = ("macd", "ZW", (12, 26, 9), ("2023-01-01", "2023-12-31"))
    cache.set(complex_key, "complex_tuple_value")
    assert cache.get(complex_key) == "complex_tuple_value"

    # Test with boolean keys
    cache.set(True, "true_value")
    cache.set(False, "false_value")
    assert cache.get(True) == "true_value"
    assert cache.get(False) == "false_value"

    # Test with None as a key
    cache.set(None, "none_value")
    assert cache.get(None) == "none_value"

    # Verify all keys are in the cache
    assert cache.size() == 8
    assert cache.contains("string_key")
    assert cache.contains(42)
    assert cache.contains(3.14)
    assert cache.contains((1, 2, 3))
    assert cache.contains(complex_key)
    assert cache.contains(True)
    assert cache.contains(False)
    assert cache.contains(None)


def test_cache_no_size_limit():
    """Test that the cache doesn't limit size when max_size is None."""
    # Create a cache with no size limit
    cache = Cache("test_no_size_limit", max_size=None)

    # Add a large number of items to the cache
    num_items = 100
    for i in range(num_items):
        cache.set(f"key_{i}", f"value_{i}")

    # Verify all items are in the cache
    assert cache.size() == num_items
    for i in range(num_items):
        assert cache.contains(f"key_{i}")
        assert cache.get(f"key_{i}") == f"value_{i}"

    # Test that _enforce_size_limit doesn't remove anything
    cache._enforce_size_limit()

    # Verify all items are still in the cache
    assert cache.size() == num_items
    for i in range(num_items):
        assert cache.contains(f"key_{i}")
        assert cache.get(f"key_{i}") == f"value_{i}"


def test_cache_no_expiration():
    """Test that the cache doesn't expire items when max_age is None."""
    # Create a cache with no expiration
    cache = Cache("test_no_expiration", max_age=None)

    # Add items to the cache
    cache.set("key1", "value1")
    cache.set("key2", "value2")

    # Verify items are in the cache
    assert cache.size() == 2
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"

    # Wait for a short time
    time.sleep(0.5)

    # Verify items are still in the cache
    assert cache.contains("key1")
    assert cache.contains("key2")
    assert cache.size() == 2

    # Test that _remove_expired_items doesn't remove anything
    cache._remove_expired_items()

    # Verify items are still in the cache
    assert cache.size() == 2
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"


def test_cache_max_age_and_expiration():
    """Test that the cache enforces the max_age limit and expires old items."""
    # Create a cache with a small max_age (1 second)
    cache = Cache("test_max_age", max_age=1)

    # Add items to the cache
    cache.set("key1", "value1")
    cache.set("key2", "value2")

    # Verify items are in the cache
    assert cache.size() == 2
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"

    # Wait for items to expire
    time.sleep(1.5)

    # Verify items are expired when accessed
    assert not cache.contains("key1")
    assert not cache.contains("key2")
    assert cache.size() == 0

    # Add new items
    cache.set("key3", "value3")

    # Verify new items are in the cache
    assert cache.size() == 1
    assert cache.get("key3") == "value3"

    # Test that _remove_expired_items works
    cache.set("key4", "value4")
    time.sleep(1.5)

    # This should trigger _remove_expired_items
    cache._remove_expired_items()

    # Verify all items are expired
    assert cache.size() == 0


def test_cache_max_size_and_lru():
    """Test that the cache enforces the max_size limit and uses LRU eviction."""
    # Create a cache with a small max_size
    cache = Cache("test_max_size", max_size=3)

    # Add items to the cache
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")

    # Verify all items are in the cache
    assert cache.size() == 3
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"

    # Access key1 to make it the most recently used
    cache.get("key1")

    # Add a new item, which should evict the least recently used item (key2)
    cache.set("key4", "value4")

    # Verify the cache size is still 3
    assert cache.size() == 3

    # Verify key2 was evicted
    assert not cache.contains("key2")

    # Verify the other keys are still in the cache
    assert cache.get("key1") == "value1"
    assert cache.get("key3") == "value3"
    assert cache.get("key4") == "value4"

    # Access key3 to make it the most recently used
    cache.get("key3")

    # Add another item, which should evict the least recently used item (key1)
    cache.set("key5", "value5")

    # Verify the cache size is still 3
    assert cache.size() == 3

    # Verify key1 was evicted
    assert not cache.contains("key1")

    # Verify the other keys are still in the cache
    assert cache.get("key3") == "value3"
    assert cache.get("key4") == "value4"
    assert cache.get("key5") == "value5"


def test_cache_update_operations():
    """Test updating existing values in the cache."""
    cache = Cache("test_updates")

    # Add initial values
    cache.set("key1", "value1")
    cache.set("key2", [1, 2, 3])
    cache.set("key3", {"a": 1, "b": 2})

    # Verify initial values
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == [1, 2, 3]
    assert cache.get("key3") == {"a": 1, "b": 2}
    assert cache.size() == 3

    # Update values
    cache.set("key1", "updated_value1")
    cache.set("key2", [4, 5, 6])
    cache.set("key3", {"c": 3, "d": 4})

    # Verify updated values
    assert cache.get("key1") == "updated_value1"
    assert cache.get("key2") == [4, 5, 6]
    assert cache.get("key3") == {"c": 3, "d": 4}
    assert cache.size() == 3  # Size should remain the same

    # Test updating mutable objects
    list_value = [1, 2, 3]
    dict_value = {"a": 1, "b": 2}

    cache.set("mutable_list", list_value)
    cache.set("mutable_dict", dict_value)

    # Modify the original objects
    list_value.append(4)
    dict_value["c"] = 3

    # Verify that the cached values reflect the changes (since they're the same objects)
    assert cache.get("mutable_list") == [1, 2, 3, 4]
    assert cache.get("mutable_dict") == {"a": 1, "b": 2, "c": 3}

    # Test replacing with a different type
    cache.set("type_change", "string_value")
    assert cache.get("type_change") == "string_value"

    cache.set("type_change", 42)
    assert cache.get("type_change") == 42

    cache.set("type_change", [1, 2, 3])
    assert cache.get("type_change") == [1, 2, 3]
