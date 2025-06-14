import multiprocessing
import os
import pickle
import random
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from app.backtesting.cache.cache_base import Cache
from config import CACHE_DIR


def worker_process(worker_id, cache_name, iterations=10):
    """Worker process that reads and writes to the cache."""
    # Create a test cache instance
    test_cache = Cache(cache_name, cache_version=1, max_size=100, max_age=3600)

    for i in range(iterations):
        # Add some randomness to increase the chance of collision
        time.sleep(random.uniform(0.01, 0.1))

        # Read from cache
        value = test_cache.get(f"key_{i % 5}")

        # Write to cache
        test_cache.set(f"key_{i % 5}", f"value_from_worker_{worker_id}_iteration_{i}")

        # Save cache
        test_cache.save_cache()

    return worker_id


def check_cache_integrity(cache_name):
    """Check if the cache file is valid by trying to load it."""
    cache_file = os.path.join(CACHE_DIR, f"{cache_name}_cache_v1.pkl")
    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
            if isinstance(cache_data, dict):
                return True, len(cache_data)
            else:
                return False, 0
    except Exception:
        return False, 0


@pytest.fixture
def clean_test_cache():
    """Fixture to ensure a clean test environment."""
    # Ensure the cache directory exists
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

    # Generate a unique cache name for this test run
    cache_name = f"concurrency_test_{int(time.time())}"

    # Remove any existing test cache file
    cache_file = os.path.join(CACHE_DIR, f"{cache_name}_cache_v1.pkl")
    if os.path.exists(cache_file):
        os.remove(cache_file)

    yield cache_name

    # Cleanup after a test
    if os.path.exists(cache_file):
        os.remove(cache_file)

    lock_file = os.path.join(CACHE_DIR, f"{cache_name}_cache_v1.lock")
    if os.path.exists(lock_file):
        os.remove(lock_file)


def test_cache_concurrency(clean_test_cache):
    """Test that the cache handles concurrent access correctly."""
    cache_name = clean_test_cache
    num_workers = 4
    iterations = 10

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Start the worker processes
        results = pool.starmap(worker_process, [(i, cache_name, iterations) for i in range(num_workers)])

    # Verify all workers completed successfully
    assert len(results) == num_workers
    for i in range(num_workers):
        assert i in results

    # Check cache integrity
    is_valid, num_items = check_cache_integrity(cache_name)
    assert is_valid, "Cache file is corrupted or invalid"
    assert num_items > 0, "Cache is empty"
    assert num_items <= 5, f"Cache should have at most 5 items, but has {num_items}"

    # Load the cache and verify its contents
    test_cache = Cache(cache_name, cache_version=1)
    assert test_cache.size() > 0, "Cache is empty"
    assert test_cache.size() <= 5, f"Cache should have at most 5 items, but has {test_cache.size()}"

    # Verify that the cache contains valid data
    for i in range(5):
        key = f"key_{i}"
        if test_cache.contains(key):
            value = test_cache.get(key)
            assert isinstance(value, str), f"Cache value for {key} is not a string: {value}"
            assert value.startswith("value_from_worker_"), f"Unexpected value format: {value}"


def test_cache_lock_release_on_exception():
    """Test that the lock is released when an exception occurs."""
    # Generate a unique cache name for this test
    cache_name = f"lock_release_test_{int(time.time())}"

    # Create a cache instance
    test_cache = Cache(cache_name, cache_version=1, max_size=100, max_age=3600)

    # Add some data to the cache
    test_cache.set("key1", "value1")

    # Mock the open function to raise an exception during save
    with patch('builtins.open') as mock_open:
        mock_open.side_effect = Exception("Test exception")

        # Mock the FileLock to verify it's used and released
        with patch('app.backtesting.cache.cache_base.FileLock', autospec=True) as mock_filelock:
            # Set up the mock to return a context manager
            mock_lock_instance = MagicMock()
            mock_filelock.return_value = mock_lock_instance
            mock_lock_instance.__enter__.return_value = mock_lock_instance
            mock_lock_instance.__exit__.return_value = None

            # Try to save the cache, which should handle the exception
            test_cache.save_cache()

            # Verify FileLock was called
            mock_filelock.assert_called_once()
            # Verify the lock's __enter__ method was called (context manager was used)
            mock_lock_instance.__enter__.assert_called_once()
            # Verify the lock's __exit__ method was called (lock was released)
            mock_lock_instance.__exit__.assert_called_once()

    # Clean up
    cache_file = os.path.join(CACHE_DIR, f"{cache_name}_cache_v1.pkl")
    if os.path.exists(cache_file):
        os.remove(cache_file)

    lock_file = os.path.join(CACHE_DIR, f"{cache_name}_cache_v1.lock")
    if os.path.exists(lock_file):
        os.remove(lock_file)
