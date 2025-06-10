import os
import pickle
from unittest.mock import patch, mock_open

import pytest

from app.backtesting.cache.cache_base import Cache
from config import CACHE_DIR


@pytest.fixture
def temp_cache_file(tmp_path):
    """Create a temporary cache file for testing."""
    cache_file = tmp_path / "test_cache_v1.pkl"
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
    cache = Cache("test", 1)
    assert cache.cache_name == "test"
    assert cache.cache_version == 1
    assert cache.cache_file == os.path.join(CACHE_DIR, "test_cache_v1.pkl")
    assert cache.cache_data == {}


def test_cache_set_and_get():
    """Test setting and getting values from the cache."""
    cache = Cache("test", 1)

    # Test setting and getting a value
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"

    # Test getting a non-existent key
    assert cache.get("non_existent_key") is None

    # Test getting a non-existent key with a default value
    assert cache.get("non_existent_key", "default") == "default"


def test_cache_contains():
    """Test checking if a key exists in the cache."""
    cache = Cache("test", 1)

    # Key doesn't exist initially
    assert not cache.contains("key1")

    # Add the key
    cache.set("key1", "value1")
    assert cache.contains("key1")


def test_cache_clear():
    """Test clearing the cache."""
    cache = Cache("test", 1)

    # Add some items to the cache
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    assert cache.size() == 2

    # Clear the cache
    cache.clear()
    assert cache.size() == 0
    assert not cache.contains("key1")
    assert not cache.contains("key2")


def test_cache_size():
    """Test getting the size of the cache."""
    cache = Cache("test", 1)

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
    cache_file = os.path.join(mock_cache_dir, "test_cache_v1.pkl")

    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

    # Initialize cache, which should load the file
    cache = Cache("test", 1)

    # Verify the data was loaded
    assert cache.cache_data == cache_data
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"


def test_load_cache_file_does_not_exist():
    """Test loading a cache file that doesn't exist."""
    # Use a unique name to ensure the file doesn't exist
    cache = Cache("nonexistent_test", 999)

    # Verify the cache is empty
    assert cache.cache_data == {}
    assert cache.size() == 0


def test_load_cache_invalid_data(mock_cache_dir):
    """Test loading a cache file with invalid data."""
    # Create a cache file with invalid data (not a dictionary)
    cache_file = os.path.join(mock_cache_dir, "test_cache_v1.pkl")

    with open(cache_file, 'wb') as f:
        pickle.dump("not a dictionary", f)

    # Initialize cache, which should handle the invalid data
    cache = Cache("test", 1)

    # Verify the cache is empty
    assert cache.cache_data == {}
    assert cache.size() == 0


def test_load_cache_exception_handling():
    """Test handling exceptions when loading the cache."""
    with patch("builtins.open", mock_open()) as mock_file:
        mock_file.side_effect = Exception("Test exception")

        # Initialize cache, which should handle the exception
        cache = Cache("test", 1)

        # Verify the cache is empty
        assert cache.cache_data == {}
        assert cache.size() == 0


def test_save_cache(mock_cache_dir):
    """Test saving the cache to disk."""
    cache = Cache("test", 1)

    # Add some data to the cache
    cache.set("key1", "value1")
    cache.set("key2", "value2")

    # Save the cache
    cache.save_cache()

    # Verify the file was created
    cache_file = os.path.join(mock_cache_dir, "test_cache_v1.pkl")
    assert os.path.exists(cache_file)

    # Load the file and verify the data
    with open(cache_file, 'rb') as f:
        loaded_data = pickle.load(f)

    assert loaded_data == {"key1": "value1", "key2": "value2"}


def test_save_cache_exception_handling():
    """Test handling exceptions when saving the cache."""
    cache = Cache("test", 1)
    cache.set("key1", "value1")

    with patch("builtins.open", mock_open()) as mock_file:
        mock_file.side_effect = Exception("Test exception")

        # This should not raise an exception
        cache.save_cache()

        # The cache data should still be intact
        assert cache.get("key1") == "value1"


def test_cache_with_complex_data():
    """Test the cache with complex data types."""
    cache = Cache("test", 1)

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
