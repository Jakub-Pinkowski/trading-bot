import os
from unittest.mock import patch

import pandas as pd
import pytest

from app.backtesting.cache.dataframe_cache import dataframe_cache, get_preprocessed_dataframe, CACHE_VERSION


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing."""
    return pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=5),
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [95, 96, 97, 98, 99],
        'close': [102, 103, 104, 105, 106],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })


def test_dataframe_cache_instance():
    """Test that the dataframe_cache is properly initialized."""
    assert dataframe_cache.cache_name == "dataframe"
    assert dataframe_cache.cache_version == CACHE_VERSION


def test_dataframe_cache_operations():
    """Test basic operations on the dataframe cache."""
    # Clear the cache to start with a clean state
    dataframe_cache.clear()

    # Test that the cache is empty
    assert dataframe_cache.size() == 0

    # Test setting and getting a value
    test_key = "test_file.parquet"
    test_value = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

    dataframe_cache.set(test_key, test_value)
    assert dataframe_cache.contains(test_key)

    # Get the value and verify it matches the original
    cached_df = dataframe_cache.get(test_key)
    pd.testing.assert_frame_equal(cached_df, test_value)
    # Note: The base Cache.get() method doesn't make a copy, unlike get_preprocessed_dataframe

    # Test cache size
    assert dataframe_cache.size() == 1

    # Test getting a non-existent key
    non_existent_key = "non_existent.parquet"
    assert dataframe_cache.get(non_existent_key) is None
    assert dataframe_cache.get(non_existent_key, "default") == "default"

    # Clear the cache again
    dataframe_cache.clear()
    assert dataframe_cache.size() == 0
    assert not dataframe_cache.contains(test_key)


@patch('app.backtesting.cache.cache_base.Cache.save_cache')
def test_dataframe_cache_save(mock_save_cache):
    """Test saving the dataframe cache."""
    # Clear the cache to start with a clean state
    dataframe_cache.clear()

    # Add some data to the cache
    test_key = "test_file.parquet"
    test_value = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

    dataframe_cache.set(test_key, test_value)

    # Save the cache
    dataframe_cache.save_cache()

    # Verify that save_cache was called
    mock_save_cache.assert_called_once()


@patch('pandas.read_parquet')
@patch('app.backtesting.cache.dataframe_cache.dataframe_cache.save_cache')
def test_get_preprocessed_dataframe_cache_miss(mock_save_cache, mock_read_parquet, sample_dataframe):
    """Test get_preprocessed_dataframe when the dataframe is not in the cache."""
    # Clear the cache to start with a clean state
    dataframe_cache.clear()

    # Set up the mock to return our sample dataframe
    mock_read_parquet.return_value = sample_dataframe

    # Call the function with a filepath that's not in the cache
    filepath = "test_file.parquet"
    result_df = get_preprocessed_dataframe(filepath)

    # Verify that read_parquet was called with the correct filepath
    mock_read_parquet.assert_called_once_with(filepath)

    # Verify that the result is a copy of the sample dataframe
    pd.testing.assert_frame_equal(result_df, sample_dataframe)
    assert id(result_df) != id(sample_dataframe)  # Should be a different object (copy)

    # Verify that the dataframe was added to the cache
    assert dataframe_cache.contains(filepath)
    cached_df = dataframe_cache.get(filepath)
    pd.testing.assert_frame_equal(cached_df, sample_dataframe)

    # Verify that save_cache was called
    mock_save_cache.assert_called_once()


def test_get_preprocessed_dataframe_cache_hit(sample_dataframe):
    """Test get_preprocessed_dataframe when the dataframe is already in the cache."""
    # Clear the cache to start with a clean state
    dataframe_cache.clear()

    # Add the dataframe to the cache
    filepath = "test_file.parquet"
    dataframe_cache.set(filepath, sample_dataframe)

    # Call the function with the filepath that's in the cache
    with patch('pandas.read_parquet') as mock_read_parquet:
        result_df = get_preprocessed_dataframe(filepath)

        # Verify that read_parquet was NOT called
        mock_read_parquet.assert_not_called()

    # Verify that the result is a copy of the sample dataframe
    pd.testing.assert_frame_equal(result_df, sample_dataframe)
    assert id(result_df) != id(sample_dataframe)  # Should be a different object (copy)


@patch('pandas.read_parquet')
def test_get_preprocessed_dataframe_error_handling(mock_read_parquet):
    """Test error handling in get_preprocessed_dataframe."""
    # Clear the cache to start with a clean state
    dataframe_cache.clear()

    # Set up the mock to raise an exception
    mock_read_parquet.side_effect = Exception("Test exception")

    # Call the function and verify that the exception is propagated
    filepath = "test_file.parquet"
    with pytest.raises(Exception, match="Test exception"):
        get_preprocessed_dataframe(filepath)

    # Verify that the dataframe was NOT added to the cache
    assert not dataframe_cache.contains(filepath)


def test_get_preprocessed_dataframe_with_real_file(tmp_path, sample_dataframe):
    """Test get_preprocessed_dataframe with a real parquet file."""
    # Create a temporary parquet file
    filepath = os.path.join(tmp_path, "test_file.parquet")
    sample_dataframe.to_parquet(filepath)

    # Clear the cache to start with a clean state
    dataframe_cache.clear()

    # Call the function with the filepath
    with patch('app.backtesting.cache.dataframe_cache.dataframe_cache.save_cache') as mock_save_cache:
        result_df = get_preprocessed_dataframe(filepath)

        # Verify that save_cache was called
        mock_save_cache.assert_called_once()

    # Verify that the result matches the sample dataframe
    pd.testing.assert_frame_equal(result_df, sample_dataframe)

    # Verify that the dataframe was added to the cache
    assert dataframe_cache.contains(filepath)

    # Call the function again and verify that it uses the cached value
    with patch('pandas.read_parquet') as mock_read_parquet:
        result_df2 = get_preprocessed_dataframe(filepath)

        # Verify that read_parquet was NOT called
        mock_read_parquet.assert_not_called()

    # Verify that the result matches the sample dataframe
    pd.testing.assert_frame_equal(result_df2, sample_dataframe)
