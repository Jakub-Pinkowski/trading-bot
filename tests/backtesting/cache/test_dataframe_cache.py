import os
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.backtesting.cache.cache_base import Cache
from app.backtesting.cache.dataframe_cache import dataframe_cache, get_cached_dataframe


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
    assert dataframe_cache.max_size == 50
    assert dataframe_cache.max_age == 604800  # 7 days in seconds


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
def test_get_cached_dataframe_cache_miss(mock_save_cache, mock_read_parquet, sample_dataframe):
    """Test get_cached_dataframe when the dataframe is not in the cache."""
    # Clear the cache to start with a clean state
    dataframe_cache.clear()

    # Set up the mock to return our sample dataframe
    mock_read_parquet.return_value = sample_dataframe

    # Call the function with a filepath that's not in the cache
    filepath = "test_file.parquet"
    result_df = get_cached_dataframe(filepath)

    # Verify that read_parquet was called with the correct filepath
    mock_read_parquet.assert_called_once_with(filepath)

    # Verify that the result is a copy of the sample dataframe
    pd.testing.assert_frame_equal(result_df, sample_dataframe)
    assert id(result_df) != id(sample_dataframe)  # Should be a different object (copy)

    # Verify that the dataframe was added to the cache
    assert dataframe_cache.contains(filepath)
    cached_df = dataframe_cache.get(filepath)
    pd.testing.assert_frame_equal(cached_df, sample_dataframe)

    # Verify that save_cache was NOT called - this is now handled periodically by MassTester
    mock_save_cache.assert_not_called()


def test_get_cached_dataframe_cache_hit(sample_dataframe):
    """Test get_cached_dataframe when the dataframe is already in the cache."""
    # Clear the cache to start with a clean state
    dataframe_cache.clear()

    # Add the dataframe to the cache
    filepath = "test_file.parquet"
    dataframe_cache.set(filepath, sample_dataframe)

    # Call the function with the filepath that's in the cache
    with patch('pandas.read_parquet') as mock_read_parquet:
        result_df = get_cached_dataframe(filepath)

        # Verify that read_parquet was NOT called
        mock_read_parquet.assert_not_called()

    # Verify that the result is a copy of the sample dataframe
    pd.testing.assert_frame_equal(result_df, sample_dataframe)
    assert id(result_df) != id(sample_dataframe)  # Should be a different object (copy)


@patch('pandas.read_parquet')
def test_get_cached_dataframe_error_handling(mock_read_parquet):
    """Test error handling in get_cached_dataframe."""
    # Clear the cache to start with a clean state
    dataframe_cache.clear()

    # Set up the mock to raise an exception
    mock_read_parquet.side_effect = Exception("Test exception")

    # Call the function and verify that the exception is propagated
    filepath = "test_file.parquet"
    with pytest.raises(Exception, match="Test exception"):
        get_cached_dataframe(filepath)

    # Verify that the dataframe was NOT added to the cache
    assert not dataframe_cache.contains(filepath)


def test_get_cached_dataframe_with_real_file(tmp_path, sample_dataframe):
    """Test get_cached_dataframe with a real parquet file."""
    # Create a temporary parquet file
    filepath = os.path.join(tmp_path, "test_file.parquet")
    sample_dataframe.to_parquet(filepath)

    # Clear the cache to start with a clean state
    dataframe_cache.clear()

    # Call the function with the filepath
    with patch('app.backtesting.cache.dataframe_cache.dataframe_cache.save_cache') as mock_save_cache:
        result_df = get_cached_dataframe(filepath)

        # Verify that save_cache was NOT called - this is now handled periodically by MassTester
        mock_save_cache.assert_not_called()

    # Verify that the result matches the sample dataframe
    pd.testing.assert_frame_equal(result_df, sample_dataframe)

    # Verify that the dataframe was added to the cache
    assert dataframe_cache.contains(filepath)

    # Call the function again and verify that it uses the cached value
    with patch('pandas.read_parquet') as mock_read_parquet:
        result_df2 = get_cached_dataframe(filepath)

        # Verify that read_parquet was NOT called
        mock_read_parquet.assert_not_called()

    # Verify that the result matches the sample dataframe
    pd.testing.assert_frame_equal(result_df2, sample_dataframe)


def test_dataframe_cache_with_different_sizes_and_structures(tmp_path):
    """Test the dataframe cache with dataframes of different sizes and structures."""
    # Clear the cache to start with a clean state
    dataframe_cache.clear()

    # Test with a small dataframe
    small_df = pd.DataFrame({
        'A': range(10),
        'B': range(10, 20)
    })
    small_filepath = os.path.join(tmp_path, "small_df.parquet")
    small_df.to_parquet(small_filepath)

    # Test with a medium dataframe
    medium_df = pd.DataFrame({
        'A': range(1000),
        'B': range(1000, 2000),
        'C': range(2000, 3000)
    })
    medium_filepath = os.path.join(tmp_path, "medium_df.parquet")
    medium_df.to_parquet(medium_filepath)

    # Test with a large dataframe
    large_df = pd.DataFrame({
        'A': range(10000),
        'B': range(10000, 20000),
        'C': range(20000, 30000),
        'D': range(30000, 40000),
        'E': range(40000, 50000)
    })
    large_filepath = os.path.join(tmp_path, "large_df.parquet")
    large_df.to_parquet(large_filepath)

    # Test with a dataframe with a different structure
    different_structure_df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=100),
        'symbol': ['ZC'] * 100,
        'open': np.random.rand(100) * 100,
        'high': np.random.rand(100) * 100,
        'low': np.random.rand(100) * 100,
        'close': np.random.rand(100) * 100,
        'volume': np.random.randint(1000, 10000, 100)
    })
    different_structure_filepath = os.path.join(tmp_path, "different_structure_df.parquet")
    different_structure_df.to_parquet(different_structure_filepath)

    # Load all dataframes into cache
    with patch('app.backtesting.cache.dataframe_cache.dataframe_cache.save_cache'):
        small_result = get_cached_dataframe(small_filepath)
        medium_result = get_cached_dataframe(medium_filepath)
        large_result = get_cached_dataframe(large_filepath)
        different_structure_result = get_cached_dataframe(different_structure_filepath)

    # Verify all dataframes were loaded correctly
    pd.testing.assert_frame_equal(small_result, small_df)
    pd.testing.assert_frame_equal(medium_result, medium_df)
    pd.testing.assert_frame_equal(large_result, large_df)
    pd.testing.assert_frame_equal(different_structure_result, different_structure_df)

    # Verify all dataframes are in the cache
    assert dataframe_cache.size() == 4
    assert dataframe_cache.contains(small_filepath)
    assert dataframe_cache.contains(medium_filepath)
    assert dataframe_cache.contains(large_filepath)
    assert dataframe_cache.contains(different_structure_filepath)

    # Test retrieval performance for different sizes
    start_time = time.time()
    small_cached = get_cached_dataframe(small_filepath)
    small_time = time.time() - start_time

    start_time = time.time()
    medium_cached = get_cached_dataframe(medium_filepath)
    medium_time = time.time() - start_time

    start_time = time.time()
    large_cached = get_cached_dataframe(large_filepath)
    large_time = time.time() - start_time

    # Log performance metrics (these are not strict assertions, just informational)
    # Performance metrics are measured but not printed to keep test output clean


def test_dataframe_cache_with_different_data_types(tmp_path):
    """Test the dataframe cache with dataframes containing different data types."""
    # Clear the cache to start with a clean state
    dataframe_cache.clear()

    # Create a dataframe with various data types
    df_with_types = pd.DataFrame({
        # Numeric types
        'int': [1, 2, 3, 4, 5],
        'float': [1.1, 2.2, 3.3, 4.4, 5.5],
        'complex': [complex(1, 2), complex(2, 3), complex(3, 4), complex(4, 5), complex(5, 6)],

        # Boolean type
        'bool': [True, False, True, False, True],

        # String type
        'string': ['a', 'b', 'c', 'd', 'e'],

        # Date and time types
        'date': pd.date_range(start='2023-01-01', periods=5),
        'datetime': pd.date_range(start='2023-01-01 12:00:00', periods=5),
        'timedelta': [pd.Timedelta(days=i) for i in range(5)],

        # Categorical type
        'category': pd.Categorical(['A', 'B', 'C', 'A', 'B']),

        # Object type
        'object': [{'a': 1}, {'b': 2}, {'c': 3}, {'d': 4}, {'e': 5}]
    })

    # Save to parquet (note: some types like complex and object might not be preserved in parquet)
    # We'll exclude those columns for the parquet file
    df_for_parquet = df_with_types.drop(columns=['complex', 'object'])
    filepath = os.path.join(tmp_path, "different_types.parquet")
    df_for_parquet.to_parquet(filepath)

    # Load the dataframe into cache
    with patch('app.backtesting.cache.dataframe_cache.dataframe_cache.save_cache'):
        result_df = get_cached_dataframe(filepath)

    # Verify the dataframe was loaded correctly
    pd.testing.assert_frame_equal(result_df, df_for_parquet)

    # Verify the dataframe is in the cache
    assert dataframe_cache.contains(filepath)
    cached_df = dataframe_cache.get(filepath)
    pd.testing.assert_frame_equal(cached_df, df_for_parquet)

    # Test with a dataframe containing NaN, None, and Inf values
    df_with_special_values = pd.DataFrame({
        'with_nan': [1.0, np.nan, 3.0, np.nan, 5.0],
        'with_none': [1, None, 3, None, 5],
        'with_inf': [1.0, np.inf, 3.0, -np.inf, 5.0],
    })

    special_filepath = os.path.join(tmp_path, "special_values.parquet")
    df_with_special_values.to_parquet(special_filepath)

    # Load the dataframe into cache
    with patch('app.backtesting.cache.dataframe_cache.dataframe_cache.save_cache'):
        special_result_df = get_cached_dataframe(special_filepath)

    # Verify the dataframe was loaded correctly
    pd.testing.assert_frame_equal(special_result_df, df_with_special_values)

    # Verify the dataframe is in the cache
    assert dataframe_cache.contains(special_filepath)
    special_cached_df = dataframe_cache.get(special_filepath)
    pd.testing.assert_frame_equal(special_cached_df, df_with_special_values)


def test_multiple_dataframes_in_cache(tmp_path):
    """Test handling multiple dataframes in the cache simultaneously."""
    # Clear the cache to start with a clean state
    dataframe_cache.clear()

    # Create multiple dataframes for different futures contracts
    contracts = ['ZC', 'ZS', 'CL', 'GC', 'SI']
    dataframes = {}
    filepaths = {}

    # Create and save dataframes for each contract
    for contract in contracts:
        # Create a dataframe with OHLCV data
        df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=100),
            'symbol': [contract] * 100,
            'open': np.random.rand(100) * 100,
            'high': np.random.rand(100) * 100 + 10,
            'low': np.random.rand(100) * 100 - 10,
            'close': np.random.rand(100) * 100,
            'volume': np.random.randint(1000, 10000, 100)
        })

        # Save to parquet
        filepath = os.path.join(tmp_path, f"{contract}.parquet")
        df.to_parquet(filepath)

        # Store for later comparison
        dataframes[contract] = df
        filepaths[contract] = filepath

    # Load all dataframes into cache
    with patch('app.backtesting.cache.dataframe_cache.dataframe_cache.save_cache'):
        for contract in contracts:
            result_df = get_cached_dataframe(filepaths[contract])
            # Verify the dataframe was loaded correctly
            pd.testing.assert_frame_equal(result_df, dataframes[contract])

    # Verify all dataframes are in the cache
    assert dataframe_cache.size() == len(contracts)
    for contract in contracts:
        assert dataframe_cache.contains(filepaths[contract])

    # Retrieve all dataframes from the cache and verify they match the originals
    for contract in contracts:
        cached_df = dataframe_cache.get(filepaths[contract])
        pd.testing.assert_frame_equal(cached_df, dataframes[contract])

    # Modify one dataframe and verify others remain unchanged
    modified_contract = 'ZC'
    modified_df = dataframes[modified_contract].copy()
    modified_df['close'] = modified_df['close'] * 1.1  # Increase close prices by 10%

    # Update the cache with the modified dataframe
    dataframe_cache.set(filepaths[modified_contract], modified_df)

    # Verify the modified dataframe is updated in the cache
    cached_modified_df = dataframe_cache.get(filepaths[modified_contract])
    pd.testing.assert_frame_equal(cached_modified_df, modified_df)

    # Verify other dataframes remain unchanged
    for contract in contracts:
        if contract != modified_contract:
            cached_df = dataframe_cache.get(filepaths[contract])
            pd.testing.assert_frame_equal(cached_df, dataframes[contract])

    # Clear the cache and verify it's empty
    dataframe_cache.clear()
    assert dataframe_cache.size() == 0
    for contract in contracts:
        assert not dataframe_cache.contains(filepaths[contract])


def test_dataframe_cache_with_non_existent_files():
    """Test handling of non-existent files."""
    # Clear the cache to start with a clean state
    dataframe_cache.clear()

    # Test with a non-existent file
    non_existent_filepath = "/path/to/non_existent_file.parquet"

    # Attempt to load the non-existent file
    with patch('pandas.read_parquet') as mock_read_parquet:
        # Set up the mock to raise a FileNotFoundError
        mock_read_parquet.side_effect = FileNotFoundError("File not found")

        # Verify that the exception is propagated
        with pytest.raises(FileNotFoundError, match="File not found"):
            get_cached_dataframe(non_existent_filepath)

    # Verify that the non-existent file was not added to the cache
    assert not dataframe_cache.contains(non_existent_filepath)
    assert dataframe_cache.size() == 0

    # Test with a file that exists but is not a valid parquet file
    with patch('pandas.read_parquet') as mock_read_parquet:
        # Set up the mock to raise a different exception
        mock_read_parquet.side_effect = Exception("Invalid parquet file")

        # Verify that the exception is propagated
        with pytest.raises(Exception, match="Invalid parquet file"):
            get_cached_dataframe("invalid_parquet_file.parquet")

    # Verify that the invalid file was not added to the cache
    assert not dataframe_cache.contains("invalid_parquet_file.parquet")
    assert dataframe_cache.size() == 0


def test_dataframe_cache_corrupted_file():
    """Test handling of corrupted cache files."""
    # Clear the cache to start with a clean state
    dataframe_cache.clear()

    # Add some data to the cache
    test_key = "test_file.parquet"
    test_value = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    dataframe_cache.set(test_key, test_value)

    # Save the cache
    dataframe_cache.save_cache()

    # Simulate a corrupted cache file by mocking pickle.load to raise an exception
    with patch('pickle.load', side_effect=Exception("Corrupted file")):
        # Create a new cache instance, which will try to load the corrupted file
        new_cache = Cache("dataframe")

        # Verify the new cache is empty (couldn't load the corrupted data)
        assert not new_cache.contains(test_key)
        assert new_cache.size() == 0


def test_dataframe_cache_interrupted_save():
    """Test handling of interrupted save operations."""
    # Clear the cache to start with a clean state
    dataframe_cache.clear()

    # Add some data to the cache
    test_key = "test_file.parquet"
    test_value = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    dataframe_cache.set(test_key, test_value)

    # Simulate an interrupted save by mocking open to raise an exception
    with patch('builtins.open', side_effect=Exception("Interrupted save")):
        # Try to save the cache, which should handle the exception gracefully
        dataframe_cache.save_cache()

    # Verify the cache still contains the data in memory
    assert dataframe_cache.contains(test_key)
    pd.testing.assert_frame_equal(dataframe_cache.get(test_key), test_value)


def test_dataframe_cache_large_objects():
    """Test handling of large objects in the cache to check for memory leaks."""
    # Clear the cache to start with a clean state
    dataframe_cache.clear()

    # Create a large dataframe (100,000 rows x 10 columns)
    large_df = pd.DataFrame({
        f'col_{i}': np.random.rand(100_000) for i in range(10)
    })

    # Add the large dataframe to the cache
    test_key = "large_file.parquet"
    dataframe_cache.set(test_key, large_df)

    # Verify the dataframe is in the cache
    assert dataframe_cache.contains(test_key)

    # Get the dataframe from the cache and verify it matches the original
    cached_df = dataframe_cache.get(test_key)
    pd.testing.assert_frame_equal(cached_df, large_df)

    # Clear the reference to the original dataframe to allow garbage collection
    del large_df

    # Clear the cache to free memory
    dataframe_cache.clear()
    assert not dataframe_cache.contains(test_key)
    assert dataframe_cache.size() == 0


def test_dataframe_cache_concurrent_access():
    """Test handling of concurrent access to the cache."""
    import threading

    # Clear the cache to start with a clean state
    dataframe_cache.clear()

    # Create a function that adds dataframes to the cache
    def add_to_cache(start_idx, end_idx):
        for i in range(start_idx, end_idx):
            df = pd.DataFrame({
                'A': range(i, i + 10),
                'B': range(i + 10, i + 20)
            })
            dataframe_cache.set(f'file_{i}.parquet', df)

    # Create threads to add dataframes to the cache concurrently
    threads = []
    for i in range(5):
        t = threading.Thread(target=add_to_cache, args=(i * 10, (i + 1) * 10))
        threads.append(t)

    # Start all threads
    for t in threads:
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    # Verify that all dataframes were added to the cache
    assert dataframe_cache.size() == 50  # 5 threads * 10 dataframes each

    # Verify that we can retrieve some of the dataframes
    for i in range(0, 50, 10):
        key = f'file_{i}.parquet'
        assert dataframe_cache.contains(key)
        df = dataframe_cache.get(key)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (10, 2)
        assert list(df.columns) == ['A', 'B']
        assert df['A'].iloc[0] == i
        assert df['B'].iloc[0] == i + 10

    # Clear the cache
    dataframe_cache.clear()
    assert dataframe_cache.size() == 0


def test_dataframe_cache_invalid_keys():
    """Test handling of invalid keys in the cache."""
    # Clear the cache to start with a clean state
    dataframe_cache.clear()

    # Try to set a non-hashable key (dictionary is not hashable)
    with pytest.raises(TypeError):
        dataframe_cache.set({'non_hashable': 'key'}, pd.DataFrame())

    # Try to set a None key
    test_df = pd.DataFrame({'A': [1, 2, 3]})
    dataframe_cache.set(None, test_df)
    assert dataframe_cache.contains(None)
    pd.testing.assert_frame_equal(dataframe_cache.get(None), test_df)

    # Try to set an empty string key
    dataframe_cache.set('', test_df)
    assert dataframe_cache.contains('')
    pd.testing.assert_frame_equal(dataframe_cache.get(''), test_df)

    # Verify the cache size
    assert dataframe_cache.size() == 2  # None and empty string keys

    # Clear the cache
    dataframe_cache.clear()
    assert dataframe_cache.size() == 0
