import json
import os
from unittest.mock import patch, mock_open, MagicMock

import pandas as pd
import pytest

from app.utils.file_utils import (
    load_file, save_file, json_to_dataframe, save_to_csv, load_data_from_json_files,
    save_to_parquet
)


@pytest.fixture
def sample_json_data():
    return {
        "item1": {"name": "Item 1", "value": 100},
        "item2": {"name": "Item 2", "value": 200}
    }


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "name": ["Item 1", "Item 2"],
        "value": [100, 200]
    })


def test_load_file_existing_file(sample_json_data):
    """Test that load_file correctly loads and returns JSON data from an existing file"""

    # Mock file existence and open operation, prepare JSON data for loading
    with patch("os.path.exists", return_value=True), \
            patch("builtins.open", mock_open(read_data=json.dumps(sample_json_data))), \
            patch("json.load", return_value=sample_json_data):
        # Call load_file function with a test filename
        result = load_file("test_file.json")
        # Verify the function returns the expected JSON data
        assert result == sample_json_data


def test_load_file_nonexistent_file():
    """Test that load_file returns an empty dictionary when the file doesn't exist"""

    # Mock file existence to return False (file doesn't exist)
    with patch("os.path.exists", return_value=False):
        # Call load_file function with a nonexistent filename
        result = load_file("nonexistent_file.json")
        # Verify the function returns an empty dictionary when file doesn't exist
        assert result == {}


def test_save_file(sample_json_data):
    """Test that save_file creates directories and writes JSON data to the specified file"""

    # Create mock objects for file operations and directory creation
    m = mock_open()
    with patch("os.makedirs") as mock_makedirs, \
            patch("builtins.open", m), \
            patch("json.dump") as mock_json_dump:
        # Call save_file function with sample data and a test filepath
        save_file(sample_json_data, "test_dir/test_file.json")
        # Verify directory was created, file was opened correctly, and json.dump was called
        mock_makedirs.assert_called_once_with("test_dir", exist_ok=True)
        m.assert_called_once_with("test_dir/test_file.json", "w")
        mock_json_dump.assert_called_once()


def test_json_to_dataframe_dict(sample_json_data):
    """Test that json_to_dataframe correctly converts a dictionary to a DataFrame"""

    # No additional setup needed, using the sample_json_data fixture
    # Convert the dictionary data to a DataFrame using json_to_dataframe
    result = json_to_dataframe(sample_json_data)
    # Verify the result is a DataFrame with expected columns from the dictionary keys
    assert isinstance(result, pd.DataFrame)
    assert "item1" in result.columns
    assert "item2" in result.columns


def test_json_to_dataframe_list():
    """Test that json_to_dataframe correctly converts a list of dictionaries to a DataFrame"""

    # Create a list of dictionaries to test list-type JSON data conversion
    data = [{"name": "Item 1", "value": 100}, {"name": "Item 2", "value": 200}]
    # Convert the list data to a DataFrame using json_to_dataframe
    result = json_to_dataframe(data)
    # Verify the result is a DataFrame with expected columns and values from the list data
    assert isinstance(result, pd.DataFrame)
    assert "name" in result.columns
    assert "value" in result.columns
    assert result["name"].tolist() == ["Item 1", "Item 2"]
    assert result["value"].tolist() == [100, 200]


def test_json_to_dataframe_empty():
    """Test that json_to_dataframe returns an empty DataFrame when given an empty dictionary"""

    # Convert an empty dictionary to a DataFrame using json_to_dataframe
    result = json_to_dataframe({})
    # Verify the result is an empty DataFrame
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_json_to_dataframe_with_date_fields():
    """Test that json_to_dataframe correctly converts date string fields to datetime objects"""

    # Create test data with date strings that should be converted to datetime objects
    data = [
        {"name": "Item 1", "date": "2023-01-01"},
        {"name": "Item 2", "date": "2023-01-02"}
    ]
    # Convert to DataFrame with date_fields parameter to specify which fields should be parsed as dates
    result = json_to_dataframe(data, date_fields=["date"])
    # Verify the result is a DataFrame and the date column has been converted to datetime type
    assert isinstance(result, pd.DataFrame)
    assert pd.api.types.is_datetime64_dtype(result["date"])


def test_json_to_dataframe_unsupported_format():
    """Test that json_to_dataframe raises ValueError when given an unsupported data format"""

    # Test with a string (unsupported format)
    with pytest.raises(ValueError, match="Unsupported data format. Provide either dictionary or list."):
        json_to_dataframe("unsupported_format")


def test_json_to_dataframe_dict_with_index_orient():
    """Test that json_to_dataframe correctly handles orient='index' parameter"""

    # Create test data
    data = {
        "item1": {"name": "Item 1", "value": 100},
        "item2": {"name": "Item 2", "value": 200}
    }

    # Convert to DataFrame with orient='index'
    result = json_to_dataframe(data, orient='index', index_name='custom_index')

    # Verify the result has the expected structure
    assert isinstance(result, pd.DataFrame)
    assert 'custom_index' in result.columns
    assert 'name' in result.columns
    assert 'value' in result.columns
    assert result['custom_index'].tolist() == ['item1', 'item2']
    assert result['name'].tolist() == ['Item 1', 'Item 2']
    assert result['value'].tolist() == [100, 200]


def test_save_to_csv_new_file(sample_dataframe):
    """Test that save_to_csv correctly saves a DataFrame to a new CSV file"""

    # Mock file existence check to return False (file doesn't exist) and patch DataFrame.to_csv
    with patch("os.path.exists", return_value=False), \
            patch.object(pd.DataFrame, "to_csv") as mock_to_csv:
        # Call save_to_csv with sample dataframe and a test filename
        save_to_csv(sample_dataframe, "test_file.csv")
        # Verify to_csv was called once with the correct filename and index=False
        mock_to_csv.assert_called_once_with("test_file.csv", index=False)


def test_save_to_csv_existing_file(sample_dataframe):
    """Test that save_to_csv appends data to an existing CSV file"""

    # Create an existing dataframe and mock file existence, read_csv, to_csv, and concat operations
    existing_df = pd.DataFrame({
        "name": ["Item 3"],
        "value": [300]
    })
    with patch("os.path.exists", return_value=True), \
            patch("pandas.read_csv", return_value=existing_df), \
            patch.object(pd.DataFrame, "to_csv") as mock_to_csv, \
            patch("pandas.concat", return_value=pd.DataFrame()) as mock_concat:
        # Call save_to_csv with sample dataframe and a test filename that "already exists"
        save_to_csv(sample_dataframe, "test_file.csv")
        # Verify concat was called to merge existing and new data, and to_csv was called to save the result
        mock_concat.assert_called_once()
        mock_to_csv.assert_called_once()


def test_save_to_csv_existing_file_read_error(sample_dataframe):
    """Test that save_to_csv handles exceptions when reading an existing CSV file"""

    # Mock file existence to return True, but read_csv to raise an exception
    with patch("os.path.exists", return_value=True), \
            patch("pandas.read_csv", side_effect=Exception("CSV read error")), \
            patch("app.utils.file_utils.logger.error") as mock_logger_error, \
            patch.object(pd.DataFrame, "to_csv") as mock_to_csv:
        # Call save_to_csv with sample dataframe and a test filename that "exists but can't be read"
        save_to_csv(sample_dataframe, "test_file.csv")

        # Verify error was logged and to_csv was still called to save the new data
        mock_logger_error.assert_called_once()
        mock_to_csv.assert_called_once()


def test_save_to_csv_with_dict():
    """Test that save_to_csv correctly converts a dictionary to a DataFrame before saving"""

    # Create a dictionary to test dictionary-to-csv conversion and mock file operations
    data = {"key1": "value1", "key2": "value2"}
    expected_df = pd.DataFrame([("key1", "value1"), ("key2", "value2")], columns=["Key", "Value"])
    with patch("os.path.exists", return_value=False), \
            patch.object(pd.DataFrame, "to_csv") as mock_to_csv:
        # Call save_to_csv with a dictionary instead of a DataFrame
        save_to_csv(data, "test_file.csv")
        # Verify to_csv was called with the correct filename and parameters
        assert mock_to_csv.called, "to_csv was not called"
        args, kwargs = mock_to_csv.call_args
        assert args[0] == "test_file.csv"
        assert kwargs["index"] == False


def test_save_to_csv_invalid_data_type():
    """Test that save_to_csv raises ValueError when given an invalid data type"""

    # No setup needed, testing error handling with invalid data type
    # Call save_to_csv with a string (invalid data type) and verify it raises the expected ValueError
    with pytest.raises(ValueError, match="Data must be either a Pandas DataFrame or a dictionary."):
        save_to_csv("invalid_data", "test_file.csv")


@patch("app.utils.file_utils.glob")
@patch("app.utils.file_utils.load_file")
@patch("app.utils.file_utils.json_to_dataframe")
@patch("pandas.concat")
def test_load_data_from_json_files(mock_concat, mock_json_to_df, mock_load_file, mock_glob):
    """Test that load_data_from_json_files correctly loads and combines data from multiple JSON files"""

    # Mock all dependencies: glob to find files, load_file to read JSON, json_to_dataframe for conversion, and pandas.concat
    mock_glob.return_value = ["file1.json", "file2.json"]
    mock_load_file.side_effect = [{"data1": "value1"}, {"data2": "value2"}]
    mock_df1 = MagicMock()
    mock_df2 = MagicMock()
    mock_json_to_df.side_effect = [mock_df1, mock_df2]
    mock_concat_result = MagicMock()
    mock_concat.return_value = mock_concat_result
    mock_concat_result.sort_index.return_value = mock_concat_result
    mock_concat_result.reset_index.return_value = mock_concat_result

    # Call load_data_from_json_files with test parameters
    result = load_data_from_json_files(
        "test_dir", "prefix", ["date"], "YYYY-MM-DD", "timestamp"
    )

    # Verify all mocked functions were called correctly and the expected result is returned
    mock_glob.assert_called_once_with(os.path.join("test_dir", "prefix_*.json"))
    assert mock_load_file.call_count == 2
    assert mock_json_to_df.call_count == 2
    mock_concat.assert_called_once_with([mock_df1, mock_df2])
    assert result == mock_concat_result


@patch("app.utils.file_utils.glob")
def test_load_data_from_json_files_no_files(mock_glob):
    """Test that load_data_from_json_files returns an empty DataFrame when no files are found"""

    # Mock glob to return an empty list, simulating no matching files found
    mock_glob.return_value = []

    # Call load_data_from_json_files with test parameters when no files exist
    result = load_data_from_json_files(
        "test_dir", "prefix", ["date"], "YYYY-MM-DD", "timestamp"
    )

    # Verify the function returns an empty DataFrame when no files are found
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_save_to_parquet_new_file(sample_dataframe):
    """Test that save_to_parquet correctly saves a DataFrame to a new parquet file"""

    # Mock file lock and file existence check to return False (file doesn't exist) and patch DataFrame.to_parquet
    mock_lock = MagicMock()
    with patch("os.path.exists", return_value=False), \
            patch("os.makedirs") as mock_makedirs, \
            patch("app.utils.file_utils.FileLock", return_value=mock_lock), \
            patch.object(pd.DataFrame, "to_parquet") as mock_to_parquet:
        # Call save_to_parquet with sample dataframe and a test filename
        save_to_parquet(sample_dataframe, "test_dir/test_file.parquet")

        # Verify directory was created and to_parquet was called with the correct parameters
        mock_makedirs.assert_called_once_with("test_dir", exist_ok=True)
        mock_to_parquet.assert_called_once_with("test_dir/test_file.parquet", index=False)
        # Verify FileLock was used
        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()


def test_save_to_parquet_existing_file(sample_dataframe):
    """Test that save_to_parquet appends data to an existing parquet file"""

    # Create an existing dataframe and mock file existence, read_parquet, to_parquet, and concat operations
    existing_df = pd.DataFrame({
        "name": ["Item 3"],
        "value": [300]
    })
    mock_lock = MagicMock()
    with patch("os.path.exists", return_value=True), \
            patch("os.makedirs") as mock_makedirs, \
            patch("app.utils.file_utils.FileLock", return_value=mock_lock), \
            patch("pandas.read_parquet", return_value=existing_df), \
            patch.object(pd.DataFrame, "to_parquet") as mock_to_parquet, \
            patch("pandas.concat", return_value=pd.DataFrame()) as mock_concat:
        # Call save_to_parquet with sample dataframe and a test filename that "already exists"
        save_to_parquet(sample_dataframe, "test_dir/test_file.parquet")

        # Verify directory was created, concat was called to merge existing and new data, 
        # and to_parquet was called to save the result
        mock_makedirs.assert_called_once_with("test_dir", exist_ok=True)
        mock_concat.assert_called_once()
        mock_to_parquet.assert_called_once()
        # Verify FileLock was used
        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()


def test_save_to_parquet_existing_file_read_error(sample_dataframe):
    """Test that save_to_parquet handles exceptions when reading an existing parquet file"""

    # Mock file existence to return True, but read_parquet to raise an exception
    mock_lock = MagicMock()
    with patch("os.path.exists", return_value=True), \
            patch("os.makedirs") as mock_makedirs, \
            patch("app.utils.file_utils.FileLock", return_value=mock_lock), \
            patch("pandas.read_parquet", side_effect=Exception("Parquet read error")), \
            patch("app.utils.file_utils.logger.error") as mock_logger_error, \
            patch.object(pd.DataFrame, "to_parquet") as mock_to_parquet:
        # Call save_to_parquet with sample dataframe and a test filename that "exists but can't be read"
        save_to_parquet(sample_dataframe, "test_dir/test_file.parquet")

        # Verify directory was created, error was logged, and to_parquet was still called to save the new data
        mock_makedirs.assert_called_once_with("test_dir", exist_ok=True)
        mock_logger_error.assert_called_once()
        mock_to_parquet.assert_called_once()
        # Verify FileLock was used
        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()


def test_save_to_parquet_invalid_data_type():
    """Test that save_to_parquet raises ValueError when given an invalid data type"""

    # Call save_to_parquet with a dictionary (invalid data type) and verify it raises the expected ValueError
    # Note: validation happens before acquiring lock, so we don't need to mock FileLock
    with pytest.raises(ValueError, match="Data must be a Pandas DataFrame for parquet format."):
        save_to_parquet({"key": "value"}, "test_dir/test_file.parquet")


def test_save_to_parquet_lock_timeout(sample_dataframe):
    """Test that save_to_parquet handles FileLock timeout gracefully"""

    from filelock import Timeout as FileLockTimeout

    # Mock FileLock to raise timeout exception
    mock_lock = MagicMock()
    mock_lock.__enter__.side_effect = FileLockTimeout("test_file.parquet.lock")

    with patch("os.makedirs"), \
            patch("app.utils.file_utils.FileLock", return_value=mock_lock), \
            patch("app.utils.file_utils.logger.error") as mock_logger_error, \
            pytest.raises(FileLockTimeout):
        save_to_parquet(sample_dataframe, "test_dir/test_file.parquet")

        # Verify error was logged
        mock_logger_error.assert_called_once()


def test_save_to_parquet_general_exception(sample_dataframe):
    """Test that save_to_parquet handles general exceptions during save"""

    mock_lock = MagicMock()

    with patch("os.path.exists", return_value=False), \
            patch("os.makedirs"), \
            patch("app.utils.file_utils.FileLock", return_value=mock_lock), \
            patch.object(pd.DataFrame, "to_parquet", side_effect=Exception("Write error")), \
            patch("app.utils.file_utils.logger.error") as mock_logger_error, \
            pytest.raises(Exception, match="Write error"):
        save_to_parquet(sample_dataframe, "test_dir/test_file.parquet")

        # Verify error was logged
        assert mock_logger_error.call_count >= 1


def test_save_to_parquet_uses_absolute_path(sample_dataframe):
    """Test that save_to_parquet uses absolute path for lock file"""

    mock_lock = MagicMock()

    with patch("os.path.exists", return_value=False), \
            patch("os.makedirs"), \
            patch("os.path.abspath", return_value="/absolute/path/test_file.parquet") as mock_abspath, \
            patch("app.utils.file_utils.FileLock", return_value=mock_lock) as mock_filelock, \
            patch.object(pd.DataFrame, "to_parquet"):
        save_to_parquet(sample_dataframe, "relative/path/test_file.parquet")

        # Verify absolute path was computed
        mock_abspath.assert_called_once_with("relative/path/test_file.parquet")

        # Verify FileLock was called with absolute path + .lock
        mock_filelock.assert_called_once_with("/absolute/path/test_file.parquet.lock", timeout=120)
