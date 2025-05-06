import json
import os
from unittest.mock import patch, mock_open, MagicMock

import pandas as pd
import pytest

from app.utils.file_utils import (
    load_file, save_file, json_to_dataframe, save_to_csv, load_data_from_json_files
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
    # Mock file existence and open operation, prepare JSON data for loading
    with patch("os.path.exists", return_value=True), \
            patch("builtins.open", mock_open(read_data=json.dumps(sample_json_data))), \
            patch("json.load", return_value=sample_json_data):
        # Call load_file function with a test filename
        result = load_file("test_file.json")
        # Verify the function returns the expected JSON data
        assert result == sample_json_data


def test_load_file_nonexistent_file():
    # Mock file existence to return False (file doesn't exist)
    with patch("os.path.exists", return_value=False):
        # Call load_file function with a nonexistent filename
        result = load_file("nonexistent_file.json")
        # Verify the function returns an empty dictionary when file doesn't exist
        assert result == {}


def test_save_file(sample_json_data):
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
    # No additional setup needed, using the sample_json_data fixture
    # Convert the dictionary data to a DataFrame using json_to_dataframe
    result = json_to_dataframe(sample_json_data)
    # Verify the result is a DataFrame with expected columns from the dictionary keys
    assert isinstance(result, pd.DataFrame)
    assert "item1" in result.columns
    assert "item2" in result.columns


def test_json_to_dataframe_list():
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
    # No setup needed, testing with an empty dictionary
    # Convert an empty dictionary to a DataFrame using json_to_dataframe
    result = json_to_dataframe({})
    # Verify the result is an empty DataFrame
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_json_to_dataframe_with_date_fields():
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


def test_save_to_csv_new_file(sample_dataframe):
    # Mock file existence check to return False (file doesn't exist) and patch DataFrame.to_csv
    with patch("os.path.exists", return_value=False), \
            patch.object(pd.DataFrame, "to_csv") as mock_to_csv:
        # Call save_to_csv with sample dataframe and a test filename
        save_to_csv(sample_dataframe, "test_file.csv")
        # Verify to_csv was called once with the correct filename and index=False
        mock_to_csv.assert_called_once_with("test_file.csv", index=False)


def test_save_to_csv_existing_file(sample_dataframe):
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


def test_save_to_csv_with_dict():
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
    # No setup needed, testing error handling with invalid data type
    # Call save_to_csv with a string (invalid data type) and verify it raises the expected ValueError
    with pytest.raises(ValueError, match="Data must be either a Pandas DataFrame or a dictionary."):
        save_to_csv("invalid_data", "test_file.csv")


@patch("app.utils.file_utils.glob")
@patch("app.utils.file_utils.load_file")
@patch("app.utils.file_utils.json_to_dataframe")
@patch("pandas.concat")
def test_load_data_from_json_files(mock_concat, mock_json_to_df, mock_load_file, mock_glob):
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
    # Mock glob to return an empty list, simulating no matching files found
    mock_glob.return_value = []

    # Call load_data_from_json_files with test parameters when no files exist
    result = load_data_from_json_files(
        "test_dir", "prefix", ["date"], "YYYY-MM-DD", "timestamp"
    )

    # Verify the function returns an empty DataFrame when no files are found
    assert isinstance(result, pd.DataFrame)
    assert result.empty
