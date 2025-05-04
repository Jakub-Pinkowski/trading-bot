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
    # Setup
    with patch("os.path.exists", return_value=True), \
            patch("builtins.open", mock_open(read_data=json.dumps(sample_json_data))), \
            patch("json.load", return_value=sample_json_data):
        # Execute
        result = load_file("test_file.json")
        # Assert
        assert result == sample_json_data


def test_load_file_nonexistent_file():
    # Setup
    with patch("os.path.exists", return_value=False):
        # Execute
        result = load_file("nonexistent_file.json")
        # Assert
        assert result == {}


def test_save_file(sample_json_data):
    # Setup
    m = mock_open()
    with patch("os.makedirs") as mock_makedirs, \
            patch("builtins.open", m), \
            patch("json.dump") as mock_json_dump:
        # Execute
        save_file(sample_json_data, "test_dir/test_file.json")
        # Assert
        mock_makedirs.assert_called_once_with("test_dir", exist_ok=True)
        m.assert_called_once_with("test_dir/test_file.json", "w")
        mock_json_dump.assert_called_once()


def test_json_to_dataframe_dict(sample_json_data):
    # Setup
    # Execute
    result = json_to_dataframe(sample_json_data)
    # Assert
    assert isinstance(result, pd.DataFrame)
    assert "item1" in result.columns
    assert "item2" in result.columns


def test_json_to_dataframe_list():
    # Setup
    data = [{"name": "Item 1", "value": 100}, {"name": "Item 2", "value": 200}]
    # Execute
    result = json_to_dataframe(data)
    # Assert
    assert isinstance(result, pd.DataFrame)
    assert "name" in result.columns
    assert "value" in result.columns
    assert result["name"].tolist() == ["Item 1", "Item 2"]
    assert result["value"].tolist() == [100, 200]


def test_json_to_dataframe_empty():
    # Setup
    # Execute
    result = json_to_dataframe({})
    # Assert
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_json_to_dataframe_with_date_fields():
    # Setup
    data = [
        {"name": "Item 1", "date": "2023-01-01"},
        {"name": "Item 2", "date": "2023-01-02"}
    ]
    # Execute
    result = json_to_dataframe(data, date_fields=["date"])
    # Assert
    assert isinstance(result, pd.DataFrame)
    assert pd.api.types.is_datetime64_dtype(result["date"])


def test_save_to_csv_new_file(sample_dataframe):
    # Setup
    with patch("os.path.exists", return_value=False), \
            patch.object(pd.DataFrame, "to_csv") as mock_to_csv:
        # Execute
        save_to_csv(sample_dataframe, "test_file.csv")
        # Assert
        mock_to_csv.assert_called_once_with("test_file.csv", index=False)


def test_save_to_csv_existing_file(sample_dataframe):
    # Setup
    existing_df = pd.DataFrame({
        "name": ["Item 3"],
        "value": [300]
    })
    with patch("os.path.exists", return_value=True), \
            patch("pandas.read_csv", return_value=existing_df), \
            patch.object(pd.DataFrame, "to_csv") as mock_to_csv, \
            patch("pandas.concat", return_value=pd.DataFrame()) as mock_concat:
        # Execute
        save_to_csv(sample_dataframe, "test_file.csv")
        # Assert
        mock_concat.assert_called_once()
        mock_to_csv.assert_called_once()


def test_save_to_csv_with_dict():
    # Setup
    data = {"key1": "value1", "key2": "value2"}
    expected_df = pd.DataFrame([("key1", "value1"), ("key2", "value2")], columns=["Key", "Value"])
    with patch("os.path.exists", return_value=False), \
            patch.object(pd.DataFrame, "to_csv") as mock_to_csv:
        # Execute
        save_to_csv(data, "test_file.csv")
        # Assert
        assert mock_to_csv.called, "to_csv was not called"
        args, kwargs = mock_to_csv.call_args
        assert args[0] == "test_file.csv"
        assert kwargs["index"] == False


def test_save_to_csv_invalid_data_type():
    # Setup
    # Execute & Assert
    with pytest.raises(ValueError, match="Data must be either a Pandas DataFrame or a dictionary."):
        save_to_csv("invalid_data", "test_file.csv")


@patch("app.utils.file_utils.glob")
@patch("app.utils.file_utils.load_file")
@patch("app.utils.file_utils.json_to_dataframe")
@patch("pandas.concat")
def test_load_data_from_json_files(mock_concat, mock_json_to_df, mock_load_file, mock_glob):
    # Setup
    mock_glob.return_value = ["file1.json", "file2.json"]
    mock_load_file.side_effect = [{"data1": "value1"}, {"data2": "value2"}]
    mock_df1 = MagicMock()
    mock_df2 = MagicMock()
    mock_json_to_df.side_effect = [mock_df1, mock_df2]
    mock_concat_result = MagicMock()
    mock_concat.return_value = mock_concat_result
    mock_concat_result.sort_index.return_value = mock_concat_result
    mock_concat_result.reset_index.return_value = mock_concat_result

    # Execute
    result = load_data_from_json_files(
        "test_dir", "prefix", ["date"], "YYYY-MM-DD", "timestamp"
    )

    # Assert
    mock_glob.assert_called_once_with(os.path.join("test_dir", "prefix_*.json"))
    assert mock_load_file.call_count == 2
    assert mock_json_to_df.call_count == 2
    mock_concat.assert_called_once_with([mock_df1, mock_df2])
    assert result == mock_concat_result


@patch("app.utils.file_utils.glob")
def test_load_data_from_json_files_no_files(mock_glob):
    # Setup
    mock_glob.return_value = []

    # Execute
    result = load_data_from_json_files(
        "test_dir", "prefix", ["date"], "YYYY-MM-DD", "timestamp"
    )

    # Assert
    assert isinstance(result, pd.DataFrame)
    assert result.empty
