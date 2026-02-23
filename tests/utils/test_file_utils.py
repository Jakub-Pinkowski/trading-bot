"""
Tests for File Utils Module.

Tests cover:
- JSON file loading: existing and nonexistent files
- JSON file saving with directory creation
- JSON to DataFrame conversion: dict, list, empty, date fields, orient modes, unsupported formats
- CSV saving: new file, append to existing, read error fallback, dict input, invalid type
- JSON file batch loading, concatenation, and empty-file skipping
- Parquet saving: new file, append, read error fallback, invalid type, lock timeout, general error, absolute path
"""
import json
import os
from unittest.mock import MagicMock, mock_open

import pandas as pd
import pytest

from app.utils.file_utils import (
    load_file,
    save_file,
    json_to_dataframe,
    save_to_csv,
    load_data_from_json_files,
    save_to_parquet,
)


# ==================== Test Classes ====================

class TestLoadFile:
    """Test JSON file loading."""

    def test_existing_file_returns_json(self, monkeypatch, sample_json_data):
        """Test load_file returns parsed JSON data from an existing file."""
        monkeypatch.setattr(os.path, "exists", lambda path: True)
        monkeypatch.setattr("builtins.open", mock_open(read_data=json.dumps(sample_json_data)))
        monkeypatch.setattr(json, "load", lambda f: sample_json_data)

        result = load_file("test_file.json")

        assert result == sample_json_data

    def test_nonexistent_file_returns_empty_dict(self, monkeypatch):
        """Test load_file returns an empty dict when the file does not exist."""
        monkeypatch.setattr(os.path, "exists", lambda path: False)

        result = load_file("nonexistent_file.json")

        assert result == {}


class TestSaveFile:
    """Test JSON file saving."""

    def test_creates_directory_and_writes_json(self, monkeypatch, sample_json_data):
        """Test save_file creates the parent directory and writes JSON data."""
        mock_makedirs = MagicMock()
        mock_json_dump = MagicMock()
        m = mock_open()

        monkeypatch.setattr(os, "makedirs", mock_makedirs)
        monkeypatch.setattr("builtins.open", m)
        monkeypatch.setattr(json, "dump", mock_json_dump)

        save_file(sample_json_data, "test_dir/test_file.json")

        mock_makedirs.assert_called_once_with("test_dir", exist_ok=True)
        m.assert_called_once_with("test_dir/test_file.json", "w")
        mock_json_dump.assert_called_once()


class TestJsonToDataframe:
    """Test JSON to DataFrame conversion."""

    def test_dict_columns_orient(self, sample_json_data):
        """Test dict input with default columns orient produces expected columns."""
        result = json_to_dataframe(sample_json_data)

        assert isinstance(result, pd.DataFrame)
        assert "item1" in result.columns
        assert "item2" in result.columns

    def test_list_of_dicts(self):
        """Test list input produces DataFrame with expected columns and values."""
        data = [{"name": "Item 1", "value": 100}, {"name": "Item 2", "value": 200}]

        result = json_to_dataframe(data)

        assert isinstance(result, pd.DataFrame)
        assert result["name"].tolist() == ["Item 1", "Item 2"]
        assert result["value"].tolist() == [100, 200]

    def test_empty_dict_returns_empty_dataframe(self):
        """Test empty dict input returns an empty DataFrame."""
        result = json_to_dataframe({})

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_date_fields_converted_to_datetime(self):
        """Test date_fields parameter converts string columns to datetime type."""
        data = [
            {"name": "Item 1", "date": "2023-01-01"},
            {"name": "Item 2", "date": "2023-01-02"},
        ]

        result = json_to_dataframe(data, date_fields=["date"])

        assert pd.api.types.is_datetime64_dtype(result["date"])

    def test_unsupported_format_raises_value_error(self):
        """Test ValueError raised for unsupported data types."""
        with pytest.raises(ValueError, match="Unsupported data format"):
            json_to_dataframe("unsupported_format")

    def test_dict_with_index_orient_and_custom_index_name(self, sample_json_data):
        """Test index orient with a custom index name produces the expected column."""
        result = json_to_dataframe(sample_json_data, orient="index", index_name="custom_index")

        assert isinstance(result, pd.DataFrame)
        assert "custom_index" in result.columns
        assert result["custom_index"].tolist() == ["item1", "item2"]


class TestSaveToCsv:
    """Test CSV file saving and appending."""

    def test_new_file_saves_dataframe(self, monkeypatch, sample_dataframe):
        """Test new file path causes direct save without concatenation."""
        mock_to_csv = MagicMock()
        monkeypatch.setattr(os.path, "exists", lambda path: False)
        monkeypatch.setattr(pd.DataFrame, "to_csv", mock_to_csv)

        save_to_csv(sample_dataframe, "test_file.csv")

        mock_to_csv.assert_called_once_with("test_file.csv", index=False)

    def test_existing_file_appends_data(self, monkeypatch, sample_dataframe):
        """Test existing file path reads, concatenates, deduplicates, then saves."""
        existing_df = pd.DataFrame({"name": ["Item 3"], "value": [300]})
        mock_to_csv = MagicMock()
        mock_concat = MagicMock(return_value=pd.DataFrame())

        monkeypatch.setattr(os.path, "exists", lambda path: True)
        monkeypatch.setattr(pd, "read_csv", MagicMock(return_value=existing_df))
        monkeypatch.setattr(pd.DataFrame, "to_csv", mock_to_csv)
        monkeypatch.setattr(pd, "concat", mock_concat)

        save_to_csv(sample_dataframe, "test_file.csv")

        mock_concat.assert_called_once()
        mock_to_csv.assert_called_once()

    def test_existing_file_read_error_logs_and_saves(self, monkeypatch, sample_dataframe):
        """Test read error on existing file is logged and save still proceeds."""
        mock_logger = MagicMock()
        mock_to_csv = MagicMock()

        monkeypatch.setattr(os.path, "exists", lambda path: True)
        monkeypatch.setattr(pd, "read_csv", MagicMock(side_effect=Exception("CSV read error")))
        monkeypatch.setattr("app.utils.file_utils.logger", mock_logger)
        monkeypatch.setattr(pd.DataFrame, "to_csv", mock_to_csv)

        save_to_csv(sample_dataframe, "test_file.csv")

        mock_logger.error.assert_called_once()
        mock_to_csv.assert_called_once()

    def test_dict_converted_and_saved(self, monkeypatch):
        """Test dict input is converted to DataFrame and saved."""
        mock_to_csv = MagicMock()
        monkeypatch.setattr(os.path, "exists", lambda path: False)
        monkeypatch.setattr(pd.DataFrame, "to_csv", mock_to_csv)

        save_to_csv({"key1": "value1", "key2": "value2"}, "test_file.csv")

        assert mock_to_csv.called
        _, kwargs = mock_to_csv.call_args
        assert kwargs["index"] is False

    def test_invalid_data_type_raises_value_error(self):
        """Test ValueError raised for data that is neither DataFrame nor dict."""
        with pytest.raises(ValueError, match="Data must be either a Pandas DataFrame or a dictionary"):
            save_to_csv("invalid_data", "test_file.csv")


class TestLoadDataFromJsonFiles:
    """Test batch JSON file loading and DataFrame concatenation."""

    def test_loads_and_combines_multiple_files(self, monkeypatch):
        """Test multiple JSON files are loaded, converted, and concatenated."""
        mock_df1 = MagicMock()
        mock_df2 = MagicMock()
        mock_concat_result = MagicMock()
        mock_concat_result.sort_index.return_value = mock_concat_result
        mock_concat_result.reset_index.return_value = mock_concat_result

        mock_glob = MagicMock(return_value=["file1.json", "file2.json"])
        monkeypatch.setattr("app.utils.file_utils.glob", mock_glob)
        monkeypatch.setattr(
            "app.utils.file_utils.load_file",
            MagicMock(side_effect=[{"data1": "value1"}, {"data2": "value2"}]),
        )
        monkeypatch.setattr(
            "app.utils.file_utils.json_to_dataframe",
            MagicMock(side_effect=[mock_df1, mock_df2]),
        )
        monkeypatch.setattr(pd, "concat", MagicMock(return_value=mock_concat_result))

        result = load_data_from_json_files("test_dir", "prefix", ["date"], "YYYY-MM-DD", "timestamp")

        mock_glob.assert_called_once_with(os.path.join("test_dir", "prefix_*.json"))
        assert result == mock_concat_result

    def test_no_files_returns_empty_dataframe(self, monkeypatch):
        """Test empty glob result returns an empty DataFrame."""
        monkeypatch.setattr("app.utils.file_utils.glob", MagicMock(return_value=[]))

        result = load_data_from_json_files("test_dir", "prefix", ["date"], "YYYY-MM-DD", "timestamp")

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_skips_empty_json_files(self, monkeypatch):
        """Test files that load as empty dicts are excluded from concatenation."""
        mock_df = MagicMock()
        mock_concat_result = MagicMock()
        mock_concat_result.sort_index.return_value = mock_concat_result
        mock_concat_result.reset_index.return_value = mock_concat_result

        monkeypatch.setattr("app.utils.file_utils.glob", MagicMock(return_value=["empty.json", "valid.json"]))
        monkeypatch.setattr(
            "app.utils.file_utils.load_file",
            MagicMock(side_effect=[{}, {"data": "value"}]),
        )
        mock_json_to_df = MagicMock(return_value=mock_df)
        monkeypatch.setattr("app.utils.file_utils.json_to_dataframe", mock_json_to_df)
        monkeypatch.setattr(pd, "concat", MagicMock(return_value=mock_concat_result))

        load_data_from_json_files("test_dir", "prefix", ["date"], "YYYY-MM-DD", "timestamp")

        # Empty file is skipped — json_to_dataframe only called once for the valid file
        assert mock_json_to_df.call_count == 1


class TestSaveToParquet:
    """Test Parquet file saving with file locking."""

    def test_new_file_saves_dataframe(self, monkeypatch, sample_dataframe):
        """Test new file path creates directory and writes directly."""
        mock_lock = MagicMock()
        mock_makedirs = MagicMock()
        mock_to_parquet = MagicMock()

        monkeypatch.setattr(os, "makedirs", mock_makedirs)
        monkeypatch.setattr(os.path, "exists", lambda path: False)
        monkeypatch.setattr("app.utils.file_utils.FileLock", MagicMock(return_value=mock_lock))
        monkeypatch.setattr(pd.DataFrame, "to_parquet", mock_to_parquet)

        save_to_parquet(sample_dataframe, "test_dir/test_file.parquet")

        mock_makedirs.assert_called_once_with("test_dir", exist_ok=True)
        mock_to_parquet.assert_called_once_with("test_dir/test_file.parquet", index=False)
        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()

    def test_existing_file_appends_data(self, monkeypatch, sample_dataframe):
        """Test existing file path reads, concatenates, deduplicates, then saves."""
        existing_df = pd.DataFrame({"name": ["Item 3"], "value": [300]})
        mock_lock = MagicMock()
        mock_to_parquet = MagicMock()
        mock_concat = MagicMock(return_value=pd.DataFrame())

        monkeypatch.setattr(os, "makedirs", MagicMock())
        monkeypatch.setattr(os.path, "exists", lambda path: True)
        monkeypatch.setattr("app.utils.file_utils.FileLock", MagicMock(return_value=mock_lock))
        monkeypatch.setattr(pd, "read_parquet", MagicMock(return_value=existing_df))
        monkeypatch.setattr(pd.DataFrame, "to_parquet", mock_to_parquet)
        monkeypatch.setattr(pd, "concat", mock_concat)

        save_to_parquet(sample_dataframe, "test_dir/test_file.parquet")

        mock_concat.assert_called_once()
        mock_to_parquet.assert_called_once()
        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()

    def test_existing_file_read_error_logs_and_saves(self, monkeypatch, sample_dataframe):
        """Test read error on existing file is logged and save still proceeds."""
        mock_lock = MagicMock()
        mock_logger = MagicMock()
        mock_to_parquet = MagicMock()

        monkeypatch.setattr(os, "makedirs", MagicMock())
        monkeypatch.setattr(os.path, "exists", lambda path: True)
        monkeypatch.setattr("app.utils.file_utils.FileLock", MagicMock(return_value=mock_lock))
        monkeypatch.setattr(pd, "read_parquet", MagicMock(side_effect=Exception("Parquet read error")))
        monkeypatch.setattr("app.utils.file_utils.logger", mock_logger)
        monkeypatch.setattr(pd.DataFrame, "to_parquet", mock_to_parquet)

        save_to_parquet(sample_dataframe, "test_dir/test_file.parquet")

        mock_logger.error.assert_called_once()
        mock_to_parquet.assert_called_once()
        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()

    def test_invalid_data_type_raises_value_error(self):
        """Test ValueError raised when data is not a DataFrame."""
        with pytest.raises(ValueError, match="Data must be a Pandas DataFrame for parquet format"):
            save_to_parquet({"key": "value"}, "test_dir/test_file.parquet")

    def test_lock_timeout_logs_and_reraises(self, monkeypatch, sample_dataframe):
        """Test FileLock timeout is logged and re-raised."""
        from filelock import Timeout as FileLockTimeout

        mock_lock = MagicMock()
        mock_lock.__enter__.side_effect = FileLockTimeout("test_file.parquet.lock")
        mock_logger = MagicMock()

        monkeypatch.setattr(os, "makedirs", MagicMock())
        monkeypatch.setattr("app.utils.file_utils.FileLock", MagicMock(return_value=mock_lock))
        monkeypatch.setattr("app.utils.file_utils.logger", mock_logger)

        with pytest.raises(FileLockTimeout):
            save_to_parquet(sample_dataframe, "test_dir/test_file.parquet")

        mock_logger.error.assert_called_once()

    def test_general_exception_logs_and_reraises(self, monkeypatch, sample_dataframe):
        """Test unexpected write error is logged and re-raised."""
        mock_lock = MagicMock()
        mock_logger = MagicMock()

        monkeypatch.setattr(os, "makedirs", MagicMock())
        monkeypatch.setattr(os.path, "exists", lambda path: False)
        monkeypatch.setattr("app.utils.file_utils.FileLock", MagicMock(return_value=mock_lock))
        monkeypatch.setattr(pd.DataFrame, "to_parquet", MagicMock(side_effect=Exception("Write error")))
        monkeypatch.setattr("app.utils.file_utils.logger", mock_logger)

        with pytest.raises(Exception, match="Write error"):
            save_to_parquet(sample_dataframe, "test_dir/test_file.parquet")

        assert mock_logger.error.call_count >= 1

    def test_uses_absolute_path_for_lock(self, monkeypatch, sample_dataframe):
        """Test FileLock is called with the absolute path and .lock suffix."""
        mock_lock = MagicMock()
        mock_filelock = MagicMock(return_value=mock_lock)

        monkeypatch.setattr(os, "makedirs", MagicMock())
        monkeypatch.setattr(os.path, "exists", lambda path: False)
        monkeypatch.setattr(
            os.path, "abspath", MagicMock(return_value="/absolute/path/test_file.parquet")
        )
        monkeypatch.setattr("app.utils.file_utils.FileLock", mock_filelock)
        monkeypatch.setattr(pd.DataFrame, "to_parquet", MagicMock())

        save_to_parquet(sample_dataframe, "relative/path/test_file.parquet")

        mock_filelock.assert_called_once_with("/absolute/path/test_file.parquet.lock", timeout=120)
