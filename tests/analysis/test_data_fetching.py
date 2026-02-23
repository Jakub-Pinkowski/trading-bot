import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.analysis.data_fetching import (
    get_ibkr_alerts_data,
    get_tv_alerts_data,
    get_trades_data,
    fetch_trades_data,
    json_to_dataframe,
    load_data_from_json_files,
)


@pytest.fixture
def sample_alerts_data():
    """Sample ibkr_alerts data for testing."""

    return pd.DataFrame([
        {"timestamp": "23-05-01 10:30:45", "symbol": "ZW1!", "side": "B", "price": "34.20"},
        {"timestamp": "23-05-01 11:45:30", "symbol": "ZC1!", "side": "S", "price": "423.20"}
    ])


@pytest.fixture
def sample_tv_alerts_data():
    """Sample TradingView ibkr_alerts data for testing."""

    return pd.DataFrame([
        {
            "Alert ID": "2223583442",
            "Ticker": "NYMEX:MCL1!, 15m",
            "Name": "",
            "Description": '{"symbol":"MCL1!","side":"S","price":56.98}',
            "Time": "2025-05-05T14:07:00Z"
        }
    ])


@pytest.fixture
def sample_trades_data():
    """Sample trades data for testing."""

    return pd.DataFrame([
        {
            "trade_time": "20231211-18:00:49",
            "symbol": "AAPL",
            "side": "S",
            "price": "192.26",
            "size": 5,
            "commission": "1.01",
            "net_amount": 961.3
        }
    ])


@pytest.fixture
def sample_trades_json():
    """Sample trades JSON data for testing."""

    return [
        {
            "execution_id": "0000e0d5.6576fd38.01.01",
            "symbol": "AAPL",
            "side": "S",
            "trade_time": "20231211-18:00:49",
            "size": 5,
            "price": "192.26",
            "commission": "1.01",
            "net_amount": 961.3
        }
    ]


@patch('app.analysis.data_fetching.load_data_from_json_files')
def test_get_ibkr_alerts_data(mock_load_data, sample_alerts_data):
    """Test getting ibkr_alerts data."""

    # Setup mock
    mock_load_data.return_value = sample_alerts_data

    # Call the function
    result = get_ibkr_alerts_data()

    # Verify the result
    assert not result.empty
    assert list(result.columns) == list(sample_alerts_data.columns)
    assert len(result) == len(sample_alerts_data)

    # Verify the mock was called with the correct arguments
    mock_load_data.assert_called_once()


@patch('app.analysis.data_fetching.load_data_from_json_files')
def test_get_ibkr_alerts_data_empty(mock_load_data):
    """Test getting ibkr_alerts data when no data is available."""

    # Setup mock to return empty DataFrame
    mock_load_data.return_value = pd.DataFrame()

    # Call the function
    result = get_ibkr_alerts_data()

    # Verify the result is an empty DataFrame with expected columns
    assert result.empty
    assert list(result.columns) == ['timestamp', 'symbol', 'side', 'price']

    # Verify the mock was called
    mock_load_data.assert_called_once()


@patch('app.analysis.data_fetching.os.listdir')
@patch('app.analysis.data_fetching.os.path.exists')
@patch('app.analysis.data_fetching.pd.read_csv')
def test_get_tv_alerts_data(mock_read_csv, mock_exists, mock_listdir, sample_tv_alerts_data):
    """Test getting TradingView ibkr_alerts data."""

    # Setup mocks
    mock_listdir.return_value = ['TradingView_Alerts_Log_2025-05-05.csv']
    mock_exists.return_value = True
    mock_read_csv.return_value = sample_tv_alerts_data

    # Call the function
    result = get_tv_alerts_data()

    # Verify the result
    assert not result.empty
    assert list(result.columns) == list(sample_tv_alerts_data.columns)
    assert len(result) == len(sample_tv_alerts_data)

    # Verify the mocks were called
    mock_listdir.assert_called_once()
    mock_exists.assert_called_once()
    mock_read_csv.assert_called_once()


@patch('app.analysis.data_fetching.os.listdir')
def test_get_tv_alerts_data_no_files(mock_listdir):
    """Test getting TradingView ibkr_alerts data when no files are available."""

    # Setup mock to return empty list
    mock_listdir.return_value = []

    # Call the function
    result = get_tv_alerts_data()

    # Verify the result is an empty DataFrame
    assert result.empty

    # Verify the mock was called
    mock_listdir.assert_called_once()


@patch('app.analysis.data_fetching.os.listdir')
def test_get_tv_alerts_data_invalid_date_format(mock_listdir):
    """Test getting TradingView ibkr_alerts data with invalid date format in filename."""

    # Setup mock to return files with invalid date format
    mock_listdir.return_value = ['TradingView_Alerts_Log_invalid-date.csv']

    # Call the function
    result = get_tv_alerts_data()

    # Verify the result is an empty DataFrame
    assert result.empty

    # Verify the mock was called
    mock_listdir.assert_called_once()


@patch('app.analysis.data_fetching.os.listdir')
@patch('app.analysis.data_fetching.os.path.exists')
def test_get_tv_alerts_data_file_not_exists(mock_exists, mock_listdir):
    """Test getting TradingView ibkr_alerts data when file doesn't exist."""

    # Setup mocks
    mock_listdir.return_value = ['TradingView_Alerts_Log_2025-05-05.csv']
    mock_exists.return_value = False

    # Call the function
    result = get_tv_alerts_data()

    # Verify the result is an empty DataFrame
    assert result.empty

    # Verify the mocks were called
    mock_listdir.assert_called_once()
    mock_exists.assert_called_once()


@patch('app.analysis.data_fetching.os.listdir')
@patch('app.analysis.data_fetching.os.path.exists')
@patch('app.analysis.data_fetching.pd.read_csv')
def test_get_tv_alerts_data_read_exception(mock_read_csv, mock_exists, mock_listdir):
    """Test getting TradingView ibkr_alerts data when reading file raises exception."""

    # Setup mocks
    mock_listdir.return_value = ['TradingView_Alerts_Log_2025-05-05.csv']
    mock_exists.return_value = True
    mock_read_csv.side_effect = Exception("Error reading file")

    # Call the function
    result = get_tv_alerts_data()

    # Verify the result is an empty DataFrame
    assert result.empty

    # Verify the mocks were called
    mock_listdir.assert_called_once()
    mock_exists.assert_called_once()
    mock_read_csv.assert_called_once()


@patch('app.analysis.data_fetching.load_data_from_json_files')
@patch('app.analysis.data_fetching.fetch_trades_data')
def test_get_trades_data(mock_fetch_trades, mock_load_data, sample_trades_data):
    """Test getting trades data."""

    # Make sure 'trade_time' is within the last 24 hours to avoid recursion
    now = pd.Timestamp.now()
    sample_trades_data['trade_time'] = [now - pd.Timedelta(hours=1)]

    mock_load_data.return_value = sample_trades_data

    # Call the function
    result = get_trades_data()

    # Verify the result
    assert not result.empty
    assert list(result.columns) == list(sample_trades_data.columns)
    assert len(result) == len(sample_trades_data)

    # Verify the mock was called
    mock_load_data.assert_called_once()
    # Fetch trades should not be called if data is available
    mock_fetch_trades.assert_not_called()


@patch('app.analysis.data_fetching.load_data_from_json_files')
@patch('app.analysis.data_fetching.fetch_trades_data')
def test_get_trades_data_empty_fetch_success(mock_fetch_trades, mock_load_data):
    """Test getting trades data when no data is available but fetch succeeds."""

    # Setup mocks for first call (empty) and second call (with data)
    mock_load_data.side_effect = [pd.DataFrame(), pd.DataFrame([{"trade_time": datetime.now(), "symbol": "AAPL"}])]
    mock_fetch_trades.return_value = {"success": True}

    # Call the function
    result = get_trades_data()

    # Verify the result
    assert not result.empty

    # Verify the mocks were called
    assert mock_load_data.call_count == 2
    mock_fetch_trades.assert_called_once()


@patch('app.analysis.data_fetching.load_data_from_json_files')
@patch('app.analysis.data_fetching.fetch_trades_data')
def test_get_trades_data_empty_fetch_fail(mock_fetch_trades, mock_load_data):
    """Test getting trades data when no data is available and fetch fails."""

    # Setup mocks
    mock_load_data.return_value = pd.DataFrame()
    mock_fetch_trades.return_value = {"success": False, "error": "API error"}

    # Call the function
    result = get_trades_data()

    # Verify the result is an empty DataFrame with expected columns
    assert result.empty
    assert list(result.columns) == ['conid', 'side', 'price', 'trade_time']

    # Verify the mocks were called
    mock_load_data.assert_called_once()
    mock_fetch_trades.assert_called_once()


@patch('app.analysis.data_fetching.api_get')
@patch('app.analysis.data_fetching.save_trades_data')
def test_fetch_trades_data_success(mock_save_trades, mock_api_get, sample_trades_json):
    """Test fetching trades data successfully."""

    # Setup mocks
    mock_api_get.return_value = sample_trades_json

    # Call the function
    result = fetch_trades_data()

    # Verify the result
    assert result["success"] is True

    # Verify the mocks were called
    mock_api_get.assert_called_once()
    mock_save_trades.assert_called_once_with(sample_trades_json, mock_save_trades.call_args[0][1])


@patch('app.analysis.data_fetching.api_get')
def test_fetch_trades_data_api_error(mock_api_get):
    """Test fetching trades data with API error."""

    # Setup mock to raise exception
    mock_api_get.side_effect = Exception("API error")

    # Call the function
    result = fetch_trades_data()

    # Verify the result
    assert result["success"] is False
    assert "error" in result

    # Verify the mock was called
    mock_api_get.assert_called_once()


@patch('app.analysis.data_fetching.api_get')
def test_fetch_trades_data_no_data(mock_api_get):
    """Test fetching trades data with no data returned."""

    # Setup mock to return None
    mock_api_get.return_value = None

    # Call the function
    result = fetch_trades_data(max_retries=1)

    # Verify the result
    assert result["success"] is False
    assert "error" in result

    # Verify the mock was called
    mock_api_get.assert_called_once()


@patch('app.analysis.data_fetching.api_get')
@patch('app.analysis.data_fetching.time.sleep')
def test_fetch_trades_data_retry_success(mock_sleep, mock_api_get, sample_trades_json):
    """Test fetching trades data with retry that succeeds."""

    # Setup mock to return None on the first call, then data on the second call
    mock_api_get.side_effect = [None, sample_trades_json]

    # Call the function
    result = fetch_trades_data(max_retries=2, retry_delay=0.1)

    # Verify the result
    assert result["success"] is True

    # Verify the mocks were called
    assert mock_api_get.call_count == 2
    mock_sleep.assert_called_once_with(0.1)


# ==================== Test Classes ====================

class TestJsonToDataframe:
    """Test JSON to DataFrame conversion."""

    def test_dict_columns_orient(self):
        """Test dict input with default columns orient produces expected columns."""
        data = {"item1": {"value": 1}, "item2": {"value": 2}}

        result = json_to_dataframe(data)

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

    def test_dict_with_index_orient_and_custom_index_name(self):
        """Test index orient with a custom index name produces the expected column."""
        data = {"item1": {"value": 1}, "item2": {"value": 2}}

        result = json_to_dataframe(data, orient="index", index_name="custom_index")

        assert isinstance(result, pd.DataFrame)
        assert "custom_index" in result.columns
        assert result["custom_index"].tolist() == ["item1", "item2"]


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
        monkeypatch.setattr("app.analysis.data_fetching.glob", mock_glob)
        monkeypatch.setattr(
            "app.analysis.data_fetching.load_file",
            MagicMock(side_effect=[{"data1": "value1"}, {"data2": "value2"}]),
        )
        monkeypatch.setattr(
            "app.analysis.data_fetching.json_to_dataframe",
            MagicMock(side_effect=[mock_df1, mock_df2]),
        )
        monkeypatch.setattr(pd, "concat", MagicMock(return_value=mock_concat_result))

        result = load_data_from_json_files("test_dir", "prefix", ["date"], "YYYY-MM-DD", "timestamp")

        mock_glob.assert_called_once_with(os.path.join("test_dir", "prefix_*.json"))
        assert result == mock_concat_result

    def test_no_files_returns_empty_dataframe(self, monkeypatch):
        """Test empty glob result returns an empty DataFrame."""
        monkeypatch.setattr("app.analysis.data_fetching.glob", MagicMock(return_value=[]))

        result = load_data_from_json_files("test_dir", "prefix", ["date"], "YYYY-MM-DD", "timestamp")

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_skips_empty_json_files(self, monkeypatch):
        """Test files that load as empty dicts are excluded from concatenation."""
        mock_df = MagicMock()
        mock_concat_result = MagicMock()
        mock_concat_result.sort_index.return_value = mock_concat_result
        mock_concat_result.reset_index.return_value = mock_concat_result

        monkeypatch.setattr(
            "app.analysis.data_fetching.glob",
            MagicMock(return_value=["empty.json", "valid.json"]),
        )
        monkeypatch.setattr(
            "app.analysis.data_fetching.load_file",
            MagicMock(side_effect=[{}, {"data": "value"}]),
        )
        mock_json_to_df = MagicMock(return_value=mock_df)
        monkeypatch.setattr("app.analysis.data_fetching.json_to_dataframe", mock_json_to_df)
        monkeypatch.setattr(pd, "concat", MagicMock(return_value=mock_concat_result))

        load_data_from_json_files("test_dir", "prefix", ["date"], "YYYY-MM-DD", "timestamp")

        # Empty file is skipped — json_to_dataframe only called once for the valid file
        assert mock_json_to_df.call_count == 1
