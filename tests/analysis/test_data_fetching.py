from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from app.analysis.data_fetching import (
    get_alerts_data,
    get_tw_alerts_data,
    get_trades_data,
    fetch_trades_data
)


@pytest.fixture
def sample_alerts_data():
    """Sample alerts data for testing."""

    return pd.DataFrame([
        {"timestamp": "23-05-01 10:30:45", "symbol": "ZW1!", "side": "B", "price": "34.20"},
        {"timestamp": "23-05-01 11:45:30", "symbol": "ZC1!", "side": "S", "price": "423.20"}
    ])


@pytest.fixture
def sample_tw_alerts_data():
    """Sample TradingView alerts data for testing."""

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
def test_get_alerts_data(mock_load_data, sample_alerts_data):
    """Test getting alerts data."""

    # Setup mock
    mock_load_data.return_value = sample_alerts_data

    # Call the function
    result = get_alerts_data()

    # Verify the result
    assert not result.empty
    assert list(result.columns) == list(sample_alerts_data.columns)
    assert len(result) == len(sample_alerts_data)

    # Verify the mock was called with the correct arguments
    mock_load_data.assert_called_once()


@patch('app.analysis.data_fetching.load_data_from_json_files')
def test_get_alerts_data_empty(mock_load_data):
    """Test getting alerts data when no data is available."""

    # Setup mock to return empty DataFrame
    mock_load_data.return_value = pd.DataFrame()

    # Call the function
    result = get_alerts_data()

    # Verify the result is an empty DataFrame with expected columns
    assert result.empty
    assert list(result.columns) == ['timestamp', 'symbol', 'side', 'price']

    # Verify the mock was called
    mock_load_data.assert_called_once()


@patch('app.analysis.data_fetching.os.listdir')
@patch('app.analysis.data_fetching.os.path.exists')
@patch('app.analysis.data_fetching.pd.read_csv')
def test_get_tw_alerts_data(mock_read_csv, mock_exists, mock_listdir, sample_tw_alerts_data):
    """Test getting TradingView alerts data."""

    # Setup mocks
    mock_listdir.return_value = ['TradingView_Alerts_Log_2025-05-05.csv']
    mock_exists.return_value = True
    mock_read_csv.return_value = sample_tw_alerts_data

    # Call the function
    result = get_tw_alerts_data()

    # Verify the result
    assert not result.empty
    assert list(result.columns) == list(sample_tw_alerts_data.columns)
    assert len(result) == len(sample_tw_alerts_data)

    # Verify the mocks were called
    mock_listdir.assert_called_once()
    mock_exists.assert_called_once()
    mock_read_csv.assert_called_once()


@patch('app.analysis.data_fetching.os.listdir')
def test_get_tw_alerts_data_no_files(mock_listdir):
    """Test getting TradingView alerts data when no files are available."""

    # Setup mock to return empty list
    mock_listdir.return_value = []

    # Call the function
    result = get_tw_alerts_data()

    # Verify the result is an empty DataFrame
    assert result.empty

    # Verify the mock was called
    mock_listdir.assert_called_once()


@patch('app.analysis.data_fetching.os.listdir')
def test_get_tw_alerts_data_invalid_date_format(mock_listdir):
    """Test getting TradingView alerts data with invalid date format in filename."""

    # Setup mock to return files with invalid date format
    mock_listdir.return_value = ['TradingView_Alerts_Log_invalid-date.csv']

    # Call the function
    result = get_tw_alerts_data()

    # Verify the result is an empty DataFrame
    assert result.empty

    # Verify the mock was called
    mock_listdir.assert_called_once()


@patch('app.analysis.data_fetching.os.listdir')
@patch('app.analysis.data_fetching.os.path.exists')
def test_get_tw_alerts_data_file_not_exists(mock_exists, mock_listdir):
    """Test getting TradingView alerts data when file doesn't exist."""

    # Setup mocks
    mock_listdir.return_value = ['TradingView_Alerts_Log_2025-05-05.csv']
    mock_exists.return_value = False

    # Call the function
    result = get_tw_alerts_data()

    # Verify the result is an empty DataFrame
    assert result.empty

    # Verify the mocks were called
    mock_listdir.assert_called_once()
    mock_exists.assert_called_once()


@patch('app.analysis.data_fetching.os.listdir')
@patch('app.analysis.data_fetching.os.path.exists')
@patch('app.analysis.data_fetching.pd.read_csv')
def test_get_tw_alerts_data_read_exception(mock_read_csv, mock_exists, mock_listdir):
    """Test getting TradingView alerts data when reading file raises exception."""

    # Setup mocks
    mock_listdir.return_value = ['TradingView_Alerts_Log_2025-05-05.csv']
    mock_exists.return_value = True
    mock_read_csv.side_effect = Exception("Error reading file")

    # Call the function
    result = get_tw_alerts_data()

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

    # Setup mock to return None on first call, then data on second call
    mock_api_get.side_effect = [None, sample_trades_json]

    # Call the function
    result = fetch_trades_data(max_retries=2, retry_delay=0.1)

    # Verify the result
    assert result["success"] is True

    # Verify the mocks were called
    assert mock_api_get.call_count == 2
    mock_sleep.assert_called_once_with(0.1)
