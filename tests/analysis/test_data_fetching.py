import os
from datetime import datetime, timedelta
from io import StringIO
from unittest.mock import patch

import pandas as pd
import pytest

from app.analysis.data_fetching import get_alerts_data, get_tw_alerts_data, get_trades_data, fetch_trades_data
from config import TIMEFRAME_TO_ANALYZE, TW_ALERTS_DIR


@pytest.fixture
def sample_alerts():
    return [
        {
            "symbol": "ZW1!",
            "side": "B",
            "price": "34.20",
            "dummy": "YES",
            "timestamp": "23-05-05 14:07:00"
        },
        {
            "symbol": "ZC1!",
            "side": "S",
            "price": "32.20",
            "timestamp": "23-05-05 12:47:43"
        }
    ]


@pytest.fixture
def sample_tw_alerts_csv():
    return """Alert ID\tTicker\tName\tDescription\tTime
2223583442\tNYMEX:MCL1!, 15m\t\t{"symbol":"MCL1!","side":"S","price":56.98}\t2025-05-05T14:07:00Z
2223583442\tNYMEX:MCL1!, 15m\t\t{"symbol":"MCL1!","side":"S","price":56.98}\t2025-05-05T12:47:43Z
2223583442\tNYMEX:MCL1!, 15m\t\t{"symbol":"MCL1!","side":"S","price":56.98}\t2025-05-05T12:34:56Z
2223584382\tNYMEX:MNG1!, 15m\t\t{"symbol":"MNG1!","side":"S","price":3.694}\t2025-05-05T12:06:09Z
2223584182\tNYMEX:MNG1!, 15m\t\t{"symbol":"MNG1!","side":"B","price":3.706}\t2025-05-05T10:12:34Z"""


@pytest.fixture
def sample_trades():
    return [
        {
            "execution_id": "0000e0d5.6576fd38.01.01",
            "symbol": "AAPL",
            "supports_tax_opt": "1",
            "side": "S",
            "order_description": "Sold 5 @ 192.26 on ISLAND",
            "trade_time": "20231211-18:00:49",
            "trade_time_r": 1702317649000,
            "size": 5,
            "price": "192.26",
            "order_ref": "Order123",
            "submitter": "user1234",
            "exchange": "ISLAND",
            "commission": "1.01",
            "net_amount": 961.3,
            "account": "U1234567",
            "accountCode": "U1234567",
            "account_allocation_name": "U1234567",
            "company_name": "APPLE INC",
            "contract_description_1": "AAPL",
            "sec_type": "STK",
            "listing_exchange": "NASDAQ.NMS",
            "conid": 265598,
            "conidEx": "265598",
            "clearing_id": "IB",
            "clearing_name": "IB",
            "liquidation_trade": "0",
            "is_event_trading": "0"
        }
    ]


def test_get_alerts_data_with_data(sample_alerts):
    # Mock the load_data_from_json_files function to return a DataFrame with sample alerts
    with patch('app.analysis.data_fetching.load_data_from_json_files') as mock_load:
        # Create a DataFrame from sample_alerts
        df = pd.DataFrame(sample_alerts)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%y-%m-%d %H:%M:%S')
        df.set_index('timestamp', inplace=True)
        mock_load.return_value = df

        # Call the function
        result = get_alerts_data()

        # Verify the result
        assert not result.empty
        assert len(result) == 2
        assert 'symbol' in result.columns
        assert 'side' in result.columns
        assert 'price' in result.columns
        assert result.iloc[0]['symbol'] == 'ZW1!'
        assert result.iloc[0]['side'] == 'B'
        assert result.iloc[0]['price'] == '34.20'
        assert result.iloc[0]['dummy'] == 'YES'
        assert result.iloc[1]['symbol'] == 'ZC1!'
        assert result.iloc[1]['side'] == 'S'
        assert result.iloc[1]['price'] == '32.20'


def test_get_alerts_data_no_data():
    # Mock the load_data_from_json_files function to return an empty DataFrame
    with patch('app.analysis.data_fetching.load_data_from_json_files') as mock_load:
        mock_load.return_value = pd.DataFrame()

        # Call the function
        result = get_alerts_data()

        # Verify the result
        assert result.empty
        assert list(result.columns) == ['timestamp', 'symbol', 'side', 'price']


def test_get_tw_alerts_data_success(sample_tw_alerts_csv):
    # Create a DataFrame from the sample CSV outside the patch context
    expected_df = pd.read_csv(StringIO(sample_tw_alerts_csv), sep='\t')

    # Mock os.listdir to return a list of files
    with patch('os.listdir') as mock_listdir, \
            patch('os.path.exists') as mock_exists, \
            patch('pandas.read_csv') as mock_read_csv:
        mock_listdir.return_value = ['TradingView_Alerts_Log_2025-05-05.csv', 'TradingView_Alerts_Log_2025-05-04.csv']
        mock_exists.return_value = True

        # Set the return value of mock_read_csv
        mock_read_csv.return_value = expected_df

        # Call the function
        result = get_tw_alerts_data()

        # Verify that mock_read_csv was called with the correct file path
        mock_read_csv.assert_called_once_with(os.path.join(TW_ALERTS_DIR, 'TradingView_Alerts_Log_2025-05-05.csv'))

        # Verify that the result is the expected DataFrame
        assert result is expected_df

        # Verify the content of the DataFrame
        assert not expected_df.empty
        assert len(expected_df) == 5
        assert 'Alert ID' in expected_df.columns
        assert 'Ticker' in expected_df.columns
        assert 'Description' in expected_df.columns
        assert 'Time' in expected_df.columns
        assert expected_df.iloc[0]['Description'] == '{"symbol":"MCL1!","side":"S","price":56.98}'
        assert expected_df.iloc[0]['Time'] == '2025-05-05T14:07:00Z'


def test_get_tw_alerts_data_no_files():
    # Mock os.listdir to return an empty list
    with patch('os.listdir') as mock_listdir:
        mock_listdir.return_value = []

        # Call the function and expect a FileNotFoundError
        with pytest.raises(FileNotFoundError):
            get_tw_alerts_data()


def test_get_tw_alerts_data_file_not_found():
    # Mock os.listdir to return a list of files but os.path.exists to return False
    with patch('os.listdir') as mock_listdir, \
            patch('os.path.exists') as mock_exists:
        mock_listdir.return_value = ['TradingView_Alerts_Log_2025-05-05.csv']
        mock_exists.return_value = False

        # Call the function and expect a FileNotFoundError
        with pytest.raises(FileNotFoundError):
            get_tw_alerts_data()


def test_get_tw_alerts_data_read_error(sample_tw_alerts_csv):
    # Mock os.listdir to return a list of files
    with patch('os.listdir') as mock_listdir, \
            patch('os.path.exists') as mock_exists, \
            patch('pandas.read_csv') as mock_read_csv:
        mock_listdir.return_value = ['TradingView_Alerts_Log_2025-05-05.csv']
        mock_exists.return_value = True
        mock_read_csv.side_effect = Exception("Error reading CSV")

        # Call the function and expect a ValueError
        with pytest.raises(ValueError):
            get_tw_alerts_data()


def test_get_trades_data_with_recent_data(sample_trades):
    # Mock the load_data_from_json_files function to return a DataFrame with sample trades
    with patch('app.analysis.data_fetching.load_data_from_json_files') as mock_load:
        # Create a DataFrame from sample_trades
        df = pd.DataFrame(sample_trades)
        df['trade_time'] = pd.to_datetime(df['trade_time'], format='%Y%m%d-%H:%M:%S')

        # Set trade_time to a recent date within TIMEFRAME_TO_ANALYZE
        recent_time = datetime.now() - timedelta(days=1)
        df['trade_time'] = recent_time

        # Create a copy of the DataFrame with trade_time as index
        df_indexed = df.copy()
        df_indexed.set_index('trade_time', inplace=True)

        # Reset the index but keep trade_time as a column
        df_with_column = df_indexed.reset_index()
        mock_load.return_value = df_with_column

        # Call the function
        result = get_trades_data()

        # Verify the result
        assert not result.empty
        assert len(result) == 1
        assert 'symbol' in result.columns
        assert 'side' in result.columns
        assert 'price' in result.columns
        assert result.iloc[0]['symbol'] == 'AAPL'
        assert result.iloc[0]['side'] == 'S'
        assert result.iloc[0]['price'] == '192.26'


def test_get_trades_data_with_old_data(sample_trades):
    # Mock the load_data_from_json_files function to return a DataFrame with sample trades
    # and fetch_trades_data to return success
    with patch('app.analysis.data_fetching.load_data_from_json_files') as mock_load, \
            patch('app.analysis.data_fetching.fetch_trades_data') as mock_fetch, \
            patch('app.analysis.data_fetching.get_trades_data', wraps=get_trades_data) as wrapped_get_trades:
        # First call to load_data_from_json_files returns trades with old dates
        df_old = pd.DataFrame(sample_trades)
        df_old['trade_time'] = pd.to_datetime(df_old['trade_time'], format='%Y%m%d-%H:%M:%S')

        # Set trade_time to an old date outside TIMEFRAME_TO_ANALYZE
        old_time = datetime.now() - timedelta(days=TIMEFRAME_TO_ANALYZE + 1)
        df_old['trade_time'] = old_time

        df_old = df_old.sort_values('trade_time').reset_index(drop=True)

        # Second call to load_data_from_json_files returns trades with recent dates
        df_recent = pd.DataFrame(sample_trades)
        df_recent['trade_time'] = pd.to_datetime(df_recent['trade_time'], format='%Y%m%d-%H:%M:%S')

        # Set trade_time to a recent date within TIMEFRAME_TO_ANALYZE
        recent_time = datetime.now() - timedelta(days=1)
        df_recent['trade_time'] = recent_time

        df_recent = df_recent.sort_values('trade_time').reset_index(drop=True)

        # Set up the mock to return df_old on first call and df_recent on second call
        mock_load.side_effect = [df_old, df_recent]

        # Mock fetch_trades_data to return success
        mock_fetch.return_value = {"success": True, "message": "Trades fetched successfully"}

        # Prevent infinite recursion by limiting the number of calls to get_trades_data
        wrapped_get_trades.side_effect = lambda: df_recent

        # Call the function
        result = get_trades_data()

        # Verify that fetch_trades_data was called
        mock_fetch.assert_called_once()

        # Verify the result
        assert not result.empty
        assert len(result) == 1
        assert result.iloc[0]['symbol'] == 'AAPL'


def test_get_trades_data_no_data_fetch_failure():
    # Mock the load_data_from_json_files function to return an empty DataFrame
    # and fetch_trades_data to return failure
    with patch('app.analysis.data_fetching.load_data_from_json_files') as mock_load, \
            patch('app.analysis.data_fetching.fetch_trades_data') as mock_fetch:
        mock_load.return_value = pd.DataFrame()
        mock_fetch.return_value = {"success": False, "error": "No data returned from IBKR API after multiple retries"}

        # Call the function
        result = get_trades_data()

        # Verify the result
        assert result.empty
        assert list(result.columns) == ['conid', 'side', 'price', 'trade_time']


def test_fetch_trades_data_success(sample_trades):
    # Mock api_get to return sample trades and save_trades_data to do nothing
    with patch('app.analysis.data_fetching.api_get') as mock_api_get, \
            patch('app.analysis.data_fetching.save_trades_data') as mock_save:
        mock_api_get.return_value = sample_trades

        # Call the function
        result = fetch_trades_data()

        # Verify the result
        assert result["success"] is True
        assert result["message"] == "Trades fetched successfully"

        # Verify that api_get and save_trades_data were called with the correct arguments
        mock_api_get.assert_called_once_with("iserver/account/trades?days=7")
        mock_save.assert_called_once()


def test_fetch_trades_data_empty_response():
    # Mock api_get to return an empty list
    with patch('app.analysis.data_fetching.api_get') as mock_api_get, \
            patch('app.analysis.data_fetching.time.sleep') as mock_sleep:
        mock_api_get.return_value = []

        # Call the function
        result = fetch_trades_data(max_retries=2, retry_delay=0)

        # Verify the result
        assert result["success"] is False
        assert result["error"] == "No data returned from IBKR API after multiple retries"

        # Verify that api_get was called twice (initial + 1 retry)
        assert mock_api_get.call_count == 2

        # Verify that time.sleep was called once
        mock_sleep.assert_called_once_with(0)


def test_fetch_trades_data_api_exception():
    # Mock api_get to raise an exception
    with patch('app.analysis.data_fetching.api_get') as mock_api_get:
        mock_api_get.side_effect = Exception("API error")

        # Call the function
        result = fetch_trades_data()

        # Verify the result
        assert result["success"] is False
        assert "Unexpected error: API error" in result["error"]

        # Verify that api_get was called once
        mock_api_get.assert_called_once()
