import json
import os
from unittest.mock import patch, mock_open, MagicMock, call
from zoneinfo import ZoneInfo

import pytest

from app.analysis.analysis_utils.data_fetching_utils import save_trades_data


@pytest.fixture
def sample_trades():
    return [
        {
            "execution_id": "1",
            "trade_time": "20250401-10:30:00",
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 100,
            "price": 150.0
        },
        {
            "execution_id": "2",
            "trade_time": "20250401-14:45:00",
            "symbol": "MSFT",
            "side": "SELL",
            "quantity": 50,
            "price": 300.0
        },
        {
            "execution_id": "3",
            "trade_time": "20250402-09:15:00",
            "symbol": "GOOGL",
            "side": "BUY",
            "quantity": 25,
            "price": 2500.0
        }
    ]


def test_save_trades_data_new_files(sample_trades):
    # Test saving trades to new files (no existing files)
    with patch("os.path.exists", return_value=False), \
            patch("builtins.open", mock_open()) as mock_file, \
            patch("json.dump") as mock_json_dump:
        save_trades_data(sample_trades, "test_dir")

        # Check that files were opened for writing
        assert mock_file.call_count == 2  # Two different dates in sample_trades
        mock_file.assert_has_calls([
            call(os.path.join("test_dir", "trades_2025-04-01.json"), "w"),
            call(os.path.join("test_dir", "trades_2025-04-02.json"), "w")
        ], any_order=True)

        # Check that json.dump was called twice (once for each date)
        assert mock_json_dump.call_count == 2


def test_save_trades_data_existing_files(sample_trades):
    # Test saving trades to existing files with some overlapping trades
    existing_data = [
        {
            "execution_id": "1",  # Same as in sample_trades
            "trade_time": "20250401-10:30:00",
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 100,
            "price": 150.0
        },
        {
            "execution_id": "4",  # Different from sample_trades
            "trade_time": "20250401-16:00:00",
            "symbol": "TSLA",
            "side": "BUY",
            "quantity": 10,
            "price": 800.0
        }
    ]

    # Mock file operations
    m = mock_open()
    m.side_effect = [
        mock_open(read_data=json.dumps(existing_data)).return_value,  # First open for reading
        mock_open().return_value,  # Second open for writing
        mock_open(read_data="[]").return_value,  # Third open for reading (empty file)
        mock_open().return_value  # Fourth open for writing
    ]

    with patch("os.path.exists", return_value=True), \
            patch("builtins.open", m), \
            patch("json.load", side_effect=[existing_data, []]), \
            patch("json.dump") as mock_json_dump:
        save_trades_data(sample_trades, "test_dir")

        # Check that json.dump was called twice (once for each date)
        assert mock_json_dump.call_count == 2

        # For the first date (2025-04-01), we should have 3 unique trades
        # (1 from sample_trades that's also in existing_data, 1 unique from sample_trades, 1 from existing_data)
        first_call_args = mock_json_dump.call_args_list[0][0]
        assert len(first_call_args[0]) == 3

        # For the second date (2025-04-02), we should have 1 trade
        second_call_args = mock_json_dump.call_args_list[1][0]
        assert len(second_call_args[0]) == 1


def test_save_trades_data_timezone(sample_trades):
    with patch("os.path.exists", return_value=False), \
            patch("builtins.open", mock_open()), \
            patch("json.dump") as mock_json_dump, \
            patch("app.analysis.analysis_utils.data_fetching_utils.datetime") as mock_datetime:
        # Setup the side effect for strptime
        mock_dt1 = MagicMock()
        mock_dt1.astimezone.return_value.strftime.return_value = "2025-04-01"
        mock_dt2 = MagicMock()
        mock_dt2.astimezone.return_value.strftime.return_value = "2025-04-01"
        mock_dt3 = MagicMock()
        mock_dt3.astimezone.return_value.strftime.return_value = "2025-04-02"
        mock_datetime.strptime.side_effect = [mock_dt1, mock_dt2, mock_dt3]

        from app.analysis.analysis_utils.data_fetching_utils import save_trades_data

        save_trades_data(sample_trades, "test_dir", timezone="America/New_York")

        mock_dt1.astimezone.assert_called_once_with(ZoneInfo("America/New_York"))
        mock_dt2.astimezone.assert_called_once_with(ZoneInfo("America/New_York"))
        mock_dt3.astimezone.assert_called_once_with(ZoneInfo("America/New_York"))
