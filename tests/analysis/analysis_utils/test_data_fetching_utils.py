import os
from unittest.mock import MagicMock, mock_open, call
from zoneinfo import ZoneInfo

from app.analysis.analysis_utils.data_fetching_utils import save_trades_data


class TestSaveTradesData:
    """Tests for save_trades_data — grouping, deduplication, and timezone handling."""

    def test_new_files_writes_once_per_date(self, monkeypatch, sample_trades):
        """Test that one file is written per unique trade date when no files exist."""
        mock_json_dump = MagicMock()

        monkeypatch.setattr("app.analysis.analysis_utils.data_fetching_utils.os.path.exists",
                            MagicMock(return_value=False))
        monkeypatch.setattr("builtins.open", mock_open())
        monkeypatch.setattr("app.analysis.analysis_utils.data_fetching_utils.json.dump", mock_json_dump)

        save_trades_data(sample_trades, "test_dir")

        # Two different dates in sample_trades → two writes
        assert mock_json_dump.call_count == 2

    def test_new_files_uses_correct_date_based_paths(self, monkeypatch, sample_trades):
        """Test that output files are named with the correct date extracted from trade_time."""
        mock_open_func = mock_open()

        monkeypatch.setattr("app.analysis.analysis_utils.data_fetching_utils.os.path.exists",
                            MagicMock(return_value=False))
        monkeypatch.setattr("builtins.open", mock_open_func)
        monkeypatch.setattr("app.analysis.analysis_utils.data_fetching_utils.json.dump", MagicMock())

        save_trades_data(sample_trades, "test_dir")

        mock_open_func.assert_has_calls([
            call(os.path.join("test_dir", "trades_2025-04-01.json"), "w"),
            call(os.path.join("test_dir", "trades_2025-04-02.json"), "w")
        ], any_order=True)

    def test_existing_files_deduplicates_by_execution_id(self, monkeypatch, sample_trades):
        """Test that trades from existing files are merged and deduplicated by execution_id."""
        existing_data = [
            {
                "execution_id": "1",  # Same as in sample_trades — should not duplicate
                "trade_time": "20250401-10:30:00",
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 100,
                "price": 150.0
            },
            {
                "execution_id": "4",  # Unique to existing file — should be preserved
                "trade_time": "20250401-16:00:00",
                "symbol": "TSLA",
                "side": "BUY",
                "quantity": 10,
                "price": 800.0
            }
        ]
        mock_json_dump = MagicMock()
        mock_json_load = MagicMock(side_effect=[existing_data, []])

        monkeypatch.setattr("app.analysis.analysis_utils.data_fetching_utils.os.path.exists",
                            MagicMock(return_value=True))
        monkeypatch.setattr("builtins.open", mock_open())
        monkeypatch.setattr("app.analysis.analysis_utils.data_fetching_utils.json.load", mock_json_load)
        monkeypatch.setattr("app.analysis.analysis_utils.data_fetching_utils.json.dump", mock_json_dump)

        save_trades_data(sample_trades, "test_dir")

        assert mock_json_dump.call_count == 2

        # 2025-04-01: ids 1 (overlap), 2 (new) from sample_trades + id 4 (existing) = 3 unique
        first_call_args = mock_json_dump.call_args_list[0][0]
        assert len(first_call_args[0]) == 3

        # 2025-04-02: id 3 from sample_trades, empty existing = 1
        second_call_args = mock_json_dump.call_args_list[1][0]
        assert len(second_call_args[0]) == 1

    def test_timezone_conversion(self, monkeypatch, sample_trades):
        """Test that trade times are converted to the specified timezone before grouping."""
        mock_dt1 = MagicMock()
        mock_dt1.astimezone.return_value.strftime.return_value = "2025-04-01"
        mock_dt2 = MagicMock()
        mock_dt2.astimezone.return_value.strftime.return_value = "2025-04-01"
        mock_dt3 = MagicMock()
        mock_dt3.astimezone.return_value.strftime.return_value = "2025-04-02"

        mock_datetime = MagicMock()
        mock_datetime.strptime.side_effect = [mock_dt1, mock_dt2, mock_dt3]

        monkeypatch.setattr("app.analysis.analysis_utils.data_fetching_utils.os.path.exists",
                            MagicMock(return_value=False))
        monkeypatch.setattr("builtins.open", mock_open())
        monkeypatch.setattr("app.analysis.analysis_utils.data_fetching_utils.json.dump", MagicMock())
        monkeypatch.setattr("app.analysis.analysis_utils.data_fetching_utils.datetime", mock_datetime)

        save_trades_data(sample_trades, "test_dir", timezone="America/New_York")

        mock_dt1.astimezone.assert_called_once_with(ZoneInfo("America/New_York"))
        mock_dt2.astimezone.assert_called_once_with(ZoneInfo("America/New_York"))
        mock_dt3.astimezone.assert_called_once_with(ZoneInfo("America/New_York"))
