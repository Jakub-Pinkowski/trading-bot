import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.analysis.analysis_runner import run_analysis, save_to_csv


@pytest.fixture
def sample_alerts_data():
    """Sample alerts data for testing."""
    return pd.DataFrame([
        {"timestamp": "23-05-01 10:30:45", "symbol": "ZW1!", "side": "B", "price": "34.20"},
        {"timestamp": "23-05-01 11:45:30", "symbol": "ZC1!", "side": "S", "price": "423.20"}
    ])


@pytest.fixture
def sample_tv_alerts_data():
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
            "trade_time": pd.Timestamp("2023-12-11 18:00:49"),
            "symbol": "AAPL",
            "side": "S",
            "price": "192.26",
            "size": 5,
            "commission": "1.01",
            "net_amount": 961.3
        }
    ])


@pytest.fixture
def sample_cleaned_alerts():
    """Sample cleaned ibkr_alerts data for testing."""
    return pd.DataFrame([
        {"trade_time": pd.Timestamp("2023-05-01 10:30:45"), "symbol": "ZW", "side": "B", "price": 34.20}
    ])


@pytest.fixture
def sample_cleaned_tv_alerts():
    """Sample cleaned TradingView alerts data for testing."""
    return pd.DataFrame([
        {"trade_time": pd.Timestamp("2025-05-05 14:07:00"), "symbol": "MCL", "side": "S", "price": 56.98}
    ])


@pytest.fixture
def sample_cleaned_trades():
    """Sample cleaned trades data for testing."""
    return pd.DataFrame([
        {
            "trade_time": pd.Timestamp("2023-12-11 18:00:49"),
            "symbol": "AAPL",
            "side": "S",
            "price": 192.26,
            "size": 5.0,
            "commission": 1.01,
            "net_amount": 961.3
        }
    ])


@pytest.fixture
def sample_matched_trades():
    """Sample matched trades data for testing."""
    return pd.DataFrame([
        {
            "symbol": "AAPL",
            "entry_time": pd.Timestamp("2023-12-11 18:00:49"),
            "entry_side": "B",
            "entry_price": 192.26,
            "entry_net_amount": 961.3,
            "exit_time": pd.Timestamp("2023-12-11 19:15:30"),
            "exit_side": "S",
            "exit_price": 195.50,
            "exit_net_amount": 586.5,
            "size": 3,
            "total_commission": 1.76
        }
    ])


@pytest.fixture
def sample_trades_with_metrics():
    """Sample trades data with metrics for testing."""
    return pd.DataFrame([
        {
            "symbol": "AAPL",
            "entry_time": pd.Timestamp("2023-12-11 18:00:49"),
            "entry_side": "B",
            "entry_price": 192.26,
            "entry_net_amount": 961.3,
            "exit_time": pd.Timestamp("2023-12-11 19:15:30"),
            "exit_side": "S",
            "exit_price": 195.50,
            "exit_net_amount": 586.5,
            "size": 3,
            "total_commission": 1.76,
            "pnl": 15.98,
            "pnl_pct": 0.0166,
            "trade_duration": 74.68
        }
    ])


@pytest.fixture
def sample_dataset_metrics():
    """Sample dataset metrics for testing."""
    return {
        "win_rate": 100.0,
        "average_win": 15.98,
        "average_loss": 0.0,
        "profit_factor": float('nan'),
        "sharpe_ratio": float('nan'),
        "sortino_ratio": float('nan'),
        "cumulative_pnl": 15.98,
        "max_drawdown": 0.0,
        "total_commission": 1.76
    }


@patch('app.analysis.analysis_runner.get_ibkr_alerts_data')
@patch('app.analysis.analysis_runner.get_tv_alerts_data')
@patch('app.analysis.analysis_runner.get_trades_data')
@patch('app.analysis.analysis_runner.clean_ibkr_alerts_data')
@patch('app.analysis.analysis_runner.clean_tv_alerts_data')
@patch('app.analysis.analysis_runner.clean_trades_data')
@patch('app.analysis.analysis_runner.match_trades')
@patch('app.analysis.analysis_runner.add_per_trade_metrics')
@patch('app.analysis.analysis_runner.calculate_dataset_metrics')
@patch('app.analysis.analysis_runner.save_to_csv')
@patch('app.analysis.analysis_runner.is_nonempty')
def test_run_analysis_with_all_data(
    mock_is_nonempty,
    mock_save_to_csv,
    mock_calculate_dataset_metrics,
    mock_add_per_trade_metrics,
    mock_match_trades,
    mock_clean_trades,
    mock_clean_tv_alerts,
    mock_clean_ibkr_alerts,
    mock_get_trades_data,
    mock_get_tv_alerts_data,
    mock_get_ibkr_alerts_data,
    sample_alerts_data,
    sample_tv_alerts_data,
    sample_trades_data,
    sample_cleaned_alerts,
    sample_cleaned_tv_alerts,
    sample_cleaned_trades,
    sample_matched_trades,
    sample_trades_with_metrics,
    sample_dataset_metrics
):
    """Test running analysis with all data available."""

    # Setup mocks
    mock_get_ibkr_alerts_data.return_value = sample_alerts_data
    mock_get_tv_alerts_data.return_value = sample_tv_alerts_data
    mock_get_trades_data.return_value = sample_trades_data

    mock_clean_ibkr_alerts.return_value = sample_cleaned_alerts
    mock_clean_tv_alerts.return_value = sample_cleaned_tv_alerts
    mock_clean_trades.return_value = sample_cleaned_trades

    mock_match_trades.side_effect = [sample_matched_trades, sample_matched_trades, sample_matched_trades]

    mock_add_per_trade_metrics.return_value = sample_trades_with_metrics

    mock_calculate_dataset_metrics.return_value = sample_dataset_metrics

    # Mock is_nonempty to return True for all calls
    mock_is_nonempty.return_value = True

    # Call the function
    run_analysis()

    # Verify the mocks were called
    mock_get_ibkr_alerts_data.assert_called_once()
    mock_get_tv_alerts_data.assert_called_once()
    mock_get_trades_data.assert_called_once()

    # Verify cleaning functions were called
    mock_clean_ibkr_alerts.assert_called_once()
    mock_clean_tv_alerts.assert_called_once()
    mock_clean_trades.assert_called_once()

    # Verify match_trades was called for ibkr_alerts, TradingView alerts, and trades
    assert mock_match_trades.call_count == 3
    mock_match_trades.assert_any_call(sample_cleaned_alerts, is_ibkr_alerts=True)
    # The actual DataFrame passed might be slightly different due to processing, so we check positional/keyword args
    # but mock_match_trades.assert_any_call(sample_cleaned_tv_alerts, is_tv_alerts=True) failed due to DF comparison

    # Let's try to verify by call args instead
    calls = mock_match_trades.call_args_list
    assert any(c.kwargs.get('is_ibkr_alerts') is True for c in calls)
    assert any(c.kwargs.get('is_tv_alerts') is True for c in calls)
    assert any(not c.kwargs.get('is_ibkr_alerts') and not c.kwargs.get('is_tv_alerts') for c in calls)

    # Verify add_per_trade_metrics was called three times
    assert mock_add_per_trade_metrics.call_count == 3

    # Verify calculate_dataset_metrics was called three times
    assert mock_calculate_dataset_metrics.call_count == 3

    # Verify save_to_csv was called 6 times (3 for per-trade metrics, 3 for dataset metrics)
    assert mock_save_to_csv.call_count == 6


@patch('app.analysis.analysis_runner.get_ibkr_alerts_data')
@patch('app.analysis.analysis_runner.get_tv_alerts_data')
@patch('app.analysis.analysis_runner.get_trades_data')
@patch('app.analysis.analysis_runner.clean_ibkr_alerts_data')
@patch('app.analysis.analysis_runner.clean_tv_alerts_data')
@patch('app.analysis.analysis_runner.clean_trades_data')
@patch('app.analysis.analysis_runner.match_trades')
@patch('app.analysis.analysis_runner.add_per_trade_metrics')
@patch('app.analysis.analysis_runner.calculate_dataset_metrics')
@patch('app.analysis.analysis_runner.save_to_csv')
@patch('app.analysis.analysis_runner.is_nonempty')
def test_run_analysis_with_no_data(
    mock_is_nonempty,
    mock_save_to_csv,
    mock_calculate_dataset_metrics,
    mock_add_per_trade_metrics,
    mock_match_trades,
    mock_clean_trades,
    mock_clean_tv_alerts,
    mock_clean_ibkr_alerts,
    mock_get_trades_data,
    mock_get_tv_alerts_data,
    mock_get_ibkr_alerts_data
):
    """Test running analysis with no data available."""

    # Setup mocks to return empty DataFrames
    mock_get_ibkr_alerts_data.return_value = pd.DataFrame()
    mock_get_tv_alerts_data.return_value = pd.DataFrame()
    mock_get_trades_data.return_value = pd.DataFrame()

    # Mock is_nonempty to return False for all calls
    mock_is_nonempty.return_value = False

    # Call the function
    run_analysis()

    # Verify the mocks were called
    mock_get_ibkr_alerts_data.assert_called_once()
    mock_get_tv_alerts_data.assert_called_once()
    mock_get_trades_data.assert_called_once()

    # Verify other functions were not called
    mock_clean_ibkr_alerts.assert_not_called()
    mock_clean_tv_alerts.assert_not_called()
    mock_clean_trades.assert_not_called()
    mock_match_trades.assert_not_called()
    mock_add_per_trade_metrics.assert_not_called()
    mock_calculate_dataset_metrics.assert_not_called()
    mock_save_to_csv.assert_not_called()


@patch('app.analysis.analysis_runner.get_ibkr_alerts_data')
@patch('app.analysis.analysis_runner.get_tv_alerts_data')
@patch('app.analysis.analysis_runner.get_trades_data')
@patch('app.analysis.analysis_runner.clean_ibkr_alerts_data')
@patch('app.analysis.analysis_runner.clean_tv_alerts_data')
@patch('app.analysis.analysis_runner.clean_trades_data')
@patch('app.analysis.analysis_runner.match_trades')
@patch('app.analysis.analysis_runner.add_per_trade_metrics')
@patch('app.analysis.analysis_runner.calculate_dataset_metrics')
@patch('app.analysis.analysis_runner.save_to_csv')
@patch('app.analysis.analysis_runner.is_nonempty')
def test_run_analysis_with_only_tv_alerts(
    mock_is_nonempty,
    mock_save_to_csv,
    mock_calculate_dataset_metrics,
    mock_add_per_trade_metrics,
    mock_match_trades,
    mock_clean_trades,
    mock_clean_tv_alerts,
    mock_clean_ibkr_alerts,
    mock_get_trades_data,
    mock_get_tv_alerts_data,
    mock_get_ibkr_alerts_data,
    sample_tv_alerts_data,
    sample_cleaned_tv_alerts,
    sample_matched_trades,
    sample_trades_with_metrics,
    sample_dataset_metrics
):
    """Test running analysis with only TradingView alerts data available."""

    # Setup mocks
    mock_get_ibkr_alerts_data.return_value = pd.DataFrame()
    mock_get_tv_alerts_data.return_value = sample_tv_alerts_data
    mock_get_trades_data.return_value = pd.DataFrame()

    mock_clean_tv_alerts.return_value = sample_cleaned_tv_alerts

    mock_match_trades.return_value = sample_matched_trades

    mock_add_per_trade_metrics.return_value = sample_trades_with_metrics

    mock_calculate_dataset_metrics.return_value = sample_dataset_metrics

    # Mock is_nonempty to return True for TradingView alerts and False for others
    mock_is_nonempty.side_effect = lambda x: not x.empty if isinstance(x, pd.DataFrame) else bool(x)

    # Call the function
    run_analysis()

    # Verify the mocks were called
    mock_get_ibkr_alerts_data.assert_called_once()
    mock_get_tv_alerts_data.assert_called_once()
    mock_get_trades_data.assert_called_once()

    # Verify clean_tv_alerts_data was called once for TradingView alerts
    mock_clean_tv_alerts.assert_called_once_with(sample_tv_alerts_data)

    # Verify other cleaning functions were not called
    mock_clean_trades.assert_not_called()
    mock_clean_ibkr_alerts.assert_not_called()

    # Verify match_trades was called once for TradingView alerts
    # mock_match_trades.assert_called_once_with(sample_cleaned_tv_alerts, is_tv_alerts=True)
    assert mock_match_trades.call_count == 1
    assert mock_match_trades.call_args.kwargs.get('is_tv_alerts') is True

    # Verify add_per_trade_metrics was called once
    mock_add_per_trade_metrics.assert_called_once_with(sample_matched_trades)

    # Verify calculate_dataset_metrics was called once
    mock_calculate_dataset_metrics.assert_called_once_with(sample_trades_with_metrics)

    # Verify save_to_csv was called twice (once for per-trade metrics, once for dataset metrics)
    assert mock_save_to_csv.call_count == 2


# ==================== Test Classes ====================

class TestSaveToCsv:
    """Test CSV file saving and appending."""

    def test_new_file_saves_dataframe(self, monkeypatch):
        """Test new file path causes direct save without concatenation."""
        sample_df = pd.DataFrame({"name": ["Item 1", "Item 2"], "value": [100, 200]})
        mock_to_csv = MagicMock()

        monkeypatch.setattr(os.path, "exists", lambda path: False)
        monkeypatch.setattr(pd.DataFrame, "to_csv", mock_to_csv)

        save_to_csv(sample_df, "test_file.csv")

        mock_to_csv.assert_called_once_with("test_file.csv", index=False)

    def test_existing_file_appends_data(self, monkeypatch):
        """Test existing file path reads, concatenates, deduplicates, then saves."""
        sample_df = pd.DataFrame({"name": ["Item 1", "Item 2"], "value": [100, 200]})
        existing_df = pd.DataFrame({"name": ["Item 3"], "value": [300]})
        mock_to_csv = MagicMock()
        mock_concat = MagicMock(return_value=pd.DataFrame())

        monkeypatch.setattr(os.path, "exists", lambda path: True)
        monkeypatch.setattr(pd, "read_csv", MagicMock(return_value=existing_df))
        monkeypatch.setattr(pd.DataFrame, "to_csv", mock_to_csv)
        monkeypatch.setattr(pd, "concat", mock_concat)

        save_to_csv(sample_df, "test_file.csv")

        mock_concat.assert_called_once()
        mock_to_csv.assert_called_once()

    def test_existing_file_read_error_logs_and_saves(self, monkeypatch):
        """Test read error on existing file is logged and save still proceeds."""
        sample_df = pd.DataFrame({"name": ["Item 1"], "value": [100]})
        mock_logger = MagicMock()
        mock_to_csv = MagicMock()

        monkeypatch.setattr(os.path, "exists", lambda path: True)
        monkeypatch.setattr(pd, "read_csv", MagicMock(side_effect=Exception("CSV read error")))
        monkeypatch.setattr("app.analysis.analysis_runner.logger", mock_logger)
        monkeypatch.setattr(pd.DataFrame, "to_csv", mock_to_csv)

        save_to_csv(sample_df, "test_file.csv")

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
