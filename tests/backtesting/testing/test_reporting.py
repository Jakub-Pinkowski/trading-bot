"""
Tests for reporting module.

Tests cover:
- results_to_dataframe conversion
- DataFrame structure and column validation
- Numeric type validation and coercion
- Missing metrics handling
- Invalid value handling (NaN, inf)
- Empty results handling
- save_shard function
- merge_shards function
- File saving integration
- Error handling
"""
import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import app.backtesting.testing.reporting as reporting_module
from app.backtesting.testing.reporting import (
    merge_shards,
    results_to_dataframe,
    save_shard,
    save_to_parquet,
)


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({"name": ["Item 1", "Item 2"], "value": [100, 200]})


# ==================== Results to DataFrame Tests ====================

class TestResultsToDataFrame:
    """Test results_to_dataframe conversion."""

    def test_results_to_dataframe_basic(self):
        """Test basic conversion of results to DataFrame."""
        results = [
            {
                'month': '1!',
                'symbol': 'ZS',
                'interval': '1h',
                'strategy': 'RSI_14_30_70',
                'metrics': {
                    'total_trades': 10,
                    'win_rate': 60.0,
                    'average_trade_duration_bars': 4.5,
                    'total_wins_percentage_of_contract': 100.0,
                    'total_losses_percentage_of_contract': 50.0,
                    'total_return_percentage_of_contract': 50.0,
                    'average_trade_return_percentage_of_contract': 5.0,
                    'average_win_percentage_of_contract': 10.0,
                    'average_loss_percentage_of_contract': -5.0,
                    'profit_factor': 2.0,
                    'maximum_drawdown_percentage': 15.0,
                    'sharpe_ratio': 1.5,
                    'sortino_ratio': 2.0,
                    'calmar_ratio': 1.2,
                    'value_at_risk': 10.0,
                    'expected_shortfall': 12.0,
                    'ulcer_index': 5.0
                }
            }
        ]

        df = results_to_dataframe(results)

        assert len(df) == 1
        assert df.loc[0, 'month'] == '1!'
        assert df.loc[0, 'symbol'] == 'ZS'
        assert df.loc[0, 'interval'] == '1h'
        assert df.loc[0, 'strategy'] == 'RSI_14_30_70'
        assert df.loc[0, 'total_trades'] == 10
        assert df.loc[0, 'win_rate'] == 60.0

    def test_results_to_dataframe_all_columns_present(self):
        """Test that all expected columns are present in DataFrame."""
        results = [
            {
                'month': '1!',
                'symbol': 'ZS',
                'interval': '1h',
                'strategy': 'RSI_14_30_70',
                'metrics': {
                    'total_trades': 10,
                    'win_rate': 60.0,
                    'average_trade_duration_bars': 4.5,
                    'total_wins_percentage_of_contract': 100.0,
                    'total_losses_percentage_of_contract': 50.0,
                    'total_return_percentage_of_contract': 50.0,
                    'average_trade_return_percentage_of_contract': 5.0,
                    'average_win_percentage_of_contract': 10.0,
                    'average_loss_percentage_of_contract': -5.0,
                    'profit_factor': 2.0,
                    'maximum_drawdown_percentage': 15.0,
                    'sharpe_ratio': 1.5,
                    'sortino_ratio': 2.0,
                    'calmar_ratio': 1.2,
                    'value_at_risk': 10.0,
                    'expected_shortfall': 12.0,
                    'ulcer_index': 5.0
                }
            }
        ]

        df = results_to_dataframe(results)

        expected_columns = [
            'month', 'symbol', 'interval', 'strategy',
            'total_trades', 'win_rate', 'average_trade_duration_bars',
            'win_loss_ratio', 'max_consecutive_wins', 'max_consecutive_losses',
            'total_wins_percentage_of_contract', 'total_losses_percentage_of_contract',
            'total_return_percentage_of_contract', 'average_trade_return_percentage_of_contract',
            'average_win_percentage_of_contract', 'average_loss_percentage_of_contract',
            'profit_factor', 'expectancy_per_bar',
            'maximum_drawdown_percentage',
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'value_at_risk', 'expected_shortfall', 'ulcer_index', 'time_in_market_percentage'
        ]

        for col in expected_columns:
            assert col in df.columns

    def test_results_to_dataframe_multiple_results(self):
        """Test conversion with multiple results."""
        results = [
            {
                'month': '1!', 'symbol': 'ZS', 'interval': '1h', 'strategy': 'RSI_14_30_70',
                'metrics': {'total_trades': 10, 'win_rate': 60.0, 'average_trade_duration_bars': 4.5,
                            'total_wins_percentage_of_contract': 100.0, 'total_losses_percentage_of_contract': 50.0,
                            'total_return_percentage_of_contract': 50.0,
                            'average_trade_return_percentage_of_contract': 5.0,
                            'average_win_percentage_of_contract': 10.0, 'average_loss_percentage_of_contract': -5.0,
                            'profit_factor': 2.0, 'maximum_drawdown_percentage': 15.0, 'sharpe_ratio': 1.5,
                            'sortino_ratio': 2.0, 'calmar_ratio': 1.2, 'value_at_risk': 10.0,
                            'expected_shortfall': 12.0, 'ulcer_index': 5.0}
            },
            {
                'month': '2!', 'symbol': 'CL', 'interval': '15m', 'strategy': 'EMA_9_21',
                'metrics': {'total_trades': 15, 'win_rate': 55.0, 'average_trade_duration_bars': 3.0,
                            'total_wins_percentage_of_contract': 120.0, 'total_losses_percentage_of_contract': 60.0,
                            'total_return_percentage_of_contract': 60.0,
                            'average_trade_return_percentage_of_contract': 4.0,
                            'average_win_percentage_of_contract': 8.0, 'average_loss_percentage_of_contract': -4.0,
                            'profit_factor': 2.5, 'maximum_drawdown_percentage': 12.0, 'sharpe_ratio': 1.8,
                            'sortino_ratio': 2.2, 'calmar_ratio': 1.5, 'value_at_risk': 8.0,
                            'expected_shortfall': 10.0, 'ulcer_index': 4.0}
            }
        ]

        df = results_to_dataframe(results)

        assert len(df) == 2
        assert df.loc[0, 'symbol'] == 'ZS'
        assert df.loc[1, 'symbol'] == 'CL'

    def test_results_to_dataframe_empty_results(self):
        """Test that empty results return empty DataFrame."""
        results = []

        df = results_to_dataframe(results)

        assert df.empty
        assert isinstance(df, pd.DataFrame)


# ==================== Missing Metrics Handling ====================

class TestMissingMetricsHandling:
    """Test handling of missing metrics."""

    def test_missing_metric_replaced_with_zero(self):
        """Test that missing metrics are replaced with 0."""
        results = [
            {
                'month': '1!',
                'symbol': 'ZS',
                'interval': '1h',
                'strategy': 'RSI_14_30_70',
                'metrics': {
                    'total_trades': 10,
                    # Missing most metrics
                }
            }
        ]

        df = results_to_dataframe(results)

        # Missing metrics should be 0
        assert df.loc[0, 'win_rate'] == 0
        assert df.loc[0, 'profit_factor'] == 0
        assert df.loc[0, 'sharpe_ratio'] == 0

    def test_partially_missing_metrics(self):
        """Test handling of partially complete metrics."""
        results = [
            {
                'month': '1!',
                'symbol': 'ZS',
                'interval': '1h',
                'strategy': 'RSI_14_30_70',
                'metrics': {
                    'total_trades': 10,
                    'win_rate': 60.0,
                    # Missing other metrics
                }
            }
        ]

        df = results_to_dataframe(results)

        # Present metrics should have correct values
        assert df.loc[0, 'total_trades'] == 10
        assert df.loc[0, 'win_rate'] == 60.0
        # Missing metrics should be 0
        assert df.loc[0, 'profit_factor'] == 0

    def test_missing_critical_metrics_logged(self):
        """Test that missing critical metrics are logged."""
        results = [
            {
                'month': '1!',
                'symbol': 'ZS',
                'interval': '1h',
                'strategy': 'RSI_14_30_70',
                'metrics': {}  # All metrics missing
            }
        ]

        # Should not raise, but log warnings
        df = results_to_dataframe(results)

        assert len(df) == 1
        assert df.loc[0, 'total_trades'] == 0


# ==================== Invalid Value Handling ====================

class TestInvalidValueHandling:
    """Test handling of invalid values (NaN, inf, wrong types)."""

    def test_nan_value_replaced_with_zero(self):
        """Test that NaN values are replaced with 0."""
        results = [
            {
                'month': '1!',
                'symbol': 'ZS',
                'interval': '1h',
                'strategy': 'RSI_14_30_70',
                'metrics': {
                    'total_trades': 10,
                    'win_rate': np.nan,  # NaN value
                    'profit_factor': 2.0,
                    'sharpe_ratio': 1.5,
                    'average_trade_duration_bars': 4.0,
                    'total_wins_percentage_of_contract': 100.0,
                    'total_losses_percentage_of_contract': 50.0,
                    'total_return_percentage_of_contract': 50.0,
                    'average_trade_return_percentage_of_contract': 5.0,
                    'average_win_percentage_of_contract': 10.0,
                    'average_loss_percentage_of_contract': -5.0,
                    'maximum_drawdown_percentage': 15.0,
                    'sortino_ratio': 2.0,
                    'calmar_ratio': 1.2,
                    'value_at_risk': 10.0,
                    'expected_shortfall': 12.0,
                    'ulcer_index': 5.0
                }
            }
        ]

        df = results_to_dataframe(results)

        assert df.loc[0, 'win_rate'] == 0

    def test_inf_value_replaced_with_zero(self):
        """Test that infinity values are replaced with 0."""
        results = [
            {
                'month': '1!',
                'symbol': 'ZS',
                'interval': '1h',
                'strategy': 'RSI_14_30_70',
                'metrics': {
                    'total_trades': 10,
                    'win_rate': 60.0,
                    'profit_factor': np.inf,  # Infinity
                    'sharpe_ratio': 1.5,
                    'average_trade_duration_bars': 4.0,
                    'total_wins_percentage_of_contract': 100.0,
                    'total_losses_percentage_of_contract': 50.0,
                    'total_return_percentage_of_contract': 50.0,
                    'average_trade_return_percentage_of_contract': 5.0,
                    'average_win_percentage_of_contract': 10.0,
                    'average_loss_percentage_of_contract': -5.0,
                    'maximum_drawdown_percentage': 15.0,
                    'sortino_ratio': 2.0,
                    'calmar_ratio': 1.2,
                    'value_at_risk': 10.0,
                    'expected_shortfall': 12.0,
                    'ulcer_index': 5.0
                }
            }
        ]

        df = results_to_dataframe(results)

        assert df.loc[0, 'profit_factor'] == 0

    def test_non_numeric_type_replaced_with_zero(self):
        """Test that non-numeric types are replaced with 0."""
        results = [
            {
                'month': '1!',
                'symbol': 'ZS',
                'interval': '1h',
                'strategy': 'RSI_14_30_70',
                'metrics': {
                    'total_trades': 10,
                    'win_rate': "invalid",  # String instead of number
                    'profit_factor': 2.0,
                    'sharpe_ratio': 1.5,
                    'average_trade_duration_bars': 4.0,
                    'total_wins_percentage_of_contract': 100.0,
                    'total_losses_percentage_of_contract': 50.0,
                    'total_return_percentage_of_contract': 50.0,
                    'average_trade_return_percentage_of_contract': 5.0,
                    'average_win_percentage_of_contract': 10.0,
                    'average_loss_percentage_of_contract': -5.0,
                    'maximum_drawdown_percentage': 15.0,
                    'sortino_ratio': 2.0,
                    'calmar_ratio': 1.2,
                    'value_at_risk': 10.0,
                    'expected_shortfall': 12.0,
                    'ulcer_index': 5.0
                }
            }
        ]

        df = results_to_dataframe(results)

        assert df.loc[0, 'win_rate'] == 0

    def test_multiple_invalid_values(self):
        """Test handling of multiple invalid values in one result."""
        results = [
            {
                'month': '1!',
                'symbol': 'ZS',
                'interval': '1h',
                'strategy': 'RSI_14_30_70',
                'metrics': {
                    'total_trades': 10,
                    'win_rate': np.nan,
                    'profit_factor': np.inf,
                    'sharpe_ratio': "invalid",
                    'sortino_ratio': 2.0,
                    'average_trade_duration_bars': 4.0,
                    'total_wins_percentage_of_contract': 100.0,
                    'total_losses_percentage_of_contract': 50.0,
                    'total_return_percentage_of_contract': 50.0,
                    'average_trade_return_percentage_of_contract': 5.0,
                    'average_win_percentage_of_contract': 10.0,
                    'average_loss_percentage_of_contract': -5.0,
                    'maximum_drawdown_percentage': 15.0,
                    'calmar_ratio': 1.2,
                    'value_at_risk': 10.0,
                    'expected_shortfall': 12.0,
                    'ulcer_index': 5.0
                }
            }
        ]

        df = results_to_dataframe(results)

        # All invalid values should be 0
        assert df.loc[0, 'win_rate'] == 0
        assert df.loc[0, 'profit_factor'] == 0
        assert df.loc[0, 'sharpe_ratio'] == 0
        # Valid value should remain
        assert df.loc[0, 'sortino_ratio'] == 2.0


# ==================== DataFrame Structure Tests ====================

class TestDataFrameStructure:
    """Test DataFrame structure and types."""

    def test_dataframe_column_order(self):
        """Test that columns are in expected order."""
        results = [
            {
                'month': '1!', 'symbol': 'ZS', 'interval': '1h', 'strategy': 'RSI_14_30_70',
                'metrics': {
                    'total_trades': 10, 'win_rate': 60.0, 'average_trade_duration_bars': 4.5,
                    'total_wins_percentage_of_contract': 100.0, 'total_losses_percentage_of_contract': 50.0,
                    'total_return_percentage_of_contract': 50.0, 'average_trade_return_percentage_of_contract': 5.0,
                    'average_win_percentage_of_contract': 10.0, 'average_loss_percentage_of_contract': -5.0,
                    'profit_factor': 2.0, 'maximum_drawdown_percentage': 15.0, 'sharpe_ratio': 1.5,
                    'sortino_ratio': 2.0, 'calmar_ratio': 1.2, 'value_at_risk': 10.0,
                    'expected_shortfall': 12.0, 'ulcer_index': 5.0
                }
            }
        ]

        df = results_to_dataframe(results)

        # First 4 columns should be identifiers
        assert df.columns[0] == 'month'
        assert df.columns[1] == 'symbol'
        assert df.columns[2] == 'interval'
        assert df.columns[3] == 'strategy'

    def test_numeric_columns_are_float(self):
        """Test that numeric columns have correct dtype."""
        results = [
            {
                'month': '1!', 'symbol': 'ZS', 'interval': '1h', 'strategy': 'RSI_14_30_70',
                'metrics': {
                    'total_trades': 10, 'win_rate': 60.0, 'average_trade_duration_bars': 4.5,
                    'total_wins_percentage_of_contract': 100.0, 'total_losses_percentage_of_contract': 50.0,
                    'total_return_percentage_of_contract': 50.0, 'average_trade_return_percentage_of_contract': 5.0,
                    'average_win_percentage_of_contract': 10.0, 'average_loss_percentage_of_contract': -5.0,
                    'profit_factor': 2.0, 'maximum_drawdown_percentage': 15.0, 'sharpe_ratio': 1.5,
                    'sortino_ratio': 2.0, 'calmar_ratio': 1.2, 'value_at_risk': 10.0,
                    'expected_shortfall': 12.0, 'ulcer_index': 5.0
                }
            }
        ]

        df = results_to_dataframe(results)

        # Numeric columns should be numeric dtype
        numeric_cols = ['total_trades', 'win_rate', 'profit_factor', 'sharpe_ratio']
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(df[col])

    def test_string_columns_are_object(self):
        """Test that string columns have object or string dtype."""
        results = [
            {
                'month': '1!', 'symbol': 'ZS', 'interval': '1h', 'strategy': 'RSI_14_30_70',
                'metrics': {
                    'total_trades': 10, 'win_rate': 60.0, 'average_trade_duration_bars': 4.5,
                    'total_wins_percentage_of_contract': 100.0, 'total_losses_percentage_of_contract': 50.0,
                    'total_return_percentage_of_contract': 50.0, 'average_trade_return_percentage_of_contract': 5.0,
                    'average_win_percentage_of_contract': 10.0, 'average_loss_percentage_of_contract': -5.0,
                    'profit_factor': 2.0, 'maximum_drawdown_percentage': 15.0, 'sharpe_ratio': 1.5,
                    'sortino_ratio': 2.0, 'calmar_ratio': 1.2, 'value_at_risk': 10.0,
                    'expected_shortfall': 12.0, 'ulcer_index': 5.0
                }
            }
        ]

        df = results_to_dataframe(results)

        # String columns should be object or string dtype (pandas 2.0+)
        for col in ['month', 'symbol', 'interval', 'strategy']:
            assert pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object


# ==================== Integration Tests ====================

class TestReportingIntegration:
    """Test reporting integration scenarios."""

    def test_complete_workflow(self):
        """Test complete workflow from results to DataFrame."""
        results = [
            {
                'month': '1!', 'symbol': 'ZS', 'interval': '1h', 'strategy': 'RSI_14_30_70',
                'metrics': {'total_trades': 10, 'win_rate': 60.0, 'average_trade_duration_bars': 4.5,
                            'total_wins_percentage_of_contract': 100.0, 'total_losses_percentage_of_contract': 50.0,
                            'total_return_percentage_of_contract': 50.0,
                            'average_trade_return_percentage_of_contract': 5.0,
                            'average_win_percentage_of_contract': 10.0, 'average_loss_percentage_of_contract': -5.0,
                            'profit_factor': 2.0, 'maximum_drawdown_percentage': 15.0, 'sharpe_ratio': 1.5,
                            'sortino_ratio': 2.0, 'calmar_ratio': 1.2, 'value_at_risk': 10.0,
                            'expected_shortfall': 12.0, 'ulcer_index': 5.0}
            },
            {
                'month': '2!', 'symbol': 'CL', 'interval': '15m', 'strategy': 'EMA_9_21',
                'metrics': {'total_trades': 15, 'win_rate': 55.0, 'average_trade_duration_bars': 3.0,
                            'total_wins_percentage_of_contract': 120.0, 'total_losses_percentage_of_contract': 60.0,
                            'total_return_percentage_of_contract': 60.0,
                            'average_trade_return_percentage_of_contract': 4.0,
                            'average_win_percentage_of_contract': 8.0, 'average_loss_percentage_of_contract': -4.0,
                            'profit_factor': 2.5, 'maximum_drawdown_percentage': 12.0, 'sharpe_ratio': 1.8,
                            'sortino_ratio': 2.2, 'calmar_ratio': 1.5, 'value_at_risk': 8.0,
                            'expected_shortfall': 10.0, 'ulcer_index': 4.0}
            }
        ]

        df = results_to_dataframe(results)

        assert len(df) == 2
        assert all(col in df.columns for col in ['month', 'symbol', 'interval', 'strategy'])

    def test_large_results_batch(self):
        """Test handling of large batch of results."""
        # Create 100 results
        results = []
        for i in range(100):
            results.append({
                'month': f'{i % 3 + 1}!',
                'symbol': ['ZS', 'CL', 'GC'][i % 3],
                'interval': ['15m', '1h', '4h'][i % 3],
                'strategy': f'Strategy_{i}',
                'metrics': {'total_trades': i, 'win_rate': 50.0 + i % 50, 'average_trade_duration_bars': 4.0,
                            'total_wins_percentage_of_contract': 100.0, 'total_losses_percentage_of_contract': 50.0,
                            'total_return_percentage_of_contract': 50.0,
                            'average_trade_return_percentage_of_contract': 5.0,
                            'average_win_percentage_of_contract': 10.0, 'average_loss_percentage_of_contract': -5.0,
                            'profit_factor': 2.0, 'maximum_drawdown_percentage': 15.0, 'sharpe_ratio': 1.5,
                            'sortino_ratio': 2.0, 'calmar_ratio': 1.2, 'value_at_risk': 10.0,
                            'expected_shortfall': 12.0, 'ulcer_index': 5.0}
            })

        df = results_to_dataframe(results)

        assert len(df) == 100
        assert df['total_trades'].tolist() == list(range(100))


# ==================== Save to Parquet Tests ====================

class TestSaveToParquet:
    """Test save_to_parquet function."""

    def test_new_file_saves_dataframe(self, monkeypatch, sample_dataframe):
        """Test new file path creates directory and writes directly."""
        mock_lock = MagicMock()
        mock_makedirs = MagicMock()
        mock_to_parquet = MagicMock()

        monkeypatch.setattr(os, "makedirs", mock_makedirs)
        monkeypatch.setattr(os.path, "exists", lambda path: False)
        monkeypatch.setattr("app.backtesting.testing.reporting.FileLock", MagicMock(return_value=mock_lock))
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
        monkeypatch.setattr("app.backtesting.testing.reporting.FileLock", MagicMock(return_value=mock_lock))
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
        monkeypatch.setattr("app.backtesting.testing.reporting.FileLock", MagicMock(return_value=mock_lock))
        monkeypatch.setattr(pd, "read_parquet", MagicMock(side_effect=Exception("Parquet read error")))
        monkeypatch.setattr("app.backtesting.testing.reporting.logger", mock_logger)
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
        monkeypatch.setattr("app.backtesting.testing.reporting.FileLock", MagicMock(return_value=mock_lock))
        monkeypatch.setattr("app.backtesting.testing.reporting.logger", mock_logger)

        with pytest.raises(FileLockTimeout):
            save_to_parquet(sample_dataframe, "test_dir/test_file.parquet")

        mock_logger.error.assert_called_once()

    def test_general_exception_logs_and_reraises(self, monkeypatch, sample_dataframe):
        """Test unexpected write error is logged and re-raised."""
        mock_lock = MagicMock()
        mock_logger = MagicMock()

        monkeypatch.setattr(os, "makedirs", MagicMock())
        monkeypatch.setattr(os.path, "exists", lambda path: False)
        monkeypatch.setattr("app.backtesting.testing.reporting.FileLock", MagicMock(return_value=mock_lock))
        monkeypatch.setattr(pd.DataFrame, "to_parquet", MagicMock(side_effect=Exception("Write error")))
        monkeypatch.setattr("app.backtesting.testing.reporting.logger", mock_logger)

        with pytest.raises(Exception, match="Write error"):
            save_to_parquet(sample_dataframe, "test_dir/test_file.parquet")

        assert mock_logger.error.call_count >= 1

    def test_uses_absolute_path_for_lock(self, monkeypatch, sample_dataframe):
        """Test FileLock is called with the absolute path and .lock suffix."""
        mock_lock = MagicMock()
        mock_filelock = MagicMock(return_value=mock_lock)

        monkeypatch.setattr(os, "makedirs", MagicMock())
        monkeypatch.setattr(os.path, "exists", lambda path: False)
        monkeypatch.setattr(os.path, "abspath", MagicMock(return_value="/absolute/path/test_file.parquet"))
        monkeypatch.setattr("app.backtesting.testing.reporting.FileLock", mock_filelock)
        monkeypatch.setattr(pd.DataFrame, "to_parquet", MagicMock())

        save_to_parquet(sample_dataframe, "relative/path/test_file.parquet")

        mock_filelock.assert_called_once_with("/absolute/path/test_file.parquet.lock", timeout=120)


# ==================== Save Shard Tests ====================

class TestSaveShard:
    """Test save_shard function."""

    def test_writes_shard_file_without_reading_existing(self, monkeypatch, sample_test_results):
        """Test that save_shard writes directly without reading any existing file."""
        mock_makedirs = MagicMock()
        mock_to_parquet = MagicMock()
        mock_read_parquet = MagicMock()

        monkeypatch.setattr(os, "makedirs", mock_makedirs)
        monkeypatch.setattr(pd.DataFrame, "to_parquet", mock_to_parquet)
        monkeypatch.setattr(pd, "read_parquet", mock_read_parquet)

        save_shard(sample_test_results, shard_index=0)

        mock_to_parquet.assert_called_once()
        mock_read_parquet.assert_not_called()

    def test_shard_filename_uses_zero_padded_index(self, monkeypatch, sample_test_results):
        """Test shard file is named with 4-digit zero-padded index."""
        mock_to_parquet = MagicMock()
        monkeypatch.setattr(os, "makedirs", MagicMock())
        monkeypatch.setattr(pd.DataFrame, "to_parquet", mock_to_parquet)

        result = save_shard(sample_test_results, shard_index=3)

        assert result is not None
        assert 'shard_0003.parquet' in str(result)

    def test_returns_none_for_empty_results(self, monkeypatch):
        """Test None returned when results produce an empty DataFrame."""
        monkeypatch.setattr(os, "makedirs", MagicMock())
        mock_to_parquet = MagicMock()
        monkeypatch.setattr(pd.DataFrame, "to_parquet", mock_to_parquet)

        result = save_shard([], shard_index=0)

        assert result is None
        mock_to_parquet.assert_not_called()

    def test_creates_shards_directory(self, monkeypatch, sample_test_results):
        """Test that the shards directory is created if it does not exist."""
        mock_makedirs = MagicMock()
        monkeypatch.setattr(os, "makedirs", mock_makedirs)
        monkeypatch.setattr(pd.DataFrame, "to_parquet", MagicMock())

        save_shard(sample_test_results, shard_index=0)

        mock_makedirs.assert_called_once()
        assert mock_makedirs.call_args[1].get('exist_ok') is True


# ==================== Merge Shards Tests ====================

class TestMergeShards:
    """Test merge_shards function."""

    def test_merges_all_shards_into_final_file(self, monkeypatch, tmp_path):
        """Test that all shard DataFrames are concatenated and written to the final file."""
        shard1 = pd.DataFrame({'month': ['1!'], 'symbol': ['ZS']})
        shard2 = pd.DataFrame({'month': ['2!'], 'symbol': ['CL']})

        mock_read = MagicMock(side_effect=[shard1, shard2])
        mock_to_parquet = MagicMock()
        mock_remove = MagicMock()

        monkeypatch.setattr(pd, "read_parquet", mock_read)
        monkeypatch.setattr(pd.DataFrame, "to_parquet", mock_to_parquet)
        monkeypatch.setattr(os, "remove", mock_remove)
        monkeypatch.setattr(os.path, "exists", lambda path: False)

        shard_paths = [tmp_path / 'shard_0000.parquet', tmp_path / 'shard_0001.parquet']
        merge_shards(shard_paths)

        assert mock_read.call_count == 2
        mock_to_parquet.assert_called_once()

    def test_merges_with_existing_final_file(self, monkeypatch, tmp_path):
        """Test that existing final parquet is merged with shards before writing."""
        existing = pd.DataFrame({'month': ['1!'], 'symbol': ['ZS']})
        shard = pd.DataFrame({'month': ['2!'], 'symbol': ['CL']})

        written = []

        read_count = {'n': 0}

        def mock_read(path):
            read_count['n'] += 1
            return shard if read_count['n'] == 1 else existing

        def capture_write(self_df, path, **kwargs):
            written.append(self_df.copy())

        monkeypatch.setattr(pd, "read_parquet", mock_read)
        monkeypatch.setattr(pd.DataFrame, "to_parquet", capture_write)
        monkeypatch.setattr(os, "remove", MagicMock())
        monkeypatch.setattr(os.path, "exists", lambda path: True)

        merge_shards([tmp_path / 'shard_0000.parquet'])

        assert len(written) == 1
        assert len(written[0]) == 2

    def test_deduplicates_on_merge(self, monkeypatch, tmp_path):
        """Test that duplicate rows are removed in the final output."""
        row = {'month': '1!', 'symbol': 'ZS'}
        shard1 = pd.DataFrame([row])
        shard2 = pd.DataFrame([row])  # identical duplicate

        written = []

        def capture_write(self_df, path, **kwargs):
            written.append(self_df.copy())

        monkeypatch.setattr(pd, "read_parquet", MagicMock(side_effect=[shard1, shard2]))
        monkeypatch.setattr(pd.DataFrame, "to_parquet", capture_write)
        monkeypatch.setattr(os, "remove", MagicMock())
        monkeypatch.setattr(os.path, "exists", lambda path: False)

        merge_shards([tmp_path / 'shard_0000.parquet', tmp_path / 'shard_0001.parquet'])

        assert len(written) == 1
        assert len(written[0]) == 1

    def test_removes_shard_files_after_merge(self, monkeypatch, tmp_path):
        """Test that shard files are deleted after a successful merge."""
        shard = pd.DataFrame({'month': ['1!'], 'symbol': ['ZS']})
        mock_remove = MagicMock()

        monkeypatch.setattr(pd, "read_parquet", MagicMock(return_value=shard))
        monkeypatch.setattr(pd.DataFrame, "to_parquet", MagicMock())
        monkeypatch.setattr(os, "remove", mock_remove)
        monkeypatch.setattr(os.path, "exists", lambda path: False)

        shard_paths = [tmp_path / 'shard_0000.parquet', tmp_path / 'shard_0001.parquet']
        merge_shards(shard_paths)

        assert mock_remove.call_count == 2

    def test_no_op_when_shard_paths_empty(self, monkeypatch):
        """Test that nothing is written when shard_paths is empty."""
        mock_to_parquet = MagicMock()
        monkeypatch.setattr(pd.DataFrame, "to_parquet", mock_to_parquet)

        merge_shards([])

        mock_to_parquet.assert_not_called()

    def test_skips_unreadable_shards_and_merges_rest(self, monkeypatch, tmp_path):
        """Test that a corrupt shard is excluded from merge and only readable shards are removed."""
        good_shard = pd.DataFrame({'month': ['2!'], 'symbol': ['CL']})
        bad_path = tmp_path / 'bad.parquet'
        good_path = tmp_path / 'good.parquet'
        written = []
        removed = []

        read_count = {'n': 0}

        def mock_read(path):
            read_count['n'] += 1
            if read_count['n'] == 1:
                raise Exception('Corrupt shard')
            return good_shard

        def capture_write(self_df, path, **kwargs):
            written.append(self_df.copy())

        def capture_remove(path):
            removed.append(path)

        monkeypatch.setattr(pd, 'read_parquet', mock_read)
        monkeypatch.setattr(pd.DataFrame, 'to_parquet', capture_write)
        monkeypatch.setattr(os, 'remove', capture_remove)
        monkeypatch.setattr(os.path, 'exists', lambda path: False)

        merge_shards([bad_path, good_path])

        # Good shard written, corrupt shard excluded
        assert len(written) == 1
        assert len(written[0]) == 1
        # Only the readable shard is removed; corrupt shard left on disk
        assert good_path in removed
        assert bad_path not in removed

    def test_logs_warning_for_corrupt_shard_left_on_disk(self, monkeypatch, tmp_path):
        """Test that corrupt shards not removed are explicitly logged as a warning."""
        good_shard = pd.DataFrame({'month': ['1!'], 'symbol': ['ZS']})
        bad_path = tmp_path / 'bad.parquet'
        good_path = tmp_path / 'good.parquet'
        mock_logger = MagicMock()

        read_count = {'n': 0}

        def mock_read(path):
            read_count['n'] += 1
            if read_count['n'] == 1:
                raise Exception('Corrupt shard')
            return good_shard

        monkeypatch.setattr(pd, 'read_parquet', mock_read)
        monkeypatch.setattr(pd.DataFrame, 'to_parquet', MagicMock())
        monkeypatch.setattr(os, 'remove', MagicMock())
        monkeypatch.setattr(os.path, 'exists', lambda path: False)
        monkeypatch.setattr(reporting_module, 'logger', mock_logger)

        merge_shards([bad_path, good_path])

        warning_messages = [str(call) for call in mock_logger.warning.call_args_list]
        assert any(str(bad_path) in msg for msg in warning_messages)

    def test_logs_error_and_returns_when_all_shards_unreadable(self, monkeypatch, tmp_path):
        """Test that merge returns early and logs when all shards fail to read."""
        mock_logger = MagicMock()
        mock_to_parquet = MagicMock()

        monkeypatch.setattr(pd, 'read_parquet', MagicMock(side_effect=Exception('Corrupt')))
        monkeypatch.setattr(pd.DataFrame, 'to_parquet', mock_to_parquet)
        monkeypatch.setattr(reporting_module, 'logger', mock_logger)

        merge_shards([tmp_path / 'bad1.parquet', tmp_path / 'bad2.parquet'])

        mock_logger.error.assert_called()
        mock_to_parquet.assert_not_called()

    def test_logs_error_when_existing_final_file_unreadable(self, monkeypatch, tmp_path):
        """Test that a read failure on the existing final file is logged and merge continues."""
        shard = pd.DataFrame({'month': ['1!'], 'symbol': ['ZS']})
        written = []

        read_count = {'n': 0}

        def mock_read(path):
            read_count['n'] += 1
            if read_count['n'] == 1:
                return shard  # shard reads fine
            raise Exception('Existing file corrupt')  # final file read fails

        def capture_write(self_df, path, **kwargs):
            written.append(self_df.copy())

        mock_logger = MagicMock()
        monkeypatch.setattr(pd, 'read_parquet', mock_read)
        monkeypatch.setattr(pd.DataFrame, 'to_parquet', capture_write)
        monkeypatch.setattr(os, 'remove', MagicMock())
        monkeypatch.setattr(os.path, 'exists', lambda path: True)
        monkeypatch.setattr(reporting_module, 'logger', mock_logger)

        merge_shards([tmp_path / 'shard_0000.parquet'])

        # Error logged for the unreadable final file
        mock_logger.error.assert_called()
        # Merge still writes with just the shard data
        assert len(written) == 1

    def test_logs_warning_when_shard_removal_fails(self, monkeypatch, tmp_path):
        """Test that a failure to remove a shard file after merge is logged as a warning."""
        shard = pd.DataFrame({'month': ['1!'], 'symbol': ['ZS']})
        mock_logger = MagicMock()

        monkeypatch.setattr(pd, 'read_parquet', MagicMock(return_value=shard))
        monkeypatch.setattr(pd.DataFrame, 'to_parquet', MagicMock())
        monkeypatch.setattr(os, 'remove', MagicMock(side_effect=OSError('Permission denied')))
        monkeypatch.setattr(os.path, 'exists', lambda path: False)
        monkeypatch.setattr(reporting_module, 'logger', mock_logger)

        merge_shards([tmp_path / 'shard_0000.parquet'])

        mock_logger.warning.assert_called()
