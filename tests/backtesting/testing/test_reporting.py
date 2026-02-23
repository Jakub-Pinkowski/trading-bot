"""
Tests for reporting module.

Tests cover:
- results_to_dataframe conversion
- DataFrame structure and column validation
- Numeric type validation and coercion
- Missing metrics handling
- Invalid value handling (NaN, inf)
- Empty results handling
- save_results function
- File saving integration
- Error handling
"""
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.backtesting.testing.reporting import results_to_dataframe, save_results, save_to_parquet


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
                    'average_trade_duration_hours': 4.5,
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
                    'average_trade_duration_hours': 4.5,
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
            'total_trades', 'win_rate', 'average_trade_duration_hours',
            'total_wins_percentage_of_contract', 'total_losses_percentage_of_contract',
            'total_return_percentage_of_contract', 'average_trade_return_percentage_of_contract',
            'average_win_percentage_of_contract', 'average_loss_percentage_of_contract',
            'profit_factor', 'maximum_drawdown_percentage',
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'value_at_risk', 'expected_shortfall', 'ulcer_index'
        ]

        for col in expected_columns:
            assert col in df.columns

    def test_results_to_dataframe_multiple_results(self):
        """Test conversion with multiple results."""
        results = [
            {
                'month': '1!', 'symbol': 'ZS', 'interval': '1h', 'strategy': 'RSI_14_30_70',
                'metrics': {'total_trades': 10, 'win_rate': 60.0, 'average_trade_duration_hours': 4.5,
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
                'metrics': {'total_trades': 15, 'win_rate': 55.0, 'average_trade_duration_hours': 3.0,
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
                    'average_trade_duration_hours': 4.0,
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
                    'average_trade_duration_hours': 4.0,
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
                    'average_trade_duration_hours': 4.0,
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
                    'average_trade_duration_hours': 4.0,
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
                    'total_trades': 10, 'win_rate': 60.0, 'average_trade_duration_hours': 4.5,
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
                    'total_trades': 10, 'win_rate': 60.0, 'average_trade_duration_hours': 4.5,
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
                    'total_trades': 10, 'win_rate': 60.0, 'average_trade_duration_hours': 4.5,
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


# ==================== Save Results Tests ====================

class TestSaveResults:
    """Test save_results function."""

    def test_save_results_calls_results_to_dataframe(self, sample_test_results, mock_reporting_dependencies):
        """Test that save_results calls results_to_dataframe."""
        mocks = mock_reporting_dependencies

        save_results(sample_test_results)

        # Verify results_to_dataframe was called
        mocks['results_to_dataframe'].assert_called_once_with(sample_test_results)

    def test_save_results_calls_save_to_parquet(self, sample_test_results, mock_reporting_dependencies):
        """Test that save_results calls save_to_parquet."""
        mocks = mock_reporting_dependencies

        save_results(sample_test_results)

        # Verify save_to_parquet was called with DataFrame
        mocks['save_to_parquet'].assert_called_once()
        assert isinstance(mocks['save_to_parquet'].call_args[0][0], pd.DataFrame)

    def test_save_results_with_empty_dataframe(self, capsys, mock_reporting_dependencies):
        """Test that empty DataFrame doesn't trigger save."""
        results = []
        mocks = mock_reporting_dependencies
        mocks['results_to_dataframe'].return_value = pd.DataFrame()

        save_results(results)

        # save_to_parquet should not be called
        mocks['save_to_parquet'].assert_not_called()

        captured = capsys.readouterr()
        assert 'No results to save' in captured.out

    def test_save_results_error_handling(self, sample_test_results):
        """Test that errors during save are caught and logged."""
        with patch('app.backtesting.testing.reporting.results_to_dataframe') as mock_to_df:
            mock_to_df.side_effect = Exception("Test error")

            # Should not raise, should handle gracefully
            save_results(sample_test_results)

    def test_save_results_uses_correct_filepath(self, sample_test_results, mock_reporting_dependencies):
        """Test that save_results uses correct file path."""
        mocks = mock_reporting_dependencies

        save_results(sample_test_results)

        # Verify correct filepath used
        filepath = mocks['save_to_parquet'].call_args[0][1]
        assert 'mass_test_results_all.parquet' in filepath


# ==================== Integration Tests ====================

class TestReportingIntegration:
    """Test reporting integration scenarios."""

    def test_complete_workflow(self):
        """Test complete workflow from results to saved file."""
        results = [
            {
                'month': '1!', 'symbol': 'ZS', 'interval': '1h', 'strategy': 'RSI_14_30_70',
                'metrics': {'total_trades': 10, 'win_rate': 60.0, 'average_trade_duration_hours': 4.5,
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
                'metrics': {'total_trades': 15, 'win_rate': 55.0, 'average_trade_duration_hours': 3.0,
                            'total_wins_percentage_of_contract': 120.0, 'total_losses_percentage_of_contract': 60.0,
                            'total_return_percentage_of_contract': 60.0,
                            'average_trade_return_percentage_of_contract': 4.0,
                            'average_win_percentage_of_contract': 8.0, 'average_loss_percentage_of_contract': -4.0,
                            'profit_factor': 2.5, 'maximum_drawdown_percentage': 12.0, 'sharpe_ratio': 1.8,
                            'sortino_ratio': 2.2, 'calmar_ratio': 1.5, 'value_at_risk': 8.0,
                            'expected_shortfall': 10.0, 'ulcer_index': 4.0}
            }
        ]

        # Convert to DataFrame
        df = results_to_dataframe(results)

        # Verify DataFrame structure
        assert len(df) == 2
        assert all(col in df.columns for col in ['month', 'symbol', 'interval', 'strategy'])

        # Mock save operation
        with patch('app.backtesting.testing.reporting.save_to_parquet') as mock_save:
            save_results(results)
            assert mock_save.called

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
                'metrics': {'total_trades': i, 'win_rate': 50.0 + i % 50, 'average_trade_duration_hours': 4.0,
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
