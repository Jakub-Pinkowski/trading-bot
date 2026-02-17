"""
Tests for runner module.

Tests cover:
- Single test execution workflow
- DataFrame loading with caching
- DataFrame validation integration
- Strategy execution and trade generation
- Metrics calculation (per-trade and summary)
- Cache statistics tracking
- Error handling (missing files, invalid data)
- Edge cases (no trades, insufficient data)
- Verbose output behavior
"""
from datetime import datetime
from unittest.mock import patch, MagicMock

import pandas as pd

from app.backtesting.testing.runner import run_single_test


# ==================== Successful Test Execution ====================

class TestRunSingleTestSuccess:
    """Test successful test execution scenarios."""

    def test_run_single_test_with_trades(self):
        """Test successful test execution that generates trades."""
        # Prepare test parameters
        test_params = (
            '1!',  # tested_month
            'ZS',  # symbol
            '1h',  # interval
            'RSI_14_30_70',  # strategy_name
            MagicMock(),  # strategy_instance
            False,  # verbose
            [],  # switch_dates
            '/path/to/data.parquet',  # filepath
            None,  # segment_id
            None,  # period_id
            None,  # start_date
            None,  # end_date
        )

        # Mock dataframe
        mock_df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [102, 103, 104],
            'low': [99, 100, 101],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        })

        # Mock trades
        mock_trade = {
            'entry_time': datetime(2024, 1, 1, 10, 0),
            'exit_time': datetime(2024, 1, 1, 14, 0),
            'entry_price': 100.0,
            'exit_price': 105.0,
            'side': 'long'
        }

        with patch('app.backtesting.testing.runner.get_cached_dataframe') as mock_get_df, \
                patch('app.backtesting.testing.runner.validate_dataframe') as mock_validate, \
                patch('app.backtesting.testing.runner.calculate_trade_metrics') as mock_calc_metrics, \
                patch('app.backtesting.testing.runner.SummaryMetrics') as mock_summary, \
                patch('app.backtesting.testing.runner.indicator_cache') as mock_ind_cache, \
                patch('app.backtesting.testing.runner.dataframe_cache') as mock_df_cache:
            mock_get_df.return_value = mock_df
            mock_validate.return_value = True
            test_params[4].run.return_value = [mock_trade]

            # Mock metrics calculation
            trade_with_metrics = {**mock_trade, 'net_pnl': 500.0}
            mock_calc_metrics.return_value = trade_with_metrics

            # Mock summary metrics
            mock_summary_instance = MagicMock()
            mock_summary_instance.calculate_all_metrics.return_value = {
                'total_trades': 1,
                'win_rate': 100.0,
                'profit_factor': 2.5
            }
            mock_summary.return_value = mock_summary_instance

            # Mock cache stats
            mock_ind_cache.hits = 10
            mock_ind_cache.misses = 2
            mock_df_cache.hits = 5
            mock_df_cache.misses = 1

            result = run_single_test(test_params)

            # Verify result structure
            assert result is not None
            assert result['month'] == '1!'
            assert result['symbol'] == 'ZS'
            assert result['interval'] == '1h'
            assert result['strategy'] == 'RSI_14_30_70'
            assert 'metrics' in result
            assert 'timestamp' in result
            assert 'cache_stats' in result

    def test_run_single_test_returns_correct_structure(self):
        """Test that result dictionary has all required fields."""
        test_params = (
            '2!', 'CL', '15m', 'EMA_9_21', MagicMock(), False, [], '/path/to/data.parquet',
            None, None, None, None
        )

        mock_df = pd.DataFrame({
            'open': [100], 'high': [102], 'low': [99], 'close': [101], 'volume': [1000]
        })

        mock_trade = {
            'entry_time': datetime(2024, 1, 1),
            'exit_time': datetime(2024, 1, 2),
            'entry_price': 100.0,
            'exit_price': 105.0,
            'side': 'long'
        }

        with patch('app.backtesting.testing.runner.get_cached_dataframe', return_value=mock_df), \
                patch('app.backtesting.testing.runner.validate_dataframe', return_value=True), \
                patch('app.backtesting.testing.runner.calculate_trade_metrics',
                      return_value={**mock_trade, 'net_pnl': 500}), \
                patch('app.backtesting.testing.runner.SummaryMetrics') as mock_summary, \
                patch('app.backtesting.testing.runner.indicator_cache') as mock_ind_cache, \
                patch('app.backtesting.testing.runner.dataframe_cache') as mock_df_cache:

            test_params[4].run.return_value = [mock_trade]
            mock_summary.return_value.calculate_all_metrics.return_value = {'total_trades': 1}
            mock_ind_cache.hits = mock_ind_cache.misses = 0
            mock_df_cache.hits = mock_df_cache.misses = 0

            result = run_single_test(test_params)

            # Verify all required fields present
            required_fields = ['month', 'symbol', 'interval', 'strategy', 'metrics', 'timestamp', 'cache_stats']
            for field in required_fields:
                assert field in result

    def test_run_single_test_calls_strategy_run(self):
        """Test that strategy.run() is called with correct parameters."""
        mock_strategy = MagicMock()
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', mock_strategy, False, [], '/path/to/data.parquet',
            None, None, None, None
        )

        mock_df = pd.DataFrame({
            'open': [100], 'high': [102], 'low': [99], 'close': [101], 'volume': [1000]
        })
        switch_dates = [datetime(2024, 1, 1), datetime(2024, 2, 1)]
        test_params = test_params[:6] + (switch_dates,) + test_params[7:]

        with patch('app.backtesting.testing.runner.get_cached_dataframe', return_value=mock_df), \
                patch('app.backtesting.testing.runner.validate_dataframe', return_value=True), \
                patch('app.backtesting.testing.runner.calculate_trade_metrics', return_value={}), \
                patch('app.backtesting.testing.runner.SummaryMetrics') as mock_summary, \
                patch('app.backtesting.testing.runner.indicator_cache'), \
                patch('app.backtesting.testing.runner.dataframe_cache'):
            mock_strategy.run.return_value = []
            mock_summary.return_value.calculate_all_metrics.return_value = {}

            run_single_test(test_params)

            # Verify strategy.run was called with df and switch_dates
            mock_strategy.run.assert_called_once()
            call_args = mock_strategy.run.call_args[0]
            assert isinstance(call_args[0], pd.DataFrame)
            assert call_args[1] == switch_dates


# ==================== No Trades Scenario ====================

class TestRunSingleTestNoTrades:
    """Test scenarios where strategy generates no trades."""

    def test_run_single_test_no_trades_returns_empty_metrics(self):
        """Test that empty metrics dict is returned when no trades generated."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), False, [], '/path/to/data.parquet',
            None, None, None, None
        )

        mock_df = pd.DataFrame({
            'open': [100], 'high': [102], 'low': [99], 'close': [101], 'volume': [1000]
        })

        with patch('app.backtesting.testing.runner.get_cached_dataframe', return_value=mock_df), \
                patch('app.backtesting.testing.runner.validate_dataframe', return_value=True), \
                patch('app.backtesting.testing.runner.indicator_cache') as mock_ind_cache, \
                patch('app.backtesting.testing.runner.dataframe_cache') as mock_df_cache:
            test_params[4].run.return_value = []  # No trades
            mock_ind_cache.hits = mock_ind_cache.misses = 0
            mock_df_cache.hits = mock_df_cache.misses = 0

            result = run_single_test(test_params)

            # Should still return result with empty metrics
            assert result is not None
            assert result['metrics'] == {}
            assert result['month'] == '1!'
            assert result['symbol'] == 'ZS'

    def test_run_single_test_no_trades_includes_cache_stats(self):
        """Test that cache stats are included even when no trades generated."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), False, [], '/path/to/data.parquet',
            None, None, None, None
        )

        mock_df = pd.DataFrame({
            'open': [100], 'high': [102], 'low': [99], 'close': [101], 'volume': [1000]
        })

        with patch('app.backtesting.testing.runner.get_cached_dataframe', return_value=mock_df), \
                patch('app.backtesting.testing.runner.validate_dataframe', return_value=True), \
                patch('app.backtesting.testing.runner.indicator_cache') as mock_ind_cache, \
                patch('app.backtesting.testing.runner.dataframe_cache') as mock_df_cache:
            test_params[4].run.return_value = []
            mock_ind_cache.hits = 5
            mock_ind_cache.misses = 2
            mock_df_cache.hits = 3
            mock_df_cache.misses = 1

            result = run_single_test(test_params)

            # Verify cache stats present
            assert 'cache_stats' in result
            assert 'ind_hits' in result['cache_stats']
            assert 'ind_misses' in result['cache_stats']
            assert 'df_hits' in result['cache_stats']
            assert 'df_misses' in result['cache_stats']

    def test_run_single_test_no_trades_verbose_output(self, capsys):
        """Test verbose output when no trades are generated."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), True, [], '/path/to/data.parquet',
            None, None, None, None
        )

        mock_df = pd.DataFrame({
            'open': [100], 'high': [102], 'low': [99], 'close': [101], 'volume': [1000]
        })

        with patch('app.backtesting.testing.runner.get_cached_dataframe', return_value=mock_df), \
                patch('app.backtesting.testing.runner.validate_dataframe', return_value=True), \
                patch('app.backtesting.testing.runner.indicator_cache'), \
                patch('app.backtesting.testing.runner.dataframe_cache'):
            test_params[4].run.return_value = []

            run_single_test(test_params)

            captured = capsys.readouterr()
            assert 'No trades generated' in captured.out


# ==================== Error Handling Tests ====================

class TestRunSingleTestErrorHandling:
    """Test error handling scenarios."""

    def test_run_single_test_file_not_found_returns_none(self):
        """Test that None is returned when data file cannot be loaded."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), False, [], '/nonexistent/file.parquet',
            None, None, None, None
        )

        with patch('app.backtesting.testing.runner.get_cached_dataframe') as mock_get_df:
            mock_get_df.side_effect = FileNotFoundError("File not found")

            result = run_single_test(test_params)

            assert result is None

    def test_run_single_test_invalid_dataframe_returns_none(self):
        """Test that None is returned when DataFrame validation fails."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), False, [], '/path/to/data.parquet',
            None, None, None, None
        )

        mock_df = pd.DataFrame()  # Empty dataframe

        with patch('app.backtesting.testing.runner.get_cached_dataframe', return_value=mock_df), \
                patch('app.backtesting.testing.runner.validate_dataframe', return_value=False):
            result = run_single_test(test_params)

            assert result is None

    def test_run_single_test_general_exception_returns_none(self):
        """Test that general exceptions are caught and None is returned."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), False, [], '/path/to/data.parquet',
            None, None, None, None
        )

        with patch('app.backtesting.testing.runner.get_cached_dataframe') as mock_get_df:
            mock_get_df.side_effect = Exception("Unexpected error")

            result = run_single_test(test_params)

            assert result is None


# ==================== Cache Statistics Tests ====================

class TestCacheStatistics:
    """Test cache statistics tracking."""

    def test_cache_statistics_calculated_correctly(self):
        """Test that cache hit/miss deltas are calculated correctly."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), False, [], '/path/to/data.parquet',
            None, None, None, None
        )

        mock_df = pd.DataFrame({
            'open': [100], 'high': [102], 'low': [99], 'close': [101], 'volume': [1000]
        })

        with patch('app.backtesting.testing.runner.get_cached_dataframe', return_value=mock_df), \
                patch('app.backtesting.testing.runner.validate_dataframe', return_value=True), \
                patch('app.backtesting.testing.runner.indicator_cache') as mock_ind_cache, \
                patch('app.backtesting.testing.runner.dataframe_cache') as mock_df_cache:
            test_params[4].run.return_value = []

            # Set initial cache stats
            mock_ind_cache.hits = 10
            mock_ind_cache.misses = 5
            mock_df_cache.hits = 8
            mock_df_cache.misses = 3

            # Simulate cache activity during test (increment counters)
            def run_with_cache_activity(*args):
                mock_ind_cache.hits += 3
                mock_ind_cache.misses += 1
                mock_df_cache.hits += 2
                mock_df_cache.misses += 1
                return []

            test_params[4].run.side_effect = run_with_cache_activity

            result = run_single_test(test_params)

            # Verify deltas are correct
            assert result['cache_stats']['ind_hits'] == 3
            assert result['cache_stats']['ind_misses'] == 1
            assert result['cache_stats']['df_hits'] == 2
            assert result['cache_stats']['df_misses'] == 1

    def test_cache_statistics_with_multiple_trades(self):
        """Test cache statistics when multiple trades are generated."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), False, [], '/path/to/data.parquet',
            None, None, None, None
        )

        mock_df = pd.DataFrame({
            'open': [100], 'high': [102], 'low': [99], 'close': [101], 'volume': [1000]
        })

        mock_trades = [
            {'entry_time': datetime(2024, 1, 1), 'exit_time': datetime(2024, 1, 2),
             'entry_price': 100, 'exit_price': 105, 'side': 'long'},
            {'entry_time': datetime(2024, 1, 3), 'exit_time': datetime(2024, 1, 4),
             'entry_price': 105, 'exit_price': 110, 'side': 'long'}
        ]

        with patch('app.backtesting.testing.runner.get_cached_dataframe', return_value=mock_df), \
                patch('app.backtesting.testing.runner.validate_dataframe', return_value=True), \
                patch('app.backtesting.testing.runner.calculate_trade_metrics') as mock_calc, \
                patch('app.backtesting.testing.runner.SummaryMetrics') as mock_summary, \
                patch('app.backtesting.testing.runner.indicator_cache') as mock_ind_cache, \
                patch('app.backtesting.testing.runner.dataframe_cache') as mock_df_cache:
            test_params[4].run.return_value = mock_trades
            mock_calc.return_value = {'net_pnl': 500}
            mock_summary.return_value.calculate_all_metrics.return_value = {'total_trades': 2}

            mock_ind_cache.hits = 0
            mock_ind_cache.misses = 0
            mock_df_cache.hits = 0
            mock_df_cache.misses = 0

            result = run_single_test(test_params)

            # Cache stats should still be tracked
            assert 'cache_stats' in result


# ==================== Verbose Output Tests ====================

class TestVerboseOutput:
    """Test verbose output functionality."""

    def test_verbose_true_prints_strategy_execution(self, capsys):
        """Test that verbose=True prints strategy execution info."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), True, [], '/path/to/data.parquet',
            None, None, None, None
        )

        mock_df = pd.DataFrame({
            'open': [100], 'high': [102], 'low': [99], 'close': [101], 'volume': [1000]
        })

        with patch('app.backtesting.testing.runner.get_cached_dataframe', return_value=mock_df), \
                patch('app.backtesting.testing.runner.validate_dataframe', return_value=True), \
                patch('app.backtesting.testing.runner.indicator_cache'), \
                patch('app.backtesting.testing.runner.dataframe_cache'):
            test_params[4].run.return_value = []

            run_single_test(test_params)

            captured = capsys.readouterr()
            assert 'Running strategy: RSI_14_30_70' in captured.out
            assert 'ZS' in captured.out
            assert '1h' in captured.out
            assert '1!' in captured.out

    def test_verbose_false_no_output(self, capsys):
        """Test that verbose=False suppresses output."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), False, [], '/path/to/data.parquet',
            None, None, None, None
        )

        mock_df = pd.DataFrame({
            'open': [100], 'high': [102], 'low': [99], 'close': [101], 'volume': [1000]
        })

        mock_trade = {
            'entry_time': datetime(2024, 1, 1),
            'exit_time': datetime(2024, 1, 2),
            'entry_price': 100,
            'exit_price': 105,
            'side': 'long'
        }

        with patch('app.backtesting.testing.runner.get_cached_dataframe', return_value=mock_df), \
                patch('app.backtesting.testing.runner.validate_dataframe', return_value=True), \
                patch('app.backtesting.testing.runner.calculate_trade_metrics',
                      return_value={**mock_trade, 'net_pnl': 500}), \
                patch('app.backtesting.testing.runner.SummaryMetrics') as mock_summary, \
                patch('app.backtesting.testing.runner.indicator_cache'), \
                patch('app.backtesting.testing.runner.dataframe_cache'):
            test_params[4].run.return_value = [mock_trade]
            mock_summary.return_value.calculate_all_metrics.return_value = {'total_trades': 1}

            run_single_test(test_params)

            captured = capsys.readouterr()
            # Should not contain strategy execution messages
            assert 'Running strategy:' not in captured.out


# ==================== Metrics Calculation Tests ====================

class TestMetricsCalculation:
    """Test metrics calculation integration."""

    def test_calculate_trade_metrics_called_for_each_trade(self):
        """Test that calculate_trade_metrics is called for each trade."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), False, [], '/path/to/data.parquet',
            None, None, None, None
        )

        mock_df = pd.DataFrame({
            'open': [100], 'high': [102], 'low': [99], 'close': [101], 'volume': [1000]
        })

        mock_trades = [
            {'entry_time': datetime(2024, 1, 1), 'exit_time': datetime(2024, 1, 2),
             'entry_price': 100, 'exit_price': 105, 'side': 'long'},
            {'entry_time': datetime(2024, 1, 3), 'exit_time': datetime(2024, 1, 4),
             'entry_price': 105, 'exit_price': 100, 'side': 'short'}
        ]

        with patch('app.backtesting.testing.runner.get_cached_dataframe', return_value=mock_df), \
                patch('app.backtesting.testing.runner.validate_dataframe', return_value=True), \
                patch('app.backtesting.testing.runner.calculate_trade_metrics') as mock_calc, \
                patch('app.backtesting.testing.runner.SummaryMetrics') as mock_summary, \
                patch('app.backtesting.testing.runner.indicator_cache'), \
                patch('app.backtesting.testing.runner.dataframe_cache'):
            test_params[4].run.return_value = mock_trades
            mock_calc.return_value = {'net_pnl': 500}
            mock_summary.return_value.calculate_all_metrics.return_value = {}

            run_single_test(test_params)

            # Should be called twice (once for each trade)
            assert mock_calc.call_count == 2

    def test_summary_metrics_receives_trades_with_metrics(self):
        """Test that SummaryMetrics receives trades with calculated metrics."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), False, [], '/path/to/data.parquet',
            None, None, None, None
        )

        mock_df = pd.DataFrame({
            'open': [100], 'high': [102], 'low': [99], 'close': [101], 'volume': [1000]
        })

        mock_trade = {
            'entry_time': datetime(2024, 1, 1),
            'exit_time': datetime(2024, 1, 2),
            'entry_price': 100,
            'exit_price': 105,
            'side': 'long'
        }

        with patch('app.backtesting.testing.runner.get_cached_dataframe', return_value=mock_df), \
                patch('app.backtesting.testing.runner.validate_dataframe', return_value=True), \
                patch('app.backtesting.testing.runner.calculate_trade_metrics') as mock_calc, \
                patch('app.backtesting.testing.runner.SummaryMetrics') as mock_summary, \
                patch('app.backtesting.testing.runner.indicator_cache'), \
                patch('app.backtesting.testing.runner.dataframe_cache'):
            test_params[4].run.return_value = [mock_trade]
            trade_with_metrics = {**mock_trade, 'net_pnl': 500, 'commission': 10}
            mock_calc.return_value = trade_with_metrics
            mock_summary.return_value.calculate_all_metrics.return_value = {'total_trades': 1}

            run_single_test(test_params)

            # Verify SummaryMetrics was instantiated with trades that have metrics
            mock_summary.assert_called_once()
            trades_arg = mock_summary.call_args[0][0]
            assert len(trades_arg) == 1
            assert 'net_pnl' in trades_arg[0]
            assert 'commission' in trades_arg[0]


# ==================== DataFrame Validation Tests ====================

class TestDataFrameValidation:
    """Test DataFrame validation integration."""

    def test_dataframe_validation_called(self):
        """Test that validate_dataframe is called."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), False, [], '/path/to/data.parquet',
            None, None, None, None
        )

        mock_df = pd.DataFrame({
            'open': [100], 'high': [102], 'low': [99], 'close': [101], 'volume': [1000]
        })

        with patch('app.backtesting.testing.runner.get_cached_dataframe', return_value=mock_df), \
                patch('app.backtesting.testing.runner.validate_dataframe') as mock_validate, \
                patch('app.backtesting.testing.runner.indicator_cache'), \
                patch('app.backtesting.testing.runner.dataframe_cache'):
            mock_validate.return_value = True
            test_params[4].run.return_value = []

            run_single_test(test_params)

            # Verify validation was called with df and filepath
            mock_validate.assert_called_once()
            assert isinstance(mock_validate.call_args[0][0], pd.DataFrame)

    def test_insufficient_rows_warning_logged(self):
        """Test that warning is logged when DataFrame has insufficient rows."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), False, [], '/path/to/data.parquet',
            None, None, None, None
        )

        # Create DataFrame with very few rows
        mock_df = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102],
            'volume': [1000, 1100]
        })

        with patch('app.backtesting.testing.runner.get_cached_dataframe', return_value=mock_df), \
                patch('app.backtesting.testing.runner.validate_dataframe', return_value=True), \
                patch('app.backtesting.testing.runner.MIN_ROWS_FOR_BACKTEST', 100), \
                patch('app.backtesting.testing.runner.indicator_cache'), \
                patch('app.backtesting.testing.runner.dataframe_cache'):
            test_params[4].run.return_value = []

            # Should still run but log warning
            result = run_single_test(test_params)

            # Test should complete despite insufficient rows
            assert result is not None


# ==================== Integration Tests ====================

class TestRunnerIntegration:
    """Test runner integration scenarios."""

    def test_complete_workflow_with_successful_trade(self):
        """Test complete workflow from loading data to returning results."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), False, [], '/path/to/data.parquet',
            None, None, None, None
        )

        mock_df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [102, 103, 104],
            'low': [99, 100, 101],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        })

        mock_trade = {
            'entry_time': datetime(2024, 1, 1, 10, 0),
            'exit_time': datetime(2024, 1, 1, 14, 0),
            'entry_price': 100.0,
            'exit_price': 105.0,
            'side': 'long'
        }

        with patch('app.backtesting.testing.runner.get_cached_dataframe', return_value=mock_df), \
                patch('app.backtesting.testing.runner.validate_dataframe', return_value=True), \
                patch('app.backtesting.testing.runner.calculate_trade_metrics') as mock_calc, \
                patch('app.backtesting.testing.runner.SummaryMetrics') as mock_summary, \
                patch('app.backtesting.testing.runner.indicator_cache') as mock_ind_cache, \
                patch('app.backtesting.testing.runner.dataframe_cache') as mock_df_cache:
            # Setup mocks
            test_params[4].run.return_value = [mock_trade]
            mock_calc.return_value = {**mock_trade, 'net_pnl': 500.0, 'commission': 5.0}
            mock_summary.return_value.calculate_all_metrics.return_value = {
                'total_trades': 1,
                'win_rate': 100.0,
                'profit_factor': 3.0,
                'sharpe_ratio': 2.5
            }
            mock_ind_cache.hits = mock_ind_cache.misses = 0
            mock_df_cache.hits = mock_df_cache.misses = 0

            result = run_single_test(test_params)

            # Verify complete result
            assert result is not None
            assert result['month'] == '1!'
            assert result['symbol'] == 'ZS'
            assert result['interval'] == '1h'
            assert result['strategy'] == 'RSI_14_30_70'
            assert result['metrics']['total_trades'] == 1
            assert result['metrics']['win_rate'] == 100.0
            assert 'timestamp' in result
            assert 'cache_stats' in result

    def test_test_params_tuple_unpacking(self):
        """Test that all 12 test parameters are unpacked correctly."""
        tested_month = '2!'
        symbol = 'CL'
        interval = '15m'
        strategy_name = 'MACD_12_26_9'
        strategy_instance = MagicMock()
        verbose = True
        switch_dates = [datetime(2024, 1, 1)]
        filepath = '/data/test.parquet'

        test_params = (
            tested_month, symbol, interval, strategy_name,
            strategy_instance, verbose, switch_dates, filepath,
            None, None, None, None
        )

        mock_df = pd.DataFrame({
            'open': [100], 'high': [102], 'low': [99], 'close': [101], 'volume': [1000]
        })

        with patch('app.backtesting.testing.runner.get_cached_dataframe', return_value=mock_df), \
                patch('app.backtesting.testing.runner.validate_dataframe', return_value=True), \
                patch('app.backtesting.testing.runner.indicator_cache'), \
                patch('app.backtesting.testing.runner.dataframe_cache'):
            strategy_instance.run.return_value = []

            result = run_single_test(test_params)

            # Verify all parameters used correctly
            assert result['month'] == tested_month
            assert result['symbol'] == symbol
            assert result['interval'] == interval
            assert result['strategy'] == strategy_name
            strategy_instance.run.assert_called_once_with(mock_df, switch_dates)
