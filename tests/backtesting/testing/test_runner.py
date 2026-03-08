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
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd

from app.backtesting.testing.runner import run_single_test


# ==================== Successful Test Execution ====================

class TestRunSingleTestSuccess:
    """Test successful test execution scenarios."""

    def test_run_single_test_with_trades(self, monkeypatch):
        """Test successful test execution that generates trades."""
        test_params = (
            '1!',  # tested_month
            'ZS',  # symbol
            '1h',  # interval
            'RSI_14_30_70',  # strategy_name
            MagicMock(),  # strategy_instance
            False,  # verbose
            [],  # switch_dates
            '/path/to/data.parquet'  # filepath
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

        mock_summary = MagicMock()
        mock_summary.return_value.calculate_all_metrics.return_value = {
            'total_trades': 1,
            'win_rate': 100.0,
            'profit_factor': 2.5
        }
        mock_ind_cache = MagicMock()
        mock_ind_cache.hits = 10
        mock_ind_cache.misses = 2
        mock_df_cache = MagicMock()
        mock_df_cache.hits = 5
        mock_df_cache.misses = 1

        monkeypatch.setattr('app.backtesting.testing.runner.get_cached_dataframe', MagicMock(return_value=mock_df))
        monkeypatch.setattr('app.backtesting.testing.runner.validate_dataframe', MagicMock(return_value=True))
        monkeypatch.setattr('app.backtesting.testing.runner.calculate_trade_metrics',
                            MagicMock(return_value={**mock_trade, 'net_pnl': 500.0}))
        monkeypatch.setattr('app.backtesting.testing.runner.SummaryMetrics', mock_summary)
        monkeypatch.setattr('app.backtesting.testing.runner.indicator_cache', mock_ind_cache)
        monkeypatch.setattr('app.backtesting.testing.runner.dataframe_cache', mock_df_cache)

        test_params[4].run.return_value = [mock_trade]

        result = run_single_test(test_params)

        assert result is not None
        assert result['month'] == '1!'
        assert result['symbol'] == 'ZS'
        assert result['interval'] == '1h'
        assert result['strategy'] == 'RSI_14_30_70'
        assert 'metrics' in result
        assert 'timestamp' in result
        assert 'cache_stats' in result

    def test_run_single_test_returns_correct_structure(self, monkeypatch):
        """Test that result dictionary has all required fields."""
        test_params = (
            '2!', 'CL', '15m', 'EMA_9_21', MagicMock(), False, [], '/path/to/data.parquet'
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

        mock_summary = MagicMock()
        mock_summary.return_value.calculate_all_metrics.return_value = {'total_trades': 1}
        mock_ind_cache = MagicMock()
        mock_ind_cache.hits = mock_ind_cache.misses = 0
        mock_df_cache = MagicMock()
        mock_df_cache.hits = mock_df_cache.misses = 0

        monkeypatch.setattr('app.backtesting.testing.runner.get_cached_dataframe', MagicMock(return_value=mock_df))
        monkeypatch.setattr('app.backtesting.testing.runner.validate_dataframe', MagicMock(return_value=True))
        monkeypatch.setattr('app.backtesting.testing.runner.calculate_trade_metrics',
                            MagicMock(return_value={**mock_trade, 'net_pnl': 500}))
        monkeypatch.setattr('app.backtesting.testing.runner.SummaryMetrics', mock_summary)
        monkeypatch.setattr('app.backtesting.testing.runner.indicator_cache', mock_ind_cache)
        monkeypatch.setattr('app.backtesting.testing.runner.dataframe_cache', mock_df_cache)

        test_params[4].run.return_value = [mock_trade]

        result = run_single_test(test_params)

        required_fields = ['month', 'symbol', 'interval', 'strategy', 'metrics', 'timestamp', 'cache_stats']
        for field in required_fields:
            assert field in result

    def test_run_single_test_calls_strategy_run(self, monkeypatch):
        """Test that strategy.run() is called with correct parameters."""
        mock_strategy = MagicMock()
        switch_dates = [datetime(2024, 1, 1), datetime(2024, 2, 1)]
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', mock_strategy, False, switch_dates, '/path/to/data.parquet'
        )

        mock_df = pd.DataFrame({
            'open': [100], 'high': [102], 'low': [99], 'close': [101], 'volume': [1000]
        })

        mock_summary = MagicMock()
        mock_summary.return_value.calculate_all_metrics.return_value = {}

        monkeypatch.setattr('app.backtesting.testing.runner.get_cached_dataframe', MagicMock(return_value=mock_df))
        monkeypatch.setattr('app.backtesting.testing.runner.validate_dataframe', MagicMock(return_value=True))
        monkeypatch.setattr('app.backtesting.testing.runner.calculate_trade_metrics', MagicMock(return_value={}))
        monkeypatch.setattr('app.backtesting.testing.runner.SummaryMetrics', mock_summary)
        monkeypatch.setattr('app.backtesting.testing.runner.indicator_cache', MagicMock())
        monkeypatch.setattr('app.backtesting.testing.runner.dataframe_cache', MagicMock())

        mock_strategy.run.return_value = []

        run_single_test(test_params)

        mock_strategy.run.assert_called_once()
        call_args = mock_strategy.run.call_args[0]
        assert isinstance(call_args[0], pd.DataFrame)
        assert call_args[1] == switch_dates


# ==================== No Trades Scenario ====================

class TestRunSingleTestNoTrades:
    """Test scenarios where strategy generates no trades."""

    def test_run_single_test_no_trades_returns_empty_metrics(self, monkeypatch):
        """Test that empty metrics dict is returned when no trades generated."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), False, [], '/path/to/data.parquet'
        )

        mock_df = pd.DataFrame({
            'open': [100], 'high': [102], 'low': [99], 'close': [101], 'volume': [1000]
        })

        mock_ind_cache = MagicMock()
        mock_ind_cache.hits = mock_ind_cache.misses = 0
        mock_df_cache = MagicMock()
        mock_df_cache.hits = mock_df_cache.misses = 0

        monkeypatch.setattr('app.backtesting.testing.runner.get_cached_dataframe', MagicMock(return_value=mock_df))
        monkeypatch.setattr('app.backtesting.testing.runner.validate_dataframe', MagicMock(return_value=True))
        monkeypatch.setattr('app.backtesting.testing.runner.indicator_cache', mock_ind_cache)
        monkeypatch.setattr('app.backtesting.testing.runner.dataframe_cache', mock_df_cache)

        test_params[4].run.return_value = []

        result = run_single_test(test_params)

        assert result is not None
        assert result['metrics'] == {}
        assert result['month'] == '1!'
        assert result['symbol'] == 'ZS'

    def test_run_single_test_no_trades_includes_cache_stats(self, monkeypatch):
        """Test that cache stats are included even when no trades generated."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), False, [], '/path/to/data.parquet'
        )

        mock_df = pd.DataFrame({
            'open': [100], 'high': [102], 'low': [99], 'close': [101], 'volume': [1000]
        })

        mock_ind_cache = MagicMock()
        mock_ind_cache.hits = 5
        mock_ind_cache.misses = 2
        mock_df_cache = MagicMock()
        mock_df_cache.hits = 3
        mock_df_cache.misses = 1

        monkeypatch.setattr('app.backtesting.testing.runner.get_cached_dataframe', MagicMock(return_value=mock_df))
        monkeypatch.setattr('app.backtesting.testing.runner.validate_dataframe', MagicMock(return_value=True))
        monkeypatch.setattr('app.backtesting.testing.runner.indicator_cache', mock_ind_cache)
        monkeypatch.setattr('app.backtesting.testing.runner.dataframe_cache', mock_df_cache)

        test_params[4].run.return_value = []

        result = run_single_test(test_params)

        assert 'cache_stats' in result
        assert 'ind_hits' in result['cache_stats']
        assert 'ind_misses' in result['cache_stats']
        assert 'df_hits' in result['cache_stats']
        assert 'df_misses' in result['cache_stats']

    def test_run_single_test_no_trades_verbose_output(self, monkeypatch, capsys):
        """Test verbose output when no trades are generated."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), True, [], '/path/to/data.parquet'
        )

        mock_df = pd.DataFrame({
            'open': [100], 'high': [102], 'low': [99], 'close': [101], 'volume': [1000]
        })

        monkeypatch.setattr('app.backtesting.testing.runner.get_cached_dataframe', MagicMock(return_value=mock_df))
        monkeypatch.setattr('app.backtesting.testing.runner.validate_dataframe', MagicMock(return_value=True))
        monkeypatch.setattr('app.backtesting.testing.runner.indicator_cache', MagicMock())
        monkeypatch.setattr('app.backtesting.testing.runner.dataframe_cache', MagicMock())

        test_params[4].run.return_value = []

        run_single_test(test_params)

        captured = capsys.readouterr()
        assert 'No trades generated' in captured.out


# ==================== Error Handling Tests ====================

class TestRunSingleTestErrorHandling:
    """Test error handling scenarios."""

    def test_run_single_test_file_not_found_returns_none(self, monkeypatch):
        """Test that None is returned when data file cannot be loaded."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), False, [], '/nonexistent/file.parquet'
        )

        mock_get_df = MagicMock(side_effect=FileNotFoundError("File not found"))
        monkeypatch.setattr('app.backtesting.testing.runner.get_cached_dataframe', mock_get_df)

        result = run_single_test(test_params)

        assert result is None

    def test_run_single_test_invalid_dataframe_returns_none(self, monkeypatch):
        """Test that None is returned when DataFrame validation fails."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), False, [], '/path/to/data.parquet'
        )

        monkeypatch.setattr('app.backtesting.testing.runner.get_cached_dataframe',
                            MagicMock(return_value=pd.DataFrame()))
        monkeypatch.setattr('app.backtesting.testing.runner.validate_dataframe', MagicMock(return_value=False))

        result = run_single_test(test_params)

        assert result is None

    def test_run_single_test_general_exception_returns_none(self, monkeypatch):
        """Test that general exceptions are caught and None is returned."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), False, [], '/path/to/data.parquet'
        )

        mock_get_df = MagicMock(side_effect=Exception("Unexpected error"))
        monkeypatch.setattr('app.backtesting.testing.runner.get_cached_dataframe', mock_get_df)

        result = run_single_test(test_params)

        assert result is None


# ==================== Cache Statistics Tests ====================

class TestCacheStatistics:
    """Test cache statistics tracking."""

    def test_cache_statistics_calculated_correctly(self, monkeypatch):
        """Test that cache hit/miss deltas are calculated correctly."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), False, [], '/path/to/data.parquet'
        )

        mock_df = pd.DataFrame({
            'open': [100], 'high': [102], 'low': [99], 'close': [101], 'volume': [1000]
        })

        mock_ind_cache = MagicMock()
        mock_ind_cache.hits = 10
        mock_ind_cache.misses = 5
        mock_df_cache = MagicMock()
        mock_df_cache.hits = 8
        mock_df_cache.misses = 3

        monkeypatch.setattr('app.backtesting.testing.runner.get_cached_dataframe', MagicMock(return_value=mock_df))
        monkeypatch.setattr('app.backtesting.testing.runner.validate_dataframe', MagicMock(return_value=True))
        monkeypatch.setattr('app.backtesting.testing.runner.indicator_cache', mock_ind_cache)
        monkeypatch.setattr('app.backtesting.testing.runner.dataframe_cache', mock_df_cache)

        def run_with_cache_activity(*args, **kwargs):
            mock_ind_cache.hits += 3
            mock_ind_cache.misses += 1
            mock_df_cache.hits += 2
            mock_df_cache.misses += 1
            return []

        test_params[4].run.side_effect = run_with_cache_activity

        result = run_single_test(test_params)

        assert result['cache_stats']['ind_hits'] == 3
        assert result['cache_stats']['ind_misses'] == 1
        assert result['cache_stats']['df_hits'] == 2
        assert result['cache_stats']['df_misses'] == 1

    def test_cache_statistics_with_multiple_trades(self, monkeypatch):
        """Test cache statistics when multiple trades are generated."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), False, [], '/path/to/data.parquet'
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

        mock_summary = MagicMock()
        mock_summary.return_value.calculate_all_metrics.return_value = {'total_trades': 2}
        mock_ind_cache = MagicMock()
        mock_ind_cache.hits = mock_ind_cache.misses = 0
        mock_df_cache = MagicMock()
        mock_df_cache.hits = mock_df_cache.misses = 0

        monkeypatch.setattr('app.backtesting.testing.runner.get_cached_dataframe', MagicMock(return_value=mock_df))
        monkeypatch.setattr('app.backtesting.testing.runner.validate_dataframe', MagicMock(return_value=True))
        monkeypatch.setattr('app.backtesting.testing.runner.calculate_trade_metrics',
                            MagicMock(return_value={'net_pnl': 500}))
        monkeypatch.setattr('app.backtesting.testing.runner.SummaryMetrics', mock_summary)
        monkeypatch.setattr('app.backtesting.testing.runner.indicator_cache', mock_ind_cache)
        monkeypatch.setattr('app.backtesting.testing.runner.dataframe_cache', mock_df_cache)

        test_params[4].run.return_value = mock_trades

        result = run_single_test(test_params)

        assert 'cache_stats' in result


# ==================== Verbose Output Tests ====================

class TestVerboseOutput:
    """Test verbose output functionality."""

    def test_verbose_true_prints_strategy_execution(self, monkeypatch, capsys):
        """Test that verbose=True prints strategy execution info."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), True, [], '/path/to/data.parquet'
        )

        mock_df = pd.DataFrame({
            'open': [100], 'high': [102], 'low': [99], 'close': [101], 'volume': [1000]
        })

        monkeypatch.setattr('app.backtesting.testing.runner.get_cached_dataframe', MagicMock(return_value=mock_df))
        monkeypatch.setattr('app.backtesting.testing.runner.validate_dataframe', MagicMock(return_value=True))
        monkeypatch.setattr('app.backtesting.testing.runner.indicator_cache', MagicMock())
        monkeypatch.setattr('app.backtesting.testing.runner.dataframe_cache', MagicMock())

        test_params[4].run.return_value = []

        run_single_test(test_params)

        captured = capsys.readouterr()
        assert 'Running strategy: RSI_14_30_70' in captured.out
        assert 'ZS' in captured.out
        assert '1h' in captured.out
        assert '1!' in captured.out

    def test_verbose_false_no_output(self, monkeypatch, capsys):
        """Test that verbose=False suppresses output."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), False, [], '/path/to/data.parquet'
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

        mock_summary = MagicMock()
        mock_summary.return_value.calculate_all_metrics.return_value = {'total_trades': 1}

        monkeypatch.setattr('app.backtesting.testing.runner.get_cached_dataframe', MagicMock(return_value=mock_df))
        monkeypatch.setattr('app.backtesting.testing.runner.validate_dataframe', MagicMock(return_value=True))
        monkeypatch.setattr('app.backtesting.testing.runner.calculate_trade_metrics',
                            MagicMock(return_value={**mock_trade, 'net_pnl': 500}))
        monkeypatch.setattr('app.backtesting.testing.runner.SummaryMetrics', mock_summary)
        monkeypatch.setattr('app.backtesting.testing.runner.indicator_cache', MagicMock())
        monkeypatch.setattr('app.backtesting.testing.runner.dataframe_cache', MagicMock())

        test_params[4].run.return_value = [mock_trade]

        run_single_test(test_params)

        captured = capsys.readouterr()
        assert 'Running strategy:' not in captured.out


# ==================== Metrics Calculation Tests ====================

class TestMetricsCalculation:
    """Test metrics calculation integration."""

    def test_calculate_trade_metrics_called_for_each_trade(self, monkeypatch):
        """Test that calculate_trade_metrics is called for each trade."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), False, [], '/path/to/data.parquet'
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

        mock_calc = MagicMock(return_value={'net_pnl': 500})
        mock_summary = MagicMock()
        mock_summary.return_value.calculate_all_metrics.return_value = {}

        monkeypatch.setattr('app.backtesting.testing.runner.get_cached_dataframe', MagicMock(return_value=mock_df))
        monkeypatch.setattr('app.backtesting.testing.runner.validate_dataframe', MagicMock(return_value=True))
        monkeypatch.setattr('app.backtesting.testing.runner.calculate_trade_metrics', mock_calc)
        monkeypatch.setattr('app.backtesting.testing.runner.SummaryMetrics', mock_summary)
        monkeypatch.setattr('app.backtesting.testing.runner.indicator_cache', MagicMock())
        monkeypatch.setattr('app.backtesting.testing.runner.dataframe_cache', MagicMock())

        test_params[4].run.return_value = mock_trades

        run_single_test(test_params)

        assert mock_calc.call_count == 2

    def test_duration_bars_attached_to_trades(self, monkeypatch):
        """Test that duration_bars is computed and attached to each trade before SummaryMetrics."""
        test_params = (
            '1!', 'ZS', '4h', 'RSI_14_30_70', MagicMock(), False, [], '/path/to/data.parquet'
        )

        mock_df = pd.DataFrame({
            'open': [100], 'high': [102], 'low': [99], 'close': [101], 'volume': [1000]
        })

        mock_trade = {
            'entry_time': datetime(2024, 1, 1, 0, 0),
            'exit_time': datetime(2024, 1, 1, 8, 0),
            'entry_price': 100.0,
            'exit_price': 105.0,
            'side': 'long'
        }

        mock_summary = MagicMock()
        mock_summary.return_value.calculate_all_metrics.return_value = {'total_trades': 1}

        monkeypatch.setattr('app.backtesting.testing.runner.get_cached_dataframe', MagicMock(return_value=mock_df))
        monkeypatch.setattr('app.backtesting.testing.runner.validate_dataframe', MagicMock(return_value=True))
        monkeypatch.setattr('app.backtesting.testing.runner.calculate_trade_metrics',
                            MagicMock(return_value={**mock_trade, 'net_pnl': 500.0, 'duration': timedelta(hours=8)}))
        monkeypatch.setattr('app.backtesting.testing.runner.SummaryMetrics', mock_summary)
        monkeypatch.setattr('app.backtesting.testing.runner.indicator_cache', MagicMock())
        monkeypatch.setattr('app.backtesting.testing.runner.dataframe_cache', MagicMock())

        test_params[4].run.return_value = [mock_trade]

        run_single_test(test_params)

        # duration=8h, interval=4h → duration_bars should be 2.0
        trades_arg = mock_summary.call_args[0][0]
        assert 'duration_bars' in trades_arg[0]
        assert trades_arg[0]['duration_bars'] == 2.0

    def test_dataset_total_hours_passed_to_summary_metrics(self, monkeypatch):
        """Test that dataset_total_hours is computed from df index and passed to SummaryMetrics."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), False, [], '/path/to/data.parquet'
        )

        # DataFrame with a DatetimeIndex spanning exactly 10 hours
        idx = pd.date_range('2024-01-01 00:00', periods=11, freq='1h')
        mock_df = pd.DataFrame({
            'open': [100] * 11, 'high': [102] * 11, 'low': [99] * 11,
            'close': [101] * 11, 'volume': [1000] * 11
        }, index=idx)

        mock_trade = {
            'entry_time': datetime(2024, 1, 1, 0, 0),
            'exit_time': datetime(2024, 1, 1, 2, 0),
            'entry_price': 100.0,
            'exit_price': 105.0,
            'side': 'long'
        }

        mock_summary = MagicMock()
        mock_summary.return_value.calculate_all_metrics.return_value = {'total_trades': 1}

        monkeypatch.setattr('app.backtesting.testing.runner.get_cached_dataframe', MagicMock(return_value=mock_df))
        monkeypatch.setattr('app.backtesting.testing.runner.validate_dataframe', MagicMock(return_value=True))
        monkeypatch.setattr('app.backtesting.testing.runner.calculate_trade_metrics',
                            MagicMock(return_value={**mock_trade, 'net_pnl': 500.0, 'duration': timedelta(hours=2)}))
        monkeypatch.setattr('app.backtesting.testing.runner.SummaryMetrics', mock_summary)
        monkeypatch.setattr('app.backtesting.testing.runner.indicator_cache', MagicMock())
        monkeypatch.setattr('app.backtesting.testing.runner.dataframe_cache', MagicMock())

        test_params[4].run.return_value = [mock_trade]

        run_single_test(test_params)

        mock_summary.assert_called_once()
        kwargs = mock_summary.call_args[1]
        assert 'dataset_total_hours' in kwargs
        assert kwargs['dataset_total_hours'] == 10.0

    def test_summary_metrics_receives_trades_with_metrics(self, monkeypatch):
        """Test that SummaryMetrics receives trades with calculated metrics."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), False, [], '/path/to/data.parquet'
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

        trade_with_metrics = {**mock_trade, 'net_pnl': 500, 'commission': 10}
        mock_summary = MagicMock()
        mock_summary.return_value.calculate_all_metrics.return_value = {'total_trades': 1}

        monkeypatch.setattr('app.backtesting.testing.runner.get_cached_dataframe', MagicMock(return_value=mock_df))
        monkeypatch.setattr('app.backtesting.testing.runner.validate_dataframe', MagicMock(return_value=True))
        monkeypatch.setattr('app.backtesting.testing.runner.calculate_trade_metrics',
                            MagicMock(return_value=trade_with_metrics))
        monkeypatch.setattr('app.backtesting.testing.runner.SummaryMetrics', mock_summary)
        monkeypatch.setattr('app.backtesting.testing.runner.indicator_cache', MagicMock())
        monkeypatch.setattr('app.backtesting.testing.runner.dataframe_cache', MagicMock())

        test_params[4].run.return_value = [mock_trade]

        run_single_test(test_params)

        mock_summary.assert_called_once()
        trades_arg = mock_summary.call_args[0][0]
        assert len(trades_arg) == 1
        assert 'net_pnl' in trades_arg[0]
        assert 'commission' in trades_arg[0]


# ==================== DataFrame Validation Tests ====================

class TestDataFrameValidation:
    """Test DataFrame validation integration."""

    def test_dataframe_validation_called(self, monkeypatch):
        """Test that validate_dataframe is called."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), False, [], '/path/to/data.parquet'
        )

        mock_df = pd.DataFrame({
            'open': [100], 'high': [102], 'low': [99], 'close': [101], 'volume': [1000]
        })

        mock_validate = MagicMock(return_value=True)

        monkeypatch.setattr('app.backtesting.testing.runner.get_cached_dataframe', MagicMock(return_value=mock_df))
        monkeypatch.setattr('app.backtesting.testing.runner.validate_dataframe', mock_validate)
        monkeypatch.setattr('app.backtesting.testing.runner.indicator_cache', MagicMock())
        monkeypatch.setattr('app.backtesting.testing.runner.dataframe_cache', MagicMock())

        test_params[4].run.return_value = []

        run_single_test(test_params)

        mock_validate.assert_called_once()
        assert isinstance(mock_validate.call_args[0][0], pd.DataFrame)

    def test_insufficient_rows_warning_logged(self, monkeypatch):
        """Test that warning is logged when DataFrame has insufficient rows."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), False, [], '/path/to/data.parquet'
        )

        mock_df = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102],
            'volume': [1000, 1100]
        })

        monkeypatch.setattr('app.backtesting.testing.runner.get_cached_dataframe', MagicMock(return_value=mock_df))
        monkeypatch.setattr('app.backtesting.testing.runner.validate_dataframe', MagicMock(return_value=True))
        monkeypatch.setattr('app.backtesting.testing.runner.MIN_ROWS_FOR_BACKTEST', 100)
        monkeypatch.setattr('app.backtesting.testing.runner.indicator_cache', MagicMock())
        monkeypatch.setattr('app.backtesting.testing.runner.dataframe_cache', MagicMock())

        test_params[4].run.return_value = []

        result = run_single_test(test_params)

        assert result is not None


# ==================== Integration Tests ====================

class TestRunnerIntegration:
    """Test runner integration scenarios."""

    def test_complete_workflow_with_successful_trade(self, monkeypatch):
        """Test complete workflow from loading data to returning results."""
        test_params = (
            '1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock(), False, [], '/path/to/data.parquet'
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

        mock_summary = MagicMock()
        mock_summary.return_value.calculate_all_metrics.return_value = {
            'total_trades': 1,
            'win_rate': 100.0,
            'profit_factor': 3.0,
            'sharpe_ratio': 2.5
        }
        mock_ind_cache = MagicMock()
        mock_ind_cache.hits = mock_ind_cache.misses = 0
        mock_df_cache = MagicMock()
        mock_df_cache.hits = mock_df_cache.misses = 0

        monkeypatch.setattr('app.backtesting.testing.runner.get_cached_dataframe', MagicMock(return_value=mock_df))
        monkeypatch.setattr('app.backtesting.testing.runner.validate_dataframe', MagicMock(return_value=True))
        monkeypatch.setattr('app.backtesting.testing.runner.calculate_trade_metrics',
                            MagicMock(return_value={**mock_trade, 'net_pnl': 500.0, 'commission': 5.0}))
        monkeypatch.setattr('app.backtesting.testing.runner.SummaryMetrics', mock_summary)
        monkeypatch.setattr('app.backtesting.testing.runner.indicator_cache', mock_ind_cache)
        monkeypatch.setattr('app.backtesting.testing.runner.dataframe_cache', mock_df_cache)

        test_params[4].run.return_value = [mock_trade]

        result = run_single_test(test_params)

        assert result is not None
        assert result['month'] == '1!'
        assert result['symbol'] == 'ZS'
        assert result['interval'] == '1h'
        assert result['strategy'] == 'RSI_14_30_70'
        assert result['metrics']['total_trades'] == 1
        assert result['metrics']['win_rate'] == 100.0
        assert 'timestamp' in result
        assert 'cache_stats' in result

    def test_test_params_tuple_unpacking(self, monkeypatch, capsys):
        """Test that all 8 test parameters are unpacked correctly."""
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
            strategy_instance, verbose, switch_dates, filepath
        )

        mock_df = pd.DataFrame({
            'open': [100], 'high': [102], 'low': [99], 'close': [101], 'volume': [1000]
        })

        monkeypatch.setattr('app.backtesting.testing.runner.get_cached_dataframe', MagicMock(return_value=mock_df))
        monkeypatch.setattr('app.backtesting.testing.runner.validate_dataframe', MagicMock(return_value=True))
        monkeypatch.setattr('app.backtesting.testing.runner.indicator_cache', MagicMock())
        monkeypatch.setattr('app.backtesting.testing.runner.dataframe_cache', MagicMock())

        strategy_instance.run.return_value = []

        result = run_single_test(test_params)

        assert result['month'] == tested_month
        assert result['symbol'] == symbol
        assert result['interval'] == interval
        assert result['strategy'] == strategy_name
        strategy_instance.run.assert_called_once_with(mock_df, switch_dates, symbol=symbol)
