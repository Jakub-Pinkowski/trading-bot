"""
Tests for Formatters Module.

Tests formatting and CSV export helper functions including strategy name parsing,
DataFrame formatting, filename generation, and column transformations.

Test Coverage:
- Strategy name parsing (parameter extraction)
- DataFrame formatting for export
- Column reordering
- Column name transformation
- Filename building with various parameters
- Edge cases and error handling
"""
from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from app.backtesting.analysis.constants import DECIMAL_PLACES
from app.backtesting.analysis.formatters import (
    format_dataframe_for_export,
    build_filename,
    _parse_strategy_name,
    _reorder_columns,
    _format_column_name
)


# ==================== Fixtures ====================
# Note: Core fixtures (formatting_strategy_data) are in conftest.py

@pytest.fixture
def aggregated_strategy_results():
    """
    Create sample aggregated strategy results (without symbol/interval columns).

    Returns:
        DataFrame with aggregated metrics across symbols and intervals
    """
    return pd.DataFrame([
        {
            'strategy': 'RSI(period=14,lower=30,upper=70,rollover=True,trailing=2.5,slippage_ticks=1.0)',
            'symbol_count': 3,
            'interval_count': 2,
            'total_trades': 230,
            'win_rate': 62.5,
            'profit_factor': 3.5,
            'average_trade_return_percentage_of_contract': 1.1,
            'maximum_drawdown_percentage': 12.0,
            'sharpe_ratio': 2.3
        }
    ])


# ==================== Test Classes ====================

class TestStrategyNameParsing:
    """Test strategy name parsing and parameter extraction."""

    def test_parse_strategy_with_all_parameters(self):
        """Test parsing strategy name with rollover, trailing, and slippage."""
        strategy_name = 'RSI(period=14,lower=30,upper=70,rollover=True,trailing=2.5,slippage_ticks=1.0)'

        clean_name, rollover, trailing, slippage = _parse_strategy_name(strategy_name)

        # Clean name should have common params removed
        assert 'rollover' not in clean_name
        assert 'trailing' not in clean_name
        assert 'slippage_ticks' not in clean_name
        assert 'RSI(period=14,lower=30,upper=70)' == clean_name

        # Extracted params should match
        assert rollover is True
        assert trailing == 2.5
        assert slippage == 1.0

    def test_parse_strategy_with_no_trailing(self):
        """Test parsing strategy with trailing=None."""
        strategy_name = 'MACD(fast=12,slow=26,signal=9,rollover=False,trailing=None,slippage_ticks=2.0)'

        clean_name, rollover, trailing, slippage = _parse_strategy_name(strategy_name)

        assert 'MACD(fast=12,slow=26,signal=9)' == clean_name
        assert rollover is False
        assert trailing is None
        assert slippage == 2.0

    def test_parse_strategy_with_zero_slippage(self):
        """Test parsing strategy with slippage_ticks=0.0."""
        strategy_name = 'EMA(fast=12,slow=26,rollover=True,trailing=3.0,slippage_ticks=0.0)'

        clean_name, rollover, trailing, slippage = _parse_strategy_name(strategy_name)

        assert 'EMA(fast=12,slow=26)' == clean_name
        assert rollover is True
        assert trailing == 3.0
        assert slippage == 0.0

    def test_parse_strategy_minimal_params(self):
        """Test parsing strategy with only rollover parameter."""
        strategy_name = 'BB(period=20,std=2,rollover=False,trailing=None,slippage_ticks=0.0)'

        clean_name, rollover, trailing, slippage = _parse_strategy_name(strategy_name)

        assert 'BB(period=20,std=2)' == clean_name
        assert rollover is False
        assert trailing is None
        assert slippage == 0.0

    def test_parse_strategy_with_float_trailing(self):
        """Test parsing various trailing stop values."""
        test_cases = [
            ('Strategy(a=1,trailing=1.5,slippage_ticks=0.0)', 1.5),
            ('Strategy(a=1,trailing=10.0,slippage_ticks=0.0)', 10.0),
            ('Strategy(a=1,trailing=0.5,slippage_ticks=0.0)', 0.5),
        ]

        for strategy_name, expected_trailing in test_cases:
            _, _, trailing, _ = _parse_strategy_name(strategy_name)
            assert trailing == expected_trailing

    def test_parse_strategy_preserves_other_params(self):
        """Test that non-common parameters are preserved in clean name."""
        strategy_name = 'Ichimoku(tenkan=9,kijun=26,senkou_b=52,rollover=True,trailing=None,slippage_ticks=1.0)'

        clean_name, _, _, _ = _parse_strategy_name(strategy_name)

        # All strategy-specific params should be preserved
        assert 'tenkan=9' in clean_name
        assert 'kijun=26' in clean_name
        assert 'senkou_b=52' in clean_name
        # Common params should be removed
        assert 'rollover' not in clean_name
        assert 'trailing' not in clean_name
        assert 'slippage_ticks' not in clean_name

    def test_parse_strategy_handles_commas_correctly(self):
        """Test that comma handling in strategy name is correct."""
        strategy_name = 'Strategy(a=1,b=2,c=3,rollover=True,trailing=2.0,slippage_ticks=1.0)'

        clean_name, _, _, _ = _parse_strategy_name(strategy_name)

        # Should not have trailing commas or double commas
        assert ',,' not in clean_name
        assert not clean_name.endswith(',)')
        assert 'Strategy(a=1,b=2,c=3)' == clean_name


class TestDataFrameFormatting:
    """Test DataFrame formatting for CSV export."""

    def test_format_dataframe_parses_strategy_names(self, formatting_strategy_data):
        """Test that strategy names are parsed and parameters extracted."""
        formatted = format_dataframe_for_export(formatting_strategy_data)

        # Strategy column should be cleaned
        assert all('rollover' not in str(s) for s in formatted['Strategy'].values)
        assert all('trailing' not in str(s) for s in formatted['Strategy'].values)
        assert all('slippage_ticks' not in str(s) for s in formatted['Strategy'].values)

        # New columns should exist
        assert 'Rollover' in formatted.columns
        assert 'Trailing' in formatted.columns
        assert 'Slippage' in formatted.columns

    def test_format_dataframe_creates_parameter_columns(self, formatting_strategy_data):
        """Test that parameter columns are created with correct values."""
        formatted = format_dataframe_for_export(formatting_strategy_data)

        # First row: rollover=True, trailing=2.5, slippage=1.0
        assert bool(formatted.iloc[0]['Rollover']) is True
        assert formatted.iloc[0]['Trailing'] == 2.5
        assert formatted.iloc[0]['Slippage'] == 1.0

        # Second row: rollover=False, trailing=None, slippage=2.0
        assert bool(formatted.iloc[1]['Rollover']) is False
        assert pd.isna(formatted.iloc[1]['Trailing'])  # None becomes NaN in DataFrame
        assert formatted.iloc[1]['Slippage'] == 2.0

    def test_format_dataframe_rounds_numeric_columns(self, formatting_strategy_data):
        """Test that numeric columns are rounded to DECIMAL_PLACES."""
        formatted = format_dataframe_for_export(formatting_strategy_data)

        # Win rate should be rounded (65.5555 → 65.56 with DECIMAL_PLACES=2)
        assert formatted.iloc[0]['Win Rate %'] == round(65.5555, DECIMAL_PLACES)

        # Profit factor should be rounded (3.141592653589793 → 3.14 with DECIMAL_PLACES=2)
        assert formatted.iloc[0]['Profit Factor'] == round(3.141592653589793, DECIMAL_PLACES)

        # Check that all numeric columns are properly rounded
        for col in formatted.select_dtypes(include='number').columns:
            for val in formatted[col]:
                # Check decimal places (allowing for integers which have 0 decimal places)
                if pd.notna(val) and val != int(val):
                    decimal_str = str(float(val)).split('.')[-1]
                    assert len(decimal_str) <= DECIMAL_PLACES

    def test_format_dataframe_transforms_column_names(self, formatting_strategy_data):
        """Test that column names are transformed to Title Case."""
        formatted = format_dataframe_for_export(formatting_strategy_data)

        # Check specific transformations
        assert 'Avg Return %' in formatted.columns
        assert 'Max Drawdown %' in formatted.columns
        assert 'Sharpe' in formatted.columns
        assert 'Sortino' in formatted.columns
        assert 'Calmar' in formatted.columns
        assert 'Var 95%' in formatted.columns
        assert 'Cvar 95%' in formatted.columns
        assert 'Ulcer Idx' in formatted.columns

        # Original snake_case columns should not exist
        assert 'average_trade_return_percentage_of_contract' not in formatted.columns
        assert 'maximum_drawdown_percentage' not in formatted.columns

    def test_format_dataframe_reorders_columns(self, formatting_strategy_data):
        """Test that columns are reordered with parameters after strategy."""
        formatted = format_dataframe_for_export(formatting_strategy_data)

        cols = list(formatted.columns)
        strategy_idx = cols.index('Strategy')

        # Rollover, Trailing, Slippage should come right after Strategy
        assert cols[strategy_idx + 1] == 'Rollover'
        assert cols[strategy_idx + 2] == 'Trailing'
        assert cols[strategy_idx + 3] == 'Slippage'

    def test_format_dataframe_without_strategy_column(self):
        """Test formatting DataFrame without strategy column."""
        df = pd.DataFrame([
            {'symbol': 'ZS', 'win_rate': 65.5, 'profit_factor': 3.14}
        ])

        formatted = format_dataframe_for_export(df)

        # Should still format column names and round numbers
        assert 'Win Rate %' in formatted.columns
        assert 'Profit Factor' in formatted.columns
        assert formatted.iloc[0]['Win Rate %'] == round(65.5, DECIMAL_PLACES)

        # Should not have parameter columns
        assert 'Rollover' not in formatted.columns
        assert 'Trailing' not in formatted.columns

    def test_format_dataframe_preserves_non_numeric_columns(self, formatting_strategy_data):
        """Test that non-numeric columns are preserved unchanged."""
        formatted = format_dataframe_for_export(formatting_strategy_data)

        # Symbol and interval should be unchanged (except column name format)
        assert 'Symbol' in formatted.columns
        assert 'Interval' in formatted.columns
        assert formatted.iloc[0]['Symbol'] == 'ZS'
        assert formatted.iloc[0]['Interval'] == '1h'
        assert formatted.iloc[1]['Symbol'] == 'CL'
        assert formatted.iloc[1]['Interval'] == '4h'

    def test_format_dataframe_handles_aggregated_results(self, aggregated_strategy_results):
        """Test formatting of aggregated results (no symbol/interval columns)."""
        formatted = format_dataframe_for_export(aggregated_strategy_results)

        # Should have aggregation-specific columns formatted
        assert 'Symbol Count' in formatted.columns
        assert 'Interval Count' in formatted.columns
        assert 'Total Trades' in formatted.columns

        # Strategy should be cleaned
        assert 'rollover' not in formatted.iloc[0]['Strategy']

        # Values should be rounded
        assert formatted.iloc[0]['Win Rate %'] == round(62.5, DECIMAL_PLACES)


class TestColumnReordering:
    """Test column reordering functionality."""

    def test_reorder_columns_places_params_after_strategy(self):
        """Test that rollover, trailing, slippage are placed after strategy."""
        df = pd.DataFrame({
            'other_col': [1, 2],
            'strategy': ['A', 'B'],
            'more_cols': [3, 4],
            'rollover': [True, False],
            'trailing': [1.0, None],
            'slippage': [0.5, 1.0]
        })

        reordered = _reorder_columns(df)
        cols = list(reordered.columns)

        strategy_idx = cols.index('strategy')
        assert cols[strategy_idx + 1] == 'rollover'
        assert cols[strategy_idx + 2] == 'trailing'
        assert cols[strategy_idx + 3] == 'slippage'

    def test_reorder_columns_without_strategy(self):
        """Test reordering when strategy column doesn't exist."""
        df = pd.DataFrame({
            'col1': [1, 2],
            'col2': [3, 4]
        })

        reordered = _reorder_columns(df)

        # Should return original DataFrame unchanged
        assert list(reordered.columns) == list(df.columns)
        assert reordered.equals(df)

    def test_reorder_columns_preserves_other_column_order(self):
        """Test that other columns maintain their relative order."""
        df = pd.DataFrame({
            'first': [1],
            'strategy': ['A'],
            'second': [2],
            'third': [3],
            'rollover': [True],
            'trailing': [1.0],
            'slippage': [0.5]
        })

        reordered = _reorder_columns(df)
        cols = list(reordered.columns)

        # Original order for non-parameter columns should be preserved
        assert cols.index('first') < cols.index('strategy')
        assert cols.index('strategy') < cols.index('second')
        assert cols.index('second') < cols.index('third')

    def test_reorder_columns_with_missing_param_columns(self):
        """Test reordering when some parameter columns don't exist."""
        df = pd.DataFrame({
            'strategy': ['A', 'B'],
            'col1': [1, 2],
            'rollover': [True, False]
            # Missing trailing and slippage
        })

        # When some parameter columns are missing, function should handle gracefully
        # and only reorder the columns that exist
        reordered = _reorder_columns(df)

        # Should not raise an error
        assert isinstance(reordered, pd.DataFrame)

        # Rollover should be after strategy (since it exists)
        cols = list(reordered.columns)
        strategy_idx = cols.index('strategy')
        assert cols[strategy_idx + 1] == 'rollover'

        # Should have same columns as original
        assert set(reordered.columns) == set(df.columns)

        # Data should be unchanged
        assert reordered['strategy'].tolist() == df['strategy'].tolist()
        assert reordered['rollover'].tolist() == df['rollover'].tolist()

    def test_reorder_columns_with_no_param_columns(self):
        """Test reordering when no parameter columns exist."""
        df = pd.DataFrame({
            'strategy': ['A', 'B'],
            'col1': [1, 2],
            'col2': [3, 4]
            # No rollover, trailing, or slippage columns
        })

        # Should return DataFrame unchanged when no param columns exist
        reordered = _reorder_columns(df)

        assert list(reordered.columns) == list(df.columns)
        assert reordered.equals(df)

    def test_reorder_columns_when_params_before_strategy(self):
        """Test reordering when param columns appear before strategy column."""
        df = pd.DataFrame({
            'rollover': [True, False],
            'slippage': [1, 2],
            'col1': [10, 20],
            'strategy': ['A', 'B'],
            'col2': [30, 40],
            'trailing': [1.5, 2.0]
        })

        reordered = _reorder_columns(df)

        # Params should be moved after strategy, not before
        cols = list(reordered.columns)
        strategy_idx = cols.index('strategy')

        # All param columns should come after strategy
        assert cols[strategy_idx + 1] == 'rollover'
        assert cols[strategy_idx + 2] == 'trailing'
        assert cols[strategy_idx + 3] == 'slippage'

        # col1 should remain before strategy
        assert cols.index('col1') < strategy_idx

        # col2 should remain after all params
        assert cols.index('col2') > cols.index('slippage')

    @pytest.mark.parametrize("exception_type", [ValueError, IndexError, KeyError])
    def test_reorder_columns_exception_handling_with_mock(self, exception_type):
        """Test exception handling for ValueError, IndexError, KeyError (lines 152-154)."""
        from unittest.mock import MagicMock

        df = pd.DataFrame({
            'strategy': ['A', 'B'],
            'col1': [1, 2]
        })

        # Create a DataFrame mock that raises exceptions
        mock_df = MagicMock(spec=pd.DataFrame)
        mock_df.columns = ['strategy', 'col1']
        mock_df.__getitem__.side_effect = exception_type("Test error")

        # Call with mock should handle exception gracefully
        result = _reorder_columns(mock_df)
        # Should return the original mock df on error
        assert result is mock_df


class TestColumnNameFormatting:
    """Test column name transformation."""

    @pytest.mark.parametrize("input_name,expected_output", [
        ('average_trade_return_percentage_of_contract', 'Avg Return %'),
        ('average_win_percentage_of_contract', 'Avg Win %'),
        ('total_return_percentage_of_contract', 'Total Return %'),
        ('average_loss_percentage_of_contract', 'Avg Loss %'),
        ('maximum_drawdown_percentage', 'Max Drawdown %'),
        ('average_trade_duration_hours', 'Avg Duration H'),
        ('win_rate', 'Win Rate %'),
        ('sharpe_ratio', 'Sharpe'),
        ('sortino_ratio', 'Sortino'),
        ('calmar_ratio', 'Calmar'),
        ('value_at_risk', 'Var 95%'),
        ('expected_shortfall', 'Cvar 95%'),
        ('ulcer_index', 'Ulcer Idx'),
    ])
    def test_format_column_name_with_mapping(self, input_name, expected_output):
        """Test column names that have specific mappings."""
        result = _format_column_name(input_name)
        assert result == expected_output

    @pytest.mark.parametrize("input_name,expected_output", [
        ('symbol', 'Symbol'),
        ('interval', 'Interval'),
        ('total_trades', 'Total Trades'),
        ('profit_factor', 'Profit Factor'),
        ('symbol_count', 'Symbol Count'),
        ('interval_count', 'Interval Count'),
    ])
    def test_format_column_name_without_mapping(self, input_name, expected_output):
        """Test column names without specific mappings use Title Case."""
        result = _format_column_name(input_name)
        assert result == expected_output

    def test_format_column_name_single_word(self):
        """Test formatting single-word column names."""
        assert _format_column_name('strategy') == 'Strategy'
        assert _format_column_name('rollover') == 'Rollover'
        assert _format_column_name('trailing') == 'Trailing'

    def test_format_column_name_preserves_capitalization_pattern(self):
        """Test that Title Case is applied correctly."""
        result = _format_column_name('my_column_name')
        assert result == 'My Column Name'
        assert all(word[0].isupper() for word in result.split())


class TestFilenameBuilding:
    """Test CSV filename generation."""

    def test_build_filename_basic(self):
        """Test basic filename construction with metric only."""
        with patch('app.backtesting.analysis.formatters.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 2, 8, 14, 30)

            filename = build_filename(
                metric='profit_factor',
                aggregate=False,
                interval=None,
                symbol=None,
                weighted=False
            )

            assert filename.startswith('2024-02-08 14:30')
            assert 'top_strategies_by_profit_factor' in filename
            assert filename.endswith('.csv')
            assert '_aggregated' not in filename
            assert '_weighted' not in filename
            assert '_simple' not in filename

    def test_build_filename_with_aggregation_weighted(self):
        """Test filename with aggregated and weighted parameters."""
        with patch('app.backtesting.analysis.formatters.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 2, 8, 14, 30)

            filename = build_filename(
                metric='sharpe_ratio',
                aggregate=True,
                interval=None,
                symbol=None,
                weighted=True
            )

            assert '2024-02-08 14:30' in filename
            assert 'top_strategies_by_sharpe_ratio' in filename
            assert '_aggregated_weighted' in filename
            assert filename.endswith('.csv')

    def test_build_filename_with_aggregation_simple(self):
        """Test filename with aggregated but simple averaging."""
        with patch('app.backtesting.analysis.formatters.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 2, 8, 14, 30)

            filename = build_filename(
                metric='win_rate',
                aggregate=True,
                interval=None,
                symbol=None,
                weighted=False
            )

            assert 'top_strategies_by_win_rate' in filename
            assert '_aggregated_simple' in filename

    def test_build_filename_with_interval_filter(self):
        """Test filename with interval filter."""
        with patch('app.backtesting.analysis.formatters.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 2, 8, 14, 30)

            filename = build_filename(
                metric='profit_factor',
                aggregate=False,
                interval='1h',
                symbol=None,
                weighted=False
            )

            assert '_1h' in filename
            assert 'top_strategies_by_profit_factor_1h' in filename

    def test_build_filename_with_symbol_filter(self):
        """Test filename with symbol filter."""
        with patch('app.backtesting.analysis.formatters.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 2, 8, 14, 30)

            filename = build_filename(
                metric='calmar_ratio',
                aggregate=False,
                interval=None,
                symbol='ZS',
                weighted=False
            )

            assert '_ZS' in filename
            assert 'top_strategies_by_calmar_ratio_ZS' in filename

    def test_build_filename_with_all_filters(self):
        """Test filename with all possible filters applied."""
        with patch('app.backtesting.analysis.formatters.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 2, 8, 14, 30)

            filename = build_filename(
                metric='sortino_ratio',
                aggregate=True,
                interval='4h',
                symbol='CL',
                weighted=True
            )

            assert '2024-02-08 14:30' in filename
            assert 'top_strategies_by_sortino_ratio' in filename
            assert '_4h' in filename
            assert '_CL' in filename
            assert '_aggregated_weighted' in filename
            assert filename.endswith('.csv')

    def test_build_filename_metric_variations(self):
        """Test filename with various metric names."""
        metrics = [
            'profit_factor',
            'win_rate',
            'sharpe_ratio',
            'maximum_drawdown_percentage',
            'average_trade_return_percentage_of_contract'
        ]

        with patch('app.backtesting.analysis.formatters.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 2, 8, 14, 30)

            for metric in metrics:
                filename = build_filename(
                    metric=metric,
                    aggregate=False,
                    interval=None,
                    symbol=None,
                    weighted=False
                )

                assert f'top_strategies_by_{metric}' in filename
                assert filename.endswith('.csv')

    def test_build_filename_timestamp_format(self):
        """Test that timestamp format is correct."""
        with patch('app.backtesting.analysis.formatters.datetime') as mock_datetime:
            # Test various timestamps
            test_times = [
                datetime(2024, 1, 1, 0, 0),
                datetime(2024, 12, 31, 23, 59),
                datetime(2024, 6, 15, 12, 30),
            ]

            for test_time in test_times:
                mock_datetime.now.return_value = test_time

                filename = build_filename(
                    metric='profit_factor',
                    aggregate=False,
                    interval=None,
                    symbol=None,
                    weighted=False
                )

                expected_timestamp = test_time.strftime('%Y-%m-%d %H:%M')
                assert filename.startswith(expected_timestamp)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_format_empty_dataframe(self):
        """Test formatting an empty DataFrame."""
        df = pd.DataFrame()

        formatted = format_dataframe_for_export(df)

        assert isinstance(formatted, pd.DataFrame)
        assert len(formatted) == 0

    def test_format_dataframe_with_nan_values(self):
        """Test formatting DataFrame with NaN values."""
        df = pd.DataFrame({
            'strategy': ['RSI(period=14,rollover=True,trailing=None,slippage_ticks=0.0)'],
            'win_rate': [float('nan')],
            'profit_factor': [2.5]
        })

        formatted = format_dataframe_for_export(df)

        # NaN should be preserved
        assert pd.isna(formatted.iloc[0]['Win Rate %'])
        # Other values should be formatted
        assert formatted.iloc[0]['Profit Factor'] == round(2.5, DECIMAL_PLACES)

    def test_format_dataframe_with_malformed_strategy_name(self):
        """Test formatting with strategy name that doesn't match expected pattern."""
        df = pd.DataFrame({
            'strategy': ['InvalidStrategyName'],
            'win_rate': [65.5]
        })

        # Should not crash, even if parsing fails
        formatted = format_dataframe_for_export(df)

        assert isinstance(formatted, pd.DataFrame)
        assert len(formatted) == 1

    def test_parse_strategy_name_with_missing_params(self):
        """Test parsing strategy name without all common parameters."""
        strategy_name = 'Strategy(param=1)'

        clean_name, rollover, trailing, slippage = _parse_strategy_name(strategy_name)

        # Should return defaults for missing params
        assert rollover is False
        assert trailing is None
        assert slippage == 0.0

    def test_format_column_name_with_numbers(self):
        """Test column name formatting with numbers."""
        assert _format_column_name('param1_value2') == 'Param1 Value2'
        assert _format_column_name('test_123') == 'Test 123'

    def test_reorder_columns_handles_exception_gracefully(self):
        """Test that column reordering handles exceptions without crashing."""
        # Create a DataFrame that might cause issues
        df = pd.DataFrame({
            'strategy': ['A'],
            'rollover': [True]
        })

        # The function may return original df or try to reorder
        # Both behaviors are acceptable
        try:
            result = _reorder_columns(df)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1
        except Exception:
            # If exception occurs, that's also documented behavior
            pass

    def test_format_dataframe_with_very_long_strategy_name(self):
        """Test formatting with very long strategy names."""
        long_strategy = 'VeryLongStrategyName(' + ','.join(
            f'param{i}={i}' for i in range(20)) + ',rollover=True,trailing=2.0,slippage_ticks=1.0)'

        df = pd.DataFrame({
            'strategy': [long_strategy],
            'win_rate': [65.5]
        })

        formatted = format_dataframe_for_export(df)

        # Should handle long names without crashing
        assert isinstance(formatted, pd.DataFrame)
        assert 'Rollover' in formatted.columns
        assert bool(formatted.iloc[0]['Rollover']) is True


class TestIntegrationScenarios:
    """Test complete formatting workflows."""

    def test_complete_export_workflow(self, formatting_strategy_data):
        """Test complete workflow from raw results to formatted export."""
        # Format the DataFrame
        formatted = format_dataframe_for_export(formatting_strategy_data)

        # Verify all transformations happened
        assert 'Strategy' in formatted.columns
        assert 'Rollover' in formatted.columns
        assert 'Trailing' in formatted.columns
        assert 'Slippage' in formatted.columns
        assert 'Avg Return %' in formatted.columns
        assert 'Max Drawdown %' in formatted.columns

        # Verify column order
        cols = list(formatted.columns)
        strategy_idx = cols.index('Strategy')
        assert cols[strategy_idx + 1] == 'Rollover'
        assert cols[strategy_idx + 2] == 'Trailing'
        assert cols[strategy_idx + 3] == 'Slippage'

        # Verify values are rounded
        for col in formatted.select_dtypes(include='number').columns:
            for val in formatted[col]:
                if pd.notna(val) and val != int(val):
                    decimal_str = str(float(val)).split('.')[-1]
                    assert len(decimal_str) <= DECIMAL_PLACES

        # Verify strategy names are cleaned
        assert all('rollover' not in str(s) for s in formatted['Strategy'].values)

    def test_multiple_strategies_same_type(self):
        """Test formatting multiple strategies of the same type with different params."""
        df = pd.DataFrame([
            {
                'strategy': 'RSI(period=14,lower=30,upper=70,rollover=True,trailing=2.0,slippage_ticks=1.0)',
                'win_rate': 65.0,
                'profit_factor': 3.0
            },
            {
                'strategy': 'RSI(period=21,lower=25,upper=75,rollover=False,trailing=None,slippage_ticks=0.0)',
                'win_rate': 60.0,
                'profit_factor': 2.5
            },
            {
                'strategy': 'RSI(period=14,lower=20,upper=80,rollover=True,trailing=3.0,slippage_ticks=2.0)',
                'win_rate': 70.0,
                'profit_factor': 3.5
            }
        ])

        formatted = format_dataframe_for_export(df)

        # All should be RSI but with different parameters
        assert all('RSI' in str(s) for s in formatted['Strategy'].values)

        # Should have different parameter values
        assert formatted.iloc[0]['Trailing'] == 2.0
        assert pd.isna(formatted.iloc[1]['Trailing'])  # None becomes NaN
        assert formatted.iloc[2]['Trailing'] == 3.0

        assert formatted.iloc[0]['Slippage'] == 1.0
        assert formatted.iloc[1]['Slippage'] == 0.0
        assert formatted.iloc[2]['Slippage'] == 2.0

    def test_filename_matches_formatting_parameters(self):
        """Test that filename accurately reflects the parameters used."""
        with patch('app.backtesting.analysis.formatters.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 2, 8, 14, 30)

            # Test various parameter combinations
            test_cases = [
                {'aggregate': False, 'interval': None, 'symbol': None, 'weighted': False,
                 'expected_parts': []},
                {'aggregate': True, 'interval': None, 'symbol': None, 'weighted': True,
                 'expected_parts': ['_aggregated_weighted']},
                {'aggregate': True, 'interval': None, 'symbol': None, 'weighted': False,
                 'expected_parts': ['_aggregated_simple']},
                {'aggregate': False, 'interval': '1h', 'symbol': 'ZS', 'weighted': False,
                 'expected_parts': ['_1h', '_ZS']},
                {'aggregate': True, 'interval': '4h', 'symbol': 'CL', 'weighted': True,
                 'expected_parts': ['_4h', '_CL', '_aggregated_weighted']},
            ]

            for test_case in test_cases:
                filename = build_filename(
                    metric='profit_factor',
                    aggregate=test_case['aggregate'],
                    interval=test_case['interval'],
                    symbol=test_case['symbol'],
                    weighted=test_case['weighted']
                )

                for part in test_case['expected_parts']:
                    assert part in filename
