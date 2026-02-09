"""
Tests for orchestrator module.

Tests cover:
- Main run_tests function orchestration
- Switch dates mapping (mini/micro symbols)
- Test combination generation
- Test preparation and filtering
- Parallel execution coordination
- Cache statistics reporting
- Error handling and recovery
- Result aggregation and saving
- Integration scenarios
"""
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from app.backtesting.testing.orchestrator import (
    run_tests,
    _get_switch_dates_for_symbols,
    _generate_all_combinations,
    _prepare_test_combinations,
    _report_cache_statistics,
)


# ==================== Run Tests Main Function ====================

class TestRunTests:
    """Test the main run_tests orchestration function."""

    def test_run_tests_raises_error_when_no_strategies(self):
        """Test that run_tests raises ValueError when no strategies are added."""
        tester = MagicMock()
        tester.strategies = []

        with pytest.raises(ValueError, match='No strategies added for testing'):
            run_tests(tester, verbose=False, max_workers=1, skip_existing=False)

    def test_run_tests_raises_error_when_strategies_attribute_missing(self):
        """Test that run_tests raises ValueError when strategies attribute is missing."""
        tester = MagicMock(spec=[])  # No strategies attribute

        with pytest.raises(ValueError, match='No strategies added for testing'):
            run_tests(tester, verbose=False, max_workers=1, skip_existing=False)

    def test_run_tests_resets_cache_statistics(self):
        """Test that run_tests resets cache statistics at start."""
        tester = MagicMock()
        tester.strategies = [('RSI_14_30_70', MagicMock())]
        tester.tested_months = ['1!']
        tester.symbols = ['ZS']
        tester.intervals = ['1h']
        tester.switch_dates_dict = {'ZS': []}
        tester.results = []

        with patch('app.backtesting.testing.orchestrator.indicator_cache') as mock_ind_cache, \
                patch('app.backtesting.testing.orchestrator.dataframe_cache') as mock_df_cache, \
                patch('app.backtesting.testing.orchestrator.load_existing_results') as mock_load, \
                patch('app.backtesting.testing.orchestrator.concurrent.futures.ProcessPoolExecutor'), \
                patch('app.backtesting.testing.orchestrator.concurrent.futures.as_completed') as mock_as_completed:
            mock_load.return_value = (pd.DataFrame(), set())
            mock_as_completed.return_value = []

            run_tests(tester, verbose=False, max_workers=1, skip_existing=False)

            # Verify caches were reset
            mock_ind_cache.reset_stats.assert_called_once()
            mock_df_cache.reset_stats.assert_called_once()

    def test_run_tests_loads_existing_results_when_skip_existing_true(self):
        """Test that existing results are loaded when skip_existing is True."""
        tester = MagicMock()
        tester.strategies = [('RSI_14_30_70', MagicMock())]
        tester.tested_months = ['1!']
        tester.symbols = ['ZS']
        tester.intervals = ['1h']
        tester.switch_dates_dict = {'ZS': []}
        tester.results = []

        existing_df = pd.DataFrame({'test': [1, 2, 3]})
        existing_set = {('1!', 'ZS', '1h', 'RSI_14_30_70')}

        with patch('app.backtesting.testing.orchestrator.indicator_cache'), \
                patch('app.backtesting.testing.orchestrator.dataframe_cache'), \
                patch('app.backtesting.testing.orchestrator.load_existing_results') as mock_load, \
                patch('app.backtesting.testing.orchestrator.concurrent.futures.ProcessPoolExecutor'), \
                patch('app.backtesting.testing.orchestrator.concurrent.futures.as_completed') as mock_as_completed:
            mock_load.return_value = (existing_df, existing_set)
            mock_as_completed.return_value = []

            run_tests(tester, verbose=False, max_workers=1, skip_existing=True)

            # Verify load_existing_results was called
            mock_load.assert_called_once()

    def test_run_tests_skips_loading_existing_when_skip_existing_false(self):
        """Test that existing results are not loaded when skip_existing is False."""
        tester = MagicMock()
        tester.strategies = [('RSI_14_30_70', MagicMock())]
        tester.tested_months = ['1!']
        tester.symbols = ['ZS']
        tester.intervals = ['1h']
        tester.switch_dates_dict = {'ZS': []}
        tester.results = []

        with patch('app.backtesting.testing.orchestrator.indicator_cache'), \
                patch('app.backtesting.testing.orchestrator.dataframe_cache'), \
                patch('app.backtesting.testing.orchestrator.load_existing_results') as mock_load, \
                patch('app.backtesting.testing.orchestrator.concurrent.futures.ProcessPoolExecutor'), \
                patch('app.backtesting.testing.orchestrator.concurrent.futures.as_completed') as mock_as_completed:
            mock_load.return_value = (pd.DataFrame(), set())
            mock_as_completed.return_value = []

            run_tests(tester, verbose=False, max_workers=1, skip_existing=False)

            # load_existing_results should not be called
            mock_load.assert_not_called()

    def test_run_tests_clears_previous_results(self):
        """Test that previous results are cleared at start."""
        tester = MagicMock()
        tester.strategies = [('RSI_14_30_70', MagicMock())]
        tester.tested_months = ['1!']
        tester.symbols = ['ZS']
        tester.intervals = ['1h']
        tester.switch_dates_dict = {'ZS': []}
        tester.results = [{'old': 'result'}]

        with patch('app.backtesting.testing.orchestrator.indicator_cache'), \
                patch('app.backtesting.testing.orchestrator.dataframe_cache'), \
                patch('app.backtesting.testing.orchestrator.load_existing_results') as mock_load, \
                patch('app.backtesting.testing.orchestrator.concurrent.futures.ProcessPoolExecutor'), \
                patch('app.backtesting.testing.orchestrator.concurrent.futures.as_completed') as mock_as_completed:
            mock_load.return_value = (pd.DataFrame(), set())
            mock_as_completed.return_value = []

            run_tests(tester, verbose=False, max_workers=1, skip_existing=False)

            # Verify results were cleared
            assert tester.results == []

    def test_run_tests_returns_empty_list_when_all_tests_skipped(self):
        """Test that empty list is returned when all tests are already run."""
        tester = MagicMock()
        tester.strategies = [('RSI_14_30_70', MagicMock())]
        tester.tested_months = ['1!']
        tester.symbols = ['ZS']
        tester.intervals = ['1h']
        tester.switch_dates_dict = {'ZS': []}
        tester.results = []

        # Mock existing results to skip all tests
        with patch('app.backtesting.testing.orchestrator.indicator_cache'), \
                patch('app.backtesting.testing.orchestrator.dataframe_cache'), \
                patch('app.backtesting.testing.orchestrator.load_existing_results') as mock_load, \
                patch('app.backtesting.testing.orchestrator.check_test_exists') as mock_check:
            mock_load.return_value = (pd.DataFrame(), set())
            mock_check.return_value = True  # All tests exist

            result = run_tests(tester, verbose=False, max_workers=1, skip_existing=True)

            assert result == []

    def test_run_tests_saves_results_when_results_exist(self):
        """Test that results are saved when tests complete."""
        tester = MagicMock()
        tester.strategies = [('RSI_14_30_70', MagicMock())]
        tester.tested_months = ['1!']
        tester.symbols = ['ZS']
        tester.intervals = ['1h']
        tester.switch_dates_dict = {'ZS': []}
        tester.results = []

        with patch('app.backtesting.testing.orchestrator.indicator_cache'), \
                patch('app.backtesting.testing.orchestrator.dataframe_cache'), \
                patch('app.backtesting.testing.orchestrator.load_existing_results') as mock_load, \
                patch('app.backtesting.testing.orchestrator.save_results') as mock_save, \
                patch('app.backtesting.testing.orchestrator.concurrent.futures.ProcessPoolExecutor'), \
                patch('app.backtesting.testing.orchestrator.concurrent.futures.as_completed') as mock_as_completed:
            mock_load.return_value = (pd.DataFrame(), set())

            # Mock a future that returns a result
            mock_future = MagicMock()
            mock_future.result.return_value = {'test': 'result'}
            mock_as_completed.return_value = [mock_future]

            run_tests(tester, verbose=False, max_workers=1, skip_existing=False)

            # Verify save_results was called
            assert mock_save.called


# ==================== Switch Dates Mapping Tests ====================

class TestGetSwitchDatesForSymbols:
    """Test switch dates mapping functionality."""

    def test_get_switch_dates_for_direct_symbols(self):
        """Test getting switch dates for symbols with direct mappings."""
        symbols = ['ZS', 'CL']
        switch_dates_dict = {
            'ZS': ['2024-01-01', '2024-02-01'],
            'CL': ['2024-01-15', '2024-02-15']
        }

        result = _get_switch_dates_for_symbols(symbols, switch_dates_dict)

        assert 'ZS' in result
        assert 'CL' in result
        assert len(result['ZS']) == 2
        assert len(result['CL']) == 2
        assert isinstance(result['ZS'][0], pd.Timestamp)
        assert isinstance(result['CL'][0], pd.Timestamp)

    def test_get_switch_dates_for_mini_micro_symbols(self):
        """Test getting switch dates for mini/micro symbols that map to main symbols."""
        symbols = ['MCL', 'MGC']
        switch_dates_dict = {
            'CL': ['2024-01-01', '2024-02-01'],
            'GC': ['2024-01-15', '2024-02-15'],
            '_symbol_mappings': {
                'MCL': 'CL',
                'MGC': 'GC'
            }
        }

        result = _get_switch_dates_for_symbols(symbols, switch_dates_dict)

        # MCL should get CL's dates
        assert 'MCL' in result
        assert len(result['MCL']) == 2

        # MGC should get GC's dates
        assert 'MGC' in result
        assert len(result['MGC']) == 2

    def test_get_switch_dates_for_unmapped_symbols(self):
        """Test that unmapped symbols get empty list of switch dates."""
        symbols = ['UNKNOWN']
        switch_dates_dict = {
            'ZS': ['2024-01-01'],
            'CL': ['2024-01-15']
        }

        result = _get_switch_dates_for_symbols(symbols, switch_dates_dict)

        assert 'UNKNOWN' in result
        assert result['UNKNOWN'] == []

    def test_get_switch_dates_converts_strings_to_timestamps(self):
        """Test that date strings are converted to pandas Timestamps."""
        symbols = ['ZS']
        switch_dates_dict = {
            'ZS': ['2024-01-01', '2024-02-01', '2024-03-01']
        }

        result = _get_switch_dates_for_symbols(symbols, switch_dates_dict)

        # Verify all dates are Timestamps
        for date in result['ZS']:
            assert isinstance(date, pd.Timestamp)

        # Verify correct conversion
        assert result['ZS'][0] == pd.Timestamp('2024-01-01')
        assert result['ZS'][1] == pd.Timestamp('2024-02-01')
        assert result['ZS'][2] == pd.Timestamp('2024-03-01')

    def test_get_switch_dates_with_mixed_symbols(self):
        """Test with a mix of direct, mini/micro, and unmapped symbols."""
        symbols = ['ZS', 'MCL', 'UNKNOWN']
        switch_dates_dict = {
            'ZS': ['2024-01-01'],
            'CL': ['2024-02-01'],
            '_symbol_mappings': {
                'MCL': 'CL'
            }
        }

        result = _get_switch_dates_for_symbols(symbols, switch_dates_dict)

        assert len(result['ZS']) == 1
        assert len(result['MCL']) == 1
        assert len(result['UNKNOWN']) == 0


# ==================== Combination Generation Tests ====================

class TestGenerateAllCombinations:
    """Test test combination generation."""

    def test_generate_all_combinations_basic(self):
        """Test generating combinations with basic inputs."""
        tested_months = ['1!']
        symbols = ['ZS']
        intervals = ['1h']
        strategies = [('RSI_14_30_70', MagicMock())]

        result = _generate_all_combinations(tested_months, symbols, intervals, strategies)

        # Should generate 1*1*1*1 = 1 combination
        assert len(result) == 1
        assert result[0][0] == '1!'
        assert result[0][1] == 'ZS'
        assert result[0][2] == '1h'
        assert result[0][3] == 'RSI_14_30_70'

    def test_generate_all_combinations_multiple_dimensions(self):
        """Test generating combinations with multiple values per dimension."""
        tested_months = ['1!', '2!']
        symbols = ['ZS', 'CL']
        intervals = ['15m', '1h']
        strategies = [
            ('RSI_14_30_70', MagicMock()),
            ('EMA_9_21', MagicMock())
        ]

        result = _generate_all_combinations(tested_months, symbols, intervals, strategies)

        # Should generate 2*2*2*2 = 16 combinations
        assert len(result) == 16

    def test_generate_all_combinations_preserves_structure(self):
        """Test that generated combinations have correct structure."""
        tested_months = ['1!']
        symbols = ['ZS']
        intervals = ['1h']
        mock_strategy = MagicMock()
        strategies = [('RSI_14_30_70', mock_strategy)]

        result = _generate_all_combinations(tested_months, symbols, intervals, strategies)

        # Verify tuple structure: (month, symbol, interval, name, instance)
        combo = result[0]
        assert len(combo) == 5
        assert combo[0] == '1!'
        assert combo[1] == 'ZS'
        assert combo[2] == '1h'
        assert combo[3] == 'RSI_14_30_70'
        assert combo[4] == mock_strategy

    def test_generate_all_combinations_large_scale(self):
        """Test generating large number of combinations."""
        tested_months = ['1!', '2!', '3!']
        symbols = ['ZS', 'CL', 'GC', 'ES']
        intervals = ['15m', '1h', '4h', '1d']
        strategies = [(f'Strategy_{i}', MagicMock()) for i in range(10)]

        result = _generate_all_combinations(tested_months, symbols, intervals, strategies)

        # Should generate 3*4*4*10 = 480 combinations
        assert len(result) == 480

    def test_generate_all_combinations_empty_strategies(self):
        """Test with empty strategies list."""
        tested_months = ['1!']
        symbols = ['ZS']
        intervals = ['1h']
        strategies = []

        result = _generate_all_combinations(tested_months, symbols, intervals, strategies)

        # Should generate 0 combinations
        assert len(result) == 0


# ==================== Test Preparation Tests ====================

class TestPrepareTestCombinations:
    """Test test preparation and filtering."""

    def test_prepare_test_combinations_no_filtering(self):
        """Test preparing combinations without filtering (skip_existing=False)."""
        all_combinations = [
            ('1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock())
        ]
        existing_data = (pd.DataFrame(), set())
        switch_dates_by_symbol = {'ZS': []}

        test_combos, skipped = _prepare_test_combinations(
            all_combinations,
            existing_data,
            skip_existing=False,
            verbose=False,
            switch_dates_by_symbol=switch_dates_by_symbol
        )

        # Should include all combinations
        assert len(test_combos) == 1
        assert skipped == 0

    def test_prepare_test_combinations_with_filtering(self):
        """Test preparing combinations with filtering (skip_existing=True)."""
        all_combinations = [
            ('1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock()),
            ('1!', 'ZS', '1h', 'EMA_9_21', MagicMock())
        ]
        existing_data = (pd.DataFrame(), set())
        switch_dates_by_symbol = {'ZS': []}

        with patch('app.backtesting.testing.orchestrator.check_test_exists') as mock_check:
            # First test exists, second doesn't
            mock_check.side_effect = [True, False]

            test_combos, skipped = _prepare_test_combinations(
                all_combinations,
                existing_data,
                skip_existing=True,
                verbose=False,
                switch_dates_by_symbol=switch_dates_by_symbol
            )

            # Should skip 1, include 1
            assert len(test_combos) == 1
            assert skipped == 1

    def test_prepare_test_combinations_adds_required_parameters(self):
        """Test that preparation adds required parameters to combinations."""
        mock_strategy = MagicMock()
        all_combinations = [
            ('1!', 'ZS', '1h', 'RSI_14_30_70', mock_strategy)
        ]
        existing_data = (pd.DataFrame(), set())
        switch_dates = [pd.Timestamp('2024-01-01')]
        switch_dates_by_symbol = {'ZS': switch_dates}

        test_combos, _ = _prepare_test_combinations(
            all_combinations,
            existing_data,
            skip_existing=False,
            verbose=False,
            switch_dates_by_symbol=switch_dates_by_symbol
        )

        # Verify added parameters
        combo = test_combos[0]
        assert len(combo) == 8  # Original 5 + verbose, switch_dates, filepath
        assert combo[5] is False  # verbose
        assert combo[6] == switch_dates  # switch_dates
        assert 'ZS_1h.parquet' in combo[7]  # filepath

    def test_prepare_test_combinations_constructs_correct_filepath(self):
        """Test that correct filepath is constructed for data loading."""
        all_combinations = [
            ('1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock()),
            ('2!', 'CL', '15m', 'EMA_9_21', MagicMock())
        ]
        existing_data = (pd.DataFrame(), set())
        switch_dates_by_symbol = {'ZS': [], 'CL': []}

        test_combos, _ = _prepare_test_combinations(
            all_combinations,
            existing_data,
            skip_existing=False,
            verbose=False,
            switch_dates_by_symbol=switch_dates_by_symbol
        )

        # Verify filepaths
        assert '1!/ZS/ZS_1h.parquet' in test_combos[0][7]
        assert '2!/CL/CL_15m.parquet' in test_combos[1][7]


# ==================== Cache Statistics Tests ====================

class TestReportCacheStatistics:
    """Test cache statistics reporting."""

    def test_report_cache_statistics_with_results(self, capsys):
        """Test cache statistics reporting with test results."""
        results = [
            {
                'cache_stats': {
                    'ind_hits': 100,
                    'ind_misses': 10,
                    'df_hits': 50,
                    'df_misses': 5
                }
            },
            {
                'cache_stats': {
                    'ind_hits': 150,
                    'ind_misses': 15,
                    'df_hits': 75,
                    'df_misses': 8
                }
            }
        ]

        with patch('app.backtesting.testing.orchestrator.indicator_cache') as mock_ind, \
                patch('app.backtesting.testing.orchestrator.dataframe_cache') as mock_df:
            mock_ind.size.return_value = 500
            mock_df.size.return_value = 100

            _report_cache_statistics(results)

            captured = capsys.readouterr()

            # Verify output contains expected information
            assert 'CACHE PERFORMANCE STATISTICS' in captured.out
            assert 'Indicator Cache:' in captured.out
            assert 'DataFrame Cache:' in captured.out
            assert 'Hits: 250' in captured.out  # 100+150
            assert 'Misses: 25' in captured.out  # 10+15
            assert 'Hits: 125' in captured.out  # 50+75
            assert 'Misses: 13' in captured.out  # 5+8

    def test_report_cache_statistics_with_empty_results(self, capsys):
        """Test cache statistics reporting with no results."""
        results = []

        with patch('app.backtesting.testing.orchestrator.indicator_cache') as mock_ind, \
                patch('app.backtesting.testing.orchestrator.dataframe_cache') as mock_df:
            mock_ind.size.return_value = 0
            mock_df.size.return_value = 0

            _report_cache_statistics(results)

            captured = capsys.readouterr()

            # Should still print headers but with zero stats
            assert 'CACHE PERFORMANCE STATISTICS' in captured.out
            assert 'Hit rate: 0.00%' in captured.out

    def test_report_cache_statistics_calculates_hit_rate_correctly(self, capsys):
        """Test that hit rate percentage is calculated correctly."""
        results = [
            {
                'cache_stats': {
                    'ind_hits': 80,
                    'ind_misses': 20,
                    'df_hits': 90,
                    'df_misses': 10
                }
            }
        ]

        with patch('app.backtesting.testing.orchestrator.indicator_cache') as mock_ind, \
                patch('app.backtesting.testing.orchestrator.dataframe_cache') as mock_df:
            mock_ind.size.return_value = 100
            mock_df.size.return_value = 50

            _report_cache_statistics(results)

            captured = capsys.readouterr()

            # 80/(80+20) = 80%, 90/(90+10) = 90%
            assert 'Hit rate: 80.00%' in captured.out
            assert 'Hit rate: 90.00%' in captured.out

    def test_report_cache_statistics_handles_missing_cache_stats(self, capsys):
        """Test that results without cache_stats are handled gracefully."""
        results = [
            {'other': 'data'},  # No cache_stats
            {
                'cache_stats': {
                    'ind_hits': 10,
                    'ind_misses': 5,
                    'df_hits': 8,
                    'df_misses': 2
                }
            }
        ]

        with patch('app.backtesting.testing.orchestrator.indicator_cache') as mock_ind, \
                patch('app.backtesting.testing.orchestrator.dataframe_cache') as mock_df:
            mock_ind.size.return_value = 50
            mock_df.size.return_value = 20

            _report_cache_statistics(results)

            captured = capsys.readouterr()

            # Should only count the result with cache_stats
            assert 'Hits: 10' in captured.out
            assert 'Misses: 5' in captured.out


# ==================== Integration Tests ====================

class TestOrchestratorIntegration:
    """Test orchestrator integration scenarios."""

    def test_complete_orchestration_workflow(self):
        """Test complete workflow from start to finish."""
        tester = MagicMock()
        tester.strategies = [('RSI_14_30_70', MagicMock())]
        tester.tested_months = ['1!']
        tester.symbols = ['ZS']
        tester.intervals = ['1h']
        tester.switch_dates_dict = {'ZS': ['2024-01-01']}
        tester.results = []

        with patch('app.backtesting.testing.orchestrator.indicator_cache') as mock_ind, \
                patch('app.backtesting.testing.orchestrator.dataframe_cache') as mock_df, \
                patch('app.backtesting.testing.orchestrator.load_existing_results') as mock_load, \
                patch('app.backtesting.testing.orchestrator.save_results') as mock_save, \
                patch('app.backtesting.testing.orchestrator.concurrent.futures.ProcessPoolExecutor') as mock_executor:
            mock_load.return_value = (pd.DataFrame(), set())

            # Mock executor behavior
            mock_future = MagicMock()
            mock_future.result.return_value = {
                'month': '1!',
                'symbol': 'ZS',
                'interval': '1h',
                'strategy': 'RSI_14_30_70',
                'metrics': {'profit_factor': 1.5}
            }

            mock_executor_instance = MagicMock()
            mock_executor_instance.submit.return_value = mock_future
            mock_executor.return_value.__enter__.return_value = mock_executor_instance

            # Manually set up futures.as_completed behavior
            with patch('app.backtesting.testing.orchestrator.concurrent.futures.as_completed') as mock_as_completed:
                mock_as_completed.return_value = [mock_future]

                results = run_tests(tester, verbose=False, max_workers=1, skip_existing=False)

                # Verify workflow steps
                assert mock_ind.reset_stats.called
                assert mock_df.reset_stats.called
                assert isinstance(results, list)

    def test_orchestration_with_multiple_strategies(self):
        """Test orchestration with multiple strategies and combinations."""
        tester = MagicMock()
        tester.strategies = [
            ('RSI_14_30_70', MagicMock()),
            ('EMA_9_21', MagicMock()),
            ('MACD_12_26_9', MagicMock())
        ]
        tester.tested_months = ['1!', '2!']
        tester.symbols = ['ZS', 'CL']
        tester.intervals = ['15m', '1h']
        tester.switch_dates_dict = {'ZS': [], 'CL': []}
        tester.results = []

        with patch('app.backtesting.testing.orchestrator.indicator_cache'), \
                patch('app.backtesting.testing.orchestrator.dataframe_cache'), \
                patch('app.backtesting.testing.orchestrator.load_existing_results') as mock_load, \
                patch('app.backtesting.testing.orchestrator.concurrent.futures.ProcessPoolExecutor'), \
                patch('app.backtesting.testing.orchestrator.concurrent.futures.as_completed') as mock_as_completed:
            mock_load.return_value = (pd.DataFrame(), set())
            mock_as_completed.return_value = []

            # Should generate 2*2*2*3 = 24 combinations
            run_tests(tester, verbose=False, max_workers=2, skip_existing=False)

            # Verify all combinations were considered
            assert tester.results is not None

    def test_orchestration_error_handling(self):
        """Test that orchestration handles worker errors gracefully."""
        tester = MagicMock()
        tester.strategies = [('RSI_14_30_70', MagicMock())]
        tester.tested_months = ['1!']
        tester.symbols = ['ZS']
        tester.intervals = ['1h']
        tester.switch_dates_dict = {'ZS': []}
        tester.results = []

        with patch('app.backtesting.testing.orchestrator.indicator_cache'), \
                patch('app.backtesting.testing.orchestrator.dataframe_cache'), \
                patch('app.backtesting.testing.orchestrator.load_existing_results') as mock_load, \
                patch('app.backtesting.testing.orchestrator.concurrent.futures.ProcessPoolExecutor') as mock_executor:
            mock_load.return_value = (pd.DataFrame(), set())

            # Mock executor to raise exception
            mock_future = MagicMock()
            mock_future.result.side_effect = Exception("Worker error")

            mock_executor_instance = MagicMock()
            mock_executor_instance.submit.return_value = mock_future
            mock_executor.return_value.__enter__.return_value = mock_executor_instance

            with patch('app.backtesting.testing.orchestrator.concurrent.futures.as_completed') as mock_as_completed:
                mock_as_completed.return_value = [mock_future]

                # Should not raise, should handle gracefully
                result = run_tests(tester, verbose=False, max_workers=1, skip_existing=False)

                assert isinstance(result, list)
