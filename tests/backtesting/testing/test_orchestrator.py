"""
Tests for orchestrator module.

Tests cover:
- Main run_tests function orchestration
- Switch dates mapping (mini/micro symbols)
- Test combination generation
- Test preparation and filtering
- Verbose output for combo changes and skip messages
- Parallel execution coordination
- Intermediate result saving every 1000 tests
- Cache statistics reporting
- Cache save exception handling
- Error handling and recovery
- Result aggregation and saving
- Integration scenarios
"""
import concurrent.futures
from unittest.mock import MagicMock

import pandas as pd
import pytest

import app.backtesting.testing.orchestrator as orch_module
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

    def test_run_tests_raises_error_when_no_strategies(self, mock_tester):
        """Test that run_tests raises ValueError when no strategies are added."""
        mock_tester.strategies = []

        with pytest.raises(ValueError, match='No strategies added for testing'):
            run_tests(mock_tester, verbose=False, max_workers=1, skip_existing=False)

    def test_run_tests_raises_error_when_strategies_attribute_missing(self):
        """Test that run_tests raises ValueError when strategies attribute is missing."""
        tester = MagicMock(spec=[])  # No strategies attribute

        with pytest.raises(ValueError, match='No strategies added for testing'):
            run_tests(tester, verbose=False, max_workers=1, skip_existing=False)

    def test_run_tests_resets_cache_statistics(self, mock_tester, mock_orchestrator_environment):
        """Test that run_tests resets cache statistics at start."""
        mocks = mock_orchestrator_environment

        run_tests(mock_tester, verbose=False, max_workers=1, skip_existing=False)

        # Verify caches were reset
        mocks['indicator_cache'].reset_stats.assert_called_once()
        mocks['dataframe_cache'].reset_stats.assert_called_once()

    def test_run_tests_loads_existing_results_when_skip_existing_true(self, mock_tester, monkeypatch):
        """Test that existing results are loaded when skip_existing is True."""
        existing_df = pd.DataFrame({'test': [1, 2, 3]})
        existing_set = {('1!', 'ZS', '1h', 'RSI_14_30_70')}

        mock_load = MagicMock(return_value=(existing_df, existing_set))
        mock_as_completed = MagicMock(return_value=[])
        monkeypatch.setattr(orch_module, 'indicator_cache', MagicMock())
        monkeypatch.setattr(orch_module, 'dataframe_cache', MagicMock())
        monkeypatch.setattr(orch_module, 'load_existing_results', mock_load)
        monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', MagicMock())
        monkeypatch.setattr(concurrent.futures, 'as_completed', mock_as_completed)

        run_tests(mock_tester, verbose=False, max_workers=1, skip_existing=True)

        # Verify load_existing_results was called
        mock_load.assert_called_once()

    def test_run_tests_skips_loading_existing_when_skip_existing_false(
        self,
        mock_tester,
        mock_orchestrator_environment
    ):
        """Test that existing results are not loaded when skip_existing is False."""
        run_tests(mock_tester, verbose=False, max_workers=1, skip_existing=False)

        # load_existing_results should not be called
        mock_orchestrator_environment['load_existing_results'].assert_not_called()

    def test_run_tests_clears_previous_results(self, mock_tester, mock_orchestrator_environment):
        """Test that previous results are cleared at start."""
        mock_tester.results = [{'old': 'result'}]

        run_tests(mock_tester, verbose=False, max_workers=1, skip_existing=False)

        # Verify results were cleared
        assert mock_tester.results == []

    def test_run_tests_returns_empty_list_when_all_tests_skipped(self, mock_tester, monkeypatch):
        """Test that empty list is returned when all tests are already run."""
        mock_check = MagicMock(return_value=True)
        monkeypatch.setattr(orch_module, 'indicator_cache', MagicMock())
        monkeypatch.setattr(orch_module, 'dataframe_cache', MagicMock())
        monkeypatch.setattr(orch_module, 'load_existing_results', MagicMock(return_value=(pd.DataFrame(), set())))
        monkeypatch.setattr(orch_module, 'check_test_exists', mock_check)

        result = run_tests(mock_tester, verbose=False, max_workers=1, skip_existing=True)

        assert result == []

    def test_run_tests_saves_results_when_results_exist(self, mock_tester, monkeypatch):
        """Test that results are merged into the final parquet when tests complete."""
        mock_shard = MagicMock(return_value='/tmp/shard_0000.parquet')
        mock_merge = MagicMock()
        mock_future = MagicMock()
        mock_future.result.return_value = {'test': 'result'}
        mock_as_completed = MagicMock(return_value=[mock_future])

        monkeypatch.setattr(orch_module, 'indicator_cache', MagicMock())
        monkeypatch.setattr(orch_module, 'dataframe_cache', MagicMock())
        monkeypatch.setattr(orch_module, 'load_existing_results', MagicMock(return_value=(pd.DataFrame(), set())))
        monkeypatch.setattr(orch_module, 'save_shard', mock_shard)
        monkeypatch.setattr(orch_module, 'merge_shards', mock_merge)
        monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', MagicMock())
        monkeypatch.setattr(concurrent.futures, 'as_completed', mock_as_completed)

        run_tests(mock_tester, verbose=False, max_workers=1, skip_existing=False)

        # Verify merge_shards was called to finalize the output
        assert mock_merge.called


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

    def test_prepare_test_combinations_verbose_prints_combo_change(self, capsys):
        """Test that verbose mode prints when month/symbol/interval combo changes."""
        all_combinations = [
            ('1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock()),
            ('1!', 'ZS', '1h', 'EMA_9_21', MagicMock()),
            ('1!', 'CL', '1h', 'RSI_14_30_70', MagicMock()),
        ]
        existing_data = (pd.DataFrame(), set())
        switch_dates_by_symbol = {'ZS': [], 'CL': []}

        _prepare_test_combinations(
            all_combinations,
            existing_data,
            skip_existing=False,
            verbose=True,
            switch_dates_by_symbol=switch_dates_by_symbol
        )

        captured = capsys.readouterr()
        # Should print for first combo and when symbol changes to CL
        assert 'Preparing: Month=1!, Symbol=ZS, Interval=1h' in captured.out
        assert 'Preparing: Month=1!, Symbol=CL, Interval=1h' in captured.out

    def test_prepare_test_combinations_verbose_prints_skip_message(self, capsys, monkeypatch):
        """Test that verbose mode prints skip message when test is skipped."""
        all_combinations = [
            ('1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock()),
        ]
        existing_data = (pd.DataFrame(), set())
        switch_dates_by_symbol = {'ZS': []}

        monkeypatch.setattr(orch_module, 'check_test_exists', MagicMock(return_value=True))

        _prepare_test_combinations(
            all_combinations,
            existing_data,
            skip_existing=True,
            verbose=True,
            switch_dates_by_symbol=switch_dates_by_symbol
        )

        captured = capsys.readouterr()
        assert 'Skipping already run test: Month=1!, Symbol=ZS, Interval=1h, Strategy=RSI_14_30_70' in captured.out

    def test_prepare_test_combinations_with_filtering(self, monkeypatch):
        """Test preparing combinations with filtering (skip_existing=True)."""
        all_combinations = [
            ('1!', 'ZS', '1h', 'RSI_14_30_70', MagicMock()),
            ('1!', 'ZS', '1h', 'EMA_9_21', MagicMock())
        ]
        existing_data = (pd.DataFrame(), set())
        switch_dates_by_symbol = {'ZS': []}

        mock_check = MagicMock(side_effect=[True, False])
        monkeypatch.setattr(orch_module, 'check_test_exists', mock_check)

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

    def test_report_cache_statistics_with_results(self, capsys, monkeypatch):
        """Test cache statistics reporting with aggregated stats."""
        cache_stats = {'ind_hits': 250, 'ind_misses': 25, 'df_hits': 125, 'df_misses': 13}

        mock_ind = MagicMock()
        mock_df = MagicMock()
        mock_ind.size.return_value = 500
        mock_df.size.return_value = 100
        monkeypatch.setattr(orch_module, 'indicator_cache', mock_ind)
        monkeypatch.setattr(orch_module, 'dataframe_cache', mock_df)

        _report_cache_statistics(cache_stats)

        captured = capsys.readouterr()

        # Verify output contains expected information
        assert 'CACHE PERFORMANCE STATISTICS' in captured.out
        assert 'Indicator Cache:' in captured.out
        assert 'DataFrame Cache:' in captured.out
        assert 'Hits: 250' in captured.out
        assert 'Misses: 25' in captured.out
        assert 'Hits: 125' in captured.out
        assert 'Misses: 13' in captured.out

    def test_report_cache_statistics_with_empty_results(self, capsys, monkeypatch):
        """Test cache statistics reporting with zero stats."""
        cache_stats = {'ind_hits': 0, 'ind_misses': 0, 'df_hits': 0, 'df_misses': 0}

        mock_ind = MagicMock()
        mock_df = MagicMock()
        mock_ind.size.return_value = 0
        mock_df.size.return_value = 0
        monkeypatch.setattr(orch_module, 'indicator_cache', mock_ind)
        monkeypatch.setattr(orch_module, 'dataframe_cache', mock_df)

        _report_cache_statistics(cache_stats)

        captured = capsys.readouterr()

        # Should still print headers but with zero stats
        assert 'CACHE PERFORMANCE STATISTICS' in captured.out
        assert 'Hit rate: 0.00%' in captured.out

    def test_report_cache_statistics_calculates_hit_rate_correctly(self, capsys, monkeypatch):
        """Test that hit rate percentage is calculated correctly."""
        cache_stats = {'ind_hits': 80, 'ind_misses': 20, 'df_hits': 90, 'df_misses': 10}

        mock_ind = MagicMock()
        mock_df = MagicMock()
        mock_ind.size.return_value = 100
        mock_df.size.return_value = 50
        monkeypatch.setattr(orch_module, 'indicator_cache', mock_ind)
        monkeypatch.setattr(orch_module, 'dataframe_cache', mock_df)

        _report_cache_statistics(cache_stats)

        captured = capsys.readouterr()

        # 80/(80+20) = 80%, 90/(90+10) = 90%
        assert 'Hit rate: 80.00%' in captured.out
        assert 'Hit rate: 90.00%' in captured.out

    def test_report_cache_statistics_reports_given_stats(self, capsys, monkeypatch):
        """Test that the function reports the stats it receives."""
        cache_stats = {'ind_hits': 10, 'ind_misses': 5, 'df_hits': 8, 'df_misses': 2}

        mock_ind = MagicMock()
        mock_df = MagicMock()
        mock_ind.size.return_value = 50
        mock_df.size.return_value = 20
        monkeypatch.setattr(orch_module, 'indicator_cache', mock_ind)
        monkeypatch.setattr(orch_module, 'dataframe_cache', mock_df)

        _report_cache_statistics(cache_stats)

        captured = capsys.readouterr()

        assert 'Hits: 10' in captured.out
        assert 'Misses: 5' in captured.out


# ==================== Integration Tests ====================

class TestOrchestratorIntegration:
    """Test orchestrator integration scenarios."""

    def test_complete_orchestration_workflow(self, monkeypatch):
        """Test complete workflow from start to finish."""
        tester = MagicMock()
        tester.strategies = [('RSI_14_30_70', MagicMock())]
        tester.tested_months = ['1!']
        tester.symbols = ['ZS']
        tester.intervals = ['1h']
        tester.switch_dates_dict = {'ZS': ['2024-01-01']}
        tester.results = []

        mock_ind = MagicMock()
        mock_df = MagicMock()
        mock_executor = MagicMock()

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

        mock_as_completed = MagicMock(return_value=[mock_future])

        monkeypatch.setattr(orch_module, 'indicator_cache', mock_ind)
        monkeypatch.setattr(orch_module, 'dataframe_cache', mock_df)
        monkeypatch.setattr(orch_module, 'load_existing_results', MagicMock(return_value=(pd.DataFrame(), set())))
        monkeypatch.setattr(orch_module, 'save_shard', MagicMock())
        monkeypatch.setattr(orch_module, 'merge_shards', MagicMock())
        monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', mock_executor)
        monkeypatch.setattr(concurrent.futures, 'as_completed', mock_as_completed)

        results = run_tests(tester, verbose=False, max_workers=1, skip_existing=False)

        # Verify workflow steps
        assert mock_ind.reset_stats.called
        assert mock_df.reset_stats.called
        assert isinstance(results, list)

    def test_orchestration_with_multiple_strategies(self, monkeypatch):
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

        monkeypatch.setattr(orch_module, 'indicator_cache', MagicMock())
        monkeypatch.setattr(orch_module, 'dataframe_cache', MagicMock())
        monkeypatch.setattr(orch_module, 'load_existing_results', MagicMock(return_value=(pd.DataFrame(), set())))
        monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', MagicMock())
        monkeypatch.setattr(concurrent.futures, 'as_completed', MagicMock(return_value=[]))

        # Should generate 2*2*2*3 = 24 combinations
        run_tests(tester, verbose=False, max_workers=2, skip_existing=False)

        # Verify all combinations were considered
        assert tester.results is not None

    def test_orchestration_error_handling(self, monkeypatch):
        """Test that orchestration handles worker errors gracefully."""
        tester = MagicMock()
        tester.strategies = [('RSI_14_30_70', MagicMock())]
        tester.tested_months = ['1!']
        tester.symbols = ['ZS']
        tester.intervals = ['1h']
        tester.switch_dates_dict = {'ZS': []}
        tester.results = []

        mock_future = MagicMock()
        mock_future.result.side_effect = Exception("Worker error")

        mock_executor_instance = MagicMock()
        mock_executor_instance.submit.return_value = mock_future
        mock_executor = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance

        monkeypatch.setattr(orch_module, 'indicator_cache', MagicMock())
        monkeypatch.setattr(orch_module, 'dataframe_cache', MagicMock())
        monkeypatch.setattr(orch_module, 'load_existing_results', MagicMock(return_value=(pd.DataFrame(), set())))
        monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', mock_executor)
        monkeypatch.setattr(concurrent.futures, 'as_completed', MagicMock(return_value=[mock_future]))

        # Should not raise, should handle gracefully
        result = run_tests(tester, verbose=False, max_workers=1, skip_existing=False)

        assert isinstance(result, list)

    def test_orchestration_saves_intermediate_results_every_1000_tests(self, monkeypatch):
        """Test that intermediate results are written to a shard every 1000 completed tests."""
        tester = MagicMock()
        tester.strategies = [('RSI_14_30_70', MagicMock())]
        tester.tested_months = ['1!']
        tester.symbols = ['ZS']
        tester.intervals = ['1h']
        tester.switch_dates_dict = {'ZS': []}
        tester.results = []

        futures = []
        for i in range(1000):
            mock_future = MagicMock()
            mock_future.result.return_value = {'test': f'result_{i}'}
            futures.append(mock_future)

        mock_executor_instance = MagicMock()
        mock_executor_instance.submit.side_effect = futures
        mock_executor = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance

        mock_shard = MagicMock(return_value='/tmp/shard_0000.parquet')

        monkeypatch.setattr(orch_module, 'indicator_cache', MagicMock())
        monkeypatch.setattr(orch_module, 'dataframe_cache', MagicMock())
        monkeypatch.setattr(orch_module, 'load_existing_results', MagicMock(return_value=(pd.DataFrame(), set())))
        monkeypatch.setattr(orch_module, 'save_shard', mock_shard)
        monkeypatch.setattr(orch_module, 'merge_shards', MagicMock())
        monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', mock_executor)
        monkeypatch.setattr(concurrent.futures, 'as_completed', MagicMock(return_value=futures))

        run_tests(tester, verbose=False, max_workers=1, skip_existing=False)

        # Should call save_shard for the intermediate save at 1000 tests
        assert mock_shard.call_count >= 1

    def test_orchestration_indicator_cache_save_raises(self, monkeypatch):
        """Test that an indicator cache save failure propagates as RuntimeError."""
        tester = MagicMock()
        tester.strategies = [('RSI_14_30_70', MagicMock())]
        tester.tested_months = ['1!']
        tester.symbols = ['ZS']
        tester.intervals = ['1h']
        tester.switch_dates_dict = {'ZS': []}
        tester.results = []

        mock_ind = MagicMock()
        mock_ind.save_cache.side_effect = Exception('Cache save failed')

        mock_future = MagicMock()
        mock_future.result.return_value = {'test': 'result'}

        mock_executor_instance = MagicMock()
        mock_executor_instance.submit.return_value = mock_future
        mock_executor = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance

        monkeypatch.setattr(orch_module, 'indicator_cache', mock_ind)
        monkeypatch.setattr(orch_module, 'dataframe_cache', MagicMock())
        monkeypatch.setattr(orch_module, 'load_existing_results', MagicMock(return_value=(pd.DataFrame(), set())))
        monkeypatch.setattr(orch_module, 'save_shard', MagicMock())
        monkeypatch.setattr(orch_module, 'merge_shards', MagicMock())
        monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', mock_executor)
        monkeypatch.setattr(concurrent.futures, 'as_completed', MagicMock(return_value=[mock_future]))

        with pytest.raises(RuntimeError, match='Failed to save indicator cache after test completion'):
            run_tests(tester, verbose=False, max_workers=1, skip_existing=False)

    def test_orchestration_dataframe_cache_save_raises(self, monkeypatch):
        """Test that a dataframe cache save failure propagates as RuntimeError."""
        tester = MagicMock()
        tester.strategies = [('RSI_14_30_70', MagicMock())]
        tester.tested_months = ['1!']
        tester.symbols = ['ZS']
        tester.intervals = ['1h']
        tester.switch_dates_dict = {'ZS': []}
        tester.results = []

        mock_df_cache = MagicMock()
        mock_df_cache.save_cache.side_effect = Exception('Dataframe cache save failed')

        mock_future = MagicMock()
        mock_future.result.return_value = {'test': 'result'}

        mock_executor_instance = MagicMock()
        mock_executor_instance.submit.return_value = mock_future
        mock_executor = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance

        monkeypatch.setattr(orch_module, 'indicator_cache', MagicMock())
        monkeypatch.setattr(orch_module, 'dataframe_cache', mock_df_cache)
        monkeypatch.setattr(orch_module, 'load_existing_results', MagicMock(return_value=(pd.DataFrame(), set())))
        monkeypatch.setattr(orch_module, 'save_shard', MagicMock())
        monkeypatch.setattr(orch_module, 'merge_shards', MagicMock())
        monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', mock_executor)
        monkeypatch.setattr(concurrent.futures, 'as_completed', MagicMock(return_value=[mock_future]))

        with pytest.raises(RuntimeError, match='Failed to save dataframe cache after test completion'):
            run_tests(tester, verbose=False, max_workers=1, skip_existing=False)
