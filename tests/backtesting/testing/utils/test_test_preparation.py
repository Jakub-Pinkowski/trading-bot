"""
Tests for test_preparation utility module.

Tests cover:
- load_existing_results function
- check_test_exists function
- File loading and error handling
- Set-based lookup optimization
- Empty/missing file handling
- Integration scenarios
"""
from unittest.mock import patch

import pandas as pd

from app.backtesting.testing.utils.test_preparation import (
    load_existing_results,
    check_test_exists
)


# ==================== Load Existing Results Tests ====================

class TestLoadExistingResults:
    """Test load_existing_results function."""

    def test_load_existing_results_file_exists(self, existing_results_factory):
        """Test loading results when file exists."""
        combinations = [
            ('1!', 'ZS', '1h', 'RSI_14_30_70'),
            ('2!', 'CL', '15m', 'EMA_9_21')
        ]
        mock_df, expected_set = existing_results_factory(combinations)

        with patch('os.path.exists', return_value=True), \
                patch('pandas.read_parquet', return_value=mock_df):
            df, combinations_set = load_existing_results()

            assert len(df) == 2
            assert len(combinations_set) == 2
            assert ('1!', 'ZS', '1h', 'RSI_14_30_70', -1) in combinations_set
            assert ('2!', 'CL', '15m', 'EMA_9_21', -1) in combinations_set

    def test_load_existing_results_file_not_exists(self):
        """Test loading results when file doesn't exist."""
        with patch('os.path.exists', return_value=False):
            df, combinations_set = load_existing_results()

            assert df.empty
            assert len(combinations_set) == 0

    def test_load_existing_results_read_error(self):
        """Test handling of read errors."""
        with patch('os.path.exists', return_value=True), \
                patch('pandas.read_parquet', side_effect=Exception("Read error")):
            df, combinations_set = load_existing_results()

            # Should return empty DataFrame and set on error
            assert df.empty
            assert len(combinations_set) == 0

    def test_load_existing_results_large_dataset(self):
        """Test loading large dataset."""
        # Create DataFrame with 1000 results
        mock_df = pd.DataFrame({
            'month': [f'{i % 3 + 1}!' for i in range(1000)],
            'symbol': ['ZS' if i % 2 == 0 else 'CL' for i in range(1000)],
            'interval': ['1h' if i % 3 == 0 else '15m' for i in range(1000)],
            'strategy': [f'Strategy_{i}' for i in range(1000)],
            'segment_id': [-1] * 1000
        })

        with patch('os.path.exists', return_value=True), \
                patch('pandas.read_parquet', return_value=mock_df):
            df, combinations_set = load_existing_results()

            assert len(df) == 1000
            assert len(combinations_set) == 1000

    def test_load_existing_results_set_structure(self):
        """Test that combinations set has correct structure."""
        mock_df = pd.DataFrame({
            'month': ['1!'],
            'symbol': ['ZS'],
            'interval': ['1h'],
            'strategy': ['RSI_14_30_70'],
            'segment_id': [-1]
        })

        with patch('os.path.exists', return_value=True), \
                patch('pandas.read_parquet', return_value=mock_df):
            df, combinations_set = load_existing_results()

            # Verify set contains tuples
            assert isinstance(combinations_set, set)
            combination = list(combinations_set)[0]
            assert isinstance(combination, tuple)
            assert len(combination) == 5


# ==================== Check Test Exists Tests ====================

class TestCheckTestExists:
    """Test check_test_exists function."""

    def test_check_test_exists_when_exists(self, existing_results_factory):
        """Test checking for test that exists."""
        combinations = [
            ('1!', 'ZS', '1h', 'RSI_14_30_70'),
            ('2!', 'CL', '15m', 'EMA_9_21')
        ]
        existing_data = existing_results_factory(combinations)

        result = check_test_exists(existing_data, '1!', 'ZS', '1h', 'RSI_14_30_70')

        assert result is True

    def test_check_test_exists_when_not_exists(self, existing_results_factory):
        """Test checking for test that doesn't exist."""
        combinations = [('1!', 'ZS', '1h', 'RSI_14_30_70')]
        existing_data = existing_results_factory(combinations)

        result = check_test_exists(existing_data, '2!', 'CL', '15m', 'EMA_9_21')

        assert result is False

    def test_check_test_exists_with_empty_data(self, existing_results_factory):
        """Test checking when no existing data."""
        existing_data = existing_results_factory([])

        result = check_test_exists(existing_data, '1!', 'ZS', '1h', 'RSI_14_30_70')

        assert result is False

    def test_check_test_exists_case_sensitive(self, existing_results_factory):
        """Test that check is case-sensitive."""
        combinations = [('1!', 'ZS', '1h', 'RSI_14_30_70')]
        existing_data = existing_results_factory(combinations)

        # Different case should not match
        result = check_test_exists(existing_data, '1!', 'zs', '1h', 'RSI_14_30_70')

        assert result is False

    def test_check_test_exists_partial_match(self, existing_results_factory):
        """Test that partial matches don't count as exists."""
        combinations = [('1!', 'ZS', '1h', 'RSI_14_30_70')]
        existing_data = existing_results_factory(combinations)

        # Same symbol/interval but different month/strategy
        result = check_test_exists(existing_data, '2!', 'ZS', '1h', 'RSI_14_30_70')

        assert result is False


# ==================== Multiple Tests Performance ====================

class TestCheckPerformance:
    """Test performance characteristics of check_test_exists."""

    def test_check_test_exists_is_fast_with_large_dataset(self):
        """Test that checking is O(1) even with large dataset."""
        # Create large dataset
        data = []
        for i in range(10000):
            data.append({
                'month': f'{i % 3 + 1}!',
                'symbol': ['ZS', 'CL', 'GC'][i % 3],
                'interval': ['15m', '1h', '4h'][i % 3],
                'strategy': f'Strategy_{i}',
                'segment_id': -1
            })

        mock_df = pd.DataFrame(data)
        combinations_set = set(zip(
            mock_df['month'].values,
            mock_df['symbol'].values,
            mock_df['interval'].values,
            mock_df['strategy'].values,
            [-1] * len(mock_df)
        ))
        existing_data = (mock_df, combinations_set)

        # Check for test at the end (should be O(1), not O(n))
        result = check_test_exists(existing_data, '1!', 'ZS', '15m', 'Strategy_9999')

        # Should complete quickly (this is just a correctness check)
        assert result is True

    def test_multiple_checks_in_sequence(self):
        """Test multiple sequential checks."""
        mock_df = pd.DataFrame({
            'month': ['1!', '2!', '3!'],
            'symbol': ['ZS', 'CL', 'GC'],
            'interval': ['1h', '15m', '4h'],
            'strategy': ['RSI_14_30_70', 'EMA_9_21', 'MACD_12_26_9'],
            'segment_id': [-1, -1, -1]
        })
        combinations_set = set(zip(
            mock_df['month'].values,
            mock_df['symbol'].values,
            mock_df['interval'].values,
            mock_df['strategy'].values,
            [-1] * len(mock_df)
        ))
        existing_data = (mock_df, combinations_set)

        # Check multiple times
        results = []
        for i in range(10):
            result = check_test_exists(existing_data, '1!', 'ZS', '1h', 'RSI_14_30_70')
            results.append(result)

        # All should return True
        assert all(results)


# ==================== Integration Tests ====================

class TestTestPreparationIntegration:
    """Test integration between load and check functions."""

    def test_load_and_check_workflow(self):
        """Test complete workflow of loading and checking."""
        mock_df = pd.DataFrame({
            'month': ['1!', '2!'],
            'symbol': ['ZS', 'CL'],
            'interval': ['1h', '15m'],
            'strategy': ['RSI_14_30_70', 'EMA_9_21'],
            'segment_id': [-1, -1]
        })

        with patch('os.path.exists', return_value=True), \
                patch('pandas.read_parquet', return_value=mock_df):
            # Load existing results
            existing_data = load_existing_results()

            # Check for existing test
            exists_1 = check_test_exists(existing_data, '1!', 'ZS', '1h', 'RSI_14_30_70')
            exists_2 = check_test_exists(existing_data, '3!', 'GC', '4h', 'MACD_12_26_9')

            assert exists_1 is True
            assert exists_2 is False

    def test_skip_existing_tests_scenario(self):
        """Test realistic scenario of skipping existing tests."""
        # Existing results
        mock_df = pd.DataFrame({
            'month': ['1!', '1!', '2!'],
            'symbol': ['ZS', 'ZS', 'CL'],
            'interval': ['1h', '4h', '15m'],
            'strategy': ['RSI_14_30_70', 'EMA_9_21', 'MACD_12_26_9'],
            'segment_id': [-1, -1, -1]
        })

        with patch('os.path.exists', return_value=True), \
                patch('pandas.read_parquet', return_value=mock_df):

            existing_data = load_existing_results()

            # New tests to run
            new_tests = [
                ('1!', 'ZS', '1h', 'RSI_14_30_70'),  # Exists
                ('1!', 'ZS', '4h', 'EMA_9_21'),  # Exists
                ('1!', 'CL', '1h', 'RSI_14_30_70'),  # New
                ('2!', 'GC', '15m', 'MACD_12_26_9'),  # New
            ]

            to_skip = []
            to_run = []

            for month, symbol, interval, strategy in new_tests:
                if check_test_exists(existing_data, month, symbol, interval, strategy):
                    to_skip.append((month, symbol, interval, strategy))
                else:
                    to_run.append((month, symbol, interval, strategy))

            assert len(to_skip) == 2
            assert len(to_run) == 2

    def test_empty_results_all_tests_run(self):
        """Test that all tests run when no existing results."""
        with patch('os.path.exists', return_value=False):
            existing_data = load_existing_results()

            # All tests should be marked as not existing
            tests = [
                ('1!', 'ZS', '1h', 'RSI_14_30_70'),
                ('2!', 'CL', '15m', 'EMA_9_21'),
                ('3!', 'GC', '4h', 'MACD_12_26_9'),
            ]

            results = [check_test_exists(existing_data, *test) for test in tests]

            # All should be False (none exist)
            assert not any(results)

    def test_duplicate_strategies_different_parameters(self):
        """Test that strategies with different parameters are treated as different."""
        mock_df = pd.DataFrame({
            'month': ['1!', '1!'],
            'symbol': ['ZS', 'ZS'],
            'interval': ['1h', '1h'],
            'strategy': ['RSI_14_30_70', 'RSI_21_30_70'],  # Different periods
            'segment_id': [-1, -1]
        })
        combinations_set = set(zip(
            mock_df['month'].values,
            mock_df['symbol'].values,
            mock_df['interval'].values,
            mock_df['strategy'].values,
            [-1] * len(mock_df)
        ))
        existing_data = (mock_df, combinations_set)

        # RSI_14_30_70 exists
        exists_14 = check_test_exists(existing_data, '1!', 'ZS', '1h', 'RSI_14_30_70')
        # RSI_21_30_70 exists
        exists_21 = check_test_exists(existing_data, '1!', 'ZS', '1h', 'RSI_21_30_70')
        # RSI_28_30_70 doesn't exist
        exists_28 = check_test_exists(existing_data, '1!', 'ZS', '1h', 'RSI_28_30_70')

        assert exists_14 is True
        assert exists_21 is True
        assert exists_28 is False


# ==================== Edge Cases ====================

class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_load_results_with_corrupted_file(self):
        """Test handling of corrupted parquet file."""
        with patch('os.path.exists', return_value=True), \
                patch('pandas.read_parquet', side_effect=Exception("Corrupted file")):
            df, combinations_set = load_existing_results()

            # Should handle gracefully
            assert df.empty
            assert len(combinations_set) == 0

    def test_check_with_none_values(self):
        """Test checking with None values in parameters."""
        mock_df = pd.DataFrame({
            'month': ['1!'],
            'symbol': ['ZS'],
            'interval': ['1h'],
            'strategy': ['RSI_14_30_70']
        })
        combinations_set = {('1!', 'ZS', '1h', 'RSI_14_30_70', None)}
        existing_data = (mock_df, combinations_set)

        # Check with None should not match
        result = check_test_exists(existing_data, None, 'ZS', '1h', 'RSI_14_30_70')

        assert result is False

    def test_load_results_with_extra_columns(self):
        """Test loading results file with extra columns."""
        mock_df = pd.DataFrame({
            'month': ['1!'],
            'symbol': ['ZS'],
            'interval': ['1h'],
            'strategy': ['RSI_14_30_70'],
            'segment_id': [-1],
            'extra_column': ['extra_data'],
            'timestamp': ['2024-01-01']
        })

        with patch('os.path.exists', return_value=True), \
                patch('pandas.read_parquet', return_value=mock_df):
            df, combinations_set = load_existing_results()

            # Should still work with extra columns
            assert len(df) == 1
            assert len(combinations_set) == 1
            assert ('1!', 'ZS', '1h', 'RSI_14_30_70', -1) in combinations_set

    def test_load_results_with_missing_columns(self):
        """Test loading results file with missing segment_id column raises error."""
        mock_df = pd.DataFrame({
            'month': ['1!'],
            'symbol': ['ZS'],
            'interval': ['1h'],
            'strategy': ['RSI_14_30_70']
            # Missing segment_id column
        })

        with patch('os.path.exists', return_value=True), \
                patch('pandas.read_parquet', return_value=mock_df):
            # Should fail when segment_id column is missing (no backward compatibility)
            df, combinations_set = load_existing_results()

            # Function catches the error and returns empty
            assert df.empty
            assert len(combinations_set) == 0

    def test_check_test_exists_with_special_characters(self):
        """Test checking with special characters in strategy name."""
        mock_df = pd.DataFrame({
            'month': ['1!'],
            'symbol': ['ZS'],
            'interval': ['1h'],
            'strategy': ['RSI_14_30_70_trailing=2.5_slippage=1'],
            'segment_id': [-1]
        })
        combinations_set = {('1!', 'ZS', '1h', 'RSI_14_30_70_trailing=2.5_slippage=1', -1)}
        existing_data = (mock_df, combinations_set)

        result = check_test_exists(
            existing_data, '1!', 'ZS', '1h', 'RSI_14_30_70_trailing=2.5_slippage=1'
        )

        assert result is True
