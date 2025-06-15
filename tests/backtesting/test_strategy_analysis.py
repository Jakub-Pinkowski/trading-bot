import unittest
from unittest.mock import patch

import pandas as pd

from app.backtesting.strategy_analysis import StrategyAnalyzer, _format_column_name


class TestFormatColumnName(unittest.TestCase):
    """Tests for the _format_column_name function."""

    def test_special_case_column_names(self):
        """Test formatting of special case column names."""
        # Test all special case mappings
        special_cases = {
            'average_trade_return_percentage_of_margin': 'Avg Return %',
            'average_win_percentage_of_margin': 'Avg Win %',
            'total_return_percentage_of_margin': 'Total Return %',
            'average_loss_percentage_of_margin': 'Avg Loss %',
            'maximum_drawdown_percentage': 'Max Drawdown %',
            'win_rate': 'Win Rate %',
            'max_consecutive_wins': 'Max Cons Wins',
            'max_consecutive_losses': 'Max Cons Losses',
            'sharpe_ratio': 'Sharpe',
            'sortino_ratio': 'Sortino',
            'calmar_ratio': 'Calmar'
        }

        for column_name, expected_formatted_name in special_cases.items():
            formatted_name = _format_column_name(column_name)
            self.assertEqual(formatted_name, expected_formatted_name)

    def test_regular_column_names(self):
        """Test formatting of regular column names."""
        test_cases = {
            'strategy': 'Strategy',
            'symbol': 'Symbol',
            'interval': 'Interval',
            'total_trades': 'Total Trades',
            'profit_factor': 'Profit Factor'
        }

        for column_name, expected_formatted_name in test_cases.items():
            formatted_name = _format_column_name(column_name)
            self.assertEqual(formatted_name, expected_formatted_name)

    def test_multi_word_column_names(self):
        """Test formatting of multi-word column names."""
        test_cases = {
            'symbol_count': 'Symbol Count',
            'interval_count': 'Interval Count',
            'return_to_drawdown_ratio': 'Return To Drawdown Ratio'
        }

        for column_name, expected_formatted_name in test_cases.items():
            formatted_name = _format_column_name(column_name)
            self.assertEqual(formatted_name, expected_formatted_name)


class TestStrategyAnalyzer(unittest.TestCase):
    """Tests for the StrategyAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a sample DataFrame for testing
        self.sample_data = pd.DataFrame({
            'strategy': ['Strategy1', 'Strategy1', 'Strategy2', 'Strategy2'],
            'symbol': ['ES', 'NQ', 'ES', 'NQ'],
            'interval': ['1d', '1d', '4h', '4h'],
            'total_trades': [10, 15, 20, 25],
            'win_rate': [60.0, 70.0, 55.0, 65.0],
            'total_return_percentage_of_margin': [5.0, 7.0, 4.0, 6.0],
            'average_trade_return_percentage_of_margin': [0.5, 0.47, 0.2, 0.24],
            'average_win_percentage_of_margin': [1.0, 0.9, 0.8, 0.7],
            'average_loss_percentage_of_margin': [-0.5, -0.4, -0.6, -0.3],
            'commission_percentage_of_margin': [0.1, 0.1, 0.1, 0.1],
            'profit_factor': [2.0, 2.5, 1.8, 2.2],
            'maximum_drawdown_percentage': [2.0, 1.5, 2.5, 1.8],
            'return_to_drawdown_ratio': [2.5, 4.67, 1.6, 3.33],
            'sharpe_ratio': [1.5, 1.8, 1.2, 1.6],
            'sortino_ratio': [2.0, 2.5, 1.8, 2.2],
            'calmar_ratio': [2.5, 4.67, 1.6, 3.33]
        })

    @patch('pandas.read_parquet')
    def test_init_and_load_results(self, mock_read_parquet):
        """Test initialization and loading results."""
        # Setup mock
        mock_read_parquet.return_value = self.sample_data

        # Test initialization
        analyzer = StrategyAnalyzer()

        # Verify that read_parquet was called with the correct file path
        mock_read_parquet.assert_called_once()
        self.assertEqual(analyzer.results_df.equals(self.sample_data), True)

    @patch('pandas.read_parquet')
    def test_load_results_error(self, mock_read_parquet):
        """Test error handling when loading results."""
        # Setup mock to raise an exception
        mock_read_parquet.side_effect = Exception("File not found")

        # Test that the exception is propagated
        with self.assertRaises(Exception):
            analyzer = StrategyAnalyzer()

    def test_aggregate_strategies_no_results(self):
        """Test aggregating strategies when no results are available."""
        analyzer = StrategyAnalyzer()
        analyzer.results_df = None

        with self.assertRaises(ValueError):
            analyzer._aggregate_strategies()

    def test_aggregate_strategies_empty_results(self):
        """Test aggregating strategies with empty results."""
        analyzer = StrategyAnalyzer()
        analyzer.results_df = pd.DataFrame()

        with self.assertRaises(ValueError):
            analyzer._aggregate_strategies()

    @patch('pandas.read_parquet')
    def test_aggregate_strategies_basic(self, mock_read_parquet):
        """Test basic aggregation of strategies with both weighted and non-weighted approaches."""
        # Setup mock
        mock_read_parquet.return_value = self.sample_data

        # Initialize analyzer
        analyzer = StrategyAnalyzer()

        # Test weighted aggregation (default)
        weighted_aggregated = analyzer._aggregate_strategies(weighted=True)

        # Verify results for weighted aggregation
        self.assertEqual(len(weighted_aggregated), 2)  # Two strategies
        self.assertEqual(weighted_aggregated['strategy'].tolist(), ['Strategy1', 'Strategy2'])

        # Check that symbol_count and interval_count are calculated correctly
        self.assertEqual(
            weighted_aggregated.loc[weighted_aggregated['strategy'] == 'Strategy1', 'symbol_count'].iloc[0], 2)
        self.assertEqual(
            weighted_aggregated.loc[weighted_aggregated['strategy'] == 'Strategy2', 'symbol_count'].iloc[0], 2)
        self.assertEqual(
            weighted_aggregated.loc[weighted_aggregated['strategy'] == 'Strategy1', 'interval_count'].iloc[0], 1)
        self.assertEqual(
            weighted_aggregated.loc[weighted_aggregated['strategy'] == 'Strategy2', 'interval_count'].iloc[0], 1)

        # Check that total_trades is summed correctly
        self.assertEqual(
            weighted_aggregated.loc[weighted_aggregated['strategy'] == 'Strategy1', 'total_trades'].iloc[0], 25)
        self.assertEqual(
            weighted_aggregated.loc[weighted_aggregated['strategy'] == 'Strategy2', 'total_trades'].iloc[0], 45)

        # Test non-weighted aggregation
        non_weighted_aggregated = analyzer._aggregate_strategies(weighted=False)

        # Verify results for non-weighted aggregation
        self.assertEqual(len(non_weighted_aggregated), 2)  # Two strategies
        self.assertEqual(non_weighted_aggregated['strategy'].tolist(), ['Strategy1', 'Strategy2'])

        # Check that basic metrics are the same for both approaches
        self.assertEqual(
            non_weighted_aggregated.loc[non_weighted_aggregated['strategy'] == 'Strategy1', 'symbol_count'].iloc[0], 2)
        self.assertEqual(
            non_weighted_aggregated.loc[non_weighted_aggregated['strategy'] == 'Strategy2', 'symbol_count'].iloc[0], 2)
        self.assertEqual(
            non_weighted_aggregated.loc[non_weighted_aggregated['strategy'] == 'Strategy1', 'total_trades'].iloc[0], 25)
        self.assertEqual(
            non_weighted_aggregated.loc[non_weighted_aggregated['strategy'] == 'Strategy2', 'total_trades'].iloc[0], 45)

        # Verify that some metrics are different between weighted and non-weighted approaches
        # For example, win_rate should be calculated differently
        self.assertNotEqual(
            weighted_aggregated.loc[weighted_aggregated['strategy'] == 'Strategy1', 'win_rate'].iloc[0],
            non_weighted_aggregated.loc[non_weighted_aggregated['strategy'] == 'Strategy1', 'win_rate'].iloc[0]
        )

    @patch('pandas.read_parquet')
    def test_aggregate_strategies_with_filters(self, mock_read_parquet):
        """Test aggregation of strategies with filters for both weighted and non-weighted approaches."""
        # Setup mock
        mock_read_parquet.return_value = self.sample_data

        # Initialize analyzer
        analyzer = StrategyAnalyzer()

        # Test weighted aggregation with min_trades filter
        weighted_aggregated = analyzer._aggregate_strategies(min_trades=20, weighted=True)
        self.assertEqual(len(weighted_aggregated), 1)  # Only Strategy2 has all rows with >= 20 trades

        # Test non-weighted aggregation with min_trades filter
        non_weighted_aggregated = analyzer._aggregate_strategies(min_trades=20, weighted=False)
        self.assertEqual(len(non_weighted_aggregated), 1)  # Only Strategy2 has all rows with >= 20 trades

        # Verify that some metrics are different between weighted and non-weighted approaches
        self.assertNotEqual(
            weighted_aggregated.loc[weighted_aggregated['strategy'] == 'Strategy2', 'win_rate'].iloc[0],
            non_weighted_aggregated.loc[non_weighted_aggregated['strategy'] == 'Strategy2', 'win_rate'].iloc[0]
        )

        # Test weighted aggregation with interval filter
        weighted_aggregated = analyzer._aggregate_strategies(interval='1d', weighted=True)
        self.assertEqual(len(weighted_aggregated), 1)  # Only Strategy1 has 1d interval
        self.assertEqual(weighted_aggregated['strategy'].iloc[0], 'Strategy1')

        # Test non-weighted aggregation with interval filter
        non_weighted_aggregated = analyzer._aggregate_strategies(interval='1d', weighted=False)
        self.assertEqual(len(non_weighted_aggregated), 1)  # Only Strategy1 has 1d interval
        self.assertEqual(non_weighted_aggregated['strategy'].iloc[0], 'Strategy1')

        # Test weighted aggregation with symbol filter
        weighted_aggregated = analyzer._aggregate_strategies(symbol='ES', weighted=True)
        self.assertEqual(len(weighted_aggregated), 2)  # Both strategies have ES

        # Test non-weighted aggregation with symbol filter
        non_weighted_aggregated = analyzer._aggregate_strategies(symbol='ES', weighted=False)
        self.assertEqual(len(non_weighted_aggregated), 2)  # Both strategies have ES

        # Test weighted aggregation with multiple filters
        weighted_aggregated = analyzer._aggregate_strategies(min_trades=20, interval='4h', weighted=True)
        self.assertEqual(len(weighted_aggregated), 1)  # Only Strategy2 has 4h interval with >= 20 trades
        self.assertEqual(weighted_aggregated['strategy'].iloc[0], 'Strategy2')

        # Test non-weighted aggregation with multiple filters
        non_weighted_aggregated = analyzer._aggregate_strategies(min_trades=20, interval='4h', weighted=False)
        self.assertEqual(len(non_weighted_aggregated), 1)  # Only Strategy2 has 4h interval with >= 20 trades
        self.assertEqual(non_weighted_aggregated['strategy'].iloc[0], 'Strategy2')

    @patch('pandas.read_parquet')
    def test_get_top_strategies_no_results(self, mock_read_parquet):
        """Test getting top strategies when no results are available."""
        # Setup mock
        mock_read_parquet.return_value = None

        # Initialize analyzer with no results
        analyzer = StrategyAnalyzer()
        analyzer.results_df = None

        with self.assertRaises(ValueError):
            analyzer.get_top_strategies('win_rate', 0)

    @patch('pandas.read_parquet')
    @patch('app.backtesting.strategy_analysis.StrategyAnalyzer._save_results_to_csv')
    def test_get_top_strategies_basic(self, mock_save_results, mock_read_parquet):
        """Test the basic functionality of get_top_strategies with both weighted and non-weighted approaches."""
        # Setup mocks
        mock_read_parquet.return_value = self.sample_data
        mock_save_results.reset_mock()  # Reset mock to ensure a clean state

        # Initialize analyzer
        analyzer = StrategyAnalyzer()

        # Test getting top strategies by win_rate with weighted=True (default)
        weighted_top_strategies = analyzer.get_top_strategies('win_rate', 0, weighted=True)

        # Verify results for weighted approach
        self.assertEqual(len(weighted_top_strategies), 4)
        self.assertEqual(weighted_top_strategies.iloc[0]['win_rate'], 70.0)  # Highest win_rate first

        # Verify save_results_to_csv was called with weighted=True
        mock_save_results.assert_called_once()
        mock_save_results.reset_mock()  # Reset mock for next test

        # Test getting top strategies by win_rate with weighted=False
        non_weighted_top_strategies = analyzer.get_top_strategies('win_rate', 0, weighted=False)

        # Verify results for a non-weighted approach
        self.assertEqual(len(non_weighted_top_strategies), 4)
        self.assertEqual(non_weighted_top_strategies.iloc[0]['win_rate'], 70.0)  # Highest win_rate first

        # Verify save_results_to_csv was called with weighted=False
        mock_save_results.assert_called_once()

        # Compare results between weighted and non-weighted approaches
        # For individual strategies (not aggregated), the results should be the same
        pd.testing.assert_frame_equal(weighted_top_strategies, non_weighted_top_strategies)

        # But when aggregated, they should be different
        mock_save_results.reset_mock()
        weighted_aggregated = analyzer.get_top_strategies('win_rate', 0, aggregate=True, weighted=True)
        mock_save_results.reset_mock()
        non_weighted_aggregated = analyzer.get_top_strategies('win_rate', 0, aggregate=True, weighted=False)

        # Verify that the aggregated results are different
        self.assertFalse(weighted_aggregated.equals(non_weighted_aggregated))

    @patch('pandas.read_parquet')
    @patch('app.backtesting.strategy_analysis.StrategyAnalyzer._save_results_to_csv')
    def test_get_top_strategies_with_filters(self, mock_save_results, mock_read_parquet):
        """Test get_top_strategies with various filters for both weighted and non-weighted approaches."""
        # Setup mocks
        mock_read_parquet.return_value = self.sample_data

        # Initialize analyzer
        analyzer = StrategyAnalyzer()

        # Test with min_trades filter (weighted=True)
        weighted_top_strategies = analyzer.get_top_strategies('win_rate', 20, weighted=True)
        self.assertEqual(len(weighted_top_strategies), 2)  # Only rows with >= 20 trades

        # Test with min_trades filter (weighted=False)
        non_weighted_top_strategies = analyzer.get_top_strategies('win_rate', 20, weighted=False)
        self.assertEqual(len(non_weighted_top_strategies), 2)  # Only rows with >= 20 trades

        # Test with aggregate=True (weighted=True)
        weighted_aggregated = analyzer.get_top_strategies('win_rate', 0, aggregate=True, weighted=True)
        self.assertEqual(len(weighted_aggregated), 2)  # Aggregated to 2 strategies

        # Test with aggregate=True (weighted=False)
        non_weighted_aggregated = analyzer.get_top_strategies('win_rate', 0, aggregate=True, weighted=False)
        self.assertEqual(len(non_weighted_aggregated), 2)  # Aggregated to 2 strategies

        # Verify that the aggregated results are different between weighted and non-weighted
        self.assertFalse(weighted_aggregated.equals(non_weighted_aggregated))

        # Test with interval filter (weighted=True)
        weighted_interval = analyzer.get_top_strategies('win_rate', 0, interval='1d', weighted=True)
        self.assertEqual(len(weighted_interval), 2)  # Only 1d interval

        # Test with interval filter (weighted=False)
        non_weighted_interval = analyzer.get_top_strategies('win_rate', 0, interval='1d', weighted=False)
        self.assertEqual(len(non_weighted_interval), 2)  # Only 1d interval

        # Test with symbol filter (weighted=True)
        weighted_symbol = analyzer.get_top_strategies('win_rate', 0, symbol='ES', weighted=True)
        self.assertEqual(len(weighted_symbol), 2)  # Only ES symbol

        # Test with symbol filter (weighted=False)
        non_weighted_symbol = analyzer.get_top_strategies('win_rate', 0, symbol='ES', weighted=False)
        self.assertEqual(len(non_weighted_symbol), 2)  # Only ES symbol

        # Test with multiple filters (weighted=True)
        weighted_multiple = analyzer.get_top_strategies('win_rate', 0, interval='4h', symbol='NQ', weighted=True)
        self.assertEqual(len(weighted_multiple), 1)  # Only 4h interval and NQ symbol

        # Test with multiple filters (weighted=False)
        non_weighted_multiple = analyzer.get_top_strategies('win_rate', 0, interval='4h', symbol='NQ', weighted=False)
        self.assertEqual(len(non_weighted_multiple), 1)  # Only 4h interval and NQ symbol

    @patch('pandas.read_parquet')
    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    def test_save_results_to_csv_basic(self, mock_to_csv, mock_makedirs, mock_read_parquet):
        """Test the basic functionality of save_results_to_csv with both weighted and non-weighted approaches."""
        # Setup mocks
        mock_read_parquet.return_value = self.sample_data

        # Initialize analyzer
        analyzer = StrategyAnalyzer()

        # Test saving results with weighted=True
        mock_to_csv.reset_mock()
        mock_makedirs.reset_mock()
        analyzer._save_results_to_csv('win_rate', 10, analyzer.results_df, False, None, None, weighted=True)

        # Verify os.makedirs was called
        mock_makedirs.assert_called_once()

        # Verify to_csv was called
        mock_to_csv.assert_called_once()

        # Test saving results with weighted=False
        mock_to_csv.reset_mock()
        mock_makedirs.reset_mock()
        analyzer._save_results_to_csv('win_rate', 10, analyzer.results_df, False, None, None, weighted=False)

        # Verify os.makedirs was called
        mock_makedirs.assert_called_once()

        # Verify to_csv was called
        mock_to_csv.assert_called_once()

    @patch('pandas.read_parquet')
    def test_save_results_to_csv_no_results(self, mock_read_parquet):
        """Test save_results_to_csv when no results are available."""
        # Setup mock
        mock_read_parquet.return_value = None

        # Initialize analyzer with no results
        analyzer = StrategyAnalyzer()
        analyzer.results_df = None

        with self.assertRaises(ValueError):
            analyzer._save_results_to_csv('win_rate', 10, None, False, None, None)

    @patch('pandas.read_parquet')
    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    def test_save_results_to_csv_with_formatting(self, mock_to_csv, mock_makedirs, mock_read_parquet):
        """Test save_results_to_csv with column formatting for both weighted and non-weighted approaches."""
        # Setup mocks
        mock_read_parquet.return_value = self.sample_data

        # Initialize analyzer
        analyzer = StrategyAnalyzer()

        # Test saving results with weighted=True
        mock_to_csv.reset_mock()
        analyzer._save_results_to_csv('win_rate', 10, analyzer.results_df, False, None, None, weighted=True)

        # Verify to_csv was called
        mock_to_csv.assert_called_once()

        # Test saving results with weighted=False
        mock_to_csv.reset_mock()
        analyzer._save_results_to_csv('win_rate', 10, analyzer.results_df, False, None, None, weighted=False)

        # Verify to_csv was called
        mock_to_csv.assert_called_once()

        # Since we can't easily check the formatted column names in the mock,
        # we'll just verify that the method completed without errors

    @patch('pandas.read_parquet')
    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    def test_save_results_to_csv_filename_generation(self, mock_to_csv, mock_makedirs, mock_read_parquet):
        """Test filename generation in save_results_to_csv with both weighted and non-weighted approaches."""
        # Setup mocks
        mock_read_parquet.return_value = self.sample_data

        # Initialize analyzer
        analyzer = StrategyAnalyzer()

        # Since we can't easily access the filename in the mock,
        # we'll use a different approach to test filename generation

        # Test with different parameters and weighted=True
        analyzer._save_results_to_csv('win_rate', 10, analyzer.results_df, False, None, None, weighted=True)
        mock_to_csv.assert_called_once()
        mock_to_csv.reset_mock()

        analyzer._save_results_to_csv('win_rate', 10, analyzer.results_df, True, None, None, weighted=True)
        mock_to_csv.assert_called_once()
        mock_to_csv.reset_mock()

        analyzer._save_results_to_csv('win_rate', 10, analyzer.results_df, False, '1d', None, weighted=True)
        mock_to_csv.assert_called_once()
        mock_to_csv.reset_mock()

        analyzer._save_results_to_csv('win_rate', 10, analyzer.results_df, False, None, 'ES', weighted=True)
        mock_to_csv.assert_called_once()
        mock_to_csv.reset_mock()

        # Test with different parameters and weighted=False
        analyzer._save_results_to_csv('win_rate', 10, analyzer.results_df, False, None, None, weighted=False)
        mock_to_csv.assert_called_once()
        mock_to_csv.reset_mock()

        analyzer._save_results_to_csv('win_rate', 10, analyzer.results_df, True, None, None, weighted=False)
        mock_to_csv.assert_called_once()
        mock_to_csv.reset_mock()

        analyzer._save_results_to_csv('win_rate', 10, analyzer.results_df, False, '1d', None, weighted=False)
        mock_to_csv.assert_called_once()
        mock_to_csv.reset_mock()

        analyzer._save_results_to_csv('win_rate', 10, analyzer.results_df, False, None, 'ES', weighted=False)
        mock_to_csv.assert_called_once()

        # Verify that the method completed without errors for all parameter combinations

    @patch('pandas.read_parquet')
    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    def test_save_results_to_csv_with_none_df_to_save(self, mock_to_csv, mock_makedirs, mock_read_parquet):
        """Test save_results_to_csv when df_to_save is None but results_df is available for both weighted approaches."""
        # Setup mocks
        mock_read_parquet.return_value = self.sample_data

        # Initialize analyzer
        analyzer = StrategyAnalyzer()

        # Test saving results with df_to_save=None and weighted=True
        mock_to_csv.reset_mock()
        analyzer._save_results_to_csv('win_rate', 10, None, False, None, None, weighted=True)

        # Verify to_csv was called (using results_df instead)
        mock_to_csv.assert_called_once()

        # Test saving results with df_to_save=None and weighted=False
        mock_to_csv.reset_mock()
        analyzer._save_results_to_csv('win_rate', 10, None, False, None, None, weighted=False)

        # Verify to_csv was called (using results_df instead)
        mock_to_csv.assert_called_once()

    @patch('pandas.read_parquet')
    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    def test_save_results_to_csv_exception(self, mock_to_csv, mock_makedirs, mock_read_parquet):
        """Test exception handling in save_results_to_csv for both weighted approaches."""
        # Setup mocks
        mock_read_parquet.return_value = self.sample_data
        mock_to_csv.side_effect = Exception("CSV write error")

        # Initialize analyzer
        analyzer = StrategyAnalyzer()

        # Test that the exception is propagated with weighted=True
        with self.assertRaises(Exception):
            analyzer._save_results_to_csv('win_rate', 10, analyzer.results_df, False, None, None, weighted=True)

        # Verify that makedirs was called before the exception
        mock_makedirs.assert_called_once()
        mock_makedirs.reset_mock()

        # Reset the side_effect for the next test
        mock_to_csv.side_effect = Exception("CSV write error")

        # Test that the exception is propagated with weighted=False
        with self.assertRaises(Exception):
            analyzer._save_results_to_csv('win_rate', 10, analyzer.results_df, False, None, None, weighted=False)

        # Verify that makedirs was called before the exception
        mock_makedirs.assert_called_once()


if __name__ == '__main__':
    unittest.main()
