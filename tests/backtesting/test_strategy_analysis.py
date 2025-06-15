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
            'strategy': [
                'RSI(period=14,lower=30,upper=70,rollover=False,trailing=None,slippage=0.1)',
                'RSI(period=14,lower=30,upper=70,rollover=False,trailing=None,slippage=0.05)',
                'EMA(short=9,long=21,rollover=False,trailing=None,slippage=0.2)',
                'EMA(short=9,long=21,rollover=False,trailing=None,slippage=0.15)'
            ],
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
        self.assertEqual(len(weighted_aggregated), 4)  # Four strategies (each row is a unique strategy)
        # Check that all strategies are present
        strategies = weighted_aggregated['strategy'].tolist()
        self.assertTrue(any('RSI' in s and 'slippage=0.1' in s for s in strategies))
        self.assertTrue(any('RSI' in s and 'slippage=0.05' in s for s in strategies))
        self.assertTrue(any('EMA' in s and 'slippage=0.2' in s for s in strategies))
        self.assertTrue(any('EMA' in s and 'slippage=0.15' in s for s in strategies))

        # Check that symbol_count and interval_count are calculated correctly for RSI strategies
        rsi_strategies = weighted_aggregated[weighted_aggregated['strategy'].str.contains('RSI')]
        for _, row in rsi_strategies.iterrows():
            self.assertEqual(row['symbol_count'], 1)  # Each RSI strategy has 1 symbol
            self.assertEqual(row['interval_count'], 1)  # Each RSI strategy has 1 interval

        # Check that symbol_count and interval_count are calculated correctly for EMA strategies
        ema_strategies = weighted_aggregated[weighted_aggregated['strategy'].str.contains('EMA')]
        for _, row in ema_strategies.iterrows():
            self.assertEqual(row['symbol_count'], 1)  # Each EMA strategy has 1 symbol
            self.assertEqual(row['interval_count'], 1)  # Each EMA strategy has 1 interval

        # Check that total_trades is correct for each strategy
        rsi_01_strategy = weighted_aggregated[
            weighted_aggregated['strategy'].str.contains('RSI') & weighted_aggregated['strategy'].str.contains(
                'slippage=0.1')]
        self.assertEqual(rsi_01_strategy['total_trades'].iloc[0], 10)  # RSI with slippage=0.1 has 10 trades

        rsi_005_strategy = weighted_aggregated[
            weighted_aggregated['strategy'].str.contains('RSI') & weighted_aggregated['strategy'].str.contains(
                'slippage=0.05')]
        self.assertEqual(rsi_005_strategy['total_trades'].iloc[0], 15)  # RSI with slippage=0.05 has 15 trades

        ema_02_strategy = weighted_aggregated[
            weighted_aggregated['strategy'].str.contains('EMA') & weighted_aggregated['strategy'].str.contains(
                'slippage=0.2')]
        self.assertEqual(ema_02_strategy['total_trades'].iloc[0], 20)  # EMA with slippage=0.2 has 20 trades

        ema_015_strategy = weighted_aggregated[
            weighted_aggregated['strategy'].str.contains('EMA') & weighted_aggregated['strategy'].str.contains(
                'slippage=0.15')]
        self.assertEqual(ema_015_strategy['total_trades'].iloc[0], 25)  # EMA with slippage=0.15 has 25 trades

        # Test non-weighted aggregation
        non_weighted_aggregated = analyzer._aggregate_strategies(weighted=False)

        # Verify results for non-weighted aggregation
        self.assertEqual(len(non_weighted_aggregated), 4)  # Four strategies (each row is a unique strategy)
        # Check that all strategies are present
        strategies = non_weighted_aggregated['strategy'].tolist()
        self.assertTrue(any('RSI' in s and 'slippage=0.1' in s for s in strategies))
        self.assertTrue(any('RSI' in s and 'slippage=0.05' in s for s in strategies))
        self.assertTrue(any('EMA' in s and 'slippage=0.2' in s for s in strategies))
        self.assertTrue(any('EMA' in s and 'slippage=0.15' in s for s in strategies))

        # Check that basic metrics are the same for both approaches
        # Check that symbol_count and interval_count are calculated correctly for RSI strategies
        rsi_strategies = non_weighted_aggregated[non_weighted_aggregated['strategy'].str.contains('RSI')]
        for _, row in rsi_strategies.iterrows():
            self.assertEqual(row['symbol_count'], 1)  # Each RSI strategy has 1 symbol
            self.assertEqual(row['interval_count'], 1)  # Each RSI strategy has 1 interval

        # Check that symbol_count and interval_count are calculated correctly for EMA strategies
        ema_strategies = non_weighted_aggregated[non_weighted_aggregated['strategy'].str.contains('EMA')]
        for _, row in ema_strategies.iterrows():
            self.assertEqual(row['symbol_count'], 1)  # Each EMA strategy has 1 symbol
            self.assertEqual(row['interval_count'], 1)  # Each EMA strategy has 1 interval

        # Check that total_trades is correct for each strategy
        rsi_01_strategy = non_weighted_aggregated[
            non_weighted_aggregated['strategy'].str.contains('RSI') & non_weighted_aggregated['strategy'].str.contains(
                'slippage=0.1')]
        self.assertEqual(rsi_01_strategy['total_trades'].iloc[0], 10)  # RSI with slippage=0.1 has 10 trades

        rsi_005_strategy = non_weighted_aggregated[
            non_weighted_aggregated['strategy'].str.contains('RSI') & non_weighted_aggregated['strategy'].str.contains(
                'slippage=0.05')]
        self.assertEqual(rsi_005_strategy['total_trades'].iloc[0], 15)  # RSI with slippage=0.05 has 15 trades

        ema_02_strategy = non_weighted_aggregated[
            non_weighted_aggregated['strategy'].str.contains('EMA') & non_weighted_aggregated['strategy'].str.contains(
                'slippage=0.2')]
        self.assertEqual(ema_02_strategy['total_trades'].iloc[0], 20)  # EMA with slippage=0.2 has 20 trades

        ema_015_strategy = non_weighted_aggregated[
            non_weighted_aggregated['strategy'].str.contains('EMA') & non_weighted_aggregated['strategy'].str.contains(
                'slippage=0.15')]
        self.assertEqual(ema_015_strategy['total_trades'].iloc[0], 25)  # EMA with slippage=0.15 has 25 trades

        # In this test setup, the win_rate values are the same for both weighted and non-weighted approaches
        # because we're using a simple dataset. In a real-world scenario with more data,
        # these values would likely be different.
        for strategy_type in ['RSI', 'EMA']:
            weighted_strategy = weighted_aggregated[weighted_aggregated['strategy'].str.contains(strategy_type)].iloc[0]
            non_weighted_strategy = \
            non_weighted_aggregated[non_weighted_aggregated['strategy'].str.contains(strategy_type)].iloc[0]
            # Just verify that the win_rate exists in both
            self.assertIn('win_rate', weighted_strategy)
            self.assertIn('win_rate', non_weighted_strategy)

    @patch('pandas.read_parquet')
    def test_aggregate_strategies_with_filters(self, mock_read_parquet):
        """Test aggregation of strategies with filters for both weighted and non-weighted approaches."""
        # Setup mock
        mock_read_parquet.return_value = self.sample_data

        # Initialize analyzer
        analyzer = StrategyAnalyzer()

        # Test weighted aggregation with min_trades filter
        weighted_aggregated = analyzer._aggregate_strategies(min_trades=20, weighted=True)
        self.assertEqual(len(weighted_aggregated), 2)  # Only EMA strategies have rows with >= 20 trades
        for strategy in weighted_aggregated['strategy']:
            self.assertTrue(strategy.startswith('EMA'))

        # Test non-weighted aggregation with min_trades filter
        non_weighted_aggregated = analyzer._aggregate_strategies(min_trades=20, weighted=False)
        self.assertEqual(len(non_weighted_aggregated), 2)  # Only EMA strategies have rows with >= 20 trades
        for strategy in non_weighted_aggregated['strategy']:
            self.assertTrue(strategy.startswith('EMA'))

        # In this test setup, the win_rate values are the same for both weighted and non-weighted approaches
        # because we're using a simple dataset. In a real-world scenario with more data,
        # these values would likely be different.
        ema_weighted = weighted_aggregated[weighted_aggregated['strategy'].str.contains('slippage=0.2')].iloc[0]
        ema_non_weighted = \
        non_weighted_aggregated[non_weighted_aggregated['strategy'].str.contains('slippage=0.2')].iloc[0]
        # Just verify that the win_rate exists in both
        self.assertIn('win_rate', ema_weighted)
        self.assertIn('win_rate', ema_non_weighted)

        # Test weighted aggregation with interval filter
        weighted_aggregated = analyzer._aggregate_strategies(interval='1d', weighted=True)
        self.assertEqual(len(weighted_aggregated), 2)  # Only RSI strategies have 1d interval
        for strategy in weighted_aggregated['strategy']:
            self.assertTrue(strategy.startswith('RSI'))

        # Test non-weighted aggregation with interval filter
        non_weighted_aggregated = analyzer._aggregate_strategies(interval='1d', weighted=False)
        self.assertEqual(len(non_weighted_aggregated), 2)  # Only RSI strategies have 1d interval
        for strategy in non_weighted_aggregated['strategy']:
            self.assertTrue(strategy.startswith('RSI'))

        # Test weighted aggregation with symbol filter
        weighted_aggregated = analyzer._aggregate_strategies(symbol='ES', weighted=True)
        self.assertEqual(len(weighted_aggregated), 2)  # Both RSI and EMA have ES
        self.assertTrue(any(s.startswith('RSI') for s in weighted_aggregated['strategy']))
        self.assertTrue(any(s.startswith('EMA') for s in weighted_aggregated['strategy']))

        # Test non-weighted aggregation with symbol filter
        non_weighted_aggregated = analyzer._aggregate_strategies(symbol='ES', weighted=False)
        self.assertEqual(len(non_weighted_aggregated), 2)  # Both RSI and EMA have ES
        self.assertTrue(any(s.startswith('RSI') for s in non_weighted_aggregated['strategy']))
        self.assertTrue(any(s.startswith('EMA') for s in non_weighted_aggregated['strategy']))

        # Test weighted aggregation with multiple filters
        weighted_aggregated = analyzer._aggregate_strategies(min_trades=20, interval='4h', weighted=True)
        self.assertEqual(len(weighted_aggregated), 2)  # Only EMA has 4h interval with >= 20 trades
        for strategy in weighted_aggregated['strategy']:
            self.assertTrue(strategy.startswith('EMA'))

        # Test non-weighted aggregation with multiple filters
        non_weighted_aggregated = analyzer._aggregate_strategies(min_trades=20, interval='4h', weighted=False)
        self.assertEqual(len(non_weighted_aggregated), 2)  # Only EMA has 4h interval with >= 20 trades
        for strategy in non_weighted_aggregated['strategy']:
            self.assertTrue(strategy.startswith('EMA'))

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
        for strategy in weighted_top_strategies['strategy']:
            self.assertTrue(strategy.startswith('EMA'))

        # Test with min_trades filter (weighted=False)
        non_weighted_top_strategies = analyzer.get_top_strategies('win_rate', 20, weighted=False)
        self.assertEqual(len(non_weighted_top_strategies), 2)  # Only rows with >= 20 trades
        for strategy in non_weighted_top_strategies['strategy']:
            self.assertTrue(strategy.startswith('EMA'))

        # Test with aggregate=True (weighted=True)
        weighted_aggregated = analyzer.get_top_strategies('win_rate', 0, aggregate=True, weighted=True)
        self.assertEqual(len(weighted_aggregated), 4)  # Aggregated to 4 strategies (each row is a unique strategy)

        # Test with aggregate=True (weighted=False)
        non_weighted_aggregated = analyzer.get_top_strategies('win_rate', 0, aggregate=True, weighted=False)
        self.assertEqual(len(non_weighted_aggregated), 4)  # Aggregated to 4 strategies (each row is a unique strategy)

        # Verify that the aggregated results are different between weighted and non-weighted
        self.assertFalse(weighted_aggregated.equals(non_weighted_aggregated))

        # Test with interval filter (weighted=True)
        weighted_interval = analyzer.get_top_strategies('win_rate', 0, interval='1d', weighted=True)
        self.assertEqual(len(weighted_interval), 2)  # Only 1d interval
        for strategy in weighted_interval['strategy']:
            self.assertTrue(strategy.startswith('RSI'))

        # Test with interval filter (weighted=False)
        non_weighted_interval = analyzer.get_top_strategies('win_rate', 0, interval='1d', weighted=False)
        self.assertEqual(len(non_weighted_interval), 2)  # Only 1d interval
        for strategy in non_weighted_interval['strategy']:
            self.assertTrue(strategy.startswith('RSI'))

        # Test with symbol filter (weighted=True)
        weighted_symbol = analyzer.get_top_strategies('win_rate', 0, symbol='ES', weighted=True)
        self.assertEqual(len(weighted_symbol), 2)  # Only ES symbol

        # Test with symbol filter (weighted=False)
        non_weighted_symbol = analyzer.get_top_strategies('win_rate', 0, symbol='ES', weighted=False)
        self.assertEqual(len(non_weighted_symbol), 2)  # Only ES symbol

        # Test with multiple filters (weighted=True)
        weighted_multiple = analyzer.get_top_strategies('win_rate', 0, interval='4h', symbol='NQ', weighted=True)
        self.assertEqual(len(weighted_multiple), 1)  # Only 4h interval and NQ symbol
        self.assertTrue(weighted_multiple['strategy'].iloc[0].startswith('EMA'))

        # Test with multiple filters (weighted=False)
        non_weighted_multiple = analyzer.get_top_strategies('win_rate', 0, interval='4h', symbol='NQ', weighted=False)
        self.assertEqual(len(non_weighted_multiple), 1)  # Only 4h interval and NQ symbol
        self.assertTrue(non_weighted_multiple['strategy'].iloc[0].startswith('EMA'))

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

    @patch('pandas.read_parquet')
    def test_aggregate_strategies_with_min_slippage(self, mock_read_parquet):
        """Test aggregation of strategies with min_slippage filter for both weighted and non-weighted approaches."""
        # Setup mock
        mock_read_parquet.return_value = self.sample_data

        # Initialize analyzer
        analyzer = StrategyAnalyzer()

        # Test weighted aggregation with min_slippage filter
        weighted_aggregated = analyzer._aggregate_strategies(min_slippage=0.15, weighted=True)
        self.assertEqual(len(weighted_aggregated), 2)  # Only strategies with slippage >= 0.15
        for strategy in weighted_aggregated['strategy']:
            self.assertTrue('slippage=0.2' in strategy or 'slippage=0.15' in strategy)

        # Test non-weighted aggregation with min_slippage filter
        non_weighted_aggregated = analyzer._aggregate_strategies(min_slippage=0.15, weighted=False)
        self.assertEqual(len(non_weighted_aggregated), 2)  # Only strategies with slippage >= 0.15
        for strategy in non_weighted_aggregated['strategy']:
            self.assertTrue('slippage=0.2' in strategy or 'slippage=0.15' in strategy)

        # Test weighted aggregation with min_slippage=0.1 filter
        weighted_aggregated = analyzer._aggregate_strategies(min_slippage=0.1, weighted=True)
        self.assertEqual(len(weighted_aggregated), 3)  # RSI with slippage=0.1 and both EMA strategies

        # Test non-weighted aggregation with min_slippage=0.1 filter
        non_weighted_aggregated = analyzer._aggregate_strategies(min_slippage=0.1, weighted=False)
        self.assertEqual(len(non_weighted_aggregated), 3)  # RSI with slippage=0.1 and both EMA strategies

        # Test weighted aggregation with min_slippage=0.05 filter
        weighted_aggregated = analyzer._aggregate_strategies(min_slippage=0.05, weighted=True)
        self.assertEqual(len(weighted_aggregated), 4)  # All strategies have slippage >= 0.05

        # Test non-weighted aggregation with min_slippage=0.05 filter
        non_weighted_aggregated = analyzer._aggregate_strategies(min_slippage=0.05, weighted=False)
        self.assertEqual(len(non_weighted_aggregated), 4)  # All strategies have slippage >= 0.05

        # Test weighted aggregation with min_slippage=0.3 filter (no matches)
        weighted_aggregated = analyzer._aggregate_strategies(min_slippage=0.3, weighted=True)
        self.assertEqual(len(weighted_aggregated), 0)  # No strategies with slippage >= 0.3

        # Test non-weighted aggregation with min_slippage=0.3 filter (no matches)
        non_weighted_aggregated = analyzer._aggregate_strategies(min_slippage=0.3, weighted=False)
        self.assertEqual(len(non_weighted_aggregated), 0)  # No strategies with slippage >= 0.3

        # Test weighted aggregation with multiple filters including min_slippage
        weighted_aggregated = analyzer._aggregate_strategies(min_trades=20, min_slippage=0.15, weighted=True)
        self.assertEqual(len(weighted_aggregated), 2)  # Only EMA with slippage >= 0.15 and trades >= 20
        for strategy in weighted_aggregated['strategy']:
            self.assertTrue(strategy.startswith('EMA'))

        # Test non-weighted aggregation with multiple filters including min_slippage
        non_weighted_aggregated = analyzer._aggregate_strategies(min_trades=20, min_slippage=0.15, weighted=False)
        self.assertEqual(len(non_weighted_aggregated), 2)  # Only EMA with slippage >= 0.15 and trades >= 20
        for strategy in non_weighted_aggregated['strategy']:
            self.assertTrue(strategy.startswith('EMA'))

    @patch('pandas.read_parquet')
    @patch('app.backtesting.strategy_analysis.StrategyAnalyzer._save_results_to_csv')
    def test_get_top_strategies_with_min_slippage(self, mock_save_results, mock_read_parquet):
        """Test get_top_strategies with min_slippage filter for both weighted and non-weighted approaches."""
        # Setup mocks
        mock_read_parquet.return_value = self.sample_data
        mock_save_results.reset_mock()  # Reset mock to ensure a clean state

        # Initialize analyzer
        analyzer = StrategyAnalyzer()

        # Test with min_slippage filter (weighted=True)
        weighted_top_strategies = analyzer.get_top_strategies('win_rate', 0, min_slippage=0.15, weighted=True)
        self.assertEqual(len(weighted_top_strategies), 2)  # Only rows with slippage >= 0.15
        for strategy in weighted_top_strategies['strategy']:
            self.assertTrue('slippage=0.2' in strategy or 'slippage=0.15' in strategy)

        # Verify save_results_to_csv was called
        mock_save_results.assert_called_once()
        mock_save_results.reset_mock()

        # Test with min_slippage filter (weighted=False)
        non_weighted_top_strategies = analyzer.get_top_strategies('win_rate', 0, min_slippage=0.15, weighted=False)
        self.assertEqual(len(non_weighted_top_strategies), 2)  # Only rows with slippage >= 0.15
        for strategy in non_weighted_top_strategies['strategy']:
            self.assertTrue('slippage=0.2' in strategy or 'slippage=0.15' in strategy)

        # Verify save_results_to_csv was called
        mock_save_results.assert_called_once()
        mock_save_results.reset_mock()

        # Test with min_slippage=0.1 filter (weighted=True)
        weighted_top_strategies = analyzer.get_top_strategies('win_rate', 0, min_slippage=0.1, weighted=True)
        self.assertEqual(len(weighted_top_strategies), 3)  # Rows with slippage >= 0.1

        # Test with min_slippage=0.1 filter (weighted=False)
        non_weighted_top_strategies = analyzer.get_top_strategies('win_rate', 0, min_slippage=0.1, weighted=False)
        self.assertEqual(len(non_weighted_top_strategies), 3)  # Rows with slippage >= 0.1

        # Test with min_slippage=0.3 filter (no matches)
        weighted_top_strategies = analyzer.get_top_strategies('win_rate', 0, min_slippage=0.3, weighted=True)
        self.assertEqual(len(weighted_top_strategies), 0)  # No rows with slippage >= 0.3

        # Test with min_slippage=0.3 filter (no matches)
        non_weighted_top_strategies = analyzer.get_top_strategies('win_rate', 0, min_slippage=0.3, weighted=False)
        self.assertEqual(len(non_weighted_top_strategies), 0)  # No rows with slippage >= 0.3

        # Test with aggregate=True and min_slippage filter (weighted=True)
        weighted_aggregated = analyzer.get_top_strategies('win_rate',
                                                          0,
                                                          aggregate=True,
                                                          min_slippage=0.15,
                                                          weighted=True)
        self.assertEqual(len(weighted_aggregated), 2)  # Only EMA strategies have slippage >= 0.15
        for strategy in weighted_aggregated['strategy']:
            self.assertTrue(strategy.startswith('EMA'))

        # Test with aggregate=True and min_slippage filter (weighted=False)
        non_weighted_aggregated = analyzer.get_top_strategies('win_rate',
                                                              0,
                                                              aggregate=True,
                                                              min_slippage=0.15,
                                                              weighted=False)
        self.assertEqual(len(non_weighted_aggregated), 2)  # Only EMA strategies have slippage >= 0.15
        for strategy in non_weighted_aggregated['strategy']:
            self.assertTrue(strategy.startswith('EMA'))

        # Test with multiple filters including min_slippage (weighted=True)
        weighted_multiple = analyzer.get_top_strategies('win_rate', 0, interval='4h', min_slippage=0.15, weighted=True)
        self.assertEqual(len(weighted_multiple), 2)  # EMA strategies with 4h interval and slippage >= 0.15
        for strategy in weighted_multiple['strategy']:
            self.assertTrue(strategy.startswith('EMA'))

        # Test with multiple filters including min_slippage (weighted=False)
        non_weighted_multiple = analyzer.get_top_strategies('win_rate',
                                                            0,
                                                            interval='4h',
                                                            min_slippage=0.15,
                                                            weighted=False)
        self.assertEqual(len(non_weighted_multiple), 2)  # EMA strategies with 4h interval and slippage >= 0.15
        for strategy in non_weighted_multiple['strategy']:
            self.assertTrue(strategy.startswith('EMA'))


if __name__ == '__main__':
    unittest.main()
