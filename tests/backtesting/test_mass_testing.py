from unittest.mock import patch, MagicMock, mock_open

import pandas as pd
import pytest

from app.backtesting.mass_testing import MassTester, _load_existing_results, _test_already_exists


class TestMassTester:
    """Tests for the MassTester class."""

    def test_initialization(self):
        """Test that the MassTester initializes correctly with the provided parameters."""
        # Test with minimal parameters
        tested_months = ['2023-01']
        symbols = ['ES']
        intervals = ['1h']

        tester = MassTester(tested_months, symbols, intervals)

        assert tester.tested_months == tested_months
        assert tester.symbols == symbols
        assert tester.intervals == intervals
        assert tester.strategies == []
        assert tester.results == []
        assert hasattr(tester, 'switch_dates_dict')

        # Test with multiple values
        tested_months = ['2023-01', '2023-02', '2023-03']
        symbols = ['ES', 'NQ', 'YM']
        intervals = ['1h', '4h', '1d']

        tester = MassTester(tested_months, symbols, intervals)

        assert tester.tested_months == tested_months
        assert tester.symbols == symbols
        assert tester.intervals == intervals
        assert tester.strategies == []
        assert tester.results == []

    @patch('yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open, read_data='{}')
    def test_initialization_with_switch_dates(self, mock_file, mock_yaml_load):
        """Test that the MassTester loads switch dates correctly."""
        mock_yaml_load.return_value = {'ES': ['2023-01-15', '2023-02-15']}

        tester = MassTester(['2023-01'], ['ES'], ['1h'])

        mock_file.assert_called_once()
        mock_yaml_load.assert_called_once()
        assert tester.switch_dates_dict == {'ES': ['2023-01-15', '2023-02-15']}

    @patch('app.backtesting.mass_testing._load_existing_results')
    @patch('app.backtesting.mass_testing._test_already_exists')
    @patch('concurrent.futures.as_completed')
    @patch('concurrent.futures.ProcessPoolExecutor')
    def test_run_tests_basic(self, mock_executor, mock_as_completed, mock_test_exists, mock_load_results):
        """Test the basic functionality of run_tests."""
        # Setup mocks
        mock_load_results.return_value = pd.DataFrame()
        mock_test_exists.return_value = False

        # Create a mock result that will be returned by the future
        mock_result = {
            'month': '2023-01',
            'symbol': 'ES',
            'interval': '1h',
            'strategy': 'Test Strategy',
            'metrics': {'total_trades': 10, 'win_rate': 60.0},
            'timestamp': '2023-01-01T00:00:00'
        }

        # Mock the future
        mock_future = MagicMock()
        mock_future.result.return_value = mock_result

        # Mock the executor instance
        mock_executor_instance = MagicMock()
        mock_executor_instance.__enter__.return_value.submit.return_value = mock_future
        mock_executor.return_value = mock_executor_instance

        # Mock as_completed to return our mock_future
        mock_as_completed.return_value = [mock_future]

        # Create a tester with a strategy
        tester = MassTester(['2023-01'], ['ES'], ['1h'])
        tester.add_strategy_tests(
            strategy_type='ema',
            param_grid={'ema_short': [9], 'ema_long': [21], 'rollover': [True], 'trailing': [None]}
        )

        # Run tests
        results = tester.run_tests(verbose=False)

        # Verify the results
        assert len(results) == 1
        assert results[0]['month'] == '2023-01'
        assert results[0]['symbol'] == 'ES'
        assert results[0]['interval'] == '1h'
        assert results[0]['strategy'] == 'Test Strategy'
        assert results[0]['metrics'] == {'total_trades': 10, 'win_rate': 60.0}

        # Verify the mocks were called correctly
        mock_load_results.assert_called_once()
        mock_executor.assert_called_once()
        mock_executor_instance.__enter__.return_value.submit.assert_called_once()
        mock_as_completed.assert_called_once()

    @patch('app.backtesting.mass_testing._load_existing_results')
    def test_run_tests_no_strategies(self, mock_load_results):
        """Test that run_tests raises an error when no strategies are added."""
        mock_load_results.return_value = pd.DataFrame()

        tester = MassTester(['2023-01'], ['ES'], ['1h'])

        with pytest.raises(ValueError, match='No strategies added for testing'):
            tester.run_tests()

    @patch('app.backtesting.mass_testing._load_existing_results')
    @patch('app.backtesting.mass_testing._test_already_exists')
    def test_run_tests_all_skipped(self, mock_test_exists, mock_load_results):
        """Test that run_tests handles the case where all tests are skipped."""
        mock_load_results.return_value = pd.DataFrame()
        mock_test_exists.return_value = True  # All tests already exist

        tester = MassTester(['2023-01'], ['ES'], ['1h'])
        tester.add_strategy_tests(
            strategy_type='ema',
            param_grid={'ema_short': [9], 'ema_long': [21], 'rollover': [True], 'trailing': [None]}
        )

        results = tester.run_tests(verbose=False)

        assert results == []
        mock_test_exists.assert_called()

    @patch('app.backtesting.mass_testing.get_preprocessed_dataframe')
    def test_run_single_test(self, mock_get_df):
        """Test the _run_single_test method."""
        # Setup mock
        mock_df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104]
        })
        mock_get_df.return_value = mock_df

        # Create a strategy that will generate a trade
        strategy = MagicMock()
        strategy.run.return_value = [
            {
                'entry_time': pd.Timestamp('2023-01-01'),
                'exit_time': pd.Timestamp('2023-01-02'),
                'entry_price': 100,
                'exit_price': 110,
                'side': 'long'
            }
        ]

        # Create a tester
        tester = MassTester(['2023-01'], ['ES'], ['1h'])

        # Run a single test
        result = tester._run_single_test(('2023-01', 'ES', '1h', 'Test Strategy', strategy, False))

        # Verify the result
        assert result is not None
        assert result['month'] == '2023-01'
        assert result['symbol'] == 'ES'
        assert result['interval'] == '1h'
        assert result['strategy'] == 'Test Strategy'
        assert 'metrics' in result
        assert 'timestamp' in result

        # Verify the strategy was called correctly
        strategy.run.assert_called_once_with(mock_df, [])

    @patch('app.backtesting.mass_testing.get_preprocessed_dataframe')
    def test_run_single_test_no_trades(self, mock_get_df):
        """Test the _run_single_test method when no trades are generated."""
        # Setup mock
        mock_df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104]
        })
        mock_get_df.return_value = mock_df

        # Create a strategy that will not generate any trades
        strategy = MagicMock()
        strategy.run.return_value = []

        # Create a tester
        tester = MassTester(['2023-01'], ['ES'], ['1h'])

        # Run a single test
        result = tester._run_single_test(('2023-01', 'ES', '1h', 'Test Strategy', strategy, False))

        # Verify the result
        assert result is not None
        assert result['month'] == '2023-01'
        assert result['symbol'] == 'ES'
        assert result['interval'] == '1h'
        assert result['strategy'] == 'Test Strategy'
        assert result['metrics'] == {}  # Empty metrics
        assert 'timestamp' in result

    @patch('app.backtesting.mass_testing.get_preprocessed_dataframe')
    def test_run_single_test_file_error(self, mock_get_df):
        """Test the _run_single_test method when there's an error reading the file."""
        # Setup mock to raise an exception
        mock_get_df.side_effect = Exception("File not found")

        # Create a tester
        tester = MassTester(['2023-01'], ['ES'], ['1h'])

        # Run a single test
        result = tester._run_single_test(('2023-01', 'ES', '1h', 'Test Strategy', MagicMock(), False))

        # Verify the result
        assert result is None

    def test_results_to_dataframe(self):
        """Test the _results_to_dataframe method."""
        # Create a tester with some results
        tester = MassTester(['2023-01'], ['ES'], ['1h'])
        tester.results = [
            {
                'month': '2023-01',
                'symbol': 'ES',
                'interval': '1h',
                'strategy': 'Strategy 1',
                'metrics': {
                    'total_trades': 10,
                    'win_rate': 60.0,
                    'profit_factor': 1.5,
                    'total_return_percentage_of_margin': 5.0,
                    'average_trade_return_percentage_of_margin': 0.5,
                    'average_win_percentage_of_margin': 1.0,
                    'average_loss_percentage_of_margin': -0.5,
                    'maximum_drawdown_percentage': 2.0,
                    'total_net_pnl': 1000.0,
                    'avg_trade_net_pnl': 100.0,
                    'max_consecutive_wins': 3,
                    'max_consecutive_losses': 1,
                    'sharpe_ratio': 1.2,
                    'sortino_ratio': 1.5,
                    'calmar_ratio': 2.5
                },
                'timestamp': '2023-01-01T00:00:00'
            },
            {
                'month': '2023-01',
                'symbol': 'ES',
                'interval': '1h',
                'strategy': 'Strategy 2',
                'metrics': {
                    'total_trades': 5,
                    'win_rate': 40.0,
                    'profit_factor': 0.8,
                    'total_return_percentage_of_margin': -2.0,
                    'average_trade_return_percentage_of_margin': -0.4,
                    'average_win_percentage_of_margin': 0.8,
                    'average_loss_percentage_of_margin': -1.0,
                    'maximum_drawdown_percentage': 3.0,
                    'total_net_pnl': -500.0,
                    'avg_trade_net_pnl': -100.0,
                    'max_consecutive_wins': 1,
                    'max_consecutive_losses': 2,
                    'sharpe_ratio': -0.8,
                    'sortino_ratio': -1.0,
                    'calmar_ratio': -0.7
                },
                'timestamp': '2023-01-01T00:00:00'
            }
        ]

        # Convert results to DataFrame
        df = tester._results_to_dataframe()

        # Verify the DataFrame
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == [
            # Basic info
            'month',
            'symbol',
            'interval',
            'strategy',
            'total_trades',
            'win_rate',

            # Percentage-based metrics
            'total_return_percentage_of_margin',
            'average_trade_return_percentage_of_margin',
            'average_win_percentage_of_margin',
            'average_loss_percentage_of_margin',
            'commission_percentage_of_margin',

            # Risk metrics
            'profit_factor',
            'maximum_drawdown_percentage',
            'return_to_drawdown_ratio',
            'sharpe_ratio',
            'sortino_ratio',
            'calmar_ratio'
        ]

        # Verify the values
        # Basic info
        assert df.iloc[0]['strategy'] == 'Strategy 1'
        # Trade counts
        assert df.iloc[0]['total_trades'] == 10
        assert df.iloc[0]['win_rate'] == 60.0
        # Percentage-based metrics
        assert df.iloc[0]['total_return_percentage_of_margin'] == 5.0
        assert df.iloc[0]['average_trade_return_percentage_of_margin'] == 0.5
        assert df.iloc[0]['average_win_percentage_of_margin'] == 1.0
        assert df.iloc[0]['average_loss_percentage_of_margin'] == -0.5
        # Risk metrics
        assert df.iloc[0]['profit_factor'] == 1.5
        assert df.iloc[0]['maximum_drawdown_percentage'] == 2.0
        assert df.iloc[0]['return_to_drawdown_ratio'] >= 0
        assert df.iloc[0]['sharpe_ratio'] == 1.2
        assert df.iloc[0]['sortino_ratio'] == 1.5
        assert df.iloc[0]['calmar_ratio'] == 2.5

        # Basic info
        assert df.iloc[1]['strategy'] == 'Strategy 2'
        # Trade counts
        assert df.iloc[1]['total_trades'] == 5
        assert df.iloc[1]['win_rate'] == 40.0
        # Percentage-based metrics
        assert df.iloc[1]['total_return_percentage_of_margin'] == -2.0
        assert df.iloc[1]['average_trade_return_percentage_of_margin'] == -0.4
        assert df.iloc[1]['average_win_percentage_of_margin'] == 0.8
        assert df.iloc[1]['average_loss_percentage_of_margin'] == -1.0
        # Risk metrics
        assert df.iloc[1]['profit_factor'] == 0.8
        assert df.iloc[1]['maximum_drawdown_percentage'] == 3.0
        assert df.iloc[1]['return_to_drawdown_ratio'] <= 0
        assert df.iloc[1]['sharpe_ratio'] == -0.8
        assert df.iloc[1]['sortino_ratio'] == -1.0
        assert df.iloc[1]['calmar_ratio'] == -0.7

    def test_results_to_dataframe_empty(self):
        """Test the _results_to_dataframe method with empty results."""
        tester = MassTester(['2023-01'], ['ES'], ['1h'])
        tester.results = []

        df = tester._results_to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    @patch('app.backtesting.mass_testing.save_to_parquet')
    def test_save_results(self, mock_save_to_parquet):
        """Test the _save_results method."""
        # Create a tester with some results
        tester = MassTester(['2023-01'], ['ES'], ['1h'])
        tester.results = [
            {
                'month': '2023-01',
                'symbol': 'ES',
                'interval': '1h',
                'strategy': 'Strategy 1',
                'metrics': {'total_trades': 10, 'win_rate': 60.0},
                'timestamp': '2023-01-01T00:00:00'
            }
        ]

        # Save results
        tester._save_results()

        # Verify save_to_parquet was called
        mock_save_to_parquet.assert_called_once()

        # Verify the DataFrame passed to save_to_parquet
        df_arg = mock_save_to_parquet.call_args[0][0]
        assert isinstance(df_arg, pd.DataFrame)
        assert len(df_arg) == 1
        assert df_arg.iloc[0]['strategy'] == 'Strategy 1'

    @patch('app.backtesting.mass_testing.save_to_parquet')
    def test_save_results_empty(self, mock_save_to_parquet):
        """Test the _save_results method with empty results."""
        tester = MassTester(['2023-01'], ['ES'], ['1h'])
        tester.results = []

        tester._save_results()

        # Verify save_to_parquet was not called
        mock_save_to_parquet.assert_not_called()


class TestHelperFunctions:
    """Tests for the helper functions in mass_testing.py."""

    @patch('os.path.exists')
    @patch('pandas.read_parquet')
    def test_load_existing_results_file_exists(self, mock_read_parquet, mock_exists):
        """Test _load_existing_results when the file exists."""
        mock_exists.return_value = True
        mock_df = pd.DataFrame({'strategy': ['Test']})
        mock_read_parquet.return_value = mock_df

        result = _load_existing_results()

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert result.iloc[0]['strategy'] == 'Test'
        mock_exists.assert_called_once()
        mock_read_parquet.assert_called_once()

    @patch('os.path.exists')
    @patch('pandas.read_parquet')
    def test_load_existing_results_file_not_exists(self, mock_read_parquet, mock_exists):
        """Test _load_existing_results when the file doesn't exist."""
        mock_exists.return_value = False

        result = _load_existing_results()

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        mock_exists.assert_called_once()
        mock_read_parquet.assert_not_called()

    @patch('os.path.exists')
    @patch('pandas.read_parquet')
    def test_load_existing_results_error(self, mock_read_parquet, mock_exists):
        """Test _load_existing_results when there's an error reading the file."""
        mock_exists.return_value = True
        mock_read_parquet.side_effect = Exception("Error reading file")

        result = _load_existing_results()

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        mock_exists.assert_called_once()
        mock_read_parquet.assert_called_once()

    def test_test_already_exists_match(self):
        """Test _test_already_exists when there's a match."""
        existing_results = pd.DataFrame({
            'month': ['2023-01', '2023-02'],
            'symbol': ['ES', 'NQ'],
            'interval': ['1h', '4h'],
            'strategy': ['Strategy 1', 'Strategy 2']
        })

        result = _test_already_exists(existing_results, '2023-01', 'ES', '1h', 'Strategy 1')

        assert result == True

    def test_test_already_exists_no_match(self):
        """Test _test_already_exists when there's no match."""
        existing_results = pd.DataFrame({
            'month': ['2023-01', '2023-02'],
            'symbol': ['ES', 'NQ'],
            'interval': ['1h', '4h'],
            'strategy': ['Strategy 1', 'Strategy 2']
        })

        result = _test_already_exists(existing_results, '2023-01', 'ES', '1h', 'Strategy 3')

        assert result == False

    def test_test_already_exists_empty_results(self):
        """Test _test_already_exists with empty results."""
        existing_results = pd.DataFrame()

        result = _test_already_exists(existing_results, '2023-01', 'ES', '1h', 'Strategy 1')

        assert result is False
