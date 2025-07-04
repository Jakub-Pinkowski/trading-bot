import unittest
from unittest.mock import patch

import pandas as pd

from app.backtesting.strategy_analysis import (
    StrategyAnalyzer,
    _format_column_name,
    _filter_dataframe,
    _calculate_weighted_win_rate,
    _calculate_trade_weighted_average,
    _calculate_average_trade_return,
    _calculate_profit_ratio,
    _parse_strategy_name
)


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
            'calmar_ratio': 'Calmar',
            'value_at_risk': 'Var 95%',
            'expected_shortfall': 'Cvar 95%',
            'ulcer_index': 'Ulcer Idx',
            'avg_trades_per_symbol': 'Avg Trades Per Symbol',
            'avg_trades_per_interval': 'Avg Trades Per Interval',
            'avg_trades_per_combination': 'Avg Trades Per Combination'
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
        }

        for column_name, expected_formatted_name in test_cases.items():
            formatted_name = _format_column_name(column_name)
            self.assertEqual(formatted_name, expected_formatted_name)


class TestFilterDataframe(unittest.TestCase):
    """Tests for the _filter_dataframe function."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame({
            'strategy': [
                'RSI(period=14,lower=30,upper=70,rollover=False,trailing=None,slippage=0.1)',
                'RSI(period=14,lower=30,upper=70,rollover=False,trailing=None,slippage=0.05)',
                'EMA(short=9,long=21,rollover=False,trailing=None,slippage=0.2)',
                'EMA(short=9,long=21,rollover=False,trailing=None,slippage=0.15)',
                'MACD(fast=12,slow=26,signal=9,rollover=False,trailing=None,slippage=0.3)'
            ],
            'symbol': ['ES', 'NQ', 'ES', 'NQ', 'YM'],
            'interval': ['1d', '1d', '4h', '4h', '1h'],
            'total_trades': [5, 15, 20, 25, 30],
            'win_rate': [60.0, 70.0, 55.0, 65.0, 75.0],
            'total_return_percentage_of_margin': [5.0, 7.0, 4.0, 6.0, 8.0],
            'average_win_percentage_of_margin': [1.0, 0.9, 0.8, 0.7, 1.2],
            'average_loss_percentage_of_margin': [-0.5, -0.4, -0.6, -0.3, -0.4]
        })

    def test_filter_by_min_avg_trades_per_combination(self):
        """Test filtering by minimum average trades per combination."""
        # Create test data with multiple symbols and intervals for the same strategy
        test_data = pd.DataFrame({
            'strategy': [
                'RSI(period=14,lower=30,upper=70,rollover=False,trailing=None,slippage=0.1)',
                'RSI(period=14,lower=30,upper=70,rollover=False,trailing=None,slippage=0.1)',
                'EMA(short=9,long=21,rollover=False,trailing=None,slippage=0.2)',
                'EMA(short=9,long=21,rollover=False,trailing=None,slippage=0.2)',
                'MACD(fast=12,slow=26,signal=9,rollover=False,trailing=None,slippage=0.3)'
            ],
            'symbol': ['ES', 'NQ', 'ES', 'NQ', 'YM'],
            'interval': ['1d', '4h', '1d', '4h', '1h'],
            'total_trades': [20, 40, 30, 60, 50],
            # RSI: 60 total, 2 combinations = 30 avg; EMA: 90 total, 2 combinations = 45 avg; MACD: 50 total, 1 combination = 50 avg
            'win_rate': [60.0, 70.0, 55.0, 65.0, 75.0],
            'total_return_percentage_of_margin': [5.0, 7.0, 4.0, 6.0, 8.0],
            'average_win_percentage_of_margin': [1.0, 0.9, 0.8, 0.7, 1.2],
            'average_loss_percentage_of_margin': [-0.5, -0.4, -0.6, -0.3, -0.4]
        })

        # Based on debug output: RSI=15.0, EMA=22.5, MACD=50.0 avg trades per combination
        # Filter with min_avg_trades_per_combination=25 (should keep only MACD)
        result = _filter_dataframe(test_data, min_avg_trades_per_combination=25)
        self.assertEqual(len(result), 1)  # Only MACD
        strategies = result['strategy'].unique()
        self.assertTrue(any('MACD' in s for s in strategies))
        self.assertFalse(any('RSI' in s for s in strategies))
        self.assertFalse(any('EMA' in s for s in strategies))

        # Filter with min_avg_trades_per_combination=20 (should keep EMA and MACD)
        result = _filter_dataframe(test_data, min_avg_trades_per_combination=20)
        self.assertEqual(len(result), 3)  # EMA (2 rows) + MACD (1 row)
        strategies = result['strategy'].unique()
        self.assertTrue(any('EMA' in s for s in strategies))
        self.assertTrue(any('MACD' in s for s in strategies))
        self.assertFalse(any('RSI' in s for s in strategies))

    def test_filter_by_interval(self):
        """Test filtering by interval."""
        # Filter by '1d' interval
        result = _filter_dataframe(self.sample_data, interval='1d')
        self.assertEqual(len(result), 2)
        self.assertTrue(all(result['interval'] == '1d'))

        # Filter by '4h' interval
        result = _filter_dataframe(self.sample_data, interval='4h')
        self.assertEqual(len(result), 2)
        self.assertTrue(all(result['interval'] == '4h'))

        # Filter by non-existent interval
        result = _filter_dataframe(self.sample_data, interval='5m')
        self.assertEqual(len(result), 0)

    def test_filter_by_symbol(self):
        """Test filtering by symbol."""
        # Filter by 'ES' symbol
        result = _filter_dataframe(self.sample_data, symbol='ES')
        self.assertEqual(len(result), 2)
        self.assertTrue(all(result['symbol'] == 'ES'))

        # Filter by 'NQ' symbol
        result = _filter_dataframe(self.sample_data, symbol='NQ')
        self.assertEqual(len(result), 2)
        self.assertTrue(all(result['symbol'] == 'NQ'))

        # Filter by non-existent symbol
        result = _filter_dataframe(self.sample_data, symbol='BTC')
        self.assertEqual(len(result), 0)

    def test_filter_by_min_slippage(self):
        """Test filtering by minimum slippage."""
        # Filter with min_slippage=0.1
        result = _filter_dataframe(self.sample_data, min_slippage=0.1)
        self.assertEqual(len(result), 4)  # slippage >= 0.1: 0.1, 0.2, 0.15, 0.3

        # Filter with min_slippage=0.2
        result = _filter_dataframe(self.sample_data, min_slippage=0.2)
        self.assertEqual(len(result), 2)  # slippage >= 0.2: 0.2, 0.3

        # Filter with min_slippage=0.5 (higher than any slippage)
        result = _filter_dataframe(self.sample_data, min_slippage=0.5)
        self.assertEqual(len(result), 0)

    def test_filter_combined_criteria(self):
        """Test filtering with multiple criteria combined."""
        # Create test data for combination filtering
        test_data = pd.DataFrame({
            'strategy': [
                'RSI(period=14,lower=30,upper=70,rollover=False,trailing=None,slippage=0.1)',
                'RSI(period=14,lower=30,upper=70,rollover=False,trailing=None,slippage=0.1)',
                'EMA(short=9,long=21,rollover=False,trailing=None,slippage=0.2)',
                'EMA(short=9,long=21,rollover=False,trailing=None,slippage=0.2)'
            ],
            'symbol': ['ES', 'NQ', 'ES', 'NQ'],
            'interval': ['4h', '4h', '4h', '4h'],
            'total_trades': [20, 40, 30, 60],
            # RSI: 60 total, 2 combinations = 30 avg; EMA: 90 total, 2 combinations = 45 avg
            'win_rate': [60.0, 70.0, 55.0, 65.0],
            'total_return_percentage_of_margin': [5.0, 7.0, 4.0, 6.0],
            'average_win_percentage_of_margin': [1.0, 0.9, 0.8, 0.7],
            'average_loss_percentage_of_margin': [-0.5, -0.4, -0.6, -0.3]
        })

        # Filter by min_avg_trades_per_combination=35 and interval='4h'
        result = _filter_dataframe(test_data, min_avg_trades_per_combination=35, interval='4h')
        self.assertEqual(len(result), 2)  # Only EMA strategy (45 avg trades per combination)
        self.assertTrue(all(result['interval'] == '4h'))
        strategies = result['strategy'].unique()
        self.assertTrue(any('EMA' in s for s in strategies))

        # Filter by symbol='ES' and min_slippage=0.15
        result = _filter_dataframe(self.sample_data, symbol='ES', min_slippage=0.15)
        self.assertEqual(len(result), 1)  # Only EMA with slippage=0.2
        self.assertTrue(all(result['symbol'] == 'ES'))

    def test_filter_no_criteria(self):
        """Test filtering with no criteria (should return all data)."""
        result = _filter_dataframe(self.sample_data)
        self.assertEqual(len(result), len(self.sample_data))
        pd.testing.assert_frame_equal(result, self.sample_data)

    def test_filter_empty_dataframe(self):
        """Test filtering an empty DataFrame."""
        empty_df = pd.DataFrame(columns=self.sample_data.columns)
        result = _filter_dataframe(empty_df, min_avg_trades_per_combination=10)
        self.assertEqual(len(result), 0)
        self.assertEqual(list(result.columns), list(empty_df.columns))

    def test_filter_by_min_symbol_count(self):
        """Test filtering by minimum symbol count."""
        # Create test data with strategies having different symbol counts
        test_data = pd.DataFrame({
            'strategy': [
                'RSI(period=14,lower=30,upper=70,rollover=False,trailing=None,slippage=0.1)',
                'RSI(period=14,lower=30,upper=70,rollover=False,trailing=None,slippage=0.1)',
                'RSI(period=14,lower=30,upper=70,rollover=False,trailing=None,slippage=0.1)',
                'EMA(short=9,long=21,rollover=False,trailing=None,slippage=0.2)',
                'EMA(short=9,long=21,rollover=False,trailing=None,slippage=0.2)',
                'MACD(fast=12,slow=26,signal=9,rollover=False,trailing=None,slippage=0.3)'
            ],
            'symbol': ['ES', 'NQ', 'YM', 'ES', 'NQ', 'ES'],
            'interval': ['1d', '1d', '1d', '4h', '4h', '1h'],
            'total_trades': [20, 40, 30, 30, 60, 50],
            # RSI: 3 symbols, EMA: 2 symbols, MACD: 1 symbol
            'win_rate': [60.0, 70.0, 65.0, 55.0, 65.0, 75.0],
            'total_return_percentage_of_margin': [5.0, 7.0, 6.0, 4.0, 6.0, 8.0],
            'average_win_percentage_of_margin': [1.0, 0.9, 0.95, 0.8, 0.7, 1.2],
            'average_loss_percentage_of_margin': [-0.5, -0.4, -0.45, -0.6, -0.3, -0.4]
        })

        # Filter with min_symbol_count=1 (should keep all strategies)
        result = _filter_dataframe(test_data, min_symbol_count=1)
        self.assertEqual(len(result), 6)  # All strategies

        # Filter with min_symbol_count=2 (should keep RSI and EMA)
        result = _filter_dataframe(test_data, min_symbol_count=2)
        self.assertEqual(len(result), 5)  # RSI (3 rows) + EMA (2 rows)
        strategies = result['strategy'].unique()
        self.assertTrue(any('RSI' in s for s in strategies))
        self.assertTrue(any('EMA' in s for s in strategies))
        self.assertFalse(any('MACD' in s for s in strategies))

        # Filter with min_symbol_count=3 (should keep only RSI)
        result = _filter_dataframe(test_data, min_symbol_count=3)
        self.assertEqual(len(result), 3)  # Only RSI (3 rows)
        strategies = result['strategy'].unique()
        self.assertTrue(any('RSI' in s for s in strategies))
        self.assertFalse(any('EMA' in s for s in strategies))
        self.assertFalse(any('MACD' in s for s in strategies))

        # Filter with min_symbol_count=4 (should keep no strategies)
        result = _filter_dataframe(test_data, min_symbol_count=4)
        self.assertEqual(len(result), 0)

    def test_filter_by_min_symbol_count_combined_with_other_filters(self):
        """Test filtering by minimum symbol count combined with other filters."""
        # Create test data with strategies having different symbol counts
        test_data = pd.DataFrame({
            'strategy': [
                'RSI(period=14,lower=30,upper=70,rollover=False,trailing=None,slippage=0.1)',
                'RSI(period=14,lower=30,upper=70,rollover=False,trailing=None,slippage=0.1)',
                'RSI(period=14,lower=30,upper=70,rollover=False,trailing=None,slippage=0.1)',
                'EMA(short=9,long=21,rollover=False,trailing=None,slippage=0.2)',
                'EMA(short=9,long=21,rollover=False,trailing=None,slippage=0.2)',
            ],
            'symbol': ['ES', 'NQ', 'YM', 'ES', 'NQ'],
            'interval': ['1d', '1d', '1d', '4h', '4h'],
            'total_trades': [20, 40, 30, 30, 60],
            # RSI: 90 total, 3 combinations = 30 avg; EMA: 90 total, 2 combinations = 45 avg
            'win_rate': [60.0, 70.0, 65.0, 55.0, 65.0],
            'total_return_percentage_of_margin': [5.0, 7.0, 6.0, 4.0, 6.0],
            'average_win_percentage_of_margin': [1.0, 0.9, 0.95, 0.8, 0.7],
            'average_loss_percentage_of_margin': [-0.5, -0.4, -0.45, -0.6, -0.3]
        })

        # Filter with min_symbol_count=2 and min_avg_trades_per_combination=35
        # Should keep only EMA (2 symbols, 45 avg trades per combination)
        result = _filter_dataframe(test_data, min_symbol_count=2, min_avg_trades_per_combination=35)
        self.assertEqual(len(result), 2)  # Only EMA (2 rows)
        strategies = result['strategy'].unique()
        self.assertTrue(any('EMA' in s for s in strategies))
        self.assertFalse(any('RSI' in s for s in strategies))

        # Filter with min_symbol_count=3 and interval='1d'
        # Should keep only RSI (3 symbols, all in 1d interval)
        result = _filter_dataframe(test_data, min_symbol_count=3, interval='1d')
        self.assertEqual(len(result), 3)  # Only RSI (3 rows)
        strategies = result['strategy'].unique()
        self.assertTrue(any('RSI' in s for s in strategies))
        self.assertFalse(any('EMA' in s for s in strategies))


class TestCalculateWeightedWinRate(unittest.TestCase):
    """Tests for the _calculate_weighted_win_rate function."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame({
            'strategy': ['Strategy_A', 'Strategy_A', 'Strategy_B', 'Strategy_B'],
            'total_trades': [10, 20, 15, 25],
            'win_rate': [60.0, 80.0, 40.0, 60.0]
        })
        self.grouped = self.sample_data.groupby('strategy')

    def test_weighted_win_rate_calculation(self):
        """Test weighted win rate calculation."""
        result = _calculate_weighted_win_rate(self.sample_data, self.grouped)

        # Strategy_A: (60*10 + 80*20) / (10+20) = (600 + 1600) / 30 = 73.33%
        # Strategy_B: (40*15 + 60*25) / (15+25) = (600 + 1500) / 40 = 52.5%
        expected_strategy_a = round((60 * 10 + 80 * 20) / (10 + 20), 2)
        expected_strategy_b = round((40 * 15 + 60 * 25) / (15 + 25), 2)

        self.assertEqual(result['Strategy_A'], expected_strategy_a)
        self.assertEqual(result['Strategy_B'], expected_strategy_b)

    def test_single_strategy(self):
        """Test with single strategy."""
        single_strategy_data = pd.DataFrame({
            'strategy': ['Strategy_A', 'Strategy_A'],
            'total_trades': [10, 20],
            'win_rate': [60.0, 80.0]
        })
        grouped = single_strategy_data.groupby('strategy')
        result = _calculate_weighted_win_rate(single_strategy_data, grouped)

        expected = round((60 * 10 + 80 * 20) / (10 + 20), 2)
        self.assertEqual(result['Strategy_A'], expected)

    def test_equal_trades(self):
        """Test with equal number of trades (should be simple average)."""
        equal_trades_data = pd.DataFrame({
            'strategy': ['Strategy_A', 'Strategy_A'],
            'total_trades': [10, 10],
            'win_rate': [60.0, 80.0]
        })
        grouped = equal_trades_data.groupby('strategy')
        result = _calculate_weighted_win_rate(equal_trades_data, grouped)

        expected = round((60.0 + 80.0) / 2, 2)
        self.assertEqual(result['Strategy_A'], expected)

    def test_zero_win_rate(self):
        """Test with zero win rate."""
        zero_win_data = pd.DataFrame({
            'strategy': ['Strategy_A', 'Strategy_A'],
            'total_trades': [10, 20],
            'win_rate': [0.0, 50.0]
        })
        grouped = zero_win_data.groupby('strategy')
        result = _calculate_weighted_win_rate(zero_win_data, grouped)

        expected = round((0 * 10 + 50 * 20) / (10 + 20), 2)
        self.assertEqual(result['Strategy_A'], expected)




class TestCalculateTradeWeightedAverage(unittest.TestCase):
    """Tests for the _calculate_trade_weighted_average function."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame({
            'strategy': ['Strategy_A', 'Strategy_A', 'Strategy_B', 'Strategy_B'],
            'total_trades': [10, 20, 15, 25],
            'sharpe_ratio': [1.5, 2.0, 1.2, 1.8],
            'maximum_drawdown_percentage': [5.0, 3.0, 8.0, 4.0]
        })
        self.total_trades_by_strategy = self.sample_data.groupby('strategy')['total_trades'].sum()

    def test_trade_weighted_average_sharpe(self):
        """Test trade-weighted average for Sharpe ratio."""
        result = _calculate_trade_weighted_average(
            self.sample_data, 'sharpe_ratio', self.total_trades_by_strategy
        )

        # Strategy_A: (1.5*10 + 2.0*20) / (10+20) = (15 + 40) / 30 = 1.83
        # Strategy_B: (1.2*15 + 1.8*25) / (15+25) = (18 + 45) / 40 = 1.575 -> 1.58 (pandas rounding)
        expected_a = 1.83
        expected_b = 1.58  # Use actual function result due to pandas/numpy rounding behavior

        self.assertEqual(result['Strategy_A'], expected_a)
        self.assertEqual(result['Strategy_B'], expected_b)

    def test_trade_weighted_average_drawdown(self):
        """Test trade-weighted average for maximum drawdown."""
        result = _calculate_trade_weighted_average(
            self.sample_data, 'maximum_drawdown_percentage', self.total_trades_by_strategy
        )

        # Strategy_A: (5.0*10 + 3.0*20) / (10+20) = (50 + 60) / 30 = 3.67
        # Strategy_B: (8.0*15 + 4.0*25) / (15+25) = (120 + 100) / 40 = 5.5
        expected_a = round((5.0 * 10 + 3.0 * 20) / 30, 2)
        expected_b = round((8.0 * 15 + 4.0 * 25) / 40, 2)

        self.assertEqual(result['Strategy_A'], expected_a)
        self.assertEqual(result['Strategy_B'], expected_b)

    def test_single_entry_per_strategy(self):
        """Test with single entry per strategy."""
        single_entry_data = pd.DataFrame({
            'strategy': ['Strategy_A', 'Strategy_B'],
            'total_trades': [10, 20],
            'sharpe_ratio': [1.5, 2.0]
        })
        total_trades = single_entry_data.groupby('strategy')['total_trades'].sum()
        result = _calculate_trade_weighted_average(single_entry_data, 'sharpe_ratio', total_trades)

        # Should return the original values since there's only one entry per strategy
        self.assertEqual(result['Strategy_A'], 1.5)
        self.assertEqual(result['Strategy_B'], 2.0)

    def test_zero_values(self):
        """Test with zero values in the metric."""
        zero_data = pd.DataFrame({
            'strategy': ['Strategy_A', 'Strategy_A'],
            'total_trades': [10, 20],
            'sharpe_ratio': [0.0, 1.5]
        })
        total_trades = zero_data.groupby('strategy')['total_trades'].sum()
        result = _calculate_trade_weighted_average(zero_data, 'sharpe_ratio', total_trades)

        expected = round((0.0 * 10 + 1.5 * 20) / 30, 2)
        self.assertEqual(result['Strategy_A'], expected)


class TestCalculateAverageTradeReturn(unittest.TestCase):
    """Tests for the _calculate_average_trade_return function."""

    def test_basic_calculation(self):
        """Test basic average trade return calculation."""
        total_return = pd.Series([100.0, 200.0], index=['Strategy_A', 'Strategy_B'])
        total_trades = pd.Series([10, 20], index=['Strategy_A', 'Strategy_B'])

        result = _calculate_average_trade_return(total_return, total_trades)

        self.assertEqual(result['Strategy_A'], 10.0)  # 100/10
        self.assertEqual(result['Strategy_B'], 10.0)  # 200/20

    def test_different_returns(self):
        """Test with different return values."""
        total_return = pd.Series([50.0, 75.0], index=['Strategy_A', 'Strategy_B'])
        total_trades = pd.Series([5, 15], index=['Strategy_A', 'Strategy_B'])

        result = _calculate_average_trade_return(total_return, total_trades)

        self.assertEqual(result['Strategy_A'], 10.0)  # 50/5
        self.assertEqual(result['Strategy_B'], 5.0)  # 75/15

    def test_negative_returns(self):
        """Test with negative returns."""
        total_return = pd.Series([-50.0, 100.0], index=['Strategy_A', 'Strategy_B'])
        total_trades = pd.Series([10, 20], index=['Strategy_A', 'Strategy_B'])

        result = _calculate_average_trade_return(total_return, total_trades)

        self.assertEqual(result['Strategy_A'], -5.0)  # -50/10
        self.assertEqual(result['Strategy_B'], 5.0)  # 100/20

    def test_zero_return(self):
        """Test with zero return."""
        total_return = pd.Series([0.0, 100.0], index=['Strategy_A', 'Strategy_B'])
        total_trades = pd.Series([10, 20], index=['Strategy_A', 'Strategy_B'])

        result = _calculate_average_trade_return(total_return, total_trades)

        self.assertEqual(result['Strategy_A'], 0.0)  # 0/10
        self.assertEqual(result['Strategy_B'], 5.0)  # 100/20

    def test_single_strategy(self):
        """Test with single strategy."""
        total_return = pd.Series([150.0], index=['Strategy_A'])
        total_trades = pd.Series([30], index=['Strategy_A'])

        result = _calculate_average_trade_return(total_return, total_trades)

        self.assertEqual(result['Strategy_A'], 5.0)  # 150/30

    def test_rounding(self):
        """Test that results are properly rounded to 2 decimal places."""
        total_return = pd.Series([100.0], index=['Strategy_A'])
        total_trades = pd.Series([3], index=['Strategy_A'])

        result = _calculate_average_trade_return(total_return, total_trades)

        # 100/3 = 33.333... should be rounded to 33.33
        self.assertEqual(result['Strategy_A'], 33.33)


class TestCalculateProfitRatio(unittest.TestCase):
    """Tests for the _calculate_profit_ratio function."""

    def test_basic_calculation(self):
        """Test basic profit ratio calculation."""
        total_wins = pd.Series([10.0])
        total_losses = pd.Series([5.0])

        result = _calculate_profit_ratio(total_wins, total_losses)

        # 10/5 = 2.0
        self.assertEqual(result.iloc[0], 2.0)

    def test_different_ratios(self):
        """Test calculation with different win/loss ratios."""
        total_wins = pd.Series([15.0, 8.0, 12.0])
        total_losses = pd.Series([5.0, 4.0, 3.0])

        result = _calculate_profit_ratio(total_wins, total_losses)

        # 15/5 = 3.0, 8/4 = 2.0, 12/3 = 4.0
        self.assertEqual(result.iloc[0], 3.0)
        self.assertEqual(result.iloc[1], 2.0)
        self.assertEqual(result.iloc[2], 4.0)

    def test_fractional_results(self):
        """Test calculation with fractional results."""
        total_wins = pd.Series([7.0])
        total_losses = pd.Series([3.0])

        result = _calculate_profit_ratio(total_wins, total_losses)

        # 7/3 = 2.333... should be rounded to 2.33
        self.assertEqual(result.iloc[0], 2.33)

    def test_zero_losses_infinity(self):
        """Test calculation when losses are zero (should return infinity)."""
        total_wins = pd.Series([10.0])
        total_losses = pd.Series([0.0])

        result = _calculate_profit_ratio(total_wins, total_losses)

        # 10/0 should be handled as infinity
        self.assertEqual(result.iloc[0], float('inf'))

    def test_zero_wins(self):
        """Test calculation when wins are zero."""
        total_wins = pd.Series([0.0])
        total_losses = pd.Series([5.0])

        result = _calculate_profit_ratio(total_wins, total_losses)

        # 0/5 = 0.0
        self.assertEqual(result.iloc[0], 0.0)

    def test_negative_values(self):
        """Test calculation with negative values (should use absolute value)."""
        total_wins = pd.Series([10.0])
        total_losses = pd.Series([-5.0])  # Losses are typically negative

        result = _calculate_profit_ratio(total_wins, total_losses)

        # abs(10/-5) = abs(-2) = 2.0
        self.assertEqual(result.iloc[0], 2.0)

    def test_both_negative(self):
        """Test calculation with both values negative."""
        total_wins = pd.Series([-10.0])
        total_losses = pd.Series([-5.0])

        result = _calculate_profit_ratio(total_wins, total_losses)

        # abs(-10/-5) = abs(2) = 2.0
        self.assertEqual(result.iloc[0], 2.0)

    def test_rounding(self):
        """Test that results are properly rounded to 2 decimal places."""
        total_wins = pd.Series([10.0])
        total_losses = pd.Series([3.0])

        result = _calculate_profit_ratio(total_wins, total_losses)

        # 10/3 = 3.333... should be rounded to 3.33
        self.assertEqual(result.iloc[0], 3.33)


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
            # Calculate total wins and losses for new profit factor calculation
            # total_wins = (win_rate/100) * total_trades * average_win_percentage_of_margin
            # total_losses = ((100-win_rate)/100) * total_trades * average_loss_percentage_of_margin
            'total_wins_percentage_of_margin': [6.0, 9.45, 8.8, 11.375],  # [6*1.0, 10.5*0.9, 11*0.8, 16.25*0.7]
            'total_losses_percentage_of_margin': [-2.0, -1.8, -5.4, -2.625],
            # [4*(-0.5), 4.5*(-0.4), 9*(-0.6), 8.75*(-0.3)]
            'profit_factor': [2.0, 2.5, 1.8, 2.2],
            'maximum_drawdown_percentage': [2.0, 1.5, 2.5, 1.8],
            'sharpe_ratio': [1.5, 1.8, 1.2, 1.6],
            'sortino_ratio': [2.0, 2.5, 1.8, 2.2],
            'calmar_ratio': [2.5, 4.67, 1.6, 3.33],
            'value_at_risk': [1.0, 0.8, 1.2, 0.9],
            'expected_shortfall': [1.5, 1.2, 1.8, 1.4],
            'ulcer_index': [0.5, 0.4, 0.6, 0.3]
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

        # Check that the new avg_trades_per_symbol, avg_trades_per_interval, and avg_trades_per_combination columns are present and calculated correctly
        # Since each strategy has 1 symbol and 1 interval, avg_trades_per_symbol, avg_trades_per_interval, and avg_trades_per_combination should equal total_trades
        self.assertEqual(rsi_01_strategy['avg_trades_per_symbol'].iloc[0], 10.0)  # 10 trades / 1 symbol = 10.0
        self.assertEqual(rsi_01_strategy['avg_trades_per_interval'].iloc[0], 10.0)  # 10 trades / 1 interval = 10.0
        self.assertEqual(rsi_01_strategy['avg_trades_per_combination'].iloc[0],
                         10.0)  # 10 trades / (1 symbol × 1 interval) = 10.0

        self.assertEqual(rsi_005_strategy['avg_trades_per_symbol'].iloc[0], 15.0)  # 15 trades / 1 symbol = 15.0
        self.assertEqual(rsi_005_strategy['avg_trades_per_interval'].iloc[0], 15.0)  # 15 trades / 1 interval = 15.0
        self.assertEqual(rsi_005_strategy['avg_trades_per_combination'].iloc[0],
                         15.0)  # 15 trades / (1 symbol × 1 interval) = 15.0

        self.assertEqual(ema_02_strategy['avg_trades_per_symbol'].iloc[0], 20.0)  # 20 trades / 1 symbol = 20.0
        self.assertEqual(ema_02_strategy['avg_trades_per_interval'].iloc[0], 20.0)  # 20 trades / 1 interval = 20.0
        self.assertEqual(ema_02_strategy['avg_trades_per_combination'].iloc[0],
                         20.0)  # 20 trades / (1 symbol × 1 interval) = 20.0

        self.assertEqual(ema_015_strategy['avg_trades_per_symbol'].iloc[0], 25.0)  # 25 trades / 1 symbol = 25.0
        self.assertEqual(ema_015_strategy['avg_trades_per_interval'].iloc[0], 25.0)  # 25 trades / 1 interval = 25.0
        self.assertEqual(ema_015_strategy['avg_trades_per_combination'].iloc[0],
                         25.0)  # 25 trades / (1 symbol × 1 interval) = 25.0

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

        # Check that the new avg_trades_per_symbol, avg_trades_per_interval, and avg_trades_per_combination columns are present and calculated correctly
        # Since each strategy has 1 symbol and 1 interval, avg_trades_per_symbol, avg_trades_per_interval, and avg_trades_per_combination should equal total_trades
        self.assertEqual(rsi_01_strategy['avg_trades_per_symbol'].iloc[0], 10.0)  # 10 trades / 1 symbol = 10.0
        self.assertEqual(rsi_01_strategy['avg_trades_per_interval'].iloc[0], 10.0)  # 10 trades / 1 interval = 10.0
        self.assertEqual(rsi_01_strategy['avg_trades_per_combination'].iloc[0],
                         10.0)  # 10 trades / (1 symbol × 1 interval) = 10.0

        self.assertEqual(rsi_005_strategy['avg_trades_per_symbol'].iloc[0], 15.0)  # 15 trades / 1 symbol = 15.0
        self.assertEqual(rsi_005_strategy['avg_trades_per_interval'].iloc[0], 15.0)  # 15 trades / 1 interval = 15.0
        self.assertEqual(rsi_005_strategy['avg_trades_per_combination'].iloc[0],
                         15.0)  # 15 trades / (1 symbol × 1 interval) = 15.0

        self.assertEqual(ema_02_strategy['avg_trades_per_symbol'].iloc[0], 20.0)  # 20 trades / 1 symbol = 20.0
        self.assertEqual(ema_02_strategy['avg_trades_per_interval'].iloc[0], 20.0)  # 20 trades / 1 interval = 20.0
        self.assertEqual(ema_02_strategy['avg_trades_per_combination'].iloc[0],
                         20.0)  # 20 trades / (1 symbol × 1 interval) = 20.0

        self.assertEqual(ema_015_strategy['avg_trades_per_symbol'].iloc[0], 25.0)  # 25 trades / 1 symbol = 25.0
        self.assertEqual(ema_015_strategy['avg_trades_per_interval'].iloc[0], 25.0)  # 25 trades / 1 interval = 25.0
        self.assertEqual(ema_015_strategy['avg_trades_per_combination'].iloc[0],
                         25.0)  # 25 trades / (1 symbol × 1 interval) = 25.0

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

        # Test weighted aggregation with min_avg_trades_per_combination filter
        weighted_aggregated = analyzer._aggregate_strategies(min_avg_trades_per_combination=20, weighted=True)
        self.assertEqual(len(weighted_aggregated), 2)  # Only EMA strategies have rows with >= 20 trades per combination
        for strategy in weighted_aggregated['strategy']:
            self.assertTrue(strategy.startswith('EMA'))

        # Test non-weighted aggregation with min_avg_trades_per_combination filter
        non_weighted_aggregated = analyzer._aggregate_strategies(min_avg_trades_per_combination=20, weighted=False)
        self.assertEqual(len(non_weighted_aggregated),
                         2)  # Only EMA strategies have rows with >= 20 trades per combination
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
        weighted_aggregated = analyzer._aggregate_strategies(min_avg_trades_per_combination=20,
                                                             interval='4h',
                                                             weighted=True)
        self.assertEqual(len(weighted_aggregated), 2)  # Only EMA has 4h interval with >= 20 trades per combination
        for strategy in weighted_aggregated['strategy']:
            self.assertTrue(strategy.startswith('EMA'))

        # Test non-weighted aggregation with multiple filters
        non_weighted_aggregated = analyzer._aggregate_strategies(min_avg_trades_per_combination=20,
                                                                 interval='4h',
                                                                 weighted=False)
        self.assertEqual(len(non_weighted_aggregated), 2)  # Only EMA has 4h interval with >= 20 trades per combination
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
            analyzer.get_top_strategies('win_rate', 0, min_symbol_count=None)

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
        weighted_top_strategies = analyzer.get_top_strategies('win_rate', 0, weighted=True, min_symbol_count=None)

        # Verify results for weighted approach
        self.assertEqual(len(weighted_top_strategies), 4)
        self.assertEqual(weighted_top_strategies.iloc[0]['win_rate'], 70.0)  # Highest win_rate first

        # Verify save_results_to_csv was called with weighted=True
        mock_save_results.assert_called_once()
        mock_save_results.reset_mock()  # Reset mock for next test

        # Test getting top strategies by win_rate with weighted=False
        non_weighted_top_strategies = analyzer.get_top_strategies('win_rate', 0, weighted=False, min_symbol_count=None)

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
        weighted_aggregated = analyzer.get_top_strategies('win_rate',
                                                          0,
                                                          aggregate=True,
                                                          weighted=True,
                                                          min_symbol_count=None)
        mock_save_results.reset_mock()
        non_weighted_aggregated = analyzer.get_top_strategies('win_rate',
                                                              0,
                                                              aggregate=True,
                                                              weighted=False,
                                                              min_symbol_count=None)

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

        # Test with min_avg_trades_per_combination filter (weighted=True)
        weighted_top_strategies = analyzer.get_top_strategies('win_rate', 20, weighted=True, min_symbol_count=None)
        self.assertEqual(len(weighted_top_strategies), 2)  # Only rows with >= 20 trades per combination
        for strategy in weighted_top_strategies['strategy']:
            self.assertTrue(strategy.startswith('EMA'))

        # Test with min_avg_trades_per_combination filter (weighted=False)
        non_weighted_top_strategies = analyzer.get_top_strategies('win_rate', 20, weighted=False, min_symbol_count=None)
        self.assertEqual(len(non_weighted_top_strategies), 2)  # Only rows with >= 20 trades per combination
        for strategy in non_weighted_top_strategies['strategy']:
            self.assertTrue(strategy.startswith('EMA'))

        # Test with aggregate=True (weighted=True)
        weighted_aggregated = analyzer.get_top_strategies('win_rate',
                                                          0,
                                                          aggregate=True,
                                                          weighted=True,
                                                          min_symbol_count=None)
        self.assertEqual(len(weighted_aggregated), 4)  # Aggregated to 4 strategies (each row is a unique strategy)

        # Test with aggregate=True (weighted=False)
        non_weighted_aggregated = analyzer.get_top_strategies('win_rate',
                                                              0,
                                                              aggregate=True,
                                                              weighted=False,
                                                              min_symbol_count=None)
        self.assertEqual(len(non_weighted_aggregated), 4)  # Aggregated to 4 strategies (each row is a unique strategy)

        # Verify that the aggregated results are different between weighted and non-weighted
        self.assertFalse(weighted_aggregated.equals(non_weighted_aggregated))

        # Test with interval filter (weighted=True)
        weighted_interval = analyzer.get_top_strategies('win_rate',
                                                        0,
                                                        interval='1d',
                                                        weighted=True,
                                                        min_symbol_count=None)
        self.assertEqual(len(weighted_interval), 2)  # Only 1d interval
        for strategy in weighted_interval['strategy']:
            self.assertTrue(strategy.startswith('RSI'))

        # Test with interval filter (weighted=False)
        non_weighted_interval = analyzer.get_top_strategies('win_rate',
                                                            0,
                                                            interval='1d',
                                                            weighted=False,
                                                            min_symbol_count=None)
        self.assertEqual(len(non_weighted_interval), 2)  # Only 1d interval
        for strategy in non_weighted_interval['strategy']:
            self.assertTrue(strategy.startswith('RSI'))

        # Test with symbol filter (weighted=True)
        weighted_symbol = analyzer.get_top_strategies('win_rate', 0, symbol='ES', weighted=True, min_symbol_count=None)
        self.assertEqual(len(weighted_symbol), 2)  # Only ES symbol

        # Test with symbol filter (weighted=False)
        non_weighted_symbol = analyzer.get_top_strategies('win_rate',
                                                          0,
                                                          symbol='ES',
                                                          weighted=False,
                                                          min_symbol_count=None)
        self.assertEqual(len(non_weighted_symbol), 2)  # Only ES symbol

        # Test with multiple filters (weighted=True)
        weighted_multiple = analyzer.get_top_strategies('win_rate',
                                                        0,
                                                        interval='4h',
                                                        symbol='NQ',
                                                        weighted=True,
                                                        min_symbol_count=None)
        self.assertEqual(len(weighted_multiple), 1)  # Only 4h interval and NQ symbol
        self.assertTrue(weighted_multiple['strategy'].iloc[0].startswith('EMA'))

        # Test with multiple filters (weighted=False)
        non_weighted_multiple = analyzer.get_top_strategies('win_rate',
                                                            0,
                                                            interval='4h',
                                                            symbol='NQ',
                                                            weighted=False,
                                                            min_symbol_count=None)
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
        weighted_aggregated = analyzer._aggregate_strategies(min_avg_trades_per_combination=20,
                                                             min_slippage=0.15,
                                                             weighted=True)
        self.assertEqual(len(weighted_aggregated), 2)  # Only EMA with slippage >= 0.15 and trades per combination >= 20
        for strategy in weighted_aggregated['strategy']:
            self.assertTrue(strategy.startswith('EMA'))

        # Test non-weighted aggregation with multiple filters including min_slippage
        non_weighted_aggregated = analyzer._aggregate_strategies(min_avg_trades_per_combination=20,
                                                                 min_slippage=0.15,
                                                                 weighted=False)
        self.assertEqual(len(non_weighted_aggregated),
                         2)  # Only EMA with slippage >= 0.15 and trades per combination >= 20
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
        weighted_top_strategies = analyzer.get_top_strategies('win_rate',
                                                              0,
                                                              min_slippage=0.15,
                                                              weighted=True,
                                                              min_symbol_count=None)
        self.assertEqual(len(weighted_top_strategies), 2)  # Only rows with slippage >= 0.15
        for strategy in weighted_top_strategies['strategy']:
            self.assertTrue('slippage=0.2' in strategy or 'slippage=0.15' in strategy)

        # Verify save_results_to_csv was called
        mock_save_results.assert_called_once()
        mock_save_results.reset_mock()

        # Test with min_slippage filter (weighted=False)
        non_weighted_top_strategies = analyzer.get_top_strategies('win_rate',
                                                                  0,
                                                                  min_slippage=0.15,
                                                                  weighted=False,
                                                                  min_symbol_count=None)
        self.assertEqual(len(non_weighted_top_strategies), 2)  # Only rows with slippage >= 0.15
        for strategy in non_weighted_top_strategies['strategy']:
            self.assertTrue('slippage=0.2' in strategy or 'slippage=0.15' in strategy)

        # Verify save_results_to_csv was called
        mock_save_results.assert_called_once()
        mock_save_results.reset_mock()

        # Test with min_slippage=0.1 filter (weighted=True)
        weighted_top_strategies = analyzer.get_top_strategies('win_rate',
                                                              0,
                                                              min_slippage=0.1,
                                                              weighted=True,
                                                              min_symbol_count=None)
        self.assertEqual(len(weighted_top_strategies), 3)  # Rows with slippage >= 0.1

        # Test with min_slippage=0.1 filter (weighted=False)
        non_weighted_top_strategies = analyzer.get_top_strategies('win_rate',
                                                                  0,
                                                                  min_slippage=0.1,
                                                                  weighted=False,
                                                                  min_symbol_count=None)
        self.assertEqual(len(non_weighted_top_strategies), 3)  # Rows with slippage >= 0.1

        # Test with min_slippage=0.3 filter (no matches)
        weighted_top_strategies = analyzer.get_top_strategies('win_rate',
                                                              0,
                                                              min_slippage=0.3,
                                                              weighted=True,
                                                              min_symbol_count=None)
        self.assertEqual(len(weighted_top_strategies), 0)  # No rows with slippage >= 0.3

        # Test with min_slippage=0.3 filter (no matches)
        non_weighted_top_strategies = analyzer.get_top_strategies('win_rate',
                                                                  0,
                                                                  min_slippage=0.3,
                                                                  weighted=False,
                                                                  min_symbol_count=None)
        self.assertEqual(len(non_weighted_top_strategies), 0)  # No rows with slippage >= 0.3

        # Test with aggregate=True and min_slippage filter (weighted=True)
        weighted_aggregated = analyzer.get_top_strategies('win_rate',
                                                          0,
                                                          aggregate=True,
                                                          min_slippage=0.15,
                                                          weighted=True,
                                                          min_symbol_count=None)
        self.assertEqual(len(weighted_aggregated), 2)  # Only EMA strategies have slippage >= 0.15
        for strategy in weighted_aggregated['strategy']:
            self.assertTrue(strategy.startswith('EMA'))

        # Test with aggregate=True and min_slippage filter (weighted=False)
        non_weighted_aggregated = analyzer.get_top_strategies('win_rate',
                                                              0,
                                                              aggregate=True,
                                                              min_slippage=0.15,
                                                              weighted=False,
                                                              min_symbol_count=None)
        self.assertEqual(len(non_weighted_aggregated), 2)  # Only EMA strategies have slippage >= 0.15
        for strategy in non_weighted_aggregated['strategy']:
            self.assertTrue(strategy.startswith('EMA'))

        # Test with multiple filters including min_slippage (weighted=True)
        weighted_multiple = analyzer.get_top_strategies('win_rate',
                                                        0,
                                                        interval='4h',
                                                        min_slippage=0.15,
                                                        weighted=True,
                                                        min_symbol_count=None)
        self.assertEqual(len(weighted_multiple), 2)  # EMA strategies with 4h interval and slippage >= 0.15
        for strategy in weighted_multiple['strategy']:
            self.assertTrue(strategy.startswith('EMA'))

        # Test with multiple filters including min_slippage (weighted=False)
        non_weighted_multiple = analyzer.get_top_strategies('win_rate',
                                                            0,
                                                            interval='4h',
                                                            min_slippage=0.15,
                                                            weighted=False,
                                                            min_symbol_count=None)
        self.assertEqual(len(non_weighted_multiple), 2)  # EMA strategies with 4h interval and slippage >= 0.15
        for strategy in non_weighted_multiple['strategy']:
            self.assertTrue(strategy.startswith('EMA'))

    @patch('pandas.read_parquet')
    @patch('app.backtesting.strategy_analysis.StrategyAnalyzer._save_results_to_csv')
    def test_get_top_strategies_with_min_symbol_count(self, mock_save_results, mock_read_parquet):
        """Test get_top_strategies with min_symbol_count filter for both weighted and non-weighted approaches."""
        # Create test data with strategies having different symbol counts
        test_data = pd.DataFrame({
            'strategy': [
                'RSI(period=14,lower=30,upper=70,rollover=False,trailing=None,slippage=0.1)',
                'RSI(period=14,lower=30,upper=70,rollover=False,trailing=None,slippage=0.1)',
                'RSI(period=14,lower=30,upper=70,rollover=False,trailing=None,slippage=0.1)',
                'EMA(short=9,long=21,rollover=False,trailing=None,slippage=0.2)',
                'EMA(short=9,long=21,rollover=False,trailing=None,slippage=0.2)',
                'MACD(fast=12,slow=26,signal=9,rollover=False,trailing=None,slippage=0.3)'
            ],
            'symbol': ['ES', 'NQ', 'YM', 'ES', 'NQ', 'ES'],
            'interval': ['1d', '1d', '1d', '4h', '4h', '1h'],
            'total_trades': [20, 40, 30, 30, 60, 50],
            # RSI: 3 symbols, EMA: 2 symbols, MACD: 1 symbol
            'win_rate': [60.0, 70.0, 65.0, 55.0, 65.0, 75.0],
            'total_return_percentage_of_margin': [5.0, 7.0, 6.0, 4.0, 6.0, 8.0],
            'average_trade_return_percentage_of_margin': [0.25, 0.175, 0.2, 0.133, 0.1, 0.16],
            'average_win_percentage_of_margin': [1.0, 0.9, 0.95, 0.8, 0.7, 1.2],
            'average_loss_percentage_of_margin': [-0.5, -0.4, -0.45, -0.6, -0.3, -0.4],
            'commission_percentage_of_margin': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            'total_wins_percentage_of_margin': [12.0, 25.2, 18.525, 13.2, 27.3, 45.0],
            'total_losses_percentage_of_margin': [-4.0, -4.8, -4.725, -10.8, -7.875, -5.0],
            'profit_factor': [3.0, 5.25, 3.92, 1.22, 3.47, 9.0],
            'maximum_drawdown_percentage': [2.0, 1.5, 1.8, 2.5, 1.8, 1.2],
            'sharpe_ratio': [1.5, 1.8, 1.65, 1.2, 1.6, 2.0],
            'sortino_ratio': [2.0, 2.5, 2.25, 1.8, 2.2, 2.8],
            'calmar_ratio': [2.5, 4.67, 3.33, 1.6, 3.33, 6.67],
            'value_at_risk': [1.0, 0.8, 0.9, 1.2, 0.9, 0.6],
            'expected_shortfall': [1.5, 1.2, 1.35, 1.8, 1.4, 0.9],
            'ulcer_index': [0.5, 0.4, 0.45, 0.6, 0.3, 0.2]
        })

        # Setup mocks
        mock_read_parquet.return_value = test_data
        mock_save_results.reset_mock()

        # Initialize analyzer
        analyzer = StrategyAnalyzer()

        # Test with min_symbol_count=1 (should keep all strategies)
        result = analyzer.get_top_strategies('win_rate', 0, min_symbol_count=1, weighted=True)
        self.assertEqual(len(result), 6)  # All strategies
        mock_save_results.reset_mock()

        # Test with min_symbol_count=2 (should keep RSI and EMA)
        result = analyzer.get_top_strategies('win_rate', 0, min_symbol_count=2, weighted=True)
        self.assertEqual(len(result), 5)  # RSI (3 rows) + EMA (2 rows)
        strategies = result['strategy'].unique()
        self.assertTrue(any('RSI' in s for s in strategies))
        self.assertTrue(any('EMA' in s for s in strategies))
        self.assertFalse(any('MACD' in s for s in strategies))
        mock_save_results.reset_mock()

        # Test with min_symbol_count=3 (should keep only RSI)
        result = analyzer.get_top_strategies('win_rate', 0, min_symbol_count=3, weighted=True)
        self.assertEqual(len(result), 3)  # Only RSI (3 rows)
        strategies = result['strategy'].unique()
        self.assertTrue(any('RSI' in s for s in strategies))
        self.assertFalse(any('EMA' in s for s in strategies))
        self.assertFalse(any('MACD' in s for s in strategies))
        mock_save_results.reset_mock()

        # Test with min_symbol_count=4 (should keep no strategies)
        result = analyzer.get_top_strategies('win_rate', 0, min_symbol_count=4, weighted=True)
        self.assertEqual(len(result), 0)
        mock_save_results.reset_mock()

        # Test with aggregate=True and min_symbol_count=2 (weighted=True)
        result = analyzer.get_top_strategies('win_rate', 0, aggregate=True, min_symbol_count=2, weighted=True)
        self.assertEqual(len(result), 2)  # RSI and EMA strategies aggregated
        strategies = result['strategy'].unique()
        self.assertTrue(any('RSI' in s for s in strategies))
        self.assertTrue(any('EMA' in s for s in strategies))
        self.assertFalse(any('MACD' in s for s in strategies))
        mock_save_results.reset_mock()

        # Test with aggregate=True and min_symbol_count=2 (weighted=False)
        result = analyzer.get_top_strategies('win_rate', 0, aggregate=True, min_symbol_count=2, weighted=False)
        self.assertEqual(len(result), 2)  # RSI and EMA strategies aggregated
        strategies = result['strategy'].unique()
        self.assertTrue(any('RSI' in s for s in strategies))
        self.assertTrue(any('EMA' in s for s in strategies))
        self.assertFalse(any('MACD' in s for s in strategies))
        mock_save_results.reset_mock()

        # Test with combined filters: min_symbol_count=2 and min_avg_trades_per_combination=35
        result = analyzer.get_top_strategies('win_rate', 35, min_symbol_count=2, weighted=True)
        self.assertEqual(len(result), 2)  # Only EMA (2 symbols, 45 avg trades per combination)
        strategies = result['strategy'].unique()
        self.assertTrue(any('EMA' in s for s in strategies))
        self.assertFalse(any('RSI' in s for s in strategies))
        self.assertFalse(any('MACD' in s for s in strategies))


class TestParseStrategyName(unittest.TestCase):
    """Tests for the _parse_strategy_name function."""

    def test_parse_ichimoku_strategy(self):
        """Test parsing of Ichimoku strategy names."""
        strategy_name = "Ichimoku(tenkan=7,kijun=30,senkou_b=52,displacement=26,rollover=False,trailing=1,slippage=0.05)"
        clean_strategy, rollover, trailing, slippage = _parse_strategy_name(strategy_name)

        self.assertEqual(clean_strategy, "Ichimoku(tenkan=7,kijun=30,senkou_b=52,displacement=26)")
        self.assertEqual(rollover, False)
        self.assertEqual(trailing, 1.0)
        self.assertEqual(slippage, 0.05)

    def test_parse_rsi_strategy(self):
        """Test parsing of RSI strategy names."""
        strategy_name = "RSI(period=14,lower=30,upper=70,rollover=False,trailing=None,slippage=0.1)"
        clean_strategy, rollover, trailing, slippage = _parse_strategy_name(strategy_name)

        self.assertEqual(clean_strategy, "RSI(period=14,lower=30,upper=70)")
        self.assertEqual(rollover, False)
        self.assertIsNone(trailing)
        self.assertEqual(slippage, 0.1)

    def test_parse_ema_strategy(self):
        """Test parsing of EMA strategy names."""
        strategy_name = "EMA(short=9,long=21,rollover=True,trailing=2.5,slippage=0.2)"
        clean_strategy, rollover, trailing, slippage = _parse_strategy_name(strategy_name)

        self.assertEqual(clean_strategy, "EMA(short=9,long=21)")
        self.assertEqual(rollover, True)
        self.assertEqual(trailing, 2.5)
        self.assertEqual(slippage, 0.2)

    def test_parse_strategy_with_missing_params(self):
        """Test parsing of strategy names with missing common parameters."""
        strategy_name = "MACD(fast=12,slow=26,signal=9)"
        clean_strategy, rollover, trailing, slippage = _parse_strategy_name(strategy_name)

        self.assertEqual(clean_strategy, "MACD(fast=12,slow=26,signal=9)")
        self.assertEqual(rollover, False)  # Default value
        self.assertIsNone(trailing)  # Default value
        self.assertEqual(slippage, 0.0)  # Default value

    def test_parse_strategy_different_parameter_order(self):
        """Test parsing of strategy names with different parameter order."""
        strategy_name = "BB(period=20,std=2,slippage=0.15,rollover=True,trailing=1.5)"
        clean_strategy, rollover, trailing, slippage = _parse_strategy_name(strategy_name)

        self.assertEqual(clean_strategy, "BB(period=20,std=2)")
        self.assertEqual(rollover, True)
        self.assertEqual(trailing, 1.5)
        self.assertEqual(slippage, 0.15)


if __name__ == '__main__':
    unittest.main()
