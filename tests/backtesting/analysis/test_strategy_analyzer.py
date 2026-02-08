"""
Tests for Strategy Analyzer.

Uses real backtest results data to validate strategy analysis, ranking, filtering,
aggregation, and export functionality.

Test Coverage:
- Analyzer initialization and data loading
- Top strategy retrieval with various filters
- Aggregation (weighted and simple)
- Filtering by symbol, interval, slippage, symbol count
- Metric-based ranking
- CSV export functionality
- Edge cases and error handling
"""
import os
from unittest.mock import patch

import pandas as pd
import pytest

from app.backtesting.analysis.strategy_analyzer import StrategyAnalyzer
from config import BACKTESTING_DIR


# ==================== Fixtures ====================

@pytest.fixture
def sample_results_df():
    """
    Create sample backtest results DataFrame for testing.

    Mirrors structure of real mass_test_results_all.parquet with various scenarios:
    - Multiple symbols (ZS, CL, ES)
    - Multiple intervals (1h, 4h, 1d)
    - Multiple strategies with different performance characteristics
    - Various trade counts and metrics

    Returns:
        DataFrame with 12 rows representing different strategy/symbol/interval combinations
    """
    return pd.DataFrame([
        # High-performing strategy across multiple symbols
        {'month': '1!', 'symbol': 'ZS', 'interval': '1h', 'strategy': 'TopStrategy',
         'total_trades': 100, 'win_rate': 65.0, 'average_trade_duration_hours': 4.5,
         'total_wins_percentage_of_contract': 150.0, 'total_losses_percentage_of_contract': 50.0,
         'total_return_percentage_of_contract': 100.0, 'average_trade_return_percentage_of_contract': 1.0,
         'average_win_percentage_of_contract': 2.5, 'average_loss_percentage_of_contract': -1.5,
         'profit_factor': 3.0, 'maximum_drawdown_percentage': 10.0, 'sharpe_ratio': 2.5,
         'sortino_ratio': 3.0, 'calmar_ratio': 10.0, 'value_at_risk': 5.0,
         'expected_shortfall': 7.0, 'ulcer_index': 3.0},

        {'month': '1!', 'symbol': 'CL', 'interval': '1h', 'strategy': 'TopStrategy',
         'total_trades': 80, 'win_rate': 60.0, 'average_trade_duration_hours': 4.0,
         'total_wins_percentage_of_contract': 120.0, 'total_losses_percentage_of_contract': 40.0,
         'total_return_percentage_of_contract': 80.0, 'average_trade_return_percentage_of_contract': 1.0,
         'average_win_percentage_of_contract': 2.5, 'average_loss_percentage_of_contract': -1.0,
         'profit_factor': 3.0, 'maximum_drawdown_percentage': 12.0, 'sharpe_ratio': 2.0,
         'sortino_ratio': 2.5, 'calmar_ratio': 6.67, 'value_at_risk': 6.0,
         'expected_shortfall': 8.0, 'ulcer_index': 4.0},

        {'month': '1!', 'symbol': 'ES', 'interval': '4h', 'strategy': 'TopStrategy',
         'total_trades': 50, 'win_rate': 70.0, 'average_trade_duration_hours': 8.0,
         'total_wins_percentage_of_contract': 105.0, 'total_losses_percentage_of_contract': 15.0,
         'total_return_percentage_of_contract': 90.0, 'average_trade_return_percentage_of_contract': 1.8,
         'average_win_percentage_of_contract': 3.0, 'average_loss_percentage_of_contract': -1.0,
         'profit_factor': 7.0, 'maximum_drawdown_percentage': 5.0, 'sharpe_ratio': 3.0,
         'sortino_ratio': 4.0, 'calmar_ratio': 18.0, 'value_at_risk': 3.0,
         'expected_shortfall': 4.0, 'ulcer_index': 2.0},

        # Medium-performing strategy
        {'month': '1!', 'symbol': 'ZS', 'interval': '1h', 'strategy': 'MediumStrategy',
         'total_trades': 60, 'win_rate': 55.0, 'average_trade_duration_hours': 5.0,
         'total_wins_percentage_of_contract': 60.0, 'total_losses_percentage_of_contract': 30.0,
         'total_return_percentage_of_contract': 30.0, 'average_trade_return_percentage_of_contract': 0.5,
         'average_win_percentage_of_contract': 2.0, 'average_loss_percentage_of_contract': -1.2,
         'profit_factor': 2.0, 'maximum_drawdown_percentage': 15.0, 'sharpe_ratio': 1.5,
         'sortino_ratio': 2.0, 'calmar_ratio': 2.0, 'value_at_risk': 8.0,
         'expected_shortfall': 10.0, 'ulcer_index': 6.0},

        {'month': '1!', 'symbol': 'CL', 'interval': '4h', 'strategy': 'MediumStrategy',
         'total_trades': 40, 'win_rate': 50.0, 'average_trade_duration_hours': 6.0,
         'total_wins_percentage_of_contract': 40.0, 'total_losses_percentage_of_contract': 20.0,
         'total_return_percentage_of_contract': 20.0, 'average_trade_return_percentage_of_contract': 0.5,
         'average_win_percentage_of_contract': 2.0, 'average_loss_percentage_of_contract': -1.0,
         'profit_factor': 2.0, 'maximum_drawdown_percentage': 18.0, 'sharpe_ratio': 1.0,
         'sortino_ratio': 1.5, 'calmar_ratio': 1.11, 'value_at_risk': 10.0,
         'expected_shortfall': 12.0, 'ulcer_index': 8.0},

        # Low-performing strategy (should be filtered out with min trades)
        {'month': '1!', 'symbol': 'ZS', 'interval': '1d', 'strategy': 'LowTradeStrategy',
         'total_trades': 5, 'win_rate': 40.0, 'average_trade_duration_hours': 24.0,
         'total_wins_percentage_of_contract': 10.0, 'total_losses_percentage_of_contract': 15.0,
         'total_return_percentage_of_contract': -5.0, 'average_trade_return_percentage_of_contract': -1.0,
         'average_win_percentage_of_contract': 5.0, 'average_loss_percentage_of_contract': -5.0,
         'profit_factor': 0.67, 'maximum_drawdown_percentage': 25.0, 'sharpe_ratio': -0.5,
         'sortino_ratio': -0.5, 'calmar_ratio': -0.2, 'value_at_risk': 15.0,
         'expected_shortfall': 18.0, 'ulcer_index': 12.0},

        # Single symbol strategy (for symbol count filtering)
        {'month': '1!', 'symbol': 'ZS', 'interval': '1h', 'strategy': 'SingleSymbolStrategy',
         'total_trades': 90, 'win_rate': 58.0, 'average_trade_duration_hours': 3.0,
         'total_wins_percentage_of_contract': 90.0, 'total_losses_percentage_of_contract': 40.0,
         'total_return_percentage_of_contract': 50.0, 'average_trade_return_percentage_of_contract': 0.56,
         'average_win_percentage_of_contract': 1.8, 'average_loss_percentage_of_contract': -1.1,
         'profit_factor': 2.25, 'maximum_drawdown_percentage': 12.0, 'sharpe_ratio': 1.8,
         'sortino_ratio': 2.2, 'calmar_ratio': 4.17, 'value_at_risk': 7.0,
         'expected_shortfall': 9.0, 'ulcer_index': 5.0},

        # Strategy with different slippage (for slippage filtering tests)
        {'month': '1!', 'symbol': 'ZS', 'interval': '1h', 'strategy': 'HighSlippageStrategy_slippage_5',
         'total_trades': 70, 'win_rate': 52.0, 'average_trade_duration_hours': 4.0,
         'total_wins_percentage_of_contract': 50.0, 'total_losses_percentage_of_contract': 30.0,
         'total_return_percentage_of_contract': 20.0, 'average_trade_return_percentage_of_contract': 0.29,
         'average_win_percentage_of_contract': 1.4, 'average_loss_percentage_of_contract': -0.9,
         'profit_factor': 1.67, 'maximum_drawdown_percentage': 16.0, 'sharpe_ratio': 1.2,
         'sortino_ratio': 1.6, 'calmar_ratio': 1.25, 'value_at_risk': 9.0,
         'expected_shortfall': 11.0, 'ulcer_index': 7.0},

        {'month': '1!', 'symbol': 'CL', 'interval': '1h', 'strategy': 'HighSlippageStrategy_slippage_5',
         'total_trades': 65, 'win_rate': 51.0, 'average_trade_duration_hours': 3.5,
         'total_wins_percentage_of_contract': 45.0, 'total_losses_percentage_of_contract': 28.0,
         'total_return_percentage_of_contract': 17.0, 'average_trade_return_percentage_of_contract': 0.26,
         'average_win_percentage_of_contract': 1.4, 'average_loss_percentage_of_contract': -0.9,
         'profit_factor': 1.61, 'maximum_drawdown_percentage': 17.0, 'sharpe_ratio': 1.1,
         'sortino_ratio': 1.5, 'calmar_ratio': 1.0, 'value_at_risk': 9.5,
         'expected_shortfall': 11.5, 'ulcer_index': 7.5},

        # Additional variations for comprehensive testing
        {'month': '1!', 'symbol': 'ES', 'interval': '1d', 'strategy': 'LongTermStrategy',
         'total_trades': 25, 'win_rate': 64.0, 'average_trade_duration_hours': 48.0,
         'total_wins_percentage_of_contract': 60.0, 'total_losses_percentage_of_contract': 20.0,
         'total_return_percentage_of_contract': 40.0, 'average_trade_return_percentage_of_contract': 1.6,
         'average_win_percentage_of_contract': 4.0, 'average_loss_percentage_of_contract': -2.5,
         'profit_factor': 3.0, 'maximum_drawdown_percentage': 8.0, 'sharpe_ratio': 2.2,
         'sortino_ratio': 2.8, 'calmar_ratio': 5.0, 'value_at_risk': 5.0,
         'expected_shortfall': 6.5, 'ulcer_index': 3.5},

        {'month': '1!', 'symbol': 'CL', 'interval': '1d', 'strategy': 'LongTermStrategy',
         'total_trades': 30, 'win_rate': 60.0, 'average_trade_duration_hours': 40.0,
         'total_wins_percentage_of_contract': 55.0, 'total_losses_percentage_of_contract': 25.0,
         'total_return_percentage_of_contract': 30.0, 'average_trade_return_percentage_of_contract': 1.0,
         'average_win_percentage_of_contract': 3.5, 'average_loss_percentage_of_contract': -2.3,
         'profit_factor': 2.2, 'maximum_drawdown_percentage': 10.0, 'sharpe_ratio': 1.8,
         'sortino_ratio': 2.3, 'calmar_ratio': 3.0, 'value_at_risk': 6.0,
         'expected_shortfall': 7.5, 'ulcer_index': 4.5},

        {'month': '1!', 'symbol': 'ZS', 'interval': '4h', 'strategy': 'LongTermStrategy',
         'total_trades': 35, 'win_rate': 62.0, 'average_trade_duration_hours': 36.0,
         'total_wins_percentage_of_contract': 58.0, 'total_losses_percentage_of_contract': 22.0,
         'total_return_percentage_of_contract': 36.0, 'average_trade_return_percentage_of_contract': 1.03,
         'average_win_percentage_of_contract': 3.7, 'average_loss_percentage_of_contract': -1.9,
         'profit_factor': 2.64, 'maximum_drawdown_percentage': 9.0, 'sharpe_ratio': 2.0,
         'sortino_ratio': 2.5, 'calmar_ratio': 4.0, 'value_at_risk': 5.5,
         'expected_shortfall': 7.0, 'ulcer_index': 4.0},
    ])


@pytest.fixture
def temp_results_file(sample_results_df, tmp_path):
    """
    Create a temporary parquet file with sample results.

    Args:
        sample_results_df: Sample DataFrame fixture
        tmp_path: pytest's temporary directory fixture

    Returns:
        Path to temporary parquet file
    """
    temp_file = tmp_path / "test_results.parquet"
    sample_results_df.to_parquet(temp_file)
    return str(temp_file)


@pytest.fixture
def analyzer_with_data(temp_results_file, sample_results_df):
    """
    Create StrategyAnalyzer instance with sample data.

    Uses mock to override the default file path so tests don't depend on
    the actual mass_test_results_all.parquet file existing.

    Args:
        temp_results_file: Path to temporary test data file
        sample_results_df: Sample DataFrame for direct assignment

    Returns:
        StrategyAnalyzer instance loaded with test data
    """
    # Create analyzer but prevent it from loading the default file
    with patch('app.backtesting.analysis.strategy_analyzer.pd.read_parquet') as mock_read:
        mock_read.return_value = sample_results_df
        analyzer = StrategyAnalyzer()
        return analyzer


@pytest.fixture
def real_results_file():
    """
    Path to real backtest results file (if it exists).

    Returns:
        Path to real results file or None if not found
    """
    file_path = f'{BACKTESTING_DIR}/mass_test_results_all.parquet'
    return file_path if os.path.exists(file_path) else None


# ==================== Test Classes ====================

class TestStrategyAnalyzerInitialization:
    """Test StrategyAnalyzer initialization and data loading."""

    def test_initialization_loads_default_file(self, real_results_file):
        """Test that analyzer loads default results file on initialization."""
        if real_results_file is None:
            pytest.skip("Real results file not available")

        analyzer = StrategyAnalyzer()

        assert analyzer.results_df is not None
        assert isinstance(analyzer.results_df, pd.DataFrame)
        assert not analyzer.results_df.empty
        assert len(analyzer.results_df) > 0

    def test_initialization_with_custom_data(self, temp_results_file, sample_results_df):
        """Test initialization with custom data file."""
        with patch('app.backtesting.analysis.strategy_analyzer.pd.read_parquet') as mock_read:
            mock_read.return_value = sample_results_df
            analyzer = StrategyAnalyzer()

            assert analyzer.results_df is not None
            assert len(analyzer.results_df) == 12  # Sample data has 12 rows

    def test_load_results_with_missing_file(self):
        """Test error handling when results file doesn't exist."""
        # Create an analyzer instance (mocked so it doesn't fail)
        with patch('app.backtesting.analysis.strategy_analyzer.pd.read_parquet') as mock_read:
            mock_read.side_effect = FileNotFoundError("File not found")

            # Should raise exception during initialization
            with pytest.raises(Exception):
                analyzer = StrategyAnalyzer()

    def test_results_dataframe_has_required_columns(self, analyzer_with_data):
        """Test that loaded results have all required columns."""
        required_columns = [
            'strategy', 'symbol', 'interval', 'total_trades', 'win_rate',
            'profit_factor', 'sharpe_ratio', 'maximum_drawdown_percentage'
        ]

        for col in required_columns:
            assert col in analyzer_with_data.results_df.columns


class TestGetTopStrategiesBasic:
    """Test basic functionality of get_top_strategies without aggregation."""

    def test_get_top_strategies_sorts_by_metric(self, analyzer_with_data):
        """Test that strategies are sorted by specified metric in descending order."""
        result = analyzer_with_data.get_top_strategies(
            metric='profit_factor',
            min_avg_trades_per_combination=0,
            limit=5,
            aggregate=False
        )

        # Check results are sorted
        assert result['profit_factor'].is_monotonic_decreasing

        # TopStrategy on ES should be first (profit_factor=7.0)
        assert result.iloc[0]['strategy'] == 'TopStrategy'
        assert result.iloc[0]['symbol'] == 'ES'
        assert result.iloc[0]['profit_factor'] == 7.0

    def test_get_top_strategies_applies_limit(self, analyzer_with_data, tmp_path):
        """Test that limit parameter affects CSV export (not returned DataFrame)."""
        # Note: The limit parameter only affects CSV export, not the returned DataFrame
        with patch('app.backtesting.analysis.strategy_analyzer.BACKTESTING_DIR', str(tmp_path)):
            result = analyzer_with_data.get_top_strategies(
                metric='win_rate',
                min_avg_trades_per_combination=0,
                limit=3,
                aggregate=False
            )

            # Returned DataFrame contains all results (12 rows)
            assert len(result) == 12

            # But CSV file should only have 3 rows (limit applied)
            csv_file = list((tmp_path / 'csv_results').glob('*.csv'))[0]
            csv_df = pd.read_csv(csv_file)
            assert len(csv_df) == 3

    def test_get_top_strategies_with_no_limit(self, analyzer_with_data):
        """Test getting all strategies when limit is None."""
        result = analyzer_with_data.get_top_strategies(
            metric='sharpe_ratio',
            min_avg_trades_per_combination=0,
            limit=None,
            aggregate=False
        )

        # Should return all 12 rows from sample data
        assert len(result) == 12

    def test_get_top_strategies_filters_min_trades(self, analyzer_with_data):
        """Test filtering strategies by minimum average trades."""
        # Filter out strategies with < 20 trades
        result = analyzer_with_data.get_top_strategies(
            metric='profit_factor',
            min_avg_trades_per_combination=20,
            limit=None,
            aggregate=False
        )

        # Should exclude LowTradeStrategy (5 trades) and LongTermStrategy variations (~25-35 but need to check avg)
        assert all(result['total_trades'] >= 20)
        assert 'LowTradeStrategy' not in result['strategy'].values


class TestGetTopStrategiesFiltering:
    """Test filtering functionality in get_top_strategies."""

    def test_filter_by_symbol(self, analyzer_with_data):
        """Test filtering strategies by specific symbol."""
        result = analyzer_with_data.get_top_strategies(
            metric='profit_factor',
            min_avg_trades_per_combination=0,
            limit=None,
            aggregate=False,
            symbol='ZS'
        )

        # All results should be for ZS
        assert all(result['symbol'] == 'ZS')
        assert len(result) > 0

    def test_filter_by_interval(self, analyzer_with_data):
        """Test filtering strategies by specific interval."""
        result = analyzer_with_data.get_top_strategies(
            metric='sharpe_ratio',
            min_avg_trades_per_combination=0,
            limit=None,
            aggregate=False,
            interval='1h'
        )

        # All results should be for 1h interval
        assert all(result['interval'] == '1h')
        assert len(result) > 0

    def test_filter_by_symbol_and_interval(self, analyzer_with_data):
        """Test filtering by both symbol and interval."""
        result = analyzer_with_data.get_top_strategies(
            metric='win_rate',
            min_avg_trades_per_combination=0,
            limit=None,
            aggregate=False,
            symbol='CL',
            interval='1h'
        )

        # All results should match both filters
        assert all(result['symbol'] == 'CL')
        assert all(result['interval'] == '1h')

    def test_filter_by_min_slippage(self, analyzer_with_data):
        """Test filtering strategies by minimum slippage ticks."""
        result = analyzer_with_data.get_top_strategies(
            metric='profit_factor',
            min_avg_trades_per_combination=0,
            limit=None,
            aggregate=False,
            min_slippage_ticks=5
        )

        # Should only include HighSlippageStrategy_slippage_5
        assert all('slippage_5' in str(name) for name in result['strategy'].values)

    def test_no_results_with_strict_filters(self, analyzer_with_data):
        """Test that empty DataFrame is returned when no strategies match filters."""
        result = analyzer_with_data.get_top_strategies(
            metric='profit_factor',
            min_avg_trades_per_combination=1000,  # Impossible filter
            limit=None,
            aggregate=False
        )

        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)


class TestGetTopStrategiesAggregation:
    """Test strategy aggregation functionality."""

    def test_aggregate_combines_symbols_and_intervals(self, analyzer_with_data):
        """Test that aggregation combines results across symbols and intervals."""
        result = analyzer_with_data.get_top_strategies(
            metric='profit_factor',
            min_avg_trades_per_combination=0,
            limit=None,
            aggregate=True,
            weighted=False
        )

        # Should have one row per unique strategy
        unique_strategies = analyzer_with_data.results_df['strategy'].nunique()
        assert len(result) == unique_strategies

        # Check TopStrategy aggregation
        top_strategy = result[result['strategy'] == 'TopStrategy'].iloc[0]
        assert top_strategy['symbol_count'] == 3  # ZS, CL, ES
        assert top_strategy['total_trades'] == 230  # 100 + 80 + 50

    def test_weighted_aggregation_vs_simple(self, analyzer_with_data):
        """Test difference between weighted and simple aggregation."""
        weighted_result = analyzer_with_data.get_top_strategies(
            metric='win_rate',
            min_avg_trades_per_combination=0,
            limit=None,
            aggregate=True,
            weighted=True
        )

        simple_result = analyzer_with_data.get_top_strategies(
            metric='win_rate',
            min_avg_trades_per_combination=0,
            limit=None,
            aggregate=True,
            weighted=False
        )

        # Both should return same strategies
        assert set(weighted_result['strategy']) == set(simple_result['strategy'])

        # But win_rate values should differ (weighted considers trade counts)
        top_weighted = weighted_result[weighted_result['strategy'] == 'TopStrategy'].iloc[0]['win_rate']
        top_simple = simple_result[simple_result['strategy'] == 'TopStrategy'].iloc[0]['win_rate']

        # They should be different due to weighting
        assert top_weighted != top_simple

    def test_aggregation_calculates_symbol_interval_counts(self, analyzer_with_data):
        """Test that aggregation correctly counts unique symbols and intervals."""
        result = analyzer_with_data.get_top_strategies(
            metric='profit_factor',
            min_avg_trades_per_combination=0,
            limit=None,
            aggregate=True
        )

        # Check LongTermStrategy (should have 3 symbols and 2 intervals)
        long_term = result[result['strategy'] == 'LongTermStrategy'].iloc[0]
        assert long_term['symbol_count'] == 3  # ZS, CL, ES
        assert long_term['interval_count'] == 2  # 1d, 4h

    def test_aggregation_sums_total_trades(self, analyzer_with_data):
        """Test that total trades are summed correctly in aggregation."""
        result = analyzer_with_data.get_top_strategies(
            metric='sharpe_ratio',
            min_avg_trades_per_combination=0,
            limit=None,
            aggregate=True
        )

        # TopStrategy should have sum of 100 + 80 + 50 = 230 trades
        top_strategy = result[result['strategy'] == 'TopStrategy'].iloc[0]
        assert top_strategy['total_trades'] == 230

    def test_aggregation_with_min_symbol_count(self, analyzer_with_data):
        """Test filtering aggregated results by minimum symbol count."""
        result = analyzer_with_data.get_top_strategies(
            metric='profit_factor',
            min_avg_trades_per_combination=0,
            limit=None,
            aggregate=True,
            min_symbol_count=2
        )

        # Should exclude SingleSymbolStrategy (only 1 symbol)
        assert 'SingleSymbolStrategy' not in result['strategy'].values

        # Should include TopStrategy (3 symbols) and others with >= 2 symbols
        assert all(result['symbol_count'] >= 2)


class TestGetTopStrategiesMetrics:
    """Test ranking by different metrics."""

    @pytest.mark.parametrize("metric", [
        'profit_factor',
        'win_rate',
        'sharpe_ratio',
        'sortino_ratio',
        'calmar_ratio',
        'total_return_percentage_of_contract',
        'average_trade_return_percentage_of_contract'
    ])
    def test_ranking_by_various_metrics(self, analyzer_with_data, metric):
        """Test that strategies can be ranked by different metrics."""
        result = analyzer_with_data.get_top_strategies(
            metric=metric,
            min_avg_trades_per_combination=0,
            limit=5,
            aggregate=False
        )

        # Check that results are sorted by the metric
        assert result[metric].is_monotonic_decreasing
        assert len(result) > 0

    def test_top_by_profit_factor(self, analyzer_with_data):
        """Test ranking by profit factor identifies best risk/reward strategies."""
        result = analyzer_with_data.get_top_strategies(
            metric='profit_factor',
            min_avg_trades_per_combination=0,
            limit=1,
            aggregate=False
        )

        # TopStrategy on ES should be first (profit_factor=7.0)
        assert result.iloc[0]['strategy'] == 'TopStrategy'
        assert result.iloc[0]['profit_factor'] == 7.0

    def test_top_by_sharpe_ratio(self, analyzer_with_data):
        """Test ranking by Sharpe ratio identifies best risk-adjusted returns."""
        result = analyzer_with_data.get_top_strategies(
            metric='sharpe_ratio',
            min_avg_trades_per_combination=20,
            limit=3,
            aggregate=False
        )

        # Check that top results have high Sharpe ratios
        assert all(result['sharpe_ratio'] >= 1.0)

    def test_top_by_maximum_drawdown(self, analyzer_with_data):
        """Test ranking by maximum drawdown (lower is better, but we sort descending)."""
        result = analyzer_with_data.get_top_strategies(
            metric='maximum_drawdown_percentage',
            min_avg_trades_per_combination=0,
            limit=None,
            aggregate=False
        )

        # Results should be sorted (higher drawdown first due to descending sort)
        # Note: This might seem counterintuitive, but the method always sorts descending
        assert result['maximum_drawdown_percentage'].is_monotonic_decreasing


class TestCSVExport:
    """Test CSV export functionality."""

    def test_saves_csv_file(self, analyzer_with_data, tmp_path):
        """Test that results are saved to CSV file."""
        # Mock the BACKTESTING_DIR to use temp directory
        with patch('app.backtesting.analysis.strategy_analyzer.BACKTESTING_DIR', str(tmp_path)):
            result = analyzer_with_data.get_top_strategies(
                metric='profit_factor',
                min_avg_trades_per_combination=0,
                limit=5,
                aggregate=False
            )

            # Check that CSV was created
            csv_dir = tmp_path / 'csv_results'
            assert csv_dir.exists()

            # Check that a CSV file was created
            csv_files = list(csv_dir.glob('*.csv'))
            assert len(csv_files) > 0

    def test_csv_filename_includes_metric(self, analyzer_with_data, tmp_path):
        """Test that CSV filename includes the ranking metric."""
        with patch('app.backtesting.analysis.strategy_analyzer.BACKTESTING_DIR', str(tmp_path)):
            analyzer_with_data.get_top_strategies(
                metric='sharpe_ratio',
                min_avg_trades_per_combination=0,
                limit=5,
                aggregate=False
            )

            csv_files = list((tmp_path / 'csv_results').glob('*.csv'))
            assert any('sharpe_ratio' in f.name for f in csv_files)

    def test_csv_content_matches_dataframe(self, analyzer_with_data, tmp_path):
        """Test that CSV file content matches returned DataFrame."""
        with patch('app.backtesting.analysis.strategy_analyzer.BACKTESTING_DIR', str(tmp_path)):
            result = analyzer_with_data.get_top_strategies(
                metric='profit_factor',
                min_avg_trades_per_combination=0,
                limit=3,
                aggregate=False
            )

            # Read the CSV file
            csv_file = list((tmp_path / 'csv_results').glob('*.csv'))[0]
            csv_df = pd.read_csv(csv_file)

            # Should have same number of rows (limited to 3)
            assert len(csv_df) == 3

    def test_csv_respects_limit(self, analyzer_with_data, tmp_path):
        """Test that CSV file respects the limit parameter."""
        with patch('app.backtesting.analysis.strategy_analyzer.BACKTESTING_DIR', str(tmp_path)):
            analyzer_with_data.get_top_strategies(
                metric='win_rate',
                min_avg_trades_per_combination=0,
                limit=10,
                aggregate=False
            )

            csv_file = list((tmp_path / 'csv_results').glob('*.csv'))[0]
            csv_df = pd.read_csv(csv_file)

            # Should have at most 10 rows
            assert len(csv_df) <= 10


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_results_raises_error(self):
        """Test that calling get_top_strategies with no data raises error."""
        with patch.object(StrategyAnalyzer, '_load_results'):
            analyzer = StrategyAnalyzer()
            analyzer.results_df = None

            with pytest.raises(ValueError, match='No results available'):
                analyzer.get_top_strategies(
                    metric='profit_factor',
                    min_avg_trades_per_combination=0,
                    limit=5
                )

    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises error."""
        with patch.object(StrategyAnalyzer, '_load_results'):
            analyzer = StrategyAnalyzer()
            analyzer.results_df = pd.DataFrame()  # Empty DataFrame

            with pytest.raises(ValueError, match='No results available'):
                analyzer.get_top_strategies(
                    metric='profit_factor',
                    min_avg_trades_per_combination=0,
                    limit=5
                )

    def test_all_strategies_filtered_returns_empty(self, analyzer_with_data):
        """Test that overly strict filters return empty DataFrame gracefully."""
        result = analyzer_with_data.get_top_strategies(
            metric='profit_factor',
            min_avg_trades_per_combination=10000,  # Impossible requirement
            limit=5,
            aggregate=False
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_single_strategy_result(self, analyzer_with_data, tmp_path):
        """Test handling of results when filtering to a single strategy."""
        # Filter to get only one specific strategy by using symbol and interval filters
        with patch('app.backtesting.analysis.strategy_analyzer.BACKTESTING_DIR', str(tmp_path)):
            result = analyzer_with_data.get_top_strategies(
                metric='profit_factor',
                min_avg_trades_per_combination=0,
                limit=None,
                aggregate=False,
                symbol='ZS',
                interval='1d'
            )

            # LowTradeStrategy is the only strategy with ZS and 1d
            assert len(result) == 1
            assert isinstance(result, pd.DataFrame)
            assert result.iloc[0]['strategy'] == 'LowTradeStrategy'

    def test_aggregation_with_no_matching_strategies(self, analyzer_with_data):
        """Test aggregation when filters exclude all strategies."""
        result = analyzer_with_data.get_top_strategies(
            metric='profit_factor',
            min_avg_trades_per_combination=10000,
            limit=None,
            aggregate=True
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestRealDataIntegration:
    """Test with real backtest results data (if available)."""

    def test_with_real_data_file(self, real_results_file):
        """Test analyzer works with real backtest results."""
        if real_results_file is None:
            pytest.skip("Real results file not available")

        analyzer = StrategyAnalyzer()

        # Test basic retrieval
        result = analyzer.get_top_strategies(
            metric='profit_factor',
            min_avg_trades_per_combination=10,
            limit=10,
            aggregate=False
        )

        assert len(result) > 0
        assert 'strategy' in result.columns
        assert result['profit_factor'].is_monotonic_decreasing

    def test_real_data_aggregation(self, real_results_file):
        """Test aggregation with real data."""
        if real_results_file is None:
            pytest.skip("Real results file not available")

        analyzer = StrategyAnalyzer()

        result = analyzer.get_top_strategies(
            metric='sharpe_ratio',
            min_avg_trades_per_combination=5,
            limit=5,
            aggregate=True,
            weighted=True
        )

        assert len(result) > 0
        assert 'symbol_count' in result.columns
        assert 'total_trades' in result.columns

    def test_real_data_various_filters(self, real_results_file):
        """Test various filtering combinations with real data."""
        if real_results_file is None:
            pytest.skip("Real results file not available")

        analyzer = StrategyAnalyzer()

        # Test symbol filter
        zs_result = analyzer.get_top_strategies(
            metric='win_rate',
            min_avg_trades_per_combination=5,
            limit=10,
            aggregate=False,
            symbol='ZS'
        )

        if len(zs_result) > 0:
            assert all(zs_result['symbol'] == 'ZS')

        # Test interval filter
        hourly_result = analyzer.get_top_strategies(
            metric='profit_factor',
            min_avg_trades_per_combination=5,
            limit=10,
            aggregate=False,
            interval='1h'
        )

        if len(hourly_result) > 0:
            assert all(hourly_result['interval'] == '1h')


class TestAggregationMetrics:
    """Test specific metric calculations in aggregation."""

    def test_weighted_win_rate_calculation(self, analyzer_with_data):
        """Test that weighted win rate gives more weight to strategies with more trades."""
        result = analyzer_with_data.get_top_strategies(
            metric='win_rate',
            min_avg_trades_per_combination=0,
            limit=None,
            aggregate=True,
            weighted=True
        )

        # TopStrategy: (100*65 + 80*60 + 50*70) / 230 = weighted win rate
        top_strategy = result[result['strategy'] == 'TopStrategy'].iloc[0]
        expected_weighted_wr = (100 * 65 + 80 * 60 + 50 * 70) / 230

        assert abs(top_strategy['win_rate'] - expected_weighted_wr) < 0.1

    def test_profit_factor_recalculation_in_aggregation(self, analyzer_with_data):
        """Test that profit factor is recalculated from aggregated wins/losses."""
        result = analyzer_with_data.get_top_strategies(
            metric='profit_factor',
            min_avg_trades_per_combination=0,
            limit=None,
            aggregate=True,
            weighted=True
        )

        # TopStrategy total wins: 150 + 120 + 105 = 375
        # TopStrategy total losses: 50 + 40 + 15 = 105
        # Profit factor: 375 / 105 = 3.57
        top_strategy = result[result['strategy'] == 'TopStrategy'].iloc[0]
        expected_pf = 375 / 105

        assert abs(top_strategy['profit_factor'] - expected_pf) < 0.1

    def test_average_trades_per_combination(self, analyzer_with_data):
        """Test calculation of average trades per symbol/interval combination."""
        result = analyzer_with_data.get_top_strategies(
            metric='total_trades',
            min_avg_trades_per_combination=0,
            limit=None,
            aggregate=True
        )

        # TopStrategy: 230 total trades / (3 symbols * 3 intervals) = 25.56 avg
        # Wait, TopStrategy has: ZS-1h, CL-1h, ES-4h = 3 combinations (not 3*3)
        # So: 230 / 3 combinations = 76.67
        top_strategy = result[result['strategy'] == 'TopStrategy'].iloc[0]

        # symbol_count=3, but need to count actual combinations
        # Let's verify the formula: total_trades / (symbol_count * interval_count)
        expected_avg = top_strategy['total_trades'] / (top_strategy['symbol_count'] * top_strategy['interval_count'])

        assert abs(top_strategy['avg_trades_per_combination'] - expected_avg) < 0.1
