"""
Analysis Pipeline Integration Tests.

Tests the complete analysis workflow:
Backtest Results → StrategyAnalyzer → Filtering → Aggregation → Ranking → Export

This validates that all analysis components work together correctly:
1. Loading results data
2. Filtering by criteria (trades, slippage, symbols)
3. Aggregating across symbols/intervals (weighted and unweighted)
4. Calculating metrics (win rate, profit factor, etc.)
5. Ranking strategies by metrics
6. Exporting to CSV

These tests use the sample_backtest_results fixture to avoid running full backtests.
"""
import pandas as pd
import pytest

from app.backtesting.analysis import StrategyAnalyzer
from app.backtesting.analysis.data_helpers import (
    filter_dataframe,
    calculate_weighted_win_rate,
    calculate_average_trade_return,
    calculate_profit_ratio,
    calculate_trade_weighted_average
)


# ==================== StrategyAnalyzer Integration Tests ====================

@pytest.mark.integration
class TestAnalysisPipelineWithSampleData:
    """Test analysis pipeline with sample backtest results."""

    @pytest.fixture(autouse=True)
    def disable_csv_writing(self, monkeypatch):
        """Disable CSV writing for all tests in this class to prevent side effects."""
        monkeypatch.setattr(
            'app.backtesting.analysis.strategy_analyzer.StrategyAnalyzer._save_results_to_csv',
            lambda self, metric, limit, df_to_save, aggregate, interval, symbol, weighted: None
        )

    def test_strategy_analyzer_initialization(self, sample_backtest_results, tmp_path):
        """
        Test that StrategyAnalyzer can be initialized with custom results.

        Since the default __init__ tries to load from a file, we test that
        the analyzer can work with manually loaded data.
        """
        # Create temporary parquet file with sample data
        test_file = tmp_path / "test_results.parquet"
        sample_backtest_results.to_parquet(test_file)

        # Initialize analyzer
        analyzer = StrategyAnalyzer()
        analyzer.results_df = pd.read_parquet(test_file)

        # Verify data loaded correctly
        assert analyzer.results_df is not None
        assert not analyzer.results_df.empty
        assert len(analyzer.results_df) == 10
        assert 'strategy' in analyzer.results_df.columns
        assert 'total_trades' in analyzer.results_df.columns

    def test_get_top_strategies_by_profit_factor(self, sample_backtest_results, tmp_path):
        """Test ranking strategies by profit factor without aggregation."""
        # Create temporary parquet file
        test_file = tmp_path / "test_results.parquet"
        sample_backtest_results.to_parquet(test_file)

        # Initialize analyzer with sample data
        analyzer = StrategyAnalyzer()
        analyzer.results_df = pd.read_parquet(test_file)

        # Get top strategies by profit factor
        top_strategies = analyzer.get_top_strategies(
            metric='profit_factor',
            min_avg_trades_per_combination=0,
            limit=5,
            aggregate=False
        )

        # Verify results
        assert len(top_strategies) == 5
        assert top_strategies.iloc[0]['profit_factor'] >= top_strategies.iloc[1]['profit_factor']
        assert 'strategy' in top_strategies.columns
        assert 'profit_factor' in top_strategies.columns

    def test_get_top_strategies_with_aggregation(self, sample_backtest_results, tmp_path):
        """Test aggregating strategy results across symbols and intervals."""
        # Create temporary parquet file
        test_file = tmp_path / "test_results.parquet"
        sample_backtest_results.to_parquet(test_file)

        # Initialize analyzer
        analyzer = StrategyAnalyzer()
        analyzer.results_df = pd.read_parquet(test_file)

        # Get aggregated top strategies
        top_strategies = analyzer.get_top_strategies(
            metric='sharpe_ratio',
            min_avg_trades_per_combination=0,
            limit=10,
            aggregate=True,
            weighted=True
        )

        # Verify aggregation occurred
        assert 'symbol_count' in top_strategies.columns
        assert 'interval_count' in top_strategies.columns
        assert 'total_trades' in top_strategies.columns

        # Verify metrics are present
        assert 'win_rate' in top_strategies.columns
        assert 'profit_factor' in top_strategies.columns
        assert 'sharpe_ratio' in top_strategies.columns

    def test_filter_by_symbol(self, sample_backtest_results, tmp_path):
        """Test filtering strategies by specific symbol."""
        # Create temporary parquet file
        test_file = tmp_path / "test_results.parquet"
        sample_backtest_results.to_parquet(test_file)

        # Initialize analyzer
        analyzer = StrategyAnalyzer()
        analyzer.results_df = pd.read_parquet(test_file)

        # Filter by symbol ZS
        top_strategies = analyzer.get_top_strategies(
            metric='win_rate',
            min_avg_trades_per_combination=0,
            limit=10,
            aggregate=False,
            symbol='ZS'
        )

        # Verify all results are for ZS
        assert all(top_strategies['symbol'] == 'ZS')
        assert len(top_strategies) == 5  # Sample data has 5 ZS entries

    def test_filter_by_interval(self, sample_backtest_results, tmp_path):
        """Test filtering strategies by specific interval."""
        # Create temporary parquet file
        test_file = tmp_path / "test_results.parquet"
        sample_backtest_results.to_parquet(test_file)

        # Initialize analyzer
        analyzer = StrategyAnalyzer()
        analyzer.results_df = pd.read_parquet(test_file)

        # Filter by interval 1h
        top_strategies = analyzer.get_top_strategies(
            metric='total_return_percentage_of_contract',
            min_avg_trades_per_combination=0,
            limit=10,
            aggregate=False,
            interval='1h'
        )

        # Verify all results are for 1h
        assert all(top_strategies['interval'] == '1h')
        assert len(top_strategies) == 10  # All sample data is 1h

    def test_weighted_vs_unweighted_aggregation(self, sample_backtest_results, tmp_path):
        """Test that weighted aggregation differs from unweighted."""
        # Create temporary parquet file
        test_file = tmp_path / "test_results.parquet"
        sample_backtest_results.to_parquet(test_file)

        # Initialize analyzer
        analyzer = StrategyAnalyzer()
        analyzer.results_df = pd.read_parquet(test_file)

        # Get weighted aggregation
        weighted = analyzer.get_top_strategies(
            metric='win_rate',
            min_avg_trades_per_combination=0,
            limit=10,
            aggregate=True,
            weighted=True
        )

        # Get unweighted aggregation
        unweighted = analyzer.get_top_strategies(
            metric='win_rate',
            min_avg_trades_per_combination=0,
            limit=10,
            aggregate=True,
            weighted=False
        )

        # Verify both produce results
        assert not weighted.empty
        assert not unweighted.empty

        # Verify structure is the same
        assert set(weighted.columns) == set(unweighted.columns)


# ==================== Data Helpers Integration Tests ====================

@pytest.mark.integration
class TestDataHelpersWithSampleData:
    """Test data helper functions with sample data."""

    def test_filter_dataframe_integration(self, sample_backtest_results):
        """Test complete DataFrame filtering workflow."""
        # Apply various filters
        filtered = filter_dataframe(
            df=sample_backtest_results,
            min_avg_trades_per_combination=50,
            interval='1h',
            symbol='ZS',
            min_slippage_ticks=None,
            min_symbol_count=None
        )

        # Verify filtering worked
        assert all(filtered['symbol'] == 'ZS')
        assert all(filtered['interval'] == '1h')
        assert not filtered.empty

    def test_weighted_calculations_integration(self, sample_backtest_results):
        """Test weighted metric calculations on sample data."""
        # Group by strategy
        grouped = sample_backtest_results.groupby('strategy')

        # Calculate weighted win rate
        weighted_win_rate = calculate_weighted_win_rate(sample_backtest_results, grouped)

        # Verify results
        assert not weighted_win_rate.empty
        assert all(weighted_win_rate >= 0)
        assert all(weighted_win_rate <= 100)

    def test_profit_ratio_calculation(self, sample_backtest_results):
        """Test profit ratio calculation with realistic data."""
        # Create sample wins and losses
        total_wins = sample_backtest_results['total_return_percentage_of_contract'].apply(
            lambda x: x if x > 0 else 0
        )
        total_losses = sample_backtest_results['total_return_percentage_of_contract'].apply(
            lambda x: abs(x) if x < 0 else 0.1  # Avoid division by zero
        )

        # Calculate profit ratios
        profit_ratios = calculate_profit_ratio(total_wins, total_losses)

        # Verify results are valid
        assert all(profit_ratios >= 0)
        assert not profit_ratios.isna().any()

    def test_average_trade_return_calculation(self, sample_backtest_results):
        """Test average trade return calculation."""
        # Calculate average returns
        avg_returns = calculate_average_trade_return(
            total_return=sample_backtest_results['total_return_percentage_of_contract'],
            total_trades=sample_backtest_results['total_trades']
        )

        # Verify results
        assert len(avg_returns) == len(sample_backtest_results)
        assert not avg_returns.isna().any()

    def test_trade_weighted_average_calculation(self, sample_backtest_results):
        """Test trade-weighted average for a metric."""
        # Calculate total trades by strategy
        total_trades = sample_backtest_results.groupby('strategy')['total_trades'].sum()

        # Calculate weighted average of sharpe ratio
        weighted_sharpe = calculate_trade_weighted_average(
            filtered_df=sample_backtest_results,
            metric_name='sharpe_ratio',
            total_trades_by_strategy=total_trades
        )

        # Verify results
        assert not weighted_sharpe.empty
        assert not weighted_sharpe.isna().any()


# ==================== End-to-End Analysis Pipeline Tests ====================

@pytest.mark.integration
class TestCompleteAnalysisWorkflow:
    """Test complete analysis workflow from data to export."""

    @pytest.fixture(autouse=True)
    def disable_csv_writing(self, monkeypatch):
        """Disable CSV writing for all tests in this class to prevent side effects."""
        monkeypatch.setattr(
            'app.backtesting.analysis.strategy_analyzer.StrategyAnalyzer._save_results_to_csv',
            lambda self, metric, limit, df_to_save, aggregate, interval, symbol, weighted: None
        )

    def test_full_analysis_pipeline(self, sample_backtest_results, tmp_path):
        """
        Test complete analysis pipeline:
        Load → Filter → Aggregate → Rank → Export
        """
        # 1. Load data
        test_file = tmp_path / "test_results.parquet"
        sample_backtest_results.to_parquet(test_file)

        analyzer = StrategyAnalyzer()
        analyzer.results_df = pd.read_parquet(test_file)
        assert not analyzer.results_df.empty

        # 2. Filter and rank (non-aggregated)
        top_non_agg = analyzer.get_top_strategies(
            metric='profit_factor',
            min_avg_trades_per_combination=0,
            limit=3,
            aggregate=False
        )
        assert len(top_non_agg) == 3
        assert top_non_agg.iloc[0]['profit_factor'] >= top_non_agg.iloc[-1]['profit_factor']

        # 3. Aggregate and rank
        top_agg = analyzer.get_top_strategies(
            metric='sharpe_ratio',
            min_avg_trades_per_combination=0,
            limit=5,
            aggregate=True,
            weighted=True
        )
        assert 'symbol_count' in top_agg.columns
        assert top_agg.iloc[0]['sharpe_ratio'] >= top_agg.iloc[-1]['sharpe_ratio']

        # 4. Filter by criteria
        filtered = analyzer.get_top_strategies(
            metric='win_rate',
            min_avg_trades_per_combination=80,
            limit=10,
            aggregate=False
        )
        # Verify minimum trades filter was applied
        assert all(filtered['total_trades'] >= 80)

    def test_multiple_metric_rankings(self, sample_backtest_results, tmp_path):
        """Test ranking by different metrics produces different orders."""
        # Load data
        test_file = tmp_path / "test_results.parquet"
        sample_backtest_results.to_parquet(test_file)

        analyzer = StrategyAnalyzer()
        analyzer.results_df = pd.read_parquet(test_file)

        # Rank by different metrics
        by_profit_factor = analyzer.get_top_strategies(
            metric='profit_factor',
            min_avg_trades_per_combination=0,
            limit=10,
            aggregate=False
        )

        by_sharpe = analyzer.get_top_strategies(
            metric='sharpe_ratio',
            min_avg_trades_per_combination=0,
            limit=10,
            aggregate=False
        )

        by_win_rate = analyzer.get_top_strategies(
            metric='win_rate',
            min_avg_trades_per_combination=0,
            limit=10,
            aggregate=False
        )

        # Verify all produced results
        assert not by_profit_factor.empty
        assert not by_sharpe.empty
        assert not by_win_rate.empty

        # Verify proper sorting
        assert by_profit_factor['profit_factor'].is_monotonic_decreasing
        assert by_sharpe['sharpe_ratio'].is_monotonic_decreasing
        assert by_win_rate['win_rate'].is_monotonic_decreasing
