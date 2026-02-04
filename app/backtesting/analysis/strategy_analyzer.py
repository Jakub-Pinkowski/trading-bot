"""
Strategy Analyzer

Main class for analyzing and exporting trading strategy results.
"""

import os

import pandas as pd

from app.backtesting.analysis.constants import DEFAULT_LIMIT, DECIMAL_PLACES
from app.backtesting.analysis.data_helpers import (
    filter_dataframe,
    calculate_weighted_win_rate,
    calculate_average_trade_return,
    calculate_profit_ratio,
    calculate_trade_weighted_average
)
from app.backtesting.analysis.formatters import (
    format_dataframe_for_export,
    build_filename
)
from app.utils.logger import get_logger
from config import BACKTESTING_DIR

logger = get_logger('backtesting/analysis')


class StrategyAnalyzer:
    """
    Analyze and process trading strategy backtest results.

    Provides methods to filter, aggregate, rank, and export strategy performance data.
    Automatically loads results from the default parquet file on initialization and
    offers various analysis options including weighted averaging, metric-based ranking,
    and filtered views by symbol/interval.
    """

    def __init__(self):
        """Initialize the strategy analyzer and load results from the default file."""
        self.results_df = None
        results_file = f'{BACKTESTING_DIR}/mass_test_results_all.parquet'
        self._load_results(results_file)

    # ==================== Public API ====================

    def get_top_strategies(
        self,
        metric,
        min_avg_trades_per_combination,
        limit=DEFAULT_LIMIT,
        aggregate=False,
        interval=None,
        symbol=None,
        weighted=True,
        min_slippage_ticks=None,
        min_symbol_count=None
    ):
        """
        Get top-performing strategies based on a specific metric.

        Args:
            metric: Metric to rank strategies by (e.g., 'profit_factor', 'win_rate',
                   'average_trade_return_percentage_of_margin', 'sharpe_ratio')
            min_avg_trades_per_combination: Minimum average trades per symbol/interval combo
                   to filter out strategies with insufficient data
            limit: Maximum number of top strategies to return
            aggregate: If True, aggregate results across all symbols and intervals.
                      If False, return individual results per symbol/interval
            interval: Filter by specific interval (e.g., '1h', '4h', '1d'). None = all intervals
            symbol: Filter by specific symbol (e.g., 'ES', 'NQ', 'ZS'). None = all symbols
            weighted: If True and aggregate=True, use trade-weighted averages.
                     If False, use simple averages
            min_slippage_ticks: Minimum slippage value to filter by. None = no filter
            min_symbol_count: Minimum number of unique symbols per strategy. None = no filter

        Returns:
            DataFrame with top strategies sorted by metric in descending order.
            Columns include strategy name, metrics, symbol counts, trade counts, etc.

        Raises:
            ValueError: If no results are loaded or results DataFrame is empty
        """
        if self.results_df is None or self.results_df.empty:
            logger.error('No results available. Load results first.')
            raise ValueError('No results available. Load results first.')

        if aggregate:
            # Get aggregated strategies
            df = self._aggregate_strategies(min_avg_trades_per_combination,
                                            interval,
                                            symbol,
                                            weighted,
                                            min_slippage,
                                            min_symbol_count)
        else:
            # Apply common filtering
            df = filter_dataframe(self.results_df,
                                  min_avg_trades_per_combination,
                                  interval,
                                  symbol,
                                  min_slippage,
                                  min_symbol_count)

        # Sort by the metric in descending order
        sorted_df = df.sort_values(by=metric, ascending=False)

        # Save results to a CSV file with formatted column names
        self._save_results_to_csv(metric,
                                  limit,
                                  df_to_save=sorted_df,
                                  aggregate=aggregate,
                                  weighted=weighted,
                                  interval=interval,
                                  symbol=symbol)

        return sorted_df

    # ==================== Private Methods ====================

    def _load_results(self, file_path):
        """Load results from a parquet file."""
        try:
            self.results_df = pd.read_parquet(file_path)
        except Exception as error:
            logger.error(f'Failed to load results from {file_path}: {error}')
            raise

    def _aggregate_strategies(
        self,
        min_avg_trades_per_combination=0,
        interval=None,
        symbol=None,
        weighted=True,
        min_slippage_ticks=None,
        min_symbol_count=None
    ):
        """
        Aggregate strategy results across different symbols and intervals.

        Combines results from multiple test runs (different symbols/intervals) into
        single metrics per strategy. Supports both simple and trade-weighted averaging.

        Args:
            min_avg_trades_per_combination: Minimum average trades per symbol/interval combo
            interval: Filter by specific interval before aggregation. None = all intervals
            symbol: Filter by specific symbol before aggregation. None = all symbols
            weighted: If True, use trade-weighted averages (strategies with more trades
                     have greater influence). If False, use simple averages
            min_slippage_ticks: Minimum slippage value to filter by. None = no filter
            min_symbol_count: Minimum number of unique symbols per strategy. None = no filter

        Returns:
            DataFrame with aggregated metrics per strategy. Each row represents one strategy
            with combined performance across all matching symbols/intervals.

        Raises:
            ValueError: If no results are loaded or results DataFrame is empty
        """
        if self.results_df is None or self.results_df.empty:
            logger.error('No results available. Load results first.')
            raise ValueError('No results available. Load results first.')

        # Apply common filtering
        filtered_df = filter_dataframe(self.results_df,
                                       min_avg_trades_per_combination,
                                       interval,
                                       symbol,
                                       min_slippage,
                                       min_symbol_count)

        # Group by strategy
        grouped = filtered_df.groupby('strategy')

        # Calculate aggregated metrics
        total_trades = grouped['total_trades'].sum()
        symbol_count = grouped['symbol'].nunique()
        interval_count = grouped['interval'].nunique()

        metrics_dict = {
            # Basic info
            'symbol_count': symbol_count,
            'interval_count': interval_count,
            'total_trades': total_trades,
            'avg_trades_per_combination': (total_trades / (symbol_count * interval_count)).round(DECIMAL_PLACES),
            'avg_trades_per_symbol': (total_trades / symbol_count).round(DECIMAL_PLACES),
            'avg_trades_per_interval': (total_trades / interval_count).round(DECIMAL_PLACES),
        }

        if weighted:
            # Calculate weighted metrics
            metrics_dict['win_rate'] = calculate_weighted_win_rate(filtered_df, grouped)

            # Return metrics (contract-based)
            metrics_dict['total_return_percentage_of_contract'] = grouped['total_return_percentage_of_contract'].sum()
            metrics_dict['average_trade_return_percentage_of_contract'] = calculate_average_trade_return(
                metrics_dict['total_return_percentage_of_contract'], metrics_dict['total_trades']
            )

            # These metrics can be averaged as they are already normalized
            metrics_dict['average_win_percentage_of_contract'] = grouped['average_win_percentage_of_contract'].mean()
            metrics_dict['average_loss_percentage_of_contract'] = grouped['average_loss_percentage_of_contract'].mean()
            metrics_dict['average_trade_duration_hours'] = grouped['average_trade_duration_hours'].mean()

            # Calculate profit factor percentage from aggregated wins and losses
            total_wins_percentage = grouped['total_wins_percentage_of_contract'].sum()
            total_losses_percentage = grouped['total_losses_percentage_of_contract'].sum()

            # Recalculate profit factor from aggregated data
            metrics_dict['profit_factor'] = calculate_profit_ratio(
                total_wins_percentage, total_losses_percentage
            )

            # Calculate trade-weighted averages for risk metrics
            risk_metrics = [
                'maximum_drawdown_percentage',
                'sharpe_ratio',
                'sortino_ratio',
                'calmar_ratio',
                'value_at_risk',
                'expected_shortfall',
                'ulcer_index'
            ]

            for metric in risk_metrics:
                metrics_dict[metric] = calculate_trade_weighted_average(
                    filtered_df, metric, total_trades
                )
        else:
            # Averages all metrics across strategies
            metrics_dict.update({
                'win_rate': grouped['win_rate'].mean(),
                'average_trade_duration_hours': grouped['average_trade_duration_hours'].mean(),

                # Return metrics (contract-based)
                'total_return_percentage_of_contract': grouped['total_return_percentage_of_contract'].sum(),
                'average_trade_return_percentage_of_contract': grouped[
                    'average_trade_return_percentage_of_contract'].mean(),
                'average_win_percentage_of_contract': grouped['average_win_percentage_of_contract'].mean(),
                'average_loss_percentage_of_contract': grouped['average_loss_percentage_of_contract'].mean(),

                # Risk metrics
                'profit_factor': grouped['profit_factor'].mean(),
                'maximum_drawdown_percentage': grouped['maximum_drawdown_percentage'].mean(),
                'sharpe_ratio': grouped['sharpe_ratio'].mean(),
                'sortino_ratio': grouped['sortino_ratio'].mean(),
                'calmar_ratio': grouped['calmar_ratio'].mean(),
                'value_at_risk': grouped['value_at_risk'].mean(),
                'expected_shortfall': grouped['expected_shortfall'].mean(),
                'ulcer_index': grouped['ulcer_index'].mean()
            })

        aggregated_df = pd.DataFrame(metrics_dict).reset_index()

        # Apply a minimum symbol count filter if specified
        if min_symbol_count is not None:
            aggregated_df = aggregated_df[aggregated_df['symbol_count'] >= min_symbol_count]

        return aggregated_df

    def _save_results_to_csv(self, metric, limit, df_to_save, aggregate, interval=None, symbol=None, weighted=True):
        """
        Save results to a human-readable CSV file with formatted column names.

        Args:
            metric: Metric name used for sorting
            limit: Maximum number of rows to save
            df_to_save: DataFrame to save
            aggregate: Whether results are aggregated
            interval: Optional interval filter
            symbol: Optional symbol filter
            weighted: Whether aggregation is weighted

        Raises:
            ValueError: If no data available to save
        """
        # Validate input
        if df_to_save is None or df_to_save.empty:
            if self.results_df is None or self.results_df.empty:
                logger.error('No results available to save. Load results first.')
                raise ValueError('No results available to save. Load results first.')
            df_to_save = self.results_df

        try:
            # Limit the number of rows
            limited_df = df_to_save.head(limit) if limit and limit > 0 else df_to_save

            # Prepare data for export
            formatted_df = format_dataframe_for_export(limited_df)

            # Generate filename
            filename = build_filename(metric, aggregate, interval, symbol, weighted)

            # Create output directory and save
            csv_dir = os.path.join(BACKTESTING_DIR, 'csv_results')
            os.makedirs(csv_dir, exist_ok=True)

            file_path = os.path.join(csv_dir, filename)
            formatted_df.to_csv(file_path, index=False)

            logger.info(f'Results saved to {file_path} (limited to {limit} rows)')
        except Exception as error:
            logger.error(f'Failed to save results to CSV: {error}')
            raise
