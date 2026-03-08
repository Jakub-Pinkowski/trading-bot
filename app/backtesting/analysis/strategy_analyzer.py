"""
Strategy Analyzer

Main class for analyzing and exporting trading strategy results.
"""

import os

import pandas as pd

from app.backtesting.analysis.constants import DEFAULT_LIMIT, DECIMAL_PLACES, MIN_TRADES_FOR_RATIO
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
from config import DATA_DIR
from futures_config.symbol_groups import filter_to_one_per_group

logger = get_logger('backtesting/analysis')

# ==================== Module Paths ====================

BACKTESTING_DIR = DATA_DIR / "backtesting"


# ==================== Helper functions ====================

def _aggregate_weighted(filtered_df, grouped, total_trades, symbol_count):
    """
    Calculate trade-weighted aggregated metrics for each strategy group.

    Args:
        filtered_df: Filtered DataFrame with per-combination rows
        grouped: DataFrame grouped by strategy
        total_trades: Series of total trade counts per strategy
        symbol_count: Series of unique symbol counts per strategy

    Returns:
        Dict of metric name → Series, ready to merge into the base metrics_dict
    """
    metrics = {}

    metrics['win_rate'] = calculate_weighted_win_rate(filtered_df, grouped)

    # total_return_all_symbols_pct_contract grows with symbol count;
    # avg_return_per_symbol_pct_contract is symbol-count-neutral for cross-strategy comparison
    if 'total_return_percentage_of_contract' in filtered_df.columns:
        _total_return_sum = grouped['total_return_percentage_of_contract'].sum()
    else:
        _total_return_sum = pd.Series(index=total_trades.index, data=float('nan'))
    metrics['total_return_all_symbols_pct_contract'] = _total_return_sum
    metrics['avg_return_per_symbol_pct_contract'] = (
            _total_return_sum / symbol_count
    ).round(DECIMAL_PLACES)
    metrics['average_trade_return_percentage_of_contract'] = calculate_average_trade_return(
        _total_return_sum, total_trades
    )

    # Guard access in case the input DataFrame does not contain those optional columns
    def _group_mean_or_nan(col_name):
        if col_name in filtered_df.columns:
            return grouped[col_name].mean()
        return pd.Series(index=total_trades.index, data=float('nan'))

    metrics['average_win_percentage_of_contract'] = _group_mean_or_nan('average_win_percentage_of_contract')
    metrics['average_loss_percentage_of_contract'] = _group_mean_or_nan('average_loss_percentage_of_contract')

    # Recalculate profit factor from aggregated wins/losses when available for accuracy;
    # fall back to trade-weighted average of per-combination profit_factor values otherwise
    if ('total_wins_percentage_of_contract' in filtered_df.columns
            and 'total_losses_percentage_of_contract' in filtered_df.columns):
        total_wins_pct = grouped['total_wins_percentage_of_contract'].sum()
        total_losses_pct = grouped['total_losses_percentage_of_contract'].sum()
        metrics['profit_factor'] = calculate_profit_ratio(total_wins_pct, total_losses_pct)
    elif 'profit_factor' in filtered_df.columns:
        metrics['profit_factor'] = calculate_trade_weighted_average(filtered_df, 'profit_factor', total_trades)
    else:
        metrics['profit_factor'] = pd.Series(index=total_trades.index, data=float('nan'))

    risk_metrics = [
        'maximum_drawdown_percentage',
        'sharpe_ratio',
        'sortino_ratio',
        'calmar_ratio',
        'value_at_risk',
        'expected_shortfall',
        'ulcer_index',
        'time_in_market_percentage',
        'win_loss_ratio',
        'max_consecutive_wins',
        'max_consecutive_losses',
        'expectancy_per_bar',
        'average_trade_duration_bars',
    ]

    for metric in risk_metrics:
        if metric in filtered_df.columns:
            if metric in ('sharpe_ratio', 'sortino_ratio', 'calmar_ratio'):
                # Exclude runs below MIN_TRADES_FOR_RATIO to reduce noise in ratio averaging.
                # Limitation: this is still an approximation — ideally these ratios would be
                # recomputed from the combined trade stream.
                metrics[metric] = calculate_trade_weighted_average(
                    filtered_df, metric, total_trades, min_trades=MIN_TRADES_FOR_RATIO
                )
            else:
                metrics[metric] = calculate_trade_weighted_average(filtered_df, metric, total_trades)
        else:
            metrics[metric] = pd.Series(index=total_trades.index, data=float('nan'))

    return metrics


def _aggregate_unweighted(filtered_df, grouped, total_trades, symbol_count):
    """
    Calculate simple-averaged aggregated metrics for each strategy group.

    Args:
        filtered_df: Filtered DataFrame with per-combination rows
        grouped: DataFrame grouped by strategy
        total_trades: Series of total trade counts per strategy
        symbol_count: Series of unique symbol counts per strategy

    Returns:
        Dict of metric name → Series, ready to merge into the base metrics_dict
    """

    def _safe_group_mean(col_name):
        if col_name in filtered_df.columns:
            return grouped[col_name].mean()
        return pd.Series(index=total_trades.index, data=float('nan'))

    # total_return_all_symbols_pct_contract grows with symbol count;
    # avg_return_per_symbol_pct_contract is symbol-count-neutral for cross-strategy comparison
    if 'total_return_percentage_of_contract' in filtered_df.columns:
        _total_return_sum = grouped['total_return_percentage_of_contract'].sum()
    else:
        _total_return_sum = pd.Series(index=total_trades.index, data=float('nan'))

    return {
        'win_rate': _safe_group_mean('win_rate'),
        'average_trade_duration_bars': _safe_group_mean('average_trade_duration_bars'),
        'win_loss_ratio': _safe_group_mean('win_loss_ratio'),
        'max_consecutive_wins': _safe_group_mean('max_consecutive_wins'),
        'max_consecutive_losses': _safe_group_mean('max_consecutive_losses'),
        'total_return_all_symbols_pct_contract': _total_return_sum,
        'avg_return_per_symbol_pct_contract': (_total_return_sum / symbol_count).round(DECIMAL_PLACES),
        'average_trade_return_percentage_of_contract': _safe_group_mean(
            'average_trade_return_percentage_of_contract'),
        'average_win_percentage_of_contract': _safe_group_mean('average_win_percentage_of_contract'),
        'average_loss_percentage_of_contract': _safe_group_mean('average_loss_percentage_of_contract'),
        'profit_factor': _safe_group_mean('profit_factor'),
        'expectancy_per_bar': _safe_group_mean('expectancy_per_bar'),
        # Note: simple means for Sharpe/Sortino/Calmar are an approximation; ideally these
        # would be recomputed from the combined trade stream.
        'maximum_drawdown_percentage': _safe_group_mean('maximum_drawdown_percentage'),
        'sharpe_ratio': _safe_group_mean('sharpe_ratio'),
        'sortino_ratio': _safe_group_mean('sortino_ratio'),
        'calmar_ratio': _safe_group_mean('calmar_ratio'),
        'value_at_risk': _safe_group_mean('value_at_risk'),
        'expected_shortfall': _safe_group_mean('expected_shortfall'),
        'ulcer_index': _safe_group_mean('ulcer_index'),
        'time_in_market_percentage': _safe_group_mean('time_in_market_percentage'),
    }


def _aggregate_strategies(
    df,
    min_avg_trades_per_combination,
    interval,
    symbol,
    weighted,
    min_slippage_ticks,
    min_symbol_count
):
    """
    Aggregate strategy results across different symbols and intervals.

    Combines results from multiple test runs (different symbols/intervals) into
    single metrics per strategy. Supports both simple and trade-weighted averaging.

    Args:
        df: DataFrame with results to aggregate
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
    if df is None or df.empty:
        logger.error('No results available. Load results first.')
        raise ValueError('No results available. Load results first.')

    filtered_df = filter_dataframe(df,
                                   min_avg_trades_per_combination,
                                   interval,
                                   symbol,
                                   min_slippage_ticks,
                                   min_symbol_count)

    grouped = filtered_df.groupby('strategy')
    total_trades = grouped['total_trades'].sum()
    symbol_count = grouped['symbol'].nunique()
    interval_count = grouped['interval'].nunique()

    metrics_dict = {
        'symbol_count': symbol_count,
        'interval_count': interval_count,
        'total_trades': total_trades,
        'avg_trades_per_combination': (total_trades / (symbol_count * interval_count)).round(DECIMAL_PLACES),
        'avg_trades_per_symbol': (total_trades / symbol_count).round(DECIMAL_PLACES),
        'avg_trades_per_interval': (total_trades / interval_count).round(DECIMAL_PLACES),
    }

    if weighted:
        metrics_dict.update(_aggregate_weighted(filtered_df, grouped, total_trades, symbol_count))
    else:
        metrics_dict.update(_aggregate_unweighted(filtered_df, grouped, total_trades, symbol_count))

    return pd.DataFrame(metrics_dict).reset_index()


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
        min_symbol_count=None,
        one_per_group=False
    ):
        """
        Get top-performing strategies based on a specific metric.

        Args:
            metric: Metric to rank strategies by (e.g., 'profit_factor', 'win_rate',
                   'average_trade_return_percentage_of_contract', 'sharpe_ratio')
            min_avg_trades_per_combination: Minimum average trades per symbol/interval combo
                   to filter out strategies with insufficient data
            limit: Maximum number of top strategies to return
            aggregate: If True, aggregate results across all symbols and intervals.
                      If False, return individual results per symbol/interval
            interval: Filter by specific interval (e.g., '1h', '4h', '1d'). None = all intervals
            symbol: Filter by specific symbol (e.g., 'ZC', 'NQ', 'ZS'). None = all symbols
            weighted: If True and aggregate=True, use trade-weighted averages.
                     If False, use simple averages
            min_slippage_ticks: Minimum slippage value to filter by. None = no filter
            min_symbol_count: Minimum number of unique symbols per strategy. None = no filter
            one_per_group: If True, filter to only one symbol per correlated group
                          (e.g., keep ZC but exclude XC/MZC). Prevents pseudo-replication
                          from mini/micro contracts. Recommended for accurate analysis

        Returns:
            DataFrame with top strategies sorted by metric in descending order.
            Columns include strategy name, metrics, symbol counts, trade counts, etc.

        Raises:
            ValueError: If no results are loaded or results DataFrame is empty
        """
        if self.results_df is None or self.results_df.empty:
            logger.error('No results available. Load results first.')
            raise ValueError('No results available. Load results first.')

        # Apply symbol group filtering if requested
        df_to_analyze = self.results_df
        if one_per_group:
            unique_symbols = df_to_analyze['symbol'].unique().tolist()
            filtered_symbols = filter_to_one_per_group(unique_symbols)
            df_to_analyze = df_to_analyze[df_to_analyze['symbol'].isin(filtered_symbols)]

            excluded_count = len(unique_symbols) - len(filtered_symbols)
            if excluded_count > 0:
                logger.info(f'Filtered to one symbol per group: kept {len(filtered_symbols)}, '
                            f'excluded {excluded_count} correlated symbols')

        if aggregate:
            # Get aggregated strategies
            df = _aggregate_strategies(df_to_analyze,
                                       min_avg_trades_per_combination,
                                       interval,
                                       symbol,
                                       weighted,
                                       min_slippage_ticks,
                                       min_symbol_count)
        else:
            # Apply common filtering
            df = filter_dataframe(df_to_analyze,
                                  min_avg_trades_per_combination,
                                  interval,
                                  symbol,
                                  min_slippage_ticks,
                                  min_symbol_count)

        # Sort by the metric in descending order
        sorted_df = df.sort_values(by=metric, ascending=False)

        # Apply limit to the results returned
        result_df = sorted_df.head(limit) if limit and limit > 0 else sorted_df

        # Save results to a CSV file with formatted column names
        self._save_results_to_csv(metric,
                                  limit,
                                  df_to_save=sorted_df,
                                  aggregate=aggregate,
                                  weighted=weighted,
                                  interval=interval,
                                  symbol=symbol)

        return result_df

    # ==================== Private Methods ====================

    def _load_results(self, file_path):
        """Load results from a parquet file."""
        try:
            self.results_df = pd.read_parquet(file_path)
        except Exception as error:
            logger.error(f'Failed to load results from {file_path}: {error}')
            raise

    def _save_results_to_csv(self, metric, limit, df_to_save, aggregate, interval, symbol, weighted):
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
            ValueError: If no data is available to save
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

            # Create an output directory and save
            csv_dir = os.path.join(BACKTESTING_DIR, 'csv_results')
            os.makedirs(csv_dir, exist_ok=True)

            file_path = os.path.join(csv_dir, filename)
            formatted_df.to_csv(file_path, index=False)

            logger.info(f'Results saved to {file_path} (limited to {limit} rows)')
        except Exception as error:
            logger.error(f'Failed to save results to CSV: {error}')
            raise
