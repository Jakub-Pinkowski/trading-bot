import os
from datetime import datetime

import pandas as pd

from app.utils.logger import get_logger
from config import BACKTESTING_DATA_DIR

logger = get_logger('backtesting/strategy_analysis')


def _format_column_name(column_name):
    """Convert snake_case column names to Title Case with spaces for better readability.
    Also provides shorter names for specific long column names."""

    # Special cases for very long column names
    column_name_mapping = {
        'average_trade_return_percentage_of_margin': 'avg_return_%',
        'average_win_percentage_of_margin': 'avg_win_%',
        'total_return_percentage_of_margin': 'total_return_%',
        'average_loss_percentage_of_margin': 'avg_loss_%',
        'maximum_drawdown_percentage': 'max_drawdown_%',
        'win_rate': 'win_rate_%',
        'max_consecutive_wins': 'max_cons_wins',
        'max_consecutive_losses': 'max_cons_losses',
        'sharpe_ratio': 'sharpe',
        'sortino_ratio': 'sortino',
        'calmar_ratio': 'calmar',
        'value_at_risk': 'var_95%',
        'expected_shortfall': 'cvar_95%',
        'ulcer_index': 'ulcer_idx'
    }

    # Check if this column has a special shorter name
    if column_name in column_name_mapping:
        # Get the shortened name and capitalize each word
        shortened_name = column_name_mapping[column_name]
        # Split by _ and capitalize each part
        return ' '.join(word.capitalize() for word in shortened_name.split('_'))

    # Default case: Replace underscores with spaces and capitalize each word
    return ' '.join(word.capitalize() for word in column_name.split('_'))


def _filter_dataframe(df, min_trades=0, interval=None, symbol=None, min_slippage=None):
    """Filter DataFrame based on common criteria."""
    # Filter by minimum trades
    filtered_df = df[df['total_trades'] >= min_trades]

    # Filter by interval if provided
    if interval:
        filtered_df = filtered_df[filtered_df['interval'] == interval]

    # Filter by symbol if provided
    if symbol:
        filtered_df = filtered_df[filtered_df['symbol'] == symbol]

    # Filter by minimum slippage if provided
    if min_slippage is not None:
        # Extract slippage from the strategy name
        filtered_df = filtered_df[
            filtered_df['strategy'].str.extract(r'slippage=([^,\)]+)')[0].astype(float) >= min_slippage]

    return filtered_df


class StrategyAnalyzer:
    """A class for analyzing and processing trading strategy results."""

    def __init__(self):
        """Initialize the strategy analyzer and load results from the default file."""
        self.results_df = None
        results_file = f'{BACKTESTING_DATA_DIR}/mass_test_results_all.parquet'
        self._load_results(results_file)

    def get_top_strategies(
        self,
        metric,
        min_trades,
        limit=30,
        aggregate=False,
        interval=None,
        symbol=None,
        weighted=True,
        min_slippage=None
    ):
        """  Get top-performing strategies based on a specific metric. """
        if self.results_df is None or self.results_df.empty:
            logger.error('No results available. Load results first.')
            raise ValueError('No results available. Load results first.')

        if aggregate:
            # Get aggregated strategies
            df = self._aggregate_strategies(min_trades, interval, symbol, weighted, min_slippage)
        else:
            # Apply common filtering
            df = _filter_dataframe(self.results_df, min_trades, interval, symbol, min_slippage)

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

        # Create a message with interval and symbol information
        interval_msg = f" for {interval} interval" if interval else ""
        symbol_msg = f" and {symbol} symbol" if symbol else ""
        weighted_msg = " with weighted aggregation" if aggregate and weighted else ""
        weighted_msg = " with simple aggregation" if aggregate and not weighted else weighted_msg
        print(f"Top strategies by {metric}{interval_msg}{symbol_msg}{weighted_msg} saved")

        return sorted_df

    # --- Private methods ---

    def _load_results(self, file_path):
        """Load results from a parquet file."""
        try:
            self.results_df = pd.read_parquet(file_path)
        except Exception as error:
            logger.error(f'Failed to load results from {file_path}: {error}')
            raise

    def _aggregate_strategies(self, min_trades=0, interval=None, symbol=None, weighted=True, min_slippage=None):
        """  Aggregate strategy results across different symbols and intervals. """
        if self.results_df is None or self.results_df.empty:
            logger.error('No results available. Load results first.')
            raise ValueError('No results available. Load results first.')

        # Apply common filtering
        filtered_df = _filter_dataframe(self.results_df, min_trades, interval, symbol, min_slippage)

        # Group by strategy
        grouped = filtered_df.groupby('strategy')

        # Calculate aggregated metrics
        metrics_dict = {
            # Basic info
            'symbol_count': grouped['symbol'].nunique(),
            'interval_count': grouped['interval'].nunique(),
            'total_trades': grouped['total_trades'].sum(),
        }

        if weighted:
            # Calculates win rate from total trades
            total_trades_by_strategy = grouped['total_trades'].sum()
            # Estimate winning trades by multiplying win_rate by total_trades for each strategy
            winning_trades_by_strategy = (filtered_df['win_rate'] * filtered_df['total_trades'] / 100).groupby(
                filtered_df['strategy']).sum()
            # Calculate aggregated win rate
            metrics_dict['win_rate'] = (winning_trades_by_strategy / total_trades_by_strategy * 100).round(2)

            # Percentage-based metrics
            metrics_dict['total_return_percentage_of_margin'] = grouped['total_return_percentage_of_margin'].sum()

            # Calculate average trade return from total return and total trades
            metrics_dict['average_trade_return_percentage_of_margin'] = (
                    metrics_dict['total_return_percentage_of_margin'] / metrics_dict['total_trades']
            ).round(2)

            # These metrics can be averaged as they are already normalized
            metrics_dict['average_win_percentage_of_margin'] = grouped['average_win_percentage_of_margin'].mean()
            metrics_dict['average_loss_percentage_of_margin'] = grouped['average_loss_percentage_of_margin'].mean()
            metrics_dict['commission_percentage_of_margin'] = grouped['commission_percentage_of_margin'].mean()

            # Recalculate risk metrics from aggregated data

            # Profit factor: total profit / total loss
            # We need to estimate total profit and total loss from the available metrics
            # Total profit = winning_trades * average_win
            # Total loss = losing_trades * average_loss
            winning_trades = winning_trades_by_strategy
            losing_trades = total_trades_by_strategy - winning_trades

            avg_win_by_strategy = filtered_df.groupby('strategy')['average_win_percentage_of_margin'].mean()
            avg_loss_by_strategy = filtered_df.groupby('strategy')['average_loss_percentage_of_margin'].mean()

            total_profit = (winning_trades * avg_win_by_strategy).abs()
            total_loss = (losing_trades * avg_loss_by_strategy).abs()

            # Calculate profit factor
            metrics_dict['profit_factor'] = (total_profit / total_loss).replace([float('inf'), float('-inf')],
                                                                                float('inf')).round(2)

            # Maximum drawdown percentage - use weighted average based on total trades
            metrics_dict['maximum_drawdown_percentage'] = (
                                                                  filtered_df['maximum_drawdown_percentage'] *
                                                                  filtered_df['total_trades']
                                                          ).groupby(
                filtered_df['strategy']).sum() / total_trades_by_strategy

            # Return to drawdown ratio - recalculate from total return and maximum drawdown
            metrics_dict['return_to_drawdown_ratio'] = (
                    metrics_dict['total_return_percentage_of_margin'] / metrics_dict['maximum_drawdown_percentage']
            ).replace([float('inf'), float('-inf')], float('inf')).round(2)

            # For Sharpe, Sortino, and Calmar ratios, we need to recalculate them based on the aggregated returns
            # Since we don't have access to individual trade returns, we'll use a weighted average based on total trades
            metrics_dict['sharpe_ratio'] = (
                                                   filtered_df['sharpe_ratio'] * filtered_df['total_trades']
                                           ).groupby(filtered_df['strategy']).sum() / total_trades_by_strategy

            metrics_dict['sortino_ratio'] = (
                                                    filtered_df['sortino_ratio'] * filtered_df['total_trades']
                                            ).groupby(filtered_df['strategy']).sum() / total_trades_by_strategy

            metrics_dict['calmar_ratio'] = (
                                                   filtered_df['calmar_ratio'] * filtered_df['total_trades']
                                           ).groupby(filtered_df['strategy']).sum() / total_trades_by_strategy

            # Add the new risk metrics with weighted average based on total trades
            metrics_dict['value_at_risk'] = (
                                                    filtered_df['value_at_risk'] * filtered_df['total_trades']
                                            ).groupby(filtered_df['strategy']).sum() / total_trades_by_strategy

            metrics_dict['expected_shortfall'] = (
                                                         filtered_df['expected_shortfall'] * filtered_df['total_trades']
                                                 ).groupby(filtered_df['strategy']).sum() / total_trades_by_strategy

            metrics_dict['ulcer_index'] = (
                                                  filtered_df['ulcer_index'] * filtered_df['total_trades']
                                          ).groupby(filtered_df['strategy']).sum() / total_trades_by_strategy

            # Round the ratio metrics
            for ratio in [
                'sharpe_ratio',
                'sortino_ratio',
                'calmar_ratio',
                'value_at_risk',
                'expected_shortfall',
                'ulcer_index'
            ]:
                metrics_dict[ratio] = metrics_dict[ratio].round(2)
        else:
            # Averages all metrics across strategies
            metrics_dict.update({
                'win_rate': grouped['win_rate'].mean(),

                # Percentage-based metrics
                'total_return_percentage_of_margin': grouped['total_return_percentage_of_margin'].sum(),
                'average_trade_return_percentage_of_margin': grouped[
                    'average_trade_return_percentage_of_margin'].mean(),
                'average_win_percentage_of_margin': grouped['average_win_percentage_of_margin'].mean(),
                'average_loss_percentage_of_margin': grouped['average_loss_percentage_of_margin'].mean(),
                'commission_percentage_of_margin': grouped['commission_percentage_of_margin'].mean(),

                # Risk metrics
                'profit_factor': grouped['profit_factor'].mean(),
                'maximum_drawdown_percentage': grouped['maximum_drawdown_percentage'].mean(),
                'return_to_drawdown_ratio': grouped['return_to_drawdown_ratio'].mean(),
                'sharpe_ratio': grouped['sharpe_ratio'].mean(),
                'sortino_ratio': grouped['sortino_ratio'].mean(),
                'calmar_ratio': grouped['calmar_ratio'].mean(),
                'value_at_risk': grouped['value_at_risk'].mean(),
                'expected_shortfall': grouped['expected_shortfall'].mean(),
                'ulcer_index': grouped['ulcer_index'].mean()
            })

        aggregated_df = pd.DataFrame(metrics_dict).reset_index()

        return aggregated_df

    def _save_results_to_csv(self, metric, limit, df_to_save, aggregate, interval=None, symbol=None, weighted=True):
        """  Save results to a human-readable CSV file with formatted column names. """
        if df_to_save is None:
            if self.results_df is None or self.results_df.empty:
                logger.error('No results available to save. Load results first.')
                raise ValueError('No results available to save. Load results first.')
            df_to_save = self.results_df

        try:
            # Create a copy of the DataFrame to avoid modifying the original
            formatted_df = df_to_save.copy()

            # Limit the number of rows
            if limit and limit > 0:
                formatted_df = formatted_df.head(limit)

            # Format all numeric columns to 2 decimal places
            numeric_cols = formatted_df.select_dtypes(include='number').columns
            formatted_df[numeric_cols] = formatted_df[numeric_cols].round(2)

            # Rename columns for better readability
            formatted_df.columns = [_format_column_name(col) for col in formatted_df.columns]

            # Create a timestamp in the format YYYY-MM-DD HH:MM
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

            # Create a filename based on the metric, interval, symbol, and whether it's aggregated
            agg_suffix = "_aggregated" if aggregate else ""
            weighted_suffix = "_weighted" if aggregate and weighted else "_simple" if aggregate else ""
            interval_suffix = f"_{interval}" if interval else ""
            symbol_suffix = f"_{symbol}" if symbol else ""
            filename = f"{timestamp} top_strategies_by_{metric}{interval_suffix}{symbol_suffix}{agg_suffix}{weighted_suffix}.csv"

            # Create the csv_results directory if it doesn't exist
            csv_dir = os.path.join(BACKTESTING_DATA_DIR, 'csv_results')
            os.makedirs(csv_dir, exist_ok=True)

            # Create the full file path
            file_path = os.path.join(csv_dir, filename)

            # Save to CSV
            formatted_df.to_csv(file_path, index=False)
            print(f'Results saved to {file_path} (limited to {limit} rows)')
        except Exception as error:
            logger.error(f'Failed to save results to CSV: {error}')
            raise
