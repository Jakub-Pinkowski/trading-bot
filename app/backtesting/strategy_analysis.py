import os
import re
from datetime import datetime

import pandas as pd

from app.utils.logger import get_logger
from config import BACKTESTING_DATA_DIR

logger = get_logger('backtesting/strategy_analysis')


def _filter_dataframe(df, min_avg_trades_per_combination=0, interval=None, symbol=None, min_slippage=None):
    """Filter DataFrame based on common criteria."""

    # Filter by minimum average trades per combination
    if min_avg_trades_per_combination > 0:
        strategy_stats = df.groupby('strategy').agg({
            'total_trades': 'sum',
            'symbol': 'nunique',
            'interval': 'nunique'
        }).reset_index()

        # Calculate combinations (symbol × interval)
        strategy_stats['combination_count'] = strategy_stats['symbol'] * strategy_stats['interval']
        strategy_stats['avg_trades_per_combination'] = (
                strategy_stats['total_trades'] / strategy_stats['combination_count']
        )

        # Filter strategies that meet the minimum
        valid_strategies = strategy_stats[
            strategy_stats['avg_trades_per_combination'] >= min_avg_trades_per_combination
            ]['strategy'].tolist()

        filtered_df = df[df['strategy'].isin(valid_strategies)]
    else:
        filtered_df = df.copy()

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


def _calculate_weighted_win_rate(filtered_df, grouped):
    """Calculate win rate weighted by total trades."""
    total_trades_by_strategy = grouped['total_trades'].sum()
    winning_trades_by_strategy = (filtered_df['win_rate'] * filtered_df['total_trades'] / 100).groupby(
        filtered_df['strategy']).sum()
    return (winning_trades_by_strategy / total_trades_by_strategy * 100).round(2)


def _calculate_average_trade_return(total_return, total_trades):
    """Calculate average trade return from total return and total trades."""
    return (total_return / total_trades).round(2)


def _calculate_profit_ratio(total_wins_percentage, total_losses_percentage):
    """Calculate a profit factor from total wins and losses percentages."""
    return abs(
        total_wins_percentage / total_losses_percentage
    ).replace([float('inf'), float('-inf')], float('inf')).round(2)


def _calculate_trade_weighted_average(filtered_df, metric_name, total_trades_by_strategy):
    """Calculate trade-weighted average for a given metric."""
    weighted_sum = (filtered_df[metric_name] * filtered_df['total_trades']).groupby(
        filtered_df['strategy']).sum()
    return (weighted_sum / total_trades_by_strategy).round(2)


def _parse_strategy_name(strategy_name):
    """ Parse strategy name to extract common parameters and clean the strategy name. """
    # Extract common parameters using regex
    rollover_match = re.search(r'rollover=([^,)]+)', strategy_name)
    trailing_match = re.search(r'trailing=([^,)]+)', strategy_name)
    slippage_match = re.search(r'slippage=([^,)]+)', strategy_name)

    # Extract values or set defaults
    rollover = rollover_match.group(1) if rollover_match else 'False'
    trailing = trailing_match.group(1) if trailing_match else 'None'
    slippage = slippage_match.group(1) if slippage_match else '0'

    # Convert string values to appropriate types
    rollover = rollover == 'True'
    trailing = None if trailing == 'None' else float(trailing)
    slippage = float(slippage)

    # Remove common parameters from the strategy name
    clean_strategy = strategy_name
    for param in ['rollover', 'trailing', 'slippage']:
        clean_strategy = re.sub(f',{param}=[^,)]+', '', clean_strategy)
        clean_strategy = re.sub(f'{param}=[^,)]+,', '', clean_strategy)
        clean_strategy = re.sub(f'{param}=[^,)]+', '', clean_strategy)

    # Clean up any double commas or trailing commas
    clean_strategy = re.sub(r',,+', ',', clean_strategy)
    clean_strategy = re.sub(r',\)', ')', clean_strategy)
    clean_strategy = re.sub(r'\(,', '(', clean_strategy)

    return clean_strategy, rollover, trailing, slippage


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
        min_avg_trades_per_combination,
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
            df = self._aggregate_strategies(min_avg_trades_per_combination, interval, symbol, weighted, min_slippage)
        else:
            # Apply common filtering
            df = _filter_dataframe(self.results_df, min_avg_trades_per_combination, interval, symbol, min_slippage)

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

    def _aggregate_strategies(
        self,
        min_avg_trades_per_combination=0,
        interval=None,
        symbol=None,
        weighted=True,
        min_slippage=None
    ):
        """  Aggregate strategy results across different symbols and intervals. """
        if self.results_df is None or self.results_df.empty:
            logger.error('No results available. Load results first.')
            raise ValueError('No results available. Load results first.')

        # Apply common filtering
        filtered_df = _filter_dataframe(self.results_df, min_avg_trades_per_combination, interval, symbol, min_slippage)

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
            'avg_trades_per_combination': (total_trades / (symbol_count * interval_count)).round(2),
            'avg_trades_per_symbol': (total_trades / symbol_count).round(2),
            'avg_trades_per_interval': (total_trades / interval_count).round(2),
        }

        if weighted:
            # Calculate weighted metrics
            metrics_dict['win_rate'] = _calculate_weighted_win_rate(filtered_df, grouped)

            # Percentage-based metrics
            metrics_dict['total_return_percentage_of_margin'] = grouped['total_return_percentage_of_margin'].sum()
            metrics_dict['average_trade_return_percentage_of_margin'] = _calculate_average_trade_return(
                metrics_dict['total_return_percentage_of_margin'], metrics_dict['total_trades']
            )

            # These metrics can be averaged as they are already normalized
            metrics_dict['average_win_percentage_of_margin'] = grouped['average_win_percentage_of_margin'].mean()
            metrics_dict['average_loss_percentage_of_margin'] = grouped['average_loss_percentage_of_margin'].mean()
            metrics_dict['commission_percentage_of_margin'] = grouped['commission_percentage_of_margin'].mean()

            # Calculate profit factor percentage from aggregated wins and losses
            total_wins_percentage = grouped['total_wins_percentage_of_margin'].sum()
            total_losses_percentage = grouped['total_losses_percentage_of_margin'].sum()

            # Recalculate profit factor from aggregated data
            metrics_dict['profit_factor'] = _calculate_profit_ratio(
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
                metrics_dict[metric] = _calculate_trade_weighted_average(
                    filtered_df, metric, total_trades
                )
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

            # Parse strategy names to extract common parameters
            if 'strategy' in formatted_df.columns:
                strategy_data = formatted_df['strategy'].apply(_parse_strategy_name)
                formatted_df['strategy'] = [data[0] for data in strategy_data]  # Clean strategy name
                formatted_df['rollover'] = [data[1] for data in strategy_data]  # Rollover parameter
                formatted_df['trailing'] = [data[2] for data in strategy_data]  # Trailing parameter
                formatted_df['slippage'] = [data[3] for data in strategy_data]  # Slippage parameter

                # Reorder columns to put common parameters after strategy
                cols = list(formatted_df.columns)
                strategy_idx = cols.index('strategy')
                # Remove the new columns from their current positions
                cols = [col for col in cols if col not in ['rollover', 'trailing', 'slippage']]
                # Insert them after strategy
                cols.insert(strategy_idx + 1, 'rollover')
                cols.insert(strategy_idx + 2, 'trailing')
                cols.insert(strategy_idx + 3, 'slippage')
                formatted_df = formatted_df[cols]

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
