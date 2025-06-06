import os
from datetime import datetime

import pandas as pd

from app.utils.logger import get_logger
from config import BACKTESTING_DATA_DIR

logger = get_logger()


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
        'win_rate': 'win_rate_%'
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
        self.load_results(results_file)

    def load_results(self, file_path):
        """Load results from a parquet file."""
        try:
            self.results_df = pd.read_parquet(file_path)
        except Exception as error:
            logger.error(f'Failed to load results from {file_path}: {error}')
            raise

    def aggregate_strategies(self, min_trades=0):
        """ Aggregate strategy results across different symbols and intervals. """
        if self.results_df is None or self.results_df.empty:
            logger.error('No results available. Load results first.')
            raise ValueError('No results available. Load results first.')

        # Filter by minimum trades
        filtered_df = self.results_df[self.results_df['total_trades'] >= min_trades]

        # Group by strategy
        grouped = filtered_df.groupby('strategy')

        # Calculate aggregated metrics
        aggregated_df = pd.DataFrame({
            'symbol_count': grouped['symbol'].nunique(),
            'interval_count': grouped['interval'].nunique(),
            'total_trades': grouped['total_trades'].sum(),
            'win_rate': grouped['win_rate'].mean(),
            'profit_factor': grouped['profit_factor'].mean(),
            'total_return_percentage_of_margin': grouped['total_return_percentage_of_margin'].sum(),
            'average_trade_return_percentage_of_margin': grouped['average_trade_return_percentage_of_margin'].mean(),
            'average_win_percentage_of_margin': grouped['average_win_percentage_of_margin'].mean(),
            'average_loss_percentage_of_margin': grouped['average_loss_percentage_of_margin'].mean(),
            'maximum_drawdown_percentage': grouped['maximum_drawdown_percentage'].mean(),
            'total_net_pnl': grouped['total_net_pnl'].sum(),
            'avg_trade_net_pnl': grouped['avg_trade_net_pnl'].mean()
        }).reset_index()

        return aggregated_df

    def get_top_strategies(self, metric, min_trades, limit=30, aggregate=False):
        """ Get top-performing strategies based on a specific metric. """
        if self.results_df is None or self.results_df.empty:
            logger.error('No results available. Load results first.')
            raise ValueError('No results available. Load results first.')

        if aggregate:
            # Get aggregated strategies
            df = self.aggregate_strategies(min_trades)
        else:
            # Filter by minimum trades
            df = self.results_df[self.results_df['total_trades'] >= min_trades]

        # Sort by the metric in descending order
        sorted_df = df.sort_values(by=metric, ascending=False)

        # Save results to a CSV file with formatted column names
        self.save_results_to_csv(metric, limit, df_to_save=sorted_df, aggregate=aggregate)
        print(f"Top strategies by {metric} saved")

        return sorted_df

    def save_results_to_csv(self, metric, limit, df_to_save, aggregate=False):
        """ Save results to a human-readable CSV file with formatted column names. """
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

            # Rename columns for better readability
            formatted_df.columns = [_format_column_name(col) for col in formatted_df.columns]

            # Create a timestamp in the format YYYY-MM-DD HH:MM
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

            # Create a filename based on the metric and whether it's aggregated
            agg_suffix = "_aggregated" if aggregate else ""
            filename = f"{timestamp} top_strategies_by_{metric}{agg_suffix}.csv"

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
