import pandas as pd

from app.utils.logger import get_logger

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


# TODO [HIGH]: Hardcode which results file to use using config
class StrategyAnalyzer:
    """A class for analyzing and processing trading strategy results."""

    def __init__(self, results_file=None):
        """Initialize the strategy analyzer with an optional result file path."""
        self.results_df = None
        if results_file:
            self.load_results(results_file)

    def load_results(self, file_path):
        """Load results from a parquet file."""
        try:
            self.results_df = pd.read_parquet(file_path)
        except Exception as error:
            logger.error(f'Failed to load results from {file_path}: {error}')
            raise

    def get_top_strategies(self, metric, min_trades=5):
        """Get top-performing strategies based on a specific metric."""
        if self.results_df is None or self.results_df.empty:
            logger.error('No results available. Load results first.')
            raise ValueError('No results available. Load results first.')

        # Filter by minimum trades
        filtered_df = self.results_df[self.results_df['total_trades'] >= min_trades]

        # Sort by the metric in descending order
        sorted_df = filtered_df.sort_values(by=metric, ascending=False)

        return sorted_df

    def save_results_to_csv(self, file_path, df=None, limit=30):
        """Save results to a human-readable CSV file with formatted column names. """
        # Use the provided DataFrame or fallback to self.results_df
        results_to_save = df if df is not None else self.results_df

        if results_to_save is None or results_to_save.empty:
            logger.error('No results available to save. Load results first.')
            raise ValueError('No results available to save. Load results first.')

        try:
            # Create a copy of the DataFrame to avoid modifying the original
            formatted_df = results_to_save.copy()

            # Limit the number of rows
            if limit and limit > 0:
                formatted_df = formatted_df.head(limit)

            # Rename columns for better readability
            formatted_df.columns = [_format_column_name(col) for col in formatted_df.columns]

            # Save to CSV
            formatted_df.to_csv(file_path, index=False)
            print(f'Results saved to {file_path} (limited to {limit} rows)')
            return True
        except Exception as error:
            logger.error(f'Failed to save results to CSV: {error}')
            raise
