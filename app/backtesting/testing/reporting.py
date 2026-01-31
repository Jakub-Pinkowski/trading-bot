import numpy as np
import pandas as pd

from app.utils.file_utils import save_to_parquet
from app.utils.logger import get_logger
from config import BACKTESTING_DIR

logger = get_logger('backtesting/testing/reporting')


# ==================== Results Conversion & Saving ====================

def results_to_dataframe(results):
    """Convert results to a pandas DataFrame."""

    if not results:
        logger.warning('No results available to convert to DataFrame.')
        return pd.DataFrame()

    # Define column names
    columns = [
        # --- Basic Trade Statistics ---
        'month',
        'symbol',
        'interval',
        'strategy',
        'total_trades',
        'win_rate',
        'average_trade_duration_hours',

        # --- Return Metrics --- (contract-based)
        'total_wins_percentage_of_contract',
        'total_losses_percentage_of_contract',
        'total_return_percentage_of_contract',
        'average_trade_return_percentage_of_contract',
        'average_win_percentage_of_contract',
        'average_loss_percentage_of_contract',
        'profit_factor',

        # --- Risk Metrics ---
        'maximum_drawdown_percentage',
        'sharpe_ratio',
        'sortino_ratio',
        'calmar_ratio',
        'value_at_risk',
        'expected_shortfall',
        'ulcer_index'
    ]

    # Pre-allocate arrays for each column
    n_results = len(results)

    # Create arrays for numeric columns
    numeric_columns = columns[4:]  # All columns except month, symbol, interval, strategy

    # Pre-allocate arrays for all columns with proper types
    # Using dict[str, list] to avoid type inference issues
    data: dict[str, list] = {
        'month': [''] * n_results,
        'symbol': [''] * n_results,
        'interval': [''] * n_results,
        'strategy': [''] * n_results,
    }

    # Add numeric columns with proper float initialization
    for col in numeric_columns:
        data[col] = [0.0] * n_results

    # Track validation issues
    missing_metrics_count = 0
    type_mismatch_count = 0

    # Fill the arrays directly
    for i, result in enumerate(results):
        metrics = result['metrics']
        data['month'][i] = result['month']
        data['symbol'][i] = result['symbol']
        data['interval'][i] = result['interval']
        data['strategy'][i] = result['strategy']

        # Fill numeric columns with validation
        for col in numeric_columns:
            if col not in metrics:
                # Log warning for missing critical metrics
                if col in ['total_trades', 'win_rate', 'total_return_percentage_of_contract']:
                    missing_metrics_count += 1
                    # Only log first 5 to avoid spam
                    if missing_metrics_count <= 5:
                        logger.warning(
                            f"Critical metric '{col}' missing for {result.get('strategy', 'unknown')} "
                            f"({result.get('symbol', 'unknown')}, {result.get('interval', 'unknown')}, "
                            f"{result.get('month', 'unknown')}). Using 0."
                        )
                data[col][i] = 0
            else:
                value = metrics[col]
                # Validate numeric type
                if not isinstance(value, (int, float, np.number)):
                    type_mismatch_count += 1
                    # Only log the first 5 to avoid spam
                    if type_mismatch_count <= 5:
                        logger.warning(
                            f"Type mismatch for metric '{col}': expected numeric, got {type(value).__name__} "
                            f"(value: {value}) for {result.get('strategy', 'unknown')}. Using 0."
                        )
                    data[col][i] = 0
                # Check for inf/NaN values
                elif np.isnan(value) or np.isinf(value):
                    logger.warning(
                        f"Invalid value ({value}) for metric '{col}' in {result.get('strategy', 'unknown')}. Using 0."
                    )
                    data[col][i] = 0
                else:
                    data[col][i] = value

    # Log summary if there were validation issues
    if missing_metrics_count > 0:
        logger.warning(f"Total missing critical metrics: {missing_metrics_count}")
    if type_mismatch_count > 0:
        logger.warning(f"Total type mismatches: {type_mismatch_count}")

    # Create DataFrame from pre-filled arrays
    return pd.DataFrame(data)


def save_results(results):
    """Save results to one big parquet file."""
    try:
        # Convert results to DataFrame
        results_df = results_to_dataframe(results)
        if not results_df.empty:
            # Save all results to one big parquet file with unique entries
            parquet_filename = f'{BACKTESTING_DIR}/mass_test_results_all.parquet'
            save_to_parquet(results_df, parquet_filename)
            print(f'Results saved to {parquet_filename}')
        else:
            print('No results to save.')
    except Exception as error:
        logger.error(f'Failed to save results: {error}')
