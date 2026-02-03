import os

import pandas as pd

from app.utils.logger import get_logger
from config import BACKTESTING_DIR

logger = get_logger('backtesting/testing/utils/test_preparation')


# ==================== Existing Results Management ====================

def load_existing_results():
    """
    Load existing test results from parquet file for duplicate detection.

    Reads the aggregated results file and creates an optimized lookup structure
    for O(1) checking of whether a specific test combination has already been run.
    This enables efficient skipping of duplicate tests.

    Args:
        None. Uses BACKTESTING_DIR config to locate the results file

    Returns:
        Tuple of (DataFrame, set):
        - DataFrame: Complete existing results with all columns
        - set: Set of (month, symbol, interval, strategy) tuples for fast lookup
        Returns (empty DataFrame, empty set) if file doesn't exist or load fails
    """
    parquet_filename = f'{BACKTESTING_DIR}/mass_test_results_all.parquet'
    if os.path.exists(parquet_filename):
        try:
            df = pd.read_parquet(parquet_filename)
            # Create set of tuples for O(1) lookup (set construction is O(n))
            existing_combinations = set(zip(
                df['month'].values,
                df['symbol'].values,
                df['interval'].values,
                df['strategy'].values
            ))
            return df, existing_combinations
        except Exception as error:
            logger.error(f'Failed to load existing results: {error}')
    return pd.DataFrame(), set()


def check_test_exists(existing_data, month, symbol, interval, strategy):
    """
    Check if a specific test combination has already been executed.

    Performs O(1) lookup in the pre-built set of existing test combinations
    to determine if a test needs to be run or can be skipped.

    Args:
        existing_data: Tuple of (DataFrame, set) from load_existing_results()
        month: Month identifier to check (e.g., '1!', '2!')
        symbol: Symbol to check (e.g., 'ZS', 'CL', 'GC')
        interval: Interval to check (e.g., '15m', '1h', '4h')
        strategy: Full strategy name with parameters to check

    Returns:
        Boolean. True if this exact combination exists in the results database,
        False if it needs to be run
    """
    existing_results, existing_combinations = existing_data

    if existing_results.empty:
        return False

    # Check if the combination exists in the set (O(1) operation)
    return (month, symbol, interval, strategy) in existing_combinations
