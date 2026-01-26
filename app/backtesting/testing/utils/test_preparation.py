import os

import pandas as pd

from app.utils.logger import get_logger
from config import BACKTESTING_DIR

logger = get_logger('backtesting/testing/utils/test_preparation')


# ==================== Existing Results Management ====================

def load_existing_results():
    """Load existing results from the parquet file."""
    parquet_filename = f'{BACKTESTING_DIR}/mass_test_results_all.parquet'
    if os.path.exists(parquet_filename):
        try:
            df = pd.read_parquet(parquet_filename)
            # Create tuples directly from DataFrame columns - O(1) operation with vectorization
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


def test_already_exists(existing_data, month, symbol, interval, strategy):
    """Check if a test with the given parameters already exists in the results."""
    existing_results, existing_combinations = existing_data

    if existing_results.empty:
        return False

    # Check if the combination exists in the set (O(1) operation)
    return (month, symbol, interval, strategy) in existing_combinations
