import os

import numpy as np
import pandas as pd
from filelock import FileLock, Timeout as FileLockTimeout

from app.utils.logger import get_logger
from config import DATA_DIR

logger = get_logger('backtesting/testing/reporting')

# ==================== Module Paths ====================

BACKTESTING_DIR = DATA_DIR / "backtesting"
SHARDS_DIR = BACKTESTING_DIR / "shards"


# ==================== Results Conversion & Saving ====================

def save_to_parquet(data, file_path):
    """Save data to a parquet file with deduplication and file locking."""
    # Validate data type before acquiring a lock
    if not isinstance(data, pd.DataFrame):
        raise ValueError('Data must be a Pandas DataFrame for parquet format.')

    # Create a directory if needed
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Use an absolute path for lock to avoid conflicts
    abs_file_path = os.path.abspath(file_path)
    lock_path = f"{abs_file_path}.lock"

    try:
        with FileLock(lock_path, timeout=120):  # 2-minute timeout
            # Load existing data if a file exists
            if os.path.exists(file_path):
                try:
                    existing = pd.read_parquet(file_path)
                except Exception as err:
                    logger.error(f'Could not read existing parquet file: {err}')
                    existing = None
            else:
                existing = None

            # Concatenate and deduplicate
            if existing is not None:
                concat = pd.concat([existing, data], ignore_index=True)
                deduped = concat.drop_duplicates()
            else:
                deduped = data

            # Save deduped data
            deduped.to_parquet(file_path, index=False)

    except FileLockTimeout:
        logger.error(f'Failed to acquire lock for {file_path} after 120s')
        raise
    except Exception as e:
        logger.error(f'Error saving parquet file {file_path}: {e}')
        raise


def results_to_dataframe(results):
    """
    Convert the list of test result dictionaries to a structured pandas DataFrame.

    Transforms raw backtest results into a tabular format with standardized columns
    for analysis. Performs validation on all numeric metrics and handles missing/invalid
    values gracefully by replacing with 0.

    Args:
        results: List of result dictionaries from run_single_test(). Each dict contains:
                - month: Month identifier (e.g., '1!')
                - symbol: Futures symbol (e.g., 'ZS', 'CL')
                - interval: Timeframe (e.g., '15m', '1h')
                - strategy: Strategy name with parameters
                - metrics: Dict of performance metrics

    Returns:
        DataFrame with standardized columns including
        - Basic identifiers: month, symbol, interval, strategy
        - Trade statistics: total_trades, win_rate, average_trade_duration_bars,
                          win_loss_ratio, max_consecutive_wins, max_consecutive_losses
        - Return metrics (contract): profit_factor, expectancy_per_bar, average/total returns
        - Risk metrics: maximum_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio,
                       value_at_risk, expected_shortfall, ulcer_index, time_in_market_percentage
        Returns empty DataFrame if results list is empty
    """

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
        'average_trade_duration_bars',
        'win_loss_ratio',
        'max_consecutive_wins',
        'max_consecutive_losses',

        # --- Return Metrics (contract-based) ---
        'total_wins_percentage_of_contract',
        'total_losses_percentage_of_contract',
        'total_return_percentage_of_contract',
        'average_trade_return_percentage_of_contract',
        'average_win_percentage_of_contract',
        'average_loss_percentage_of_contract',
        'largest_win_percentage_of_contract',
        'largest_loss_percentage_of_contract',
        'profit_factor',
        'expectancy_per_bar',
        'return_skewness',

        # --- Risk Metrics ---
        'maximum_drawdown_percentage',
        'sharpe_ratio',
        'sortino_ratio',
        'calmar_ratio',
        'value_at_risk',
        'expected_shortfall',
        'ulcer_index',
        'time_in_market_percentage'
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
                    # Only log the first 5 to avoid spam
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


def save_shard(results, shard_index):
    """
    Write intermediate results to a numbered shard file.

    Appends only — no reading or deduplication. Each call writes a self-contained
    parquet file. Call merge_shards() at the end of a run to combine all shards
    into the final output.

    Args:
        results: List of result dictionaries from run_single_test()
        shard_index: Integer used to name the shard file
            (e.g., 3 -> shard_0003.parquet)

    Returns:
        Path to the written shard file, or None if results produced an empty DataFrame
    """
    results_df = results_to_dataframe(results)
    if results_df.empty:
        return None

    os.makedirs(SHARDS_DIR, exist_ok=True)
    shard_path = SHARDS_DIR / f"shard_{shard_index:04d}.parquet"
    results_df.to_parquet(shard_path, index=False)
    print(f'Shard {shard_index:04d} saved ({len(results_df)} rows)')
    return shard_path


def merge_shards(shard_paths):
    """
    Merge all shard files into the final parquet file with deduplication.

    Reads every shard, concatenates with any existing final results file,
    deduplicates, writes the final output, and removes the shard files.

    Args:
        shard_paths: List of Path objects pointing to shard files to merge

    Returns:
        None. Saves to: {BACKTESTING_DIR}/mass_test_results_all.parquet
    """
    if not shard_paths:
        logger.warning('No shards to merge.')
        return

    final_path = BACKTESTING_DIR / 'mass_test_results_all.parquet'

    dfs = []
    failed_paths = []
    for path in shard_paths:
        try:
            dfs.append(pd.read_parquet(path))
        except Exception as err:
            logger.error(f'Could not read shard {path}: {err}')
            failed_paths.append(path)

    if not dfs:
        logger.error('All shards failed to read; skipping merge.')
        return

    combined = pd.concat(dfs, ignore_index=True)

    # Merge with existing final file if present
    if os.path.exists(final_path):
        try:
            existing = pd.read_parquet(final_path)
            combined = pd.concat([existing, combined], ignore_index=True)
        except Exception as err:
            logger.error(f'Could not read existing results file: {err}')

    deduped = combined.drop_duplicates()
    deduped.to_parquet(final_path, index=False)
    print(f'Results merged and saved to {final_path} ({len(deduped)} rows)')

    # Clean up successfully read shard files only
    readable_paths = [p for p in shard_paths if p not in failed_paths]
    for path in readable_paths:
        try:
            os.remove(path)
        except Exception as err:
            logger.warning(f'Could not remove shard {path}: {err}')

    # Log corrupt shards left on disk for manual inspection
    for path in failed_paths:
        logger.warning(f'Corrupt shard left on disk for manual inspection: {path}')

    # Remove shards directory if now empty
    try:
        SHARDS_DIR.rmdir()
    except Exception as err:
        logger.debug(f'Could not remove shards directory (may not be empty): {err}')
