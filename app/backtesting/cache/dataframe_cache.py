import pandas as pd

from app.backtesting.cache.cache_base import Cache
from app.utils.logger import get_logger

logger = get_logger('backtesting/cache/dataframe')

# Create the dataframe cache instance
# Set max_age to 7 days (7 * 24 * 60 * 60 = 604,800 seconds)
dataframe_cache = Cache("dataframe", max_size=50, max_age=604800)


def get_cached_dataframe(filepath):
    """
    Get a DataFrame from the cache or load it from disk.

    IMPORTANT: Returns a copy of the cached DataFrame to prevent cache corruption.

    Why copies are necessary:
    - Strategies modify DataFrames by adding indicator and signal columns
    - Multiple strategies may run on the same data concurrently
    - Without copies, modifications would corrupt the cached data

    Performance optimization:
    - Copy-on-write mode is enabled globally in app/__init__.py
    - This makes copies nearly free until actual modification occurs
    - Memory is shared between copies until one is modified
    - Reduces memory overhead from ~100% to <5% for read-only access

    Args:
        filepath: Path to the parquet file to load

    Returns:
        DataFrame: A copy-on-write copy of the cached DataFrame

    Example:
        >>> df = get_cached_dataframe('data/ES_1d.parquet')
        >>> df['new_column'] = 0  # Only now does the copy actually happen
    """
    if dataframe_cache.contains(filepath):
        # Return a copy-on-write copy (nearly free until modified)
        return dataframe_cache.get(filepath).copy()

    df = pd.read_parquet(filepath)

    # Store in cache
    dataframe_cache.set(filepath, df)

    # Don't save cache to disk on every miss - this will be handled periodically by MassTester
    # dataframe_cache.save_cache()

    # Return a copy-on-write copy (protects the cache from modifications)
    return df.copy()
