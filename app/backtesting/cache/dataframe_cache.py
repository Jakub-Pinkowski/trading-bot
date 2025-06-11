import pandas as pd

from app.backtesting.cache.cache_base import Cache
from app.utils.logger import get_logger

logger = get_logger('backtesting/cache/dataframe')

# Cache version - increment this when preprocessing algorithms change
CACHE_VERSION = 1

# Create the dataframe cache instance
dataframe_cache = Cache("dataframe", CACHE_VERSION)


def get_preprocessed_dataframe(filepath):
    """ Get a preprocessed dataframe from the cache or load it from the disk. """
    if dataframe_cache.contains(filepath):
        logger.debug(f"Cache hit for {filepath}")
        return dataframe_cache.get(filepath).copy()

    logger.debug(f"Cache miss for {filepath}, loading from disk")
    df = pd.read_parquet(filepath)

    # Store in cache
    dataframe_cache.set(filepath, df)

    # Save cache to disk
    dataframe_cache.save_cache()

    return df.copy()
