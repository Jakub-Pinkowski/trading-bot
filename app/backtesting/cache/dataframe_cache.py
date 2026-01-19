import pandas as pd

from app.backtesting.cache.cache_base import Cache
from app.utils.logger import get_logger

logger = get_logger('backtesting/cache/dataframe')

# Create the dataframe cache instance
# Set max_age to 7 days (7 * 24 * 60 * 60 = 604,800 seconds)
dataframe_cache = Cache("dataframe", max_size=50, max_age=604800)


def get_cached_dataframe(filepath):
    """ Get a dataframe from the cache or load it from the disk. """
    if dataframe_cache.contains(filepath):
        return dataframe_cache.get(filepath).copy()

    df = pd.read_parquet(filepath)

    # Store in cache
    dataframe_cache.set(filepath, df)

    # Don't save cache to disk on every miss - this will be handled periodically by MassTester
    # dataframe_cache.save_cache()

    return df.copy()
