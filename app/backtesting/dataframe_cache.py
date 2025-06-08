import os
import pickle

import pandas as pd

from app.utils.logger import get_logger
from config import BACKTESTING_DATA_DIR

# Get logger
logger = get_logger()

# Cache version - increment this when preprocessing algorithms change
CACHE_VERSION = 1

# Cache file path
CACHE_DIR = os.path.join(BACKTESTING_DATA_DIR, "cache")
CACHE_FILE = os.path.join(CACHE_DIR, f"dataframe_cache_v{CACHE_VERSION}.pkl")

# Ensure the cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Dictionary to store filepath -> dataframe mapping
dataframe_cache = {}

# Load cache at startup
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, 'rb') as f:
            loaded_cache = pickle.load(f)
            # Only use the cache if it's a dictionary
            if isinstance(loaded_cache, dict):
                dataframe_cache = loaded_cache
                logger.debug(f"Loaded {len(dataframe_cache)} dataframes from cache")
            else:
                logger.error(f"Cache file {CACHE_FILE} contains invalid data. Using empty cache.")
    except Exception as load_err:
        logger.error(f"Failed to load cache from {CACHE_FILE}: {load_err}. Using empty cache.")


def get_preprocessed_dataframe(filepath):
    """ Get a preprocessed dataframe from the cache or load it from the disk. """
    if filepath in dataframe_cache:
        logger.debug(f"Cache hit for {filepath}")
        return dataframe_cache[filepath].copy()

    logger.debug(f"Cache miss for {filepath}, loading from disk")
    df = pd.read_parquet(filepath)

    # Store in cache
    dataframe_cache[filepath] = df

    # Save cache to disk
    save_cache()

    return df.copy()


def save_cache():
    """Save the dataframe cache to disk."""
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(dataframe_cache, f)
        logger.debug(f"Dataframe cache saved to {CACHE_FILE} with {len(dataframe_cache)} entries")
    except Exception as save_err:
        logger.error(f"Failed to save dataframe cache to {CACHE_FILE}: {save_err}")
