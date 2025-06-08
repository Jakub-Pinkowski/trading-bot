import os
import pickle

from app.utils.logger import get_logger
from config import BACKTESTING_DATA_DIR

# TODO [MEDIUM]: Consolidate logic with dataframe_cache.py
# TODO [MEDIUM]: Create a separate folder for caching

# Get logger
logger = get_logger()

# Cache version - increment this when indicator algorithms change
CACHE_VERSION = 1

# Cache file path
CACHE_DIR = os.path.join(BACKTESTING_DATA_DIR, "cache")
CACHE_FILE = os.path.join(CACHE_DIR, f"indicator_cache_v{CACHE_VERSION}.pkl")

# Ensure the cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Load cache at startup
indicator_cache = {}
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, 'rb') as f:
            loaded_cache = pickle.load(f)
            # Only use the cache if it's a dictionary
            if isinstance(loaded_cache, dict):
                indicator_cache = loaded_cache
            else:
                logger.error(f"Cache file {CACHE_FILE} contains invalid data. Using empty cache.")
    except Exception as load_err:
        logger.error(f"Failed to load cache from {CACHE_FILE}: {load_err}. Using empty cache.")


def save_cache():
    """Save the indicator cache to disk."""
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(indicator_cache, f)
        logger.debug(f"Indicator cache saved to {CACHE_FILE}")
    except Exception as save_err:
        logger.error(f"Failed to save cache to {CACHE_FILE}: {save_err}")
