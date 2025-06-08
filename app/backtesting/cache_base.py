import os
import pickle

from app.utils.logger import get_logger
from config import BACKTESTING_DATA_DIR

# Get logger
logger = get_logger()


class Cache:
    """
    Base class for caching functionality.
    This class provides common caching operations that can be extended for specific cache types.
    """

    def __init__(self, cache_name, cache_version=1):
        """ Initialize a cache instance. """
        self.cache_name = cache_name
        self.cache_version = cache_version

        # Cache directory
        self.cache_dir = os.path.join(BACKTESTING_DATA_DIR, "cache")
        self.cache_file = os.path.join(self.cache_dir, f"{cache_name}_cache_v{cache_version}.pkl")

        # Ensure the cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize the cache dictionary
        self.cache_data = {}

        # Load cache at initialization
        self._load_cache()

    def _load_cache(self):
        """Load the cache from the disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    loaded_cache = pickle.load(f)
                    # Only use the cache if it's a dictionary
                    if isinstance(loaded_cache, dict):
                        self.cache_data = loaded_cache
                        logger.debug(f"Loaded {len(self.cache_data)} items from {self.cache_name} cache")
                    else:
                        logger.error(f"Cache file {self.cache_file} contains invalid data. Using empty cache.")
            except Exception as load_err:
                logger.error(f"Failed to load cache from {self.cache_file}: {load_err}. Using empty cache.")

    def save_cache(self):
        """Save the cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache_data, f)
            logger.debug(f"{self.cache_name.capitalize()} cache saved to {self.cache_file} with {len(self.cache_data)} entries")
        except Exception as save_err:
            logger.error(f"Failed to save {self.cache_name} cache to {self.cache_file}: {save_err}")

    def get(self, key, default=None):
        """ Get a value from the cache. """
        return self.cache_data.get(key, default)

    def set(self, key, value):
        """ Set a value in the cache.  """
        self.cache_data[key] = value

    def contains(self, key):
        """ Check if a key exists in the cache. """
        return key in self.cache_data

    def clear(self):
        """Clear the cache."""
        self.cache_data.clear()

    def size(self):
        """ Get the number of items in the cache. """
        return len(self.cache_data)
