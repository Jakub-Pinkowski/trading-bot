import os
import pickle
import time
from collections import OrderedDict

from filelock import FileLock

from app.utils.logger import get_logger
from config import CACHE_DIR

# Ensure the cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

logger = get_logger('backtesting/cache/base')

# Cache Configuration Constants
DEFAULT_CACHE_MAX_SIZE = 1000  # Default maximum number of items in the cache before LRU eviction
DEFAULT_CACHE_MAX_AGE = 86400  # Default cache expiration time in seconds (24 hours)
DEFAULT_CACHE_LOCK_TIMEOUT = 60  # File lock timeout in seconds
DEFAULT_CACHE_RETRY_ATTEMPTS = 3  # Number of retry attempts for cache save operations
DEFAULT_CACHE_RETRY_DELAY = 1  # Delay between retry attempts in seconds


def _convert_cache_format(loaded_cache):
    """ Convert the loaded cache to the new format with timestamps. """
    new_cache = OrderedDict()
    current_time = time.time()
    for key, value in loaded_cache.items():
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], (int, float)):
            # Already in the new format
            new_cache[key] = value
        else:
            # Convert to the new format
            new_cache[key] = (current_time, value)
    return new_cache


class Cache:
    """
    Base class for caching functionality.
    This class provides common caching operations that can be extended for specific cache types.
    """

    def __init__(self, cache_name, cache_version=1, max_size=DEFAULT_CACHE_MAX_SIZE, max_age=DEFAULT_CACHE_MAX_AGE):
        """ Initialize a cache instance. """
        self.cache_name = cache_name
        self.cache_version = cache_version
        self.cache_file = os.path.join(CACHE_DIR, f"{cache_name}_cache_v{cache_version}.pkl")
        self.lock_file = os.path.join(CACHE_DIR, f"{cache_name}_cache_v{cache_version}.lock")  # Add a lockfile path
        self.max_size = max_size
        self.max_age = max_age
        self.cache_data = OrderedDict()  # Use OrderedDict for LRU functionality
        self._load_cache()

    def save_cache(self, max_retries=DEFAULT_CACHE_RETRY_ATTEMPTS):
        """Save the cache to disk with file locking and retry mechanism.

        Args:
            max_retries (int): Maximum number of retry attempts (default: 3)

        Returns:
            bool: True if save was successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                # Create a lock for the file
                lock = FileLock(self.lock_file, timeout=DEFAULT_CACHE_LOCK_TIMEOUT)

                # Acquire the lock before writing
                with lock:
                    with open(self.cache_file, 'wb') as f:
                        pickle.dump(self.cache_data, f)

                # Success - log only on retry (not first attempt)
                if attempt > 0:
                    logger.info(f"Successfully saved {self.cache_name} cache on attempt {attempt + 1}")

                return True

            except Exception as save_err:
                # If this was the last attempt, log the error and return False
                if attempt == max_retries - 1:
                    logger.error(
                        f"Failed to save {self.cache_name} cache to {self.cache_file} "
                        f"after {max_retries} attempts: {save_err}"
                    )
                    return False

                # Log warning and retry
                logger.warning(
                    f"Failed to save {self.cache_name} cache (attempt {attempt + 1}/{max_retries}): {save_err}. "
                    f"Retrying in {DEFAULT_CACHE_RETRY_DELAY} second..."
                )
                time.sleep(DEFAULT_CACHE_RETRY_DELAY)  # Wait before retry

        return False

    def get(self, key, default=None):
        """ Get a value from the cache. """
        if not self.contains(key):
            return default

        # Move the item to the end of the OrderedDict to mark it as recently used
        timestamp, value = self.cache_data.pop(key)
        self.cache_data[key] = (timestamp, value)

        return value

    def set(self, key, value):
        """ Set a value in the cache. """

        # Store the value with the current timestamp
        self.cache_data[key] = (time.time(), value)

        # Enforce the cache size limit
        self._enforce_size_limit()

    def contains(self, key):
        """ Check if a key exists in the cache """
        if key not in self.cache_data:
            return False

        # Check if the item is expired
        if self.max_age is not None:
            timestamp, _ = self.cache_data[key]
            if time.time() - timestamp > self.max_age:
                # Remove the expired item
                del self.cache_data[key]
                return False

        return True

    def clear(self):
        """Clear the cache."""
        self.cache_data.clear()

    def size(self):
        """Get the number of items in the cache."""
        return len(self.cache_data)

    # --- Private methods ---

    def _load_cache(self):
        """Load the cache from the disk with file locking."""
        if os.path.exists(self.cache_file):
            # Create a lock for the file
            lock = FileLock(self.lock_file, timeout=DEFAULT_CACHE_LOCK_TIMEOUT)
            try:
                # Acquire the lock before reading
                with lock:
                    with open(self.cache_file, 'rb') as f:
                        loaded_cache = pickle.load(f)
                        # Only use the cache if it's a dictionary
                        if isinstance(loaded_cache, dict):
                            # Convert the cache to the new format
                            self.cache_data = _convert_cache_format(loaded_cache)

                            # Remove expired items
                            self._remove_expired_items()
                        else:
                            logger.error(f"Cache file {self.cache_file} contains invalid data. Using empty cache.")
            except Exception as load_err:
                logger.error(f"Failed to load cache from {self.cache_file}: {load_err}. Using empty cache.")

    def _remove_expired_items(self):
        """Remove expired items from the cache."""
        if self.max_age is None:
            return  # No expiration

        current_time = time.time()
        expired_keys = []

        # Find expired keys
        for key, (timestamp, _) in list(self.cache_data.items()):
            if current_time - timestamp > self.max_age:
                expired_keys.append(key)

        # Remove expired keys
        for key in expired_keys:
            del self.cache_data[key]

        if expired_keys:
            logger.debug(f"Removed {len(expired_keys)} expired items from {self.cache_name} cache")

    def _enforce_size_limit(self):
        """Enforce the cache size limit by removing the least recently used items."""
        if self.max_size is None:
            return  # No size limit

        # Remove the oldest items until we're under the limit
        while len(self.cache_data) > self.max_size:
            # In our implementation, the least recently used item is the first one in the OrderedDict
            # because we move items to the end when they are accessed in the get() method
            self.cache_data.popitem(last=False)  # Remove the first item (least recently used)
