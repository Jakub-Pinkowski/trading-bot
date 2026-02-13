import os
import pickle
import time
from collections import OrderedDict

from filelock import FileLock, Timeout

from app.utils.logger import get_logger
from config import CACHE_DIR

# Ensure the cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

logger = get_logger('backtesting/cache/base')

# ==================== Constants ====================

# Cache Configuration Constants
DEFAULT_CACHE_MAX_SIZE = 1000  # Default maximum number of items in the cache before LRU eviction
DEFAULT_CACHE_MAX_AGE = 86400  # Default cache expiration time in seconds (24 hours)
DEFAULT_CACHE_LOCK_TIMEOUT = 60  # File lock timeout in seconds
DEFAULT_CACHE_RETRY_ATTEMPTS = 3  # Number of retry attempts for cache save operations
DEFAULT_CACHE_RETRY_DELAY = 1  # Delay between retry attempts in seconds


# ==================== Helper Functions ====================

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

    # ==================== Initialization ====================

    def __init__(self, cache_name, max_size=DEFAULT_CACHE_MAX_SIZE, max_age=DEFAULT_CACHE_MAX_AGE):
        """
        Initialize a cache instance with LRU eviction and TTL expiration.

        Creates an in-memory cache with file persistence, file locking for
        concurrent access, and automatic loading of existing cache data.

        Args:
            cache_name: Identifier for this cache (used for file naming)
            max_size: Maximum number of items to store (LRU eviction when exceeded)
            max_age: Maximum age in seconds before items expire (TTL)

        Side Effects:
            - Creates cache file and lock file in CACHE_DIR
            - Loads existing cache data from disk if available
            - Initializes hit/miss counters to zero
        """
        self.cache_name = cache_name
        self.cache_file = os.path.join(CACHE_DIR, f"{cache_name}_cache.pkl")
        self.lock_file = os.path.join(CACHE_DIR, f"{cache_name}_cache.lock")
        self.max_size = max_size
        self.max_age = max_age
        self.cache_data = OrderedDict()  # Use OrderedDict for LRU functionality
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self._load_cache()

    # ==================== Public Methods ====================

    def save_cache(self, max_retries=DEFAULT_CACHE_RETRY_ATTEMPTS):
        """Save the cache to disk with a file locking and retry mechanism.

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
        """
        Get a value from the cache with LRU ordering update.

        Retrieves the cached value for the given key if it exists and is not expired.
        Updates the LRU ordering by moving the accessed item to the most recently used
        position. Automatically increments cache hit/miss statistics.

        Args:
            key: Cache key to retrieve
            default: Value to return if the key is not found or expired

        Returns:
            Cached value if the key exists and is not expired, otherwise the default value
        """
        if not self.contains(key):
            return default

        # Move the item to the end of the OrderedDict to mark it as recently used
        timestamp, value = self.cache_data.pop(key)
        self.cache_data[key] = (timestamp, value)

        return value

    def set(self, key, value):
        """
        Set a value in the cache with the current timestamp.

        Stores the value with a timestamp for TTL checking. Automatically enforces
        the maximum cache size by evicting the least recently used items when the
        cache is full. Updates LRU ordering.

        Args:
            key: Cache key to store
            value: Value to cache (must be picklable for disk persistence)

        Side Effects:
            - May evict old items if the cache is at max_size
            - Updates cache statistics
            - Marks item as most recently used
        """

        # Store the value with the current timestamp
        self.cache_data[key] = (time.time(), value)

        # Enforce the cache size limit
        self._enforce_size_limit()

    def contains(self, key):
        """
        Check if a key exists in the cache and is not expired.

        Validates both key presence and TTL expiration. Automatically removes
        expired entries from the cache. Updates hit/miss statistics based on
        the result.

        Args:
            key: Cache key to check

        Returns:
            Boolean. True if the key exists and is not expired, False otherwise

        Side Effects:
            - Increments hits counter if the key exists and is valid
            - Increments misses counter if the key is not found or expired
            - Removes expired items from the cache
        """
        if key not in self.cache_data:
            self.misses += 1
            return False

        # Check if the item is expired
        if self.max_age is not None:
            timestamp, _ = self.cache_data[key]
            if time.time() - timestamp > self.max_age:
                # Remove the expired item
                del self.cache_data[key]
                self.misses += 1
                return False

        self.hits += 1
        return True

    def size(self):
        """
        Get the current number of items in the cache.

        Returns the count of cached items currently stored in memory. This count
        excludes expired items that have already been removed but includes items
        that have expired but have not yet been accessed (and thus not yet removed).

        Returns:
            Integer count of items currently in the cache
        """
        return len(self.cache_data)

    def reset_stats(self):
        """
        Reset cache performance statistics to zero.

        Clears the hit-and-miss counters. Typically called at the start of
        backtest runs to measure cache performance for that specific test session.
        Does not affect the cached data itself.

        Side Effects:
            - Sets hits counter to 0
            - Sets misses counter to 0
            - Does not clear cached items
        """
        self.hits = 0
        self.misses = 0

    # ==================== Private Methods ====================

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
            except Timeout:
                logger.warning(f"Cache lock timeout for {self.cache_file} after {DEFAULT_CACHE_LOCK_TIMEOUT}s, proceeding without cache")
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
