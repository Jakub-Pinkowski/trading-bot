from app.backtesting.cache.cache_base import Cache

# Cache version - increment this when indicator algorithms change
CACHE_VERSION = 1

# Create the indicator cache instance
indicator_cache = Cache("indicator", CACHE_VERSION)
