from app.backtesting.cache.cache_base import Cache

# Cache version - increment this when indicator algorithms change
CACHE_VERSION = 1

# Create the indicator cache instance
# Set max_age to 30 days (30 * 24 * 60 * 60 = 2,592,000 seconds)
indicator_cache = Cache("indicator", CACHE_VERSION, max_size=1000, max_age=2592000)
