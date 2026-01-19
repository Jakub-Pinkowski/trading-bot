from app.backtesting.cache.cache_base import Cache

# Create the indicator cache instance
# Set max_age to 30 days (30 * 24 * 60 * 60 = 2,592,000 seconds)
indicator_cache = Cache("indicator", max_size=500, max_age=2592000)
