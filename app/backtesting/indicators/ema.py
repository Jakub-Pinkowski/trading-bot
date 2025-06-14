from app.backtesting.cache.indicators_cache import indicator_cache
from app.utils.backtesting_utils.indicators_utils import hash_series, logger


def calculate_ema(prices, period=9):
    # Validate period
    if period <= 0:
        logger.error(f"Invalid period value for EMA: {period}. Period must be a positive integer.")
        raise ValueError("Period must be a positive integer")

    # Create a hashable key for the cache
    prices_hash = hash_series(prices)

    # Check if we have this calculation cached in the global cache
    cache_key = ('ema', prices_hash, period)
    if indicator_cache.contains(cache_key):
        return indicator_cache.get(cache_key)

    # Calculate EMA
    ema = prices.ewm(span=period, adjust=False).mean()

    # Cache the result
    indicator_cache.set(cache_key, ema)

    return ema
