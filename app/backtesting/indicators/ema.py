from app.backtesting.cache.indicators_cache import indicator_cache
from app.utils.backtesting_utils.indicators_utils import hash_series, logger


def calculate_ema(prices, period=9, prices_hash=None):
    """
    Calculate EMA (Exponential Moving Average) indicator.

    Args:
        prices: pandas Series of prices
        period: EMA period (default: 9)
        prices_hash: Optional pre-computed hash of prices series for cache optimization.
                    If None, will be computed. Pass this to avoid redundant hashing
                    when calling multiple indicators on the same data.

    Returns:
        pandas Series with EMA values

    Example:
        # Without pre-computed hash (simple usage)
        ema = calculate_ema(df['close'], period=9)

        # With pre-computed hash (optimized for multiple indicators)
        close_hash = hash_series(df['close'])
        ema_short = calculate_ema(df['close'], period=9, prices_hash=close_hash)
        ema_long = calculate_ema(df['close'], period=21, prices_hash=close_hash)
    """
    # Validate period
    if period <= 0:
        logger.error(f"Invalid period value for EMA: {period}. Period must be a positive integer.")
        raise ValueError("Period must be a positive integer")

    # Create a hashable key for the cache
    if prices_hash is None:
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
