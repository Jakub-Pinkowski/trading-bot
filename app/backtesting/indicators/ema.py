from app.backtesting.cache.indicators_cache import indicator_cache
from app.utils.backtesting_utils.indicators_utils import logger


def calculate_ema(prices, period, prices_hash):
    """
    Calculate EMA (Exponential Moving Average) indicator.

    Args:
        prices: pandas Series of prices
        period: EMA period (e.g., 9, 21)
        prices_hash: Pre-computed hash of prices series (use hash_series() or
                    BaseStrategy._precompute_hashes())

    Returns:
        pandas Series with EMA values

    Example:
        # Pre-compute hash once
        hashes = strategy._precompute_hashes(df)

        # Pass to multiple indicators
        ema_short = calculate_ema(df['close'], period=9, prices_hash=hashes['close'])
        ema_long = calculate_ema(df['close'], period=21, prices_hash=hashes['close'])
    """
    # Validate period
    if period <= 0:
        logger.error(f"Invalid period value for EMA: {period}. Period must be a positive integer.")
        raise ValueError("Period must be a positive integer")


    # Check if we have this calculation cached in the global cache
    cache_key = ('ema', prices_hash, period)
    if indicator_cache.contains(cache_key):
        return indicator_cache.get(cache_key)

    # Calculate EMA
    ema = prices.ewm(span=period, adjust=False).mean()

    # Cache the result
    indicator_cache.set(cache_key, ema)

    return ema
