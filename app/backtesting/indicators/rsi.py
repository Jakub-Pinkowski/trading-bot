from app.backtesting.cache.indicators_cache import indicator_cache
from app.utils.backtesting_utils.indicators_utils import hash_series, logger


def calculate_rsi(prices, period=14, prices_hash=None):
    """
    Calculate RSI (Relative Strength Index) indicator.

    Args:
        prices: pandas Series of prices
        period: RSI period (default: 14)
        prices_hash: Optional pre-computed hash of prices series for cache optimization.
                    If None, will be computed. Pass this to avoid redundant hashing
                    when calling multiple indicators on the same data.

    Returns:
        pandas Series with RSI values

    Example:
        # Without pre-computed hash (simple usage)
        rsi = calculate_rsi(df['close'], period=14)

        # With pre-computed hash (optimized for multiple indicators)
        close_hash = hash_series(df['close'])
        rsi = calculate_rsi(df['close'], period=14, prices_hash=close_hash)
        rsi_long = calculate_rsi(df['close'], period=21, prices_hash=close_hash)
    """
    # Validate period
    if period <= 0:
        logger.error(f"Invalid period value for RSI: {period}. Period must be a positive integer.")
        raise ValueError("Period must be a positive integer")

    # Create a hashable key for the cache
    if prices_hash is None:
        prices_hash = hash_series(prices)

    # Check if we have this calculation cached in the global cache
    cache_key = ('rsi', prices_hash, period)
    if indicator_cache.contains(cache_key):
        return indicator_cache.get(cache_key)

    # Calculate RSI
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Initialize the recursive values with the rolling mean at index `period`
    for i in range(period, len(prices)):
        if i == period:
            # For the very first point, leave as is (already set by rolling)
            continue
        # Recursive calculation, starting after the first rolling value
        avg_gain.iat[i] = (avg_gain.iat[i - 1] * (period - 1) + gain.iat[i]) / period
        avg_loss.iat[i] = (avg_loss.iat[i - 1] * (period - 1) + loss.iat[i]) / period

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Cache the result
    indicator_cache.set(cache_key, rsi)

    return rsi
