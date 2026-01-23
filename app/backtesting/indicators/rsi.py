from app.backtesting.cache.indicators_cache import indicator_cache
from app.utils.backtesting_utils.indicators_utils import logger


def calculate_rsi(prices, period, prices_hash):
    """
    Calculate RSI (Relative Strength Index) indicator.

    Args:
        prices: pandas Series of prices
        period: RSI period (e.g., 14)
        prices_hash: Pre-computed hash of prices series (use hash_series() or
                    BaseStrategy._precompute_hashes())

    Returns:
        pandas Series with RSI values

    Example:
        # Pre-compute hash once
        hashes = strategy._precompute_hashes(df)

        # Pass to multiple indicators
        rsi = calculate_rsi(df['close'], period=14, prices_hash=hashes['close'])
        rsi_long = calculate_rsi(df['close'], period=21, prices_hash=hashes['close'])
    """
    # Validate period
    if period <= 0:
        logger.error(f"Invalid period value for RSI: {period}. Period must be a positive integer.")
        raise ValueError("Period must be a positive integer")

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
