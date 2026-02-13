from app.backtesting.cache.indicators_cache import indicator_cache
from app.backtesting.indicators.indicators_utils import logger


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

    # ==================== Vectorized RSI Calculation ====================
    # Calculate price changes
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use EWM (Exponential Weighted Moving Average) for Wilder's smoothing
    # alpha = 1/period gives Wilder's smoothing, adjust=False for recursive formula
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # Calculate RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Cache the result
    indicator_cache.set(cache_key, rsi)

    return rsi
