import pandas as pd

from app.backtesting.cache.indicators_cache import indicator_cache


def calculate_macd(prices, fast_period, slow_period, signal_period, prices_hash):
    """
    Calculate MACD (Moving Average Convergence Divergence) indicator.

    Args:
        prices: pandas Series of prices
        fast_period: Fast EMA period (e.g., 12)
        slow_period: Slow EMA period (e.g., 26)
        signal_period: Signal line period (e.g., 9)
        prices_hash: Pre-computed hash of prices series (use hash_series() or
                    BaseStrategy._precompute_hashes())

    Returns:
        DataFrame with columns: macd_line, signal_line, histogram

    Example:
        # Pre-compute hash once
        hashes = strategy._precompute_hashes(df)

        # Pass to indicator
        macd = calculate_macd(df['close'], fast_period=12, slow_period=26,
                            signal_period=9, prices_hash=hashes['close'])
    """

    # Check if we have this calculation cached in the global cache
    cache_key = ('macd', prices_hash, fast_period, slow_period, signal_period)
    if indicator_cache.contains(cache_key):
        return indicator_cache.get(cache_key)

    # Calculate fast and slow EMAs with min_periods to naturally produce NaN
    fast_ema = prices.ewm(span=fast_period, min_periods=fast_period, adjust=False).mean()
    slow_ema = prices.ewm(span=slow_period, min_periods=slow_period, adjust=False).mean()

    # Calculate MACD line
    macd_line = fast_ema - slow_ema

    # Calculate signal line (no min_periods - starts when macd_line has values)
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # Calculate histogram
    histogram = macd_line - signal_line

    # Create result DataFrame
    result = pd.DataFrame({
        'macd_line': macd_line,
        'signal_line': signal_line,
        'histogram': histogram
    })


    # Cache the result
    indicator_cache.set(cache_key, result)

    return result
