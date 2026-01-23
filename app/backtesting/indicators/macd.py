import numpy as np
import pandas as pd

from app.backtesting.cache.indicators_cache import indicator_cache
from app.utils.backtesting_utils.indicators_utils import hash_series


def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9, prices_hash=None):
    """
    Calculate MACD (Moving Average Convergence Divergence) indicator.

    Args:
        prices: pandas Series of prices
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)
        prices_hash: Optional pre-computed hash of prices series for cache optimization.
                    If None, will be computed. Pass this to avoid redundant hashing
                    when calling multiple indicators on the same data.

    Returns:
        DataFrame with columns: macd_line, signal_line, histogram

    Example:
        # Without pre-computed hash (simple usage)
        macd = calculate_macd(df['close'])

        # With pre-computed hash (optimized for multiple indicators)
        close_hash = hash_series(df['close'])
        macd = calculate_macd(df['close'], prices_hash=close_hash)
        rsi = calculate_rsi(df['close'], prices_hash=close_hash)
    """
    # Create a hashable key for the cache
    if prices_hash is None:
        prices_hash = hash_series(prices)

    # Check if we have this calculation cached in the global cache
    cache_key = ('macd', prices_hash, fast_period, slow_period, signal_period)
    if indicator_cache.contains(cache_key):
        return indicator_cache.get(cache_key)

    # Calculate fast and slow EMAs
    fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = prices.ewm(span=slow_period, adjust=False).mean()

    # Calculate MACD line
    macd_line = fast_ema - slow_ema

    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # Calculate histogram
    histogram = macd_line - signal_line

    # Create result DataFrame
    result = pd.DataFrame({
        'macd_line': macd_line,
        'signal_line': signal_line,
        'histogram': histogram
    })

    # The first points are undefined
    result.iloc[:slow_period - 1] = np.nan

    # Cache the result
    indicator_cache.set(cache_key, result)

    return result
