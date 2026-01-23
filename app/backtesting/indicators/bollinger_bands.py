import pandas as pd

from app.backtesting.cache.indicators_cache import indicator_cache
from app.utils.backtesting_utils.indicators_utils import hash_series


def calculate_bollinger_bands(prices, period=20, num_std=2, prices_hash=None):
    """
    Calculate Bollinger Bands indicator.

    Args:
        prices: pandas Series of prices
        period: Moving average period (default: 20)
        num_std: Number of standard deviations (default: 2)
        prices_hash: Optional pre-computed hash of prices series for cache optimization.
                    If None, will be computed. Pass this to avoid redundant hashing
                    when calling multiple indicators on the same data.

    Returns:
        Dictionary with keys: middle_band, upper_band, lower_band

    Example:
        # Without pre-computed hash (simple usage)
        bb = calculate_bollinger_bands(df['close'])

        # With pre-computed hash (optimized for multiple indicators)
        close_hash = hash_series(df['close'])
        bb = calculate_bollinger_bands(df['close'], prices_hash=close_hash)
        rsi = calculate_rsi(df['close'], prices_hash=close_hash)
    """
    # Create a hashable key for the cache
    if prices_hash is None:
        prices_hash = hash_series(prices)

    # Check if we have this calculation cached in the global cache
    cache_key = ('bb', prices_hash, period, num_std)
    if indicator_cache.contains(cache_key):
        return indicator_cache.get(cache_key)

    # Calculate a middle band (SMA)
    middle_band = prices.rolling(window=period).mean()

    # Calculate standard deviation
    std = prices.rolling(window=period).std(ddof=0)

    # Calculate upper and lower bands
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)

    # Create result DataFrame
    result = pd.DataFrame({
        'middle_band': middle_band,
        'upper_band': upper_band,
        'lower_band': lower_band
    })

    # Cache the result
    indicator_cache.set(cache_key, result)

    return result
