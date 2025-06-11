import numpy as np
import pandas as pd

from app.backtesting.cache.indicators_cache import indicator_cache
from app.utils.backtesting_utils.indicators_utils import hash_series


# MACD
def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    # Create a hashable key for the cache
    prices_hash = hash_series(prices)

    # Check if we have this calculation cached in the global cache
    cache_key = ('macd', prices_hash, fast_period, slow_period, signal_period)
    if indicator_cache.contains(cache_key):
        return indicator_cache.get(cache_key)

    # Calculate MACD
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
