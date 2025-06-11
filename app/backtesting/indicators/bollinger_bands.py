import numpy as np
import pandas as pd

from app.backtesting.cache.indicators_cache import indicator_cache
from app.utils.backtesting_utils.indicators_utils import hash_series


# Bollinger Bands
def calculate_bollinger_bands(prices, period=20, num_std=2):
    # Create a hashable key for the cache
    prices_hash = hash_series(prices)

    # Check if we have this calculation cached in the global cache
    cache_key = ('bb', prices_hash, period, num_std)
    if indicator_cache.contains(cache_key):
        return indicator_cache.get(cache_key)

    # Calculate Bollinger Bands
    # Calculate a middle band (SMA)
    middle_band = prices.rolling(window=period).mean()

    # Calculate standard deviation
    std = prices.rolling(window=period).std()

    # Calculate upper and lower bands
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)

    # Create result DataFrame
    result = pd.DataFrame({
        'middle_band': middle_band,
        'upper_band': upper_band,
        'lower_band': lower_band
    })

    # The first points are undefined
    result.iloc[:period - 1] = np.nan

    # Cache the result
    indicator_cache.set(cache_key, result)

    return result
