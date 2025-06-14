import numpy as np
import pandas as pd

from app.backtesting.cache.indicators_cache import indicator_cache
from app.utils.backtesting_utils.indicators_utils import hash_series


def calculate_atr(df, period=14):
    # Create a hashable key for the cache
    df_hash = hash_series(df['high']) + hash_series(df['low']) + hash_series(df['close'])

    # Check if we have this calculation cached in the global cache
    cache_key = ('atr', df_hash, period)
    if indicator_cache.contains(cache_key):
        return indicator_cache.get(cache_key)

    # Calculate ATR
    high = df['high']
    low = df['low']
    close = df['close']

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = true_range.ewm(span=period, adjust=False).mean()
    atr[:period - 1] = np.nan  # First period points are undefined

    # Cache the result
    indicator_cache.set(cache_key, atr)

    return atr
