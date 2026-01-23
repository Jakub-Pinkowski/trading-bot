import hashlib

import numpy as np
import pandas as pd

from app.backtesting.cache.indicators_cache import indicator_cache
from app.utils.backtesting_utils.indicators_utils import hash_series


def calculate_atr(df, period=14, high_hash=None, low_hash=None, close_hash=None):
    """
    Calculate ATR (Average True Range) indicator.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR period (default: 14)
        high_hash: Optional pre-computed hash of df['high']
        low_hash: Optional pre-computed hash of df['low']
        close_hash: Optional pre-computed hash of df['close']
                   Pass these to avoid redundant hashing when calling multiple indicators.

    Returns:
        pandas Series with ATR values

    Example:
        # Without pre-computed hashes (simple usage)
        atr = calculate_atr(df, period=14)

        # With pre-computed hashes (optimized for multiple indicators)
        high_hash = hash_series(df['high'])
        low_hash = hash_series(df['low'])
        close_hash = hash_series(df['close'])
        atr = calculate_atr(df, period=14, high_hash=high_hash,
                           low_hash=low_hash, close_hash=close_hash)
    """
    # Create a combined hash for all price series
    if high_hash is None:
        high_hash = hash_series(df['high'])
    if low_hash is None:
        low_hash = hash_series(df['low'])
    if close_hash is None:
        close_hash = hash_series(df['close'])

    combined_hash = hashlib.md5(
        f"{high_hash}{low_hash}{close_hash}".encode()
    ).hexdigest()

    # Check if we have this calculation cached in the global cache
    cache_key = ('atr', combined_hash, period)
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
