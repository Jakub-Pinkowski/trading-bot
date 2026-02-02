import hashlib

import pandas as pd

from app.backtesting.cache.indicators_cache import indicator_cache


def calculate_atr(df, period, high_hash, low_hash, close_hash):
    """
    Calculate ATR (Average True Range) indicator.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR period (e.g., 14)
        high_hash: Pre-computed hash of df['high']
        low_hash: Pre-computed hash of df['low']
        close_hash: Pre-computed hash of df['close']
                   Use BaseStrategy._precompute_hashes() to get all hashes at once.

    Returns:
        pandas Series with ATR values

    Example:
        # Pre-compute all hashes once
        hashes = strategy._precompute_hashes(df)

        # Pass to ATR
        atr = calculate_atr(df, period=14, high_hash=hashes['high'],
                           low_hash=hashes['low'], close_hash=hashes['close'])
    """
    # Create a combined hash for all price series
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

    # Use min_periods to naturally produce NaN for first period-1 values
    atr = true_range.ewm(span=period, min_periods=period, adjust=False).mean()

    # Cache the result
    indicator_cache.set(cache_key, atr)

    return atr
