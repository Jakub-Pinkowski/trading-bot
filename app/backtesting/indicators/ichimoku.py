import hashlib

from app.backtesting.cache.indicators_cache import indicator_cache
from app.utils.backtesting_utils.indicators_utils import hash_series


def calculate_ichimoku(
    high, low, close, tenkan_period=9, kijun_period=26,
    senkou_span_b_period=52, displacement=26,
    high_hash=None, low_hash=None, close_hash=None
):
    """
    Calculate Ichimoku Cloud indicator.

    Args:
        high: pandas Series of high prices
        low: pandas Series of low prices
        close: pandas Series of close prices
        tenkan_period: Conversion line period (default: 9)
        kijun_period: Base line period (default: 26)
        senkou_span_b_period: Leading span B period (default: 52)
        displacement: Displacement for cloud (default: 26)
        high_hash: Optional pre-computed hash of high series
        low_hash: Optional pre-computed hash of low series
        close_hash: Optional pre-computed hash of close series
                   Pass these to avoid redundant hashing when calling multiple indicators.

    Returns:
        Dictionary with keys: tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

    Example:
        # Without pre-computed hashes (simple usage)
        ichimoku = calculate_ichimoku(df['high'], df['low'], df['close'])

        # With pre-computed hashes (optimized for multiple indicators)
        high_hash = hash_series(df['high'])
        low_hash = hash_series(df['low'])
        close_hash = hash_series(df['close'])
        ichimoku = calculate_ichimoku(df['high'], df['low'], df['close'],
                                      high_hash=high_hash, low_hash=low_hash,
                                      close_hash=close_hash)
    """
    # Create a combined hash for all price series
    if high_hash is None:
        high_hash = hash_series(high)
    if low_hash is None:
        low_hash = hash_series(low)
    if close_hash is None:
        close_hash = hash_series(close)

    combined_hash = hashlib.md5(
        f"{high_hash}{low_hash}{close_hash}".encode()
    ).hexdigest()

    # Check if we have this calculation cached in the global cache
    cache_key = (
        'ichimoku',
        combined_hash,
        tenkan_period,
        kijun_period,
        senkou_span_b_period,
        displacement
    )
    if indicator_cache.contains(cache_key):
        return indicator_cache.get(cache_key)

    # Calculate Tenkan-sen (Conversion Line): (the highest high + the lowest low)/2 for the past tenkan_period
    tenkan_sen = (high.rolling(window=tenkan_period).max() + low.rolling(window=tenkan_period).min()) / 2

    # Calculate Kijun-sen (baseline): (the highest high + the lowest low)/2 for the past kijun_period
    kijun_sen = (high.rolling(window=kijun_period).max() + low.rolling(window=kijun_period).min()) / 2

    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 displaced forward by displacement period
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)

    # Calculate Senkou Span B (Leading Span B): (the highest high + the lowest low)/2 for the past senkou_span_b_period, displaced forward by displacement period
    senkou_span_b = ((
                             high.rolling(window=senkou_span_b_period).max() + low.rolling(window=senkou_span_b_period).min()) / 2).shift(
        displacement)

    # Calculate Chikou Span (Lagging Span): Close price displaced backwards by displacement period
    chikou_span = close.shift(-displacement)

    # Create a result dictionary
    result = {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    }

    # Cache the result
    indicator_cache.set(cache_key, result)

    return result
