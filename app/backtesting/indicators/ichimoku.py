from app.backtesting.cache.indicators_cache import indicator_cache
from app.utils.backtesting_utils.indicators_utils import hash_series


def calculate_ichimoku(high, low, close, tenkan_period=9, kijun_period=26, senkou_span_b_period=52, displacement=26):
    # Create a hashable key for the cache
    high_hash = hash_series(high)
    low_hash = hash_series(low)
    close_hash = hash_series(close)

    # Check if we have this calculation cached in the global cache
    cache_key = (
        'ichimoku',
        high_hash,
        low_hash,
        close_hash,
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
