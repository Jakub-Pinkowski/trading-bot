import hashlib

from app.backtesting.cache.indicators_cache import indicator_cache


def calculate_ichimoku(
    high, low, close, tenkan_period, kijun_period,
    senkou_span_b_period, displacement,
    high_hash, low_hash, close_hash
):
    """
    Calculate Ichimoku Cloud indicator.

    Args:
        high: pandas Series of high prices
        low: pandas Series of low prices
        close: pandas Series of close prices
        tenkan_period: Conversion line period (e.g., 9)
        kijun_period: Base line period (e.g., 26)
        senkou_span_b_period: Leading span B period (e.g., 52)
        displacement: Displacement for cloud (e.g., 26)
        high_hash: Pre-computed hash of high series
        low_hash: Pre-computed hash of low series
        close_hash: Pre-computed hash of close series
                   Use BaseStrategy._precompute_hashes() to get all hashes at once.

    Returns:
        Dictionary with keys: tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

    Example:
        # Pre-compute all hashes once
        hashes = strategy._precompute_hashes(df)

        # Pass to Ichimoku
        ichimoku = calculate_ichimoku(df['high'], df['low'], df['close'],
                                      tenkan_period=9, kijun_period=26,
                                      senkou_span_b_period=52, displacement=26,
                                      high_hash=hashes['high'], low_hash=hashes['low'],
                                      close_hash=hashes['close'])
    """
    # Create a combined hash for all price series
    combined_hash = hashlib.md5(
        f"{high_hash}{low_hash}{close_hash}".encode()
    ).hexdigest()

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
