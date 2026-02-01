import pandas as pd

from app.backtesting.cache.indicators_cache import indicator_cache


def calculate_bollinger_bands(prices, period, number_of_standard_deviations, prices_hash):
    """
    Calculate Bollinger Bands indicator.

    Args:
        prices: pandas Series of prices
        period: Moving average period (e.g., 20)
        number_of_standard_deviations: Number of standard deviations (e.g., 2)
        prices_hash: Pre-computed hash of prices series (use hash_series() or
                    BaseStrategy._precompute_hashes())

    Returns:
        Dictionary with keys: middle_band, upper_band, lower_band

    Example:
        # Pre-compute hash once
        hashes = strategy._precompute_hashes(df)

        # Pass to indicator
        bb = calculate_bollinger_bands(df['close'], period=20, number_of_standard_deviations=2,
                                      prices_hash=hashes['close'])
    """

    # Check if we have this calculation cached in the global cache
    cache_key = ('bb', prices_hash, period, number_of_standard_deviations)
    if indicator_cache.contains(cache_key):
        return indicator_cache.get(cache_key)

    # Calculate a middle band (SMA)
    middle_band = prices.rolling(window=period).mean()

    # Calculate standard deviation
    std = prices.rolling(window=period).std(ddof=0)

    # Calculate upper and lower bands
    upper_band = middle_band + (std * number_of_standard_deviations)
    lower_band = middle_band - (std * number_of_standard_deviations)

    # Create result DataFrame
    result = pd.DataFrame({
        'middle_band': middle_band,
        'upper_band': upper_band,
        'lower_band': lower_band
    })

    # Cache the result
    indicator_cache.set(cache_key, result)

    return result
