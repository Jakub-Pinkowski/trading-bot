import hashlib

import numpy as np
import pandas as pd

from app.backtesting.cache.indicators_cache import indicator_cache
from app.utils.logger import get_logger

# Initialize logger
logger = get_logger('indicators')


# Helper function to create a hashable key for pandas Series
def _hash_series(series):
    # Convert to string and hash
    series_str = str(series.values.tobytes())
    return hashlib.md5(series_str.encode()).hexdigest()


# RSI
def calculate_rsi(prices, period=14):
    # Validate period
    if period <= 0:
        logger.error(f"Invalid period value for RSI: {period}. Period must be a positive integer.")
        raise ValueError("Period must be a positive integer")

    # Create a hashable key for the cache
    prices_hash = _hash_series(prices)

    # Check if we have this calculation cached in the global cache
    cache_key = (prices_hash, period)
    if indicator_cache.contains(cache_key):
        return indicator_cache.get(cache_key)

    # Calculate RSI
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Initialize the recursive values with the rolling mean at index `period`
    for i in range(period, len(prices)):
        if i == period:
            # For the very first point, leave as is (already set by rolling)
            continue
        # Recursive calculation, starting after the first rolling value
        avg_gain.iat[i] = (avg_gain.iat[i - 1] * (period - 1) + gain.iat[i]) / period
        avg_loss.iat[i] = (avg_loss.iat[i - 1] * (period - 1) + loss.iat[i]) / period

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan  # First period points are undefined

    # Cache the result
    indicator_cache.set(cache_key, rsi)

    return rsi


# EMA
def calculate_ema(prices, period=9):
    # Validate period
    if period <= 0:
        logger.error(f"Invalid period value for EMA: {period}. Period must be a positive integer.")
        raise ValueError("Period must be a positive integer")

    # Create a hashable key for the cache
    prices_hash = _hash_series(prices)

    # Check if we have this calculation cached in the global cache
    cache_key = ('ema', prices_hash, period)
    if indicator_cache.contains(cache_key):
        return indicator_cache.get(cache_key)

    # Calculate EMA
    ema = prices.ewm(span=period, adjust=False).mean()

    # Cache the result
    indicator_cache.set(cache_key, ema)

    return ema


# ATR
def calculate_atr(df, period=14):
    # Create a hashable key for the cache
    df_hash = _hash_series(df['high']) + _hash_series(df['low']) + _hash_series(df['close'])

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


# MACD
def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    # Create a hashable key for the cache
    prices_hash = _hash_series(prices)

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


# Bollinger Bands
def calculate_bollinger_bands(prices, period=20, num_std=2):
    # Create a hashable key for the cache
    prices_hash = _hash_series(prices)

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
