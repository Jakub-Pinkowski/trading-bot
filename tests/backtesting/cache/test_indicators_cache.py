import hashlib
from unittest.mock import patch

import pandas as pd
import pytest

from app.backtesting.cache.indicators_cache import indicator_cache
from app.backtesting.indicators import (
    calculate_rsi, calculate_ema, calculate_atr,
    calculate_macd, calculate_bollinger_bands,
)
from app.utils.backtesting_utils.indicators_utils import hash_series


def test_indicator_cache_instance():
    """Test that the indicator_cache is properly initialized."""
    assert indicator_cache.cache_name == "indicator"
    assert indicator_cache.max_size == 500
    assert indicator_cache.max_age == 2592000  # 30 days in seconds


def test_indicator_cache_operations():
    """Test basic operations on the indicator cache."""
    # Clear the cache to start with a clean state
    indicator_cache.clear()

    # Test that the cache is empty
    assert indicator_cache.size() == 0

    # Test setting and getting a value
    test_key = ("rsi", "ZC", 14)  # Example key for an RSI indicator for Corn futures
    test_value = [50.0, 55.0, 60.0]  # Example RSI values

    indicator_cache.set(test_key, test_value)
    assert indicator_cache.contains(test_key)
    assert indicator_cache.get(test_key) == test_value

    # Test cache size
    assert indicator_cache.size() == 1

    # Test getting a non-existent key
    non_existent_key = ("ema", "ZS", 20)  # Soybean futures
    assert indicator_cache.get(non_existent_key) is None
    assert indicator_cache.get(non_existent_key, "default") == "default"

    # Clear the cache again
    indicator_cache.clear()
    assert indicator_cache.size() == 0
    assert not indicator_cache.contains(test_key)


@patch('app.backtesting.cache.cache_base.Cache.save_cache')
def test_indicator_cache_save(mock_save_cache):
    """Test saving the indicator cache."""
    # Clear the cache to start with a clean state
    indicator_cache.clear()

    # Add some data to the cache
    test_key = ("bollinger_bands", "CL", (20, 2))  # Crude Oil futures
    test_value = {"upper": [105.0, 110.0], "middle": [100.0, 105.0], "lower": [95.0, 100.0]}

    indicator_cache.set(test_key, test_value)

    # Save the cache
    indicator_cache.save_cache()

    # Verify that save_cache was called
    mock_save_cache.assert_called_once()


def test_indicator_cache_load():
    """Test loading the indicator cache."""
    # Since indicator_cache is a singleton that's already initialized,
    # we'll test that it has the expected properties after loading
    assert indicator_cache.cache_name == "indicator"
    assert hasattr(indicator_cache, 'cache_data')
    assert isinstance(indicator_cache.cache_data, dict)


def test_indicator_cache_with_complex_keys():
    """Test the indicator cache with complex keys."""
    # Clear the cache to start with a clean state
    indicator_cache.clear()

    # Test with a complex key structure, but use a hashable type (tuple) for the key
    # Dictionary keys must be hashable, so we convert the complex structure to a tuple
    complex_key = (
        "macd",  # indicator
        "ZW",  # symbol (Wheat futures)
        (12, 26, 9),  # parameters (fast_period, slow_period, signal_period)
        ("2023-01-01", "2023-12-31")  # date_range
    )

    complex_value = {
        "macd_line": [1.0, 1.5, 2.0],
        "signal_line": [0.5, 1.0, 1.5],
        "histogram": [0.5, 0.5, 0.5]
    }

    indicator_cache.set(complex_key, complex_value)
    assert indicator_cache.contains(complex_key)
    assert indicator_cache.get(complex_key) == complex_value

    # Clear the cache again
    indicator_cache.clear()


def test_rsi_caching():
    """Test that RSI calculations are properly cached."""
    # Clear the cache to start with a clean state
    indicator_cache.clear()

    # Create sample price data for ZC (Corn futures)
    prices = pd.Series([500, 505, 510, 515, 520, 525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580])

    # Calculate RSI
    rsi1 = calculate_rsi(prices)

    # Get the cache key
    prices_hash = hash_series(prices)
    cache_key = ('rsi', prices_hash, 14)  # Default period is 14

    # Verify the result is in the cache
    assert indicator_cache.contains(cache_key)
    cached_rsi = indicator_cache.get(cache_key)
    pd.testing.assert_series_equal(rsi1, cached_rsi)

    # Calculate again and verify it uses the cached value
    with patch('pandas.Series.rolling') as mock_rolling:
        rsi2 = calculate_rsi(prices)
        # The rolling function should not be called again
        mock_rolling.assert_not_called()

    # Verify both results are the same
    pd.testing.assert_series_equal(rsi1, rsi2)


def test_ema_caching():
    """Test that EMA calculations are properly cached."""
    # Clear the cache to start with a clean state
    indicator_cache.clear()

    # Create sample price data for ZS (Soybean futures)
    prices = pd.Series([1200, 1205, 1210, 1215, 1220, 1225, 1230, 1235, 1240, 1245, 1250])

    # Calculate EMA
    ema1 = calculate_ema(prices, period=5)

    # Get the cache key
    prices_hash = hash_series(prices)
    cache_key = ('ema', prices_hash, 5)

    # Verify the result is in the cache
    assert indicator_cache.contains(cache_key)
    cached_ema = indicator_cache.get(cache_key)
    pd.testing.assert_series_equal(ema1, cached_ema)

    # Calculate again and verify it uses the cached value
    with patch('pandas.Series.ewm') as mock_ewm:
        ema2 = calculate_ema(prices, period=5)
        # The ewm function should not be called again
        mock_ewm.assert_not_called()

    # Verify both results are the same
    pd.testing.assert_series_equal(ema1, ema2)


def test_atr_caching():
    """Test that ATR calculations are properly cached."""
    # Clear the cache to start with a clean state
    indicator_cache.clear()

    # Create sample OHLC data for CL (Crude Oil futures)
    df = pd.DataFrame({
        'open': [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
        'high': [72, 73, 74, 75, 76, 77, 78, 79, 80, 81],
        'low': [69, 70, 71, 72, 73, 74, 75, 76, 77, 78],
        'close': [71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
    })

    # Calculate ATR
    atr1 = calculate_atr(df, period=7)

    # Get the cache key
    combined_hash = hashlib.md5(
        f"{hash_series(df['high'])}{hash_series(df['low'])}{hash_series(df['close'])}".encode()
    ).hexdigest()
    cache_key = ('atr', combined_hash, 7)

    # Verify the result is in the cache
    assert indicator_cache.contains(cache_key)
    cached_atr = indicator_cache.get(cache_key)
    pd.testing.assert_series_equal(atr1, cached_atr)

    # Calculate again and verify it uses the cached value
    with patch('pandas.Series.ewm') as mock_ewm:
        atr2 = calculate_atr(df, period=7)
        # The ewm function should not be called again
        mock_ewm.assert_not_called()

    # Verify both results are the same
    pd.testing.assert_series_equal(atr1, atr2)


def test_macd_caching():
    """Test that MACD calculations are properly cached."""
    # Clear the cache to start with a clean state
    indicator_cache.clear()

    # Create sample price data for GC (Gold futures)
    prices = pd.Series([
        1800, 1805, 1810, 1815, 1820, 1825, 1830, 1835, 1840, 1845,
        1850, 1855, 1860, 1865, 1870, 1875, 1880, 1885, 1890, 1895,
        1900, 1905, 1910, 1915, 1920, 1925, 1930, 1935, 1940, 1945
    ])

    # Calculate MACD
    macd1 = calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9)

    # Get the cache key
    prices_hash = hash_series(prices)
    cache_key = ('macd', prices_hash, 12, 26, 9)

    # Verify the result is in the cache
    assert indicator_cache.contains(cache_key)
    cached_macd = indicator_cache.get(cache_key)
    pd.testing.assert_frame_equal(macd1, cached_macd)

    # Calculate again and verify it uses the cached value
    with patch('pandas.Series.ewm') as mock_ewm:
        macd2 = calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9)
        # The ewm function should not be called again
        mock_ewm.assert_not_called()

    # Verify both results are the same
    pd.testing.assert_frame_equal(macd1, macd2)


def test_bollinger_bands_caching():
    """Test that Bollinger Bands calculations are properly cached."""
    # Clear the cache to start with a clean state
    indicator_cache.clear()

    # Create sample price data for SI (Silver futures)
    prices = pd.Series([
        22, 22.1, 22.2, 22.3, 22.4, 22.5, 22.6, 22.7, 22.8, 22.9,
        23, 23.1, 23.2, 23.3, 23.4, 23.5, 23.6, 23.7, 23.8, 23.9,
        24, 24.1, 24.2, 24.3, 24.4
    ])

    # Calculate Bollinger Bands
    bb1 = calculate_bollinger_bands(prices, period=20, num_std=2)

    # Get the cache key
    prices_hash = hash_series(prices)
    cache_key = ('bb', prices_hash, 20, 2)

    # Verify the result is in the cache
    assert indicator_cache.contains(cache_key)
    cached_bb = indicator_cache.get(cache_key)
    pd.testing.assert_frame_equal(bb1, cached_bb)

    # Calculate again and verify it uses the cached value
    with patch('pandas.Series.rolling') as mock_rolling:
        bb2 = calculate_bollinger_bands(prices, period=20, num_std=2)
        # The rolling function should not be called again
        mock_rolling.assert_not_called()

    # Verify both results are the same
    pd.testing.assert_frame_equal(bb1, bb2)


def test_cache_persistence_across_calculations():
    """Test that the cache persists across multiple calculations with different parameters."""
    # Clear the cache to start with a clean state
    indicator_cache.clear()

    # Create sample price data for HG (Copper futures)
    prices = pd.Series([
        3.5, 3.51, 3.52, 3.53, 3.54, 3.55, 3.56, 3.57, 3.58, 3.59,
        3.6, 3.61, 3.62, 3.63, 3.64, 3.65, 3.66, 3.67, 3.68, 3.69
    ])

    # Calculate RSI with different periods
    rsi_14 = calculate_rsi(prices, period=14)
    rsi_7 = calculate_rsi(prices, period=7)

    # Calculate EMA with different periods
    ema_9 = calculate_ema(prices, period=9)
    ema_21 = calculate_ema(prices, period=21)

    # Verify all calculations are in the cache
    prices_hash = hash_series(prices)

    assert indicator_cache.contains(('rsi', prices_hash, 14))  # RSI 14
    assert indicator_cache.contains(('rsi', prices_hash, 7))  # RSI 7
    assert indicator_cache.contains(('ema', prices_hash, 9))  # EMA 9
    assert indicator_cache.contains(('ema', prices_hash, 21))  # EMA 21

    # Verify the cache size
    assert indicator_cache.size() == 4

    # Calculate again and verify it uses the cached values
    with patch('pandas.Series.rolling') as mock_rolling:
        with patch('pandas.Series.ewm') as mock_ewm:
            rsi_14_again = calculate_rsi(prices, period=14)
            rsi_7_again = calculate_rsi(prices, period=7)
            ema_9_again = calculate_ema(prices, period=9)
            ema_21_again = calculate_ema(prices, period=21)

            # The rolling and ewm functions should not be called again
            mock_rolling.assert_not_called()
            mock_ewm.assert_not_called()

    # Verify all results are the same
    pd.testing.assert_series_equal(rsi_14, rsi_14_again)
    pd.testing.assert_series_equal(rsi_7, rsi_7_again)
    pd.testing.assert_series_equal(ema_9, ema_9_again)
    pd.testing.assert_series_equal(ema_21, ema_21_again)


def test_integration_with_real_calculations():
    """Test integration with real indicator calculations."""
    # Clear the cache to start with a clean state
    indicator_cache.clear()

    # Create realistic price data for ES (E-mini S&P 500 futures)
    prices = pd.Series([
        4200, 4210, 4215, 4205, 4195, 4190, 4200, 4220, 4230, 4225,
        4215, 4225, 4235, 4245, 4260, 4250, 4240, 4245, 4255, 4270,
        4280, 4275, 4265, 4260, 4270, 4280, 4290, 4300, 4295, 4285
    ])

    # Calculate multiple indicators
    rsi = calculate_rsi(prices)
    ema = calculate_ema(prices)
    macd = calculate_macd(prices)
    bb = calculate_bollinger_bands(prices)

    # Verify all calculations are in the cache
    prices_hash = hash_series(prices)

    assert indicator_cache.contains(('rsi', prices_hash, 14))  # RSI
    assert indicator_cache.contains(('ema', prices_hash, 9))  # EMA
    assert indicator_cache.contains(('macd', prices_hash, 12, 26, 9))  # MACD
    assert indicator_cache.contains(('bb', prices_hash, 20, 2))  # Bollinger Bands

    # Save the cache
    with patch('app.backtesting.cache.cache_base.Cache.save_cache') as mock_save_cache:
        indicator_cache.save_cache()
        mock_save_cache.assert_called_once()

    # Calculate again and verify it uses the cached values
    with patch('pandas.Series.rolling') as mock_rolling:
        with patch('pandas.Series.ewm') as mock_ewm:
            rsi_again = calculate_rsi(prices)
            ema_again = calculate_ema(prices)
            macd_again = calculate_macd(prices)
            bb_again = calculate_bollinger_bands(prices)

            # The rolling and ewm functions should not be called again
            mock_rolling.assert_not_called()
            mock_ewm.assert_not_called()

    # Verify all results are the same
    pd.testing.assert_series_equal(rsi, rsi_again)
    pd.testing.assert_series_equal(ema, ema_again)
    pd.testing.assert_frame_equal(macd, macd_again)
    pd.testing.assert_frame_equal(bb, bb_again)



def test_indicator_cache_corrupted_file():
    """Test handling of corrupted cache files."""
    # Clear the cache to start with a clean state
    indicator_cache.clear()

    # Create sample price data
    prices = pd.Series([100, 101, 102, 103, 104, 105])

    # Calculate an indicator to add to the cache
    rsi = calculate_rsi(prices)

    # Get the cache key
    prices_hash = hash_series(prices)
    cache_key = ('rsi', prices_hash, 14)  # Default period is 14

    # Verify the result is in the cache
    assert indicator_cache.contains(cache_key)

    # Save the cache
    indicator_cache.save_cache()

    # Simulate a corrupted cache file by mocking pickle.load to raise an exception
    with patch('pickle.load', side_effect=Exception("Corrupted file")):
        # Create a new cache instance, which will try to load the corrupted file
        from app.backtesting.cache.cache_base import Cache
        new_cache = Cache("indicator")

        # Verify the new cache is empty (couldn't load the corrupted data)
        assert not new_cache.contains(cache_key)
        assert new_cache.size() == 0


def test_indicator_cache_interrupted_save():
    """Test handling of interrupted save operations."""
    # Clear the cache to start with a clean state
    indicator_cache.clear()

    # Create sample price data
    prices = pd.Series([100, 101, 102, 103, 104, 105])

    # Calculate an indicator to add to the cache
    rsi = calculate_rsi(prices)

    # Get the cache key
    prices_hash = hash_series(prices)
    cache_key = ('rsi', prices_hash, 14)  # Default period is 14

    # Verify the result is in the cache
    assert indicator_cache.contains(cache_key)
    cached_rsi = indicator_cache.get(cache_key)

    # Simulate an interrupted save by mocking open to raise an exception
    with patch('builtins.open', side_effect=Exception("Interrupted save")):
        # Try to save the cache, which should handle the exception gracefully
        indicator_cache.save_cache()

    # Verify the cache still contains the data in memory
    assert indicator_cache.contains(cache_key)
    pd.testing.assert_series_equal(indicator_cache.get(cache_key), cached_rsi)


def test_indicator_cache_large_objects():
    """Test handling of large objects in the cache to check for memory leaks."""
    # Clear the cache to start with a clean state
    indicator_cache.clear()

    # Create a large price series (100,000 points)
    large_prices = pd.Series(range(100_000))

    # Calculate RSI on the large series
    large_rsi = calculate_rsi(large_prices)

    # Get the cache key
    prices_hash = hash_series(large_prices)
    cache_key = ('rsi', prices_hash, 14)  # Default period is 14

    # Verify the result is in the cache
    assert indicator_cache.contains(cache_key)
    cached_rsi = indicator_cache.get(cache_key)
    pd.testing.assert_series_equal(cached_rsi, large_rsi)

    # Clear the reference to the original series to allow garbage collection
    del large_prices
    del large_rsi

    # Clear the cache to free memory
    indicator_cache.clear()
    assert not indicator_cache.contains(cache_key)
    assert indicator_cache.size() == 0


def test_indicator_cache_invalid_keys():
    """Test handling of invalid keys in the cache."""
    # Clear the cache to start with a clean state
    indicator_cache.clear()

    # Try to set a non-hashable key (dictionary is not hashable)
    with pytest.raises(TypeError):
        indicator_cache.set({'non_hashable': 'key'}, 'value')

    # Try to set a None key
    indicator_cache.set(None, 'value')
    assert indicator_cache.contains(None)
    assert indicator_cache.get(None) == 'value'

    # Try to set an empty string key
    indicator_cache.set('', 'value')
    assert indicator_cache.contains('')
    assert indicator_cache.get('') == 'value'

    # Verify the cache size
    assert indicator_cache.size() == 2  # None and empty string keys

    # Clear the cache
    indicator_cache.clear()
    assert indicator_cache.size() == 0


def test_indicator_cache_concurrent_access():
    """Test handling of concurrent access to the cache."""
    import threading

    # Clear the cache to start with a clean state
    indicator_cache.clear()

    # Create a function that adds items to the cache
    def add_to_cache(start_idx, end_idx):
        for i in range(start_idx, end_idx):
            prices = pd.Series(range(i, i + 10))
            indicator_cache.set(f'key_{i}', calculate_rsi(prices))

    # Create threads to add items to the cache concurrently
    threads = []
    for i in range(5):
        t = threading.Thread(target=add_to_cache, args=(i * 10, (i + 1) * 10))
        threads.append(t)

    # Start all threads
    for t in threads:
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    # Verify that all items were added to the cache
    assert indicator_cache.size() >= 50  # At least 50 items (might be more due to RSI calculations)

    # Verify that we can retrieve some items
    for i in range(0, 50, 10):
        assert indicator_cache.contains(f'key_{i}')

    # Clear the cache
    indicator_cache.clear()
    assert indicator_cache.size() == 0
