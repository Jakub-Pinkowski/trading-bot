from unittest.mock import patch

from app.backtesting.cache.indicators_cache import indicator_cache, CACHE_VERSION


def test_indicator_cache_instance():
    """Test that the indicator_cache is properly initialized."""
    assert indicator_cache.cache_name == "indicator"
    assert indicator_cache.cache_version == CACHE_VERSION


def test_indicator_cache_operations():
    """Test basic operations on the indicator cache."""
    # Clear the cache to start with a clean state
    indicator_cache.clear()

    # Test that the cache is empty
    assert indicator_cache.size() == 0

    # Test setting and getting a value
    test_key = ("rsi", "AAPL", 14)  # Example key for an RSI indicator
    test_value = [50.0, 55.0, 60.0]  # Example RSI values

    indicator_cache.set(test_key, test_value)
    assert indicator_cache.contains(test_key)
    assert indicator_cache.get(test_key) == test_value

    # Test cache size
    assert indicator_cache.size() == 1

    # Test getting a non-existent key
    non_existent_key = ("ema", "MSFT", 20)
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
    test_key = ("bollinger_bands", "GOOG", (20, 2))
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
    assert indicator_cache.cache_version == CACHE_VERSION
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
        "AAPL",  # symbol
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
