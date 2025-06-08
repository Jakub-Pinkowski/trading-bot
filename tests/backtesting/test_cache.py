import os
import pandas as pd
import numpy as np
import unittest
from unittest.mock import patch

from app.backtesting.cache.cache_base import Cache
from app.backtesting.cache.indicators_cache import indicator_cache
from app.backtesting.cache.dataframe_cache import dataframe_cache, get_preprocessed_dataframe


class TestCache(unittest.TestCase):
    """Test the caching functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a test cache
        self.test_cache = Cache("test", 1)
        self.test_cache.clear()

        # Clear the indicator and dataframe caches
        indicator_cache.clear()
        dataframe_cache.clear()

    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the test cache file
        if os.path.exists(self.test_cache.cache_file):
            os.remove(self.test_cache.cache_file)

    def test_base_cache(self):
        """Test the base Cache class."""
        # Test setting and getting values
        self.test_cache.set("key1", "value1")
        self.test_cache.set("key2", "value2")

        self.assertEqual(self.test_cache.get("key1"), "value1")
        self.assertEqual(self.test_cache.get("key2"), "value2")
        self.assertIsNone(self.test_cache.get("key3"))
        self.assertEqual(self.test_cache.get("key3", "default"), "default")

        # Test contains method
        self.assertTrue(self.test_cache.contains("key1"))
        self.assertFalse(self.test_cache.contains("key3"))

        # Test size method
        self.assertEqual(self.test_cache.size(), 2)

        # Test clear method
        self.test_cache.clear()
        self.assertEqual(self.test_cache.size(), 0)

    def test_save_and_load_cache(self):
        """Test saving and loading the cache."""
        # Set some values
        self.test_cache.set("key1", "value1")
        self.test_cache.set("key2", "value2")

        # Save the cache
        self.test_cache.save_cache()

        # Create a new cache instance that should load from the same file
        new_cache = Cache("test", 1)

        # Check that the values were loaded
        self.assertEqual(new_cache.get("key1"), "value1")
        self.assertEqual(new_cache.get("key2"), "value2")

    @patch('app.backtesting.cache.dataframe_cache.pd.read_parquet')
    def test_dataframe_cache(self, mock_read_parquet):
        """Test the dataframe cache."""
        # Mock the read_parquet function
        test_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        mock_read_parquet.return_value = test_df

        # First call should miss the cache and call read_parquet
        result1 = get_preprocessed_dataframe("test_file.parquet")
        self.assertTrue(mock_read_parquet.called)
        pd.testing.assert_frame_equal(result1, test_df)

        # Reset the mock
        mock_read_parquet.reset_mock()

        # Second call should hit the cache and not call read_parquet
        result2 = get_preprocessed_dataframe("test_file.parquet")
        self.assertFalse(mock_read_parquet.called)
        pd.testing.assert_frame_equal(result2, test_df)

        # Test save_cache
        dataframe_cache.save_cache()
        self.assertTrue(os.path.exists(os.path.join(os.path.dirname(self.test_cache.cache_file),
                                                    "dataframe_cache_v1.pkl")))

    def test_indicator_cache(self):
        """Test the indicator cache."""
        # Create a test series
        test_series = pd.Series([1, 2, 3, 4, 5])

        # Store in the cache
        cache_key = ("test", "series")
        indicator_cache.set(cache_key, test_series)

        # Check that it's in the cache
        self.assertTrue(indicator_cache.contains(cache_key))
        pd.testing.assert_series_equal(indicator_cache.get(cache_key), test_series)

        # Test save_cache
        indicator_cache.save_cache()
        self.assertTrue(os.path.exists(os.path.join(os.path.dirname(self.test_cache.cache_file),
                                                    "indicator_cache_v1.pkl")))


if __name__ == '__main__':
    unittest.main()
