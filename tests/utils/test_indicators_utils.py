"""
Tests for Indicators Utils Module.

Tests hash_series function for different data types and edge cases.
"""
import hashlib
import numpy as np
import pandas as pd
import pytest

from app.utils.backtesting_utils.indicators_utils import hash_series


def test_hash_series_numeric():
    """Test hashing numeric pandas Series."""
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = hash_series(series)
    
    # Should return a valid hex digest
    assert isinstance(result, str)
    assert len(result) == 32  # MD5 hex digest length
    
    # Same data should produce same hash
    series2 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result2 = hash_series(series2)
    assert result == result2


def test_hash_series_integer():
    """Test hashing integer pandas Series."""
    series = pd.Series([1, 2, 3, 4, 5])
    result = hash_series(series)
    
    assert isinstance(result, str)
    assert len(result) == 32


def test_hash_series_arrow_backed():
    """Test hashing Arrow-backed pandas Series (fallback path)."""
    # Create a class that mimics Arrow-backed arrays
    class MockArrowValues:
        def tobytes(self):
            raise AttributeError("Arrow arrays don't have tobytes")
    
    class MockSeries:
        def __init__(self):
            self.values = MockArrowValues()
        
        def __array__(self, dtype=None, copy=None):
            return np.array([1, 2, 3], dtype=dtype if dtype else object)
    
    mock_series = MockSeries()
    result = hash_series(mock_series)
    
    # Should successfully compute hash using fallback path
    assert isinstance(result, str)
    assert len(result) == 32


def test_hash_series_different_data_different_hash():
    """Test that different data produces different hashes."""
    series1 = pd.Series([1.0, 2.0, 3.0])
    series2 = pd.Series([1.0, 2.0, 4.0])
    
    hash1 = hash_series(series1)
    hash2 = hash_series(series2)
    
    assert hash1 != hash2


def test_hash_series_empty():
    """Test hashing empty Series."""
    series = pd.Series([], dtype=float)
    result = hash_series(series)
    
    assert isinstance(result, str)
    assert len(result) == 32
