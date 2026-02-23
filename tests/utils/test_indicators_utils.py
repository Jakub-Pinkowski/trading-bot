"""
Tests for Indicators Utils Module.

Tests cover:
- hash_series for numeric, integer, Arrow-backed, and empty Series
- Hash determinism and collision resistance
"""
import numpy as np
import pandas as pd

from app.backtesting.indicators.indicators_utils import hash_series


# ==================== Test Classes ====================

class TestHashSeries:
    """Test deterministic MD5 hashing of pandas Series."""

    def test_numeric_series_returns_md5_hex_digest(self):
        """Test numeric Series returns a 32-character hex digest."""
        result = hash_series(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))

        assert isinstance(result, str)
        assert len(result) == 32

    def test_same_data_produces_same_hash(self):
        """Test identical Series values produce the same hash."""
        hash1 = hash_series(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
        hash2 = hash_series(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))

        assert hash1 == hash2

    def test_integer_series_returns_hex_digest(self):
        """Test integer Series returns a 32-character hex digest."""
        result = hash_series(pd.Series([1, 2, 3, 4, 5]))

        assert isinstance(result, str)
        assert len(result) == 32

    def test_arrow_backed_series_uses_fallback_path(self):
        """Test Series whose .values lacks tobytes() falls back to __array__."""

        class MockArrowValues:
            def tobytes(self):
                raise AttributeError("Arrow arrays don't have tobytes")

        class MockSeries:
            def __init__(self):
                self.values = MockArrowValues()

            def __array__(self, *args, **kwargs):
                return np.array([1, 2, 3], dtype=object)

        result = hash_series(MockSeries())

        assert isinstance(result, str)
        assert len(result) == 32

    def test_different_data_produces_different_hash(self):
        """Test Series with different values produce distinct hashes."""
        hash1 = hash_series(pd.Series([1.0, 2.0, 3.0]))
        hash2 = hash_series(pd.Series([1.0, 2.0, 4.0]))

        assert hash1 != hash2

    def test_empty_series_returns_hex_digest(self):
        """Test empty Series returns a valid 32-character hex digest."""
        result = hash_series(pd.Series([], dtype=float))

        assert isinstance(result, str)
        assert len(result) == 32
