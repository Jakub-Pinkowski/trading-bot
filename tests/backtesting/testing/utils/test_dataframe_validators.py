"""
Tests for dataframe_validators utility module.

Tests cover:
- DataFrame validation for required columns
- Numeric type validation
- NaN value handling
- DatetimeIndex validation
- Index sorting validation
- Duplicate timestamp handling
- Edge cases (empty, None, invalid dtypes)
"""
from datetime import datetime

import numpy as np
import pandas as pd

from app.backtesting.testing.utils.dataframe_validators import (
    validate_dataframe,
    MIN_ROWS_FOR_BACKTEST,
    MAX_NAN_PERCENTAGE_WARNING
)


# ==================== Valid DataFrame Tests ====================

class TestValidDataFrame:
    """Test validation of valid DataFrames."""

    def test_validate_dataframe_with_valid_data(self):
        """Test that valid DataFrame passes validation."""
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [102.0, 103.0, 104.0],
            'low': [99.0, 100.0, 101.0],
            'close': [101.0, 102.0, 103.0],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1h'))

        result = validate_dataframe(df, '/path/to/test.parquet')

        assert result is True

    def test_validate_dataframe_with_only_required_columns(self):
        """Test validation with only required columns (no volume)."""
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [102.0, 103.0, 104.0],
            'low': [99.0, 100.0, 101.0],
            'close': [101.0, 102.0, 103.0]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1h'))

        result = validate_dataframe(df, '/path/to/test.parquet')

        assert result is True

    def test_validate_dataframe_with_extra_columns(self):
        """Test that extra columns don't affect validation."""
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [102.0, 103.0, 104.0],
            'low': [99.0, 100.0, 101.0],
            'close': [101.0, 102.0, 103.0],
            'volume': [1000, 1100, 1200],
            'extra_column': ['a', 'b', 'c']
        }, index=pd.date_range('2024-01-01', periods=3, freq='1h'))

        result = validate_dataframe(df, '/path/to/test.parquet')

        assert result is True

    def test_validate_dataframe_with_large_dataset(self):
        """Test validation with large dataset."""
        dates = pd.date_range('2024-01-01', periods=1000, freq='1h')
        df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 1000),
            'high': np.random.uniform(110, 120, 1000),
            'low': np.random.uniform(90, 100, 1000),
            'close': np.random.uniform(100, 110, 1000)
        }, index=dates)

        result = validate_dataframe(df, '/path/to/test.parquet')

        assert result is True


# ==================== Missing/Invalid DataFrame Tests ====================

class TestInvalidDataFrame:
    """Test validation of invalid DataFrames."""

    def test_validate_dataframe_with_none(self):
        """Test that None DataFrame fails validation."""
        result = validate_dataframe(None, '/path/to/test.parquet')

        assert result is False

    def test_validate_dataframe_with_empty_dataframe(self):
        """Test that empty DataFrame fails validation."""
        df = pd.DataFrame()

        result = validate_dataframe(df, '/path/to/test.parquet')

        assert result is False

    def test_validate_dataframe_missing_open_column(self):
        """Test that DataFrame missing 'open' column fails."""
        df = pd.DataFrame({
            'high': [102.0, 103.0, 104.0],
            'low': [99.0, 100.0, 101.0],
            'close': [101.0, 102.0, 103.0]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1h'))

        result = validate_dataframe(df, '/path/to/test.parquet')

        assert result is False

    def test_validate_dataframe_missing_high_column(self):
        """Test that DataFrame missing 'high' column fails."""
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'low': [99.0, 100.0, 101.0],
            'close': [101.0, 102.0, 103.0]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1h'))

        result = validate_dataframe(df, '/path/to/test.parquet')

        assert result is False

    def test_validate_dataframe_missing_multiple_columns(self):
        """Test that DataFrame missing multiple required columns fails."""
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1h'))

        result = validate_dataframe(df, '/path/to/test.parquet')

        assert result is False


# ==================== Data Type Tests ====================

class TestDataTypeValidation:
    """Test data type validation."""

    def test_validate_dataframe_with_non_numeric_column(self):
        """Test that non-numeric OHLC column fails validation."""
        df = pd.DataFrame({
            'open': ['100', '101', '102'],  # Strings instead of numbers
            'high': [102.0, 103.0, 104.0],
            'low': [99.0, 100.0, 101.0],
            'close': [101.0, 102.0, 103.0]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1h'))

        result = validate_dataframe(df, '/path/to/test.parquet')

        assert result is False

    def test_validate_dataframe_with_integer_columns(self):
        """Test that integer OHLC columns are valid."""
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [102, 103, 104],
            'low': [99, 100, 101],
            'close': [101, 102, 103]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1h'))

        result = validate_dataframe(df, '/path/to/test.parquet')

        assert result is True

    def test_validate_dataframe_with_mixed_numeric_types(self):
        """Test that mixed int/float columns are valid."""
        df = pd.DataFrame({
            'open': [100, 101.5, 102],
            'high': [102.0, 103, 104.5],
            'low': [99, 100.0, 101],
            'close': [101.5, 102, 103.0]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1h'))

        result = validate_dataframe(df, '/path/to/test.parquet')

        assert result is True


# ==================== NaN Value Tests ====================

class TestNaNHandling:
    """Test NaN value handling."""

    def test_validate_dataframe_with_few_nan_values(self):
        """Test that DataFrame with few NaN values passes with warning."""
        df = pd.DataFrame({
            'open': [100.0, np.nan, 102.0, 103.0, 104.0],
            'high': [102.0, 103.0, 104.0, 105.0, 106.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [101.0, 102.0, 103.0, 104.0, 105.0]
        }, index=pd.date_range('2024-01-01', periods=5, freq='1h'))

        # 1 NaN out of 5 = 20% > MAX_NAN_PERCENTAGE_WARNING (10%)
        result = validate_dataframe(df, '/path/to/test.parquet')

        # Should still pass validation (just warns)
        assert result is True

    def test_validate_dataframe_with_many_nan_values(self):
        """Test that DataFrame with many NaN values still passes (with warning)."""
        df = pd.DataFrame({
            'open': [100.0, np.nan, np.nan, np.nan, 104.0],
            'high': [102.0, 103.0, 104.0, 105.0, 106.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [101.0, 102.0, 103.0, 104.0, 105.0]
        }, index=pd.date_range('2024-01-01', periods=5, freq='1h'))

        # 3 NaN out of 5 = 60% >> MAX_NAN_PERCENTAGE_WARNING
        result = validate_dataframe(df, '/path/to/test.parquet')

        # Should still pass (validation doesn't fail on NaN, just warns)
        assert result is True

    def test_validate_dataframe_with_no_nan_values(self):
        """Test that DataFrame with no NaN values passes cleanly."""
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [102.0, 103.0, 104.0],
            'low': [99.0, 100.0, 101.0],
            'close': [101.0, 102.0, 103.0]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1h'))

        result = validate_dataframe(df, '/path/to/test.parquet')

        assert result is True


# ==================== Index Validation Tests ====================

class TestIndexValidation:
    """Test index validation."""

    def test_validate_dataframe_with_non_datetime_index(self):
        """Test that DataFrame with non-DatetimeIndex fails."""
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [102.0, 103.0, 104.0],
            'low': [99.0, 100.0, 101.0],
            'close': [101.0, 102.0, 103.0]
        })  # Default integer index

        result = validate_dataframe(df, '/path/to/test.parquet')

        assert result is False

    def test_validate_dataframe_with_unsorted_index(self):
        """Test that DataFrame with unsorted index fails."""
        dates = [
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 12, 0),
            datetime(2024, 1, 1, 11, 0),  # Out of order
        ]
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [102.0, 103.0, 104.0],
            'low': [99.0, 100.0, 101.0],
            'close': [101.0, 102.0, 103.0]
        }, index=pd.DatetimeIndex(dates))

        result = validate_dataframe(df, '/path/to/test.parquet')

        assert result is False

    def test_validate_dataframe_with_sorted_index(self):
        """Test that DataFrame with properly sorted index passes."""
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [102.0, 103.0, 104.0],
            'low': [99.0, 100.0, 101.0],
            'close': [101.0, 102.0, 103.0]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1h'))

        result = validate_dataframe(df, '/path/to/test.parquet')

        assert result is True

    def test_validate_dataframe_with_duplicate_timestamps(self):
        """Test that DataFrame with duplicate timestamps passes with warning."""
        dates = pd.DatetimeIndex([
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 11, 0),
            datetime(2024, 1, 1, 11, 0),  # Duplicate
            datetime(2024, 1, 1, 12, 0),
        ])
        df = pd.DataFrame({
            'open': [100.0, 101.0, 101.5, 102.0],
            'high': [102.0, 103.0, 103.5, 104.0],
            'low': [99.0, 100.0, 100.5, 101.0],
            'close': [101.0, 102.0, 102.5, 103.0]
        }, index=dates)

        # Should pass (just warns about duplicates)
        result = validate_dataframe(df, '/path/to/test.parquet')

        assert result is True


# ==================== Constants Tests ====================

class TestConstants:
    """Test module constants."""

    def test_min_rows_for_backtest_constant(self):
        """Test that MIN_ROWS_FOR_BACKTEST has reasonable value."""
        assert MIN_ROWS_FOR_BACKTEST > 0
        assert MIN_ROWS_FOR_BACKTEST == 150

    def test_max_nan_percentage_warning_constant(self):
        """Test that MAX_NAN_PERCENTAGE_WARNING has reasonable value."""
        assert MAX_NAN_PERCENTAGE_WARNING > 0
        assert MAX_NAN_PERCENTAGE_WARNING <= 100
        assert MAX_NAN_PERCENTAGE_WARNING == 10.0


# ==================== Integration Tests ====================

class TestDataFrameValidatorIntegration:
    """Test dataframe validator integration scenarios."""

    def test_validate_realistic_ohlc_data(self):
        """Test validation with realistic OHLC data."""
        # Create realistic price data
        dates = pd.date_range('2024-01-01', periods=200, freq='1h')
        base_price = 100.0
        prices = base_price + np.cumsum(np.random.randn(200) * 0.5)

        df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices * 1.01,
            'volume': np.random.randint(1000, 10000, 200)
        }, index=dates)

        result = validate_dataframe(df, '/path/to/realistic.parquet')

        assert result is True

    def test_validate_multiple_dataframes_in_sequence(self):
        """Test validating multiple DataFrames sequentially."""
        dfs = []
        for i in range(5):
            df = pd.DataFrame({
                'open': [100.0 + i, 101.0 + i, 102.0 + i],
                'high': [102.0 + i, 103.0 + i, 104.0 + i],
                'low': [99.0 + i, 100.0 + i, 101.0 + i],
                'close': [101.0 + i, 102.0 + i, 103.0 + i]
            }, index=pd.date_range(f'2024-01-0{i + 1}', periods=3, freq='1h'))
            dfs.append(df)

        results = [validate_dataframe(df, f'/path/to/test_{i}.parquet') for i, df in enumerate(dfs)]

        # All should pass
        assert all(results)

    def test_validate_with_different_frequencies(self):
        """Test validation with different time frequencies."""
        frequencies = ['1min', '5min', '15min', '1h', '4h', '1D']

        for freq in frequencies:
            df = pd.DataFrame({
                'open': [100.0, 101.0, 102.0],
                'high': [102.0, 103.0, 104.0],
                'low': [99.0, 100.0, 101.0],
                'close': [101.0, 102.0, 103.0]
            }, index=pd.date_range('2024-01-01', periods=3, freq=freq))

            result = validate_dataframe(df, f'/path/to/test_{freq}.parquet')

            assert result is True

    def test_validate_edge_case_single_row(self):
        """Test validation with single row DataFrame."""
        df = pd.DataFrame({
            'open': [100.0],
            'high': [102.0],
            'low': [99.0],
            'close': [101.0]
        }, index=pd.date_range('2024-01-01', periods=1, freq='1h'))

        result = validate_dataframe(df, '/path/to/single_row.parquet')

        assert result is True

    def test_validate_minimum_backtest_rows(self):
        """Test DataFrame with exactly MIN_ROWS_FOR_BACKTEST rows."""
        df = pd.DataFrame({
            'open': np.random.uniform(100, 110, MIN_ROWS_FOR_BACKTEST),
            'high': np.random.uniform(110, 120, MIN_ROWS_FOR_BACKTEST),
            'low': np.random.uniform(90, 100, MIN_ROWS_FOR_BACKTEST),
            'close': np.random.uniform(100, 110, MIN_ROWS_FOR_BACKTEST)
        }, index=pd.date_range('2024-01-01', periods=MIN_ROWS_FOR_BACKTEST, freq='1h'))

        result = validate_dataframe(df, '/path/to/minimum_rows.parquet')

        assert result is True
