"""
Tests for Data Fetching Validators Module.

Tests cover:
- Symbol validation for TradingView compatibility
- Exchange compatibility validation
- OHLCV data structure validation
- Gap detection in time series data
- Edge cases and error handling
- Integration scenarios with real symbol configurations

All tests use realistic futures symbols and actual SYMBOL_SPECS configuration.
"""
from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from app.backtesting.fetching.validators import (
    validate_symbols,
    validate_exchange_compatibility,
    validate_ohlcv_data,
    detect_gaps
)


# ==================== Test Classes ====================

class TestValidateSymbols:
    """Test symbol validation for TradingView compatibility."""

    @pytest.mark.parametrize("symbols,expected_count", [
        (['ZS'], 1),
        (['ZS', 'ZC'], 2),
        (['ZS', 'ZC', 'CL'], 3),
        (['CL', 'NG'], 2),
        (['GC', 'SI'], 2),
        (['YM', 'ZB'], 2),
    ])
    def test_valid_symbols_parametrized(self, symbols, expected_count):
        """Test validation with various valid symbol combinations."""
        result = validate_symbols(symbols)
        assert len(result) == expected_count
        assert all(s in result for s in symbols)

    def test_all_valid_symbols(self):
        """Test validation with all valid TradingView-compatible symbols."""
        symbols = ['ZS', 'ZC', 'CL', 'GC']
        result = validate_symbols(symbols)

        assert result == symbols
        assert len(result) == 4

    def test_mixed_valid_invalid_symbols(self):
        """Test validation filters out invalid symbols."""
        symbols = ['ZS', 'INVALID', 'CL', 'NOTREAL']

        with patch('app.backtesting.fetching.validators.logger') as mock_logger:
            result = validate_symbols(symbols)

            # Valid symbols returned
            assert 'ZS' in result
            assert 'CL' in result
            assert len(result) == 2

            # Warning logged for invalid symbols
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert 'INVALID' in warning_msg
            assert 'NOTREAL' in warning_msg

    def test_all_invalid_symbols_raises_error(self):
        """Test ValueError raised when no valid symbols provided."""
        symbols = ['INVALID1', 'INVALID2', 'NOTREAL']

        with pytest.raises(ValueError, match='No valid symbols provided'):
            validate_symbols(symbols)

    def test_empty_list_raises_error(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match='No valid symbols provided'):
            validate_symbols([])

    def test_tv_incompatible_symbols_filtered(self):
        """Test that symbols marked as tv_compatible=False are filtered."""
        # YC and QC are marked as tv_compatible=False in SYMBOL_SPECS
        symbols = ['ZS', 'YC', 'QC']

        with patch('app.backtesting.fetching.validators.logger') as mock_logger:
            result = validate_symbols(symbols)

            # Only ZS should be valid
            assert result == ['ZS']

            # Warning logged
            mock_logger.warning.assert_called_once()

    def test_single_valid_symbol(self):
        """Test validation with single valid symbol."""
        symbols = ['YM']
        result = validate_symbols(symbols)

        assert result == ['YM']

    def test_duplicate_symbols_preserved(self):
        """Test that duplicate symbols are preserved in output."""
        symbols = ['ZS', 'ZS', 'CL']
        result = validate_symbols(symbols)

        # Duplicates preserved
        assert len(result) == 3
        assert result.count('ZS') == 2


class TestValidateExchangeCompatibility:
    """Test exchange compatibility validation."""

    @pytest.mark.parametrize("exchange,symbols,expected_valid", [
        ('CBOT', ['ZS', 'ZC', 'ZW'], ['ZS', 'ZC', 'ZW']),
        ('NYMEX', ['CL', 'NG'], ['CL', 'NG']),
        ('CBOT', ['ZS', 'CL'], ['ZS']),  # CL is NYMEX
        ('NYMEX', ['CL', 'ZS'], ['CL']),  # ZS is CBOT
        ('CBOT', ['YM', 'ZB'], ['YM', 'ZB']),
    ])
    def test_exchange_compatibility_parametrized(self, exchange, symbols, expected_valid):
        """Test exchange compatibility with various combinations."""
        with patch('app.backtesting.fetching.validators.logger'):
            result = validate_exchange_compatibility(symbols, exchange)
            assert sorted(result) == sorted(expected_valid)

    def test_all_symbols_compatible_with_exchange(self):
        """Test validation when all symbols match the exchange."""
        # CBOT symbols
        symbols = ['ZS', 'ZC', 'ZW']
        exchange = 'CBOT'

        result = validate_exchange_compatibility(symbols, exchange)

        assert result == symbols
        assert len(result) == 3

    def test_mixed_compatible_incompatible_symbols(self):
        """Test validation filters symbols by exchange."""
        # Mix of CBOT and CME/NYMEX symbols
        symbols = ['ZS', 'YM', 'CL']  # ZS=CBOT, YM=CBOT, CL=NYMEX
        exchange = 'CBOT'

        with patch('app.backtesting.fetching.validators.logger') as mock_logger:
            result = validate_exchange_compatibility(symbols, exchange)

            # Only ZS and YM should be compatible with CBOT
            assert 'ZS' in result
            assert 'YM' in result
            assert len(result) == 2

            # Warning logged for incompatible symbols
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert 'CL' in warning_msg

    def test_no_compatible_symbols_raises_error(self):
        """Test ValueError raised when no symbols match exchange."""
        symbols = ['YM', 'ZB']  # Both CBOT symbols
        exchange = 'CME'

        with pytest.raises(ValueError, match='No symbols compatible with exchange'):
            validate_exchange_compatibility(symbols, exchange)

    def test_nymex_exchange_compatibility(self):
        """Test validation for NYMEX exchange."""
        symbols = ['CL', 'NG', 'ZS']  # CL, NG=NYMEX, ZS=CBOT
        exchange = 'NYMEX'

        with patch('app.backtesting.fetching.validators.logger'):
            result = validate_exchange_compatibility(symbols, exchange)

            # Only NYMEX symbols returned
            assert 'CL' in result
            assert 'NG' in result
            assert 'ZS' not in result

    def test_cme_exchange_compatibility(self):
        """Test validation for CME exchange."""
        symbols = ['YM', 'ZB', 'ZS']  # YM, ZB=CBOT, ZS=CBOT
        exchange = 'CBOT'

        with patch('app.backtesting.fetching.validators.logger'):
            result = validate_exchange_compatibility(symbols, exchange)

            assert 'YM' in result
            assert 'ZB' in result
            assert len(result) == 3  # All are CBOT

    def test_single_compatible_symbol(self):
        """Test with single compatible symbol."""
        symbols = ['ZS']
        exchange = 'CBOT'

        result = validate_exchange_compatibility(symbols, exchange)
        assert result == ['ZS']

    def test_empty_symbols_list_raises_error(self):
        """Test that empty symbol list raises ValueError."""
        symbols = []
        exchange = 'CBOT'

        with pytest.raises(ValueError, match='No symbols compatible'):
            validate_exchange_compatibility(symbols, exchange)


class TestValidateOHLCVData:
    """Test OHLCV data structure validation."""

    @pytest.mark.parametrize("column_case", [
        ['open', 'high', 'low', 'close', 'volume'],
        ['Open', 'High', 'Low', 'Close', 'Volume'],
        ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'],
        ['Open', 'high', 'Low', 'close', 'VOLUME'],  # Mixed case
    ])
    def test_column_names_case_insensitive_parametrized(self, column_case):
        """Test validation works with various column name cases."""
        data = pd.DataFrame({
            column_case[0]: [100.0, 101.0],
            column_case[1]: [105.0, 106.0],
            column_case[2]: [99.0, 100.0],
            column_case[3]: [103.0, 104.0],
            column_case[4]: [1000, 1100]
        })

        # Should not raise any exception
        validate_ohlcv_data(data, 'ZS', '1h')

    def test_valid_ohlcv_dataframe(self):
        """Test validation passes with valid OHLCV data."""
        data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [99.0, 100.0, 101.0],
            'close': [103.0, 104.0, 105.0],
            'volume': [1000, 1100, 1200]
        })

        # Should not raise any exception
        validate_ohlcv_data(data, 'ZS', '1h')

    def test_case_insensitive_column_names(self):
        """Test validation works with different column name cases."""
        data = pd.DataFrame({
            'Open': [100.0, 101.0],
            'High': [105.0, 106.0],
            'Low': [99.0, 100.0],
            'Close': [103.0, 104.0],
            'Volume': [1000, 1100]
        })

        # Should not raise any exception
        validate_ohlcv_data(data, 'ZS', '1h')

    def test_missing_column_raises_error(self):
        """Test ValueError raised when required column is missing."""
        data = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [105.0, 106.0],
            'low': [99.0, 100.0],
            'close': [103.0, 104.0]
            # Missing 'volume'
        })

        with pytest.raises(ValueError, match='Missing required columns'):
            validate_ohlcv_data(data, 'ZS', '1h')

    def test_multiple_missing_columns_raises_error(self):
        """Test error message includes all missing columns."""
        data = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [105.0, 106.0]
            # Missing: low, close, volume
        })

        with pytest.raises(ValueError, match='Missing required columns') as exc_info:
            validate_ohlcv_data(data, 'ZS', '1h')

        error_msg = str(exc_info.value)
        assert 'low' in error_msg
        assert 'close' in error_msg
        assert 'volume' in error_msg

    def test_none_data_raises_error(self):
        """Test ValueError raised when data is None."""
        with pytest.raises(ValueError, match='No data'):
            validate_ohlcv_data(None, 'ZS', '1h')

    def test_empty_dataframe_raises_error(self):
        """Test ValueError raised when DataFrame is empty."""
        data = pd.DataFrame()

        with pytest.raises(ValueError, match='No data'):
            validate_ohlcv_data(data, 'ZS', '1h')

    def test_non_numeric_column_raises_error(self):
        """Test ValueError raised when column is not numeric."""
        data = pd.DataFrame({
            'open': ['100', '101', '102'],  # Strings instead of numbers
            'high': [105.0, 106.0, 107.0],
            'low': [99.0, 100.0, 101.0],
            'close': [103.0, 104.0, 105.0],
            'volume': [1000, 1100, 1200]
        })

        with pytest.raises(ValueError, match='must be numeric'):
            validate_ohlcv_data(data, 'ZS', '1h')

    def test_extra_columns_allowed(self):
        """Test validation passes when DataFrame has extra columns."""
        data = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [105.0, 106.0],
            'low': [99.0, 100.0],
            'close': [103.0, 104.0],
            'volume': [1000, 1100],
            'extra_column': [1, 2],
            'another_extra': ['a', 'b']
        })

        # Should not raise any exception
        validate_ohlcv_data(data, 'ZS', '1h')

    def test_error_message_includes_symbol_and_interval(self):
        """Test error messages include symbol and interval for context."""
        data = pd.DataFrame({'open': [100.0]})

        with pytest.raises(ValueError) as exc_info:
            validate_ohlcv_data(data, 'TEST_SYMBOL', '4h')

        error_msg = str(exc_info.value)
        assert 'TEST_SYMBOL' in error_msg
        assert '4h' in error_msg


class TestDetectGaps:
    """Test gap detection in time series data."""

    def test_no_gaps_detected(self):
        """Test no warnings logged when data has no significant gaps."""
        # Create hourly data with no gaps
        dates = pd.date_range('2024-01-01', periods=24, freq='h')
        data = pd.DataFrame({
            'close': [100.0] * 24
        }, index=dates)

        with patch('app.backtesting.fetching.validators.logger') as mock_logger:
            detect_gaps(data, '1h', 'ZS1!', set())

            # No warnings should be logged
            mock_logger.warning.assert_not_called()

    def test_large_gap_detected(self):
        """Test warning logged for gap larger than threshold."""
        # Create data with a 7-day gap (larger than 4-day threshold)
        dates1 = pd.date_range('2024-01-01', periods=10, freq='h')
        dates2 = pd.date_range('2024-01-08', periods=10, freq='h')
        dates = dates1.append(dates2)

        data = pd.DataFrame({
            'close': [100.0] * 20
        }, index=dates)

        with patch('app.backtesting.fetching.validators.logger') as mock_logger:
            detect_gaps(data, '1h', 'ZS1!', set())

            # Gap should be logged
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert 'NEW data gap detected' in warning_msg
            assert 'ZS1!' in warning_msg
            assert '1h' in warning_msg

    def test_small_gap_not_logged(self):
        """Test gap smaller than threshold is not logged."""
        # Create data with a 2-day gap (smaller than 4-day threshold)
        dates1 = pd.date_range('2024-01-01', periods=10, freq='h')
        dates2 = pd.date_range('2024-01-03', periods=10, freq='h')
        dates = dates1.append(dates2)

        data = pd.DataFrame({
            'close': [100.0] * 20
        }, index=dates)

        with patch('app.backtesting.fetching.validators.logger') as mock_logger:
            detect_gaps(data, '1h', 'ZS1!', set())

            # No warning should be logged for small gaps
            mock_logger.warning.assert_not_called()

    def test_multiple_gaps_detected(self):
        """Test multiple gaps are all logged."""
        # Create data with multiple large gaps
        dates1 = pd.date_range('2024-01-01', periods=5, freq='h')
        dates2 = pd.date_range('2024-01-10', periods=5, freq='h')
        dates3 = pd.date_range('2024-01-20', periods=5, freq='h')
        dates = dates1.append(dates2).append(dates3)

        data = pd.DataFrame({
            'close': [100.0] * 15
        }, index=dates)

        with patch('app.backtesting.fetching.validators.logger') as mock_logger:
            detect_gaps(data, '1h', 'ZS1!', set())

            # Two gaps should be logged
            assert mock_logger.warning.call_count == 2

    def test_single_row_data(self):
        """Test no error with single row of data."""
        data = pd.DataFrame({
            'close': [100.0]
        }, index=[datetime(2024, 1, 1)])

        with patch('app.backtesting.fetching.validators.logger') as mock_logger:
            detect_gaps(data, '1h', 'ZS1!', set())

            # No warnings for single row
            mock_logger.warning.assert_not_called()

    def test_empty_dataframe(self):
        """Test no error with empty DataFrame."""
        data = pd.DataFrame()

        with patch('app.backtesting.fetching.validators.logger') as mock_logger:
            detect_gaps(data, '1h', 'ZS1!', set())

            # No warnings for empty data
            mock_logger.warning.assert_not_called()

    def test_unsorted_index_handled(self):
        """Test gap detection works with unsorted datetime index."""
        # Create unsorted dates
        dates = pd.DatetimeIndex([
            datetime(2024, 1, 3),
            datetime(2024, 1, 1),
            datetime(2024, 1, 15),  # Large gap
            datetime(2024, 1, 2)
        ])

        data = pd.DataFrame({
            'close': [100.0] * 4
        }, index=dates)

        with patch('app.backtesting.fetching.validators.logger') as mock_logger:
            detect_gaps(data, '1h', 'ZS1!', set())

            # Gap should still be detected after sorting
            mock_logger.warning.assert_called()

    def test_gap_at_threshold_boundary(self):
        """Test gap exactly at threshold is not logged."""
        # Create gap exactly equal to threshold (5 days)
        dates1 = pd.date_range('2024-01-01', periods=5, freq='h')
        dates2 = pd.date_range('2024-01-06', periods=5, freq='h')
        dates = dates1.append(dates2)

        data = pd.DataFrame({
            'close': [100.0] * 10
        }, index=dates)

        with patch('app.backtesting.fetching.validators.logger') as mock_logger:
            detect_gaps(data, '1h', 'ZS1!', set())

            # Gap at threshold should not be logged (only > threshold)
            mock_logger.warning.assert_not_called()

    def test_gap_slightly_above_threshold(self):
        """Test gap just above threshold is logged."""
        # Create gap just above threshold (>5 days)
        dates1 = pd.date_range('2024-01-01', periods=5, freq='h')
        dates2 = pd.date_range('2024-01-07', periods=5, freq='h')  # ~6 days gap
        dates = dates1.append(dates2)

        data = pd.DataFrame({
            'close': [100.0] * 10
        }, index=dates)

        with patch('app.backtesting.fetching.validators.logger') as mock_logger:
            detect_gaps(data, '1h', 'ZS1!', set())

            # Gap should be logged
            mock_logger.warning.assert_called_once()

    def test_warning_message_format(self):
        """Test warning message contains all required information."""
        # Create data with gap
        dates1 = pd.date_range('2024-01-01', periods=5, freq='h')
        dates2 = pd.date_range('2024-01-10', periods=5, freq='h')
        dates = dates1.append(dates2)

        data = pd.DataFrame({
            'close': [100.0] * 10
        }, index=dates)

        with patch('app.backtesting.fetching.validators.logger') as mock_logger:
            detect_gaps(data, '1h', 'CL1!', set())

            warning_msg = mock_logger.warning.call_args[0][0]
            assert 'CL1!' in warning_msg
            assert '1h' in warning_msg
            assert 'from' in warning_msg
            assert 'to' in warning_msg
            assert 'duration' in warning_msg


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple validators."""

    def test_full_validation_pipeline(self):
        """Test complete validation flow from symbols to data validation."""
        # Start with mixed symbols
        symbols = ['ZS', 'ZC', 'INVALID', 'ES']
        exchange = 'CBOT'

        # Step 1: Validate symbols
        with patch('app.backtesting.fetching.validators.logger'):
            valid_symbols = validate_symbols(symbols)
            assert 'ZS' in valid_symbols
            assert 'ZC' in valid_symbols
            assert 'INVALID' not in valid_symbols

        # Step 2: Validate exchange compatibility
        with patch('app.backtesting.fetching.validators.logger'):
            compatible_symbols = validate_exchange_compatibility(valid_symbols, exchange)
            assert 'ZS' in compatible_symbols
            assert 'ZC' in compatible_symbols
            assert 'ES' not in compatible_symbols  # CME symbol

        # Step 3: Validate OHLCV data
        data = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [105.0, 106.0],
            'low': [99.0, 100.0],
            'close': [103.0, 104.0],
            'volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2, freq='h'))

        # Should pass validation
        validate_ohlcv_data(data, compatible_symbols[0], '1h')

    def test_error_propagation(self):
        """Test that errors propagate correctly through validation chain."""
        # All invalid symbols
        symbols = ['INVALID1', 'INVALID2']

        # Should raise at first validation step
        with pytest.raises(ValueError, match='No valid symbols provided'):
            validate_symbols(symbols)

    def test_real_symbol_configurations(self):
        """Test with real symbol configurations from SYMBOL_SPECS."""
        # Grains (CBOT)
        grains = ['ZS', 'ZC', 'ZW']
        result = validate_symbols(grains)
        assert len(result) == 3

        result = validate_exchange_compatibility(result, 'CBOT')
        assert len(result) == 3

        # Energies (NYMEX)
        energies = ['CL', 'NG']
        result = validate_symbols(energies)
        assert len(result) == 2

        result = validate_exchange_compatibility(result, 'NYMEX')
        assert len(result) == 2

        # Index (CBOT - using tv_compatible symbols)
        indices = ['YM', 'ZB']
        result = validate_symbols(indices)
        assert len(result) == 2

        result = validate_exchange_compatibility(result, 'CBOT')
        assert len(result) == 2


class TestPerformanceAndStress:
    """Test performance with large datasets and edge cases."""

    def test_validate_large_symbol_list(self):
        """Test validation performance with large symbol list."""
        # Create large list with duplicates and invalid symbols
        symbols = ['ZS', 'ZC', 'CL', 'NG', 'GC'] * 100  # 500 symbols

        with patch('app.backtesting.fetching.validators.logger'):
            result = validate_symbols(symbols)

            # Should handle duplicates efficiently
            assert len(result) == 500
            assert result.count('ZS') == 100

    def test_validate_large_dataframe(self):
        """Test OHLCV validation with large dataset."""
        # Create large DataFrame (10,000 rows)
        dates = pd.date_range('2020-01-01', periods=10000, freq='h')
        data = pd.DataFrame({
            'open': range(10000),
            'high': range(10000),
            'low': range(10000),
            'close': range(10000),
            'volume': range(10000)
        }, index=dates)

        # Should validate quickly
        validate_ohlcv_data(data, 'ZS', '1h')

    def test_gap_detection_large_dataset(self):
        """Test gap detection performance with large dataset."""
        # Create large dataset with multiple gaps
        dates1 = pd.date_range('2020-01-01', periods=2000, freq='h')
        dates2 = pd.date_range('2020-05-01', periods=2000, freq='h')  # Large gap
        dates3 = pd.date_range('2020-09-01', periods=2000, freq='h')  # Another large gap
        dates = dates1.append(dates2).append(dates3)

        data = pd.DataFrame({
            'close': [100.0] * 6000
        }, index=dates)

        with patch('app.backtesting.fetching.validators.logger') as mock_logger:
            detect_gaps(data, '1h', 'ZS1!', set())

            # Should detect both gaps
            assert mock_logger.warning.call_count == 2

    def test_unsorted_large_dataset(self):
        """Test gap detection with large unsorted dataset."""
        # Create unsorted large dataset
        import random
        dates = pd.date_range('2020-01-01', periods=1000, freq='h')
        shuffled_dates = dates.to_list()
        random.shuffle(shuffled_dates)

        # Add a large gap
        extra_dates = pd.date_range('2020-03-01', periods=100, freq='h')
        all_dates = pd.DatetimeIndex(shuffled_dates + extra_dates.to_list())

        data = pd.DataFrame({
            'close': [100.0] * len(all_dates)
        }, index=all_dates)

        with patch('app.backtesting.fetching.validators.logger') as mock_logger:
            detect_gaps(data, '1h', 'ZS1!', set())

            # Should still detect gap despite unsorted data
            mock_logger.warning.assert_called()

    @pytest.mark.parametrize("num_rows", [10, 100, 1000, 5000])
    def test_validation_scales_with_data_size(self, num_rows):
        """Test validation scales efficiently with different data sizes."""
        dates = pd.date_range('2024-01-01', periods=num_rows, freq='h')
        data = pd.DataFrame({
            'open': [100.0] * num_rows,
            'high': [105.0] * num_rows,
            'low': [99.0] * num_rows,
            'close': [103.0] * num_rows,
            'volume': [1000] * num_rows
        }, index=dates)

        # Should validate efficiently regardless of size
        validate_ohlcv_data(data, 'ZS', '1h')

    def test_many_small_gaps_not_logged(self):
        """Test performance when dataset has many small gaps."""
        # Create dataset with many 1-hour gaps (weekend gaps)
        all_dates = []
        start_date = pd.Timestamp('2024-01-01')
        for day in range(100):  # 100 days
            # 12 hours per day with 12-hour gaps (like market hours)
            day_start = start_date + pd.Timedelta(days=day)
            dates = pd.date_range(day_start + pd.Timedelta(hours=9), periods=12, freq='h')
            all_dates.extend(dates.to_list())

        data = pd.DataFrame({
            'close': [100.0] * len(all_dates)
        }, index=pd.DatetimeIndex(all_dates))

        with patch('app.backtesting.fetching.validators.logger') as mock_logger:
            detect_gaps(data, '1h', 'ZS1!', set())

            # Should not log small gaps (< 4 days)
            mock_logger.warning.assert_not_called()


class TestRealDataIntegration:
    """Test validators with realistic data scenarios."""

    def test_realistic_ohlcv_structure(self):
        """Test validation with realistic OHLCV data structure."""
        # Simulate real TradingView data
        dates = pd.date_range('2024-01-01', periods=200, freq='1h')
        data = pd.DataFrame({
            'symbol': ['CBOT:ZS1!'] * 200,
            'open': [1200.0 + i * 0.5 for i in range(200)],
            'high': [1205.0 + i * 0.5 for i in range(200)],
            'low': [1195.0 + i * 0.5 for i in range(200)],
            'close': [1202.0 + i * 0.5 for i in range(200)],
            'volume': [10000 + i * 100 for i in range(200)]
        }, index=dates)

        # Should validate successfully
        validate_ohlcv_data(data, 'ZS', '1h')

    def test_weekend_gaps_typical_scenario(self):
        """Test gap detection with typical weekend gaps."""
        # Create 5 trading days with weekend gaps
        dates = []
        for week in range(4):  # 4 weeks
            for day in range(5):  # Mon-Fri
                start = pd.Timestamp('2024-01-01') + pd.Timedelta(weeks=week, days=day)
                daily_hours = pd.date_range(start + pd.Timedelta(hours=9), periods=8, freq='h')
                dates.extend(daily_hours.to_list())

        data = pd.DataFrame({
            'close': [100.0] * len(dates)
        }, index=pd.DatetimeIndex(dates))

        with patch('app.backtesting.fetching.validators.logger') as mock_logger:
            detect_gaps(data, '1h', 'ZS1!', set())

            # Weekend gaps (~2.5 days) should not trigger warnings
            mock_logger.warning.assert_not_called()

    def test_holiday_gap_scenario(self):
        """Test gap detection with extended holiday gap."""
        # Simulate week-long holiday break
        dates1 = pd.date_range('2024-01-01', periods=50, freq='h')
        dates2 = pd.date_range('2024-01-15', periods=50, freq='h')  # 14-day gap
        dates = dates1.append(dates2)

        data = pd.DataFrame({
            'close': [100.0] * 100
        }, index=dates)

        with patch('app.backtesting.fetching.validators.logger') as mock_logger:
            detect_gaps(data, '1h', 'ZS1!', set())

            # Should detect extended gap
            mock_logger.warning.assert_called_once()

    def test_mixed_symbol_categories(self):
        """Test validation with symbols from different categories."""
        # Grains, Energies, Metals
        symbols_by_category = {
            'grains': ['ZS', 'ZC', 'ZW'],
            'energies': ['CL', 'NG'],
            'metals': ['GC', 'SI']
        }

        for category, symbols in symbols_by_category.items():
            result = validate_symbols(symbols)
            assert len(result) == len(symbols)
            assert all(s in result for s in symbols)

    def test_micro_contracts_validation(self):
        """Test validation with micro contract symbols."""
        # Micro contracts
        micro_symbols = ['MZC', 'MZW', 'MZS', 'MZL']

        result = validate_symbols(micro_symbols)

        # All micro grain contracts should be valid
        assert len(result) == 4

    @pytest.mark.parametrize("interval,periods", [
        ('5m', 288),  # 1 day of 5-min data
        ('15m', 96),  # 1 day of 15-min data
        ('1h', 24),  # 1 day of hourly data
        ('4h', 42),  # 1 week of 4-hour data
        ('1d', 30),  # 1 month of daily data
    ])
    def test_different_timeframes(self, interval, periods):
        """Test validation works correctly across different timeframes."""
        data = pd.DataFrame({
            'open': [100.0] * periods,
            'high': [105.0] * periods,
            'low': [99.0] * periods,
            'close': [103.0] * periods,
            'volume': [1000] * periods
        })

        # Should validate regardless of timeframe
        validate_ohlcv_data(data, 'ZS', interval)

    def test_realistic_price_data_ranges(self):
        """Test validation with realistic price ranges for different instruments."""
        test_cases = [
            ('ZS', 1200.0, 1250.0),  # Soybeans in cents
            ('CL', 70.0, 80.0),  # Crude oil in dollars
            ('GC', 1800.0, 1900.0),  # Gold in dollars
            ('ES', 4500.0, 4600.0),  # E-mini S&P
        ]

        for symbol, low_price, high_price in test_cases:
            data = pd.DataFrame({
                'open': [low_price] * 10,
                'high': [high_price] * 10,
                'low': [low_price - 5] * 10,
                'close': [(low_price + high_price) / 2] * 10,
                'volume': [10000] * 10
            })

            # Should validate with realistic prices
            validate_ohlcv_data(data, symbol, '1h')

    def test_data_quality_edge_cases(self):
        """Test validation handles common data quality issues."""
        # Test with very small volume
        data_low_volume = pd.DataFrame({
            'open': [100.0] * 5,
            'high': [105.0] * 5,
            'low': [99.0] * 5,
            'close': [103.0] * 5,
            'volume': [1, 2, 3, 4, 5]  # Very low volume
        })
        validate_ohlcv_data(data_low_volume, 'ZS', '1h')

        # Test with zero volume (halted trading)
        data_zero_volume = pd.DataFrame({
            'open': [100.0] * 5,
            'high': [105.0] * 5,
            'low': [99.0] * 5,
            'close': [103.0] * 5,
            'volume': [0, 0, 1000, 0, 2000]  # Some zero volume bars
        })
        validate_ohlcv_data(data_zero_volume, 'ZS', '1h')
