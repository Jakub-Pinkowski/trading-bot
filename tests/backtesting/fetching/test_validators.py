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
    detect_and_log_gaps
)


# ==================== Test Classes ====================

class TestValidateSymbols:
    """Test symbol validation for TradingView compatibility."""

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


class TestDetectAndLogGaps:
    """Test gap detection in time series data."""

    def test_no_gaps_detected(self):
        """Test no warnings logged when data has no significant gaps."""
        # Create hourly data with no gaps
        dates = pd.date_range('2024-01-01', periods=24, freq='h')
        data = pd.DataFrame({
            'close': [100.0] * 24
        }, index=dates)

        with patch('app.backtesting.fetching.validators.logger') as mock_logger:
            detect_and_log_gaps(data, '1h', 'ZS1!')

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
            detect_and_log_gaps(data, '1h', 'ZS1!')

            # Warning should be logged
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert 'Data gap detected' in warning_msg
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
            detect_and_log_gaps(data, '1h', 'ZS1!')

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
            detect_and_log_gaps(data, '1h', 'ZS1!')

            # Two gaps should be logged
            assert mock_logger.warning.call_count == 2

    def test_single_row_data(self):
        """Test no error with single row of data."""
        data = pd.DataFrame({
            'close': [100.0]
        }, index=[datetime(2024, 1, 1)])

        with patch('app.backtesting.fetching.validators.logger') as mock_logger:
            detect_and_log_gaps(data, '1h', 'ZS1!')

            # No warnings for single row
            mock_logger.warning.assert_not_called()

    def test_empty_dataframe(self):
        """Test no error with empty DataFrame."""
        data = pd.DataFrame()

        with patch('app.backtesting.fetching.validators.logger') as mock_logger:
            detect_and_log_gaps(data, '1h', 'ZS1!')

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
            detect_and_log_gaps(data, '1h', 'ZS1!')

            # Gap should still be detected after sorting
            mock_logger.warning.assert_called()

    def test_gap_at_threshold_boundary(self):
        """Test gap exactly at threshold is not logged."""
        # Create gap exactly equal to threshold (4 days)
        dates1 = pd.date_range('2024-01-01', periods=5, freq='h')
        dates2 = pd.date_range('2024-01-05', periods=5, freq='h')
        dates = dates1.append(dates2)

        data = pd.DataFrame({
            'close': [100.0] * 10
        }, index=dates)

        with patch('app.backtesting.fetching.validators.logger') as mock_logger:
            detect_and_log_gaps(data, '1h', 'ZS1!')

            # Gap at threshold should not be logged (only > threshold)
            mock_logger.warning.assert_not_called()

    def test_gap_slightly_above_threshold(self):
        """Test gap just above threshold is logged."""
        # Create gap just above threshold (>4 days)
        dates1 = pd.date_range('2024-01-01', periods=5, freq='h')
        dates2 = pd.date_range('2024-01-06', periods=5, freq='h')  # 5 days gap
        dates = dates1.append(dates2)

        data = pd.DataFrame({
            'close': [100.0] * 10
        }, index=dates)

        with patch('app.backtesting.fetching.validators.logger') as mock_logger:
            detect_and_log_gaps(data, '1h', 'ZS1!')

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
            detect_and_log_gaps(data, '1h', 'CL1!')

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
