"""
Tests for Data Fetcher Module.

Tests cover:
- DataFetcher initialization and validation
- Single symbol-interval data fetching
- Bulk data fetching across multiple symbols and intervals
- Data updating and deduplication logic
- File saving and loading operations
- Error handling and recovery
- Integration with TradingView API
- Year filtering and data validation

All tests use realistic futures symbols and mock TradingView responses.
"""
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
from tvDatafeed import Interval

from app.backtesting.fetching.data_fetcher import (
    DataFetcher,
    _save_new_data,
    _update_existing_data,
    INTERVAL_MAPPING,
    DATA_START_YEAR,
    MAX_BARS
)


# ==================== Fixtures ====================

@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV DataFrame for testing."""
    dates = pd.date_range('2024-01-01', periods=100, freq='h')
    return pd.DataFrame({
        'open': [100.0 + i * 0.1 for i in range(100)],
        'high': [105.0 + i * 0.1 for i in range(100)],
        'low': [99.0 + i * 0.1 for i in range(100)],
        'close': [103.0 + i * 0.1 for i in range(100)],
        'volume': [1000 + i * 10 for i in range(100)]
    }, index=dates)


@pytest.fixture
def old_data_2020():
    """Create sample data from 2020."""
    dates = pd.date_range('2020-01-01', periods=50, freq='h')
    return pd.DataFrame({
        'open': [100.0] * 50,
        'high': [105.0] * 50,
        'low': [99.0] * 50,
        'close': [103.0] * 50,
        'volume': [1000] * 50
    }, index=dates)


@pytest.fixture
def new_data_2024():
    """Create sample data from 2024."""
    dates = pd.date_range('2024-01-01', periods=50, freq='h')
    return pd.DataFrame({
        'open': [110.0] * 50,
        'high': [115.0] * 50,
        'low': [109.0] * 50,
        'close': [113.0] * 50,
        'volume': [1500] * 50
    }, index=dates)


@pytest.fixture
def mock_tv_client():
    """Create mock TradingView client."""
    with patch('app.backtesting.fetching.data_fetcher.TvDatafeed') as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary directory for test data files."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


# ==================== Test Classes ====================

class TestDataFetcherInitialization:
    """Test DataFetcher initialization and validation."""

    def test_valid_initialization(self, mock_tv_client):
        """Test successful initialization with valid parameters."""
        fetcher = DataFetcher(
            symbols=['ZS', 'ZC'],
            contract_suffix='1!',
            exchange='CBOT'
        )

        assert fetcher.symbols == ['ZS', 'ZC']
        assert fetcher.contract_suffix == '1!'
        assert fetcher.exchange == 'CBOT'
        assert fetcher.tv_client is not None
        assert fetcher.year_threshold == datetime(DATA_START_YEAR, 1, 1)

    def test_empty_symbols_raises_error(self):
        """Test ValueError raised for empty symbols list."""
        with pytest.raises(ValueError, match='symbols list cannot be empty'):
            DataFetcher(symbols=[], contract_suffix='1!', exchange='CBOT')

    def test_empty_contract_suffix_raises_error(self):
        """Test ValueError raised for empty contract suffix."""
        with pytest.raises(ValueError, match='contract_suffix cannot be empty'):
            DataFetcher(symbols=['ZS'], contract_suffix='', exchange='CBOT')

    def test_empty_exchange_raises_error(self):
        """Test ValueError raised for empty exchange."""
        with pytest.raises(ValueError, match='exchange cannot be empty'):
            DataFetcher(symbols=['ZS'], contract_suffix='1!', exchange='')

    def test_invalid_symbols_filtered(self, mock_tv_client):
        """Test invalid symbols are filtered during initialization."""
        with patch('app.backtesting.fetching.data_fetcher.validate_symbols') as mock_validate:
            mock_validate.return_value = ['ZS']  # Only ZS is valid

            fetcher = DataFetcher(
                symbols=['ZS', 'INVALID'],
                contract_suffix='1!',
                exchange='CBOT'
            )

            # Only valid symbol included
            assert fetcher.symbols == ['ZS']
            mock_validate.assert_called_once_with(['ZS', 'INVALID'])

    def test_exchange_incompatible_symbols_filtered(self, mock_tv_client):
        """Test symbols incompatible with exchange are filtered."""
        with patch('app.backtesting.fetching.data_fetcher.validate_symbols') as mock_val_sym:
            with patch('app.backtesting.fetching.data_fetcher.validate_exchange_compatibility') as mock_val_exch:
                mock_val_sym.return_value = ['ZS', 'ES']
                mock_val_exch.return_value = ['ZS']  # ES not compatible with CBOT

                fetcher = DataFetcher(
                    symbols=['ZS', 'ES'],
                    contract_suffix='1!',
                    exchange='CBOT'
                )

                assert fetcher.symbols == ['ZS']
                mock_val_exch.assert_called_once_with(['ZS', 'ES'], 'CBOT')

    def test_multiple_valid_symbols(self, mock_tv_client):
        """Test initialization with multiple valid symbols."""
        fetcher = DataFetcher(
            symbols=['ZS', 'ZC', 'ZW'],
            contract_suffix='1!',
            exchange='CBOT'
        )

        assert len(fetcher.symbols) == 3


class TestFetchAllData:
    """Test fetch_all_data method."""

    def test_empty_intervals_raises_error(self, mock_tv_client):
        """Test ValueError raised for empty intervals list."""
        fetcher = DataFetcher(symbols=['ZS'], contract_suffix='1!', exchange='CBOT')

        with pytest.raises(ValueError, match='intervals list cannot be empty'):
            fetcher.fetch_all_data([])

    def test_invalid_interval_raises_error(self, mock_tv_client):
        """Test ValueError raised for invalid interval."""
        fetcher = DataFetcher(symbols=['ZS'], contract_suffix='1!', exchange='CBOT')

        with pytest.raises(ValueError, match='Invalid intervals'):
            fetcher.fetch_all_data(['1h', 'INVALID'])

    def test_valid_intervals_accepted(self, mock_tv_client):
        """Test all valid intervals are accepted."""
        fetcher = DataFetcher(symbols=['ZS'], contract_suffix='1!', exchange='CBOT')

        with patch.object(fetcher, '_fetch_symbol_data'):
            # Should not raise error
            fetcher.fetch_all_data(['5m', '15m', '1h', '4h', '1d'])

    def test_fetch_all_data_calls_fetch_symbol_data(self, mock_tv_client):
        """Test fetch_all_data calls _fetch_symbol_data for each symbol."""
        fetcher = DataFetcher(symbols=['ZS', 'ZC'], contract_suffix='1!', exchange='CBOT')

        with patch.object(fetcher, '_fetch_symbol_data') as mock_fetch:
            fetcher.fetch_all_data(['1h', '4h'])

            # Should be called once per symbol
            assert mock_fetch.call_count == 2
            mock_fetch.assert_any_call('ZS', ['1h', '4h'])
            mock_fetch.assert_any_call('ZC', ['1h', '4h'])

    def test_single_symbol_single_interval(self, mock_tv_client):
        """Test fetch with single symbol and interval."""
        fetcher = DataFetcher(symbols=['ZS'], contract_suffix='1!', exchange='CBOT')

        with patch.object(fetcher, '_fetch_symbol_data') as mock_fetch:
            fetcher.fetch_all_data(['1h'])

            mock_fetch.assert_called_once_with('ZS', ['1h'])


class TestFetchIntervalData:
    """Test _fetch_interval_data method."""

    def test_successful_data_fetch(self, mock_tv_client, sample_ohlcv_data, temp_data_dir):
        """Test successful data fetching and saving."""
        fetcher = DataFetcher(symbols=['ZS'], contract_suffix='1!', exchange='CBOT')
        mock_tv_client.get_hist.return_value = sample_ohlcv_data

        with patch('app.backtesting.fetching.data_fetcher.validate_ohlcv_data'):
            with patch('app.backtesting.fetching.data_fetcher.detect_and_log_gaps'):
                with patch('app.backtesting.fetching.data_fetcher.logger'):
                    output_dir = str(temp_data_dir)
                    fetcher._fetch_interval_data('ZS', 'ZS1!', '1h', output_dir)

                    # Verify TradingView API was called
                    mock_tv_client.get_hist.assert_called_once_with(
                        symbol='ZS1!',
                        exchange='CBOT',
                        interval=Interval.in_1_hour,
                        n_bars=MAX_BARS,
                        fut_contract=None
                    )

                    # Verify file was created
                    file_path = Path(output_dir) / 'ZS_1h.parquet'
                    assert file_path.exists()

    def test_none_data_logged(self, mock_tv_client, temp_data_dir):
        """Test warning logged when API returns None."""
        fetcher = DataFetcher(symbols=['ZS'], contract_suffix='1!', exchange='CBOT')
        mock_tv_client.get_hist.return_value = None

        with patch('app.backtesting.fetching.data_fetcher.logger') as mock_logger:
            output_dir = str(temp_data_dir)
            fetcher._fetch_interval_data('ZS', 'ZS1!', '1h', output_dir)

            # Warning should be logged
            mock_logger.warning.assert_called()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert 'No data received' in warning_msg

    def test_empty_dataframe_logged(self, mock_tv_client, temp_data_dir):
        """Test warning logged when API returns empty DataFrame."""
        fetcher = DataFetcher(symbols=['ZS'], contract_suffix='1!', exchange='CBOT')
        mock_tv_client.get_hist.return_value = pd.DataFrame()

        with patch('app.backtesting.fetching.data_fetcher.logger') as mock_logger:
            output_dir = str(temp_data_dir)
            fetcher._fetch_interval_data('ZS', 'ZS1!', '1h', output_dir)

            mock_logger.warning.assert_called()

    def test_year_filtering_applied(self, mock_tv_client, temp_data_dir):
        """Test data is filtered to 2020 onwards."""
        # Create data with dates before and after 2020
        dates_old = pd.date_range('2018-01-01', periods=50, freq='h')
        dates_new = pd.date_range('2023-01-01', periods=50, freq='h')
        dates = dates_old.append(dates_new)

        mixed_data = pd.DataFrame({
            'open': [100.0] * 100,
            'high': [105.0] * 100,
            'low': [99.0] * 100,
            'close': [103.0] * 100,
            'volume': [1000] * 100
        }, index=dates)

        fetcher = DataFetcher(symbols=['ZS'], contract_suffix='1!', exchange='CBOT')
        mock_tv_client.get_hist.return_value = mixed_data

        with patch('app.backtesting.fetching.data_fetcher.validate_ohlcv_data'):
            with patch('app.backtesting.fetching.data_fetcher.detect_and_log_gaps'):
                with patch('app.backtesting.fetching.data_fetcher.logger'):
                    output_dir = str(temp_data_dir)
                    fetcher._fetch_interval_data('ZS', 'ZS1!', '1h', output_dir)

                    # Load saved file and verify only 2020+ data
                    file_path = Path(output_dir) / 'ZS_1h.parquet'
                    saved_data = pd.read_parquet(file_path)

                    # Should only have 50 rows (2023 data)
                    assert len(saved_data) == 50
                    assert saved_data.index.min().year >= DATA_START_YEAR

    def test_all_old_data_filtered_logged(self, mock_tv_client, temp_data_dir):
        """Test warning when all data is filtered out by year threshold."""
        # Create data entirely before 2020
        dates = pd.date_range('2018-01-01', periods=50, freq='h')
        old_data = pd.DataFrame({
            'open': [100.0] * 50,
            'high': [105.0] * 50,
            'low': [99.0] * 50,
            'close': [103.0] * 50,
            'volume': [1000] * 50
        }, index=dates)

        fetcher = DataFetcher(symbols=['ZS'], contract_suffix='1!', exchange='CBOT')
        mock_tv_client.get_hist.return_value = old_data

        with patch('app.backtesting.fetching.data_fetcher.validate_ohlcv_data'):
            with patch('app.backtesting.fetching.data_fetcher.logger') as mock_logger:
                output_dir = str(temp_data_dir)
                fetcher._fetch_interval_data('ZS', 'ZS1!', '1h', output_dir)

                # Warning should be logged
                warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
                assert any('No data after year filtering' in msg for msg in warning_calls)

    def test_validation_error_handled(self, mock_tv_client, sample_ohlcv_data, temp_data_dir):
        """Test ValueError from validation is caught and logged."""
        fetcher = DataFetcher(symbols=['ZS'], contract_suffix='1!', exchange='CBOT')
        mock_tv_client.get_hist.return_value = sample_ohlcv_data

        with patch('app.backtesting.fetching.data_fetcher.validate_ohlcv_data') as mock_validate:
            mock_validate.side_effect = ValueError('Missing columns')

            with patch('app.backtesting.fetching.data_fetcher.logger') as mock_logger:
                output_dir = str(temp_data_dir)
                fetcher._fetch_interval_data('ZS', 'ZS1!', '1h', output_dir)

                # Error should be logged
                mock_logger.error.assert_called()
                error_msg = mock_logger.error.call_args[0][0]
                assert 'Data validation failed' in error_msg

    def test_generic_exception_handled(self, mock_tv_client, temp_data_dir):
        """Test generic exceptions are caught and logged."""
        fetcher = DataFetcher(symbols=['ZS'], contract_suffix='1!', exchange='CBOT')
        mock_tv_client.get_hist.side_effect = Exception('API Error')

        with patch('app.backtesting.fetching.data_fetcher.logger') as mock_logger:
            output_dir = str(temp_data_dir)
            fetcher._fetch_interval_data('ZS', 'ZS1!', '1h', output_dir)

            # Error should be logged
            mock_logger.error.assert_called()
            error_msg = mock_logger.error.call_args[0][0]
            assert 'Error fetching data' in error_msg

    def test_invalid_interval_logged_and_skipped(self, mock_tv_client, temp_data_dir):
        """Test invalid interval is logged and skipped."""
        fetcher = DataFetcher(symbols=['ZS'], contract_suffix='1!', exchange='CBOT')

        with patch('app.backtesting.fetching.data_fetcher.logger') as mock_logger:
            output_dir = str(temp_data_dir)
            fetcher._fetch_interval_data('ZS', 'ZS1!', 'INVALID', output_dir)

            # Warning should be logged
            mock_logger.warning.assert_called()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert 'Invalid interval' in warning_msg

            # API should not be called
            mock_tv_client.get_hist.assert_not_called()


class TestFetchSymbolData:
    """Test _fetch_symbol_data method."""

    def test_directory_created(self, mock_tv_client, temp_data_dir):
        """Test output directory is created."""
        with patch('app.backtesting.fetching.data_fetcher.HISTORICAL_DATA_DIR', str(temp_data_dir)):
            fetcher = DataFetcher(symbols=['ZS'], contract_suffix='1!', exchange='CBOT')

            with patch.object(fetcher, '_fetch_interval_data'):
                with patch('app.backtesting.fetching.data_fetcher.logger'):
                    fetcher._fetch_symbol_data('ZS', ['1h'])

                    # Directory should exist
                    expected_dir = temp_data_dir / '1!' / 'ZS'
                    assert expected_dir.exists()
                    assert expected_dir.is_dir()

    def test_fetch_interval_called_for_each_interval(self, mock_tv_client, temp_data_dir):
        """Test _fetch_interval_data called for each interval."""
        with patch('app.backtesting.fetching.data_fetcher.HISTORICAL_DATA_DIR', str(temp_data_dir)):
            fetcher = DataFetcher(symbols=['ZS'], contract_suffix='1!', exchange='CBOT')

            with patch.object(fetcher, '_fetch_interval_data') as mock_fetch:
                with patch('app.backtesting.fetching.data_fetcher.logger'):
                    fetcher._fetch_symbol_data('ZS', ['1h', '4h', '1d'])

                    # Should be called 3 times
                    assert mock_fetch.call_count == 3

    def test_full_symbol_constructed(self, mock_tv_client, temp_data_dir):
        """Test full symbol is constructed correctly."""
        with patch('app.backtesting.fetching.data_fetcher.HISTORICAL_DATA_DIR', str(temp_data_dir)):
            fetcher = DataFetcher(symbols=['ZS'], contract_suffix='1!', exchange='CBOT')

            with patch.object(fetcher, '_fetch_interval_data') as mock_fetch:
                with patch('app.backtesting.fetching.data_fetcher.logger'):
                    fetcher._fetch_symbol_data('ZS', ['1h'])

                    # Verify full symbol
                    call_args = mock_fetch.call_args[0]
                    assert call_args[1] == 'ZS1!'

    def test_progress_logging(self, mock_tv_client, temp_data_dir):
        """Test progress is logged for each interval."""
        with patch('app.backtesting.fetching.data_fetcher.HISTORICAL_DATA_DIR', str(temp_data_dir)):
            fetcher = DataFetcher(symbols=['ZS'], contract_suffix='1!', exchange='CBOT')

            with patch.object(fetcher, '_fetch_interval_data'):
                with patch('app.backtesting.fetching.data_fetcher.logger') as mock_logger:
                    fetcher._fetch_symbol_data('ZS', ['1h', '4h', '1d'])

                    # Should log progress 3 times
                    assert mock_logger.info.call_count == 3

                    # Check progress format
                    first_call = mock_logger.info.call_args_list[0][0][0]
                    assert 'ZS' in first_call
                    assert '[1/3]' in first_call


class TestSaveNewData:
    """Test _save_new_data helper function."""

    def test_file_created(self, sample_ohlcv_data, temp_data_dir):
        """Test parquet file is created."""
        file_path = temp_data_dir / 'test.parquet'

        with patch('app.backtesting.fetching.data_fetcher.detect_and_log_gaps'):
            with patch('app.backtesting.fetching.data_fetcher.logger'):
                _save_new_data(sample_ohlcv_data, str(file_path), '1h', 'ZS1!')

                assert file_path.exists()

    def test_data_content_correct(self, sample_ohlcv_data, temp_data_dir):
        """Test saved data matches input data."""
        file_path = temp_data_dir / 'test.parquet'

        with patch('app.backtesting.fetching.data_fetcher.detect_and_log_gaps'):
            with patch('app.backtesting.fetching.data_fetcher.logger'):
                _save_new_data(sample_ohlcv_data, str(file_path), '1h', 'ZS1!')

                loaded_data = pd.read_parquet(file_path)

                # Compare values and index (ignore freq metadata)
                assert len(loaded_data) == len(sample_ohlcv_data)
                assert (loaded_data.index == sample_ohlcv_data.index).all()
                pd.testing.assert_frame_equal(loaded_data, sample_ohlcv_data, check_freq=False)

    def test_gap_detection_called(self, sample_ohlcv_data, temp_data_dir):
        """Test gap detection is called before saving."""
        file_path = temp_data_dir / 'test.parquet'

        with patch('app.backtesting.fetching.data_fetcher.detect_and_log_gaps') as mock_gaps:
            with patch('app.backtesting.fetching.data_fetcher.logger'):
                _save_new_data(sample_ohlcv_data, str(file_path), '1h', 'ZS1!')

                mock_gaps.assert_called_once_with(sample_ohlcv_data, '1h', 'ZS1!')

    def test_success_logged(self, sample_ohlcv_data, temp_data_dir):
        """Test success message is logged."""
        file_path = temp_data_dir / 'test.parquet'

        with patch('app.backtesting.fetching.data_fetcher.detect_and_log_gaps'):
            with patch('app.backtesting.fetching.data_fetcher.logger') as mock_logger:
                _save_new_data(sample_ohlcv_data, str(file_path), '1h', 'ZS1!')

                mock_logger.info.assert_called_once()
                info_msg = mock_logger.info.call_args[0][0]
                assert 'Created' in info_msg
                assert str(len(sample_ohlcv_data)) in info_msg


class TestUpdateExistingData:
    """Test _update_existing_data helper function."""

    def test_new_data_added(self, old_data_2020, new_data_2024, temp_data_dir):
        """Test new data is appended to existing data."""
        file_path = temp_data_dir / 'test.parquet'
        old_data_2020.to_parquet(file_path)

        with patch('app.backtesting.fetching.data_fetcher.detect_and_log_gaps'):
            with patch('app.backtesting.fetching.data_fetcher.logger'):
                _update_existing_data(new_data_2024, str(file_path), '1h', 'ZS1!')

                combined = pd.read_parquet(file_path)

                # Should have both datasets
                assert len(combined) == 100
                assert combined.index.min().year == 2020
                assert combined.index.max().year == 2024

    def test_duplicates_removed_keep_last(self, temp_data_dir):
        """Test duplicates are removed keeping the latest data."""
        # Create overlapping data
        dates = pd.date_range('2024-01-01', periods=10, freq='h')

        old_data = pd.DataFrame({
            'close': [100.0] * 10
        }, index=dates)

        new_data = pd.DataFrame({
            'close': [200.0] * 10  # Same dates, different values
        }, index=dates)

        file_path = temp_data_dir / 'test.parquet'
        old_data.to_parquet(file_path)

        with patch('app.backtesting.fetching.data_fetcher.detect_and_log_gaps'):
            with patch('app.backtesting.fetching.data_fetcher.logger'):
                _update_existing_data(new_data, str(file_path), '1h', 'ZS1!')

                combined = pd.read_parquet(file_path)

                # Should have 10 rows (duplicates removed)
                assert len(combined) == 10
                # Should have new values (keep='last')
                assert all(combined['close'] == 200.0)

    def test_data_sorted_by_index(self, temp_data_dir):
        """Test combined data is sorted by datetime index."""
        # Create data in wrong order
        dates1 = pd.date_range('2024-01-10', periods=5, freq='h')
        dates2 = pd.date_range('2024-01-01', periods=5, freq='h')

        old_data = pd.DataFrame({'close': [100.0] * 5}, index=dates1)
        new_data = pd.DataFrame({'close': [200.0] * 5}, index=dates2)

        file_path = temp_data_dir / 'test.parquet'
        old_data.to_parquet(file_path)

        with patch('app.backtesting.fetching.data_fetcher.detect_and_log_gaps'):
            with patch('app.backtesting.fetching.data_fetcher.logger'):
                _update_existing_data(new_data, str(file_path), '1h', 'ZS1!')

                combined = pd.read_parquet(file_path)

                # Should be sorted
                assert combined.index.is_monotonic_increasing

    def test_positive_new_entries_logged(self, old_data_2020, new_data_2024, temp_data_dir):
        """Test positive new entries are logged."""
        file_path = temp_data_dir / 'test.parquet'
        old_data_2020.to_parquet(file_path)

        with patch('app.backtesting.fetching.data_fetcher.detect_and_log_gaps'):
            with patch('app.backtesting.fetching.data_fetcher.logger') as mock_logger:
                _update_existing_data(new_data_2024, str(file_path), '1h', 'ZS1!')

                mock_logger.info.assert_called()
                info_msg = mock_logger.info.call_args[0][0]
                assert '+' in info_msg
                assert '50' in info_msg

    def test_no_new_data_logged(self, old_data_2020, temp_data_dir):
        """Test message logged when no new data added."""
        file_path = temp_data_dir / 'test.parquet'
        old_data_2020.to_parquet(file_path)

        # Try to add same data again
        with patch('app.backtesting.fetching.data_fetcher.detect_and_log_gaps'):
            with patch('app.backtesting.fetching.data_fetcher.logger') as mock_logger:
                _update_existing_data(old_data_2020, str(file_path), '1h', 'ZS1!')

                mock_logger.info.assert_called()
                info_msg = mock_logger.info.call_args[0][0]
                assert 'No new data' in info_msg

    def test_overlapping_data_no_increase(self, temp_data_dir):
        """Test info message when overlapping data adds no new rows."""
        # Create scenario where new data overlaps completely with old data
        dates1 = pd.date_range('2024-01-01', periods=100, freq='h')
        dates2 = pd.date_range('2024-01-01', periods=50, freq='h')

        old_data = pd.DataFrame({'close': [100.0] * 100}, index=dates1)
        new_data = pd.DataFrame({'close': [200.0] * 50}, index=dates2)

        file_path = temp_data_dir / 'test.parquet'
        old_data.to_parquet(file_path)

        with patch('app.backtesting.fetching.data_fetcher.detect_and_log_gaps'):
            with patch('app.backtesting.fetching.data_fetcher.logger') as mock_logger:
                _update_existing_data(new_data, str(file_path), '1h', 'ZS1!')

                # Should log info for no new data (overlapping data)
                mock_logger.info.assert_called()
                info_msg = mock_logger.info.call_args[0][0]
                assert 'No new data' in info_msg

    def test_gap_detection_called_on_combined(self, old_data_2020, new_data_2024, temp_data_dir):
        """Test gap detection is called on combined data."""
        file_path = temp_data_dir / 'test.parquet'
        old_data_2020.to_parquet(file_path)

        with patch('app.backtesting.fetching.data_fetcher.detect_and_log_gaps') as mock_gaps:
            with patch('app.backtesting.fetching.data_fetcher.logger'):
                _update_existing_data(new_data_2024, str(file_path), '1h', 'ZS1!')

                # Should be called once with combined data
                mock_gaps.assert_called_once()
                call_args = mock_gaps.call_args[0]
                assert len(call_args[0]) == 100  # Combined data

    def test_error_logged_without_raising(self, new_data_2024, temp_data_dir):
        """Test error is logged but not raised on failure."""
        file_path = temp_data_dir / 'nonexistent.parquet'

        with patch('app.backtesting.fetching.data_fetcher.logger') as mock_logger:
            # Should not raise, just log
            _update_existing_data(new_data_2024, str(file_path), '1h', 'ZS1!')

            mock_logger.error.assert_called()
            error_msg = mock_logger.error.call_args[0][0]
            assert 'Error updating existing file' in error_msg


class TestIntervalMapping:
    """Test INTERVAL_MAPPING constant."""

    def test_all_expected_intervals_present(self):
        """Test all expected intervals are in mapping."""
        expected_intervals = ['5m', '15m', '30m', '1h', '2h', '4h', '1d']

        for interval in expected_intervals:
            assert interval in INTERVAL_MAPPING

    def test_mapping_values_are_interval_objects(self):
        """Test mapping values are Interval enum values."""
        assert INTERVAL_MAPPING['1h'] == Interval.in_1_hour
        assert INTERVAL_MAPPING['4h'] == Interval.in_4_hour
        assert INTERVAL_MAPPING['1d'] == Interval.in_daily


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components."""

    def test_full_fetch_workflow(self, mock_tv_client, sample_ohlcv_data, temp_data_dir):
        """Test complete workflow from initialization to data saving."""
        with patch('app.backtesting.fetching.data_fetcher.HISTORICAL_DATA_DIR', str(temp_data_dir)):
            # Initialize fetcher
            fetcher = DataFetcher(
                symbols=['ZS'],
                contract_suffix='1!',
                exchange='CBOT'
            )

            # Mock API response
            mock_tv_client.get_hist.return_value = sample_ohlcv_data

            with patch('app.backtesting.fetching.data_fetcher.validate_ohlcv_data'):
                with patch('app.backtesting.fetching.data_fetcher.detect_and_log_gaps'):
                    with patch('app.backtesting.fetching.data_fetcher.logger'):
                        # Fetch data
                        fetcher.fetch_all_data(['1h'])

                        # Verify file created
                        file_path = temp_data_dir / '1!' / 'ZS' / 'ZS_1h.parquet'
                        assert file_path.exists()

                        # Verify data content
                        saved_data = pd.read_parquet(file_path)
                        assert len(saved_data) > 0

    def test_multiple_symbols_multiple_intervals(self, mock_tv_client, sample_ohlcv_data, temp_data_dir):
        """Test fetching multiple symbols and intervals."""
        with patch('app.backtesting.fetching.data_fetcher.HISTORICAL_DATA_DIR', str(temp_data_dir)):
            fetcher = DataFetcher(
                symbols=['ZS', 'ZC'],
                contract_suffix='1!',
                exchange='CBOT'
            )

            mock_tv_client.get_hist.return_value = sample_ohlcv_data

            with patch('app.backtesting.fetching.data_fetcher.validate_ohlcv_data'):
                with patch('app.backtesting.fetching.data_fetcher.detect_and_log_gaps'):
                    with patch('app.backtesting.fetching.data_fetcher.logger'):
                        fetcher.fetch_all_data(['1h', '4h'])

                        # Should have 4 files (2 symbols × 2 intervals)
                        zs_1h = temp_data_dir / '1!' / 'ZS' / 'ZS_1h.parquet'
                        zs_4h = temp_data_dir / '1!' / 'ZS' / 'ZS_4h.parquet'
                        zc_1h = temp_data_dir / '1!' / 'ZC' / 'ZC_1h.parquet'
                        zc_4h = temp_data_dir / '1!' / 'ZC' / 'ZC_4h.parquet'

                        assert zs_1h.exists()
                        assert zs_4h.exists()
                        assert zc_1h.exists()
                        assert zc_4h.exists()

    def test_update_existing_files(self, mock_tv_client, old_data_2020, new_data_2024, temp_data_dir):
        """Test updating existing data files."""
        with patch('app.backtesting.fetching.data_fetcher.HISTORICAL_DATA_DIR', str(temp_data_dir)):
            # Create existing file
            output_dir = temp_data_dir / '1!' / 'ZS'
            output_dir.mkdir(parents=True)
            file_path = output_dir / 'ZS_1h.parquet'
            old_data_2020.to_parquet(file_path)

            # Fetch new data
            fetcher = DataFetcher(
                symbols=['ZS'],
                contract_suffix='1!',
                exchange='CBOT'
            )

            mock_tv_client.get_hist.return_value = new_data_2024

            with patch('app.backtesting.fetching.data_fetcher.validate_ohlcv_data'):
                with patch('app.backtesting.fetching.data_fetcher.detect_and_log_gaps'):
                    with patch('app.backtesting.fetching.data_fetcher.logger'):
                        fetcher.fetch_all_data(['1h'])

                        # Verify data was updated
                        combined = pd.read_parquet(file_path)
                        assert len(combined) == 100  # 50 old + 50 new
