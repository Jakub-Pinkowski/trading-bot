"""
TradingView Data Fetcher for Historical Market Data.

This module provides a class-based interface for fetching and managing
historical futures data from TradingView using tvDatafeed library.
"""
import os
from datetime import datetime, timedelta

import pandas as pd
from tvDatafeed import TvDatafeed, Interval

from app.utils.logger import get_logger
from config import HISTORICAL_DATA_DIR
from futures_config import validate_symbols, get_exchange_for_symbol

logger = get_logger('backtesting/data_fetcher')

# ==================== Constants ====================

DATA_START_YEAR = 2020  # Data filtering threshold
MAX_BARS = 100000  # Maximum bars to fetch per request
GAP_DETECTION_THRESHOLD = timedelta(days=4)  # Gap detection threshold (only log gaps larger than this)

# Interval mapping for easy reference
INTERVAL_MAPPING = {
    '5m': Interval.in_5_minute,
    '15m': Interval.in_15_minute,
    '30m': Interval.in_30_minute,
    '1h': Interval.in_1_hour,
    '2h': Interval.in_2_hour,
    '4h': Interval.in_4_hour,
    '1d': Interval.in_daily,
}


# ==================== Helper Functions ====================

def _validate_exchange_compatibility(symbols, exchange):
    """
    Validate that symbols are compatible with the specified exchange.

    Args:
        symbols: List of symbol strings to validate
        exchange: Exchange name to validate against (e.g., 'CBOT', 'NYMEX')

    Returns:
        List of symbols that are compatible with the exchange

    Raises:
        ValueError: If no symbols are compatible with the exchange
    """
    exchange_compatible_symbols = []
    exchange_incompatible_symbols = []

    for symbol in symbols:
        symbol_exchange = get_exchange_for_symbol(symbol)
        if symbol_exchange == exchange:
            exchange_compatible_symbols.append(symbol)
        else:
            exchange_incompatible_symbols.append((symbol, symbol_exchange))

    if exchange_incompatible_symbols:
        incompatible_details = ', '.join([f'{sym} (requires {exch})' for sym, exch in exchange_incompatible_symbols])
        logger.warning(f'Symbols with incompatible exchange will be skipped: {incompatible_details}')

    if not exchange_compatible_symbols:
        raise ValueError(f'No symbols compatible with exchange "{exchange}". '
                         f'Incompatible symbols: {exchange_incompatible_symbols}')

    return exchange_compatible_symbols


def _validate_ohlcv_data(data, symbol, interval_label):
    """
    Validate that DataFrame contains required OHLCV columns.

    Args:
        data: DataFrame to validate
        symbol: Symbol name for error messages
        interval_label: Interval label for error messages

    Raises:
        ValueError: If required columns are missing or data is invalid
    """
    if data is None or len(data) == 0:
        raise ValueError(f'No data for {symbol} {interval_label}')

    required_columns = {'open', 'high', 'low', 'close', 'volume'}
    actual_columns = set(data.columns.str.lower())
    missing_columns = required_columns - actual_columns

    if missing_columns:
        raise ValueError(f'Missing required columns for {symbol} {interval_label}: {missing_columns}')

    # Validate data types (should be numeric)
    for col in required_columns:
        col_name = next(c for c in data.columns if c.lower() == col)
        if not pd.api.types.is_numeric_dtype(data[col_name]):
            raise ValueError(f'Column {col_name} must be numeric for {symbol} {interval_label}')


def _detect_and_log_gaps(data, interval_label, symbol):
    """
    Detect and log significant gaps in the datetime index.

    Only logs gaps larger than the configured threshold to avoid
    noise from expected gaps (weekends, holidays, etc.).

    Args:
        data: DataFrame with datetime index
        interval_label: Interval identifier for logging
        symbol: Symbol name for logging
    """
    if len(data) < 2:
        return

    sorted_index = data.index.sort_values()

    for i in range(1, len(sorted_index)):
        current_time = sorted_index[i]
        previous_time = sorted_index[i - 1]
        actual_gap = current_time - previous_time

        # Only log gaps larger than a threshold
        if actual_gap > GAP_DETECTION_THRESHOLD:
            logger.warning(f'Data gap detected in {symbol} {interval_label}: '
                           f'from {previous_time} to {current_time} '
                           f'(duration: {actual_gap})')


def _save_new_data(data, file_path, interval_label, full_symbol):
    """Save new data to a file."""
    # Detect gaps
    _detect_and_log_gaps(data, interval_label, full_symbol)

    # Save data
    data.to_parquet(file_path)
    logger.info(f'  ✅ Created {len(data)} rows')


def _update_existing_data(new_data, file_path, interval_label, full_symbol):
    """Update the existing data file with new data."""
    try:
        # Load existing data
        existing_data = pd.read_parquet(file_path)
        existing_count = len(existing_data)

        # Combine old and new data (new_data comes last)
        combined_data = pd.concat([existing_data, new_data])

        # Remove duplicates, keeping LAST occurrence (new data takes precedence)
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')].sort_index()  # type: ignore

        # Detect gaps
        _detect_and_log_gaps(combined_data, interval_label, full_symbol)

        # Save combined data
        combined_data.to_parquet(file_path)

        # Log the results
        new_entries = len(combined_data) - existing_count

        if new_entries > 0:
            logger.info(f'  ✅ +{new_entries} rows')
        elif new_entries < 0:
            logger.warning(f'  ⚠️  {new_entries} rows')
        else:
            logger.info(f'  No new data')

    except Exception as e:
        logger.error(f'Error updating existing file {file_path}: {e}')
        raise  # Re-raise to fail loudly and preserve existing data


# ==================== Data Fetcher Class ====================


class DataFetcher:
    """
    Fetches and manages historical futures data from TradingView.

    This class handles downloading, updating, and validating historical
    market data for futures contracts, with support for multiple symbols
    and timeframes.

    Example:
        fetcher = DataFetcher(
            symbols=['MZL', 'MET', 'RTY'],
            contract_suffix='1!',
            exchange='CBOT')
        fetcher.fetch_all_data(intervals=['1h', '4h', '1d'])
    """

    # ==================== Initialization ====================

    def __init__(self, symbols, contract_suffix, exchange):
        """
        Initialize the data fetcher.

        Args:
            symbols: List of base symbol names (e.g., ['MZL', 'MET', 'RTY'])
            contract_suffix: Contract identifier suffix (e.g., '1!' for front month)
            exchange: Exchange name for data fetching (e.g., 'CBOT')

        Raises:
            ValueError: If a symbols list is empty or parameters are invalid
        """
        if not symbols:
            raise ValueError('symbols list cannot be empty')
        if not contract_suffix:
            raise ValueError('contract_suffix cannot be empty')
        if not exchange:
            raise ValueError('exchange cannot be empty')

        # Validate symbols against the allowed list
        valid_symbols, invalid_symbols = validate_symbols(symbols)

        if invalid_symbols:
            logger.warning(f'Invalid symbols will be skipped: {invalid_symbols}')

        if not valid_symbols:
            raise ValueError(f'No valid symbols provided. All symbols were invalid: {invalid_symbols}')

        # Validate exchange compatibility for each symbol
        self.symbols = _validate_exchange_compatibility(valid_symbols, exchange)
        self.contract_suffix = contract_suffix
        self.exchange = exchange
        self.tv_client = TvDatafeed()
        self.year_threshold = datetime(DATA_START_YEAR, 1, 1)

    # ==================== Public Methods ====================

    def fetch_all_data(self, intervals):
        """
        Fetch data for all configured symbols and intervals.

        Downloads historical data for each symbol-interval combination,
        handles updates to existing data, and validates data quality.

        Args:
            intervals: List of interval labels to fetch (e.g., ['5m', '1h', '1d'])

        Raises:
            ValueError: If intervals list is empty or contains invalid intervals
        """
        if not intervals:
            raise ValueError('intervals list cannot be empty')

        # Validate all intervals are supported
        invalid_intervals = [i for i in intervals if i not in INTERVAL_MAPPING]
        if invalid_intervals:
            raise ValueError(f'Invalid intervals: {invalid_intervals}. '
                             f'Supported: {list(INTERVAL_MAPPING.keys())}')

        for symbol in self.symbols:
            self._fetch_symbol_data(symbol, intervals)

    # ==================== Private Methods ====================

    def _fetch_interval_data(self, base_symbol, full_symbol, interval_label, output_dir):
        """Fetch data for a single symbol-interval combination."""
        if interval_label not in INTERVAL_MAPPING:
            logger.warning(f'Invalid interval: {interval_label}. Skipping.')
            return

        interval = INTERVAL_MAPPING[interval_label]

        try:
            # Fetch data from TradingView
            data = self.tv_client.get_hist(
                symbol=full_symbol,
                exchange=self.exchange,
                interval=interval,
                n_bars=MAX_BARS,
                fut_contract=None  # Not used for continuous contracts (1!, 2!, etc.)
            )

            if data is None or len(data) == 0:
                logger.warning(f'No data received for {full_symbol} {interval_label}')
                return

            # Validate OHLCV data structure
            _validate_ohlcv_data(data, full_symbol, interval_label)

            # Filter data from 2020 onwards
            data = data[data.index >= self.year_threshold]

            if len(data) == 0:
                logger.warning(f'No data after year filtering for {full_symbol} {interval_label}')
                return

            # Save or update the data
            file_path = os.path.join(output_dir, f'{base_symbol}_{interval_label}.parquet')

            if os.path.exists(file_path):
                _update_existing_data(data, file_path, interval_label, full_symbol)
            else:
                _save_new_data(data, file_path, interval_label, full_symbol)

        except ValueError as e:
            logger.error(f'Data validation failed for {full_symbol} {interval_label}: {e}')
        except Exception as e:
            logger.error(f'Error fetching data for {full_symbol} {interval_label}: {e}')

    def _fetch_symbol_data(self, symbol, intervals):
        """Fetch data for a single symbol across multiple intervals."""
        full_symbol = symbol + self.contract_suffix
        output_dir = os.path.join(HISTORICAL_DATA_DIR, self.contract_suffix, symbol)
        os.makedirs(output_dir, exist_ok=True)

        for idx, interval_label in enumerate(intervals, 1):
            logger.info(f'{symbol} [{idx}/{len(intervals)}] {interval_label}')
            self._fetch_interval_data(symbol, full_symbol, interval_label, output_dir)
