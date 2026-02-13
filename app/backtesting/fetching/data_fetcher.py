"""
TradingView Data Fetcher for Historical Market Data.

This module provides a class-based interface for fetching and managing
historical futures data from TradingView using tvDatafeed library.
"""
import os
from datetime import datetime

import pandas as pd
from tvDatafeed import TvDatafeed, Interval

from app.backtesting.fetching.validators import (
    validate_symbols,
    validate_exchange_compatibility,
    validate_ohlcv_data,
    detect_and_log_gaps
)
from app.utils.logger import get_logger
from config import HISTORICAL_DATA_DIR

logger = get_logger('backtesting/data_fetcher')

# ==================== Constants ====================

DATA_START_YEAR = 2020  # Data filtering threshold
MAX_BARS = 100000  # Maximum bars to fetch per request

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


def _save_new_data(data, file_path, interval_label, full_symbol):
    """Save new data to a file."""
    # Detect gaps
    detect_and_log_gaps(data, interval_label, full_symbol)

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
        detect_and_log_gaps(combined_data, interval_label, full_symbol)

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
        # Log error without re-raising because the caller handles failures gracefully
        logger.error(f'Error updating existing file {file_path}: {e}')


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
        valid_symbols = validate_symbols(symbols)

        # Validate exchange compatibility for each symbol
        self.symbols = validate_exchange_compatibility(valid_symbols, exchange)
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
            ValueError: If the intervals list is empty or contains invalid intervals
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
            validate_ohlcv_data(data, full_symbol, interval_label)

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
