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
        before_dedup_count = len(combined_data)

        # Remove duplicates, keeping LAST occurrence (new data takes precedence)
        # This ensures newer data from TradingView overwrites any existing entries
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')].sort_index()  # type: ignore
        after_dedup_count = len(combined_data)

        duplicates_removed = before_dedup_count - after_dedup_count
        if duplicates_removed > 0:
            logger.warning(f'Removed {duplicates_removed} duplicates (kept newer data)')

        # Detect gaps
        _detect_and_log_gaps(combined_data, interval_label, full_symbol)

        # Save combined data
        combined_data.to_parquet(file_path)

        new_entries = len(combined_data) - existing_count
        final_count = len(combined_data)

        if new_entries > 0:
            logger.info(f'  ✅ +{new_entries} rows ({existing_count} → {final_count})')
        elif new_entries < 0:
            logger.warning(f'  ⚠️  {new_entries} rows ({existing_count} → {final_count})')
        else:
            logger.info(f'  No new data ({final_count} rows)')

    except Exception as e:
        logger.error(f'Error updating existing file {file_path}: {e}')
        raise  # Re-raise to fail loudly and preserve existing data


def _save_or_update_data(new_data, file_path, interval_label, full_symbol):
    """Save new data or update an existing data file."""
    if os.path.exists(file_path):
        _update_existing_data(new_data, file_path, interval_label, full_symbol)
    else:
        _save_new_data(new_data, file_path, interval_label, full_symbol)


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
        """
        self.symbols = symbols
        self.contract_suffix = contract_suffix
        self.exchange = exchange
        self.tv_client = TvDatafeed()
        self.year_threshold = datetime(DATA_START_YEAR, 1, 1)

        logger.info(f'DataFetcher initialized for {len(symbols)} symbols with suffix {contract_suffix}')

    # ==================== Public Methods ====================

    def fetch_all_data(self, intervals):
        """
        Fetch data for all configured symbols and intervals.

        Downloads historical data for each symbol-interval combination,
        handles updates to existing data, and validates data quality.

        Args:
            intervals: List of interval labels to fetch (e.g., ['5m', '1h', '1d'])
        """

        for symbol in self.symbols:
            self._fetch_symbol_data(symbol, intervals)

    # ==================== Private Methods ====================

    # --- Symbol Processing ---

    def _fetch_symbol_data(self, symbol, intervals):
        """Fetch data for a single symbol across multiple intervals."""
        full_symbol = symbol + self.contract_suffix
        output_dir = os.path.join(HISTORICAL_DATA_DIR, self.contract_suffix, symbol)
        os.makedirs(output_dir, exist_ok=True)

        for idx, interval_label in enumerate(intervals, 1):
            logger.info(f'{symbol} [{idx}/{len(intervals)}] {interval_label}')
            self._fetch_interval_data(symbol, full_symbol, interval_label, output_dir)

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

            # Filter data from 2020 onwards
            data = self._filter_data_by_year(data)

            # Save or update the data
            file_path = os.path.join(output_dir, f'{base_symbol}_{interval_label}.parquet')
            _save_or_update_data(data, file_path, interval_label, full_symbol)

        except Exception as e:
            logger.error(f'Error fetching data for {full_symbol} {interval_label}: {e}')

    # --- Data Processing ---

    def _filter_data_by_year(self, data):
        """
        Filter out data points before the threshold year.

        Args:
            data: DataFrame with datetime index

        Returns:
            Filtered DataFrame containing only data from the threshold year onwards
        """
        return data[data.index >= self.year_threshold]
