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

# Data filtering threshold
DATA_START_YEAR = 2020

# Maximum bars to fetch per request
MAX_BARS = 100000

# Gap detection threshold (only log gaps larger than this)
GAP_DETECTION_THRESHOLD = timedelta(days=4)

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
            exchange='CBOT'
        )
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

        total_combinations = len(self.symbols) * len(intervals)
        logger.info(f'Starting data fetch for {len(self.symbols)} symbols across {len(intervals)} intervals '
                    f'({total_combinations} total combinations)')

        for symbol in self.symbols:
            self._fetch_symbol_data(symbol, intervals)

        logger.info(f'✅ Data fetch completed for all {len(self.symbols)} symbols and {len(intervals)} intervals')

    # ==================== Private Methods ====================

    # --- Symbol Processing ---

    def _fetch_symbol_data(self, symbol, intervals):
        """Fetch data for a single symbol across multiple intervals."""
        full_symbol = symbol + self.contract_suffix
        output_dir = self._get_output_dir(symbol)
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f'Processing {symbol} ({len(intervals)} intervals)')

        for idx, interval_label in enumerate(intervals, 1):
            logger.info(f'[{idx}/{len(intervals)}] Fetching {interval_label} data for {symbol}')
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
                fut_contract=None
            )

            if data is None or len(data) == 0:
                logger.warning(f'No data received for {full_symbol} {interval_label}')
                return

            logger.info(f'Received {len(data)} bars from TradingView for {full_symbol} {interval_label}')

            # Filter data from 2020 onwards
            data = self._filter_data_by_year(data)

            if len(data) == 0:
                logger.warning(f'No data remaining after year filter for {full_symbol} {interval_label}')
                return

            # Save or update the data
            file_path = os.path.join(output_dir, f'{base_symbol}_{interval_label}.parquet')
            self._save_or_update_data(data, file_path, base_symbol, interval_label, full_symbol)

        except Exception as e:
            logger.error(f'Error fetching data for {full_symbol} {interval_label}: {e}')

    # --- Data Processing ---

    def _save_or_update_data(self, new_data, file_path, base_symbol, interval_label, full_symbol):
        """Save new data or update existing data file."""
        if os.path.exists(file_path):
            self._update_existing_data(new_data, file_path, base_symbol, interval_label, full_symbol)
        else:
            self._save_new_data(new_data, file_path, base_symbol, interval_label, full_symbol)

    def _update_existing_data(self, new_data, file_path, base_symbol, interval_label, full_symbol):
        """Update existing data file with new data."""
        try:
            # Load existing data
            existing_data = pd.read_parquet(file_path)
            existing_count = len(existing_data)
            logger.info(f'Found existing data with {existing_count} rows for {base_symbol} {interval_label}')

            # Filter existing data to remove anything before threshold year
            existing_data = self._filter_data_by_year(existing_data)

            # Combine and deduplicate
            combined_data = pd.concat([existing_data, new_data])
            before_dedup_count = len(combined_data)
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            combined_data = combined_data.sort_index()
            after_dedup_count = len(combined_data)

            duplicates_removed = before_dedup_count - after_dedup_count
            if duplicates_removed > 0:
                logger.info(f'Removed {duplicates_removed} duplicate entries for {base_symbol} {interval_label}')

            # Final filter to ensure no pre-threshold data remains
            combined_data = self._filter_data_by_year(combined_data)

            # Detect gaps
            self._detect_and_log_gaps(combined_data, interval_label, full_symbol)

            # Save combined data
            combined_data.to_parquet(file_path)

            new_entries = len(combined_data) - existing_count
            final_count = len(combined_data)

            if new_entries > 0:
                logger.info(f'✅ Added {new_entries} new entries for {base_symbol} {interval_label} '
                            f'({existing_count} → {final_count} total rows)')
            elif new_entries < 0:
                logger.warning(f'Data count decreased by {abs(new_entries)} rows for {base_symbol} {interval_label} '
                               f'({existing_count} → {final_count} total rows) - likely due to year filtering')
            else:
                logger.info(f'No new entries added for {base_symbol} {interval_label} (total: {final_count} rows)')

            # Log date range
            self._log_date_range(combined_data, full_symbol, interval_label)

        except Exception as e:
            logger.error(f'Error updating existing file {file_path}: {e}')
            logger.info('Overwriting with new data')
            self._save_new_data(new_data, file_path, base_symbol, interval_label, full_symbol)

    def _save_new_data(self, data, file_path, base_symbol, interval_label, full_symbol):
        """Save new data to a file."""
        # Detect gaps
        self._detect_and_log_gaps(data, interval_label, full_symbol)

        # Save data
        data.to_parquet(file_path)
        logger.info(f'✅ Created new file with {len(data)} rows for {base_symbol} {interval_label}')
        logger.debug(f'File saved to: {file_path}')

        # Log date range
        self._log_date_range(data, full_symbol, interval_label)

    def _filter_data_by_year(self, data):
        """
        Filter out data points before the threshold year.

        Args:
            data: DataFrame with datetime index

        Returns:
            Filtered DataFrame containing only data from threshold year onwards
        """
        if data is None or len(data) == 0:
            return data

        filtered_data = data[data.index >= self.year_threshold]

        original_count = len(data)
        filtered_count = len(filtered_data)

        if original_count > filtered_count:
            removed_count = original_count - filtered_count
            logger.info(f'Filtered out {removed_count} rows from before {DATA_START_YEAR} '
                        f'(kept {filtered_count}/{original_count} rows)')

        return filtered_data

    # --- Data Validation ---

    def _detect_and_log_gaps(self, data, interval_label, symbol):
        """
        Detect and log significant gaps in datetime index.

        Only logs gaps larger than the configured threshold to avoid
        noise from expected gaps (weekends, holidays, etc.).

        Args:
            data: DataFrame with datetime index
            interval_label: Interval identifier for logging
            symbol: Symbol name for logging
        """
        if len(data) < 2:
            return

        gaps = []
        sorted_index = data.index.sort_values()

        for i in range(1, len(sorted_index)):
            current_time = sorted_index[i]
            previous_time = sorted_index[i - 1]
            actual_gap = current_time - previous_time

            # Only log gaps larger than threshold
            if actual_gap > GAP_DETECTION_THRESHOLD:
                gaps.append((previous_time, current_time, actual_gap))
                logger.warning(f'Data gap detected in {symbol} {interval_label}: '
                               f'from {previous_time} to {current_time} '
                               f'(duration: {actual_gap})')

    def _log_date_range(self, data, symbol, interval_label):
        """Log the first and last date in the dataset."""
        if data is None or len(data) == 0:
            logger.info(f'No data available for {symbol} {interval_label}')
            return

        first_date = data.index.min()
        last_date = data.index.max()
        logger.info(f'Data range for {symbol} {interval_label}: {first_date} to {last_date}')

    # --- Path Helpers ---

    def _get_output_dir(self, symbol):
        """Get output directory path for a symbol."""
        return os.path.join(HISTORICAL_DATA_DIR, self.contract_suffix, symbol)
