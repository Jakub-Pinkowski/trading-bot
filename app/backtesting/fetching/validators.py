"""
Validation Functions for Data Fetching.

This module provides validation utilities for ensuring data quality
and compatibility when fetching historical market data.
"""
import os
import tempfile
from datetime import datetime, timedelta

import pandas as pd
import yaml

from app.utils.logger import get_logger
from futures_config import get_exchange_for_symbol, SYMBOL_SPECS

logger = get_logger('backtesting/fetching/validators')

# ==================== Constants ====================

GAP_DETECTION_THRESHOLD = timedelta(days=5)  # Gap detection threshold (only log gaps larger than this)


# ==================== Helper Functions ====================

def load_existing_gaps(contract_suffix):
    """
    Load existing gaps from the YAML file.

    Reads the historical data gaps YAML file for the specified contract and returns
    a set of known gaps for quick lookup. Timestamps are normalized by removing
    microseconds to ensure consistent comparison across different timestamp formats
    and prevent false "NEW gap" warnings.

    Args:
        contract_suffix: Contract identifier (e.g., '1!')

    Returns:
        Set of tuples (symbol, interval, normalized_start_time) for quick lookup.
        Timestamps are normalized (microseconds removed) for consistent comparison
    """
    filename = f'historical_data_gaps_{contract_suffix}.yaml'
    file_path = os.path.join('data', filename)

    if not os.path.exists(file_path):
        return set()

    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f) or {}

        existing_gaps = set()
        for symbol, intervals in data.get('gaps', {}).items():
            for interval, gaps_list in intervals.items():
                for gap in gaps_list:
                    # Parse ISO string and normalize by removing microseconds
                    start_time_str = gap['start_time']
                    start_dt = pd.to_datetime(start_time_str)
                    normalized_dt = start_dt.replace(microsecond=0)
                    existing_gaps.add((symbol, interval, normalized_dt))

        return existing_gaps
    except Exception as e:
        logger.warning(f'Failed to load existing gaps: {e}')
        return set()


# ==================== Validation Functions ====================

def validate_symbols(symbols):
    """
    Validate that all symbols are TradingView-compatible.

    Logs warnings for invalid symbols and raises an error if no valid symbols remain.

    Args:
        symbols: List of symbol strings to validate

    Returns:
        List of valid symbols that are TradingView-compatible

    Raises:
        ValueError: If no valid symbols are provided
    """
    valid_symbols = []
    invalid_symbols = []

    for symbol in symbols:
        if symbol in SYMBOL_SPECS and SYMBOL_SPECS[symbol]['tv_compatible']:
            valid_symbols.append(symbol)
        else:
            invalid_symbols.append(symbol)

    if invalid_symbols:
        logger.warning(f'Invalid symbols will be skipped: {invalid_symbols}')

    if not valid_symbols:
        raise ValueError(f'No valid symbols provided. All symbols were invalid: {invalid_symbols}')

    return valid_symbols


def validate_exchange_compatibility(symbols, exchange):
    """
    Validate that symbols are compatible with the specified exchange.

    Filters the provided symbols to only include those that trade on the specified
    exchange. Logs warnings for incompatible symbols and raises an error if no
    compatible symbols remain.

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
        incompatible_details = ', '.join([f'{symbol_name} (requires {exchange})' for
                                          symbol_name, exchange in exchange_incompatible_symbols])
        logger.warning(f'Symbols with incompatible exchange will be skipped: {incompatible_details}')

    if not exchange_compatible_symbols:
        raise ValueError(f'No symbols compatible with exchange "{exchange}". '
                         f'Incompatible symbols: {exchange_incompatible_symbols}')

    return exchange_compatible_symbols


def validate_ohlcv_data(data, symbol, interval_label):
    """
    Validate that DataFrame contains required OHLCV columns.

    Checks for the presence of all required OHLCV (Open, High, Low, Close, Volume)
    columns and verifies they contain numeric data types.

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


def detect_gaps(data, interval_label, symbol, known_gaps):
    """
    Detect significant gaps in the datetime index.

    Analyzes the time series data for gaps larger than the threshold (5 days).
    Returns gap metadata and logs warnings for NEW gaps only (not in known_gaps).
    Timestamps are normalized for comparison to prevent false warnings.

    Args:
        data: DataFrame with datetime index
        interval_label: Interval identifier (e.g., '5m', '1h', '1d')
        symbol: Symbol name (e.g., 'ZS1!', 'CL1!')
        known_gaps: Set of (symbol, interval, start_time) tuples for existing gaps

    Returns:
        List of gap dictionaries with keys: symbol, interval, start_time, end_time, duration_days
    """
    gaps = []

    if len(data) < 2:
        return gaps

    sorted_index = data.index.sort_values()

    for i in range(1, len(sorted_index)):
        current_time = sorted_index[i]
        previous_time = sorted_index[i - 1]
        actual_gap = current_time - previous_time

        # Only detect gaps larger than a threshold
        if actual_gap > GAP_DETECTION_THRESHOLD:
            # Calculate duration in days (including fractional days)
            duration_days = actual_gap.days + (actual_gap.seconds / 86400)

            gap_info = {
                'symbol': symbol,
                'interval': interval_label,
                'start_time': previous_time.isoformat(),
                'end_time': current_time.isoformat(),
                'duration_days': round(duration_days, 2)
            }
            gaps.append(gap_info)

            # Normalize timestamp for comparison (remove microseconds, ensure timezone consistency)
            normalized_previous = previous_time.replace(microsecond=0)
            gap_key = (symbol, interval_label, normalized_previous)

            # Only warn if this is a NEW gap
            if gap_key not in known_gaps:
                logger.warning(f'NEW data gap detected in {symbol} {interval_label}: '
                               f'from {previous_time} to {current_time} '
                               f'(duration: {actual_gap})')

    return gaps


def save_gaps_to_yaml(gaps, contract_suffix):
    """
    Save gaps data in YAML format.

    Merges with existing gaps to support multiple DataFetcher instances
    running sequentially. Detects duplicates by comparing start_time.

    Args:
        gaps: List of gap dictionaries from detect_gaps
        contract_suffix: Contract identifier (e.g., '1!')
    """
    if not gaps:
        logger.info('No gaps detected - skipping save')
        return

    # Build file path
    filename = f'historical_data_gaps_{contract_suffix}.yaml'
    file_path = os.path.join('data', filename)

    # Load existing gaps file to merge
    gaps_data = {'gaps': {}, 'meta': {}}
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                gaps_data = yaml.safe_load(f) or {'gaps': {}, 'meta': {}}
        except Exception as e:
            logger.warning(f'Failed to load existing gaps file: {e}. Starting fresh.')
            gaps_data = {'gaps': {}, 'meta': {}}

    # Merge new gaps with existing
    added_count = 0
    skipped_count = 0

    for gap in gaps:
        symbol = gap['symbol']
        interval = gap['interval']

        # Initialize a nested structure if needed
        if symbol not in gaps_data['gaps']:
            gaps_data['gaps'][symbol] = {}
        if interval not in gaps_data['gaps'][symbol]:
            gaps_data['gaps'][symbol][interval] = []

        gap_entry = {
            'start_time': gap['start_time'],
            'end_time': gap['end_time'],
            'duration_days': gap['duration_days']
        }

        # Check for duplicate (same start_time)
        existing_gaps = gaps_data['gaps'][symbol][interval]
        is_duplicate = any(
            g['start_time'] == gap_entry['start_time']
            for g in existing_gaps
        )

        if not is_duplicate:
            existing_gaps.append(gap_entry)
            added_count += 1
        else:
            skipped_count += 1

    # Update metadata
    total_gaps = sum(
        len(interval_gaps)
        for symbol_data in gaps_data['gaps'].values()
        for interval_gaps in symbol_data.values()
    )

    gaps_data['meta'] = {
        'last_updated': datetime.now().replace(microsecond=0).isoformat(),
        'total_gaps': total_gaps,
        'symbols_scanned': sorted(list(gaps_data['gaps'].keys()))
    }

    # Atomic write with a temp file
    temp_path = None
    try:
        # Ensure the data directory exists before creating a temp file
        os.makedirs('data', exist_ok=True)

        # Write to a temp file first
        temp_fd, temp_path = tempfile.mkstemp(dir='data', suffix='.yaml')
        with os.fdopen(temp_fd, 'w') as f:
            yaml.dump(gaps_data, f, default_flow_style=False, sort_keys=False)

        # Rename it to a final path (atomic on POSIX systems)
        os.replace(temp_path, file_path)

        # Log summary based on what actually happened
        if added_count > 0 and skipped_count > 0:
            logger.info(f'Saved {added_count} new gap(s), skipped {skipped_count} duplicate(s) → {total_gaps} total in {file_path}')
        elif added_count > 0:
            logger.info(f'Saved {added_count} new gap(s) → {total_gaps} total in {file_path}')
        else:
            logger.info(f'All {skipped_count} gap(s) already exist → {total_gaps} total in {file_path}')
    except Exception as e:
        logger.error(f'Failed to save gaps file {file_path}: {e}')

        # Clean up the temp file if it exists (wrap in try-except to avoid masking the original exception)
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_error:
                logger.warning(f'Failed to clean up temporary file {temp_path}: {cleanup_error}')

        raise
