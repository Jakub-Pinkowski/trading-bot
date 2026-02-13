"""
Validation Functions for Data Fetching.

This module provides validation utilities for ensuring data quality
and compatibility when fetching historical market data.
"""
from datetime import timedelta

import pandas as pd

from app.utils.logger import get_logger
from futures_config import get_exchange_for_symbol, SYMBOL_SPECS

logger = get_logger('backtesting/fetching/validators')

# ==================== Constants ====================

GAP_DETECTION_THRESHOLD = timedelta(days=4)  # Gap detection threshold (only log gaps larger than this)


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


def detect_and_log_gaps(data, interval_label, symbol):
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
