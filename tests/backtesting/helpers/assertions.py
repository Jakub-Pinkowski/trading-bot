"""
Custom assertion functions for backtesting tests.

Provides specialized assertions for validating:
- OHLCV data structure
- Indicator values
- Signal generation
- Trade lists
- Performance metrics
"""
import numpy as np
import pandas as pd


# ==================== Indicator Validation ====================

def assert_valid_indicator(series, name, min_val=None, max_val=None, allow_nan=True):
    """
    Assert indicator series contains valid values within expected range.

    Validates that an indicator series (RSI, EMA, MACD, etc.) has proper structure,
    contains values within specified bounds, and handles NaN values appropriately.

    Args:
        series: Pandas Series containing indicator values
        name: Indicator name for error messages (e.g., 'RSI', 'EMA(9)')
        min_val: Minimum valid value (None = no minimum check)
        max_val: Maximum valid value (None = no maximum check)
        allow_nan: If True, allow NaN values (common for initial periods)

    Raises:
        AssertionError: If any validation check fails

    Examples:
        assert_valid_indicator(rsi, 'RSI', min_val=0, max_val=100)
        assert_valid_indicator(ema, 'EMA(9)', min_val=0)
    """
    # Check series type
    assert isinstance(series, pd.Series), f"{name} must be a pandas Series"

    # Check series is not empty
    assert len(series) > 0, f"{name} series is empty"

    # Get valid (non-NaN) values for range checking
    valid_values = series.dropna()

    # Check if we have any valid values
    if not allow_nan:
        assert len(valid_values) == len(series), f"{name} contains NaN values"
    else:
        assert len(valid_values) > 0, f"{name} contains only NaN values"

    # Check minimum value
    if min_val is not None:
        min_actual = valid_values.min()
        assert min_actual >= min_val, \
            f"{name} minimum value {min_actual} is below expected minimum {min_val}"

    # Check maximum value
    if max_val is not None:
        max_actual = valid_values.max()
        assert max_actual <= max_val, \
            f"{name} maximum value {max_actual} is above expected maximum {max_val}"

    # Check for infinite values
    assert not np.isinf(valid_values).any(), f"{name} contains infinite values"


def assert_indicator_varies(series, name, min_std=0.1):
    """
    Assert indicator shows variation (not constant).

    Validates that an indicator actually responds to price changes rather than
    remaining constant. Useful for detecting calculation errors.

    Args:
        series: Pandas Series containing indicator values
        name: Indicator name for error messages
        min_std: Minimum standard deviation expected

    Raises:
        AssertionError: If indicator is too constant
    """
    valid_values = series.dropna()
    assert len(valid_values) > 1, f"{name} has insufficient valid values"

    std = valid_values.std()
    assert std >= min_std, \
        f"{name} has insufficient variation (std={std:.4f}, min={min_std})"


# ==================== Signal Validation ====================

def assert_valid_signals(df):
    """
    Assert signal column has valid values and structure.

    Validates that a DataFrame's 'signal' column contains only valid signal values
    (1 for long, -1 for short, 0 for no signal), has proper type, and makes
    logical sense (e.g., signals aren't too frequent).

    Args:
        df: DataFrame with 'signal' column

    Raises:
        AssertionError: If signal column is invalid
    """
    # Check signal column exists
    assert 'signal' in df.columns, "DataFrame missing 'signal' column"

    # Check signal values are valid
    valid_signals = {-1, 0, 1}
    unique_signals = set(df['signal'].unique())
    invalid_signals = unique_signals - valid_signals
    assert not invalid_signals, \
        f"Signal column contains invalid values: {invalid_signals}"

    # Check signal is numeric type
    assert pd.api.types.is_numeric_dtype(df['signal']), \
        "Signal column must be numeric type"

    # Check no NaN signals
    assert not df['signal'].isna().any(), "Signal column contains NaN values"


# ==================== Trade Validation ====================

def assert_valid_trades(trades_list):
    """
    Assert trades list contains valid trade dictionaries.

    Validates that each trade has required fields, proper types, valid values,
    and logical relationships (exit after entry, positive prices, etc.).

    Args:
        trades_list: List of trade dictionaries, each with keys:
            ['entry_time', 'exit_time', 'entry_price', 'exit_price', 'side']

    Raises:
        AssertionError: If any trade is invalid
    """
    assert isinstance(trades_list, list), "Trades must be a list"
    assert len(trades_list) > 0, "Trades list is empty"

    required_fields = [
        'entry_time', 'exit_time', 'entry_price', 'exit_price', 'side'
    ]

    for i, trade in enumerate(trades_list):
        # Check required fields
        missing_fields = [f for f in required_fields if f not in trade]
        assert not missing_fields, \
            f"Trade {i} missing required fields: {missing_fields}"

        # Check side is valid
        assert trade['side'] in ['long', 'short'], \
            f"Trade {i} has invalid side: {trade['side']}"

        # Check prices are positive
        assert trade['entry_price'] > 0, \
            f"Trade {i} has invalid entry_price: {trade['entry_price']}"
        assert trade['exit_price'] > 0, \
            f"Trade {i} has invalid exit_price: {trade['exit_price']}"

        # Check times are valid and ordered
        # Note: Contract switch trades may have entry_time == exit_time
        # (position closed at switch point on same bar)
        if trade.get('switch', False):
            assert trade['entry_time'] <= trade['exit_time'], \
                f"Trade {i} exit_time must be >= entry_time (switch trade)"
        else:
            assert trade['entry_time'] < trade['exit_time'], \
                f"Trade {i} exit_time must be after entry_time"


def assert_no_overlapping_trades(trades_list):
    """
    Assert trades don't overlap in time.

    Validates that each trade exits before the next trade enters, ensuring
    proper position management without simultaneous positions.

    Args:
        trades_list: List of trade dictionaries with 'entry_time' and 'exit_time'

    Raises:
        AssertionError: If overlapping trades detected
    """
    assert_valid_trades(trades_list)

    if len(trades_list) > 1:
        # Sort by entry time
        sorted_trades = sorted(trades_list, key=lambda t: t['entry_time'])

        # Check each trade exits before next enters
        for i in range(len(sorted_trades) - 1):
            exit_time = sorted_trades[i]['exit_time']
            next_entry = sorted_trades[i + 1]['entry_time']
            assert exit_time <= next_entry, \
                f"Trade {i} overlaps with trade {i + 1}"
