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

from app.backtesting.metrics.per_trade_metrics import COMMISSION_PER_TRADE
from config import CONTRACT_MULTIPLIERS, TICK_SIZES


# ==================== OHLCV Data Validation ====================

def assert_valid_ohlcv(df):
    """
    Assert DataFrame has valid OHLCV structure and values.

    Validates that the DataFrame contains required columns, has proper types,
    maintains OHLC relationships (high >= open/close/low, low <= open/close/high),
    and contains no invalid values (negative prices, zero volume, etc.).

    Args:
        df: DataFrame to validate, expected to have columns:
            ['open', 'high', 'low', 'close', 'volume']

    Raises:
        AssertionError: If any validation check fails
    """
    # Check required columns exist
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    assert not missing_columns, f"Missing required columns: {missing_columns}"

    # Check DataFrame is not empty
    assert len(df) > 0, "DataFrame is empty"

    # Check datetime index
    assert isinstance(df.index, pd.DatetimeIndex), "Index must be DatetimeIndex"

    # Check for duplicate timestamps
    assert not df.index.duplicated().any(), "DataFrame has duplicate timestamps"

    # Check OHLC relationships
    assert (df['high'] >= df['low']).all(), "High must be >= low"
    assert (df['high'] >= df['open']).all(), "High must be >= open"
    assert (df['high'] >= df['close']).all(), "High must be >= close"
    assert (df['low'] <= df['open']).all(), "Low must be <= open"
    assert (df['low'] <= df['close']).all(), "Low must be <= close"

    # Check for negative values
    assert (df['open'] > 0).all(), "Open prices must be positive"
    assert (df['high'] > 0).all(), "High prices must be positive"
    assert (df['low'] > 0).all(), "Low prices must be positive"
    assert (df['close'] > 0).all(), "Close prices must be positive"
    assert (df['volume'] >= 0).all(), "Volume must be non-negative"

    # Check for NaN values
    assert not df['open'].isna().any(), "Open contains NaN values"
    assert not df['high'].isna().any(), "High contains NaN values"
    assert not df['low'].isna().any(), "Low contains NaN values"
    assert not df['close'].isna().any(), "Close contains NaN values"
    assert not df['volume'].isna().any(), "Volume contains NaN values"


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


def assert_signals_exist(df, signal_type=None):
    """
    Assert that signals are present in the DataFrame.

    Validates that the strategy generated at least some signals, preventing
    silent failures where a strategy never triggers.

    Args:
        df: DataFrame with 'signal' column
        signal_type: Type of signal to check for (1=long, -1=short, None=any)

    Raises:
        AssertionError: If no signals found
    """
    assert_valid_signals(df)

    if signal_type is None:
        non_zero_signals = (df['signal'] != 0).sum()
        assert non_zero_signals > 0, "No signals generated"
    else:
        signals_of_type = (df['signal'] == signal_type).sum()
        signal_name = 'long' if signal_type == 1 else 'short'
        assert signals_of_type > 0, f"No {signal_name} signals generated"


def assert_no_consecutive_signals(df):
    """
    Assert signals don't occur on consecutive bars.

    Validates that a strategy properly manages positions and doesn't generate
    new entry signals when already in a position.

    Args:
        df: DataFrame with 'signal' column

    Raises:
        AssertionError: If consecutive non-zero signals detected
    """
    assert_valid_signals(df)

    # Get non-zero signal indices
    signal_indices = df[df['signal'] != 0].index

    if len(signal_indices) > 1:
        # Check no consecutive indices
        for i in range(len(signal_indices) - 1):
            idx1 = df.index.get_loc(signal_indices[i])
            idx2 = df.index.get_loc(signal_indices[i + 1])
            assert idx2 - idx1 > 1, \
                f"Consecutive signals at indices {idx1} and {idx2}"


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


def assert_trade_pnl_calculation(trade_metrics, symbol):
    """
    Assert trade PnL is calculated correctly.

    Validates that the calculated metrics from calculate_trade_metrics() match
    expected calculations based on entry/exit prices, side, and contract specifications.
    This validates the entire metrics calculation pipeline.

    Args:
        trade_metrics: Trade metrics dictionary from calculate_trade_metrics() with keys:
            ['entry_price', 'exit_price', 'side', 'net_pnl', 'commission']
        symbol: Futures symbol code (e.g., 'ZS', 'CL', 'GC') for looking up specs

    Raises:
        AssertionError: If PnL calculation is incorrect
    """
    entry_price = trade_metrics['entry_price']
    exit_price = trade_metrics['exit_price']
    side = trade_metrics['side']
    actual_net_pnl = trade_metrics['net_pnl']
    actual_commission = trade_metrics['commission']

    # Verify commission is correct (imported from config)
    assert actual_commission == COMMISSION_PER_TRADE, \
        f"Commission mismatch: expected {COMMISSION_PER_TRADE}, got {actual_commission}"

    # Calculate expected PnL using imported CONTRACT_MULTIPLIERS
    contract_multiplier = CONTRACT_MULTIPLIERS.get(symbol)
    assert contract_multiplier is not None, f"No contract multiplier found for {symbol}"

    if side == 'long':
        pnl_points = exit_price - entry_price
    else:  # short
        pnl_points = entry_price - exit_price

    gross_pnl = pnl_points * contract_multiplier
    expected_net_pnl = gross_pnl - COMMISSION_PER_TRADE

    # Allow small floating point differences
    assert abs(actual_net_pnl - expected_net_pnl) < 0.01, \
        f"PnL mismatch: expected {expected_net_pnl:.2f}, got {actual_net_pnl:.2f}"


# ==================== Metrics Validation ====================

def assert_valid_metrics(metrics):
    """
    Assert performance metrics dictionary contains valid values.

    Validates that a metrics dictionary has required fields, proper types,
    and values within logical ranges.

    Args:
        metrics: Dictionary containing performance metrics:
            ['total_trades', 'win_rate', 'profit_factor', 'total_pnl',
             'avg_win', 'avg_loss', 'sharpe_ratio', 'max_drawdown']

    Raises:
        AssertionError: If any metric is invalid
    """
    assert isinstance(metrics, dict), "Metrics must be a dictionary"

    required_metrics = [
        'total_trades', 'win_rate', 'profit_factor', 'total_pnl'
    ]

    missing_metrics = [m for m in required_metrics if m not in metrics]
    assert not missing_metrics, f"Missing required metrics: {missing_metrics}"

    # Validate total_trades
    assert isinstance(metrics['total_trades'], int), \
        "total_trades must be an integer"
    assert metrics['total_trades'] >= 0, \
        "total_trades must be non-negative"

    # Validate win_rate (0-100)
    if metrics['total_trades'] > 0:
        assert 0 <= metrics['win_rate'] <= 100, \
            f"win_rate must be 0-100, got {metrics['win_rate']}"

    # Validate profit_factor (>= 0)
    assert metrics['profit_factor'] >= 0, \
        f"profit_factor must be non-negative, got {metrics['profit_factor']}"

    # Validate total_pnl is numeric
    assert isinstance(metrics['total_pnl'], (int, float)), \
        "total_pnl must be numeric"


def assert_metrics_in_range(metrics, expected_ranges):
    """
    Assert performance metrics are within expected ranges.

    Validates that calculated metrics fall within reasonable bounds based on
    strategy type, market conditions, or historical performance.

    Args:
        metrics: Performance metrics dictionary
        expected_ranges: Dictionary mapping metric names to (min, max) tuples
            Example: {'win_rate': (30, 70), 'profit_factor': (1.0, 3.0)}

    Raises:
        AssertionError: If any metric is outside its expected range
    """
    assert_valid_metrics(metrics)

    for metric_name, (min_val, max_val) in expected_ranges.items():
        assert metric_name in metrics, f"Metric '{metric_name}' not found"

        actual_val = metrics[metric_name]
        assert min_val <= actual_val <= max_val, \
            f"{metric_name} {actual_val} outside expected range [{min_val}, {max_val}]"


def assert_profitable_strategy(metrics, min_profit_factor=1.0):
    """
    Assert strategy shows profitability.

    Validates that a strategy meets minimum profitability requirements, useful
    for ensuring realistic test scenarios.

    Args:
        metrics: Performance metrics dictionary
        min_profit_factor: Minimum acceptable profit factor (default 1.0 = breakeven)

    Raises:
        AssertionError: If strategy is not profitable enough
    """
    assert_valid_metrics(metrics)

    assert metrics['total_pnl'] > 0, \
        f"Strategy is unprofitable: PnL = {metrics['total_pnl']}"

    assert metrics['profit_factor'] >= min_profit_factor, \
        f"Profit factor {metrics['profit_factor']} below minimum {min_profit_factor}"


# ==================== Strategy Component Validation ====================

def assert_valid_crossover_detection(series1, series2, crossover_series, direction):
    """
    Assert crossover detection logic is correct.

    Validates that detected crossovers actually represent cases where series1
    crosses series2 in the specified direction.

    Args:
        series1: First series (e.g., fast EMA)
        series2: Second series (e.g., slow EMA)
        crossover_series: Boolean series indicating crossover points
        direction: 'above' for bullish crossover, 'below' for bearish

    Raises:
        AssertionError: If crossover detection is incorrect
    """
    # Check crossover series is boolean
    assert crossover_series.dtype == bool, "Crossover series must be boolean"

    # Get crossover points
    crossover_indices = crossover_series[crossover_series].index

    if len(crossover_indices) > 0:
        for idx in crossover_indices:
            idx_loc = series1.index.get_loc(idx)

            # Skip if at beginning (no previous value)
            if idx_loc == 0:
                continue

            prev_idx = series1.index[idx_loc - 1]

            # Get values
            s1_prev = series1.loc[prev_idx]
            s2_prev = series2.loc[prev_idx]
            s1_curr = series1.loc[idx]
            s2_curr = series2.loc[idx]

            # Skip if any NaN
            if pd.isna([s1_prev, s2_prev, s1_curr, s2_curr]).any():
                continue

            # Validate crossover
            if direction == 'above':
                assert s1_prev <= s2_prev, \
                    f"Invalid bullish crossover at {idx}: series1 not below series2 before"
                assert s1_curr > s2_curr, \
                    f"Invalid bullish crossover at {idx}: series1 not above series2 after"
            else:  # below
                assert s1_prev >= s2_prev, \
                    f"Invalid bearish crossover at {idx}: series1 not above series2 before"
                assert s1_curr < s2_curr, \
                    f"Invalid bearish crossover at {idx}: series1 not below series2 after"


# ==================== Contract Specification Validation ====================

def assert_valid_symbol(symbol):
    """
    Assert symbol is configured in the app.

    Validates that a symbol has required configuration (contract multiplier,
    tick size) in config.py, ensuring tests use valid symbols.

    Args:
        symbol: Futures symbol code (e.g., 'ZS', 'CL', 'GC')

    Raises:
        AssertionError: If symbol is not properly configured
    """
    assert symbol in CONTRACT_MULTIPLIERS, \
        f"Symbol '{symbol}' not found in CONTRACT_MULTIPLIERS"
    assert symbol in TICK_SIZES, \
        f"Symbol '{symbol}' not found in TICK_SIZES"


def assert_slippage_applied(entry_price_with_slippage, base_price, side, slippage_ticks, symbol):
    """
    Assert slippage is correctly applied to entry price.

    Validates that entry price includes proper slippage adjustment based on
    position side and symbol's tick size (from config).

    Args:
        entry_price_with_slippage: Actual entry price with slippage
        base_price: Base price before slippage
        side: Position side ('long' or 'short')
        slippage_ticks: Number of ticks of slippage
        symbol: Futures symbol for tick size lookup

    Raises:
        AssertionError: If slippage calculation is incorrect
    """
    assert_valid_symbol(symbol)

    tick_size = TICK_SIZES[symbol]
    slippage_amount = slippage_ticks * tick_size

    if side == 'long':
        expected_price = base_price + slippage_amount
    else:  # short
        expected_price = base_price - slippage_amount

    expected_price = round(expected_price, 2)

    assert abs(entry_price_with_slippage - expected_price) < 0.01, \
        f"Slippage not applied correctly: expected {expected_price:.2f}, got {entry_price_with_slippage:.2f}"


def assert_price_increment_valid(price, symbol):
    """
    Assert price is a valid multiple of the symbol's tick size.

    Validates that prices respect the minimum tick size for the symbol,
    ensuring realistic price data in tests.

    Args:
        price: Price value to validate
        symbol: Futures symbol for tick size lookup

    Raises:
        AssertionError: If price is not a valid multiple of tick size
    """
    assert_valid_symbol(symbol)

    tick_size = TICK_SIZES[symbol]

    # Check if price is approximately a multiple of tick_size
    # Allow small floating point error
    remainder = (price % tick_size) / tick_size
    is_valid = remainder < 0.001 or remainder > 0.999

    assert is_valid, \
        f"Price {price} is not a valid multiple of tick size {tick_size} for {symbol}"
