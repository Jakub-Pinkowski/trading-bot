"""
Data manipulation utilities for backtesting tests.

Provides helper functions for:
- Extracting data subsets
- Finding market scenarios in real data
- Creating synthetic test scenarios
- Modifying data to trigger specific conditions
"""
import pandas as pd


# ==================== Data Modification for Testing ====================

def inject_price_spike(df, index, spike_pct, direction='up'):
    """
    Inject artificial price spike at specific index.

    Useful for testing strategy behavior during extreme movements.
    Modifies high/low/close prices at specified bar.

    Args:
        df: DataFrame with OHLCV data
        index: Index position or timestamp where spike occurs
        spike_pct: Percentage size of spike
        direction: 'up' for spike up, 'down' for spike down

    Returns:
        Modified DataFrame with price spike

    Example:
        # Inject 5% spike upward at bar 100
        modified_df = inject_price_spike(df.copy(), 100, 5.0, 'up')

        # Test stop loss trigger
        strategy.run(modified_df)
        assert strategy.position_closed_by_stop()
    """
    df = df.copy()

    if isinstance(index, (pd.Timestamp, str)):
        index = df.index.get_loc(index)

    base_close = df.iloc[index]['close']
    spike_amount = base_close * (spike_pct / 100)

    if direction == 'up':
        df.iloc[index, df.columns.get_loc('high')] = base_close + spike_amount
        df.iloc[index, df.columns.get_loc('close')] = base_close + (spike_amount * 0.5)
    else:
        df.iloc[index, df.columns.get_loc('low')] = base_close - spike_amount
        df.iloc[index, df.columns.get_loc('close')] = base_close - (spike_amount * 0.5)

    return df


def inject_gap(df, index, gap_pct, direction='up'):
    """
    Inject price gap between bars.

    Creates gap between previous close and next open. Useful for testing
    gap-related logic and ATR true range calculation.

    Args:
        df: DataFrame with OHLCV data
        index: Index where gap occurs (gap is between index-1 and index)
        gap_pct: Percentage size of gap
        direction: 'up' for gap up, 'down' for gap down

    Returns:
        Modified DataFrame with price gap

    Example:
        # Create 3% gap up at bar 50
        modified_df = inject_gap(df.copy(), 50, 3.0, 'up')

        # Test gap handling logic
        strategy.run(modified_df)
        assert strategy.position_adjusted_for_gap()
    """
    df = df.copy()

    if isinstance(index, (pd.Timestamp, str)):
        index = df.index.get_loc(index)

    if index == 0:
        raise ValueError("Cannot inject gap at first bar")

    prev_close = df.iloc[index - 1]['close']
    gap_amount = prev_close * (gap_pct / 100)

    if direction == 'up':
        new_low = prev_close + gap_amount
        # Set low to gap up from previous close
        # This ensures true range captures the gap
        df.iloc[index, df.columns.get_loc('low')] = new_low
        # Adjust high and close proportionally above the new low
        df.iloc[index, df.columns.get_loc('high')] = new_low + 2.0
        df.iloc[index, df.columns.get_loc('close')] = new_low + 1.0
        df.iloc[index, df.columns.get_loc('open')] = new_low + 0.5
    else:
        new_high = prev_close - gap_amount
        # Set high to gap down from previous close
        df.iloc[index, df.columns.get_loc('high')] = new_high
        # Adjust low and close proportionally below the new high
        df.iloc[index, df.columns.get_loc('low')] = new_high - 2.0
        df.iloc[index, df.columns.get_loc('close')] = new_high - 1.0
        df.iloc[index, df.columns.get_loc('open')] = new_high - 0.5

    return df
