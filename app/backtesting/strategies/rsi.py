import pandas as pd

from app.backtesting.indicators import calculate_rsi
from app.utils.backtesting_utils.backtesting_utils import format_trades

# Define parameters
RSI_PERIOD = 14
LOWER = 30
UPPER = 70


def add_rsi_indicator(df, rsi_period=RSI_PERIOD):
    df = df.copy()
    df['rsi'] = calculate_rsi(df["close"], period=rsi_period)
    return df


def generate_signals(df, lower=LOWER, upper=UPPER):
    """
    Signals:
        1: Long entry
       -1: Short entry
        0: No action
    """
    df = df.copy()
    df['signal'] = 0
    prev_rsi = df['rsi'].shift(1)

    # Buy signal: RSI crosses below a lower threshold
    df.loc[(prev_rsi > lower) & (df['rsi'] <= lower), 'signal'] = 1

    # Sell signal: RSI crosses above an upper threshold
    df.loc[(prev_rsi < upper) & (df['rsi'] >= upper), 'signal'] = -1

    return df


# TODO: Outsource repetitive logic
def extract_trades(df, switch_dates, rollover):
    trades = []
    position = None
    entry_time = None
    entry_price = None
    entry_rsi = None  # Store entry RSI
    next_switch_idx = 0
    next_switch = switch_dates[next_switch_idx] if switch_dates else None
    must_reopen = None  # Track if we need to reopen after a roll

    prev_row = None
    skip_signal_this_bar = False
    queued_signal = None  # Holds a signal to execute on the next bar
    queued_signal_row = None  # Holds the row at which the signal was queued

    for idx, row in df.iterrows():
        current_time = pd.to_datetime(idx)
        signal = row['signal']
        price_open = row['open']
        price_close = row['close']
        rsi = row.get('rsi', None)

        # Handle contract switches
        while next_switch and current_time >= next_switch:
            # On rollover: close at the price of *last bar before switch* (prev_row)
            if position is not None and entry_time is not None and prev_row is not None:
                exit_price = prev_row['open']
                exit_rsi = prev_row.get('rsi', None)

                pnl = (exit_price - entry_price) * position
                trades.append({
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "entry_rsi": entry_rsi,
                    "exit_time": current_time,
                    "exit_price": exit_price,
                    "exit_rsi": exit_rsi,
                    "side": "long" if position == 1 else "short",
                    "pnl": pnl,
                    "switch": True,
                })
                if rollover:
                    must_reopen = position  # Mark to reopen with the same direction
                    skip_signal_this_bar = True  # Skip signal for this bar, only one trade per bar allowed
                else:
                    must_reopen = None  # Do NOT reopen if ROLLOVER is False
                entry_time = None
                entry_price = None
                entry_rsi = None
                position = None
            next_switch_idx += 1
            next_switch = switch_dates[next_switch_idx] if next_switch_idx < len(switch_dates) else None

        # Open a new position on the next iteration (only if rollover enabled)
        if must_reopen is not None and position is None:
            if rollover:
                position = must_reopen
                entry_time = idx
                entry_price = price_open  # CHANGED: open new trade at open
                entry_rsi = rsi
            must_reopen = None

        if skip_signal_this_bar:
            skip_signal_this_bar = False  # skip *this* bar only
            prev_row = row
            continue

        # Execute queued signal from the previous bar
        if queued_signal is not None:
            flip = None
            if queued_signal == 1 and position != 1:
                flip = 1
            elif queued_signal == -1 and position != -1:
                flip = -1

            if flip is not None:
                # Close if currently in position
                if position is not None and entry_time is not None:
                    exit_price = price_open
                    exit_rsi = rsi
                    side = position
                    pnl = (exit_price - entry_price) * side
                    trades.append({
                        "entry_time": entry_time,
                        "entry_price": entry_price,
                        "entry_rsi": entry_rsi,
                        "exit_time": idx,
                        "exit_price": exit_price,
                        "exit_rsi": exit_rsi,
                        "side": "long" if side == 1 else "short",
                        "pnl": pnl,
                    })
                # Open a new position at this (current) bar
                position = flip
                entry_time = idx
                entry_price = price_open
                entry_rsi = rsi

            # Reset after using
            queued_signal = None
            queued_signal_row = None

        # Set/overwrite queued_signal for next bar execution
        if signal != 0:
            queued_signal = signal

        prev_row = row

    for trade in trades:
        print(trade)

    trades = format_trades(trades)

    return trades


def compute_summary(trades):
    total_pnl = sum(trade['pnl'] for trade in trades)
    summary = {
        "num_trades": len(trades),
        "total_pnl": total_pnl
    }
    return summary


def rsi_strategy_trades(df, switch_dates, rollover, rsi_period=RSI_PERIOD, lower=LOWER, upper=UPPER):
    df = add_rsi_indicator(df, rsi_period)
    df = generate_signals(df, lower, upper)
    trades = extract_trades(df, switch_dates, rollover)
    summary = compute_summary(trades)
    return trades
