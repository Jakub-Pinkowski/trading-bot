import pandas as pd

from app.backtesting.indicators import calculate_rsi
from app.utils.backtesting_utils.backtesting_utils import format_trades

# Define parameters
RSI_PERIOD = 14
LOWER = 30
UPPER = 70


# TODO: Add some more options later. For example rollover vs close the position


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


def extract_trades(df, switch_dates):
    print(switch_dates)
    trades = []
    position = None
    entry_time = None
    entry_price = None
    next_switch_idx = 0
    next_switch = switch_dates[next_switch_idx] if switch_dates else None
    rows = list(df.iterrows())
    must_reopen = None  # Track if we need to reopen after a roll

    for idx, row in df.iterrows():
        current_time = pd.to_datetime(idx)
        signal = row['signal']
        price = row['close']

        # Roll logic: close at switch, and defer re-open to the next row
        while next_switch and current_time >= next_switch:
            if position is not None and entry_time is not None:
                exit_price = price
                pnl = (exit_price - entry_price) * position
                trades.append({
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "exit_time": current_time,
                    "exit_price": exit_price,
                    "side": "long" if position == 1 else "short",
                    "pnl": pnl,
                    "rolled": True,
                })
                must_reopen = position  # Mark to reopen with the same direction
                entry_time = None
                entry_price = None
                position = None
            next_switch_idx += 1
            next_switch = switch_dates[next_switch_idx] if next_switch_idx < len(switch_dates) else None

        # Open a new position on the next iteration
        if must_reopen is not None and position is None:
            print(must_reopen, current_time)
            position = must_reopen
            entry_time = idx
            entry_price = price
            must_reopen = None

        flip = None
        if signal == 1 and position != 1:
            flip = 1  # Long
        elif signal == -1 and position != -1:
            flip = -1  # Short

        if flip is not None:
            # Close the existing position
            if position is not None and entry_time is not None:
                exit_price = price
                side = position
                pnl = (exit_price - entry_price) * side
                trades.append({
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "exit_time": idx,
                    "exit_price": exit_price,
                    "side": "long" if side == 1 else "short",
                    "pnl": pnl,
                })
            # Open a new position
            position = flip
            entry_time = idx
            entry_price = price

    trades = format_trades(trades)

    for trade in trades:
        print(trade)

    return trades


# TODO: Add way more data
def compute_summary(trades):
    total_pnl = sum(trade['pnl'] for trade in trades)
    summary = {
        "num_trades": len(trades),
        "total_pnl": total_pnl
    }
    print(f"Summary: {summary}")
    return summary


def rsi_strategy_trades(df, switch_dates, rsi_period=RSI_PERIOD, lower=LOWER, upper=UPPER):
    df = add_rsi_indicator(df, rsi_period)
    df = generate_signals(df, lower, upper)
    trades = extract_trades(df, switch_dates)
    summary = compute_summary(trades)
    return trades
