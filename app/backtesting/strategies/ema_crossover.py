from app.backtesting.indicators import calculate_ema
from app.utils.backtesting_utils.backtesting_utils import format_trades

# Define parameters
EMA_SHORT = 9
EMA_LONG = 21
TRAIL = 0.02  # in %


def add_ema_indicators(df, ema_short=EMA_SHORT, ema_long=EMA_LONG):
    df = df.copy()
    df['ema_short'] = calculate_ema(df['close'], ema_short)
    df['ema_long'] = calculate_ema(df['close'], ema_long)
    return df


def generate_signals(df, ema_short=EMA_SHORT, ema_long=EMA_LONG):
    """
    Signals:
        1: Long entry
       -1: Short entry
        2: Exit long
       -2: Exit short
        0: No action
    """
    df = df.copy()
    df['signal'] = 0

    pos = 0  # +1=long, -1=short, 0=flat
    trail_stop = None
    trail_percentage = TRAIL

    for i in range(1, len(df)):
        ema_short_now = df['ema_short'].iloc[i]
        ema_long_now = df['ema_long'].iloc[i]
        ema_short_prev = df['ema_short'].iloc[i - 1]
        ema_long_prev = df['ema_long'].iloc[i - 1]
        close = df['close'].iloc[i]
        idx = df.index[i]

        buy_signal = ema_short_prev <= ema_long_prev and ema_short_now > ema_long_now and pos == 0
        sell_signal = ema_short_prev >= ema_long_prev and ema_short_now < ema_long_now and pos == 0
        exit_long = False
        exit_short = False

        # Manage trailing stop for long
        if pos == 1:
            trail_stop = max(trail_stop, close * (1 - trail_percentage))
            if close < trail_stop:
                exit_long = True

        # Manage trailing stop for short
        if pos == -1:
            trail_stop = min(trail_stop, close * (1 + trail_percentage))
            if close > trail_stop:
                exit_short = True

        # Signal logic
        if buy_signal:
            df.at[idx, 'signal'] = 1
            pos = 1
            trail_stop = close * (1 - trail_percentage)
        elif sell_signal:
            df.at[idx, 'signal'] = -1
            pos = -1
            trail_stop = close * (1 + trail_percentage)
        elif exit_long:
            df.at[idx, 'signal'] = 2
            pos = 0
            trail_stop = None
        elif exit_short:
            df.at[idx, 'signal'] = -2
            pos = 0
            trail_stop = None
        # else: signal is 0 by default

    return df


def extract_trades(df):
    trades = []
    position = None
    entry_time = None
    entry_price = None

    for idx, row in df.iterrows():
        signal = row['signal']
        price = row['close']

        if signal == 1 and position is None:
            # Long entry
            position = 1
            entry_time = idx
            entry_price = price
        elif signal == -1 and position is None:
            # Short entry
            position = -1
            entry_time = idx
            entry_price = price
        elif signal == 2 and position == 1:
            # Exit long
            pnl = price - entry_price
            trades.append({
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_time": idx,
                "exit_price": price,
                "side": "long",
                "pnl": pnl,
            })
            position = None
            entry_time = None
            entry_price = None
        elif signal == -2 and position == -1:
            # Exit short
            pnl = entry_price - price
            trades.append({
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_time": idx,
                "exit_price": price,
                "side": "short",
                "pnl": pnl,
            })
            position = None
            entry_time = None
            entry_price = None

    trades = format_trades(trades)

    return trades


def ema_crossover_strategy_trades(df, ema_short=EMA_SHORT, ema_long=EMA_LONG):
    df = add_ema_indicators(df, ema_short, ema_long)
    df = generate_signals(df, ema_short, ema_long)
    trades = extract_trades(df)
    return trades
