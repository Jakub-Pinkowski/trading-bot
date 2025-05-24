from app.backtesting.indicators.rsi import calculate_rsi

# Define parameters
RSI_PERIOD = 14
LOWER = 30
UPPER = 70


def format_trades(trades):
    formatted_trades = []
    for trade in trades:
        formatted_trade = {
            "entry_time": str(trade["entry_time"]),
            "entry_price": float(trade["entry_price"]),
            "exit_time": str(trade["exit_time"]),
            "exit_price": float(trade["exit_price"]),
            "side": str(trade["side"]),
            "pnl": float(trade["pnl"]),
        }
        formatted_trades.append(formatted_trade)

    return formatted_trades


def add_rsi_indicator(df, rsi_period=RSI_PERIOD):
    df = df.copy()
    df['rsi'] = calculate_rsi(df["close"], period=rsi_period)
    return df


def generate_signals(df, lower=LOWER, upper=UPPER):
    df = df.copy()
    df['signal'] = 0
    prev_rsi = df['rsi'].shift(1)

    # Buy signal: RSI crosses below a lower threshold
    df.loc[(prev_rsi > lower) & (df['rsi'] <= lower), 'signal'] = 1

    # Sell signal: RSI crosses above an upper threshold
    df.loc[(prev_rsi < upper) & (df['rsi'] >= upper), 'signal'] = -1

    return df


def extract_trades_from_signals(df):
    trades = []
    position = None  # +1 for long, -1 for short
    entry_time = None
    entry_price = None

    for i, row in df.iterrows():
        signal = row['signal']
        price = row['close']
        idx = row.name

        flip = None
        if signal == 1 and position != 1:
            flip = 1  # Long
        elif signal == -1 and position != -1:
            flip = -1  # Short

        if flip is not None:
            # Close existing position
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
            # Open new position
            position = flip
            entry_time = idx
            entry_price = price

    trades = format_trades(trades)

    return trades


def rsi_strategy_trades(df, rsi_period=RSI_PERIOD, lower=LOWER, upper=UPPER):
    """
    1. Receives a DataFrame with OHLCV data (index should be datetime).
    2. Adds RSI as an indicator to the DataFrame.
    3. Generates buy/sell signals using RSI crossovers.
    4. Extracts the resulting trades.
    """
    df = add_rsi_indicator(df, rsi_period)
    df = generate_signals(df, lower, upper)
    trades = extract_trades_from_signals(df)
    return trades
