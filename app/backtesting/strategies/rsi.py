import pandas as pd

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


def rsi_strategy_trades(
        df: pd.DataFrame,
        rsi_period: int = RSI_PERIOD,
        lower: int = LOWER,
        upper: int = UPPER,
):
    rsi = calculate_rsi(df["close"], period=rsi_period)
    trades = []
    position = None  # +1 for long, -1 for short
    entry_time = None
    entry_price = None

    prev_rsi = rsi.iloc[0]
    for i in range(1, len(df)):
        current_rsi = rsi.iloc[i]
        price = df["close"].iloc[i]
        idx = df.index[i]

        # Ignore until RSI is available
        if pd.isna(prev_rsi) or pd.isna(current_rsi):
            prev_rsi = current_rsi
            continue

        flip = None

        # Buy entry: RSI crosses BELOW oversold level
        if position != 1 and prev_rsi > lower >= current_rsi:
            flip = 1  # Go long

        # Sell entry: RSI crosses ABOVE overbought level
        elif position != -1 and prev_rsi < upper <= current_rsi:
            flip = -1  # Go short

        if flip is not None:
            # Close existing, open new flip
            if position is not None and entry_time is not None:
                exit_price = price
                side = position  # The side we *held* until now
                pnl = (exit_price - entry_price) * side
                trades.append({
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "exit_time": idx,
                    "exit_price": exit_price,
                    "side": "long" if side == 1 else "short",
                    "pnl": pnl,
                })
            # Open a new position after flip
            position = flip
            entry_time = idx
            entry_price = price

        prev_rsi = current_rsi

    trades = format_trades(trades)

    return trades
