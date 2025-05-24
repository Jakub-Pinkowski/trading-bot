import pandas as pd

from app.backtesting.indicators.rsi import calculate_rsi


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
        rsi_period: int = 14,
        lower: int = 30,
        upper: int = 70,
        price_col: str = "close"
):
    rsi = calculate_rsi(df[price_col], period=rsi_period)
    trades = []
    position = None  # +1 for long, -1 for short
    entry_idx = None
    entry_price = None

    prev_rsi = rsi.iloc[0]
    for i in range(1, len(df)):
        current_rsi = rsi.iloc[i]
        price = df[price_col].iloc[i]
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
            if position is not None and entry_idx is not None:
                exit_price = price
                side = position  # The side we *held* until now
                pnl = (exit_price - entry_price) * side
                trades.append({
                    "entry_idx": entry_idx,
                    "entry_time": df.index[entry_idx],
                    "entry_price": entry_price,
                    "exit_idx": i,
                    "exit_time": idx,
                    "exit_price": exit_price,
                    "side": "long" if side == 1 else "short",
                    "pnl": pnl,
                })
            # Open a new position after flip
            position = flip
            entry_idx = i
            entry_price = price

        prev_rsi = current_rsi

    # Close open trade at the end of data
    if position is not None and entry_idx is not None and entry_idx < len(df) - 1:
        price = df[price_col].iloc[-1]
        idx = df.index[-1]
        side = position
        pnl = (price - entry_price) * side
        trades.append({
            "entry_idx": entry_idx,
            "entry_time": df.index[entry_idx],
            "entry_price": entry_price,
            "exit_idx": len(df) - 1,
            "exit_time": idx,
            "exit_price": price,
            "side": "long" if side == 1 else "short",
            "pnl": pnl,
        })

    trades = format_trades()

    return trades
