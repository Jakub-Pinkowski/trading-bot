import numpy as np
import pandas as pd


# NOTE: RSI works fine
# TODO: I migth want to instead keep on adding new indicators like EMA, RSI etc. to my dt and then apply strategy
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Initialize the recursive values with the rolling mean at index `period`
    for i in range(period, len(prices)):
        if i == period:
            # For the very first point, leave as is (already set by rolling)
            continue
        # Recursive calculation, starting after first rolling value
        avg_gain.iat[i] = (avg_gain.iat[i - 1] * (period - 1) + gain.iat[i]) / period
        avg_loss.iat[i] = (avg_loss.iat[i - 1] * (period - 1) + loss.iat[i]) / period

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan  # First period points are undefined
    return rsi


def format_rsi_trades(trades):
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
        if position != 1 and prev_rsi > lower and current_rsi <= lower:
            print(f"RSI crossed below {lower} (OVERSOLD, BUY) on {idx} (RSI={current_rsi:.2f})")

            flip = 1  # Go long

        # Sell entry: RSI crosses ABOVE overbought level
        elif position != -1 and prev_rsi < upper and current_rsi >= upper:
            print(f"RSI crossed above {upper} (OVERBOUGHT, SELL) on {idx} (RSI={current_rsi:.2f})")

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
            # Open new position after flip
            position = flip
            entry_idx = i
            entry_price = price

        prev_rsi = current_rsi

    # Close open trade at end of data
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

    return format_rsi_trades(trades)
