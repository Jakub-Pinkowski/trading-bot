import numpy as np
import pandas as pd


# RSI
def calculate_rsi(prices, period=14):
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


# EMA
def calculate_ema(prices, period=9):
    ema = prices.ewm(span=period, adjust=False).mean()
    ema[:period - 1] = np.nan  # First period points are undefined
    return ema


# ATR
def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = true_range.ewm(span=period, adjust=False).mean()
    atr[:period - 1] = np.nan  # First period points are undefined

    return atr
