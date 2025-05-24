import numpy as np


def calculate_ema(prices, period=9):
    ema = prices.ewm(span=period, adjust=False).mean()
    ema[:period - 1] = np.nan  # First period points are undefined
    return ema
