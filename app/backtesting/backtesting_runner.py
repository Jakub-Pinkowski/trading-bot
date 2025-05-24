import pandas as pd

from app.backtesting.strategies.ema_crossover import ema_crossover_strategy_trades
from app.backtesting.strategies.rsi import rsi_strategy_trades
from config import HISTORICAL_DATA_DIR


def run_backtesting():
    print("run backtesting")
    tested_month = "1!"
    symbol = "ZW"
    interval = '4h'

    # Load parquet file
    filepath = f"{HISTORICAL_DATA_DIR}/{tested_month}/{symbol}/{symbol}_{interval}.parquet"

    df = pd.read_parquet(filepath)

    trades_rsi = rsi_strategy_trades(df)
    trades_ema = ema_crossover_strategy_trades(df)

    # for trade in trades_rsi:
    #     print(trade)
    #
    # for trade in trades_ema:
    #     print(trade)
