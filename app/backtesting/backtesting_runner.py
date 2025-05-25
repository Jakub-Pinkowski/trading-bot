import pandas as pd

from app.backtesting.strategies.rsi import rsi_strategy_trades
from config import HISTORICAL_DATA_DIR

# Define parameters
tested_months = ["1!"]
symbols = ["ZW", "ZS"]
intervals = ["4h"]
strategies = [
    ("RSI", rsi_strategy_trades),
    # ("EMA Crossover", ema_crossover_strategy_trades),
]


# TODO: Add contract switch dates for all symbols
# TODO: Consider closing contracts before contract switch dates instead of rolling it
# TODO: Improve strategies to return not only a list of trades but some more info, summaries etc.
# TODO: Improve the backtesting so I can easily load them up in Google Sheets later for graphs etc.
def run_backtesting():
    for tested_month in tested_months:
        for symbol in symbols:
            for interval in intervals:
                print(f"Processing: Month={tested_month}, Symbol={symbol}, Interval={interval}")

                filepath = f"{HISTORICAL_DATA_DIR}/{tested_month}/{symbol}/{symbol}_{interval}.parquet"
                try:
                    df = pd.read_parquet(filepath)
                except Exception as e:
                    print(f"Failed to read file: {filepath}\nReason: {e}")
                    continue

                for strategy_name, strategy_function in strategies:
                    print(f"Running strategy: {strategy_name}")
                    print(df)
                    trades = strategy_function(df)
