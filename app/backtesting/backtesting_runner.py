import pandas as pd
import yaml

from app.backtesting.strategies.rsi import RSIStrategy
from app.utils.logger import get_logger
from config import HISTORICAL_DATA_DIR, SWITCH_DATES_FILE_PATH

logger = get_logger()

# Define parameters
tested_months = ["1!"]
symbols = ['ZL']
intervals = ["4h"]

# Strategy parameters
rollover = False

# Strategies setup
strategies = [("RSI", RSIStrategy(rollover=rollover))]

with open(SWITCH_DATES_FILE_PATH) as f:
    switch_dates_dict = yaml.safe_load(f)


# TODO: Build separate logic for metrics/summaries
# TODO: Add contract switch dates for missing symbols
# TODO: Improve the backtesting so I can easily load them up in Google Sheets later for graphs etc.
def run_backtesting():
    for tested_month in tested_months:
        for symbol in symbols:
            switch_dates = switch_dates_dict.get(symbol, [])
            switch_dates = [pd.to_datetime(dt) for dt in switch_dates]

            for interval in intervals:
                logger.info(f"Processing: Month={tested_month}, Symbol={symbol}, Interval={interval}")

                filepath = f"{HISTORICAL_DATA_DIR}/{tested_month}/{symbol}/{symbol}_{interval}.parquet"
                try:
                    df = pd.read_parquet(filepath)
                except Exception as e:
                    logger.error(f"Failed to read file: {filepath}\nReason: {e}")
                    continue

                for strategy_name, strategy_instance in strategies:
                    trades = strategy_instance.run(df, switch_dates)

                    for trade in trades:
                        print(trade)
