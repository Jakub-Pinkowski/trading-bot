import json
import os
from datetime import datetime
from fractions import Fraction
from zoneinfo import ZoneInfo

import pandas as pd

from app.utils.generic_utils import parse_symbol


def fractional_to_decimal(price_str):
    if isinstance(price_str, str) and ' ' in price_str:
        whole, frac = price_str.split(' ', 1)
        return float(whole) + float(Fraction(frac))
    elif isinstance(price_str, str) and '/' in price_str:
        return float(Fraction(price_str))
    else:
        return float(price_str)

# TODO: It's all fucking WRONG!!!
def clean_alerts_data(df):
    # Remove the dummy column if it exists
    if "dummy" in df.columns:
        df = df.drop(columns=["dummy"])

    # Clean symbol
    df["symbol"] = df["symbol"].apply(parse_symbol)

    # Remove consecutive orders on the same side for a given symbol
    df['prev_symbol'] = df['symbol'].shift(1)
    df['prev_side'] = df['side'].shift(1)
    df = df[~((df['symbol'] == df['prev_symbol']) & (df['side'] == df['prev_side']))]
    df = df.drop(columns=['prev_symbol', 'prev_side'])

    return df


def clean_trades_data(trades_df):
    columns_needed = [
        "symbol",
        "side",
        "size",
        "price",
        "trade_time",
        "commission",
        "net_amount",
    ]

    cleaned_df = trades_df[columns_needed].copy()

    # Ensure correct datatypes explicitly
    cleaned_df["trade_time"] = pd.to_datetime(cleaned_df["trade_time"])
    cleaned_df["size"] = cleaned_df["size"].astype(float)
    cleaned_df["commission"] = cleaned_df["commission"].astype(float)
    cleaned_df["net_amount"] = cleaned_df["net_amount"].astype(float)

    # Convert fractional price strings properly
    cleaned_df["price"] = cleaned_df["price"].apply(fractional_to_decimal)

    # Reorder columns
    reordered_columns = [
        "trade_time",
        "symbol",
        "side",
        "size",
        "price",
        "commission",
        "net_amount",
    ]
    cleaned_df = cleaned_df[reordered_columns]

    return cleaned_df


def save_trades_data(trades_json, trades_dir, timezone="Europe/Berlin"):
    trades_by_day = {}

    # Organize trades by extracting the date from each trade's timestamp
    for trade in trades_json:
        trade_datetime = datetime.strptime(trade["trade_time"], "%Y%m%d-%H:%M:%S").astimezone(ZoneInfo(timezone))
        trade_date = trade_datetime.strftime("%Y-%m-%d")
        trades_by_day.setdefault(trade_date, []).append(trade)

    # Save unique trades only
    for date, daily_trades in trades_by_day.items():
        daily_file_path = os.path.join(trades_dir, f'trades_{date}.json')

        # Use execution_id as unique identifier
        unique_trades = {}

        # Load existing trades if file exists to avoid duplicates
        if os.path.exists(daily_file_path):
            with open(daily_file_path, 'r') as file:
                existing_data = json.load(file)
                if isinstance(existing_data, list):
                    for trade in existing_data:
                        unique_trades[trade["execution_id"]] = trade

        # Add current trades (automatically overriding duplicates)
        for trade in daily_trades:
            unique_trades[trade["execution_id"]] = trade

        # Save back to file
        with open(daily_file_path, 'w') as file:
            json.dump(list(unique_trades.values()), file, indent=4)

    print(f"Trades successfully saved in {trades_dir}")
