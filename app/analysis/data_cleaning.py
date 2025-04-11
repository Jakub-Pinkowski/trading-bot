from fractions import Fraction

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


def pre_clean_tw_data(df):
    # Parse 'Time' column into a datetime object
    df['timestamp'] = pd.to_datetime(df['Time'], errors='coerce')

    # Format the timestamps
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Select only the relevant columns
    df = df[['timestamp', 'symbol', 'side', 'price']]

    # Sort the DataFrame by 'timestamp' (oldest to newest)
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    return df


# TODO: This whole function is fucking WRONG!!!
def clean_alerts_data(df, tw_alerts=False):
    if tw_alerts:
        df = pre_clean_tw_data(df)

    # Remove the dummy column if it exists
    if "dummy" in df.columns:
        df = df.drop(columns=["dummy"])

    # Remove the order column if it exists
    if "order" in df.columns:
        df = df.drop(columns=["order"])

    # CHange the name of the column to match the trades data
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "trade_time"})

    # Drop rows with missing or invalid data
    df = df.dropna().reset_index(drop=True)

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
        "price",
        "size",
        "commission",
        "net_amount",
    ]
    cleaned_df = cleaned_df[reordered_columns]

    return cleaned_df
