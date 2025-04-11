import json
import os
import time
from datetime import datetime, timedelta

import pandas as pd

from app.utils.analysis_utils.data_fetching_utils import clean_alerts_data, clean_trade_data, save_trades_data
from app.utils.api_utils import api_get
from app.utils.file_utils import load_data_from_json_files
from config import BASE_URL, ALERTS_DIR, TRADES_DIR, TIMEFRAME_TO_ANALYZE, TW_ALERTS_DIR


def get_alerts_data():
    alerts_df = load_data_from_json_files(
        directory=ALERTS_DIR,
        file_prefix="alerts",
        date_fields=['timestamp'],
        datetime_format='%y-%m-%d %H:%M:%S',
        index_name='timestamp'
    )

    if not alerts_df.empty:
        # Sort the DataFrame by 'timestamp' before cleaning
        alerts_df = alerts_df.sort_values('timestamp').reset_index(drop=True)

        # Clean the data
        alerts_df = clean_alerts_data(alerts_df)

        return alerts_df
    else:
        return pd.DataFrame(columns=['symbol', 'order', 'price', 'timestamp'])

# TODO: Figure out a solution for the dates
def get_tw_alerts_data():
    # Target the TradingView alerts CSV
    alerts_file_path = os.path.join(TW_ALERTS_DIR, "TradingView_Alerts_Log_2025-04-11.csv")

    # Check if the file exists
    if not os.path.exists(alerts_file_path):
        raise FileNotFoundError(f"The file '{alerts_file_path}' does not exist.")

    # Read the CSV file
    try:
        alerts_df = pd.read_csv(alerts_file_path)
    except Exception as e:
        raise ValueError(f"Error reading the alerts file: {e}")

    # Parse the 'Description' column (JSON data)
    def parse_description(description):
        try:
            if pd.notna(description):  # Ensure description is not null
                return json.loads(description)  # Convert JSON to a Python dict
            return {}
        except json.JSONDecodeError:
            return {}

    # Extract relevant fields from the 'Description' column
    alerts_df['description_parsed'] = alerts_df['Description'].apply(parse_description)
    alerts_df['symbol'] = alerts_df['description_parsed'].apply(lambda x: x.get('symbol'))
    alerts_df['side'] = alerts_df['description_parsed'].apply(lambda x: x.get('side'))
    alerts_df['price'] = alerts_df['description_parsed'].apply(lambda x: x.get('price'))

    # Parse 'Time' column into a datetime object
    alerts_df['timestamp'] = pd.to_datetime(alerts_df['Time'], errors='coerce')

    # Select only the relevant columns
    alerts_df = alerts_df[['symbol', 'side', 'price', 'timestamp']]

    # Drop rows with missing or invalid data
    alerts_df = alerts_df.dropna().reset_index(drop=True)

    # Sort the DataFrame by 'timestamp' (oldest to newest)
    alerts_df = alerts_df.sort_values(by='timestamp').reset_index(drop=True)

    # Remove consecutive trades on the same side for the same symbol
    def remove_consecutive_sides(group):
        # Keep rows where the side is different from the previous one
        return group.loc[group['side'] != group['side'].shift()]

    # Apply the cleaning logic group by group
    cleaned_df = alerts_df.groupby('symbol', group_keys=False).apply(remove_consecutive_sides)

    # Define the output file path
    cleaned_file_path = os.path.join(TW_ALERTS_DIR, "Cleaning_TradingView_Alerts_2025-04-11.csv")

    # Save the cleaned DataFrame to CSV
    try:
        cleaned_df.to_csv(cleaned_file_path, index=False)
    except Exception as e:
        raise ValueError(f"Error saving the cleaned data to CSV: {e}")

    # Return the cleaned DataFrame
    return cleaned_df


def get_trades_data():
    trades_df = load_data_from_json_files(
        directory=TRADES_DIR,
        file_prefix="trades",
        date_fields=['trade_time'],
        datetime_format='%Y%m%d-%H:%M:%S',
        index_name='trade_time'
    )

    if not trades_df.empty:
        trades_df = clean_trade_data(trades_df)
        trades_df = trades_df.sort_values('trade_time').reset_index(drop=True)

        # Filter data to only include trades from the last 7 days
        seven_days_ago = datetime.now() - timedelta(days=TIMEFRAME_TO_ANALYZE)
        trades_last_7_days = trades_df[trades_df['trade_time'] >= seven_days_ago]

        # Return if any data exists within the last 7 days
        if not trades_last_7_days.empty:
            return trades_last_7_days

    # If no data from the last 7 days, fetch new trades data
    fetch_result = fetch_trades_data()
    if fetch_result.get("success"):
        # Reload the data after fetching new trades
        return get_trades_data()
    else:
        # Return an empty DataFrame if all attempts fail
        return pd.DataFrame(columns=['conid', 'side', 'price', 'trade_time'])


def fetch_trades_data(max_retries=3, retry_delay=2):
    endpoint = "iserver/account/trades?days=7"
    attempt = 0

    while attempt < max_retries:
        try:
            trades_json = api_get(BASE_URL + endpoint)

            # Validate if response is not empty
            if trades_json:
                save_trades_data(trades_json, TRADES_DIR)
                return {"success": True, "message": "Trades fetched successfully"}

            # If no data returned, increment attempt counter and retry after delay
            attempt += 1
            if attempt < max_retries:
                time.sleep(retry_delay)  # Wait before making the next attempt

        except Exception as err:
            return {"success": False, "error": f"Unexpected error: {err}"}

    # After maximum retries, return error
    return {"success": False, "error": "No data returned from IBKR API after multiple retries"}
