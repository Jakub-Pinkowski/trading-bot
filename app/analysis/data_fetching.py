import os
import time
from datetime import datetime, timedelta

import pandas as pd

from app.utils.analysis_utils.data_fetching_utils import save_trades_data
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
        return alerts_df
    else:
        return pd.DataFrame(columns=['timestamp', 'symbol', 'side', 'price'])


def get_tw_alerts_data():
    # Retrieve all files in the TW_ALERTS_DIR directory
    files = [f for f in os.listdir(TW_ALERTS_DIR) if f.startswith("TradingView_Alerts_Log_") and f.endswith(".csv")]

    if not files:
        raise FileNotFoundError(f"No files found in '{TW_ALERTS_DIR}' with prefix 'TradingView_Alerts_Log_'.")

    # Extract the date portion from filenames and sort files by date
    try:
        # Sort files based on the date in the filename
        files.sort(key=lambda x: datetime.strptime(x.replace("TradingView_Alerts_Log_", "").replace(".csv", ""), "%Y-%m-%d"), reverse=True)
    except ValueError as e:
        raise ValueError(f"Error parsing dates from filenames: {e}")

    # The latest file based on the date
    latest_file = files[0]
    alerts_file_path = os.path.join(TW_ALERTS_DIR, latest_file)

    # Ensure the file exists
    if not os.path.exists(alerts_file_path):
        raise FileNotFoundError(f"The file '{alerts_file_path}' does not exist.")

    # Read the CSV file
    try:
        alerts_df = pd.read_csv(alerts_file_path)

        return alerts_df
    except Exception as e:
        raise ValueError(f"Error reading the alerts file: {e}")


def get_trades_data():
    trades_df = load_data_from_json_files(
        directory=TRADES_DIR,
        file_prefix="trades",
        date_fields=['trade_time'],
        datetime_format='%Y%m%d-%H:%M:%S',
        index_name='trade_time'
    )

    if not trades_df.empty:
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
