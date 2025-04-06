import time
from datetime import datetime, timedelta

import pandas as pd

from app.utils.analysis_utils.data_fetching_utils import clean_alerts_data, clean_trade_data, save_trades_data
from app.utils.api_utils import api_get
from app.utils.file_utils import load_data_from_json_files
from config import ALERTS_DIR
from config import BASE_URL, TRADES_DIR


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


# NOTE: Returns data from the last 7 days
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
        # Explicitly sort by 'trade_time'
        trades_df = trades_df.sort_values('trade_time').reset_index(drop=True)

        # Filter data to include trades only from the last 7 days
        seven_days_ago = datetime.now() - timedelta(days=7)
        trades_df = trades_df[trades_df['trade_time'] >= seven_days_ago]

        return trades_df
    else:
        return pd.DataFrame(columns=['symbol', 'order', 'price', 'trade_time'])


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
