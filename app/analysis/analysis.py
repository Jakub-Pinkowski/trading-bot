from datetime import datetime, timedelta

import pandas as pd

from app.utils.analisys_utils import clean_trade_data
from app.utils.api_utils import api_get
from app.utils.file_utils import load_file, save_to_csv
from config import BASE_URL, ALERTS_FILE_PATH, TRADES_FILE_PATH

def get_alerts_data():
    alerts_data = load_file(ALERTS_FILE_PATH)
    return alerts_data



def get_recent_trades():
    # Get yesterday's and today's data
    endpoint = "iserver/account/trades?days=2"

    # BUG: Sometimes the api returns an empty array, but works on a second/third try

    try:
        trades_response = api_get(BASE_URL + endpoint)

        if not trades_response:
            return {"success": False, "error": "No data returned from IBKR API"}

        # Create DataFrame from returned data and convert trade_time to datetime object
        trades_df = pd.DataFrame(trades_response)
        trades_df['trade_time'] = pd.to_datetime(trades_df['trade_time'], format='%Y%m%d-%H:%M:%S')

        # Cleaning/preparing
        cleaned_df = clean_trade_data(trades_df)

        print("cleaned_df:", cleaned_df)

        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        start_time = datetime.combine(yesterday, datetime.min.time())
        end_time = datetime.combine(today, datetime.min.time())

        yesterdays_trades = cleaned_df[
            (cleaned_df['trade_time'] >= start_time) &
            (cleaned_df['trade_time'] < end_time)
            ]

        if yesterdays_trades.empty:
            return {"success": True, "data": yesterdays_trades, "message": "No trades found for yesterday"}

        save_to_csv(yesterdays_trades, TRADES_FILE_PATH)

        return {"success": True, "data": yesterdays_trades}

    except Exception as err:
        return {"success": False, "error": f"Unexpected error: {err}"}



def run_analysis():
    alerts_data = get_alerts_data()
    print("alerts_data:", alerts_data)
    trades_data = get_recent_trades()
