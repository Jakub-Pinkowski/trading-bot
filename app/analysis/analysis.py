from datetime import datetime, timedelta

import pandas as pd

from app.utils.api_utils import api_get
from app.utils.file_utils import save_to_csv
from config import BASE_URL, TRADES_FILE_PATH


def get_recent_trades():
    # Get yesterday's and today's data
    endpoint = "iserver/account/trades?days=2"

    # BUG: Sometimes the api returns an empty array, but works on a second/third try

    try:
        trades_response = api_get(BASE_URL + endpoint)

        if not trades_response:
            return {"success": False, "error": "No data returned from IBKR API"}

        # Create DataFrame from returned data
        trades_df = pd.DataFrame(trades_response)

        # Convert trade_time to datetime object
        trades_df['trade_time'] = pd.to_datetime(trades_df['trade_time'], format='%Y%m%d-%H:%M:%S')


        # Filter to exactly yesterday's trades
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)

        # Selecting trades between yesterday start and today start
        start_time = datetime.combine(yesterday, datetime.min.time())
        end_time = datetime.combine(today, datetime.min.time())

        yesterdays_trades = trades_df[
            (trades_df['trade_time'] >= start_time) &
            (trades_df['trade_time'] < end_time)
            ]

        # If no trades found for yesterday, inform clearly
        if yesterdays_trades.empty:
            return {"success": True,  "message": "No trades found for yesterday"}

        # Save filtered data into the configured trade log file
        save_to_csv(trades_df, TRADES_FILE_PATH)

        return {"success": True, "data": yesterdays_trades}


    except Exception as err:
        return {"success": False, "error": f"Unexpected error: {err}"}
