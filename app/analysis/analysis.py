from app.utils.analisys_utils import clean_alerts_data, clean_trade_data, filter_yesterdays_data
from app.utils.api_utils import api_get
from app.utils.file_utils import load_file, save_to_csv, json_to_dataframe
from config import BASE_URL, ALERTS_FILE_PATH, TRADES_FILE_PATH


def get_alerts_data():
    alerts_json = load_file(ALERTS_FILE_PATH)
    alerts_df = json_to_dataframe(
        alerts_json,
        date_fields=['timestamp'],
        datetime_format='%y-%m-%d %H:%M:%S',
        orient='index',
        index_name='timestamp'
    )

    alerts_df = clean_alerts_data(alerts_df)

    return alerts_df


def get_recent_trades():
    # Get yesterday's and today's data
    endpoint = "iserver/account/trades?days=3"

    # BUG: Sometimes the api returns an empty array, but works on a second/third try

    try:
        trades_json = api_get(BASE_URL + endpoint)

        if not trades_json:
            return {"success": False, "error": "No data returned from IBKR API"}

        # Create DataFrame from returned data and convert trade_time to datetime object
        trades_df = json_to_dataframe(
            trades_json,
            date_fields=['trade_time'],
            datetime_format='%Y%m%d-%H:%M:%S'
        )

        # Clean the DataFrame
        cleaned_df = clean_trade_data(trades_df)

        print("cleaned_df:", cleaned_df)

        # TODO: Later on change to yesterday's data
        # Filter yesterday's trades
        # yesterdays_trades = filter_yesterdays_data(cleaned_df, 'trade_time')
        #
        # if yesterdays_trades.empty:
        #     return {"success": True, "data": yesterdays_trades, "message": "No trades found for yesterday"}

        save_to_csv(cleaned_df, TRADES_FILE_PATH)

        return {"success": True, "data": cleaned_df}

    except Exception as err:
        return {"success": False, "error": f"Unexpected error: {err}"}


def run_analysis():
    alerts_data = get_alerts_data()
    print("alerts_data:", alerts_data)
    trades_data = get_recent_trades()
