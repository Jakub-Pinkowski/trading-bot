import os
from glob import glob

import pandas as pd

from app.utils.analisys_utils import clean_alerts_data, clean_trade_data
from app.utils.api_utils import api_get
from app.utils.file_utils import load_file, save_to_csv, json_to_dataframe
from config import BASE_URL, ALERTS_DIR, TRADES_DIR


def get_alerts_data():
    # Fetch all json files in ALERTS_DIR
    file_pattern = os.path.join(ALERTS_DIR, "alerts_*.json")
    alert_files = sorted(glob(file_pattern))

    data_frames = []
    for alert_file in alert_files:
        # load individual daily alerts json file
        daily_alerts_json = load_file(alert_file)

        if daily_alerts_json:  # Check if data exists for the day
            daily_alerts_df = json_to_dataframe(
                daily_alerts_json,
                date_fields=['timestamp'],
                datetime_format='%y-%m-%d %H:%M:%S',
                orient='index',
                index_name='timestamp'
            )
            data_frames.append(daily_alerts_df)

    # Combine all daily dataframes if there's data
    if data_frames:
        alerts_df = pd.concat(data_frames).sort_index()
        # Cleaning the combined alerts DataFrame
        alerts_df = clean_alerts_data(alerts_df)
        return alerts_df
    else:
        # Return empty dataframe if no data found
        return pd.DataFrame(columns=['symbol', 'order', 'price', 'timestamp'])


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

        # save_to_csv(cleaned_df, TRADES_FILE_PATH)

        return {"success": True, "data": cleaned_df}

    except Exception as err:
        return {"success": False, "error": f"Unexpected error: {err}"}


# TODO: Make it cleaner
def calculate_alerts_pnl(alerts_df):
    alerts_df = alerts_df.sort_values('timestamp')

    positions = {}  # track open positions
    pnl_records = []  # record trades and PnL

    for idx, row in alerts_df.iterrows():
        symbol = row['symbol']
        order = row['order']
        price = row['price']
        timestamp = row['timestamp']

        position_size = 1  # fixed size — adjust later if necessary

        if symbol not in positions:
            # Open a new position if the symbol has no position yet
            positions[symbol] = {
                'order_type': order,
                'entry_price': price,
                'entry_time': timestamp
            }
        else:
            current_position = positions[symbol]

            # If the new alert is opposite to the current position
            if current_position['order_type'] != order:
                entry_price = current_position['entry_price']
                entry_order = current_position['order_type']

                pnl = (price - entry_price) * position_size

                # Proper adjustment for short entries
                if entry_order == 'SELL':
                    pnl = -pnl

                pnl_records.append({
                    'symbol': symbol,
                    'entry_time': current_position['entry_time'],
                    'entry_order': entry_order,
                    'entry_price': entry_price,
                    'exit_time': timestamp,
                    'exit_order': order,
                    'exit_price': price,
                    'pnl': pnl
                })

                # Immediately open new position after closing
                positions[symbol] = {
                    'order_type': order,
                    'entry_price': price,
                    'entry_time': timestamp
                }

    pnl_df = pd.DataFrame(pnl_records)
    return pnl_df


def run_analysis():
    alerts_data = get_alerts_data()
    print("alerts_data: ", alerts_data)
    trades_data = get_recent_trades()

    # pnl_alerts = calculate_alerts_pnl(alerts_data)
