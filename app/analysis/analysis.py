import json
import os
from datetime import datetime
from glob import glob
from zoneinfo import ZoneInfo

import pandas as pd

from app.utils.analisys_utils import clean_alerts_data, clean_trade_data
from app.utils.api_utils import api_get
from app.utils.file_utils import load_file, json_to_dataframe
from config import BASE_URL, ALERTS_DIR, TRADES_DIR


# TODO: The whole file is a mess, clean it up
# TODO: Consider splitting analysis into separate files for alerts and trades

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


def split_and_save_trades_by_date(trades_json, trades_dir, timezone="Europe/Berlin"):
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

    print(f"Trades successfully separated, deduplicated by execution_id, and saved by date in {trades_dir}")


# TODO: Actually use it somewhere

def fetch_trades_data():
    # Fetch trades from the last week
    endpoint = "iserver/account/trades?days=7"

    # BUG: Sometimes the API returns an empty array, but works on a second/third try

    try:
        trades_json = api_get(BASE_URL + endpoint)

        if not trades_json:
            return {"success": False, "error": "No data returned from IBKR API"}

        split_and_save_trades_by_date(trades_json, TRADES_DIR)

        return {"success": True, "message": "Trades fetched successfully"}

    except Exception as err:
        return {"success": False, "error": f"Unexpected error: {err}"}


def get_trades_data():
    # Fetch all json files in TRADES_DIR
    file_pattern = os.path.join(TRADES_DIR, "trades_*.json")
    trade_files = sorted(glob(file_pattern))

    data_frames = []
    for trade_file in trade_files:
        # load individual daily trades json file
        daily_trades_json = load_file(trade_file)

        if daily_trades_json:
            daily_trades_df = json_to_dataframe(
                daily_trades_json,
                date_fields=['trade_time'],
                datetime_format='%Y%m%d-%H:%M:%S',
                orient='index',
                index_name='trade_time'
            )
            data_frames.append(daily_trades_df)

    # Combine all daily dataframes if there's data
    if data_frames:
        trades_df = pd.concat(data_frames).sort_index()
        print(trades_df)

        # Cleaning the combined alerts DataFrame
        trades_df = clean_trade_data(trades_df)

        # Explicitly sort by 'trade_time'
        trades_df = trades_df.sort_values('trade_time').reset_index(drop=True)

        return trades_df
    else:
        # Return empty dataframe if no data found
        return pd.DataFrame(columns=['symbol', 'order', 'price', 'timestamp'])


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
    # alerts_data = get_alerts_data()
    fetch_trades_data()
    trades_data = get_trades_data()
    print(trades_data)

    # pnl_alerts = calculate_alerts_pnl(alerts_data)
