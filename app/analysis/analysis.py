import pandas as pd

from app.utils.analisys_utils import clean_alerts_data, clean_trade_data, save_trades_data
from app.utils.api_utils import api_get
from app.utils.file_utils import load_data_from_json_files
from config import BASE_URL, ALERTS_DIR, TRADES_DIR


# TODO: The whole file is a mess, clean it up
# TODO: Consider splitting analysis into separate files for alerts and trades

def get_alerts_data():
    alerts_df = load_data_from_json_files(
        directory=ALERTS_DIR,
        file_prefix="alerts",
        date_fields=['timestamp'],
        datetime_format='%y-%m-%d %H:%M:%S',
        index_name='timestamp'
    )

    if not alerts_df.empty:
        alerts_df = clean_alerts_data(alerts_df)
        # Explicitly sort by 'timestamp'
        alerts_df = alerts_df.sort_values('timestamp').reset_index(drop=True)
        return alerts_df
    else:
        return pd.DataFrame(columns=['symbol', 'order', 'price', 'timestamp'])


# TODO: Actually use it somewhere
# BUG: Sometimes the API returns an empty array, but works on a second/third try
def fetch_trades_data():
    # Fetch trades from the last week
    endpoint = "iserver/account/trades?days=7"

    try:
        trades_json = api_get(BASE_URL + endpoint)

        if not trades_json:
            return {"success": False, "error": "No data returned from IBKR API"}

        save_trades_data(trades_json, TRADES_DIR)

        return {"success": True, "message": "Trades fetched successfully"}

    except Exception as err:
        return {"success": False, "error": f"Unexpected error: {err}"}


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
        return trades_df
    else:
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
    alerts_data = get_alerts_data()
    fetch_trades_data()

    trades_data = get_trades_data()

    # pnl_alerts = calculate_alerts_pnl(alerts_data)
