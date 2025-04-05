import pandas as pd

from app.utils.analisys_utils import clean_alerts_data, clean_trade_data
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

            # If the new alert is opposite to the current position, we record the trade
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
            else:
                # Ignore consecutive same-side signals, clearly stated:
                print(f"Ignoring consecutive '{order}' signal for symbol '{symbol}' at {timestamp}")

    pnl_df = pd.DataFrame(pnl_records)
    return pnl_df


def run_analysis():
    alerts_data = get_alerts_data()
    print("alerts_data:", alerts_data)
    trades_data = get_recent_trades()

    pnl_alerts = calculate_alerts_pnl(alerts_data)
    print("pnl_alerts:", pnl_alerts)
