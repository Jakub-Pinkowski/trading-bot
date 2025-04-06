import pandas as pd

from app.analysis.data_fetching import get_alerts_data, get_trades_data

# TODO: Move this to a different file
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
    print(alerts_data)
    trades_data = get_trades_data()
    print(trades_data)


    # pnl_alerts = calculate_alerts_pnl(alerts_data)
