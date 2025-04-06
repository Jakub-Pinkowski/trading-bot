import pandas as pd


def calculate_pnl(trades):
    trades = trades.sort_values(by="trade_time").reset_index(drop=True)

    results = []
    inventory = {}  # symbol -> list of open buys: each is a dict with size, price, commission, time

    for _, row in trades.iterrows():
        symbol = row["symbol"]
        side = row["side"]
        size = row["size"]
        price = row["price"]
        commission = row["commission"]
        time = row["trade_time"]

        if symbol not in inventory:
            inventory[symbol] = []

        if side == "B":
            inventory[symbol].append({
                "size": size,
                "price": price,
                "commission": commission,
                "time": time
            })

        elif side == "S":
            remaining_size = size
            while remaining_size > 0 and inventory[symbol]:
                buy = inventory[symbol][0]
                matched_size = min(remaining_size, buy["size"])

                pnl = (
                        (price - buy["price"]) * matched_size
                        - buy["commission"] * (matched_size / buy["size"])
                        - commission * (matched_size / size)
                )

                results.append({
                    "symbol": symbol,
                    "buy_time": buy["time"],
                    "sell_time": time,
                    "buy_price": buy["price"],
                    "sell_price": price,
                    "size": matched_size,
                    "buy_commission": buy["commission"] * (matched_size / buy["size"]),
                    "sell_commission": commission * (matched_size / size),
                    "pnl": pnl
                })

                buy["size"] -= matched_size
                remaining_size -= matched_size

                if buy["size"] <= 0:
                    inventory[symbol].pop(0)  # remove fully matched buy

    return pd.DataFrame(results)


# NOTE: Deprecated function
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
