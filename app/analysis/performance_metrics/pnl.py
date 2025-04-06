import pandas as pd


# NOTE: No commision for now
def calculate_pnl(trades):
    print(trades)
    open_trades = {}
    closed_trades = []

    # Define contract multipliers per symbol
    contract_multipliers = {
        'CL': 1000,
        'SI': 5000,
        'ZW': 50,
        'ZC': 50,
        'MZC': 10,
        'HG': 25000,
        'GC': 100,
        'PL': 50,
        'ZS': 50
    }

    for _, row in trades.iterrows():
        symbol = row['symbol']
        side = row['side']
        size = row['size']
        price = row['price']
        commission = row['commission']
        trade_time = row['trade_time']
        multiplier = contract_multipliers.get(symbol, 1)

        if symbol not in open_trades:
            open_trades[symbol] = []

        if side == 'B':
            if open_trades[symbol] and open_trades[symbol][0]['side'] == 'S':
                # Closing short position
                open_trade = open_trades[symbol].pop(0)
                total_commission = commission + open_trade['commission']
                entry_net = open_trade['price'] * size * multiplier
                exit_net = price * size * multiplier
                pnl = (open_trade['price'] - price) * size * multiplier - total_commission
                pnl_pct = round((pnl / entry_net) * 100, 2) if entry_net else 0
                closed_trades.append({
                    'start_time': open_trade['trade_time'],
                    'end_time': trade_time,
                    'symbol': symbol,
                    'entry_side': open_trade['side'],
                    'entry_price': open_trade['price'],
                    'exit_side': side,
                    'exit_price': price,
                    'size': size,
                    'entry_net_amount': entry_net,
                    'exit_net_amount': exit_net,
                    'total_commission': total_commission,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                })
            else:
                open_trades[symbol].append({
                    'side': side,
                    'price': price,
                    'commission': commission,
                    'trade_time': trade_time
                })

        elif side == 'S':
            if open_trades[symbol] and open_trades[symbol][0]['side'] == 'B':
                # Closing long position
                open_trade = open_trades[symbol].pop(0)
                total_commission = commission + open_trade['commission']
                entry_net = open_trade['price'] * size * multiplier
                exit_net = price * size * multiplier
                pnl = (price - open_trade['price']) * size * multiplier - total_commission
                pnl_pct = round((pnl / entry_net) * 100, 2) if entry_net else 0
                closed_trades.append({
                    'start_time': open_trade['trade_time'],
                    'end_time': trade_time,
                    'symbol': symbol,
                    'entry_side': open_trade['side'],
                    'entry_price': open_trade['price'],
                    'exit_side': side,
                    'exit_price': price,
                    'size': size,
                    'entry_net_amount': entry_net,
                    'exit_net_amount': exit_net,
                    'total_commission': total_commission,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                })
            else:
                open_trades[symbol].append({
                    'side': side,
                    'price': price,
                    'commission': commission,
                    'trade_time': trade_time
                })

    df = pd.DataFrame(closed_trades)
    df = df.sort_values(by='start_time').reset_index(drop=True)
    return df


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
