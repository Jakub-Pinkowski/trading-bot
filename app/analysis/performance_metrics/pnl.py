import pandas as pd


def match_trades(trades):
    open_trades = {}
    processed_trades = []

    # Contract multipliers
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

    # TODO: Fix the errors
    def process_trade(symbol, side, size, price, commission, trade_time, multiplier):
        if symbol not in open_trades:
            open_trades[symbol] = []

        if open_trades[symbol] and ((side == 'B' and open_trades[symbol][0]['side'] == 'S') or
                                    (side == 'S' and open_trades[symbol][0]['side'] == 'B')):
            # Closing an opposite position (B -> S or S -> B)
            open_trade = open_trades[symbol].pop(0)
            closing_size = min(open_trade['size'], size)  # Close as many as possible
            processed_trades.append({
                'start_time': open_trade['trade_time'],
                'end_time': trade_time,
                'symbol': symbol,
                'entry_side': open_trade['side'],
                'entry_price': open_trade['price'],
                'entry_net_amount': open_trade['price'] * closing_size * multiplier,
                'exit_side': side,
                'exit_price': price,
                'size': closing_size,
                'exit_net_amount': price * closing_size * multiplier,
                'total_commission': commission + open_trade['commission'],
            })
            size -= closing_size  # Remaining size to open a new position

        # If there's still remaining size, open a new position
        if size > 0:
            open_trades[symbol].append({
                'side': side,
                'price': price,
                'commission': commission,
                'size': size,
                'trade_time': trade_time
            })

    # Process each trade
    for _, row in trades.iterrows():
        symbol = row['symbol']
        side = row['side']
        size = row['size']
        price = row['price']
        commission = row['commission']
        trade_time = row['trade_time']
        multiplier = contract_multipliers.get(symbol, 1)

        process_trade(symbol, side, size, price, commission, trade_time, multiplier)

    df = pd.DataFrame(processed_trades)
    df = df.sort_values(by='start_time').reset_index(drop=True)
    return df


# TODO: Use matched trades instead of raw data
def calculate_pnl(trades):
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
