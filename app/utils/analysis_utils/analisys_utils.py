import pandas as pd


def process_trade(symbol_, side_, size_, price_, commission_, trade_time_, multiplier_, open_trades, processed_trades):
    if symbol_ not in open_trades:
        open_trades[symbol_] = []

    if open_trades[symbol_] and ((side_ == 'B' and open_trades[symbol_][0]['side'] == 'S') or
                                 (side_ == 'S' and open_trades[symbol_][0]['side'] == 'B')):
        # Closing an opposite position (B -> S or S -> B)
        open_trade = open_trades[symbol_].pop(0)
        closing_size = min(open_trade['size'], size_)  # Close as many as possible
        processed_trades.append({
            'start_time': open_trade['trade_time'],
            'symbol': symbol_,
            'entry_side': open_trade['side'],
            'entry_price': open_trade['price'],
            'entry_net_amount': open_trade['price'] * closing_size * multiplier_,
            'end_time': trade_time_,
            'exit_side': side_,
            'exit_price': price_,
            'size': closing_size,
            'exit_net_amount': price_ * closing_size * multiplier_,
            'total_commission': commission_ + open_trade['commission'],
        })
        size_ -= closing_size  # Remaining size to open a new position

    # If there's still remaining size, open a new position
    if size_ > 0:
        open_trades[symbol_].append({
            'side': side_,
            'price': price_,
            'commission': commission_,
            'size': size_,
            'trade_time': trade_time_
        })


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

    # Process each trade
    for _, row in trades.iterrows():
        symbol = row['symbol']
        side = row['side']
        size = row['size']
        price = row['price']
        commission = row['commission']
        trade_time = row['trade_time']
        multiplier = contract_multipliers.get(symbol, 1)

        process_trade(symbol, side, size, price, commission, trade_time, multiplier, open_trades, processed_trades)

    df = pd.DataFrame(processed_trades)
    df = df.sort_values(by='start_time').reset_index(drop=True)
    return df
