import pandas as pd

# NOTE: Those are TW multipliers, might not with properly with actual trades
# Contract multipliers
contract_multipliers = {
    'CL': 1000,
    'NG': 10000,
    'GC': 100,
    'SI': 5000,
    'HG': 25000,
    'PL': 50,
    'ZC': 50,
    'ZS': 50,
    'ZL': 600,
    'ZW': 50,
    'SB': 1120,
    'NQ': 20,
    'ES': 50,
    'YM': 5,
    'RTY': 50,
    'ZB': 114,
    'MCL': 100,
    'MNG': 1000,
    'MGC': 10,
    'SIL': 1000,
    'MHG': 2500,
    'MHNG': 2500,
    'PLM': 10,
    'MZC': 500,
    'MZS': 500,
    'MZL': 6000,
    'MZW': 500,
    'MBT': 0.1,
    'MET': 0.1,
    'MNQ': 2,
    'MES': 5,
    'MYM': 0.5,
    'M2K': 5,
}



def process_trade(symbol_, side_, size_, price_, commission_, trade_time_, multiplier_, open_trades, processed_trades):
    if symbol_ not in open_trades:
        open_trades[symbol_] = []

    if open_trades[symbol_] and ((side_ == 'B' and open_trades[symbol_][0]['side'] == 'S') or
                                 (side_ == 'S' and open_trades[symbol_][0]['side'] == 'B')):
        # Closing an opposite position (B -> S or S -> B)
        open_trade = open_trades[symbol_].pop(0)
        closing_size = min(open_trade['size'], size_)  # Close as many as possible
        processed_trades.append({
            'symbol': symbol_,
            'entry_time': open_trade['trade_time'],
            'entry_side': open_trade['side'],
            'entry_price': open_trade['price'],
            'entry_net_amount': open_trade['price'] * closing_size * multiplier_,
            'exit_time': trade_time_,
            'exit_side': side_,
            'exit_price': price_,
            'exit_net_amount': price_ * closing_size * multiplier_,
            'size': closing_size,
            'total_commission': commission_ + open_trade['commission'],
        })
        size_ -= closing_size  # Remaining size to open a new position

    # If there's still a remaining size, open a new position
    if size_ > 0:
        open_trades[symbol_].append({
            'side': side_,
            'price': price_,
            'commission': commission_,
            'size': size_,
            'trade_time': trade_time_
        })


def format_processed_trades(processed_trades):
    df = pd.DataFrame(processed_trades)

    if not df.empty:
        # Format prices to 2 decimals and net amounts to 0 decimals
        df['entry_price'] = df['entry_price'].round(2)
        df['exit_price'] = df['exit_price'].round(2)
        df['entry_net_amount'] = df['entry_net_amount'].round(0).astype(int)
        df['exit_net_amount'] = df['exit_net_amount'].round(0).astype(int)
        df = df.sort_values(by='entry_time').reset_index(drop=True)

    return df


def match_trades(trades, is_alerts=False):
    open_trades = {}
    processed_trades = []

    for _, row in trades.iterrows():
        # Extract fields with defaults for alerts if necessary
        trade_time = row['trade_time']
        symbol = row['symbol']
        side = row['side']
        price = row['price']

        # Defaults for alerts
        size = 1 if is_alerts else row['size']
        commission = 0 if is_alerts else row['commission']
        multiplier = contract_multipliers.get(symbol, 1)

        process_trade(symbol, side, size, price, commission, trade_time, multiplier, open_trades, processed_trades)

    formatted_trades = format_processed_trades(processed_trades)
    return formatted_trades
