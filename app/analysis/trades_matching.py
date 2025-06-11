import pandas as pd

from app.utils.logger import get_logger
from config import CONTRACT_MULTIPLIERS

logger = get_logger('analysis/trades_matching')


def process_trade(symbol_, side_, size_, price_, commission_, trade_time_, multiplier_, open_trades, processed_trades):
    if symbol_ not in open_trades:
        open_trades[symbol_] = []

    # Try to close with existing open positions (opposite side only)
    while open_trades[symbol_] and ((side_ == 'B' and open_trades[symbol_][0]['side'] == 'S') or
                                    (side_ == 'S' and open_trades[symbol_][0]['side'] == 'B')) and size_ > 0:
        open_trade = open_trades[symbol_][0]
        closing_size = min(open_trade['size'], size_)
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

        # Reduce sizes
        open_trade['size'] -= closing_size
        size_ -= closing_size

        # Remove open trade if fully consumed
        if open_trade['size'] == 0:
            open_trades[symbol_].pop(0)

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
        multiplier = CONTRACT_MULTIPLIERS.get(symbol, 1)
        if symbol not in CONTRACT_MULTIPLIERS:
            logger.warning(f'Symbol \'{symbol}\' not in contract_multipliers. Using multiplier=1.')

        process_trade(symbol, side, size, price, commission, trade_time, multiplier, open_trades, processed_trades)

    formatted_trades = format_processed_trades(processed_trades)
    return formatted_trades
