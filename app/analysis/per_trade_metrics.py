import pandas as pd


def calculate_pnl(entry_side, exit_side, entry_net_amount, exit_net_amount, size, total_commission):
    if entry_side == 'B' and exit_side == 'S':  # Long trade
        pnl = (exit_net_amount - entry_net_amount) * size - total_commission
    elif entry_side == 'S' and exit_side == 'B':  # Short trade
        pnl = (entry_net_amount - exit_net_amount) * size - total_commission
    else:
        pnl = 0  # Handling unexpected conditions gracefully

    pnl_pct = round((pnl / entry_net_amount), 4) if entry_net_amount else 0

    return pnl, pnl_pct


def calculate_trade_duration(start_time, end_time):
    return round((end_time - start_time).total_seconds() / 60.0, 2)

def calculate_absolute_return(entry_net_amount, exit_net_amount, entry_side):
    abs_return = exit_net_amount - entry_net_amount
    if entry_side == 'S':  # For short trades, reverse the sign
        abs_return *= -1
    return abs_return


def calculate_commission_pct(total_commission, entry_net_amount):
    if entry_net_amount == 0:  # Avoid division by zero
        return 0
    return round((total_commission / entry_net_amount) * 100, 4)


def calculate_price_move_pct(entry_price, exit_price):
    if entry_price == 0:  # Avoid division by zero
        return 0
    return round(((exit_price - entry_price) / entry_price) * 100, 4)


def add_per_trade_metrics(matched_trades):
    # List to hold processed trades
    trades = []

    # Process each matched trade
    for _, row in matched_trades.iterrows():
        symbol = row['symbol']
        entry_side = row['entry_side']
        entry_price = row['entry_price']
        entry_net_amount = row['entry_net_amount']
        exit_side = row['exit_side']
        exit_price = row['exit_price']
        exit_net_amount = row['exit_net_amount']
        size = row['size']
        total_commission = row['total_commission']
        entry_trade_time = row['start_time']
        exit_trade_time = row['end_time']

        # TODO: Remove unused/unnecessary metrics later
        # Calculate all the metrics
        pnl, pnl_pct = calculate_pnl(entry_side, exit_side, entry_net_amount, exit_net_amount, size, total_commission)
        trade_duration = calculate_trade_duration(entry_trade_time, exit_trade_time)
        abs_return = calculate_absolute_return(entry_net_amount, exit_net_amount, entry_side)
        commission_pct = calculate_commission_pct(total_commission, entry_net_amount)
        price_move_pct = calculate_price_move_pct(entry_price, exit_price)

        # Append the results to the trades list
        trades.append({
            'symbol': symbol,
            'entry_time': entry_trade_time,
            'entry_side': entry_side,
            'entry_price': entry_price,
            'entry_net_amount': entry_net_amount,
            'exit_time': exit_trade_time,
            'exit_side': exit_side,
            'exit_price': exit_price,
            'exit_net_amount': exit_net_amount,
            'size': size,
            'total_commission': total_commission,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'trade_duration': trade_duration,
        })

    # Convert the results to a DataFrame
    df = pd.DataFrame(trades)

    # Sort by start time and reset index
    df = df.sort_values(by='entry_time').reset_index(drop=True)

    return df
