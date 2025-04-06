import pandas as pd


def calculate_pnl(entry_side, exit_side, entry_net_amount, exit_net_amount, size, total_commission):
    if entry_side == 'B' and exit_side == 'S':  # Long trade
        pnl = (exit_net_amount - entry_net_amount) * size - total_commission
    elif entry_side == 'S' and exit_side == 'B':  # Short trade
        pnl = (entry_net_amount - exit_net_amount) * size - total_commission
    else:
        pnl = 0  # Handling unexpected conditions gracefully

    pnl_pct = round((pnl / entry_net_amount) * 100, 2) if entry_net_amount else 0

    return pnl, pnl_pct


def add_metrics(matched_trades):
    # List to hold processed trades with PnL
    trades = []

    # Process each matched trade
    for _, row in matched_trades.iterrows():
        symbol = row['symbol']
        entry_side = row['entry_side']
        entry_price = row['entry_price']
        entry_net_amount = row['entry_net_amount']
        exit_side = row['exit_side']
        exit_price = row['exit_price']
        size = row['size']
        exit_net_amount = row['exit_net_amount']
        total_commission = row['total_commission']
        entry_trade_time = row['start_time']
        exit_trade_time = row['end_time']

        # Calculate PnL
        pnl, pnl_pct = calculate_pnl(entry_side, exit_side, entry_net_amount, exit_net_amount, size, total_commission)

        # Append the results to the trades list
        trades.append({
            'start_time': entry_trade_time,
            'symbol': symbol,
            'entry_side': entry_side,
            'entry_price': entry_price,
            'entry_net_amount': entry_net_amount,
            'end_time': exit_trade_time,
            'exit_side': exit_side,
            'exit_price': exit_price,
            'size': size,
            'exit_net_amount': exit_net_amount,
            'total_commission': total_commission,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
        })

    # Convert the results to a DataFrame
    df = pd.DataFrame(trades)

    # Sort by start time and reset index
    df = df.sort_values(by='start_time').reset_index(drop=True)

    return df
