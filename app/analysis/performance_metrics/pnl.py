import pandas as pd


def calculate_pnl(matched_trades):
    # List to hold processed trades with PnL
    pnl_trades = []

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

        # Calculate PnL based on net amounts and price difference
        if entry_side == 'B' and exit_side == 'S':  # Long -> Short
            pnl = (exit_net_amount - entry_net_amount) * size - total_commission
        elif entry_side == 'S' and exit_side == 'B':  # Short -> Long
            pnl = (entry_net_amount - exit_net_amount) * size - total_commission
        else:
            pnl = 0  # In case of any unexpected conditions

        # Calculate PnL percentage based on net amounts
        pnl_pct = round((pnl / entry_net_amount) * 100, 2) if entry_net_amount else 0

        # Append the results to the pnl_trades list
        pnl_trades.append({
            'start_time': entry_trade_time,
            'end_time': exit_trade_time,
            'symbol': symbol,
            'entry_side': entry_side,
            'entry_price': entry_price,
            'entry_net_amount': entry_net_amount,
            'exit_side': exit_side,
            'exit_price': exit_price,
            'size': size,
            'exit_net_amount': exit_net_amount,
            'total_commission': total_commission,
            'pnl': pnl,
            'pnl_pct': pnl_pct
        })

    # Convert the results to a DataFrame
    df_pnl = pd.DataFrame(pnl_trades)

    # Sort by start time and reset index
    df_pnl = df_pnl.sort_values(by='start_time').reset_index(drop=True)

    return df_pnl