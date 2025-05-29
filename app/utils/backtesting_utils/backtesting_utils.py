def format_trades(trades):
    formatted_trades = []
    for trade in trades:
        formatted_trades.append({
            "entry_time": str(trade["entry_time"]),
            "entry_price": float(trade["entry_price"]),
            "exit_time": str(trade["exit_time"]),
            "exit_price": float(trade["exit_price"]),
            "side": str(trade["side"]),
        })
    return formatted_trades
