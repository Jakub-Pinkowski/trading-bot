def calculate_summary_metrics(trades):
    """ Calculate summary metrics for a list of trades. """

    if not trades:
        return {
            "total_trades": 0,
            "net_pnl": 0,
            "win_rate": 0,
            "avg_trade_duration_hours": 0,
            "profit_factor": 0,
            "avg_profit_per_trade": 0,
            "max_drawdown": 0,
            "sharpe_ratio": 0
        }

    # Ensure all trades have metrics calculated
    for trade in trades:
        if 'pnl_points' not in trade:
            raise ValueError("All trades must have metrics calculated before calling calculate_summary_metrics")

    trades_with_metrics = trades

    # Basic metrics
    total_trades = len(trades_with_metrics)
    winning_trades = [t for t in trades_with_metrics if t['net_pnl'] > 0]
    losing_trades = [t for t in trades_with_metrics if t['net_pnl'] <= 0]

    win_count = len(winning_trades)
    loss_count = len(losing_trades)

    # Win rate
    win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0

    # PnL metrics
    total_net_pnl = sum(t['net_pnl'] for t in trades_with_metrics)
    total_gross_profit = sum(t['net_pnl'] for t in winning_trades) if winning_trades else 0
    total_gross_loss = sum(t['net_pnl'] for t in losing_trades) if losing_trades else 0

    # Profit factor
    profit_factor = abs(total_gross_profit / total_gross_loss) if total_gross_loss != 0 else float('inf')

    # Average metrics
    avg_trade_pnl = total_net_pnl / total_trades if total_trades > 0 else 0
    avg_win = total_gross_profit / win_count if win_count > 0 else 0
    avg_loss = total_gross_loss / loss_count if loss_count > 0 else 0

    # Duration metrics
    avg_duration_hours = sum(t['duration_hours'] for t in trades_with_metrics) / total_trades if total_trades > 0 else 0

    # Calculate drawdown
    equity_curve = []
    current_equity = 0
    max_equity = 0
    max_drawdown = 0

    for trade in trades_with_metrics:
        current_equity += trade['net_pnl']
        equity_curve.append(current_equity)

        if current_equity > max_equity:
            max_equity = current_equity

        drawdown = max_equity - current_equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # Calculate Sharpe ratio (simplified, assuming risk-free rate = 0)
    if total_trades > 1:
        returns = [t['net_pnl'] for t in trades_with_metrics]
        avg_return = sum(returns) / len(returns)
        std_dev = (sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)) ** 0.5
        sharpe_ratio = avg_return / std_dev if std_dev > 0 else 0
    else:
        sharpe_ratio = 0

    return {
        "total_trades": total_trades,
        "winning_trades": win_count,
        "losing_trades": loss_count,
        "win_rate": round(win_rate, 2),
        "total_net_pnl": round(total_net_pnl, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_trade_pnl": round(avg_trade_pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "avg_trade_duration_hours": round(avg_duration_hours, 2),
        "max_drawdown": round(max_drawdown, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
    }


def print_summary_metrics(summary):
    """ Print summary metrics in a formatted way."""
    print("\n====== SUMMARY METRICS ======")
    print(f"Total Trades: {summary['total_trades']}")
    print(f"Winning Trades: {summary['winning_trades']} ({summary['win_rate']}%)")
    print(f"Losing Trades: {summary['losing_trades']}")
    print(f"Total Net PnL: ${summary['total_net_pnl']}")
    print(f"Profit Factor: {summary['profit_factor']}")
    print(f"Avg Trade PnL: ${summary['avg_trade_pnl']}")
    print(f"Avg Win: ${summary['avg_win']}")
    print(f"Avg Loss: ${summary['avg_loss']}")
    print(f"Avg Trade Duration: {summary['avg_trade_duration_hours']} hours")
    print(f"Max Drawdown: ${summary['max_drawdown']}")
    print(f"Sharpe Ratio: {summary['sharpe_ratio']}")
    print("=============================\n")
