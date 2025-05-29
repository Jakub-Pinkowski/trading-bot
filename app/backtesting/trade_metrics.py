from app.utils.logger import get_logger
from config import CONTRACT_MULTIPLIERS

logger = get_logger()

# Fixed commission per trade in dollars
COMMISSION_PER_TRADE = 2.0


def calculate_trade_metrics(trade, symbol):
    """ Calculate metrics for a single trade. """

    # Create a copy of the trade to avoid modifying the original
    trade_with_metrics = trade.copy()

    # Determine contract value based on the symbol
    if symbol in CONTRACT_MULTIPLIERS:
        contract_value = CONTRACT_MULTIPLIERS[symbol]
    else:
        logger.error(f"No contract multiplier found for symbol: {symbol}")
        raise ValueError(f"No contract multiplier found for symbol: {symbol}")

    # Calculate trade duration
    trade_duration = trade['exit_time'] - trade['entry_time']
    trade_with_metrics['duration'] = trade_duration
    trade_with_metrics['duration_hours'] = trade_duration.total_seconds() / 3600

    # Calculate PnL in points
    if trade['side'] == 'long':
        pnl_points = trade['exit_price'] - trade['entry_price']
    else:  # short
        pnl_points = trade['entry_price'] - trade['exit_price']

    trade_with_metrics['pnl_points'] = round(pnl_points, 2)

    # Calculate PnL in dollars
    pnl_dollars = pnl_points * contract_value
    trade_with_metrics['pnl_dollars'] = round(pnl_dollars, 2)

    # Use fixed commission per trade
    total_commission = COMMISSION_PER_TRADE

    trade_with_metrics['commission'] = round(total_commission, 2)

    # Calculate net PnL (after commission)
    net_pnl = pnl_dollars - total_commission
    trade_with_metrics['net_pnl'] = round(net_pnl, 2)

    # Calculate return percentage
    initial_capital_used = trade['entry_price'] * contract_value

    if initial_capital_used != 0:
        return_pct = (net_pnl / initial_capital_used) * 100
    else:
        return_pct = 0

    trade_with_metrics['return_pct'] = round(return_pct, 2)

    return trade_with_metrics


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
    trades_with_metrics = [
        calculate_trade_metrics(trade) if 'pnl_points' not in trade else trade
        for trade in trades
    ]

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


def print_trade_metrics(trade):
    """ Print metrics for a single trade in a formatted way. """
    if 'pnl_points' not in trade:
        trade = calculate_trade_metrics(trade)

    print("\n=== TRADE METRICS ===")
    print(f"Entry: {trade['entry_time']} at {trade['entry_price']}")
    print(f"Exit: {trade['exit_time']} at {trade['exit_price']}")
    print(f"Side: {trade['side']}")
    print(f"Duration: {trade['duration']} ({trade['duration_hours']:.2f} hours)")
    print(f"PnL (points): {trade['pnl_points']}")
    print(f"PnL (dollars): ${trade['pnl_dollars']}")
    print(f"Commission: ${trade['commission']}")
    print(f"Net PnL: ${trade['net_pnl']}")
    print(f"Return: {trade['return_pct']}%")
    print("=====================\n")


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
