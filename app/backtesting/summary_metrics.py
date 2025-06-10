from app.utils.logger import get_logger

logger = get_logger()


def calculate_max_drawdown(trades):
    """Calculate the maximum drawdown given a list of trades. """
    if not trades:
        return 0, 0

    # Calculate drawdown using both dollar values and percentages
    cumulative_pnl_dollars = []
    cumulative_pnl_pct = []
    cum_sum_dollars = 0
    cum_sum_pct = 0

    for trade in trades:
        cum_sum_dollars += trade['net_pnl']  # Using net_pnl instead of gross_pnl
        cum_sum_pct += trade['return_percentage_of_margin']
        cumulative_pnl_dollars.append(cum_sum_dollars)
        cumulative_pnl_pct.append(cum_sum_pct)

    # Calculate max drawdown in dollars
    peak_dollars = cumulative_pnl_dollars[0] if cumulative_pnl_dollars else 0
    max_drawdown = 0
    for val in cumulative_pnl_dollars:
        if val > peak_dollars:
            peak_dollars = val
        drawdown = peak_dollars - val
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # Calculate max drawdown in percentage (for normalized comparison)
    peak_pct = cumulative_pnl_pct[0] if cumulative_pnl_pct else 0
    maximum_drawdown_percentage = 0
    for val in cumulative_pnl_pct:
        if val > peak_pct:
            peak_pct = val
        drawdown_percentage = peak_pct - val
        if drawdown_percentage > maximum_drawdown_percentage:
            maximum_drawdown_percentage = drawdown_percentage

    return round(max_drawdown, 2), round(maximum_drawdown_percentage, 2)


def calculate_max_consecutive(trades, win=True):
    """Calculate maximum consecutive wins or losses."""
    if not trades:
        return 0

    # Sort trades by date if available
    if 'date' in trades[0]:
        sorted_trades = sorted(trades, key=lambda x: x['date'])
    else:
        sorted_trades = trades

    # Track consecutive wins/losses
    current_streak = 0
    max_streak = 0

    for trade in sorted_trades:
        is_win = trade['return_percentage_of_margin'] > 0

        if (win and is_win) or (not win and not is_win):
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return max_streak


def calculate_sharpe_ratio(trades, risk_free_rate=0.0):
    """Calculate Sharpe ratio: (Average Return - Risk Free Rate) / Standard Deviation of Returns."""
    if not trades or len(trades) < 2:  # Need at least 2 trades for standard deviation
        return 0

    returns = [trade['return_percentage_of_margin'] for trade in trades]
    avg_return = sum(returns) / len(returns)

    # Calculate standard deviation
    variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
    std_dev = variance ** 0.5

    if std_dev == 0:
        return 0  # Avoid division by zero

    sharpe_ratio = (avg_return - risk_free_rate) / std_dev
    return sharpe_ratio


def calculate_sortino_ratio(trades, risk_free_rate=0.0):
    """Calculate Sortino ratio: (Average Return - Risk Free Rate) / Standard Deviation of Negative Returns."""
    if not trades:
        return 0

    returns = [trade['return_percentage_of_margin'] for trade in trades]
    avg_return = sum(returns) / len(returns)

    # Calculate downside deviation (only negative returns)
    negative_returns = [r - avg_return for r in returns if r < avg_return]

    if not negative_returns:
        return float('inf')  # No negative returns

    downside_variance = sum(r ** 2 for r in negative_returns) / len(negative_returns)
    downside_deviation = downside_variance ** 0.5

    if downside_deviation == 0:
        return 0  # Avoid division by zero

    sortino_ratio = (avg_return - risk_free_rate) / downside_deviation
    return sortino_ratio


def calculate_calmar_ratio(trades):
    """Calculate Calmar ratio: Annualized Return / Maximum Drawdown."""
    if not trades:
        return 0

    # Calculate annualized return (assuming percentage returns)
    total_return = sum(trade['return_percentage_of_margin'] for trade in trades)

    # Get maximum drawdown percentage
    _, max_drawdown_pct = calculate_max_drawdown(trades)

    if max_drawdown_pct == 0:
        return float('inf')  # No drawdown

    calmar_ratio = total_return / max_drawdown_pct
    return calmar_ratio


def calculate_summary_metrics(trades):
    """ Calculate summary metrics for a list of trades """

    if not trades or len(trades) == 0:
        logger.error('No trades provided to calculate_summary_metrics')
        return {}

    # ===== BASIC TRADE STATISTICS =====
    total_trades = len(trades)

    # Calculate win rate
    winning_trades = [trade for trade in trades if trade['return_percentage_of_margin'] > 0]
    losing_trades = [trade for trade in trades if trade['return_percentage_of_margin'] <= 0]
    win_count = len(winning_trades)
    loss_count = len(losing_trades)

    win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0

    # Calculate average trade duration
    avg_duration_hours = sum(trade['duration_hours'] for trade in trades) / total_trades if total_trades > 0 else 0

    # ===== DOLLAR-BASED METRICS =====
    # Net PnL calculations
    total_net_pnl = sum(trade['net_pnl'] for trade in trades)
    total_net_profit = sum(trade['net_pnl'] for trade in winning_trades)
    total_net_loss = sum(trade['net_pnl'] for trade in losing_trades)

    # Averages based on net values
    avg_trade_pnl = total_net_pnl / total_trades if total_trades > 0 else 0
    avg_win = total_net_profit / win_count if win_count > 0 else 0
    avg_loss = total_net_loss / loss_count if loss_count > 0 else 0

    # Total margin used (sum of individual trade margin requirements)
    total_margin_used = sum(trade.get('margin_requirement', 0) for trade in trades)

    # Average margin requirement per trade
    avg_margin_requirement = total_margin_used / total_trades if total_trades > 0 else 0

    # ===== NORMALIZED METRICS (PERCENTAGES) =====
    # Return percentages
    total_return_percentage_of_margin = sum(trade['return_percentage_of_margin'] for trade in trades)
    average_trade_return_percentage_of_margin = total_return_percentage_of_margin / total_trades if total_trades > 0 else 0
    average_win_percentage_of_margin = sum(
        trade['return_percentage_of_margin'] for trade in winning_trades
    ) / win_count if win_count > 0 else 0
    average_loss_percentage_of_margin = sum(
        trade['return_percentage_of_margin'] for trade in losing_trades
    ) / loss_count if loss_count > 0 else 0

    # ===== COMMISSION METRICS =====
    # Total commission paid
    total_commission_paid = sum(trade.get('commission', 0) for trade in trades)

    # Commission as percentage of margin
    commission_percentage_of_margin = (total_commission_paid / total_margin_used) * 100 if total_margin_used > 0 else 0

    # ===== RISK METRICS =====
    # Profit factor
    profit_factor = abs(total_net_profit / total_net_loss) if total_net_loss != 0 else float('inf')

    # Calculate drawdown
    max_drawdown, maximum_drawdown_percentage = calculate_max_drawdown(trades)

    # Calculate return to a drawdown ratio (reward to risk)
    return_to_drawdown_ratio = total_return_percentage_of_margin / maximum_drawdown_percentage if maximum_drawdown_percentage > 0 else float(
        'inf'
    )

    # Calculate performance ratios
    sharpe_ratio = calculate_sharpe_ratio(trades)
    sortino_ratio = calculate_sortino_ratio(trades)
    calmar_ratio = calculate_calmar_ratio(trades)

    return {
        # Basic info
        'total_trades': total_trades,
        'winning_trades': win_count,
        'losing_trades': loss_count,
        'win_rate': round(win_rate, 2),
        'avg_trade_duration_hours': round(avg_duration_hours, 2),

        # Percentage-based metrics
        'total_return_percentage_of_margin': round(total_return_percentage_of_margin, 2),
        'average_trade_return_percentage_of_margin': round(average_trade_return_percentage_of_margin, 2),
        'average_win_percentage_of_margin': round(average_win_percentage_of_margin, 2),
        'average_loss_percentage_of_margin': round(average_loss_percentage_of_margin, 2),
        'commission_percentage_of_margin': round(commission_percentage_of_margin, 2),

        # Risk metrics
        'profit_factor': round(profit_factor, 2),
        'maximum_drawdown_percentage': maximum_drawdown_percentage,
        'return_to_drawdown_ratio': round(return_to_drawdown_ratio, 2),
        'sharpe_ratio': round(sharpe_ratio, 2),
        'sortino_ratio': round(sortino_ratio, 2),
        'calmar_ratio': round(calmar_ratio, 2),
    }


def print_summary_metrics(summary):
    """ Print summary metrics in a formatted way. """
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'

    average_trade_return_percentage = summary['average_trade_return_percentage_of_margin']
    if average_trade_return_percentage > 0:
        color = GREEN
    elif average_trade_return_percentage < 0:
        color = RED
    else:
        color = RESET  # default terminal color

    print('\n====== SUMMARY METRICS ======')

    # ===== BASIC TRADE STATISTICS =====
    print('\n--- BASIC TRADE STATISTICS ---')
    print(f'Total Trades: {summary["total_trades"]}')
    print(f'Winning Trades: {summary["winning_trades"]} ({summary["win_rate"]}%)')
    print(f'Losing Trades: {summary["losing_trades"]}')
    print(f'Avg Trade Duration: {summary["avg_trade_duration_hours"]} hours')

    # ===== PERCENTAGE-BASED METRICS =====
    print('\n--- PERCENTAGE-BASED METRICS ---')
    print(f'Total Return Percentage of Margin: {color}{summary["total_return_percentage_of_margin"]}%{RESET}')
    print(f'Average Trade Return Percentage of Margin: {color}{average_trade_return_percentage}%{RESET}')
    print(f'Average Win Percentage of Margin: {GREEN}{summary["average_win_percentage_of_margin"]}%{RESET}')
    print(f'Average Loss Percentage of Margin: {RED}{summary["average_loss_percentage_of_margin"]}%{RESET}')
    print(f'Commission Percentage of Margin: {summary["commission_percentage_of_margin"]}%')

    # ===== RISK METRICS =====
    print('\n--- RISK METRICS ---')
    print(f'Profit Factor: {summary["profit_factor"]}')
    print(f'Maximum Drawdown Percentage: {summary.get("maximum_drawdown_percentage", 0)}%')
    print(f'Return to Drawdown Ratio: {summary.get("return_to_drawdown_ratio", 0)}')
    print(f'Max Consecutive Wins: {summary.get("max_consecutive_wins", 0)}')
    print(f'Max Consecutive Losses: {summary.get("max_consecutive_losses", 0)}')
    print(f'Sharpe Ratio: {summary.get("sharpe_ratio", 0)}')
    print(f'Sortino Ratio: {summary.get("sortino_ratio", 0)}')
    print(f'Calmar Ratio: {summary.get("calmar_ratio", 0)}')

    print('=============================\n')
