from app.utils.logger import get_logger

logger = get_logger('backtesting/summary_metrics')


def calculate_max_drawdown(trades):
    """Calculate the maximum drawdown given a list of trades. """

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
    # Initialize peak to the first value in the array
    peak_dollars = cumulative_pnl_dollars[0]
    max_drawdown = 0
    for val in cumulative_pnl_dollars:
        if val > peak_dollars:
            peak_dollars = val
        drawdown = peak_dollars - val
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # Calculate max drawdown in percentage (for normalized comparison)
    # Initialize peak to the first value in the array
    peak_pct = cumulative_pnl_pct[0]
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
    """Calculate Sharpe ratio: (Average Return - Risk-Free Rate) / Standard Deviation of Returns."""
    if len(trades) < 2:  # Need at least 2 trades for standard deviation
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
    """Calculate Sortino ratio: (Average Return - Risk-Free Rate) / Standard Deviation of Negative Returns."""

    returns = [trade['return_percentage_of_margin'] for trade in trades]
    avg_return = sum(returns) / len(returns)

    # Calculate downside deviation (returns below the risk-free rate)
    negative_returns = [r - risk_free_rate for r in returns if r < risk_free_rate]

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

    # Calculate annualized return (assuming percentage returns)
    total_return = sum(trade['return_percentage_of_margin'] for trade in trades)

    # Get maximum drawdown percentage
    _, max_drawdown_pct = calculate_max_drawdown(trades)

    if max_drawdown_pct == 0:
        return float('inf')  # No drawdown

    calmar_ratio = total_return / max_drawdown_pct
    return calmar_ratio


def calculate_profit_factor(trades):
    """Calculate profit factor: Total Net Profit / Total Net Loss."""

    winning_trades = [trade for trade in trades if trade['return_percentage_of_margin'] > 0]
    losing_trades = [trade for trade in trades if trade['return_percentage_of_margin'] <= 0]

    total_net_profit = sum(trade['net_pnl'] for trade in winning_trades)
    total_net_loss = sum(trade['net_pnl'] for trade in losing_trades)

    if total_net_loss == 0:
        return float('inf')  # No losses

    profit_factor = abs(total_net_profit / total_net_loss)
    return profit_factor


def calculate_return_to_drawdown_ratio(trades):
    """Calculate return-to-drawdown ratio: Total Return Percentage / Maximum Drawdown Percentage."""

    total_return_percentage_of_margin = sum(trade['return_percentage_of_margin'] for trade in trades)

    # Get maximum drawdown percentage
    _, maximum_drawdown_percentage = calculate_max_drawdown(trades)

    if maximum_drawdown_percentage == 0:
        return float('inf')  # No drawdown

    return_to_drawdown_ratio = total_return_percentage_of_margin / maximum_drawdown_percentage
    return return_to_drawdown_ratio


def calculate_win_rate(trades):
    """Calculate win rate: (Number of Winning Trades / Total Trades) * 100."""

    total_trades = len(trades)
    winning_trades = [trade for trade in trades if trade['return_percentage_of_margin'] > 0]
    losing_trades = [trade for trade in trades if trade['return_percentage_of_margin'] <= 0]
    win_count = len(winning_trades)
    loss_count = len(losing_trades)

    win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0

    return win_rate, win_count, loss_count, winning_trades, losing_trades


def calculate_average_trade_duration(trades):
    """Calculate average trade duration in hours."""

    total_trades = len(trades)
    avg_duration_hours = sum(trade['duration_hours'] for trade in trades) / total_trades if total_trades > 0 else 0

    return avg_duration_hours


def calculate_total_margin_used(trades):
    """Calculate the total margin used across all trades."""

    total_margin_used = sum(trade.get('margin_requirement', 0) for trade in trades)

    return total_margin_used


def calculate_total_return_percentage_of_margin(trades):
    """Calculate total return percentage of margin."""

    total_return_percentage_of_margin = sum(trade['return_percentage_of_margin'] for trade in trades)

    return total_return_percentage_of_margin


def calculate_average_trade_return_percentage_of_margin(trades):
    """Calculate average trade return percentage of margin."""

    total_trades = len(trades)
    total_return_percentage_of_margin = calculate_total_return_percentage_of_margin(trades)
    average_trade_return_percentage_of_margin = total_return_percentage_of_margin / total_trades if total_trades > 0 else 0

    return average_trade_return_percentage_of_margin


def calculate_average_win_percentage_of_margin(winning_trades):
    """Calculate average win percentage of margin."""

    win_count = len(winning_trades)
    average_win_percentage_of_margin = sum(
        trade['return_percentage_of_margin'] for trade in winning_trades
    ) / win_count if win_count > 0 else 0

    return average_win_percentage_of_margin


def calculate_average_loss_percentage_of_margin(losing_trades):
    """Calculate average loss percentage of margin."""

    loss_count = len(losing_trades)
    average_loss_percentage_of_margin = sum(
        trade['return_percentage_of_margin'] for trade in losing_trades
    ) / loss_count if loss_count > 0 else 0

    return average_loss_percentage_of_margin


def calculate_commission_percentage_of_margin(trades):
    """Calculate commission as percentage of margin."""

    total_commission_paid = sum(trade.get('commission', 0) for trade in trades)
    total_margin_used = calculate_total_margin_used(trades)

    commission_percentage_of_margin = (total_commission_paid / total_margin_used) * 100 if total_margin_used > 0 else 0

    return commission_percentage_of_margin


def calculate_summary_metrics(trades):
    """ Calculate summary metrics for a list of trades """

    if not trades or len(trades) == 0:
        logger.error('No trades provided to calculate_summary_metrics')
        return {}

    # ===== BASIC TRADE STATISTICS =====
    total_trades = len(trades)

    # Calculate win rate and get winning/losing trades
    win_rate, win_count, loss_count, winning_trades, losing_trades = calculate_win_rate(trades)

    # Calculate average trade duration
    avg_duration_hours = calculate_average_trade_duration(trades)

    # ===== DOLLAR-BASED METRICS =====
    # Total margin used (sum of individual trade margin requirements)
    total_margin_used = calculate_total_margin_used(trades)

    # ===== NORMALIZED METRICS (PERCENTAGES) =====
    total_return_percentage_of_margin = calculate_total_return_percentage_of_margin(trades)
    average_trade_return_percentage_of_margin = calculate_average_trade_return_percentage_of_margin(trades)
    average_win_percentage_of_margin = calculate_average_win_percentage_of_margin(winning_trades)
    average_loss_percentage_of_margin = calculate_average_loss_percentage_of_margin(losing_trades)

    # ===== COMMISSION METRICS =====
    commission_percentage_of_margin = calculate_commission_percentage_of_margin(trades)

    # ===== RISK METRICS =====
    # Profit factor
    profit_factor = calculate_profit_factor(trades)

    # Drawdown
    max_drawdown, maximum_drawdown_percentage = calculate_max_drawdown(trades)
    return_to_drawdown_ratio = calculate_return_to_drawdown_ratio(trades)

    # Sharpe Ratio
    sharpe_ratio = calculate_sharpe_ratio(trades)

    # Sortino Ratio
    sortino_ratio = calculate_sortino_ratio(trades)

    # Calmar Ratio
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
    if summary["total_return_percentage_of_margin"] == 0:
        print(f'Total Return Percentage of Margin: 0.0%')
    else:
        print(f'Total Return Percentage of Margin: {color}{summary["total_return_percentage_of_margin"]}%{RESET}')
    if average_trade_return_percentage == 0:
        print(f'Average Trade Return Percentage of Margin: 0.0%')
    else:
        print(f'Average Trade Return Percentage of Margin: {color}{average_trade_return_percentage}%{RESET}')
    print(f'Average Win Percentage of Margin: {GREEN}{summary["average_win_percentage_of_margin"]}%{RESET}')
    print(f'Average Loss Percentage of Margin: {RED}{summary["average_loss_percentage_of_margin"]}%{RESET}')
    print(f'Commission Percentage of Margin: {summary["commission_percentage_of_margin"]}%')

    # ===== RISK METRICS =====
    print('\n--- RISK METRICS ---')
    print(f'Profit Factor: {summary["profit_factor"]}')
    print(f'Maximum Drawdown Percentage: {summary.get("maximum_drawdown_percentage", 0)}%')
    print(f'Return to Drawdown Ratio: {summary.get("return_to_drawdown_ratio", 0)}')
    print(f'Sharpe Ratio: {summary.get("sharpe_ratio", 0)}')
    print(f'Sortino Ratio: {summary.get("sortino_ratio", 0)}')
    print(f'Calmar Ratio: {summary.get("calmar_ratio", 0)}')

    print('=============================\n')
