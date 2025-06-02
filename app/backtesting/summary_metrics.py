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
        cum_sum_dollars += trade['gross_pnl']
        cum_sum_pct += trade['return_percentage_of_margin']
        cumulative_pnl_dollars.append(cum_sum_dollars)
        cumulative_pnl_pct.append(cum_sum_pct)

    # Calculate max drawdown in dollars (for backward compatibility)
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


# TODO [MEDIUM]: Get rid of all those 0-value checks
def calculate_summary_metrics(trades):
    """ Calculate summary metrics for a list of trades """

    if not trades or len(trades) == 0:
        logger.error("No trades provided to calculate_summary_metrics")

    # ===== BASIC TRADE STATISTICS =====
    total_trades = len(trades)

    # Calculate win rate
    winning_trades = [trade for trade in trades if trade['return_percentage_of_margin'] > 0]
    losing_trades = [trade for trade in trades if trade['return_percentage_of_margin'] <= 0]
    win_count = len(winning_trades)
    loss_count = len(losing_trades)

    win_rate = (win_count / total_trades) * 100

    # Calculate average trade duration
    avg_duration_hours = sum(trade['duration_hours'] for trade in trades) / total_trades

    # TODO [MEDIUM]: Should be calculated based on net data instead
    # ===== DOLLAR-BASED METRICS =====
    # Gross PnL calculations
    total_gross_pnl = sum(trade['gross_pnl'] for trade in trades)
    total_gross_profit = sum(trade['gross_pnl'] for trade in winning_trades) if winning_trades else 0
    total_gross_loss = sum(trade['gross_pnl'] for trade in losing_trades) if losing_trades else 0

    # Averages based on gross values
    avg_trade_pnl = total_gross_pnl / total_trades
    avg_win = total_gross_profit / win_count if win_count > 0 else 0
    avg_loss = total_gross_loss / loss_count if loss_count > 0 else 0

    # Total margin used (sum of individual trade margin requirements)
    total_margin_used = sum(trade.get('margin_requirement', 0) for trade in trades)

    # ===== NORMALIZED METRICS (PERCENTAGES) =====
    # Return percentages
    total_return_percentage_of_margin = sum(trade['return_percentage_of_margin'] for trade in trades)
    average_trade_return_percentage_of_margin = total_return_percentage_of_margin / total_trades
    average_win_percentage_of_margin = sum(
        trade['return_percentage_of_margin'] for trade in winning_trades) / win_count if win_count > 0 else 0
    average_loss_percentage_of_margin = sum(
        trade['return_percentage_of_margin'] for trade in losing_trades) / loss_count if loss_count > 0 else 0

    # ===== COMMISSION METRICS =====
    # Total commission paid
    total_commission_paid = sum(trade.get('commission', 0) for trade in trades)

    # Commission as percentage of margin
    commission_percentage_of_margin = (
        (total_commission_paid / total_margin_used) * 100
        if total_margin_used > 0 else 0
    )

    # TODO [MEDIUM]: Add more risk metrics
    # ===== RISK METRICS =====
    # Profit factor
    profit_factor = abs(total_gross_profit / total_gross_loss) if total_gross_loss != 0 else float('inf')

    # Calculate drawdown
    max_drawdown, maximum_drawdown_percentage = calculate_max_drawdown(trades)

    return {
        # Basic trade statistics
        "total_trades": total_trades,
        "winning_trades": win_count,
        "losing_trades": loss_count,
        "win_rate": round(win_rate, 2),
        "avg_trade_duration_hours": round(avg_duration_hours, 2),

        # Dollar-based metrics
        "total_margin_used": round(total_margin_used, 2),
        "total_gross_pnl": round(total_gross_pnl, 2),
        "avg_trade_gross_pnl": round(avg_trade_pnl, 2),
        "avg_win_gross": round(avg_win, 2),
        "avg_loss_gross": round(avg_loss, 2),

        # Normalized metrics (percentages)
        "total_return_percentage_of_margin": round(total_return_percentage_of_margin, 2),
        "average_trade_return_percentage_of_margin": round(average_trade_return_percentage_of_margin, 2),
        "average_win_percentage_of_margin": round(average_win_percentage_of_margin, 2),
        "average_loss_percentage_of_margin": round(average_loss_percentage_of_margin, 2),

        # Commission metrics
        "total_commission_paid": round(total_commission_paid, 2),
        "commission_percentage_of_margin": round(commission_percentage_of_margin, 2),

        # Risk metrics
        "profit_factor": round(profit_factor, 2),
        "max_drawdown": max_drawdown,
        "maximum_drawdown_percentage": maximum_drawdown_percentage,
    }


def print_summary_metrics(summary):
    """ Print summary metrics in a formatted way. """
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    average_trade_return_percentage = summary['average_trade_return_percentage_of_margin']
    if average_trade_return_percentage > 0:
        color = GREEN
    elif average_trade_return_percentage < 0:
        color = RED
    else:
        color = RESET  # default terminal color

    print("\n====== SUMMARY METRICS ======")

    # ===== BASIC TRADE STATISTICS =====
    print("\n--- BASIC TRADE STATISTICS ---")
    print(f"Total Trades: {summary['total_trades']}")
    print(f"Winning Trades: {summary['winning_trades']} ({summary['win_rate']}%)")
    print(f"Losing Trades: {summary['losing_trades']}")
    print(f"Avg Trade Duration: {summary['avg_trade_duration_hours']} hours")

    # ===== DOLLAR-BASED METRICS =====
    print("\n--- DOLLAR-BASED METRICS ---")
    print(f"Total Money Invested (Margin): ${summary.get('total_margin_used', 0)}")
    print(f"Total Gross PnL: ${summary['total_gross_pnl']} ")
    print(f"Avg Trade PnL: ${summary['avg_trade_gross_pnl']} ")
    print(f"Avg Win: ${summary['avg_win_gross']}")
    print(f"Avg Loss: ${summary['avg_loss_gross']}")

    # ===== NORMALIZED METRICS (PERCENTAGES) =====
    print("\n--- NORMALIZED METRICS (PERCENTAGES) ---")
    print(f"Total Return Percentage of Margin: {color}{summary['total_return_percentage_of_margin']}%{RESET}")
    print(f"Average Trade Return Percentage of Margin: {color}{average_trade_return_percentage}%{RESET}")
    print(f"Average Win Percentage of Margin: {GREEN}{summary['average_win_percentage_of_margin']}%{RESET}")
    print(f"Average Loss Percentage of Margin: {RED}{summary['average_loss_percentage_of_margin']}%{RESET}")

    # ===== COMMISSION METRICS =====
    print("\n--- COMMISSION METRICS ---")
    print(f"Total Commission Paid: ${summary['total_commission_paid']}")
    print(f"Commission Percentage of Margin: {summary['commission_percentage_of_margin']}%")

    # ===== RISK METRICS =====
    print("\n--- RISK METRICS ---")
    print(f"Profit Factor: {summary['profit_factor']}")
    print(f"Max Drawdown: ${summary.get('max_drawdown', 0)}")
    print(f"Maximum Drawdown Percentage: {summary.get('maximum_drawdown_percentage', 0)}%")

    print("=============================\n")
