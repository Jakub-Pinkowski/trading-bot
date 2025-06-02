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
        cum_sum_dollars += trade['pnl_dollars']
        cum_sum_pct += trade['return_pct']
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
    max_drawdown_pct = 0
    for val in cumulative_pnl_pct:
        if val > peak_pct:
            peak_pct = val
        drawdown_pct = peak_pct - val
        if drawdown_pct > max_drawdown_pct:
            max_drawdown_pct = drawdown_pct

    return round(max_drawdown, 2), round(max_drawdown_pct, 2)


def calculate_summary_metrics(trades):
    """ Calculate summary metrics for a list of trades """

    if not trades:
        logger.error("No trades provided to calculate_summary_metrics")

    total_trades = len(trades)

    # Use percentage returns for determining winning/losing trades
    winning_trades = [trade for trade in trades if trade['return_pct'] > 0]
    losing_trades = [trade for trade in trades if trade['return_pct'] <= 0]
    win_count = len(winning_trades)
    loss_count = len(losing_trades)

    win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0

    # Gross PnL sums (keep for backward compatibility)
    total_gross_pnl = sum(trade['pnl_dollars'] for trade in trades)
    total_gross_profit = sum(trade['pnl_dollars'] for trade in winning_trades) if winning_trades else 0
    total_gross_loss = sum(trade['pnl_dollars'] for trade in losing_trades) if losing_trades else 0

    # Profit factor
    profit_factor = abs(total_gross_profit / total_gross_loss) if total_gross_loss != 0 else float('inf')

    # Averages based on gross values (for backward compatibility)
    avg_trade_pnl = total_gross_pnl / total_trades if total_trades > 0 else 0
    avg_win = total_gross_profit / win_count if win_count > 0 else 0
    avg_loss = total_gross_loss / loss_count if loss_count > 0 else 0

    avg_duration_hours = sum(trade['duration_hours'] for trade in trades) / total_trades if total_trades > 0 else 0

    # Percentage-based metrics (for normalized comparison)
    total_return_pct = sum(trade['return_pct'] for trade in trades)
    avg_trade_return_pct = total_return_pct / total_trades if total_trades > 0 else 0
    avg_win_pct = sum(trade['return_pct'] for trade in winning_trades) / win_count if win_count > 0 else 0
    avg_loss_pct = sum(trade['return_pct'] for trade in losing_trades) / loss_count if loss_count > 0 else 0

    # Total commission paid
    total_commission_paid = sum(trade.get('commission', 0) for trade in trades)

    # Total margin used (sum of individual trade margin requirements)
    total_margin_used = sum(trade.get('margin_requirement', 0) for trade in trades)

    # Commission as % of margin
    commission_pct_on_margin = (
        (total_commission_paid / total_margin_used) * 100
        if total_margin_used > 0 else 0
    )

    # Calculate drawdown
    max_drawdown, max_drawdown_pct = calculate_max_drawdown(trades)

    return {
        "total_trades": total_trades,
        "winning_trades": win_count,
        "losing_trades": loss_count,
        "win_rate": round(win_rate, 2),
        "avg_trade_duration_hours": round(avg_duration_hours, 2),
        "total_margin_used": round(total_margin_used, 2),
        # Dollar-based metrics (for backward compatibility)
        "total_gross_pnl": round(total_gross_pnl, 2),
        "avg_trade_gross_pnl": round(avg_trade_pnl, 2),
        "avg_win_gross": round(avg_win, 2),
        "avg_loss_gross": round(avg_loss, 2),
        # Percentage-based metrics (for normalized comparison)
        "total_return_pct": round(total_return_pct, 2),
        "avg_trade_return_pct": round(avg_trade_return_pct, 2),
        "avg_win_pct": round(avg_win_pct, 2),
        "avg_loss_pct": round(avg_loss_pct, 2),
        # Other metrics
        "profit_factor": round(profit_factor, 2),
        "total_commission_paid": round(total_commission_paid, 2),
        "commission_pct_on_margin": round(commission_pct_on_margin, 2),
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
    }


def print_summary_metrics(summary):
    """ Print summary metrics in a formatted way. """
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    avg_trade_return_pct = summary['avg_trade_return_pct']
    if avg_trade_return_pct > 0:
        color = GREEN
    elif avg_trade_return_pct < 0:
        color = RED
    else:
        color = RESET  # default terminal color

    print("\n====== SUMMARY METRICS ======")
    print(f"Total Trades: {summary['total_trades']}")
    print(f"Winning Trades: {summary['winning_trades']} ({summary['win_rate']}%)")
    print(f"Losing Trades: {summary['losing_trades']}")
    print(f"Avg Trade Duration: {summary['avg_trade_duration_hours']} hours")

    # Percentage-based metrics (normalized)
    print("\n--- NORMALIZED METRICS (PERCENTAGES) ---")
    print(f"Total Return: {color}{summary['total_return_pct']}%{RESET}")
    print(f"Avg Trade Return: {color}{avg_trade_return_pct}%{RESET}")
    print(f"Avg Win: {GREEN}{summary['avg_win_pct']}%{RESET}")
    print(f"Avg Loss: {RED}{summary['avg_loss_pct']}%{RESET}")
    print(f"Max Drawdown: {RED}{summary.get('max_drawdown_pct', 0)}%{RESET}")

    # Dollar-based metrics (for reference)
    print("\n--- DOLLAR-BASED METRICS (FOR REFERENCE) ---")
    print(f"Total Money Invested (Margin): ${summary.get('total_margin_used', 0)}")
    print(f"Total Gross PnL: ${summary['total_gross_pnl']} ")
    print(f"Avg Trade PnL: ${summary['avg_trade_gross_pnl']} ")
    print(f"Avg Win: ${summary['avg_win_gross']}")
    print(f"Avg Loss: ${summary['avg_loss_gross']}")
    print(f"Max Drawdown: ${summary.get('max_drawdown', 0)}")

    # Other metrics
    print("\n--- OTHER METRICS ---")
    print(f"Profit Factor: {summary['profit_factor']}")
    print(f"Total Commission Paid: ${summary['total_commission_paid']}")
    print(f"Commission as % of Margin: {summary['commission_pct_on_margin']}%")
    print("=============================\n")
