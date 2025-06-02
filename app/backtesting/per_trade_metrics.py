from app.utils.logger import get_logger
from config import CONTRACT_MULTIPLIERS, MARGIN_REQUIREMENTS

logger = get_logger()

# Fixed commission per trade in dollars
COMMISSION_PER_TRADE = 4


def calculate_trade_metrics(trade, symbol):
    """ Calculate metrics for a single trade. """

    # Create a copy of the trade to avoid modifying the original
    trade_with_metrics = trade.copy()

    # Get the contract multiplier for the symbol
    contract_multiplier = CONTRACT_MULTIPLIERS.get(symbol)
    if contract_multiplier is None or contract_multiplier == 0:
        logger.error(f"No contract multiplier found for symbol: {symbol}")
        raise ValueError(f"No contract multiplier found for symbol: {symbol}")

    # Get the margin requirement for the symbol
    margin_requirement = MARGIN_REQUIREMENTS.get(symbol)
    if margin_requirement is None or margin_requirement == 0:
        logger.error(f"No margin requirement found for symbol: {symbol}")
        raise ValueError(f"No margin requirement found for symbol: {symbol}")

    # ===== TRADE DETAILS =====
    # Calculate trade duration
    trade_duration = trade['exit_time'] - trade['entry_time']
    trade_with_metrics['duration'] = trade_duration
    trade_with_metrics['duration_hours'] = trade_duration.total_seconds() / 3600

    # ===== NORMALIZED METRICS (PERCENTAGES) =====
    # Fixed commission per trade
    total_commission = COMMISSION_PER_TRADE

    # Calculate commission as percentage of margin
    commission_percentage_of_margin = (total_commission / margin_requirement) * 100
    trade_with_metrics['commission_percentage_of_margin'] = round(commission_percentage_of_margin, 2)

    # Calculate PnL in points
    if trade['side'] == 'long':
        pnl_points = trade['exit_price'] - trade['entry_price']
    elif trade['side'] == 'short':
        pnl_points = trade['entry_price'] - trade['exit_price']
    else:
        logger.error(f"Unknown trade side: {trade['side']}")
        raise ValueError(f"Unknown trade side: {trade['side']}")

    # Calculate gross PnL (before commission)
    gross_pnl = pnl_points * contract_multiplier

    # Calculate net PnL (after commission)
    net_pnl = gross_pnl - total_commission

    # Calculate return percentage as a percentage of the margin requirement
    return_percentage_of_margin = (net_pnl / margin_requirement) * 100
    trade_with_metrics['return_percentage_of_margin'] = round(return_percentage_of_margin, 2)

    # Calculate return percentage as a percentage of the contract value (entry price)
    return_percentage_of_contract = (net_pnl / (trade['entry_price'] * contract_multiplier)) * 100
    trade_with_metrics['return_percentage_of_contract'] = round(return_percentage_of_contract, 2)

    # ===== DOLLAR-BASED METRICS (FOR REFERENCE) =====
    trade_with_metrics['margin_requirement'] = round(margin_requirement, 2)
    trade_with_metrics['commission'] = round(total_commission, 2)
    trade_with_metrics['pnl_points'] = round(pnl_points, 2)
    trade_with_metrics['gross_pnl'] = round(gross_pnl, 2)
    trade_with_metrics['net_pnl'] = round(net_pnl, 2)

    print_trade_metrics(trade_with_metrics)

    # Structure of trade_with_metrics dictionary
    return {
        # Original trade data
        "entry_time": trade_with_metrics['entry_time'],
        "exit_time": trade_with_metrics['exit_time'],
        "side": trade_with_metrics['side'],
        "entry_price": trade_with_metrics['entry_price'],
        "exit_price": trade_with_metrics['exit_price'],
        # Trade details
        "duration": trade_with_metrics['duration'],
        "duration_hours": round(trade_with_metrics['duration_hours'], 2),
        # Normalized metrics (percentages)
        "commission_percentage_of_margin": trade_with_metrics['commission_percentage_of_margin'],
        "return_percentage_of_margin": trade_with_metrics['return_percentage_of_margin'],
        "return_percentage_of_contract": trade_with_metrics['return_percentage_of_contract'],
        # Dollar-based metrics
        "margin_requirement": trade_with_metrics['margin_requirement'],
        "commission": trade_with_metrics['commission'],
        "pnl_points": trade_with_metrics['pnl_points'],
        "gross_pnl": trade_with_metrics['gross_pnl'],
        "net_pnl": trade_with_metrics['net_pnl']
    }


def print_trade_metrics(trade):
    """ Print metrics for a single trade in a formatted way. """
    # Define colors for better visualization
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    # Determine colors based on return percentage
    return_percentage = trade['return_percentage_of_margin']
    if return_percentage > 0:
        color = GREEN
    elif return_percentage < 0:
        color = RED
    else:
        color = RESET

    print("\n====== TRADE METRICS ======")

    # Trade details
    print(f"Entry Time: {trade['entry_time']}")
    print(f"Exit Time: {trade['exit_time']}")
    print(f"Duration: {trade['duration']} ({trade['duration_hours']:.2f} hours)")
    print(f"Side: {trade['side']}")
    print(f"Entry Price: {trade['entry_price']}")
    print(f"Exit Price: {trade['exit_price']}")

    # Normalized metrics (percentages)
    print("\n--- NORMALIZED METRICS (PERCENTAGES) ---")
    print(f"Commission as % of Margin: {trade['commission_percentage_of_margin']}%")
    print(f"Net Return % of Margin: {color}{trade['return_percentage_of_margin']}%{RESET}")
    print(f"Return % of Contract: {color}{trade['return_percentage_of_contract']}%{RESET}")

    # Dollar-based metrics (for reference)
    print("\n--- DOLLAR-BASED METRICS (FOR REFERENCE) ---")
    print(f"Margin Requirement: ${trade['margin_requirement']}")
    print(f"Commission (dollars): ${trade['commission']}")
    print(f"PnL (points): {color}{trade['pnl_points']}{RESET}")
    print(f"Gross PnL (dollars): {color}${trade['gross_pnl']}{RESET}")
    print(f"Net PnL (dollars): {color}${trade['net_pnl']}{RESET}")

    print("=============================\n")
