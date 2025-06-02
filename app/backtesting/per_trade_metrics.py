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
    if contract_multiplier is None:
        logger.error(f"No contract multiplier found for symbol: {symbol}")
        raise ValueError(f"No contract multiplier found for symbol: {symbol}")

    # Get the margin requirement for the symbol
    margin_requirement = MARGIN_REQUIREMENTS.get(symbol)
    if margin_requirement is None:
        logger.error(f"No margin requirement found for symbol: {symbol}")
        raise ValueError(f"No margin requirement found for symbol: {symbol}")
    trade_with_metrics['margin_requirement'] = round(margin_requirement, 2)

    # Notional value of the contract
    contract_value = trade['entry_price'] * contract_multiplier
    trade_with_metrics['contract_value'] = round(contract_value, 2)

    # Calculate trade duration
    trade_duration = trade['exit_time'] - trade['entry_time']
    trade_with_metrics['duration'] = trade_duration
    trade_with_metrics['duration_hours'] = trade_duration.total_seconds() / 3600

    # Calculate PnL in points
    if trade['side'] == 'long':
        pnl_points = trade['exit_price'] - trade['entry_price']
    elif trade['side'] == 'short':
        pnl_points = trade['entry_price'] - trade['exit_price']
    else:
        logger.error(f"Unknown trade side: {trade['side']}")
        raise ValueError(f"Unknown trade side: {trade['side']}")

    trade_with_metrics['pnl_points'] = round(pnl_points, 2)

    # Calculate PnL as a percentage of entry price (normalized metric)
    pnl_pct_of_price = (pnl_points / trade['entry_price']) * 100
    trade_with_metrics['pnl_pct_of_price'] = round(pnl_pct_of_price, 2)

    # Calculate PnL in dollars (for backward compatibility)
    pnl_dollars = pnl_points * contract_multiplier
    trade_with_metrics['pnl_dollars'] = round(pnl_dollars, 2)

    # Fixed commission per trade
    total_commission = COMMISSION_PER_TRADE
    trade_with_metrics['commission'] = round(total_commission, 2)

    # Calculate commission as percentage of margin (normalized metric)
    commission_pct_of_margin = (total_commission / margin_requirement) * 100
    trade_with_metrics['commission_pct_of_margin'] = round(commission_pct_of_margin, 2)

    # Calculate net PnL (after commission)
    net_pnl = pnl_dollars - total_commission
    trade_with_metrics['net_pnl'] = round(net_pnl, 2)

    # Calculate return percentage based on the initial margin requirement (primary normalized metric)
    return_pct = (net_pnl / margin_requirement) * 100
    trade_with_metrics['return_pct'] = round(return_pct, 2)

    print_trade_metrics(trade_with_metrics)

    return trade_with_metrics


def print_trade_metrics(trade):
    """ Print metrics for a single trade in a formatted way. """
    # Define colors for better visualization
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    # Determine colors based on return percentage
    return_pct = trade['return_pct']
    if return_pct > 0:
        color = GREEN
    elif return_pct < 0:
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
    print(f"Return on Margin: {color}{trade['return_pct']}%{RESET}")
    print(f"PnL as % of Price: {color}{trade['pnl_pct_of_price']}%{RESET}")
    print(f"Commission as % of Margin: {trade['commission_pct_of_margin']}%")

    # Dollar-based metrics (for reference)
    print("\n--- DOLLAR-BASED METRICS (FOR REFERENCE) ---")
    print(f"Margin Requirement: ${trade['margin_requirement']}")
    print(f"Commission: ${trade['commission']}")
    print(f"PnL (points): {color}{trade['pnl_points']}{RESET}")
    print(f"PnL (dollars): {color}${trade['pnl_dollars']}{RESET}")
    print(f"Net PnL: {color}${trade['net_pnl']}{RESET}")

    print("=============================\n")
