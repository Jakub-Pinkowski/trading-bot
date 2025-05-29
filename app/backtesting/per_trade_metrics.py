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

    # Notional value of the contract
    contract_value = trade['entry_price'] * contract_value
    trade_with_metrics['contract_value'] = round(contract_value, 2)

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

    # Fixed commission per trade
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


def print_trade_metrics(trade):
    """ Print metrics for a single trade in a formatted way. """
    print("\n=== TRADE METRICS ===")
    print(f"Entry: {trade['entry_time']} at {trade['entry_price']}")
    print(f"Exit: {trade['exit_time']} at {trade['exit_price']}")
    print(f"Side: {trade['side']}")
    print(f"Contract value: ${trade['contract_value']}")
    print(f"Duration: {trade['duration']} ({trade['duration_hours']:.2f} hours)")
    print(f"PnL (points): {trade['pnl_points']}")
    print(f"PnL (dollars): ${trade['pnl_dollars']}")
    print(f"Commission: ${trade['commission']}")
    print(f"Net PnL: ${trade['net_pnl']}")
    print(f"Return: {trade['return_pct']}%")
    print("=====================\n")
