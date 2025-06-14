from app.utils.logger import get_logger
from config import CONTRACT_MULTIPLIERS, MARGIN_REQUIREMENTS

logger = get_logger('backtesting/per_trade_metrics')

# Fixed commission per trade in dollars
COMMISSION_PER_TRADE = 4


def calculate_trade_metrics(trade, symbol):
    """ Calculate metrics for a single trade. """

    # Create a copy of the trade to avoid modifying the original
    trade = trade.copy()

    # Get the contract multiplier for the symbol
    contract_multiplier = CONTRACT_MULTIPLIERS.get(symbol)
    if contract_multiplier is None or contract_multiplier == 0:
        logger.error(f'No contract multiplier found for symbol: {symbol}')
        raise ValueError(f'No contract multiplier found for symbol: {symbol}')

    # Get the margin requirement for the symbol
    margin_requirement = MARGIN_REQUIREMENTS.get(symbol)
    if margin_requirement is None or margin_requirement == 0:
        logger.error(f'No margin requirement found for symbol: {symbol}')
        raise ValueError(f'No margin requirement found for symbol: {symbol}')

    # ===== TRADE DETAILS =====
    # Calculate trade duration
    trade_duration = trade['exit_time'] - trade['entry_time']
    trade_duration_hours = round(trade_duration.total_seconds() / 3600)

    # ===== NORMALIZED METRICS (PERCENTAGES) =====
    # Fixed commission per trade
    total_commission = COMMISSION_PER_TRADE

    # Calculate PnL in points
    if trade['side'] == 'long':
        pnl_points = trade['exit_price'] - trade['entry_price']
    elif trade['side'] == 'short':
        pnl_points = trade['entry_price'] - trade['exit_price']
    else:
        logger.error(f'Unknown trade side: {trade['side']}')
        raise ValueError(f'Unknown trade side: {trade['side']}')

    # Calculate gross PnL (before commission)
    gross_pnl = pnl_points * contract_multiplier

    # Calculate net PnL (after commission)
    net_pnl = gross_pnl - total_commission

    # Calculate return percentage as a percentage of the margin requirement
    return_percentage_of_margin = round((net_pnl / margin_requirement) * 100, 2)

    # Calculate return percentage as a percentage of the contract value (entry price)
    contract_value = trade['entry_price'] * contract_multiplier
    return_percentage_of_contract = round((net_pnl / contract_value) * 100, 2)

    return {
        # Original trade data
        'entry_time': trade['entry_time'],
        'exit_time': trade['exit_time'],
        'entry_price': trade['entry_price'],
        'exit_price': trade['exit_price'],
        'side': trade['side'],

        # Trade details
        'duration': trade_duration,
        'duration_hours': trade_duration_hours,

        # Normalized metrics (percentages)
        'return_percentage_of_margin': return_percentage_of_margin,
        'return_percentage_of_contract': return_percentage_of_contract,

        # We need these for internal calculations in summary_metrics
        'net_pnl': net_pnl,
        'margin_requirement': margin_requirement,
        'commission': total_commission
    }


def print_trade_metrics(trade):
    """ Print metrics for a single trade in a formatted way. """
    # Define colors for better visualization
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'

    # Determine colors based on return percentage
    return_percentage = trade.get('return_percentage_of_margin', 0.0)
    if return_percentage > 0:
        color = GREEN
    elif return_percentage < 0:
        color = RED
    else:
        color = RESET

    print('\n====== TRADE METRICS ======')

    # Trade details
    print(f'Entry Time: {trade.get('entry_time', 'N/A')}')
    print(f'Exit Time: {trade.get('exit_time', 'N/A')}')

    # Print duration if available
    if 'duration' in trade and 'duration_hours' in trade:
        print(f'Duration: {trade['duration']} ({trade['duration_hours']:.2f} hours)')
    elif 'duration_hours' in trade:
        print(f'Duration: {trade['duration_hours']:.2f} hours')

    print(f'Side: {trade.get('side', 'N/A')}')
    print(f'Entry Price: {trade.get('entry_price', 'N/A')}')
    print(f'Exit Price: {trade.get('exit_price', 'N/A')}')

    # Percentage-based metrics
    print('\n--- PERCENTAGE-BASED METRICS ---')
    print(f'Net Return % of Margin: {color}{return_percentage}%{RESET}')
    print(f'Return % of Contract: {color}{trade.get('return_percentage_of_contract', 0.0)}%{RESET}')

    print('=============================\n')
