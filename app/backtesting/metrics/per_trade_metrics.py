from app.utils.logger import get_logger
from app.utils.math_utils import calculate_percentage
from config import CONTRACT_MULTIPLIERS

logger = get_logger('backtesting/per_trade_metrics')

# ==================== Constants ====================

# Fixed commission per trade in dollars
COMMISSION_PER_TRADE = 4

# Target margin-to-contract-value ratios based on asset classes
# These ratios are derived from current market data (01.2026) to provide a realistic historical estimate
MARGIN_RATIOS = {
    'energies': 0.25,  # ~25% (CL, NG, MCL, MNG,)
    'metals': 0.12,  # ~12% (GC, SI, HG, PL, MGC, SIL, MHG)
    'indices': 0.08,  # ~8%  (ES, NQ, YM, RTY, MES, MNQ, MYM, ZB)
    'forex': 0.04,  # ~4%  (6E, 6J, 6B, 6A, 6C, 6S, M6E, M6J, M6B, M6A, M6C, M6S)
    'crypto': 0.40,  # ~40% (BTC, ETH, MBT, MET)
    'grains': 0.08,  # ~8%  (ZC, ZW, ZS, ZL, XC, XK, XW, MZC, MZL, MZS, MZW)
    'softs': 0.10,  # ~10% (SB, KC, CC)
    'default': 0.10  # ~10% Default fallback
}


# ==================== Helper Functions ====================

def get_symbol_category(symbol):
    """Categorize symbols into asset classes to apply appropriate margin ratios."""
    categories = {
        'energies': ['CL', 'NG', 'MCL', 'MNG', 'HO', 'RB'],
        'metals': ['GC', 'SI', 'HG', 'PL', 'MGC', 'MHG', 'SIL'],
        'indices': ['ES', 'NQ', 'YM', 'RTY', 'MES', 'MNQ', 'MYM', 'ZB'],
        'forex': ['6E', '6J', '6B', '6A', '6C', '6S', 'M6E', 'M6J', 'M6B', 'M6A', 'M6C', 'M6S'],
        'crypto': ['BTC', 'ETH', 'MBT', 'MET'],
        'grains': ['ZC', 'ZW', 'ZS', 'ZL', 'XC', 'XK', 'XW', 'MZC', 'MZL', 'MZS', 'MZW'],
        'softs': ['SB', 'KC', 'CC']
    }
    for category, symbols in categories.items():
        if symbol in symbols:
            return category
    return 'default'


def estimate_margin(symbol, entry_price, contract_multiplier):
    """
    Estimate the margin requirement for a symbol at any given time based on its contract value.
    """
    category = get_symbol_category(symbol)
    ratio = MARGIN_RATIOS.get(category, MARGIN_RATIOS['default'])
    contract_value = entry_price * contract_multiplier
    return contract_value * ratio


# ==================== Public API ====================

def calculate_trade_metrics(trade, symbol):
    """Calculate metrics for a single trade"""

    # Create a copy of the trade to avoid modifying the original
    trade = trade.copy()

    # Get the contract multiplier for the symbol
    contract_multiplier = CONTRACT_MULTIPLIERS.get(symbol)
    if contract_multiplier is None or contract_multiplier == 0:
        logger.error(f'No contract multiplier found for symbol: {symbol}')
        raise ValueError(f'No contract multiplier found for symbol: {symbol}')

    # Estimate the margin requirement for the symbol based on the contract at the time of entry
    margin_requirement = estimate_margin(symbol, trade['entry_price'], contract_multiplier)

    # Calculate trade duration
    trade_duration = trade['exit_time'] - trade['entry_time']
    trade_duration_hours = round(trade_duration.total_seconds() / 3600)

    # Apply fixed commission per trade
    total_commission = COMMISSION_PER_TRADE

    # Calculate PnL in points
    if trade['side'] == 'long':
        pnl_points = trade['exit_price'] - trade['entry_price']
    elif trade['side'] == 'short':
        pnl_points = trade['entry_price'] - trade['exit_price']
    else:
        logger.error(f"Unknown trade side: {trade['side']}")
        raise ValueError(f'Unknown trade side: {trade['side']}')

    # Calculate gross PnL (before commission)
    gross_pnl = pnl_points * contract_multiplier

    # Calculate net PnL (after commission)
    net_pnl = gross_pnl - total_commission

    # Calculate return percentage as a percentage of the margin requirement
    return_percentage_of_margin = calculate_percentage(net_pnl, margin_requirement)

    # Calculate return percentage as a percentage of the contract value (entry price)
    contract_value = trade['entry_price'] * contract_multiplier
    return_percentage_of_contract = calculate_percentage(net_pnl, contract_value)

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
    """Print metrics for a single trade in a formatted way"""
    # Define colors for better visualization
    green = '\033[92m'
    red = '\033[91m'
    reset = '\033[0m'

    # Determine colors based on return percentage
    return_percentage = trade.get('return_percentage_of_margin', 0.0)
    if return_percentage > 0:
        color = green
    elif return_percentage < 0:
        color = red
    else:
        color = reset

    print('\n====== TRADE METRICS ======')

    # Trade details
    print(f"Entry Time: {trade.get('entry_time', 'N/A')}")
    print(f"Exit Time: {trade.get('exit_time', 'N/A')}")

    # Print duration if available
    if 'duration' in trade and 'duration_hours' in trade:
        print(f"Duration: {trade['duration']} ({trade['duration_hours']:.2f} hours)")
    elif 'duration_hours' in trade:
        print(f"Duration: {trade['duration_hours']:.2f} hours")

    print(f"Side: {trade.get('side', 'N/A')}")
    print(f"Entry Price: {trade.get('entry_price', 'N/A')}")
    print(f"Exit Price: {trade.get('exit_price', 'N/A')}")

    # Percentage-based metrics
    print('\n--- PERCENTAGE-BASED METRICS ---')
    print(f"Net Return % of Margin: {color}{return_percentage}%{reset}")
    print(f"Return % of Contract: {color}{trade.get('return_percentage_of_contract', 0.0)}%{reset}")

    print('=============================\n')
