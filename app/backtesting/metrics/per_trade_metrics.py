from app.utils.logger import get_logger
from app.utils.math_utils import calculate_percentage
from config import get_contract_multiplier

logger = get_logger('backtesting/per_trade_metrics')

# ==================== Constants ====================

# Fixed commission per trade in dollars
COMMISSION_PER_TRADE = 4

# Target margin-to-contract-value ratios based on asset classes
# These ratios are derived from current market data (01.2026) to provide a realistic historical estimate
MARGIN_RATIOS = {
    'energies': 0.25,  # ~25% (CL, NG, MCL, MNG)
    'metals': 0.12,  # ~12% (GC, SI, HG, PL, MGC, SIL, MHG)
    'indices': 0.08,  # ~8%  (ES, NQ, YM, RTY, MES, MNQ, MYM, ZB)
    'forex': 0.04,  # ~4%  (6E, 6J, 6B, 6A, 6C, 6S, M6E, M6J, M6B, M6A, M6C, M6S)
    'crypto': 0.40,  # ~40% (BTC, ETH, MBT, MET)
    'grains': 0.08,  # ~8%  (ZC, ZW, ZS, ZL, XC, XK, XW, MZC, MZL, MZS, MZW)
    'softs': 0.10,  # ~10% (SB, KC, CC)
    'default': 0.10  # ~10% Default fallback
}


# ==================== Public API ====================

def calculate_trade_metrics(trade, symbol):
    """
    Calculate comprehensive performance metrics for a single completed trade.

    Computes P&L, returns, duration, and other metrics needed for strategy evaluation.
    Applies commission costs and calculates returns both as percentage of margin (for
    risk-adjusted comparison) and percentage of contract value (for understanding leverage).

    Args:
        trade: Dictionary containing trade details:
              - entry_time: Datetime of entry
              - exit_time: Datetime of exit
              - entry_price: Entry price in symbol units
              - exit_price: Exit price in symbol units
              - side: 'long' or 'short'
        symbol: Futures symbol code (e.g., 'ZS', 'CL', 'GC') for looking up contract specs

    Returns:
        Dictionary with calculated metrics:
        - entry_time, exit_time, entry_price, exit_price, side: Original trade data
        - duration: Timedelta object of trade duration
        - duration_hours: Trade duration in hours (float)
        - return_percentage_of_margin: Return as % of estimated margin (risk-adjusted)
        - return_percentage_of_contract: Return as % of contract value (leverage-aware)
        - net_pnl: Net profit/loss in dollars after commission
        - margin_requirement: Estimated margin in dollars
        - commission: Commission cost in dollars

    Raises:
        ValueError: If symbol not found or has no contract multiplier, margin requirement is invalid,
                   or trade side is not 'long' or 'short'
    """

    # Create a copy of the trade to avoid modifying the original
    trade = trade.copy()

    # Get the contract multiplier for the symbol using helper function
    contract_multiplier = get_contract_multiplier(symbol)
    if contract_multiplier is None:
        logger.error(f'Symbol {symbol} has no contract multiplier defined')
        raise ValueError(f'Symbol {symbol} has no contract multiplier defined')

    # Estimate the margin requirement for the symbol based on the contract at the time of entry
    margin_requirement = _estimate_margin(symbol, trade['entry_price'], contract_multiplier)

    # Validate margin requirement
    if margin_requirement <= 0:
        logger.error(f'Invalid margin requirement: {margin_requirement} for symbol: {symbol}')
        raise ValueError(f'Margin requirement must be positive, got {margin_requirement} for {symbol}')

    # Calculate trade duration
    trade_duration = trade['exit_time'] - trade['entry_time']
    trade_duration_hours = round(trade_duration.total_seconds() / 3600, 2)

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
    """
    Print formatted trade metrics to console with color-coded profit/loss.

    Displays human-readable trade information including entry/exit details, duration,
    prices, side, and return percentages. Uses ANSI color codes to highlight profitable
    (green) vs losing (red) trades.

    Args:
        trade: Dictionary with trade metrics from calculate_trade_metrics().
              Expected keys: entry_time, exit_time, duration, duration_hours,
              entry_price, exit_price, side, return_percentage_of_margin,
              return_percentage_of_contract

    Returns:
        None. Outputs formatted text directly to stdout

    Side Effects:
        Prints formatted trade information to console with ANSI color codes
    """
    # Define colors for better visualization
    green = '\033[92m'
    red = '\033[91m'
    reset = '\033[0m'

    # Determine colors based on return percentage
    return_percentage = trade.get('return_percentage_of_contract', 0.0)
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
    print('\n--- RETURN METRICS ---')
    print(f"Return % of Contract: {color}{return_percentage}%{reset}")

    print('=============================\n')


# ==================== Private Helper Functions ====================

def _get_symbol_category(symbol):
    """
    Categorize futures symbols into asset classes for margin calculation (internal use only).

    Maps each symbol to its asset class category (energies, metals, indices, etc.)
    which determines the appropriate margin ratio to use for estimating requirements.

    Args:
        symbol: Futures symbol code (e.g., 'ZS', 'CL', 'GC', 'ES')

    Returns:
        String category name: 'energies', 'metals', 'indices', 'forex', 'crypto',
        'grains', 'softs', or 'default' if symbol not found in mapping
    """
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


def _estimate_margin(symbol, entry_price, contract_multiplier):
    """
    Estimate margin requirement for a futures contract based on historical market ratios (internal use only).

    Calculates approximate margin needed to trade one contract by applying asset-class-specific
    ratios to the contract value. Uses historical ratios from Jan 2026 market data to estimate
    margin requirements for backtesting periods where actual margin data is unavailable.

    Args:
        symbol: Futures symbol code (e.g., 'ZS', 'CL', 'GC')
        entry_price: Contract entry price in symbol's quote units
        contract_multiplier: Number of units per contract (e.g., 5000 for ZS bushels)

    Returns:
        Float representing estimated margin requirement in dollars.
        Calculated as: contract_value * category_margin_ratio
        where contract_value = entry_price * contract_multiplier
    """
    category = _get_symbol_category(symbol)
    ratio = MARGIN_RATIOS.get(category, MARGIN_RATIOS['default'])
    contract_value = entry_price * contract_multiplier
    return contract_value * ratio
