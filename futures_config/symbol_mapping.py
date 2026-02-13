"""
Symbol Mapping Between TradingView and IBKR.

TradingView symbols are the source of truth throughout the codebase.
These mappings are only used when communicating with IBKR API for
order placement or when receiving position data from IBKR.

Symbol Differences:
- Mini Grains: TradingView uses X prefix (XC, XW, XK), IBKR uses Y prefix (YC, YW, YK)
- Micro Silver: TradingView uses SIL, IBKR uses QI
"""

# TradingView to IBKR symbol mapping
# Only use when placing orders through IBKR API
TV_TO_IBKR_MAPPING = {
    'XC': 'YC',  # Mini Corn
    'XK': 'YK',  # Mini Soybeans
    'XW': 'YW',  # Mini Wheat
    'SIL': 'QI',  # Micro Silver
}

# IBKR to TradingView symbol mapping (reverse)
# Used when receiving position or trade data from IBKR
IBKR_TO_TV_MAPPING = {
    'YC': 'XC',  # Mini Corn
    'YK': 'XK',  # Mini Soybeans
    'YW': 'XW',  # Mini Wheat
    'QI': 'SIL',  # Micro Silver
}


def map_tv_to_ibkr(symbol):
    """
    Map TradingView symbol to IBKR symbol for order placement.

    TradingView symbols are the source of truth. This function should only
    be called when communicating with IBKR API for order placement.

    Args:
        symbol: TradingView symbol (e.g., 'XC', 'XK', 'XW', 'SIL')

    Returns:
        IBKR symbol (e.g., 'YC', 'YK', 'YW', 'QI') or original symbol if no mapping exists

    Example:
        >>> map_tv_to_ibkr('XC')
        'YC'
        >>> map_tv_to_ibkr('ZS')  # No mapping needed
        'ZS'
    """
    return TV_TO_IBKR_MAPPING.get(symbol, symbol)


def map_ibkr_to_tv(symbol):
    """
    Map IBKR symbol to TradingView symbol for position tracking.

    Used when receiving position or trade data from IBKR that needs to be
    matched with TradingView alerts or data.

    Args:
        symbol: IBKR symbol (e.g., 'YC', 'YK', 'YW', 'QI')

    Returns:
        TradingView symbol (e.g., 'XC', 'XK', 'XW', 'SIL') or original symbol if no mapping exists

    Example:
        >>> map_ibkr_to_tv('YC')
        'XC'
        >>> map_ibkr_to_tv('ZS')  # No mapping needed
        'ZS'
    """
    return IBKR_TO_TV_MAPPING.get(symbol, symbol)
