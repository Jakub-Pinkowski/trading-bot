"""
Helper Functions for Futures Symbol Configuration.

Provides utility functions for querying symbol specifications,
retrieving contract details, and validating symbols.
"""

from futures_config.symbol_specs import SYMBOL_SPECS, DEFAULT_TICK_SIZE


def get_exchange_for_symbol(symbol):
    """
    Get the exchange for a given symbol.

    Args:
        symbol: Futures symbol (e.g., 'ZS', 'CL', 'GC')

    Returns:
        Exchange name (e.g., 'CBOT', 'NYMEX', 'COMEX')

    Raises:
        ValueError: If the symbol is not recognized

    Example:
        get_exchange_for_symbol('ZS')
        'CBOT'
        get_exchange_for_symbol('CL')
        'NYMEX'
    """
    if symbol not in SYMBOL_SPECS:
        raise ValueError(f'Unknown symbol: {symbol}')
    return SYMBOL_SPECS[symbol]['exchange']


def get_category_for_symbol(symbol):
    """
    Get the category for a given symbol.

    Args:
        symbol: Futures symbol (e.g., 'ZS', 'CL', 'GC')

    Returns:
        Category name (e.g., 'Grains', 'Energy', 'Metals')

    Raises:
        ValueError: If the symbol is not recognized

    Example:
        get_category_for_symbol('ZS')
        'Grains'
         get_category_for_symbol('GC')
        'Metals'
    """
    if symbol not in SYMBOL_SPECS:
        raise ValueError(f'Unknown symbol: {symbol}')
    return SYMBOL_SPECS[symbol]['category']


def get_tick_size(symbol):
    """
    Get the tick size for a given symbol.

    Returns the configured tick size for known symbols. If the symbol is
    unknown or its tick_size is not configured (None), this function
    falls back to DEFAULT_TICK_SIZE as a defensive default, so callers
    that rely on a numeric tick size can continue to operate. Callers that
    need to distinguish unknown symbols should validate with
    get_exchange_for_symbol() or check SYMBOL_SPECS directly before
    calling this helper.

    Args:
        symbol: Futures symbol (e.g., 'ZS', 'CL', 'GC')

    Returns:
        Tick size as float. Returns DEFAULT_TICK_SIZE when the symbol is
        unknown or when no tick_size is configured in SYMBOL_SPECS

    Example:
        get_tick_size('ZS')
        0.25
        get_tick_size('UNKNOWN') # Falls back to default
        0.01
    """
    if symbol in SYMBOL_SPECS and SYMBOL_SPECS[symbol]['tick_size'] is not None:
        return SYMBOL_SPECS[symbol]['tick_size']
    # Fallback to DEFAULT_TICK_SIZE for unknown symbols or unset tick sizes
    return DEFAULT_TICK_SIZE


def get_contract_multiplier(symbol):
    """
    Get the contract multiplier for a given symbol.

    Args:
        symbol: Futures symbol (e.g., 'ZS', 'CL', 'GC')

    Returns:
        Contract multiplier (int/float) or None if not available

    Raises:
        ValueError: If the symbol is not recognized

    Example:
        get_contract_multiplier('ZS')
        50
        get_contract_multiplier('CL')
        1000
    """
    if symbol not in SYMBOL_SPECS:
        raise ValueError(f'Unknown symbol: {symbol}')
    return SYMBOL_SPECS[symbol]['multiplier']


def get_margin_requirement(symbol):
    """
    Get the margin requirement for a given symbol.

    Args:
        symbol: Futures symbol (e.g., 'ZS', 'CL', 'GC')

    Returns:
        Margin requirement (float) or None if not available

    Raises:
        ValueError: If symbol is not recognized

    Example:
        get_margin_requirement('ZS')
        3377.88
        get_margin_requirement('CL')
        16250.0
    """
    if symbol not in SYMBOL_SPECS:
        raise ValueError(f'Unknown symbol: {symbol}')
    return SYMBOL_SPECS[symbol]['margin']


def is_tradingview_compatible(symbol):
    """
    Check if a symbol is TradingView-compatible.

    Args:
        symbol: Futures symbol (e.g., 'ZS', 'CL', 'GC')

    Returns:
        Boolean indicating TradingView compatibility

    Raises:
        ValueError: If the symbol is not recognized

    Example:
        is_tradingview_compatible('ZS')
        True
        is_tradingview_compatible('ES')
        False
    """
    if symbol not in SYMBOL_SPECS:
        raise ValueError(f'Unknown symbol: {symbol}')
    return SYMBOL_SPECS[symbol]['tv_compatible']
