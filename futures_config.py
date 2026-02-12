"""
Futures Symbols Configuration.

This module contains configuration for all tradable futures symbols,
organized by category with their respective exchanges.
"""

# ==================== All Available Symbols ====================

ALL_TRADINGVIEW_SYMBOLS = [
    'ZC', 'XC', 'MZC', 'ZW', 'XW', 'MZW', 'ZS', 'XK', 'MZS', 'ZL', 'MZL', 'ZM',
    'SB', 'KC', 'CC',
    'CL', 'MCL', 'NG', 'MNG',
    'GC', 'SI', 'HG', 'PL',
    'BTC', 'ETH', 'MET',
    'YM', 'MYM', 'ZB',
    '6E', '6J', '6B', '6A', '6C', '6S',
]

# ==================== Symbols by Category ====================

GRAINS = [
    'ZC', 'XC', 'MZC',
    'ZW', 'XW', 'MZW',
    'ZS', 'XK', 'MZS',
    'ZL', 'MZL',
    'ZM',
]

SOFTS = [
    'SB',
    'KC',
    'CC',
]

ENERGY = [
    'CL', 'MCL',
    'NG', 'MNG',
]

METALS = [
    'GC',
    'SI',
    'HG',
    'PL',
]

CRYPTO = [
    'BTC',
    'ETH',
    'MET',
]

INDEX = [
    'YM', 'MYM',
    'ZB',
]

FOREX = [
    '6E',
    '6J',
    '6B',
    '6A',
    '6C',
    '6S',
]

# ==================== Exchange Mapping ====================

SYMBOL_EXCHANGE_MAP = {
    # Grains - CBOT
    'ZC': 'CBOT',
    'XC': 'CBOT',
    'MZC': 'CBOT',
    'ZW': 'CBOT',
    'XW': 'CBOT',
    'MZW': 'CBOT',
    'ZS': 'CBOT',
    'XK': 'CBOT',
    'MZS': 'CBOT',
    'ZL': 'CBOT',
    'MZL': 'CBOT',
    'ZM': 'CBOT',

    # Softs - ICEUS
    'SB': 'ICEUS',
    'KC': 'ICEUS',
    'CC': 'ICEUS',

    # Energy - NYMEX
    'CL': 'NYMEX',
    'MCL': 'NYMEX',
    'NG': 'NYMEX',
    'MNG': 'NYMEX',
    'PL': 'NYMEX',

    # Metals - COMEX
    'GC': 'COMEX',
    'SI': 'COMEX',
    'HG': 'COMEX',

    # Crypto - CME
    'BTC': 'CME',
    'ETH': 'CME',
    'MET': 'CME',

    # Index - CBOT
    'YM': 'CBOT',
    'MYM': 'CBOT',
    'ZB': 'CBOT',

    # Forex - CME
    '6E': 'CME',
    '6J': 'CME',
    '6B': 'CME',
    '6A': 'CME',
    '6C': 'CME',
    '6S': 'CME',
}

# ==================== Category Mapping ====================

SYMBOL_CATEGORY_MAP = {}

for symbol in GRAINS:
    SYMBOL_CATEGORY_MAP[symbol] = 'Grains'

for symbol in SOFTS:
    SYMBOL_CATEGORY_MAP[symbol] = 'Softs'

for symbol in ENERGY:
    SYMBOL_CATEGORY_MAP[symbol] = 'Energy'

for symbol in METALS:
    SYMBOL_CATEGORY_MAP[symbol] = 'Metals'

for symbol in CRYPTO:
    SYMBOL_CATEGORY_MAP[symbol] = 'Crypto'

for symbol in INDEX:
    SYMBOL_CATEGORY_MAP[symbol] = 'Index'

for symbol in FOREX:
    SYMBOL_CATEGORY_MAP[symbol] = 'Forex'


# ==================== Helper Functions ====================

def get_exchange_for_symbol(symbol):
    """
    Get the exchange for a given symbol.

    Args:
        symbol: Futures symbol (e.g., 'ZS', 'CL', 'GC')

    Returns:
        Exchange name (e.g., 'CBOT', 'NYMEX', 'COMEX')

    Raises:
        ValueError: If symbol is not recognized
    """
    if symbol not in SYMBOL_EXCHANGE_MAP:
        raise ValueError(f'Unknown symbol: {symbol}. Must be one of: {ALL_TRADINGVIEW_SYMBOLS}')
    return SYMBOL_EXCHANGE_MAP[symbol]


def get_category_for_symbol(symbol):
    """
    Get the category for a given symbol.

    Args:
        symbol: Futures symbol (e.g., 'ZS', 'CL', 'GC')

    Returns:
        Category name (e.g., 'Grains', 'Energy', 'Metals')

    Raises:
        ValueError: If symbol is not recognized
    """
    if symbol not in SYMBOL_CATEGORY_MAP:
        raise ValueError(f'Unknown symbol: {symbol}. Must be one of: {ALL_TRADINGVIEW_SYMBOLS}')
    return SYMBOL_CATEGORY_MAP[symbol]


def validate_symbols(symbols):
    """
    Validate that all symbols are recognized.

    Args:
        symbols: List of symbol strings to validate

    Returns:
        Tuple of (valid_symbols, invalid_symbols)
    """
    valid_symbols = []
    invalid_symbols = []

    for symbol in symbols:
        if symbol in ALL_TRADINGVIEW_SYMBOLS:
            valid_symbols.append(symbol)
        else:
            invalid_symbols.append(symbol)

    return valid_symbols, invalid_symbols
