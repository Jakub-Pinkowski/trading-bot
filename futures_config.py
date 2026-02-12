"""
Futures Symbols Configuration.

This module contains configuration for all tradable futures symbols,
organized by category with their respective exchanges.
"""

# ==================== Futures Specifications ====================

# Combined contract specifications for all symbols
# Format: symbol: {'multiplier': int/float, 'tick_size': float, 'margin': float, 'tv_compatible': bool}
# Order: Grains, Softs, Energy, Metals, Crypto, Index, Forex
# Within each category: Normal, Mini, Micro
SYMBOL_SPECS = {
    # Grains - Normal
    'ZC': {'multiplier': 50, 'tick_size': 0.25, 'margin': 1617.97, 'tv_compatible': True},
    'ZW': {'multiplier': 50, 'tick_size': 0.25, 'margin': 2453.56, 'tv_compatible': True},
    'ZS': {'multiplier': 50, 'tick_size': 0.25, 'margin': 3377.88, 'tv_compatible': True},
    'ZL': {'multiplier': 600, 'tick_size': 0.01, 'margin': 3252.26, 'tv_compatible': True},
    'ZM': {'multiplier': None, 'tick_size': 0.01, 'margin': None, 'tv_compatible': True},
    # Grains - Mini
    'XC': {'multiplier': 1000, 'tick_size': 0.125, 'margin': 323.594, 'tv_compatible': True},
    'XW': {'multiplier': 1000, 'tick_size': 0.125, 'margin': 490.712, 'tv_compatible': True},
    'XK': {'multiplier': 1000, 'tick_size': 0.125, 'margin': 675.576, 'tv_compatible': True},
    'YC': {'multiplier': None, 'tick_size': 0.125, 'margin': None, 'tv_compatible': False},
    'QC': {'multiplier': None, 'tick_size': 0.125, 'margin': None, 'tv_compatible': False},
    # Grains - Micro
    'MZC': {'multiplier': 500, 'tick_size': 0.50, 'margin': 163.047, 'tv_compatible': True},
    'MZW': {'multiplier': 500, 'tick_size': 0.50, 'margin': 245.356, 'tv_compatible': True},
    'MZS': {'multiplier': 500, 'tick_size': 0.50, 'margin': 337.788, 'tv_compatible': True},
    'MZL': {'multiplier': 6000, 'tick_size': 0.02, 'margin': 343.801, 'tv_compatible': True},

    # Softs - Normal
    'SB': {'multiplier': 1120, 'tick_size': 0.01, 'margin': 1470.56, 'tv_compatible': True},
    'KC': {'multiplier': 37500, 'tick_size': 0.05, 'margin': 23399.86, 'tv_compatible': True},
    'CC': {'multiplier': 10, 'tick_size': 1.00, 'margin': 10638.56, 'tv_compatible': True},

    # Energy - Normal
    'CL': {'multiplier': 1000, 'tick_size': 0.01, 'margin': 16250, 'tv_compatible': True},
    'NG': {'multiplier': 10000, 'tick_size': 0.001, 'margin': 10735.02, 'tv_compatible': True},
    'PL': {'multiplier': 50, 'tick_size': 0.10, 'margin': 21928.24, 'tv_compatible': True},
    # Energy - Mini
    'PLM': {'multiplier': 10, 'tick_size': 0.10, 'margin': None, 'tv_compatible': False},
    # Energy - Micro
    'MCL': {'multiplier': 100, 'tick_size': 0.01, 'margin': 1625, 'tv_compatible': True},
    'MNG': {'multiplier': 1000, 'tick_size': 0.001, 'margin': 1073.50, 'tv_compatible': True},

    # Metals - Normal
    'GC': {'multiplier': 100, 'tick_size': 0.10, 'margin': 37605.98, 'tv_compatible': True},
    'SI': {'multiplier': 5000, 'tick_size': 0.005, 'margin': 74460.96, 'tv_compatible': True},
    'HG': {'multiplier': 25000, 'tick_size': 0.0005, 'margin': 18786.14, 'tv_compatible': True},
    # Metals - Micro
    'MGC': {'multiplier': 10, 'tick_size': 0.10, 'margin': 3760.60, 'tv_compatible': False},
    'SIL': {'multiplier': 1000, 'tick_size': 0.005, 'margin': 14892.19, 'tv_compatible': False},
    'MHG': {'multiplier': 2500, 'tick_size': 0.0005, 'margin': 1878.61, 'tv_compatible': False},

    # Crypto - Normal
    'BTC': {'multiplier': None, 'tick_size': 5.00, 'margin': 157483.39, 'tv_compatible': True},
    'ETH': {'multiplier': None, 'tick_size': 0.50, 'margin': 90450.05, 'tv_compatible': True},
    # Crypto - Micro
    'MET': {'multiplier': 0.1, 'tick_size': 0.50, 'margin': 180.90, 'tv_compatible': True},
    'MBT': {'multiplier': 0.1, 'tick_size': 5.00, 'margin': 3260.81, 'tv_compatible': False},

    # Index - Normal
    'ES': {'multiplier': 50, 'tick_size': 0.25, 'margin': 23102.20, 'tv_compatible': False},
    'NQ': {'multiplier': 20, 'tick_size': 0.25, 'margin': 35512.40, 'tv_compatible': False},
    'YM': {'multiplier': 5, 'tick_size': 1.00, 'margin': 15758.45, 'tv_compatible': True},
    'RTY': {'multiplier': 50, 'tick_size': 0.10, 'margin': 10462.70, 'tv_compatible': False},
    'ZB': {'multiplier': 114, 'tick_size': 0.03125, 'margin': 4255, 'tv_compatible': True},
    # Index - Micro
    'MES': {'multiplier': 5, 'tick_size': 0.25, 'margin': 2310, 'tv_compatible': False},
    'MNQ': {'multiplier': 2, 'tick_size': 0.25, 'margin': 3550.80, 'tv_compatible': False},
    'MYM': {'multiplier': 0.5, 'tick_size': 1.00, 'margin': 1575.50, 'tv_compatible': True},
    'M2K': {'multiplier': 5, 'tick_size': 0.10, 'margin': 1046.50, 'tv_compatible': False},
    'MHNG': {'multiplier': 2500, 'tick_size': None, 'margin': None, 'tv_compatible': False},

    # Forex - Normal
    '6E': {'multiplier': None, 'tick_size': 0.00005, 'margin': 3435.30, 'tv_compatible': True},
    '6J': {'multiplier': None, 'tick_size': 0.0000005, 'margin': 3789.01, 'tv_compatible': True},
    '6B': {'multiplier': None, 'tick_size': 0.0001, 'margin': 2520.64, 'tv_compatible': True},
    '6A': {'multiplier': None, 'tick_size': 0.00005, 'margin': 2344.58, 'tv_compatible': True},
    '6C': {'multiplier': None, 'tick_size': 0.00005, 'margin': 1555.28, 'tv_compatible': True},
    '6S': {'multiplier': None, 'tick_size': 0.00005, 'margin': 5762.84, 'tv_compatible': True},
    # Forex - Micro
    'M6E': {'multiplier': None, 'tick_size': None, 'margin': 343.53, 'tv_compatible': False},
    'M6A': {'multiplier': None, 'tick_size': None, 'margin': 234.458, 'tv_compatible': False},
    'M6B': {'multiplier': None, 'tick_size': None, 'margin': 252.064, 'tv_compatible': False},
}

DEFAULT_TICK_SIZE = 0.01

# ==================== Symbols by Category ====================
# Auto-generated from SYMBOL_SPECS - lists only TradingView-compatible symbols

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
    'PL',
]

METALS = [
    'GC',
    'SI',
    'HG',
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
# Maps each category to its primary exchange

CATEGORY_EXCHANGE_MAP = {
    'Grains': 'CBOT',
    'Softs': 'ICEUS',
    'Energy': 'NYMEX',
    'Metals': 'COMEX',
    'Crypto': 'CME',
    'Index': 'CBOT',
    'Forex': 'CME',
}

# ==================== Auto-generated Mappings ====================
# SYMBOL_CATEGORY_MAP and SYMBOL_EXCHANGE_MAP are built from category lists

# Build SYMBOL_CATEGORY_MAP from category lists
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

# Build SYMBOL_EXCHANGE_MAP from categories
SYMBOL_EXCHANGE_MAP = {}
for symbol, category in SYMBOL_CATEGORY_MAP.items():
    SYMBOL_EXCHANGE_MAP[symbol] = CATEGORY_EXCHANGE_MAP[category]


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


# ==================== Backward Compatibility ====================
# Auto-generate individual dictionaries from SYMBOL_SPECS for backward compatibility

CONTRACT_MULTIPLIERS = {k: v['multiplier'] for k, v in SYMBOL_SPECS.items() if v['multiplier'] is not None}
TICK_SIZES = {k: v['tick_size'] for k, v in SYMBOL_SPECS.items() if v['tick_size'] is not None}
MARGIN_REQUIREMENTS = {k: v['margin'] for k, v in SYMBOL_SPECS.items() if v['margin'] is not None}

# Auto-generate ALL_TRADINGVIEW_SYMBOLS from SYMBOL_SPECS
ALL_TRADINGVIEW_SYMBOLS = sorted([k for k, v in SYMBOL_SPECS.items() if v['tv_compatible']])
