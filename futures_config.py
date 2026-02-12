"""
Futures Symbols Configuration.

This module contains configuration for all tradable futures symbols,
organized by category with their respective exchanges.
"""

# ==================== Futures Specifications ====================

# Single source of truth for all futures contract specifications
# Format: symbol: {'category': str, 'exchange': str, 'multiplier': int/float, 'tick_size': float, 'margin': float, 'tv_compatible': bool}
# Order: Grains, Softs, Energy, Metals, Crypto, Index, Forex
# Within each category: Normal, Mini, Micro
SYMBOL_SPECS = {
    # Grains - Normal
    'ZC': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 50, 'tick_size': 0.25, 'margin': 1617.97,
           'tv_compatible': True},
    'ZW': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 50, 'tick_size': 0.25, 'margin': 2453.56,
           'tv_compatible': True},
    'ZS': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 50, 'tick_size': 0.25, 'margin': 3377.88,
           'tv_compatible': True},
    'ZL': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 600, 'tick_size': 0.01, 'margin': 3252.26,
           'tv_compatible': True},
    'ZM': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': None, 'tick_size': 0.01, 'margin': None,
           'tv_compatible': True},
    # Grains - Mini
    'XC': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 1000, 'tick_size': 0.125, 'margin': 323.594,
           'tv_compatible': True},
    'XW': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 1000, 'tick_size': 0.125, 'margin': 490.712,
           'tv_compatible': True},
    'XK': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 1000, 'tick_size': 0.125, 'margin': 675.576,
           'tv_compatible': True},
    'YC': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': None, 'tick_size': 0.125, 'margin': None,
           'tv_compatible': False},
    'QC': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': None, 'tick_size': 0.125, 'margin': None,
           'tv_compatible': False},
    # Grains - Micro
    'MZC': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 500, 'tick_size': 0.50, 'margin': 163.047,
            'tv_compatible': True},
    'MZW': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 500, 'tick_size': 0.50, 'margin': 245.356,
            'tv_compatible': True},
    'MZS': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 500, 'tick_size': 0.50, 'margin': 337.788,
            'tv_compatible': True},
    'MZL': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 6000, 'tick_size': 0.02, 'margin': 343.801,
            'tv_compatible': True},

    # Softs - Normal
    'SB': {'category': 'Softs', 'exchange': 'ICEUS', 'multiplier': 1120, 'tick_size': 0.01, 'margin': 1470.56,
           'tv_compatible': True},
    'KC': {'category': 'Softs', 'exchange': 'ICEUS', 'multiplier': 37500, 'tick_size': 0.05, 'margin': 23399.86,
           'tv_compatible': True},
    'CC': {'category': 'Softs', 'exchange': 'ICEUS', 'multiplier': 10, 'tick_size': 1.00, 'margin': 10638.56,
           'tv_compatible': True},

    # Energy - Normal
    'CL': {'category': 'Energy', 'exchange': 'NYMEX', 'multiplier': 1000, 'tick_size': 0.01, 'margin': 16250,
           'tv_compatible': True},
    'NG': {'category': 'Energy', 'exchange': 'NYMEX', 'multiplier': 10000, 'tick_size': 0.001, 'margin': 10735.02,
           'tv_compatible': True},
    'PL': {'category': 'Energy', 'exchange': 'NYMEX', 'multiplier': 50, 'tick_size': 0.10, 'margin': 21928.24,
           'tv_compatible': True},
    # Energy - Mini
    'PLM': {'category': 'Energy', 'exchange': 'NYMEX', 'multiplier': 10, 'tick_size': 0.10, 'margin': None,
            'tv_compatible': False},
    # Energy - Micro
    'MCL': {'category': 'Energy', 'exchange': 'NYMEX', 'multiplier': 100, 'tick_size': 0.01, 'margin': 1625,
            'tv_compatible': True},
    'MNG': {'category': 'Energy', 'exchange': 'NYMEX', 'multiplier': 1000, 'tick_size': 0.001, 'margin': 1073.50,
            'tv_compatible': True},

    # Metals - Normal
    'GC': {'category': 'Metals', 'exchange': 'COMEX', 'multiplier': 100, 'tick_size': 0.10, 'margin': 37605.98,
           'tv_compatible': True},
    'SI': {'category': 'Metals', 'exchange': 'COMEX', 'multiplier': 5000, 'tick_size': 0.005, 'margin': 74460.96,
           'tv_compatible': True},
    'HG': {'category': 'Metals', 'exchange': 'COMEX', 'multiplier': 25000, 'tick_size': 0.0005, 'margin': 18786.14,
           'tv_compatible': True},
    # Metals - Micro
    'MGC': {'category': 'Metals', 'exchange': 'COMEX', 'multiplier': 10, 'tick_size': 0.10, 'margin': 3760.60,
            'tv_compatible': False},
    'SIL': {'category': 'Metals', 'exchange': 'COMEX', 'multiplier': 1000, 'tick_size': 0.005, 'margin': 14892.19,
            'tv_compatible': False},
    'MHG': {'category': 'Metals', 'exchange': 'COMEX', 'multiplier': 2500, 'tick_size': 0.0005, 'margin': 1878.61,
            'tv_compatible': False},

    # Crypto - Normal
    'BTC': {'category': 'Crypto', 'exchange': 'CME', 'multiplier': None, 'tick_size': 5.00, 'margin': 157483.39,
            'tv_compatible': True},
    'ETH': {'category': 'Crypto', 'exchange': 'CME', 'multiplier': None, 'tick_size': 0.50, 'margin': 90450.05,
            'tv_compatible': True},
    # Crypto - Micro
    'MET': {'category': 'Crypto', 'exchange': 'CME', 'multiplier': 0.1, 'tick_size': 0.50, 'margin': 180.90,
            'tv_compatible': True},
    'MBT': {'category': 'Crypto', 'exchange': 'CME', 'multiplier': 0.1, 'tick_size': 5.00, 'margin': 3260.81,
            'tv_compatible': False},

    # Index - Normal
    'ES': {'category': 'Index', 'exchange': 'CME', 'multiplier': 50, 'tick_size': 0.25, 'margin': 23102.20,
           'tv_compatible': False},
    'NQ': {'category': 'Index', 'exchange': 'CME', 'multiplier': 20, 'tick_size': 0.25, 'margin': 35512.40,
           'tv_compatible': False},
    'YM': {'category': 'Index', 'exchange': 'CBOT', 'multiplier': 5, 'tick_size': 1.00, 'margin': 15758.45,
           'tv_compatible': True},
    'RTY': {'category': 'Index', 'exchange': 'CME', 'multiplier': 50, 'tick_size': 0.10, 'margin': 10462.70,
            'tv_compatible': False},
    'ZB': {'category': 'Index', 'exchange': 'CBOT', 'multiplier': 114, 'tick_size': 0.03125, 'margin': 4255,
           'tv_compatible': True},
    # Index - Micro
    'MES': {'category': 'Index', 'exchange': 'CME', 'multiplier': 5, 'tick_size': 0.25, 'margin': 2310,
            'tv_compatible': False},
    'MNQ': {'category': 'Index', 'exchange': 'CME', 'multiplier': 2, 'tick_size': 0.25, 'margin': 3550.80,
            'tv_compatible': False},
    'MYM': {'category': 'Index', 'exchange': 'CBOT', 'multiplier': 0.5, 'tick_size': 1.00, 'margin': 1575.50,
            'tv_compatible': True},
    'M2K': {'category': 'Index', 'exchange': 'CME', 'multiplier': 5, 'tick_size': 0.10, 'margin': 1046.50,
            'tv_compatible': False},
    'MHNG': {'category': 'Index', 'exchange': 'CME', 'multiplier': 2500, 'tick_size': None, 'margin': None,
             'tv_compatible': False},

    # Forex - Normal
    '6E': {'category': 'Forex', 'exchange': 'CME', 'multiplier': None, 'tick_size': 0.00005, 'margin': 3435.30,
           'tv_compatible': True},
    '6J': {'category': 'Forex', 'exchange': 'CME', 'multiplier': None, 'tick_size': 0.0000005, 'margin': 3789.01,
           'tv_compatible': True},
    '6B': {'category': 'Forex', 'exchange': 'CME', 'multiplier': None, 'tick_size': 0.0001, 'margin': 2520.64,
           'tv_compatible': True},
    '6A': {'category': 'Forex', 'exchange': 'CME', 'multiplier': None, 'tick_size': 0.00005, 'margin': 2344.58,
           'tv_compatible': True},
    '6C': {'category': 'Forex', 'exchange': 'CME', 'multiplier': None, 'tick_size': 0.00005, 'margin': 1555.28,
           'tv_compatible': True},
    '6S': {'category': 'Forex', 'exchange': 'CME', 'multiplier': None, 'tick_size': 0.00005, 'margin': 5762.84,
           'tv_compatible': True},
    # Forex - Micro
    'M6E': {'category': 'Forex', 'exchange': 'CME', 'multiplier': None, 'tick_size': None, 'margin': 343.53,
            'tv_compatible': False},
    'M6A': {'category': 'Forex', 'exchange': 'CME', 'multiplier': None, 'tick_size': None, 'margin': 234.458,
            'tv_compatible': False},
    'M6B': {'category': 'Forex', 'exchange': 'CME', 'multiplier': None, 'tick_size': None, 'margin': 252.064,
            'tv_compatible': False},
}

DEFAULT_TICK_SIZE = 0.01

# ==================== Auto-generated Category Lists ====================
# All category lists are auto-generated from SYMBOL_SPECS - only TradingView-compatible symbols

GRAINS = sorted([k for k, v in SYMBOL_SPECS.items() if v['category'] == 'Grains' and v['tv_compatible']])
SOFTS = sorted([k for k, v in SYMBOL_SPECS.items() if v['category'] == 'Softs' and v['tv_compatible']])
ENERGY = sorted([k for k, v in SYMBOL_SPECS.items() if v['category'] == 'Energy' and v['tv_compatible']])
METALS = sorted([k for k, v in SYMBOL_SPECS.items() if v['category'] == 'Metals' and v['tv_compatible']])
CRYPTO = sorted([k for k, v in SYMBOL_SPECS.items() if v['category'] == 'Crypto' and v['tv_compatible']])
INDEX = sorted([k for k, v in SYMBOL_SPECS.items() if v['category'] == 'Index' and v['tv_compatible']])
FOREX = sorted([k for k, v in SYMBOL_SPECS.items() if v['category'] == 'Forex' and v['tv_compatible']])

# ==================== Auto-generated Mappings ====================
# All mappings are auto-generated from SYMBOL_SPECS - single source of truth

# Category and exchange mappings
SYMBOL_CATEGORY_MAP = {symbol: specs['category'] for symbol, specs in SYMBOL_SPECS.items()}
SYMBOL_EXCHANGE_MAP = {symbol: specs['exchange'] for symbol, specs in SYMBOL_SPECS.items()}

# Build CATEGORY_EXCHANGE_MAP from SYMBOL_SPECS (for reference)
CATEGORY_EXCHANGE_MAP = {}
for symbol, specs in SYMBOL_SPECS.items():
    if specs['category'] not in CATEGORY_EXCHANGE_MAP:
        CATEGORY_EXCHANGE_MAP[specs['category']] = specs['exchange']

# ==================== Backward Compatibility ====================
# Auto-generate individual dictionaries from SYMBOL_SPECS for backward compatibility

CONTRACT_MULTIPLIERS = {k: v['multiplier'] for k, v in SYMBOL_SPECS.items() if v['multiplier'] is not None}
TICK_SIZES = {k: v['tick_size'] for k, v in SYMBOL_SPECS.items() if v['tick_size'] is not None}
MARGIN_REQUIREMENTS = {k: v['margin'] for k, v in SYMBOL_SPECS.items() if v['margin'] is not None}

# Auto-generate ALL_TRADINGVIEW_SYMBOLS from SYMBOL_SPECS
ALL_TRADINGVIEW_SYMBOLS = sorted([k for k, v in SYMBOL_SPECS.items() if v['tv_compatible']])


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
        ValueError: If symbol is not recognized
    """
    if symbol not in SYMBOL_SPECS:
        raise ValueError(f'Unknown symbol: {symbol}')
    return SYMBOL_SPECS[symbol]['category']


def validate_symbols(symbols):
    """
    Validate that all symbols are TradingView-compatible.

    Args:
        symbols: List of symbol strings to validate

    Returns:
        Tuple of (valid_symbols, invalid_symbols)
    """
    valid_symbols = []
    invalid_symbols = []

    for symbol in symbols:
        if symbol in SYMBOL_SPECS and SYMBOL_SPECS[symbol]['tv_compatible']:
            valid_symbols.append(symbol)
        else:
            invalid_symbols.append(symbol)

    return valid_symbols, invalid_symbols
