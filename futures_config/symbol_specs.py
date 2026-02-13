"""
Futures Symbol Specifications.

Single source of truth for all futures contract specifications.
TradingView symbols are used as the primary identifiers.

Format: symbol: {
    'category': str,
    'exchange': str,
    'multiplier': int/float | None,
    'tick_size': float | None,
    'margin': float | None,
    'tv_compatible': bool
}

Note: None indicates that a contract specification value is intentionally unspecified or not yet configured.

Organization:
- Order: Grains, Softs, Energy, Metals, Crypto, Index, Forex
- Within each category: Normal, Mini, Micro
"""

SYMBOL_SPECS = {
    # Grains - Normal
    'ZC': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 50, 'tick_size': 0.25, 'margin': 1617.97,
           'tv_compatible': True},  # Corn
    'ZW': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 50, 'tick_size': 0.25, 'margin': 2453.56,
           'tv_compatible': True},  # Wheat
    'ZS': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 50, 'tick_size': 0.25, 'margin': 3377.88,
           'tv_compatible': True},  # Soybeans
    'ZL': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 600, 'tick_size': 0.01, 'margin': 3252.26,
           'tv_compatible': True},  # Soybean Oil
    'ZM': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': None, 'tick_size': 0.01, 'margin': None,
           'tv_compatible': True},  # Soybean Meal
    # Grains - Mini
    'XC': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 1000, 'tick_size': 0.125, 'margin': 323.594,
           'tv_compatible': True},  # Mini Corn (TradingView symbol, maps to YC in IBKR)
    'XW': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 1000, 'tick_size': 0.125, 'margin': 490.712,
           'tv_compatible': True},  # Mini Wheat (TradingView symbol, maps to YW in IBKR)
    'XK': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 1000, 'tick_size': 0.125, 'margin': 675.576,
           'tv_compatible': True},  # Mini Soybeans (TradingView symbol, maps to YK in IBKR)
    # Grains - Micro
    'MZC': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 500, 'tick_size': 0.50, 'margin': 163.047,
            'tv_compatible': True},  # Micro Corn
    'MZW': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 500, 'tick_size': 0.50, 'margin': 245.356,
            'tv_compatible': True},  # Micro Wheat
    'MZS': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 500, 'tick_size': 0.50, 'margin': 337.788,
            'tv_compatible': True},  # Micro Soybeans
    'MZL': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 6000, 'tick_size': 0.02, 'margin': 343.801,
            'tv_compatible': True},  # Micro Soybean Oil

    # Softs - Normal
    'SB': {'category': 'Softs', 'exchange': 'ICEUS', 'multiplier': 1120, 'tick_size': 0.01, 'margin': 1470.56,
           'tv_compatible': True},  # Sugar
    'KC': {'category': 'Softs', 'exchange': 'ICEUS', 'multiplier': 37500, 'tick_size': 0.05, 'margin': 23399.86,
           'tv_compatible': True},  # Coffee
    'CC': {'category': 'Softs', 'exchange': 'ICEUS', 'multiplier': 10, 'tick_size': 1.00, 'margin': 10638.56,
           'tv_compatible': True},  # Cocoa

    # Energy - Normal
    'CL': {'category': 'Energy', 'exchange': 'NYMEX', 'multiplier': 1000, 'tick_size': 0.01, 'margin': 16250,
           'tv_compatible': True},  # Crude Oil
    'NG': {'category': 'Energy', 'exchange': 'NYMEX', 'multiplier': 10000, 'tick_size': 0.001, 'margin': 10735.02,
           'tv_compatible': True},  # Natural Gas
    # Energy - Micro
    'MCL': {'category': 'Energy', 'exchange': 'NYMEX', 'multiplier': 100, 'tick_size': 0.01, 'margin': 1625,
            'tv_compatible': True},  # Micro Crude Oil
    'MNG': {'category': 'Energy', 'exchange': 'NYMEX', 'multiplier': 1000, 'tick_size': 0.001, 'margin': 1073.50,
            'tv_compatible': True},  # Micro Natural Gas

    # Metals - Normal
    'GC': {'category': 'Metals', 'exchange': 'COMEX', 'multiplier': 100, 'tick_size': 0.10, 'margin': 37605.98,
           'tv_compatible': True},  # Gold
    'SI': {'category': 'Metals', 'exchange': 'COMEX', 'multiplier': 5000, 'tick_size': 0.005, 'margin': 74460.96,
           'tv_compatible': True},  # Silver
    'HG': {'category': 'Metals', 'exchange': 'COMEX', 'multiplier': 25000, 'tick_size': 0.0005, 'margin': 18786.14,
           'tv_compatible': True},  # Copper
    'PL': {'category': 'Metals', 'exchange': 'NYMEX', 'multiplier': 50, 'tick_size': 0.10, 'margin': 21928.24,
           'tv_compatible': True},  # Platinum
    # Metals - Mini
    'PLM': {'category': 'Metals', 'exchange': 'NYMEX', 'multiplier': 10, 'tick_size': 0.10, 'margin': None,
            'tv_compatible': False},  # Mini Platinum
    # Metals - Micro
    'MGC': {'category': 'Metals', 'exchange': 'COMEX', 'multiplier': 10, 'tick_size': 0.10, 'margin': 3760.60,
            'tv_compatible': True},  # Micro Gold
    'SIL': {'category': 'Metals', 'exchange': 'COMEX', 'multiplier': 1000, 'tick_size': 0.005, 'margin': 14892.19,
            'tv_compatible': True},  # Micro Silver (TradingView symbol, maps to QI in IBKR)
    'MHG': {'category': 'Metals', 'exchange': 'COMEX', 'multiplier': 2500, 'tick_size': 0.0005, 'margin': 1878.61,
            'tv_compatible': False},  # Micro Copper

    # Crypto - Normal
    'BTC': {'category': 'Crypto', 'exchange': 'CME', 'multiplier': None, 'tick_size': 5.00, 'margin': 157483.39,
            'tv_compatible': True},  # Bitcoin
    'ETH': {'category': 'Crypto', 'exchange': 'CME', 'multiplier': None, 'tick_size': 0.50, 'margin': 90450.05,
            'tv_compatible': True},  # Ethereum
    # Crypto - Micro
    'MET': {'category': 'Crypto', 'exchange': 'CME', 'multiplier': 0.1, 'tick_size': 0.50, 'margin': 180.90,
            'tv_compatible': True},  # Micro Ethereum
    'MBT': {'category': 'Crypto', 'exchange': 'CME', 'multiplier': 0.1, 'tick_size': 5.00, 'margin': 3260.81,
            'tv_compatible': False},  # Micro Bitcoin

    # Index - Normal
    'ES': {'category': 'Index', 'exchange': 'CME', 'multiplier': 50, 'tick_size': 0.25, 'margin': 23102.20,
           'tv_compatible': False},  # E-mini S&P 500
    'NQ': {'category': 'Index', 'exchange': 'CME', 'multiplier': 20, 'tick_size': 0.25, 'margin': 35512.40,
           'tv_compatible': False},  # E-mini NASDAQ-100
    'YM': {'category': 'Index', 'exchange': 'CBOT', 'multiplier': 5, 'tick_size': 1.00, 'margin': 15758.45,
           'tv_compatible': True},  # E-mini Dow
    'RTY': {'category': 'Index', 'exchange': 'CME', 'multiplier': 50, 'tick_size': 0.10, 'margin': 10462.70,
            'tv_compatible': False},  # E-mini Russell 2000
    'ZB': {'category': 'Index', 'exchange': 'CBOT', 'multiplier': 114, 'tick_size': 0.03125, 'margin': 4255,
           'tv_compatible': True},  # 30-Year T-Bond
    # Index - Micro
    'MES': {'category': 'Index', 'exchange': 'CME', 'multiplier': 5, 'tick_size': 0.25, 'margin': 2310,
            'tv_compatible': False},  # Micro E-mini S&P 500
    'MNQ': {'category': 'Index', 'exchange': 'CME', 'multiplier': 2, 'tick_size': 0.25, 'margin': 3550.80,
            'tv_compatible': False},  # Micro E-mini NASDAQ-100
    'MYM': {'category': 'Index', 'exchange': 'CBOT', 'multiplier': 0.5, 'tick_size': 1.00, 'margin': 1575.50,
            'tv_compatible': True},  # Micro E-mini Dow
    'M2K': {'category': 'Index', 'exchange': 'CME', 'multiplier': 5, 'tick_size': 0.10, 'margin': 1046.50,
            'tv_compatible': False},  # Micro E-mini Russell 2000
    'MHNG': {'category': 'Index', 'exchange': 'CME', 'multiplier': 2500, 'tick_size': None, 'margin': None,
             'tv_compatible': False},  # Micro Henry Hub Natural Gas

    # Forex - Normal
    '6E': {'category': 'Forex', 'exchange': 'CME', 'multiplier': None, 'tick_size': 0.00005, 'margin': 3435.30,
           'tv_compatible': True},  # Euro FX
    '6J': {'category': 'Forex', 'exchange': 'CME', 'multiplier': None, 'tick_size': 0.0000005, 'margin': 3789.01,
           'tv_compatible': True},  # Japanese Yen
    '6B': {'category': 'Forex', 'exchange': 'CME', 'multiplier': None, 'tick_size': 0.0001, 'margin': 2520.64,
           'tv_compatible': True},  # British Pound
    '6A': {'category': 'Forex', 'exchange': 'CME', 'multiplier': None, 'tick_size': 0.00005, 'margin': 2344.58,
           'tv_compatible': True},  # Australian Dollar
    '6C': {'category': 'Forex', 'exchange': 'CME', 'multiplier': None, 'tick_size': 0.00005, 'margin': 1555.28,
           'tv_compatible': True},  # Canadian Dollar
    '6S': {'category': 'Forex', 'exchange': 'CME', 'multiplier': None, 'tick_size': 0.00005, 'margin': 5762.84,
           'tv_compatible': True},  # Swiss Franc
    # Forex - Micro
    'M6E': {'category': 'Forex', 'exchange': 'CME', 'multiplier': None, 'tick_size': None, 'margin': 343.53,
            'tv_compatible': False},  # Micro Euro FX
    'M6A': {'category': 'Forex', 'exchange': 'CME', 'multiplier': None, 'tick_size': None, 'margin': 234.458,
            'tv_compatible': False},  # Micro Australian Dollar
    'M6B': {'category': 'Forex', 'exchange': 'CME', 'multiplier': None, 'tick_size': None, 'margin': 252.064,
            'tv_compatible': False},  # Micro British Pound
}

# Default tick size used as fallback when symbol is unknown or tick_size is None
DEFAULT_TICK_SIZE = 0.01
