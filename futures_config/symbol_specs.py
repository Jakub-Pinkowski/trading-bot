# noinspection ALL
# @formatter:off
"""
Futures Symbol Specifications.

Single source of truth for all futures contract specifications.
TradingView symbols are used as the primary identifiers.

Format: symbol: {
    'category': str,
    'exchange': str,
    'multiplier': int/float,
    'tick_size': float,
    'margin': float,
}

Organization:
- Order: Grains, Softs, Energy, Metals, Crypto, Index, Forex
- Within each category: Normal, Mini, Micro
"""

SYMBOL_SPECS = {
    # ==================== Grains ====================

    # --- Normal ---
    'ZC': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 50, 'tick_size': 0.25, 'margin': 1617.97},    # Corn
    'ZW': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 50, 'tick_size': 0.25, 'margin': 2453.56},    # Wheat
    'ZS': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 50, 'tick_size': 0.25, 'margin': 3377.88},    # Soybeans
    'ZL': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 600, 'tick_size': 0.01, 'margin': 3252.26},   # Soybean Oil

    # --- Mini ---
    'XC': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 1000, 'tick_size': 0.125, 'margin': 323.594},  # Mini Corn
    'XW': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 1000, 'tick_size': 0.125, 'margin': 490.712},  # Mini Wheat
    'XK': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 1000, 'tick_size': 0.125, 'margin': 675.576},  # Mini Soybeans

    # --- Micro ---
    'MZC': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 500, 'tick_size': 0.50, 'margin': 163.047},   # Micro Corn
    'MZW': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 500, 'tick_size': 0.50, 'margin': 245.356},   # Micro Wheat
    'MZS': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 500, 'tick_size': 0.50, 'margin': 337.788},   # Micro Soybeans
    'MZL': {'category': 'Grains', 'exchange': 'CBOT', 'multiplier': 6000, 'tick_size': 0.02, 'margin': 343.801},  # Micro Soybean Oil

    # ==================== Softs ====================

    # --- Normal ---
    'SB': {'category': 'Softs', 'exchange': 'ICEUS', 'multiplier': 1120, 'tick_size': 0.01, 'margin': 1470.56},     # Sugar
    'KC': {'category': 'Softs', 'exchange': 'ICEUS', 'multiplier': 37500, 'tick_size': 0.05, 'margin': 23399.86},   # Coffee
    'CC': {'category': 'Softs', 'exchange': 'ICEUS', 'multiplier': 10, 'tick_size': 1.00, 'margin': 10638.56},      # Cocoa

    # ==================== Energy ====================

    # --- Normal ---
    'CL': {'category': 'Energy', 'exchange': 'NYMEX', 'multiplier': 1000, 'tick_size': 0.01, 'margin': 16250},      # Crude Oil
    'NG': {'category': 'Energy', 'exchange': 'NYMEX', 'multiplier': 10000, 'tick_size': 0.001, 'margin': 10735.02}, # Natural Gas

    # --- Micro ---
    'MCL': {'category': 'Energy', 'exchange': 'NYMEX', 'multiplier': 100, 'tick_size': 0.01, 'margin': 1625},       # Micro Crude Oil
    'MNG': {'category': 'Energy', 'exchange': 'NYMEX', 'multiplier': 1000, 'tick_size': 0.001, 'margin': 1073.50},  # Micro Natural Gas

    # ==================== Metals ====================

    # --- Normal ---
    'GC': {'category': 'Metals', 'exchange': 'COMEX', 'multiplier': 100, 'tick_size': 0.10, 'margin': 37605.98},     # Gold
    'SI': {'category': 'Metals', 'exchange': 'COMEX', 'multiplier': 5000, 'tick_size': 0.005, 'margin': 74460.96},   # Silver
    'HG': {'category': 'Metals', 'exchange': 'COMEX', 'multiplier': 25000, 'tick_size': 0.0005, 'margin': 18786.14}, # Copper
    'PL': {'category': 'Metals', 'exchange': 'NYMEX', 'multiplier': 50, 'tick_size': 0.10, 'margin': 21928.24},      # Platinum

    # --- Micro ---
    'MGC': {'category': 'Metals', 'exchange': 'COMEX_MINI', 'multiplier': 10, 'tick_size': 0.10, 'margin': 3760.60},    # Micro Gold
    'SIL': {'category': 'Metals', 'exchange': 'COMEX_MINI', 'multiplier': 1000, 'tick_size': 0.005, 'margin': 14892.19},# Micro Silver
    'MHG': {'category': 'Metals', 'exchange': 'COMEX_MINI', 'multiplier': 2500, 'tick_size': 0.0005, 'margin': 1878.61},# Micro Copper

    # ==================== Crypto ====================

    # --- Normal ---
    'BTC': {'category': 'Crypto', 'exchange': 'CME', 'multiplier': 5, 'tick_size': 5.00, 'margin': 157483.39},   # Bitcoin
    'ETH': {'category': 'Crypto', 'exchange': 'CME', 'multiplier': 50, 'tick_size': 0.50, 'margin': 90450.05},    # Ethereum

    # --- Micro ---
    'MET': {'category': 'Crypto', 'exchange': 'CME', 'multiplier': 0.1, 'tick_size': 0.50, 'margin': 180.90},       # Micro Ethereum
    'MBT': {'category': 'Crypto', 'exchange': 'CME', 'multiplier': 0.1, 'tick_size': 5.00, 'margin': 3260.81},      # Micro Bitcoin

    # ==================== Index ====================

    # --- Normal ---
    'ES': {'category': 'Index', 'exchange': 'CME_MINI', 'multiplier': 50, 'tick_size': 0.25, 'margin': 23102.20},   # E-mini S&P 500
    'NQ': {'category': 'Index', 'exchange': 'CME_MINI', 'multiplier': 20, 'tick_size': 0.25, 'margin': 35512.40},   # E-mini NASDAQ-100
    'YM': {'category': 'Index', 'exchange': 'CBOT', 'multiplier': 5, 'tick_size': 1.00, 'margin': 15758.45},        # E-mini Dow
    'RTY': {'category': 'Index', 'exchange': 'CME_MINI', 'multiplier': 50, 'tick_size': 0.10, 'margin': 10462.70},  # E-mini Russell 2000
    'ZB': {'category': 'Index', 'exchange': 'CBOT', 'multiplier': 114, 'tick_size': 0.03125, 'margin': 4255},       # 30-Year T-Bond

    # --- Micro ---
    'MES': {'category': 'Index', 'exchange': 'CME_MINI', 'multiplier': 5, 'tick_size': 0.25, 'margin': 2310},       # Micro E-mini S&P 500
    'MNQ': {'category': 'Index', 'exchange': 'CME_MINI', 'multiplier': 2, 'tick_size': 0.25, 'margin': 3550.80},    # Micro E-mini NASDAQ-100
    'MYM': {'category': 'Index', 'exchange': 'CBOT', 'multiplier': 0.5, 'tick_size': 1.00, 'margin': 1575.50},      # Micro E-mini Dow
    'M2K': {'category': 'Index', 'exchange': 'CME_MINI', 'multiplier': 5, 'tick_size': 0.10, 'margin': 1046.50},    # Micro E-mini Russell 2000

    # ==================== Forex ====================

    # --- Normal ---
    '6E': {'category': 'Forex', 'exchange': 'CME', 'multiplier': 125000, 'tick_size': 0.00005, 'margin': 3435.30},    # Euro FX
    '6J': {'category': 'Forex', 'exchange': 'CME', 'multiplier': 12500000, 'tick_size': 0.0000005, 'margin': 3789.01},  # Japanese Yen
    '6B': {'category': 'Forex', 'exchange': 'CME', 'multiplier': 62500, 'tick_size': 0.0001, 'margin': 2520.64},     # British Pound
    '6A': {'category': 'Forex', 'exchange': 'CME', 'multiplier': 100000, 'tick_size': 0.00005, 'margin': 2344.58},    # Australian Dollar
    '6C': {'category': 'Forex', 'exchange': 'CME', 'multiplier': 100000, 'tick_size': 0.00005, 'margin': 1555.28},    # Canadian Dollar
    '6S': {'category': 'Forex', 'exchange': 'CME', 'multiplier': 125000, 'tick_size': 0.00005, 'margin': 5762.84},    # Swiss Franc

    # --- Micro ---
    'M6E': {'category': 'Forex', 'exchange': 'CME_MINI', 'multiplier': 12500, 'tick_size': 0.000005, 'margin': 343.53},  # Micro Euro FX
    'M6A': {'category': 'Forex', 'exchange': 'CME_MINI', 'multiplier': 10000, 'tick_size': 0.000005, 'margin': 234.458}, # Micro Australian Dollar
    'M6B': {'category': 'Forex', 'exchange': 'CME_MINI', 'multiplier': 6250, 'tick_size': 0.00001, 'margin': 252.064}, # Micro British Pound
}
