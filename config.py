import os

from dotenv import load_dotenv

load_dotenv()

# Environment Variables
DEBUG = os.getenv('DEBUG')
PORT = os.getenv('PORT')
BASE_URL = os.getenv('BASE_URL')

# IBKR setup
ACCOUNT_ID = 'DUE343675'
ALLOWED_IPS = {
    '52.89.214.238',
    '34.212.75.30',
    '54.218.53.128',
    '52.32.178.7',
    '127.0.0.1',
    'localhost',
    '64.225.97.130',
    '95.91.215.169',
    '95.91.215.232'
}

# Data directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Alerts
ALERTS_DIR = os.path.join(DATA_DIR, "alerts")
IBKR_ALERTS_DIR = os.path.join(ALERTS_DIR, "ibkr_alerts")
TRADES_DIR = os.path.join(ALERTS_DIR, "trades")
TW_ALERTS_DIR = os.path.join(ALERTS_DIR, "tw_alerts")

# Analysis
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")
ANALYSIS_IBKR_ALERTS_DIR = os.path.join(ANALYSIS_DIR, "analysis_ibkr_alerts")
ANALYSIS_TRADES_DIR = os.path.join(ANALYSIS_DIR, "analysis_trades")
ANALYSIS_TW_ALERTS_DIR = os.path.join(ANALYSIS_DIR, "analysis_tw_alerts")
IBKR_ALERTS_PER_TRADE_METRICS_FILE_PATH = os.path.join(ANALYSIS_IBKR_ALERTS_DIR, "ibkr_alerts_per_trade_metrics.csv")
IBKR_ALERTS_DATASET_METRICS_FILE_PATH = os.path.join(ANALYSIS_IBKR_ALERTS_DIR, "ibkr_alerts_dataset_metrics.csv")
TRADES_PER_TRADE_METRICS_FILE_PATH = os.path.join(ANALYSIS_TRADES_DIR, "trades_per_trade_metrics.csv")
TRADES_DATASET_METRICS_FILE_PATH = os.path.join(ANALYSIS_TRADES_DIR, "trades_dataset_metrics.csv")
TW_ALERTS_PER_TRADE_METRICS_FILE_PATH = os.path.join(ANALYSIS_TW_ALERTS_DIR, "tw_alerts_per_trade_metrics.csv")
TW_ALERTS_DATASET_METRICS_FILE_PATH = os.path.join(ANALYSIS_TW_ALERTS_DIR, "tw_alerts_dataset_metrics.csv")

# Backtesting
BACKTESTING_DIR = os.path.join(DATA_DIR, "backtesting")
CACHE_DIR = os.path.join(BACKTESTING_DIR, "cache")
INDICATOR_CACHE_LOCK_FILE = os.path.join(CACHE_DIR, "indicator_cache.lock")
DATAFRAME_CACHE_LOCK_FILE = os.path.join(CACHE_DIR, "dataframe_cache.lock")

# Contracts
CONTRACTS_DIR = os.path.join(DATA_DIR, "contracts")
CONTRACTS_FILE_PATH = os.path.join(CONTRACTS_DIR, "contracts.json")

# Historical data
HISTORICAL_DATA_DIR = os.path.join(DATA_DIR, "historical_data")
SWITCH_DATES_FILE_PATH = os.path.join(HISTORICAL_DATA_DIR, "contract_switch_dates.yaml")

# Strategy
# TODO [LOW]: Aggressive trading is obsolete, needs to be removed
MIN_DAYS_UNTIL_EXPIRY = 60
QUANTITY_TO_TRADE = 1
AGGRESSIVE_TRADING = True

# Analysis
TIMEFRAME_TO_ANALYZE = 7  # In days

# NOTE: TradingView multipliers, might not work properly with actual trades
CONTRACT_MULTIPLIERS = {
    'CC': 10,
    'CL': 1000,
    'ES': 50,
    'GC': 100,
    'HG': 25000,
    'KC': 37500,
    'MBT': 0.1,
    'M2K': 5,
    'MCL': 100,
    'MES': 5,
    'MET': 0.1,
    'MGC': 10,
    'MHG': 2500,
    'MHNG': 2500,
    'MNQ': 2,
    'MNG': 1000,
    'MZC': 500,
    'MZL': 6000,
    'MZS': 500,
    'MZW': 500,
    'MYM': 0.5,
    'NG': 10000,
    'NQ': 20,
    'PL': 50,
    'PLM': 10,
    'RTY': 50,
    'SB': 1120,
    'SI': 5000,
    'SIL': 1000,
    'XC': 1000,
    'XK': 1000,
    'XW': 1000,
    'YM': 5,
    'ZB': 114,
    'ZC': 50,
    'ZL': 600,
    'ZS': 50,
    'ZW': 50,
}

# ==================== Tick Sizes ====================

# Minimum tick sizes for futures contracts (in price points)
# Organized by asset class for easy maintenance

# Grains
TICK_SIZES_GRAINS = {
    'ZC': 0.25,  # Corn
    'MZC': 0.50,  # Micro Corn
    'XC': 0.125,  # Mini Corn
    'YC': 0.125,  # Mini Corn (e-Corn)
    'QC': 0.125,  # Mini-sized Corn
    'ZW': 0.25,  # Wheat
    'MZW': 0.50,  # Micro Wheat
    'XW': 0.125,  # Mini Wheat
    'ZS': 0.25,  # Soybeans
    'MZS': 0.50,  # Micro Soybeans
    'XK': 0.125,  # Mini Soybeans
    'ZL': 0.01,  # Soybean Oil
    'MZL': 0.02,  # Micro Soybean Oil
}

# Softs
TICK_SIZES_SOFTS = {
    'SB': 0.01,  # Sugar
    'KC': 0.05,  # Coffee
    'CC': 1.00,  # Cocoa
}

# Energy
TICK_SIZES_ENERGY = {
    'CL': 0.01,  # Crude Oil
    'MCL': 0.01,  # Micro Crude Oil
    'NG': 0.001,  # Natural Gas
    'MNG': 0.001,  # Micro Natural Gas
}

# Metals
TICK_SIZES_METALS = {
    'GC': 0.10,  # Gold
    'MGC': 0.10,  # Micro Gold
    'SI': 0.005,  # Silver
    'SIL': 0.005,  # Micro Silver
    'HG': 0.0005,  # Copper
    'MHG': 0.0005,  # Micro Copper
    'PL': 0.10,  # Platinum
    'PLM': 0.10,  # Micro Platinum
}

# Crypto
TICK_SIZES_CRYPTO = {
    'BTC': 5.00,  # Bitcoin
    'MBT': 5.00,  # Micro Bitcoin
    'ETH': 0.50,  # Ethereum
    'MET': 0.50,  # Micro Ethereum
}

# Indices
TICK_SIZES_INDICES = {
    'ES': 0.25,  # E-mini S&P 500
    'MES': 0.25,  # Micro E-mini S&P 500
    'NQ': 0.25,  # E-mini NASDAQ-100
    'MNQ': 0.25,  # Micro E-mini NASDAQ-100
    'YM': 1.00,  # E-mini Dow
    'MYM': 1.00,  # Micro E-mini Dow
    'RTY': 0.10,  # E-mini Russell 2000
    'M2K': 0.10,  # Micro E-mini Russell 2000
    'ZB': 0.03125,  # 30-Year T-Bond (1/32)
}

# Forex
TICK_SIZES_FOREX = {
    '6E': 0.00005,  # Euro FX
    '6J': 0.0000005,  # Japanese Yen
    '6B': 0.0001,  # British Pound
    '6A': 0.00005,  # Australian Dollar
    '6C': 0.00005,  # Canadian Dollar
    '6S': 0.00005,  # Swiss Franc
}

# Combined tick sizes dictionary
TICK_SIZES = {
    **TICK_SIZES_GRAINS,
    **TICK_SIZES_SOFTS,
    **TICK_SIZES_ENERGY,
    **TICK_SIZES_METALS,
    **TICK_SIZES_CRYPTO,
    **TICK_SIZES_INDICES,
    **TICK_SIZES_FOREX,
}

# Default tick size for unknown symbols
DEFAULT_TICK_SIZE = 0.01

# NOTE: Only used once to calculate the average margin as a percent of the contract value
MARGIN_REQUIREMENTS = {
    '6A': 2344.58,
    '6B': 2520.64,
    '6C': 1555.28,
    '6E': 3435.30,
    '6J': 3789.01,
    '6S': 5762.84,
    'BTC': 157483.39,
    'CC': 10638.56,
    'CL': 16250,
    'ES': 23102.20,
    'ETH': 90450.05,
    'GC': 37605.98,
    'HG': 18786.14,
    'KC': 23399.86,
    'MBT': 3260.81,
    'M2K': 1046.50,
    'M6A': 234.458,
    'M6B': 252.064,
    'M6E': 343.53,
    'MCL': 1625,
    'MGC': 3760.60,
    'MET': 180.90,
    'MES': 2310,
    'MNQ': 3550.80,
    'MHG': 1878.61,
    'MNG': 1073.50,
    'MZC': 163.047,
    'MZL': 343.801,
    'MZS': 337.788,
    'MZW': 245.356,
    'MYM': 1575.50,
    'NG': 10735.02,
    'NQ': 35512.40,
    'PL': 21928.24,
    'RTY': 10462.70,
    'SB': 1470.56,
    'SI': 74460.96,
    'SIL': 14892.19,
    'XC': 323.594,
    'XK': 675.576,
    'XW': 490.712,
    'YM': 15758.45,
    'ZB': 4255,
    'ZC': 1617.97,
    'ZL': 3252.26,
    'ZS': 3377.88,
    'ZW': 2453.56,
}
