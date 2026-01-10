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
BACKTESTING_DATA_DIR = os.path.join(BASE_DIR, "backtesting_data")
DATA_DIR = os.path.join(BASE_DIR, "data")
HISTORICAL_DATA_DIR = os.path.join(BASE_DIR, "historical_data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

CACHE_DIR = os.path.join(BACKTESTING_DATA_DIR, "cache")
INDICATOR_CACHE_LOCK_FILE = os.path.join(CACHE_DIR, "indicator_cache.lock")
DATAFRAME_CACHE_LOCK_FILE = os.path.join(CACHE_DIR, "dataframe_cache.lock")

ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")
CONTRACTS_DIR = os.path.join(DATA_DIR, "contracts")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_data")

TRADES_ANALYSIS_DIR = os.path.join(ANALYSIS_DIR, "trades")
TW_ALERTS_ANALYSIS_DIR = os.path.join(ANALYSIS_DIR, "tw_alerts")

ALERTS_DIR = os.path.join(RAW_DATA_DIR, "alerts")
TRADES_DIR = os.path.join(RAW_DATA_DIR, "trades")
TW_ALERTS_DIR = os.path.join(RAW_DATA_DIR, "tw_alerts")

# File paths
CONTRACTS_FILE_PATH = os.path.join(CONTRACTS_DIR, "contracts.json")
TW_ALERTS_PER_TRADE_METRICS_FILE_PATH = os.path.join(TW_ALERTS_ANALYSIS_DIR, "tw_alerts_per_trade_metrics.csv")
TW_ALERTS_DATASET_METRICS_FILE_PATH = os.path.join(TW_ALERTS_ANALYSIS_DIR, "tw_alerts_dataset_metrics.csv")
TRADES_PER_TRADE_METRICS_FILE_PATH = os.path.join(TRADES_ANALYSIS_DIR, "trades_per_trade_metrics.csv")
TRADES_DATASET_METRICS_FILE_PATH = os.path.join(TRADES_ANALYSIS_DIR, "trades_dataset_metrics.csv")
SWITCH_DATES_FILE_PATH = os.path.join(HISTORICAL_DATA_DIR, "contract_switch_dates.yaml")

# Strategy
# TODO [LOW]: Aggressive trading is obsolete, needs to be removed
MIN_DAYS_UNTIL_EXPIRY = 60
QUANTITY_TO_TRADE = 1
AGGRESSIVE_TRADING = True

# Analysis
TIMEFRAME_TO_ANALYZE = 7  # In days

# NOTE: TW multipliers, might not work properly with actual trades
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

# NOTE: Short Overnight Initial as it's the highest
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
