import os

from dotenv import load_dotenv

load_dotenv()

# Environment Variables
DEBUG = os.getenv('DEBUG')
PORT = os.getenv('PORT')
BASE_URL = os.getenv('BASE_URL')

# IBKR setup
ACCOUNT_ID = "DUE343675"
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
MIN_DAYS_UNTIL_EXPIRY = 60
QUANTITY_TO_TRADE = 1
AGGRESSIVE_TRADING = True

# Analysis
TIMEFRAME_TO_ANALYZE = 7  # In days

# NOTE: TW multipliers, might not with properly with actual trades
CONTRACT_MULTIPLIERS = {
    'CL': 1000,
    'ES': 50,
    'GC': 100,
    'HG': 25000,
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
    'YM': 5,
    'ZB': 114,
    'ZC': 50,
    'ZL': 600,
    'ZS': 50,
    'ZW': 50,
}

# TODO [MEDIUM]: Add margin requirements for each contract from official IBKR website
MARGIN_REQUIREMENTS = {
    '6A': 2589.98,
    '6B': 2540.51,
    '6C': 2008.65,
    '6E': 4675.60,
    '6J': 4370.00,
    '6S': 6210,
    'BTC': 188972.00,
    'CC': 17791.36,
    'CL': 16250,
    'ES': 16889.88,
    'ETH': 79531.03,
    'GC': 25338.86,
    'HG': 11366.41,
    'KC': 19968.09,
    'MBT': 3779.44,
    'M2K': 747.202,
    'M6A': 258.998,
    'M6B': 254.051,
    'M6E': 467.56,
    'MCL': 1625,
    'MGC': 2513.58,
    'MET': 159.061,
    'MES': 1688.99,
    'MHG': 1136.64,
    'MNQ': 2445.81,
    'MNG': 805.827,
    'MZC': 176.701,
    'MZL': 368.232,
    'MZS': 356.959,
    'MZW': 291.296,
    'MYM': 1101.00,
    'NG': 8058.27,
    'NQ': 24458.12,
    'PL': 6109.41,
    'RTY': 7472.02,
    'SB': 1860.49,
    'SI': 24022.78,
    'SIL': 4804.56,
    'XC': 369.738,
    'XK': 713.918,
    'XW': 580.092,
    'YM:': 11009.96,
    'ZB': 5707.23,
    'ZC': 1713.66,
    'ZL': 3587.25,
    'ZS': 3504.81,
    'ZW': 2968.06,
}
