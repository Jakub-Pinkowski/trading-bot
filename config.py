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
    'NG': 10000,
    'GC': 100,
    'SI': 5000,
    'HG': 25000,
    'PL': 50,
    'ZC': 50,
    'ZS': 50,
    'ZL': 600,
    'ZW': 50,
    'SB': 1120,
    'NQ': 20,
    'ES': 50,
    'YM': 5,
    'RTY': 50,
    'ZB': 114,
    'MCL': 100,
    'MNG': 1000,
    'MGC': 10,
    'SIL': 1000,
    'MHG': 2500,
    'MHNG': 2500,
    'PLM': 10,
    'MZC': 500,
    'MZS': 500,
    'MZL': 6000,
    'MZW': 500,
    'MBT': 0.1,
    'MET': 0.1,
    'MNQ': 2,
    'MES': 5,
    'MYM': 0.5,
    'M2K': 5,
}

# TODO: Add margin requirements for each contract from official IBKR website
MARGIN_REQUIREMENTS = {
    'ZW': 2968.06,
    'ZC': 1713.66,
    'ZS': 3504.81,
    'ZL': 3587.25,
}
