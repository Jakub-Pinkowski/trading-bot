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
MIN_DAYS_UNTIL_EXPIRY = 60
QUANTITY_TO_TRADE = 1
AGGRESSIVE_TRADING = True

# Analysis
TIMEFRAME_TO_ANALYZE = 7

# ==================== Futures Configuration ====================
# Import futures-specific data from futures_config.py

from futures_config import (
    SYMBOL_SPECS,
    DEFAULT_TICK_SIZE,
    get_tick_size,
    get_contract_multiplier,
    get_margin_requirement,
    is_tradingview_compatible,
)

__all__ = [
    'SYMBOL_SPECS',
    'DEFAULT_TICK_SIZE',
    'get_tick_size',
    'get_contract_multiplier',
    'get_margin_requirement',
    'is_tradingview_compatible',
]
