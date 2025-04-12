import os

# API Setup
BASE_URL = "https://localhost:5001/v1/api/"
ACCOUNT_ID = "DUE343675"

# Data Directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
ALERTS_DIR = os.path.join(DATA_DIR, "alerts")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")
CONTRACTS_DIR = os.path.join(DATA_DIR, "contracts")
TRADES_DIR = os.path.join(DATA_DIR, "trades")
TW_ALERTS_DIR = os.path.join(DATA_DIR, "tw_alerts")
TW_ALERTS_ANALYSIS_DIR = os.path.join(ANALYSIS_DIR, "tw_alerts")
TRADES_ANALYSIS_DIR = os.path.join(ANALYSIS_DIR, "trades")

# File paths
CONTRACTS_FILE_PATH = os.path.join(CONTRACTS_DIR, "contracts.json")
TW_ALERTS_PER_TRADE_METRICS_FILE_PATH = os.path.join(TW_ALERTS_ANALYSIS_DIR, "tw_alerts_per_trade_metrics.csv")
TW_ALERTS_DATASET_METRICS_FILE_PATH = os.path.join(TW_ALERTS_ANALYSIS_DIR, "tw_alerts_dataset_metrics.csv")
TRADES_PER_TRADE_METRICS_FILE_PATH = os.path.join(TRADES_ANALYSIS_DIR, "trades_per_trade_metrics.csv")
TRADES_DATASET_METRICS_FILE_PATH = os.path.join(TRADES_ANALYSIS_DIR, "trades_dataset_metrics.csv")

# Strategy
MIN_DAYS_UNTIL_EXPIRY = 60
AGGRESSIVE_TRADING = True
QUANTITY_TO_TRADE = 1

# Analysis
TIMEFRAME_TO_ANALYZE = 7  # In days
