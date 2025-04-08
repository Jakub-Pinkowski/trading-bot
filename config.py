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

# File paths
CONTRACTS_FILE_PATH = os.path.join(CONTRACTS_DIR, "contracts.json")
PER_TRADE_METRICS_FILE_PATH = os.path.join(ANALYSIS_DIR, "per_trade_metrics.csv")
DATASET_METRICS_FILE_PATH = os.path.join(ANALYSIS_DIR, "dataset_metrics.csv")

# Strategy
MIN_DAYS_UNTIL_EXPIRY = 60
AGGRESSIVE_TRADING = True
QUANTITY_TO_TRADE = 1

# Analysis
TIMEFRAME_TO_ANALYZE = 7  # In days
