import os

# API Setup
BASE_URL = "https://localhost:5001/v1/api/"
ACCOUNT_ID = "DUE343675"

# Data Directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")
ALERTS_DIR = os.path.join(DATA_DIR, "alerts")
CONTRACTS_DIR = os.path.join(DATA_DIR, "contracts")
TRADES_DIR = os.path.join(DATA_DIR, "trades")

# File paths
CONTRACTS_FILE_PATH = os.path.join(CONTRACTS_DIR, "contracts.json")

# Strategy
MIN_DAYS_UNTIL_EXPIRY = 60
AGGRESSIVE_TRADING = True
QUANTITY_TO_TRADE = 1

# Analysis
TIMEFRAME_TO_ANALYZE = 7  # In days
