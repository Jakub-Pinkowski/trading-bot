import os

# API Setup
BASE_URL = "https://localhost:5001/v1/api/"
ACCOUNT_ID = "DUE343675"

# Data Directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ALERTS_DIR = os.path.join(DATA_DIR, "alerts")
CONTRACTS_DIR = os.path.join(DATA_DIR, "contracts")
RAW_TRADES_DIR = os.path.join(DATA_DIR, "raw_trades")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")
ANALYSIS_ALERTS_DIR = os.path.join(ANALYSIS_DIR, "alerts_analysis")
ANALYSIS_TRADES_DIR = os.path.join(ANALYSIS_DIR, "trades_analysis")

# File paths
CONTRACTS_FILE_PATH = os.path.join(CONTRACTS_DIR, "contracts.json")



# TODO: Update paths
# TODO: Save alerts, trades etc. per day. So each day gets its own file
ALERTS_FILE_PATH = os.path.join(BASE_DIR, "data", "alerts.json")
TRADES_FILE_PATH = os.path.join(BASE_DIR, "data", "trades.csv")


# Strategy
MIN_DAYS_UNTIL_EXPIRY = 60
AGGRESSIVE_TRADING = True
QUANTITY_TO_TRADE = 1
