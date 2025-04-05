import os

# Setup
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# TODO: Update paths
# TODO: Save alerts, trades etc. per day. So each day gets its own file
ALERTS_FILE_PATH = os.path.join(BASE_DIR, "data", "alerts.json")
CONTRACTS_FILE_PATH = os.path.join(BASE_DIR, "data", "contracts.json")
TRADES_FILE_PATH = os.path.join(BASE_DIR, "data", "trades.csv")
BASE_URL = "https://localhost:5001/v1/api/"
ACCOUNT_ID = "DUE343675"


# Strategy
MIN_DAYS_UNTIL_EXPIRY = 60
AGGRESSIVE_TRADING = True
QUANTITY_TO_TRADE = 1
