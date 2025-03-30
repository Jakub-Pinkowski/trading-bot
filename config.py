import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_URL = "https://localhost:5001/v1/api/"
ACCOUNT_ID = "DUE343675"
MIN_DAYS_UNTIL_EXPIRY = 60
CONTRACTS_FILE_PATH = os.path.join(BASE_DIR, "data", "contracts.json")
ALERTS_FILE_PATH = os.path.join(BASE_DIR, "data", "alerts.json")