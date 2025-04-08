import logging
import os

# Logging
logging.basicConfig(
    level=logging.INFO,  # Set log level to INFO; use DEBUG for more detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log timestamp, level, and message
    handlers=[
        logging.FileHandler("errors.log", mode="a"),  # Save logs to a file (append mode)
        logging.StreamHandler()  # Also log to the console
    ]
)

# API Setup
BASE_URL = "https://localhost:5001/v1/api/"
ACCOUNT_ID = "DUE343675"

# Data Directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
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
