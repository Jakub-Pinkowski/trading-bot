import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ==================== Environment Configuration ====================

DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
PORT = int(os.getenv('PORT', 5000))
BASE_URL = os.getenv('BASE_URL', 'http://localhost:5000')

# ==================== Directory Configuration ====================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# ==================== IBKR Configuration ====================

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

# ==================== Strategy Configuration ====================

MIN_DAYS_UNTIL_EXPIRY = 60
QUANTITY_TO_TRADE = 1
AGGRESSIVE_TRADING = True

# ==================== Analysis Configuration ====================

TIMEFRAME_TO_ANALYZE = 7
