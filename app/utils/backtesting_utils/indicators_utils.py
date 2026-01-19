import hashlib

from app.utils.logger import get_logger

# Initialize logger
logger = get_logger('backtesting/indicators')


# Helper function to create a hashable key for pandas Series
def hash_series(series):
    return hashlib.md5(series.values.tobytes()).hexdigest()
