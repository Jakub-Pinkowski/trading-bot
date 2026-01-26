import hashlib

import numpy as np

from app.utils.logger import get_logger

# Initialize logger
logger = get_logger('backtesting/indicators')


# Helper function to create a hashable key for pandas Series
def hash_series(series):
    try:
        # Try direct tobytes() for numeric types
        return hashlib.md5(series.values.tobytes()).hexdigest()
    except AttributeError:
        # Handle Arrow-backed arrays (pandas 3.0+) by converting to numpy first
        return hashlib.md5(np.array(series, dtype=object).tobytes()).hexdigest()
