import logging
import os
import sys

from config import BASE_DIR

# ==================== Module Paths ====================

LOGS_DIR = BASE_DIR / "logs"


# ==================== Logger Setup ====================

def get_logger(name='app'):
    # Create the logs directory if it doesn't exist
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set the logger to the lowest level to handle all logs

    # Formatter to ensure uniform log format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Check if we're running in a test environment.
    # os.environ check covers spawned subprocess workers (ProcessPoolExecutor)
    # where sys.modules does not contain pytest.
    is_test_environment = 'pytest' in sys.modules or os.environ.get('PYTEST_RUNNING') == '1'

    # Add handlers to the logger in the correct order
    if not logger.handlers:
        logger.propagate = False
        if not is_test_environment:
            # File handler for DEBUG only (logs DEBUG)
            debug_handler = logging.FileHandler(str(LOGS_DIR / 'debug.log'))
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.addFilter(lambda record: record.levelno == logging.DEBUG)
            debug_handler.setFormatter(formatter)
            logger.addHandler(debug_handler)

            # File handler for INFO only (logs INFO and WARNING)
            info_handler = logging.FileHandler(str(LOGS_DIR / 'info.log'))
            info_handler.setLevel(logging.INFO)
            info_handler.addFilter(lambda record: record.levelno < logging.ERROR)
            info_handler.setFormatter(formatter)
            logger.addHandler(info_handler)

            # File handler for ERROR and CRITICAL only
            error_handler = logging.FileHandler(str(LOGS_DIR / 'error.log'))
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            logger.addHandler(error_handler)

            # Console handler for WARNING, ERROR and CRITICAL (not added during tests)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

    return logger
