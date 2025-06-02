import logging
import os
import sys

from config import LOGS_DIR


# Custom logger
def get_logger(name='app'):
    # Create the logs directory if it doesn't exist
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set the logger to the lowest level to handle all logs

    # Formatter to ensure uniform log format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Check if we're running in a test environment
    is_test_environment = 'pytest' in sys.modules

    # Add handlers to the logger in the correct order
    if not logger.handlers:
        if not is_test_environment:
            # File handler for DEBUG only (logs DEBUG)
            debug_handler = logging.FileHandler(os.path.join(LOGS_DIR, "debug.log"))
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.addFilter(lambda record: record.levelno == logging.DEBUG)
            debug_handler.setFormatter(formatter)
            logger.addHandler(debug_handler)

            # File handler for INFO only (logs INFO and WARNING)
            info_handler = logging.FileHandler(os.path.join(LOGS_DIR, "info.log"))
            info_handler.setLevel(logging.INFO)
            info_handler.addFilter(lambda record: record.levelno < logging.ERROR)
            info_handler.setFormatter(formatter)
            logger.addHandler(info_handler)

            # File handler for ERROR and CRITICAL only
            error_handler = logging.FileHandler(os.path.join(LOGS_DIR, "error.log"))
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            logger.addHandler(error_handler)

        # Console handler for ERROR and CRITICAL (always added, even in tests)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
