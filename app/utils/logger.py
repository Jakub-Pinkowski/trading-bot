import logging
import os

from config import LOGS_DIR

# Create logs directory if it doesn't exist
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)


# Custom logger
def get_logger(name="app"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set the logger to the lowest level to handle all logs

    # Formatter to ensure uniform log format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File handler for DEBUG only (logs DEBUG)
    debug_handler = logging.FileHandler(os.path.join(LOGS_DIR, "debug.log"))
    debug_handler.setLevel(logging.DEBUG)  # Handles DEBUG only
    debug_handler.addFilter(lambda record: record.levelno == logging.DEBUG)
    debug_handler.setFormatter(formatter)

    # File handler for INFO only (logs INFO and WARNING)
    info_handler = logging.FileHandler(os.path.join(LOGS_DIR, "info.log"))
    info_handler.setLevel(logging.INFO)  # Handles INFO and WARNING
    info_handler.addFilter(lambda record: record.levelno < logging.ERROR)  # Exclude ERROR and CRITICAL
    info_handler.setFormatter(formatter)

    # File handler for ERROR and CRITICAL only
    error_handler = logging.FileHandler(os.path.join(LOGS_DIR, "error.log"))
    error_handler.setLevel(logging.ERROR)  # Handles ERROR and CRITICAL
    error_handler.setFormatter(formatter)

    # Add handlers to logger (prevent duplicate log entries)
    if not logger.handlers:
        logger.addHandler(info_handler)
        logger.addHandler(error_handler)
        logger.addHandler(debug_handler)

    return logger
