import logging
import os

from config import LOGS_DIR

# Create logs directory if it doesn't exist
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)


# Configure a custom logger
def get_logger(name="app"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # File handler - write logs to a file
    file_handler = logging.FileHandler("logs/app.log")
    file_handler.setLevel(logging.ERROR)  # Log only errors to the file

    # Stream handler - for console output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Log everything to the console

    # Formatter - uniform log format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    if not logger.handlers:  # Avoid duplicate log entries
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
