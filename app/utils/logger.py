import logging
import os

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)


# Configure a custom logger
def get_logger(name="app"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # File handler - write logs to a file
    file_handler = logging.FileHandler("logs/app.log")
    file_handler.setLevel(logging.ERROR)  # Example: log only errors to the file

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


# Example usage:
logger = get_logger()
logger.info("Logger initialized")
