import logging
import os
from unittest.mock import patch, MagicMock, call

import pytest

from app.utils.logger import get_logger


@pytest.fixture
def mock_logging_setup():
    """Fixture to mock all logging components"""
    # Create a mock sys.modules that doesn't include 'pytest'
    mock_sys_modules = {}

    with patch("app.utils.logger.logging") as mock_logging, \
            patch("app.utils.logger.os.path.exists", return_value=True), \
            patch("app.utils.logger.os.makedirs") as mock_makedirs, \
            patch("app.utils.logger.LOGS_DIR", "/mock/logs"), \
            patch("app.utils.logger.sys.modules", mock_sys_modules):  # Mock sys.modules to not include 'pytest'
        # Create mock logger and handlers
        mock_logger = MagicMock(spec=logging.Logger)
        mock_logger.handlers = []
        mock_logging.getLogger.return_value = mock_logger

        # Create mock handlers
        mock_debug_handler = MagicMock(spec=logging.FileHandler)
        mock_info_handler = MagicMock(spec=logging.FileHandler)
        mock_error_handler = MagicMock(spec=logging.FileHandler)
        mock_console_handler = MagicMock(spec=logging.StreamHandler)

        # Setup FileHandler and StreamHandler mocks
        mock_logging.FileHandler.side_effect = [
            mock_debug_handler, mock_info_handler, mock_error_handler
        ]
        mock_logging.StreamHandler.return_value = mock_console_handler

        # Setup formatter mock
        mock_formatter = MagicMock(spec=logging.Formatter)
        mock_logging.Formatter.return_value = mock_formatter

        yield {
            'logging': mock_logging,
            'makedirs': mock_makedirs,
            'logger': mock_logger,
            'debug_handler': mock_debug_handler,
            'info_handler': mock_info_handler,
            'error_handler': mock_error_handler,
            'console_handler': mock_console_handler,
            'formatter': mock_formatter
        }


def test_get_logger_creates_logs_dir():
    """Test that get_logger creates the logs directory if it doesn't exist"""

    with patch("app.utils.logger.os.path.exists", return_value=False), \
            patch("app.utils.logger.os.makedirs") as mock_makedirs, \
            patch("app.utils.logger.logging"):
        get_logger()
        mock_makedirs.assert_called_once()


def test_get_logger_default_name(mock_logging_setup):
    """Test that get_logger uses the default name 'app' when no name is provided"""

    logger = get_logger()

    mock_logging_setup['logging'].getLogger.assert_called_once_with("app")


def test_get_logger_custom_name(mock_logging_setup):
    """Test that get_logger uses a custom name if provided"""

    logger = get_logger("custom_logger")

    mock_logging_setup['logging'].getLogger.assert_called_once_with("custom_logger")


def test_get_logger_sets_level(mock_logging_setup):
    """Test that get_logger sets the logger level to DEBUG"""

    logger = get_logger()

    mock_logging_setup['logger'].setLevel.assert_called_once_with(mock_logging_setup['logging'].DEBUG)


def test_get_logger_creates_formatter(mock_logging_setup):
    """Test that get_logger creates a formatter with the expected format"""

    logger = get_logger()

    mock_logging_setup['logging'].Formatter.assert_called_once_with(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def test_get_logger_creates_debug_handler(mock_logging_setup):
    """Test that get_logger creates a debug handler with the expected configuration"""

    logger = get_logger()

    # Check that the debug handler was created with the correct file path
    mock_logging_setup['logging'].FileHandler.assert_any_call(os.path.join("/mock/logs", "debug.log"))

    # Check that the debug handler was configured correctly
    debug_handler = mock_logging_setup['debug_handler']
    debug_handler.setLevel.assert_called_once_with(mock_logging_setup['logging'].DEBUG)
    debug_handler.addFilter.assert_called_once()
    debug_handler.setFormatter.assert_called_once_with(mock_logging_setup['formatter'])


def test_get_logger_creates_info_handler(mock_logging_setup):
    """Test that get_logger creates an info handler with the expected configuration"""

    logger = get_logger()

    # Check that the info handler was created with the correct file path
    mock_logging_setup['logging'].FileHandler.assert_any_call(os.path.join("/mock/logs", "info.log"))

    # Check that the info handler was configured correctly
    info_handler = mock_logging_setup['info_handler']
    info_handler.setLevel.assert_called_once_with(mock_logging_setup['logging'].INFO)
    info_handler.addFilter.assert_called_once()
    info_handler.setFormatter.assert_called_once_with(mock_logging_setup['formatter'])


def test_get_logger_creates_error_handler(mock_logging_setup):
    """Test that get_logger creates an error handler with the expected configuration"""

    logger = get_logger()

    # Check that the error handler was created with the correct file path
    mock_logging_setup['logging'].FileHandler.assert_any_call(os.path.join("/mock/logs", "error.log"))

    # Check that the error handler was configured correctly
    error_handler = mock_logging_setup['error_handler']
    error_handler.setLevel.assert_called_once_with(mock_logging_setup['logging'].ERROR)
    error_handler.setFormatter.assert_called_once_with(mock_logging_setup['formatter'])


def test_get_logger_creates_console_handler(mock_logging_setup):
    """Test that get_logger creates a console handler with the expected configuration"""

    logger = get_logger()

    # Check that the console handler was created
    mock_logging_setup['logging'].StreamHandler.assert_called_once()

    # Check that the console handler was configured correctly
    console_handler = mock_logging_setup['console_handler']
    console_handler.setLevel.assert_called_once_with(mock_logging_setup['logging'].ERROR)
    console_handler.setFormatter.assert_called_once_with(mock_logging_setup['formatter'])


def test_get_logger_adds_handlers(mock_logging_setup):
    """Test that get_logger adds all handlers to the logger"""
    logger = get_logger()

    # Check that all handlers were added to the logger
    mock_logger = mock_logging_setup['logger']
    assert mock_logger.addHandler.call_count == 4
    mock_logger.addHandler.assert_has_calls([
        call(mock_logging_setup['debug_handler']),
        call(mock_logging_setup['info_handler']),
        call(mock_logging_setup['error_handler']),
        call(mock_logging_setup['console_handler'])
    ])


def test_get_logger_reuses_existing_logger(mock_logging_setup):
    """Test that get_logger doesn't add handlers if they already exist"""
    # Set up a logger with existing handlers
    mock_logging_setup['logger'].handlers = [MagicMock()]

    logger = get_logger()

    # Check that no handlers were added
    mock_logger = mock_logging_setup['logger']
    mock_logger.addHandler.assert_not_called()
