"""
Tests for Logger Module.

Tests cover:
- Log directory creation
- Logger name: default and custom
- Logger level configuration
- Formatter setup
- File handler creation: debug, info, error
- Console handler creation
- All handlers attached to the logger
- Handler reuse when the logger already has handlers
"""
from unittest.mock import MagicMock, call

from app.utils.logger import get_logger


# ==================== Test Classes ====================

class TestGetLogger:
    """Test get_logger configuration and handler setup."""

    def test_creates_logs_directory(self, monkeypatch, tmp_path):
        """Test the logs directory is created if it does not exist."""
        mock_logs_dir = tmp_path / "logs"
        monkeypatch.setattr("app.utils.logger.LOGS_DIR", mock_logs_dir)
        monkeypatch.setattr("app.utils.logger.logging", MagicMock())

        get_logger()

        assert mock_logs_dir.exists()

    def test_default_name_is_app(self, mock_logging_setup):
        """Test get_logger uses 'app' as the default logger name."""
        get_logger()

        mock_logging_setup["logging"].getLogger.assert_called_once_with("app")

    def test_custom_name_used(self, mock_logging_setup):
        """Test get_logger uses the provided name when given."""
        get_logger("custom_logger")

        mock_logging_setup["logging"].getLogger.assert_called_once_with("custom_logger")

    def test_logger_level_set_to_debug(self, mock_logging_setup):
        """Test get_logger sets the logger level to DEBUG."""
        get_logger()

        mock_logging_setup["logger"].setLevel.assert_called_once_with(
            mock_logging_setup["logging"].DEBUG
        )

    def test_formatter_created_with_expected_format(self, mock_logging_setup):
        """Test get_logger creates a formatter with the standard log format string."""
        get_logger()

        mock_logging_setup["logging"].Formatter.assert_called_once_with(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def test_debug_handler_configured(self, mock_logging_setup):
        """Test debug file handler is created, levelled, filtered, and formatted."""
        get_logger()

        logs_dir = mock_logging_setup["logs_dir"]
        mock_logging_setup["logging"].FileHandler.assert_any_call(str(logs_dir / "debug.log"))

        debug_handler = mock_logging_setup["debug_handler"]
        debug_handler.setLevel.assert_called_once_with(mock_logging_setup["logging"].DEBUG)
        debug_handler.addFilter.assert_called_once()
        debug_handler.setFormatter.assert_called_once_with(mock_logging_setup["formatter"])

    def test_info_handler_configured(self, mock_logging_setup):
        """Test info file handler is created, levelled, filtered, and formatted."""
        get_logger()

        logs_dir = mock_logging_setup["logs_dir"]
        mock_logging_setup["logging"].FileHandler.assert_any_call(str(logs_dir / "info.log"))

        info_handler = mock_logging_setup["info_handler"]
        info_handler.setLevel.assert_called_once_with(mock_logging_setup["logging"].INFO)
        info_handler.addFilter.assert_called_once()
        info_handler.setFormatter.assert_called_once_with(mock_logging_setup["formatter"])

    def test_error_handler_configured(self, mock_logging_setup):
        """Test error file handler is created, levelled, and formatted."""
        get_logger()

        logs_dir = mock_logging_setup["logs_dir"]
        mock_logging_setup["logging"].FileHandler.assert_any_call(str(logs_dir / "error.log"))

        error_handler = mock_logging_setup["error_handler"]
        error_handler.setLevel.assert_called_once_with(mock_logging_setup["logging"].ERROR)
        error_handler.setFormatter.assert_called_once_with(mock_logging_setup["formatter"])

    def test_console_handler_configured(self, mock_logging_setup):
        """Test console handler is created, levelled at WARNING, and formatted."""
        get_logger()

        mock_logging_setup["logging"].StreamHandler.assert_called_once()

        console_handler = mock_logging_setup["console_handler"]
        console_handler.setLevel.assert_called_once_with(mock_logging_setup["logging"].WARNING)
        console_handler.setFormatter.assert_called_once_with(mock_logging_setup["formatter"])

    def test_all_four_handlers_added(self, mock_logging_setup):
        """Test all four handlers are added to the logger."""
        get_logger()

        mock_logger = mock_logging_setup["logger"]
        assert mock_logger.addHandler.call_count == 4
        mock_logger.addHandler.assert_has_calls([
            call(mock_logging_setup["debug_handler"]),
            call(mock_logging_setup["info_handler"]),
            call(mock_logging_setup["error_handler"]),
            call(mock_logging_setup["console_handler"]),
        ])

    def test_existing_handlers_prevent_re_setup(self, mock_logging_setup):
        """Test get_logger skips handler setup when handlers already exist."""
        mock_logging_setup["logger"].handlers = [MagicMock()]

        get_logger()

        mock_logging_setup["logger"].addHandler.assert_not_called()
