"""
Shared fixtures for utility module tests.

Provides test data and mock helpers for API, file, and logging utilities.
"""
import logging
from unittest.mock import MagicMock

import pandas as pd
import pytest


# ==================== API Utils Fixtures ====================

@pytest.fixture
def mock_response_factory():
    """Factory that produces mock HTTP response objects."""

    def _make_mock_response(data=None, raise_exc=None):
        def json_func():
            return data if data is not None else {"data": "test_data"}

        def raise_for_status_func():
            if raise_exc:
                raise raise_exc

        response = type("MockResponse", (), {})()
        response.json = json_func
        response.raise_for_status = raise_for_status_func
        return response

    return _make_mock_response


# ==================== File Utils Fixtures ====================

@pytest.fixture
def sample_json_data():
    """Sample nested JSON data for file I/O tests."""
    return {
        "item1": {"name": "Item 1", "value": 100},
        "item2": {"name": "Item 2", "value": 200},
    }


@pytest.fixture
def sample_dataframe():
    """Sample two-column DataFrame for CSV/Parquet tests."""
    return pd.DataFrame({
        "name": ["Item 1", "Item 2"],
        "value": [100, 200],
    })


# ==================== Logger Fixtures ====================

@pytest.fixture
def mock_logging_setup(monkeypatch, tmp_path):
    """Mock all logging components for get_logger tests."""
    mock_logs_dir = tmp_path / "logs"

    mock_logging = MagicMock()
    mock_logger = MagicMock(spec=logging.Logger)
    mock_logger.handlers = []
    mock_logging.getLogger.return_value = mock_logger

    mock_debug_handler = MagicMock(spec=logging.FileHandler)
    mock_info_handler = MagicMock(spec=logging.FileHandler)
    mock_error_handler = MagicMock(spec=logging.FileHandler)
    mock_console_handler = MagicMock(spec=logging.StreamHandler)

    mock_logging.FileHandler.side_effect = [mock_debug_handler, mock_info_handler, mock_error_handler]
    mock_logging.StreamHandler.return_value = mock_console_handler

    mock_formatter = MagicMock(spec=logging.Formatter)
    mock_logging.Formatter.return_value = mock_formatter

    # Replace sys in the logger module with a mock that excludes 'pytest' from modules check
    mock_sys = MagicMock()
    mock_sys.modules = {}

    monkeypatch.setattr("app.utils.logger.logging", mock_logging)
    monkeypatch.setattr("app.utils.logger.LOGS_DIR", mock_logs_dir)
    monkeypatch.setattr("app.utils.logger.sys", mock_sys)
    monkeypatch.delenv("PYTEST_RUNNING", raising=False)

    return {
        "logging": mock_logging,
        "logs_dir": mock_logs_dir,
        "logger": mock_logger,
        "debug_handler": mock_debug_handler,
        "info_handler": mock_info_handler,
        "error_handler": mock_error_handler,
        "console_handler": mock_console_handler,
        "formatter": mock_formatter,
    }
