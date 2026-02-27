"""
Shared fixtures for route handler tests.

Provides a Flask test client and monkeypatched mocks for all
webhook module dependencies.
"""
from unittest.mock import MagicMock

import pytest
from flask import Flask

from app.routes.webhook import webhook_blueprint


# ==================== Flask Fixtures ====================

@pytest.fixture
def app():
    """Flask application instance with webhook blueprint registered."""
    app = Flask(__name__)
    app.register_blueprint(webhook_blueprint)
    return app


@pytest.fixture
def client(app):
    """Flask test client for making HTTP requests."""
    return app.test_client()


# ==================== Webhook Module Fixtures ====================

@pytest.fixture
def mock_process_trading_data(monkeypatch):
    """Mock process_trading_data dependency in webhook module."""
    mock = MagicMock()
    monkeypatch.setattr("app.routes.webhook.process_trading_data", mock)
    return mock


@pytest.fixture
def mock_process_rollover_data(monkeypatch):
    """Mock process_rollover_data dependency in webhook module."""
    mock = MagicMock()
    monkeypatch.setattr("app.routes.webhook.process_rollover_data", mock)
    return mock


@pytest.fixture
def mock_load_file_webhook(monkeypatch):
    """Mock load_file dependency in webhook module."""
    mock = MagicMock()
    monkeypatch.setattr("app.routes.webhook.load_file", mock)
    return mock


@pytest.fixture
def mock_save_file_webhook(monkeypatch):
    """Mock save_file dependency in webhook module."""
    mock = MagicMock()
    monkeypatch.setattr("app.routes.webhook.save_file", mock)
    return mock


@pytest.fixture
def mock_datetime_webhook(monkeypatch):
    """Mock datetime class in webhook module for deterministic timestamps."""
    mock = MagicMock()
    monkeypatch.setattr("app.routes.webhook.datetime", mock)
    return mock


@pytest.fixture
def mock_save_alert_to_file(monkeypatch):
    """Mock save_alert_data_to_file in webhook module to isolate route tests from file I/O."""
    mock = MagicMock()
    monkeypatch.setattr("app.routes.webhook.save_alert_data_to_file", mock)
    return mock
