from unittest.mock import MagicMock

import pytest
from flask import Flask

from app.routes.webhook import webhook_blueprint


# IBKR Orders fixtures
@pytest.fixture(autouse=False)
def mock_get_contract_position(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("app.services.ibkr.orders.get_contract_position", mock)
    return mock


@pytest.fixture
def mock_api_post_orders(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("app.services.ibkr.orders.api_post", mock)
    return mock


@pytest.fixture
def mock_logger_orders(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("app.services.ibkr.orders.logger", mock)
    return mock


@pytest.fixture
def mock_suppress_messages(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("app.services.ibkr.orders.suppress_messages", mock)
    return mock


@pytest.fixture(autouse=True)
def reset_aggressive_trading(monkeypatch):
    # Reset AGGRESSIVE_TRADING to False unless individually overridden
    monkeypatch.setattr("app.services.ibkr.orders.AGGRESSIVE_TRADING", False)


# IBKR Connection fixtures
@pytest.fixture
def mock_logger_connection(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr("app.services.ibkr.connection.logger", logger)
    return logger


@pytest.fixture
def mock_api_post_connection(monkeypatch):
    api_post = MagicMock()
    monkeypatch.setattr("app.services.ibkr.connection.api_post", api_post)
    return api_post


@pytest.fixture
def mock_scheduler(monkeypatch):
    scheduler = MagicMock()
    monkeypatch.setattr("app.services.ibkr.connection.scheduler", scheduler)
    return scheduler


# IBKR Contracts fixtures
@pytest.fixture
def mock_logger_contracts(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr("app.services.ibkr.contracts.logger", logger)
    return logger


@pytest.fixture
def mock_load_file(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("app.services.ibkr.contracts.load_file", mock)
    return mock


@pytest.fixture
def mock_save_file(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("app.services.ibkr.contracts.save_file", mock)
    return mock


@pytest.fixture
def mock_parse_symbol(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("app.services.ibkr.contracts.parse_symbol", mock)
    return mock


@pytest.fixture
def mock_fetch_contract(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("app.services.ibkr.contracts.fetch_contract", mock)
    return mock


@pytest.fixture
def mock_get_closest_contract(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("app.services.ibkr.contracts.get_closest_contract", mock)
    return mock


# Flask app fixtures
@pytest.fixture
def app():
    app = Flask(__name__)
    app.register_blueprint(webhook_blueprint)
    return app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def flask_app():
    app = Flask(__name__)
    with app.app_context():
        yield app
