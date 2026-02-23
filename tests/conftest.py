from unittest.mock import MagicMock

import pytest
from flask import Flask

from app.routes.webhook import webhook_blueprint

# ==================== Backtesting Fixtures Plugin Registration ====================
# Import all backtesting fixtures for availability across all test modules
pytest_plugins = [
    'tests.backtesting.fixtures.data_fixtures',
    'tests.backtesting.fixtures.strategy_fixtures',
]


# ==================== Flask Fixtures ====================

@pytest.fixture
def mock_process_trading_data(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("app.routes.webhook.process_trading_data", mock)
    return mock


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
