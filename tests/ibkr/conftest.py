"""
Shared fixtures for IBKR integration tests.

Provides monkeypatched mocks for all IBKR module dependencies,
organized by module: trading, connection, contracts, orders, and rollover.
"""
from unittest.mock import MagicMock

import pytest


# ==================== Trading Fixtures ====================

@pytest.fixture
def mock_logger_trading(monkeypatch):
    """Mock logger for trading module."""
    mock = MagicMock()
    monkeypatch.setattr('app.ibkr.trading.logger', mock)
    return mock


@pytest.fixture
def mock_contract_resolver(monkeypatch):
    """Mock ContractResolver in trading module, returning a controllable instance."""
    mock_instance = MagicMock()
    mock_class = MagicMock(return_value=mock_instance)
    monkeypatch.setattr('app.ibkr.trading.ContractResolver', mock_class)
    mock_instance._class = mock_class
    return mock_instance


@pytest.fixture
def mock_place_order(monkeypatch):
    """Mock place_order dependency in trading module."""
    mock = MagicMock()
    monkeypatch.setattr('app.ibkr.trading.place_order', mock)
    return mock


# ==================== Connection Fixtures ====================

@pytest.fixture
def mock_logger_connection(monkeypatch):
    """Mock logger for connection module."""
    mock = MagicMock()
    monkeypatch.setattr('app.ibkr.connection.logger', mock)
    return mock


@pytest.fixture
def mock_api_post_connection(monkeypatch):
    """Mock api_post dependency in connection module."""
    mock = MagicMock()
    monkeypatch.setattr('app.ibkr.connection.api_post', mock)
    return mock


@pytest.fixture
def mock_scheduler(monkeypatch):
    """Mock APScheduler instance in connection module."""
    mock = MagicMock()
    monkeypatch.setattr('app.ibkr.connection.scheduler', mock)
    return mock


# ==================== Contracts Fixtures ====================

@pytest.fixture
def mock_logger_contracts(monkeypatch):
    """Mock logger for contracts module."""
    mock = MagicMock()
    monkeypatch.setattr('app.ibkr.contracts.logger', mock)
    return mock


@pytest.fixture
def mock_api_get_contracts(monkeypatch):
    """Mock api_get dependency in contracts module."""
    mock = MagicMock()
    monkeypatch.setattr('app.ibkr.contracts.api_get', mock)
    return mock


@pytest.fixture
def mock_load_file(monkeypatch):
    """Mock load_file dependency in contracts module."""
    mock = MagicMock()
    monkeypatch.setattr('app.ibkr.contracts.load_file', mock)
    return mock


@pytest.fixture
def mock_save_file(monkeypatch):
    """Mock save_file dependency in contracts module."""
    mock = MagicMock()
    monkeypatch.setattr('app.ibkr.contracts.save_file', mock)
    return mock


@pytest.fixture
def mock_yaml_load(monkeypatch):
    """Mock builtins.open + yaml.safe_load so _load_next_switch_date never touches disk."""
    from unittest.mock import mock_open
    mock = MagicMock()
    monkeypatch.setattr('builtins.open', mock_open())
    monkeypatch.setattr('app.ibkr.contracts.yaml.safe_load', mock)
    return mock


# ==================== Orders Fixtures ====================

@pytest.fixture
def mock_logger_orders(monkeypatch):
    """Mock logger for orders module."""
    mock = MagicMock()
    monkeypatch.setattr('app.ibkr.orders.logger', mock)
    return mock


@pytest.fixture
def mock_api_post_orders(monkeypatch):
    """Mock api_post dependency in orders module."""
    mock = MagicMock()
    monkeypatch.setattr('app.ibkr.orders.api_post', mock)
    return mock


@pytest.fixture
def mock_api_get_orders(monkeypatch):
    """Mock api_get dependency in orders module."""
    mock = MagicMock()
    monkeypatch.setattr('app.ibkr.orders.api_get', mock)
    return mock


@pytest.fixture
def mock_get_contract_position(monkeypatch):
    """Mock get_contract_position dependency in orders module."""
    mock = MagicMock()
    monkeypatch.setattr('app.ibkr.orders._get_contract_position', mock)
    return mock


@pytest.fixture
def mock_suppress_messages(monkeypatch):
    """Mock suppress_messages dependency in orders module."""
    mock = MagicMock()
    monkeypatch.setattr('app.ibkr.orders._suppress_messages', mock)
    return mock


@pytest.fixture
def mock_invalidate_cache(monkeypatch):
    """Mock invalidate_cache dependency in orders module."""
    mock = MagicMock()
    monkeypatch.setattr('app.ibkr.orders._invalidate_cache', mock)
    return mock


# ==================== Rollover Fixtures ====================

@pytest.fixture
def mock_logger_rollover(monkeypatch):
    """Mock logger for rollover module."""
    mock = MagicMock()
    monkeypatch.setattr('app.ibkr.rollover.logger', mock)
    return mock


@pytest.fixture
def mock_contract_resolver_rollover(monkeypatch):
    """Mock ContractResolver in rollover module, returning a controllable instance."""
    mock_instance = MagicMock()
    mock_class = MagicMock(return_value=mock_instance)
    monkeypatch.setattr('app.ibkr.rollover.ContractResolver', mock_class)
    mock_instance._class = mock_class
    return mock_instance


@pytest.fixture
def mock_place_order_rollover(monkeypatch):
    """Mock place_order dependency in rollover module."""
    mock = MagicMock()
    monkeypatch.setattr('app.ibkr.rollover.place_order', mock)
    return mock


@pytest.fixture
def mock_get_contract_position_rollover(monkeypatch):
    """Mock _get_contract_position dependency in rollover module."""
    mock = MagicMock()
    monkeypatch.setattr('app.ibkr.rollover._get_contract_position', mock)
    return mock


@pytest.fixture
def mock_check_and_rollover_position(monkeypatch):
    """Mock check_and_rollover_position in rollover module."""
    mock = MagicMock()
    monkeypatch.setattr('app.ibkr.rollover.check_and_rollover_position', mock)
    return mock
