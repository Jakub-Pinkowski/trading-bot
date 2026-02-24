"""
Shared fixtures for IBKR integration tests.

Provides monkeypatched mocks for all IBKR module dependencies,
organized by module: ibkr_service, connection, contracts, and orders.
"""
from unittest.mock import MagicMock

import pytest


# ==================== IBKR Service Fixtures ====================

@pytest.fixture
def mock_logger_ibkr_service(monkeypatch):
    """Mock logger for ibkr_service module."""
    mock = MagicMock()
    monkeypatch.setattr('app.ibkr.ibkr_service.logger', mock)
    return mock


@pytest.fixture
def mock_get_contract_id(monkeypatch):
    """Mock get_contract_id dependency in ibkr_service module."""
    mock = MagicMock()
    monkeypatch.setattr('app.ibkr.ibkr_service.get_contract_id', mock)
    return mock


@pytest.fixture
def mock_place_order(monkeypatch):
    """Mock place_order dependency in ibkr_service module."""
    mock = MagicMock()
    monkeypatch.setattr('app.ibkr.ibkr_service.place_order', mock)
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
def mock_parse_symbol(monkeypatch):
    """Mock parse_symbol dependency in contracts module."""
    mock = MagicMock()
    monkeypatch.setattr('app.ibkr.contracts.parse_symbol', mock)
    return mock


@pytest.fixture
def mock_map_tv_to_ibkr(monkeypatch):
    """Mock map_tv_to_ibkr dependency in contracts module."""
    mock = MagicMock()
    monkeypatch.setattr('app.ibkr.contracts.map_tv_to_ibkr', mock)
    return mock


@pytest.fixture
def mock_fetch_contract(monkeypatch):
    """Mock fetch_contract dependency in contracts module."""
    mock = MagicMock()
    monkeypatch.setattr('app.ibkr.contracts._fetch_contract', mock)
    return mock


@pytest.fixture
def mock_get_closest_contract(monkeypatch):
    """Mock get_closest_contract dependency in contracts module."""
    mock = MagicMock()
    monkeypatch.setattr('app.ibkr.contracts._get_closest_contract', mock)
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
