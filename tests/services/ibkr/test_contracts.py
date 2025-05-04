from unittest.mock import MagicMock

import pytest

from app.services.ibkr.contracts import get_contract_id
from config import MIN_DAYS_UNTIL_EXPIRY, CONTRACTS_FILE_PATH


@pytest.fixture
def mock_logger(monkeypatch):
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


def test_get_contract_id_from_cache(
        mock_logger, mock_load_file, mock_parse_symbol, mock_get_closest_contract
):
    mock_parse_symbol.return_value = "ES"
    mock_load_file.return_value = {"ES": [{"conid": "123456", "expiry": "20231215"}]}
    mock_get_closest_contract.return_value = {"conid": "123456", "expiry": "20231215"}

    result = get_contract_id("ES", MIN_DAYS_UNTIL_EXPIRY)

    assert result == "123456"
    mock_parse_symbol.assert_called_once_with("ES")
    mock_load_file.assert_called_once()
    mock_get_closest_contract.assert_called_once_with(
        [{"conid": "123456", "expiry": "20231215"}], MIN_DAYS_UNTIL_EXPIRY
    )
    mock_logger.warning.assert_not_called()


def test_get_contract_id_cache_miss(
        mock_logger, mock_load_file, mock_save_file, mock_parse_symbol, mock_fetch_contract, mock_get_closest_contract
):
    mock_parse_symbol.return_value = "ES"
    mock_load_file.return_value = {}  # Empty cache
    mock_fetch_contract.return_value = [{"conid": "123456", "expiry": "20231215"}]
    mock_get_closest_contract.return_value = {"conid": "123456", "expiry": "20231215"}

    result = get_contract_id("ES", MIN_DAYS_UNTIL_EXPIRY)

    assert result == "123456"
    mock_parse_symbol.assert_called_once_with("ES")
    mock_load_file.assert_called_once()
    mock_fetch_contract.assert_called_once_with("ES")
    mock_save_file.assert_called_once_with({"ES": [{"conid": "123456", "expiry": "20231215"}]}, CONTRACTS_FILE_PATH)
    mock_get_closest_contract.assert_called_once_with([{"conid": "123456", "expiry": "20231215"}], MIN_DAYS_UNTIL_EXPIRY)
    mock_logger.warning.assert_not_called()
    mock_logger.error.assert_not_called()


def test_get_contract_id_cache_invalid(
        mock_logger, mock_load_file, mock_save_file, mock_parse_symbol, mock_fetch_contract, mock_get_closest_contract
):
    mock_parse_symbol.return_value = "ES"
    mock_load_file.return_value = {"ES": "invalid"}  # Invalid cache entry (not a list)
    mock_fetch_contract.return_value = [{"conid": "123456", "expiry": "20231215"}]
    mock_get_closest_contract.return_value = {"conid": "123456", "expiry": "20231215"}

    result = get_contract_id("ES", MIN_DAYS_UNTIL_EXPIRY)

    assert result == "123456"
    mock_parse_symbol.assert_called_once_with("ES")
    mock_load_file.assert_called_once()
    mock_fetch_contract.assert_called_once_with("ES")
    mock_save_file.assert_called_once_with({"ES": [{"conid": "123456", "expiry": "20231215"}]}, CONTRACTS_FILE_PATH)
    mock_get_closest_contract.assert_called_once_with([{"conid": "123456", "expiry": "20231215"}], MIN_DAYS_UNTIL_EXPIRY)
    mock_logger.warning.assert_not_called()
    mock_logger.error.assert_not_called()


def test_get_contract_id_cache_value_error(
        mock_logger, mock_load_file, mock_save_file, mock_parse_symbol, mock_fetch_contract, mock_get_closest_contract
):
    mock_parse_symbol.return_value = "ES"
    mock_load_file.return_value = {"ES": [{"conid": "123456", "expiry": "20231215"}]}
    mock_get_closest_contract.side_effect = [
        ValueError("No valid contract found"),
        {"conid": "123456", "expiry": "20231215"},
    ]
    mock_fetch_contract.return_value = [{"conid": "123456", "expiry": "20231215"}]

    result = get_contract_id("ES", MIN_DAYS_UNTIL_EXPIRY)

    assert result == "123456"
    mock_parse_symbol.assert_called_once_with("ES")
    mock_load_file.assert_called_once_with(CONTRACTS_FILE_PATH)
    mock_fetch_contract.assert_called_once_with("ES")
    contracts_cache_saved = {"ES": [{"conid": "123456", "expiry": "20231215"}]}
    mock_save_file.assert_called_once_with(contracts_cache_saved, CONTRACTS_FILE_PATH)
    assert mock_get_closest_contract.call_count == 2
    assert mock_get_closest_contract.call_args_list[0][0][0] == [{"conid": "123456", "expiry": "20231215"}]
    assert mock_get_closest_contract.call_args_list[1][0][0] == [{"conid": "123456", "expiry": "20231215"}]
    mock_logger.warning.assert_called_once()
    mock_logger.error.assert_not_called()


def test_get_contract_id_no_contracts_found(
        mock_logger, mock_load_file, mock_save_file, mock_parse_symbol, mock_fetch_contract
):
    mock_parse_symbol.return_value = "ES"
    mock_load_file.return_value = {}  # Empty cache
    mock_fetch_contract.return_value = []  # No contracts found

    with pytest.raises(ValueError) as context:
        get_contract_id("ES", MIN_DAYS_UNTIL_EXPIRY)

    assert "No contracts found for symbol" in str(context.value)
    mock_parse_symbol.assert_called_once_with("ES")
    mock_load_file.assert_called_once()
    mock_fetch_contract.assert_called_once_with("ES")
    mock_save_file.assert_not_called()
    mock_logger.error.assert_called_once()
    mock_logger.warning.assert_not_called()
