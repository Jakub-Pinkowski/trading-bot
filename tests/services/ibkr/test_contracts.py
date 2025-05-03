from unittest.mock import patch

import pytest

from app.services.ibkr.contracts import get_contract_id
from config import MIN_DAYS_UNTIL_EXPIRY


@patch('app.services.ibkr.contracts.load_file')
@patch('app.services.ibkr.contracts.parse_symbol')
@patch('app.services.ibkr.contracts.get_closest_contract')
def test_get_contract_id_from_cache(mock_get_closest_contract, mock_parse_symbol, mock_load_file):
    # Setup
    mock_parse_symbol.return_value = "ES"
    mock_load_file.return_value = {"ES": [{"conid": "123456", "expiry": "20231215"}]}
    mock_get_closest_contract.return_value = {"conid": "123456", "expiry": "20231215"}

    # Execute
    result = get_contract_id("ES", MIN_DAYS_UNTIL_EXPIRY)

    # Assert
    assert result == "123456"
    mock_parse_symbol.assert_called_once_with("ES")
    mock_load_file.assert_called_once()
    mock_get_closest_contract.assert_called_once_with([{"conid": "123456", "expiry": "20231215"}], MIN_DAYS_UNTIL_EXPIRY)


@patch('app.services.ibkr.contracts.load_file')
@patch('app.services.ibkr.contracts.save_file')
@patch('app.services.ibkr.contracts.parse_symbol')
@patch('app.services.ibkr.contracts.fetch_contract')
@patch('app.services.ibkr.contracts.get_closest_contract')
def test_get_contract_id_cache_miss(mock_get_closest_contract, mock_fetch_contract,
                                    mock_parse_symbol, mock_save_file, mock_load_file):
    # Setup
    mock_parse_symbol.return_value = "ES"
    mock_load_file.return_value = {}  # Empty cache
    mock_fetch_contract.return_value = [{"conid": "123456", "expiry": "20231215"}]
    mock_get_closest_contract.return_value = {"conid": "123456", "expiry": "20231215"}

    # Execute
    result = get_contract_id("ES", MIN_DAYS_UNTIL_EXPIRY)

    # Assert
    assert result == "123456"
    mock_parse_symbol.assert_called_once_with("ES")
    mock_load_file.assert_called_once()
    mock_fetch_contract.assert_called_once_with("ES")
    mock_save_file.assert_called_once()
    mock_get_closest_contract.assert_called_once_with([{"conid": "123456", "expiry": "20231215"}], MIN_DAYS_UNTIL_EXPIRY)


@patch('app.services.ibkr.contracts.load_file')
@patch('app.services.ibkr.contracts.save_file')
@patch('app.services.ibkr.contracts.parse_symbol')
@patch('app.services.ibkr.contracts.fetch_contract')
@patch('app.services.ibkr.contracts.get_closest_contract')
def test_get_contract_id_cache_invalid(mock_get_closest_contract, mock_fetch_contract,
                                       mock_parse_symbol, mock_save_file, mock_load_file):
    # Setup
    mock_parse_symbol.return_value = "ES"
    mock_load_file.return_value = {"ES": "invalid"}  # Invalid cache entry (not a list)
    mock_fetch_contract.return_value = [{"conid": "123456", "expiry": "20231215"}]
    mock_get_closest_contract.return_value = {"conid": "123456", "expiry": "20231215"}

    # Execute
    result = get_contract_id("ES", MIN_DAYS_UNTIL_EXPIRY)

    # Assert
    assert result == "123456"
    mock_parse_symbol.assert_called_once_with("ES")
    mock_load_file.assert_called_once()
    mock_fetch_contract.assert_called_once_with("ES")
    mock_save_file.assert_called_once()
    mock_get_closest_contract.assert_called_once_with([{"conid": "123456", "expiry": "20231215"}], MIN_DAYS_UNTIL_EXPIRY)


@patch('app.services.ibkr.contracts.load_file')
@patch('app.services.ibkr.contracts.save_file')
@patch('app.services.ibkr.contracts.parse_symbol')
@patch('app.services.ibkr.contracts.get_closest_contract')
@patch('app.services.ibkr.contracts.fetch_contract')
def test_get_contract_id_cache_value_error(mock_fetch_contract, mock_get_closest_contract,
                                           mock_parse_symbol, mock_save_file, mock_load_file):
    # Setup
    mock_parse_symbol.return_value = "ES"
    mock_load_file.return_value = {"ES": [{"conid": "123456", "expiry": "20231215"}]}
    # First call raises ValueError, second call returns valid contract
    mock_get_closest_contract.side_effect = [
        ValueError("No valid contract found"),
        {"conid": "123456", "expiry": "20231215"}
    ]
    mock_fetch_contract.return_value = [{"conid": "123456", "expiry": "20231215"}]

    # Execute
    result = get_contract_id("ES", MIN_DAYS_UNTIL_EXPIRY)

    # Assert
    assert result == "123456"
    mock_parse_symbol.assert_called_once_with("ES")
    mock_load_file.assert_called_once()
    mock_fetch_contract.assert_called_once_with("ES")
    mock_save_file.assert_called_once()
    assert mock_get_closest_contract.call_count == 2


@patch('app.services.ibkr.contracts.load_file')
@patch('app.services.ibkr.contracts.save_file')
@patch('app.services.ibkr.contracts.parse_symbol')
@patch('app.services.ibkr.contracts.fetch_contract')
def test_get_contract_id_no_contracts_found(mock_fetch_contract, mock_parse_symbol,
                                            mock_save_file, mock_load_file):
    # Setup
    mock_parse_symbol.return_value = "ES"
    mock_load_file.return_value = {}  # Empty cache
    mock_fetch_contract.return_value = []  # No contracts found

    # Execute and Assert
    with pytest.raises(ValueError) as context:
        get_contract_id("ES", MIN_DAYS_UNTIL_EXPIRY)

    assert "No contracts found for symbol" in str(context.value)
    mock_parse_symbol.assert_called_once_with("ES")
    mock_load_file.assert_called_once()
    mock_fetch_contract.assert_called_once_with("ES")
    mock_save_file.assert_called_once()
