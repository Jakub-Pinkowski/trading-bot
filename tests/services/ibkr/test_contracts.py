from unittest.mock import patch

import pytest

from app.services.ibkr.contracts import get_contract_id
from config import MIN_DAYS_UNTIL_EXPIRY, CONTRACTS_FILE_PATH


@patch('app.services.ibkr.contracts.logger')
@patch('app.services.ibkr.contracts.load_file')
@patch('app.services.ibkr.contracts.parse_symbol')
@patch('app.services.ibkr.contracts.get_closest_contract')
def test_get_contract_id_from_cache(mock_get_closest_contract, mock_parse_symbol, mock_load_file, mock_logger):
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
    mock_logger.warning.assert_not_called()


@patch('app.services.ibkr.contracts.logger')
@patch('app.services.ibkr.contracts.load_file')
@patch('app.services.ibkr.contracts.save_file')
@patch('app.services.ibkr.contracts.parse_symbol')
@patch('app.services.ibkr.contracts.fetch_contract')
@patch('app.services.ibkr.contracts.get_closest_contract')
def test_get_contract_id_cache_miss(mock_get_closest_contract, mock_fetch_contract,
                                    mock_parse_symbol, mock_save_file, mock_load_file, mock_logger):
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
    mock_save_file.assert_called_once_with({"ES": [{"conid": "123456", "expiry": "20231215"}]}, CONTRACTS_FILE_PATH)
    mock_get_closest_contract.assert_called_once_with([{"conid": "123456", "expiry": "20231215"}], MIN_DAYS_UNTIL_EXPIRY)
    mock_logger.warning.assert_not_called()
    mock_logger.error.assert_not_called()


@patch('app.services.ibkr.contracts.logger')
@patch('app.services.ibkr.contracts.load_file')
@patch('app.services.ibkr.contracts.save_file')
@patch('app.services.ibkr.contracts.parse_symbol')
@patch('app.services.ibkr.contracts.fetch_contract')
@patch('app.services.ibkr.contracts.get_closest_contract')
def test_get_contract_id_cache_invalid(mock_get_closest_contract, mock_fetch_contract,
                                       mock_parse_symbol, mock_save_file, mock_load_file, mock_logger):
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
    mock_save_file.assert_called_once_with({"ES": [{"conid": "123456", "expiry": "20231215"}]}, CONTRACTS_FILE_PATH)
    mock_get_closest_contract.assert_called_once_with([{"conid": "123456", "expiry": "20231215"}], MIN_DAYS_UNTIL_EXPIRY)
    # Logger should not be called since we're not even attempting to use the invalid cache
    mock_logger.warning.assert_not_called()
    mock_logger.error.assert_not_called()


@patch('app.services.ibkr.contracts.logger')
@patch('app.services.ibkr.contracts.load_file')
@patch('app.services.ibkr.contracts.save_file')
@patch('app.services.ibkr.contracts.parse_symbol')
@patch('app.services.ibkr.contracts.get_closest_contract')
@patch('app.services.ibkr.contracts.fetch_contract')
def test_get_contract_id_cache_value_error(
        mock_fetch_contract,
        mock_get_closest_contract,
        mock_parse_symbol,
        mock_save_file,
        mock_load_file,
        mock_logger,
):
    # Setup
    mock_parse_symbol.return_value = "ES"
    mock_load_file.return_value = {"ES": [{"conid": "123456", "expiry": "20231215"}]}
    # First call on cache fails, second call (on fresh contracts) succeeds
    mock_get_closest_contract.side_effect = [
        ValueError("No valid contract found"),
        {"conid": "123456", "expiry": "20231215"},
    ]
    mock_fetch_contract.return_value = [{"conid": "123456", "expiry": "20231215"}]

    # Execute
    result = get_contract_id("ES", MIN_DAYS_UNTIL_EXPIRY)

    # Assert
    assert result == "123456"
    mock_parse_symbol.assert_called_once_with("ES")
    mock_load_file.assert_called_once_with(CONTRACTS_FILE_PATH)
    mock_fetch_contract.assert_called_once_with("ES")
    # contracts_cache gets updated with fresh contracts and saved
    contracts_cache_saved = {"ES": [{"conid": "123456", "expiry": "20231215"}]}
    mock_save_file.assert_called_once_with(contracts_cache_saved, CONTRACTS_FILE_PATH)
    assert mock_get_closest_contract.call_count == 2
    assert mock_get_closest_contract.call_args_list[0][0][0] == [{"conid": "123456", "expiry": "20231215"}]
    assert mock_get_closest_contract.call_args_list[1][0][0] == [{"conid": "123456", "expiry": "20231215"}]
    # Logger should warn about the cache being invalid
    mock_logger.warning.assert_called_once()
    mock_logger.error.assert_not_called()


@patch('app.services.ibkr.contracts.logger')
@patch('app.services.ibkr.contracts.load_file')
@patch('app.services.ibkr.contracts.save_file')
@patch('app.services.ibkr.contracts.parse_symbol')
@patch('app.services.ibkr.contracts.fetch_contract')
def test_get_contract_id_no_contracts_found(mock_fetch_contract, mock_parse_symbol,
                                            mock_save_file, mock_load_file, mock_logger):
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
    # In the updated implementation, save_file is not called when no contracts are found
    mock_save_file.assert_not_called()
    # Logger should log an error about no contracts found
    mock_logger.error.assert_called_once()
    mock_logger.warning.assert_not_called()
