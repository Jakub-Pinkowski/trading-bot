import pytest

from app.services.ibkr.contracts import get_contract_id
from config import MIN_DAYS_UNTIL_EXPIRY, CONTRACTS_FILE_PATH


def test_get_contract_id_from_cache(
    mock_logger_contracts, mock_load_file, mock_parse_symbol, mock_get_closest_contract
):
    """Test that get_contract_id returns the correct contract ID from cache"""

    # Mock symbol parsing, cache loading with valid data, and contract selection
    mock_parse_symbol.return_value = "ES"
    mock_load_file.return_value = {"ES": [{"conid": "123456", "expiry": "20231215"}]}
    mock_get_closest_contract.return_value = {"conid": "123456", "expiry": "20231215"}

    # Call get_contract_id with a symbol and default expiry days
    result = get_contract_id("ES", MIN_DAYS_UNTIL_EXPIRY)

    # Verify correct contract ID is returned, cache is used, and no warnings are logged
    assert result == "123456"
    mock_parse_symbol.assert_called_once_with("ES")
    mock_load_file.assert_called_once()
    mock_get_closest_contract.assert_called_once_with(
        [{"conid": "123456", "expiry": "20231215"}], MIN_DAYS_UNTIL_EXPIRY
    )
    mock_logger_contracts.warning.assert_not_called()


def test_get_contract_id_cache_miss(
    mock_logger_contracts,
    mock_load_file,
    mock_save_file,
    mock_parse_symbol,
    mock_fetch_contract,
    mock_get_closest_contract
):
    """Test that get_contract_id fetches and caches contracts when not found in cache"""

    # Mock symbol parsing, empty cache, contract fetching, and contract selection
    mock_parse_symbol.return_value = "ES"
    mock_load_file.return_value = {}  # Empty cache
    mock_fetch_contract.return_value = [{"conid": "123456", "expiry": "20231215"}]
    mock_get_closest_contract.return_value = {"conid": "123456", "expiry": "20231215"}

    # Call get_contract_id with a symbol when the contract is not in cache
    result = get_contract_id("ES", MIN_DAYS_UNTIL_EXPIRY)

    # Verify correct contract ID is returned, cache is updated, and no errors are logged
    assert result == "123456"
    mock_parse_symbol.assert_called_once_with("ES")
    mock_load_file.assert_called_once()
    mock_fetch_contract.assert_called_once_with("ES")
    mock_save_file.assert_called_once_with({"ES": [{"conid": "123456", "expiry": "20231215"}]}, CONTRACTS_FILE_PATH)
    mock_get_closest_contract.assert_called_once_with([{"conid": "123456", "expiry": "20231215"}],
                                                      MIN_DAYS_UNTIL_EXPIRY)
    mock_logger_contracts.warning.assert_not_called()
    mock_logger_contracts.error.assert_not_called()


def test_get_contract_id_cache_invalid(
    mock_logger_contracts,
    mock_load_file,
    mock_save_file,
    mock_parse_symbol,
    mock_fetch_contract,
    mock_get_closest_contract
):
    """Test that get_contract_id handles invalid cache data by refreshing contracts"""

    # Mock symbol parsing, invalid cache data, contract fetching, and contract selection
    mock_parse_symbol.return_value = "ES"
    mock_load_file.return_value = {"ES": "invalid"}  # Invalid cache entry (not a list)
    mock_fetch_contract.return_value = [{"conid": "123456", "expiry": "20231215"}]
    mock_get_closest_contract.return_value = {"conid": "123456", "expiry": "20231215"}

    # Call get_contract_id with a symbol when the cache contains invalid data
    result = get_contract_id("ES", MIN_DAYS_UNTIL_EXPIRY)

    # Verify correct contract ID is returned, cache is refreshed with valid data, and no errors are logged
    assert result == "123456"
    mock_parse_symbol.assert_called_once_with("ES")
    mock_load_file.assert_called_once()
    mock_fetch_contract.assert_called_once_with("ES")
    mock_save_file.assert_called_once_with({"ES": [{"conid": "123456", "expiry": "20231215"}]}, CONTRACTS_FILE_PATH)
    mock_get_closest_contract.assert_called_once_with([{"conid": "123456", "expiry": "20231215"}],
                                                      MIN_DAYS_UNTIL_EXPIRY)
    mock_logger_contracts.warning.assert_not_called()
    mock_logger_contracts.error.assert_not_called()


def test_get_contract_id_cache_value_error(
    mock_logger_contracts,
    mock_load_file,
    mock_save_file,
    mock_parse_symbol,
    mock_fetch_contract,
    mock_get_closest_contract
):
    """Test that get_contract_id refreshes contracts when get_closest_contract raises ValueError"""

    # Mock symbol parsing, cache data, and contract selection that first fails then succeeds
    mock_parse_symbol.return_value = "ES"
    mock_load_file.return_value = {"ES": [{"conid": "123456", "expiry": "20231215"}]}
    mock_get_closest_contract.side_effect = [
        ValueError("No valid contract found"),
        {"conid": "123456", "expiry": "20231215"},
    ]
    mock_fetch_contract.return_value = [{"conid": "123456", "expiry": "20231215"}]

    # Call get_contract_id with a symbol when get_closest_contract initially raises ValueError
    result = get_contract_id("ES", MIN_DAYS_UNTIL_EXPIRY)

    # Verify correct contract ID is returned after refreshing contracts, warning is logged, and no errors
    assert result == "123456"
    mock_parse_symbol.assert_called_once_with("ES")
    mock_load_file.assert_called_once_with(CONTRACTS_FILE_PATH)
    mock_fetch_contract.assert_called_once_with("ES")
    contracts_cache_saved = {"ES": [{"conid": "123456", "expiry": "20231215"}]}
    mock_save_file.assert_called_once_with(contracts_cache_saved, CONTRACTS_FILE_PATH)
    assert mock_get_closest_contract.call_count == 2
    assert mock_get_closest_contract.call_args_list[0][0][0] == [{"conid": "123456", "expiry": "20231215"}]
    assert mock_get_closest_contract.call_args_list[1][0][0] == [{"conid": "123456", "expiry": "20231215"}]
    mock_logger_contracts.warning.assert_called_once()
    mock_logger_contracts.error.assert_not_called()


def test_get_contract_id_no_contracts_found(
    mock_logger_contracts, mock_load_file, mock_save_file, mock_parse_symbol, mock_fetch_contract
):
    """Test that get_contract_id raises ValueError when no contracts are found for the symbol"""

    # Mock symbol parsing, empty cache, and contract fetching that returns no contracts
    mock_parse_symbol.return_value = "ES"
    mock_load_file.return_value = {}  # Empty cache
    mock_fetch_contract.return_value = []  # No contracts found

    # Verify get_contract_id raises ValueError when no contracts are found for the symbol
    with pytest.raises(ValueError) as context:
        get_contract_id("ES", MIN_DAYS_UNTIL_EXPIRY)

    # Verify error message, API calls, and that error is logged but cache is not updated
    assert "No contracts found for symbol" in str(context.value)
    mock_parse_symbol.assert_called_once_with("ES")
    mock_load_file.assert_called_once()
    mock_fetch_contract.assert_called_once_with("ES")
    mock_save_file.assert_not_called()
    mock_logger_contracts.error.assert_called_once()
    mock_logger_contracts.warning.assert_not_called()


def test_get_contract_id_no_valid_contract_in_fresh_data(
    mock_logger_contracts,
    mock_load_file,
    mock_save_file,
    mock_parse_symbol,
    mock_fetch_contract,
    mock_get_closest_contract
):
    """Test that get_contract_id raises ValueError when no valid contract is found in fresh data"""

    # Mock symbol parsing, empty cache, contract fetching, and contract selection that fails
    mock_parse_symbol.return_value = "ES"
    mock_load_file.return_value = {}  # Empty cache
    mock_fetch_contract.return_value = [{"conid": "123456", "expiry": "20231215"}]
    mock_get_closest_contract.side_effect = ValueError("No valid contract found")

    # Verify get_contract_id raises ValueError when no valid contract is found
    with pytest.raises(ValueError) as context:
        get_contract_id("ES", MIN_DAYS_UNTIL_EXPIRY)

    # Verify error message, API calls, and that error is logged
    assert "No valid contract found" in str(context.value)
    mock_parse_symbol.assert_called_once_with("ES")
    mock_load_file.assert_called_once()
    mock_fetch_contract.assert_called_once_with("ES")
    mock_save_file.assert_called_once()
    mock_logger_contracts.error.assert_called_once()
    mock_logger_contracts.warning.assert_not_called()
