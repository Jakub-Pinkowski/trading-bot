"""
Tests for IBKR Contracts Module.

Tests cover:
- Contract fetching from the IBKR API
- Selecting the closest valid contract by expiration date
- Contract ID retrieval with caching, cache miss, invalid cache data,
  and error handling for missing or expired contracts
"""
from datetime import datetime, timedelta

import pytest

from app.ibkr.contracts import (
    fetch_contract,
    get_closest_contract,
    get_contract_id,
    CONTRACTS_FILE_PATH,
    MIN_DAYS_UNTIL_EXPIRY,
)


# ==================== Test Classes ====================

class TestFetchContract:
    """Test contract fetching from IBKR API."""

    def test_success(self, mock_api_get_contracts):
        """Test successful fetch returns contract list for the symbol."""
        mock_api_get_contracts.return_value = {"ZC": [{"conid": "123456", "expirationDate": "20231215"}]}

        result = fetch_contract("ZC")

        assert result == [{"conid": "123456", "expirationDate": "20231215"}]
        mock_api_get_contracts.assert_called_once_with("/trsrv/futures?symbols=ZC")

    def test_api_error_returns_empty_list(self, mock_api_get_contracts):
        """Test API error is caught and empty list returned."""
        mock_api_get_contracts.side_effect = Exception("API error")

        result = fetch_contract("ZC")

        assert result == []
        mock_api_get_contracts.assert_called_once_with("/trsrv/futures?symbols=ZC")

    def test_empty_response_returns_empty_list(self, mock_api_get_contracts):
        """Test empty API response returns empty list."""
        mock_api_get_contracts.return_value = {}

        result = fetch_contract("ZC")

        assert result == []
        mock_api_get_contracts.assert_called_once_with("/trsrv/futures?symbols=ZC")


class TestGetClosestContract:
    """Test selection of the closest valid contract by expiration date."""

    def test_returns_earliest_valid_contract(self):
        """Test the contract with the earliest valid expiration is selected."""
        today = datetime.today()
        earlier = today + timedelta(days=MIN_DAYS_UNTIL_EXPIRY + 10)
        later = today + timedelta(days=MIN_DAYS_UNTIL_EXPIRY + 30)

        contracts = [
            {"conid": "123456", "expirationDate": earlier.strftime("%Y%m%d")},
            {"conid": "789012", "expirationDate": later.strftime("%Y%m%d")},
        ]

        result = get_closest_contract(contracts)

        # Earlier expiration contract should be selected
        assert result["conid"] == "123456"

    def test_expired_contracts_raise_value_error(self):
        """Test ValueError raised when all contracts are past the expiry cutoff."""
        today = datetime.today()
        past_date = today - timedelta(days=1)

        contracts = [{"conid": "123456", "expirationDate": past_date.strftime("%Y%m%d")}]

        with pytest.raises(ValueError, match="No valid contracts available for expiry cutoff"):
            get_closest_contract(contracts)

    def test_empty_list_raises_value_error(self):
        """Test ValueError raised when contract list is empty."""
        with pytest.raises(ValueError, match="No valid contracts available for expiry cutoff"):
            get_closest_contract([])

    def test_custom_min_days_filters_near_contracts(self):
        """Test custom min_days_until_expiry excludes contracts expiring too soon."""
        today = datetime.today()
        near = today + timedelta(days=10)
        far = today + timedelta(days=30)

        contracts = [
            {"conid": "123456", "expirationDate": near.strftime("%Y%m%d")},
            {"conid": "789012", "expirationDate": far.strftime("%Y%m%d")},
        ]

        # min_days=15 should exclude the near contract and return the far one
        result = get_closest_contract(contracts, min_days_until_expiry=15)

        assert result["conid"] == "789012"


class TestGetContractId:
    """Test contract ID retrieval with caching and error handling."""

    def test_returns_id_from_cache(
        self, mock_logger_contracts, mock_load_file, mock_parse_symbol, mock_get_closest_contract
    ):
        """Test contract ID returned from cache without fetching from API."""
        mock_parse_symbol.return_value = "ZC"
        mock_load_file.return_value = {"ZC": [{"conid": "123456", "expiry": "20231215"}]}
        mock_get_closest_contract.return_value = {"conid": "123456", "expiry": "20231215"}

        result = get_contract_id("ZC", MIN_DAYS_UNTIL_EXPIRY)

        assert result == "123456"
        mock_parse_symbol.assert_called_once_with("ZC")
        mock_load_file.assert_called_once()
        mock_get_closest_contract.assert_called_once_with(
            [{"conid": "123456", "expiry": "20231215"}], MIN_DAYS_UNTIL_EXPIRY
        )
        mock_logger_contracts.warning.assert_not_called()

    def test_fetches_and_caches_on_cache_miss(
        self,
        mock_logger_contracts,
        mock_load_file,
        mock_save_file,
        mock_parse_symbol,
        mock_fetch_contract,
        mock_get_closest_contract,
    ):
        """Test contracts fetched from API and saved to cache when cache is empty."""
        mock_parse_symbol.return_value = "ZC"
        mock_load_file.return_value = {}
        mock_fetch_contract.return_value = [{"conid": "123456", "expiry": "20231215"}]
        mock_get_closest_contract.return_value = {"conid": "123456", "expiry": "20231215"}

        result = get_contract_id("ZC", MIN_DAYS_UNTIL_EXPIRY)

        assert result == "123456"
        mock_fetch_contract.assert_called_once_with("ZC")
        mock_save_file.assert_called_once_with(
            {"ZC": [{"conid": "123456", "expiry": "20231215"}]}, CONTRACTS_FILE_PATH
        )
        mock_logger_contracts.warning.assert_not_called()
        mock_logger_contracts.error.assert_not_called()

    def test_refreshes_on_invalid_cache_entry(
        self,
        mock_logger_contracts,
        mock_load_file,
        mock_save_file,
        mock_parse_symbol,
        mock_fetch_contract,
        mock_get_closest_contract,
    ):
        """Test invalid cache entry (non-list) triggers a fresh API fetch."""
        mock_parse_symbol.return_value = "ZC"
        # Non-list value is treated as invalid cache data
        mock_load_file.return_value = {"ZC": "invalid"}
        mock_fetch_contract.return_value = [{"conid": "123456", "expiry": "20231215"}]
        mock_get_closest_contract.return_value = {"conid": "123456", "expiry": "20231215"}

        result = get_contract_id("ZC", MIN_DAYS_UNTIL_EXPIRY)

        assert result == "123456"
        mock_fetch_contract.assert_called_once_with("ZC")
        mock_save_file.assert_called_once_with(
            {"ZC": [{"conid": "123456", "expiry": "20231215"}]}, CONTRACTS_FILE_PATH
        )
        mock_logger_contracts.warning.assert_not_called()
        mock_logger_contracts.error.assert_not_called()

    def test_refreshes_when_cached_contracts_are_expired(
        self,
        mock_logger_contracts,
        mock_load_file,
        mock_save_file,
        mock_parse_symbol,
        mock_fetch_contract,
        mock_get_closest_contract,
    ):
        """Test fresh API fetch triggered and warning logged when cached contracts are expired."""
        mock_parse_symbol.return_value = "ZC"
        mock_load_file.return_value = {"ZC": [{"conid": "123456", "expiry": "20231215"}]}
        # First call raises ValueError (cached contracts expired), second call succeeds
        mock_get_closest_contract.side_effect = [
            ValueError("No valid contract found"),
            {"conid": "123456", "expiry": "20231215"},
        ]
        mock_fetch_contract.return_value = [{"conid": "123456", "expiry": "20231215"}]

        result = get_contract_id("ZC", MIN_DAYS_UNTIL_EXPIRY)

        assert result == "123456"
        mock_parse_symbol.assert_called_once_with("ZC")
        mock_load_file.assert_called_once_with(CONTRACTS_FILE_PATH)
        mock_fetch_contract.assert_called_once_with("ZC")
        mock_save_file.assert_called_once_with(
            {"ZC": [{"conid": "123456", "expiry": "20231215"}]}, CONTRACTS_FILE_PATH
        )
        assert mock_get_closest_contract.call_count == 2
        # Warning logged when falling back to a fresh fetch
        mock_logger_contracts.warning.assert_called_once()
        mock_logger_contracts.error.assert_not_called()

    def test_raises_when_no_contracts_found(
        self,
        mock_logger_contracts,
        mock_load_file,
        mock_save_file,
        mock_parse_symbol,
        mock_fetch_contract,
    ):
        """Test ValueError raised and error logged when API returns no contracts for the symbol."""
        mock_parse_symbol.return_value = "ZC"
        mock_load_file.return_value = {}
        mock_fetch_contract.return_value = []

        with pytest.raises(ValueError) as exc_info:
            get_contract_id("ZC", MIN_DAYS_UNTIL_EXPIRY)

        assert "No contracts found for symbol" in str(exc_info.value)
        mock_fetch_contract.assert_called_once_with("ZC")
        mock_save_file.assert_not_called()
        mock_logger_contracts.error.assert_called_once()
        mock_logger_contracts.warning.assert_not_called()

    def test_raises_when_no_valid_contract_in_fresh_data(
        self,
        mock_logger_contracts,
        mock_load_file,
        mock_save_file,
        mock_parse_symbol,
        mock_fetch_contract,
        mock_get_closest_contract,
    ):
        """Test ValueError raised and error logged when fresh data has no valid contract."""
        mock_parse_symbol.return_value = "ZC"
        mock_load_file.return_value = {}
        mock_fetch_contract.return_value = [{"conid": "123456", "expiry": "20231215"}]
        mock_get_closest_contract.side_effect = ValueError("No valid contract found")

        with pytest.raises(ValueError) as exc_info:
            get_contract_id("ZC", MIN_DAYS_UNTIL_EXPIRY)

        assert "No valid contract found" in str(exc_info.value)
        mock_save_file.assert_called_once()
        mock_logger_contracts.error.assert_called_once()
        mock_logger_contracts.warning.assert_not_called()
