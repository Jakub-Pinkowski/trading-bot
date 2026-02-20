from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from app.services.ibkr.utils.contracts_utils import (get_closest_contract,
                                                     fetch_contract,
                                                     MIN_DAYS_UNTIL_EXPIRY)


@patch('app.services.ibkr.utils.contracts_utils.parse_symbol')
@patch('app.services.ibkr.utils.contracts_utils.api_get')
def test_fetch_contract_success(mock_api_get, mock_parse_symbol):
    """Test that fetch_contract successfully fetches and returns contract data."""

    # Setup mocks
    mock_parse_symbol.return_value = "ES"
    mock_api_get.return_value = {"ES": [{"conid": "123456", "expirationDate": "20231215"}]}

    # Call the function
    result = fetch_contract("ES1!")

    # Verify the result
    assert result == [{"conid": "123456", "expirationDate": "20231215"}]
    mock_parse_symbol.assert_called_once_with("ES1!")
    mock_api_get.assert_called_once_with("/trsrv/futures?symbols=ES")


@patch('app.services.ibkr.utils.contracts_utils.parse_symbol')
@patch('app.services.ibkr.utils.contracts_utils.api_get')
def test_fetch_contract_api_error(mock_api_get, mock_parse_symbol):
    """Test that fetch_contract handles API errors gracefully."""

    # Setup mocks
    mock_parse_symbol.return_value = "ES"
    mock_api_get.side_effect = Exception("API error")

    # Call the function
    result = fetch_contract("ES1!")

    # Verify the result
    assert result == []
    mock_parse_symbol.assert_called_once_with("ES1!")
    mock_api_get.assert_called_once_with("/trsrv/futures?symbols=ES")


@patch('app.services.ibkr.utils.contracts_utils.parse_symbol')
@patch('app.services.ibkr.utils.contracts_utils.api_get')
def test_fetch_contract_empty_response(mock_api_get, mock_parse_symbol):
    """Test that fetch_contract handles empty API responses."""

    # Setup mocks
    mock_parse_symbol.return_value = "ES"
    mock_api_get.return_value = {}

    # Call the function
    result = fetch_contract("ES1!")

    # Verify the result
    assert result == []
    mock_parse_symbol.assert_called_once_with("ES1!")
    mock_api_get.assert_called_once_with("/trsrv/futures?symbols=ES")


def test_get_closest_contract_valid():
    """Test that get_closest_contract returns the closest valid contract."""

    # Create test data with multiple contracts
    today = datetime.today()
    future_date1 = today + timedelta(days=MIN_DAYS_UNTIL_EXPIRY + 10)
    future_date2 = today + timedelta(days=MIN_DAYS_UNTIL_EXPIRY + 30)

    contracts = [
        {"conid": "123456", "expirationDate": future_date1.strftime("%Y%m%d")},
        {"conid": "789012", "expirationDate": future_date2.strftime("%Y%m%d")}
    ]

    # Call the function
    result = get_closest_contract(contracts)

    # Verify the result is the contract with the earlier expiration date
    assert result["conid"] == "123456"


def test_get_closest_contract_no_valid_contracts():
    """Test that get_closest_contract raises ValueError when no valid contracts are available."""

    # Create test data with only expired contracts
    today = datetime.today()
    past_date = today - timedelta(days=1)

    contracts = [
        {"conid": "123456", "expirationDate": past_date.strftime("%Y%m%d")}
    ]

    # Verify the function raises ValueError
    with pytest.raises(ValueError, match="No valid .* contracts available"):
        get_closest_contract(contracts)


def test_get_closest_contract_empty_list():
    """Test that get_closest_contract raises ValueError when given an empty list."""

    # Verify the function raises ValueError when given an empty list
    with pytest.raises(ValueError, match="No valid .* contracts available"):
        get_closest_contract([])


def test_get_closest_contract_custom_min_days():
    """Test that get_closest_contract respects custom min_days_until_expiry."""

    # Create test data with contracts at different future dates
    today = datetime.today()
    future_date1 = today + timedelta(days=10)  # This would be valid with default MIN_DAYS_UNTIL_EXPIRY
    future_date2 = today + timedelta(days=30)

    contracts = [
        {"conid": "123456", "expirationDate": future_date1.strftime("%Y%m%d")},
        {"conid": "789012", "expirationDate": future_date2.strftime("%Y%m%d")}
    ]

    # Call the function with a custom min_days_until_expiry that makes the first contract invalid
    result = get_closest_contract(contracts, min_days_until_expiry=15)

    # Verify the result is the contract with the later expiration date
    assert result["conid"] == "789012"
