from datetime import datetime, timedelta
from unittest.mock import patch, Mock

import pytest

from app.utils.ibkr_utils.contracts_utils import parse_symbol, fetch_contract, get_closest_contract
from config import BASE_URL


def test_parse_symbol_valid():
    # Setup
    symbol = "ESM2024"

    # Execute & Assert
    assert parse_symbol(symbol) == "ESM"


def test_parse_symbol_invalid():
    # Setup
    symbol = "1234"

    # Execute & Assert
    with pytest.raises(ValueError, match="Invalid symbol format"):
        parse_symbol(symbol)


@patch('app.utils.ibkr_utils.contracts_utils.api_get')
def test_fetch_contract(mock_api_get):
    # Setup
    parsed_symbol = "ESM"
    mock_response = Mock()
    mock_response.json.return_value = {
        "ESM": [
            {"symbol": "ESM2024", "expirationDate": "20240621"},
            {"symbol": "ESM2023", "expirationDate": "20230616"},
        ]
    }
    mock_api_get.return_value = mock_response

    # Execute
    contracts = fetch_contract("ESM2024")

    # Assert
    mock_api_get.assert_called_once_with(f"{BASE_URL}/trsrv/futures?symbols={parsed_symbol}")
    assert len(contracts) == 2
    assert contracts[0]['symbol'] == "ESM2024"


def test_get_closest_contract_success():
    # Setup
    today = datetime.today()
    contracts = [
        {"symbol": "ESM2024", "expirationDate": (today + timedelta(days=60)).strftime("%Y%m%d")},
        {"symbol": "ESU2024", "expirationDate": (today + timedelta(days=150)).strftime("%Y%m%d")},
        {"symbol": "ESH2024", "expirationDate": (today + timedelta(days=10)).strftime("%Y%m%d")},
    ]

    # Execute
    selected_contract = get_closest_contract(contracts, min_days_until_expiry=30)

    # Assert
    assert selected_contract["symbol"] == "ESM2024"


def test_get_closest_contract_no_valid_contract():
    # Setup
    today = datetime.today()
    contracts = [
        {"symbol": "ESM2024", "expirationDate": (today + timedelta(days=5)).strftime("%Y%m%d")},
        {"symbol": "ESH2024", "expirationDate": (today + timedelta(days=6)).strftime("%Y%m%d")},
    ]

    # Execute & Assert
    with pytest.raises(ValueError, match="No valid \\(liquid, distant enough\\) contracts available"):
        get_closest_contract(contracts, min_days_until_expiry=30)
