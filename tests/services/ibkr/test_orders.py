from unittest.mock import MagicMock

import pytest

from app.services.ibkr.orders import place_order
from config import QUANTITY_TO_TRADE


@pytest.fixture
def mock_get_contract_position(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("app.services.ibkr.orders.get_contract_position", mock)
    return mock


@pytest.fixture
def mock_api_post(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("app.services.ibkr.orders.api_post", mock)
    return mock


@pytest.fixture
def mock_logger(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("app.services.ibkr.orders.logger", mock)
    return mock


@pytest.fixture
def mock_suppress_messages(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("app.services.ibkr.orders.suppress_messages", mock)
    return mock


@pytest.fixture(autouse=True)
def reset_aggressive_trading(monkeypatch):
    # Reset AGGRESSIVE_TRADING to False unless individually overridden
    monkeypatch.setattr("app.services.ibkr.orders.AGGRESSIVE_TRADING", False)


def test_place_order_new_buy_position(mock_get_contract_position, mock_api_post):
    mock_get_contract_position.return_value = 0  # No existing position
    mock_api_post.return_value = {"id": "123456"}

    result = place_order("123456", "B")
    assert result == {"id": "123456"}
    mock_get_contract_position.assert_called_once_with("123456")
    mock_api_post.assert_called_once()
    order_details = mock_api_post.call_args[0][1]
    assert order_details["orders"][0]["side"] == "BUY"
    assert order_details["orders"][0]["quantity"] == QUANTITY_TO_TRADE


def test_place_order_new_sell_position(mock_get_contract_position, mock_api_post):
    mock_get_contract_position.return_value = 0
    mock_api_post.return_value = {"id": "123456"}

    result = place_order("123456", "S")
    assert result == {"id": "123456"}
    mock_get_contract_position.assert_called_once_with("123456")
    mock_api_post.assert_called_once()
    order_details = mock_api_post.call_args[0][1]
    assert order_details["orders"][0]["side"] == "SELL"
    assert order_details["orders"][0]["quantity"] == QUANTITY_TO_TRADE


def test_place_order_existing_same_position(mock_get_contract_position):
    # existing long, trying to buy more
    mock_get_contract_position.return_value = 1

    result = place_order("123456", "B")
    assert result == {"success": True, "message": "No action needed: already in desired position"}
    mock_get_contract_position.assert_called_once_with("123456")

    mock_get_contract_position.reset_mock()
    mock_get_contract_position.return_value = -1

    result = place_order("123456", "S")
    assert result == {"success": True, "message": "No action needed: already in desired position"}
    mock_get_contract_position.assert_called_once_with("123456")


def test_place_order_reverse_position_aggressive(
        monkeypatch, mock_get_contract_position, mock_api_post
):
    monkeypatch.setattr("app.services.ibkr.orders.AGGRESSIVE_TRADING", True)
    mock_get_contract_position.return_value = 1  # existing long
    mock_api_post.return_value = {"id": "123456"}

    result = place_order("123456", "S")
    assert result == {"id": "123456"}
    mock_get_contract_position.assert_called_once_with("123456")
    mock_api_post.assert_called_once()
    order_details = mock_api_post.call_args[0][1]
    assert order_details["orders"][0]["side"] == "SELL"
    assert order_details["orders"][0]["quantity"] == QUANTITY_TO_TRADE * 2


def test_place_order_reverse_position_not_aggressive(
        mock_get_contract_position, mock_api_post
):
    mock_get_contract_position.return_value = -1  # existing short
    mock_api_post.return_value = {"id": "123456"}

    result = place_order("123456", "B")
    assert result == {"id": "123456"}
    mock_get_contract_position.assert_called_once_with("123456")
    mock_api_post.assert_called_once()
    order_details = mock_api_post.call_args[0][1]
    assert order_details["orders"][0]["side"] == "BUY"
    assert order_details["orders"][0]["quantity"] == QUANTITY_TO_TRADE


def test_place_order_with_message_suppression(
        mock_get_contract_position, mock_api_post, mock_suppress_messages
):
    mock_get_contract_position.return_value = 0
    # First call returns message IDs, second call returns success
    mock_api_post.side_effect = [
        [{"messageIds": ["1", "2"]}],
        {"id": "123456"},
    ]

    result = place_order("123456", "B")
    assert result == {"id": "123456"}
    mock_get_contract_position.assert_called_once_with("123456")
    assert mock_api_post.call_count == 2
    mock_suppress_messages.assert_called_once_with(["1", "2"])


def test_place_order_insufficient_funds_error(
        mock_logger, mock_api_post, mock_get_contract_position
):
    mock_get_contract_position.return_value = 0
    mock_api_post.return_value = {"error": "available funds are insufficient"}

    result = place_order("123456", "B")
    assert result["success"] is False
    assert result["error"] == "Insufficient funds"
    mock_get_contract_position.assert_called_once_with("123456")
    mock_api_post.assert_called_once()
    mock_logger.error.assert_called_once()


def test_place_order_derivative_rules_error(
        mock_logger, mock_api_post, mock_get_contract_position
):
    mock_get_contract_position.return_value = 0
    mock_api_post.return_value = {"error": "does not comply with our order handling rules for derivatives"}

    result = place_order("123456", "B")
    assert result["success"] is False
    assert result["error"] == "Non-compliance with derivative rules"
    mock_get_contract_position.assert_called_once_with("123456")
    mock_api_post.assert_called_once()
    mock_logger.error.assert_called_once()


def test_place_order_unhandled_error(
        mock_logger, mock_api_post, mock_get_contract_position
):
    mock_get_contract_position.return_value = 0
    mock_api_post.return_value = {"error": "some other error"}

    result = place_order("123456", "B")
    assert result["success"] is False
    assert result["error"] == "Unhandled error"
    mock_get_contract_position.assert_called_once_with("123456")
    mock_api_post.assert_called_once()
    mock_logger.error.assert_called_once()


def test_place_order_unexpected_exception(
        mock_logger, mock_api_post, mock_get_contract_position
):
    mock_get_contract_position.return_value = 0
    mock_api_post.side_effect = Exception("Test error")

    result = place_order("123456", "B")
    assert result["success"] is False
    assert result["error"] == "An unexpected error occurred"
    mock_get_contract_position.assert_called_once_with("123456")
    mock_api_post.assert_called_once()
    mock_logger.exception.assert_called_once()
