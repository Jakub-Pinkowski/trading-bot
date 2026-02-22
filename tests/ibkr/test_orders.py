from unittest.mock import patch

import pytest

from app.ibkr.orders import (place_order, invalidate_cache, get_contract_position,
                             suppress_messages, QUANTITY_TO_TRADE, MAX_SUPPRESS_RETRIES)
from config import ACCOUNT_ID


# ==================== invalidate_cache Tests ====================

@patch('app.ibkr.orders.api_post')
def test_invalidate_cache_success(mock_api_post):
    """Test that invalidate_cache successfully calls the API."""

    # Call the function
    invalidate_cache()

    # Verify the API was called with the correct endpoint and data
    mock_api_post.assert_called_once_with(f"portfolio/{ACCOUNT_ID}/positions/invalidate", {})


@patch('app.ibkr.orders.api_post')
def test_invalidate_cache_error(mock_api_post):
    """Test that invalidate_cache logs and propagates API errors."""

    # Setup mock to raise an exception
    mock_api_post.side_effect = Exception("API error")

    # Verify the exception propagates so callers are not left with stale data
    with pytest.raises(Exception, match="API error"):
        invalidate_cache()

    mock_api_post.assert_called_once()


# ==================== get_contract_position Tests ====================

@patch('app.ibkr.orders.invalidate_cache')
@patch('app.ibkr.orders.api_get')
def test_get_contract_position_found(mock_api_get, mock_invalidate_cache):
    """Test that get_contract_position returns the correct position when found."""

    # Setup mocks
    mock_api_get.return_value = [
        {"conid": "123456", "position": 5},
        {"conid": "789012", "position": -3}
    ]

    # Call the function for a contract that exists
    result = get_contract_position("123456")

    # Verify the result and that the cache was invalidated
    assert result == 5
    mock_invalidate_cache.assert_called_once()
    mock_api_get.assert_called_once_with(f"portfolio/{ACCOUNT_ID}/positions")


@patch('app.ibkr.orders.invalidate_cache')
@patch('app.ibkr.orders.api_get')
def test_get_contract_position_not_found(mock_api_get, mock_invalidate_cache):
    """Test that get_contract_position returns 0 when the contract is not found."""

    # Setup mocks
    mock_api_get.return_value = [
        {"conid": "789012", "position": -3}
    ]

    # Call the function for a contract that doesn't exist
    result = get_contract_position("123456")

    # Verify the result is 0 and that the cache was invalidated
    assert result == 0
    mock_invalidate_cache.assert_called_once()
    mock_api_get.assert_called_once_with(f"portfolio/{ACCOUNT_ID}/positions")


@patch('app.ibkr.orders.invalidate_cache')
@patch('app.ibkr.orders.api_get')
def test_get_contract_position_api_error(mock_api_get, mock_invalidate_cache):
    """Test that get_contract_position handles API errors gracefully."""

    # Setup mock to raise an exception
    mock_api_get.side_effect = Exception("API error")

    # Call the function
    result = get_contract_position("123456")

    # Verify the result is 0 and that the cache was invalidated
    assert result == 0
    mock_invalidate_cache.assert_called_once()
    mock_api_get.assert_called_once_with(f"portfolio/{ACCOUNT_ID}/positions")


@patch('app.ibkr.orders.invalidate_cache')
def test_get_contract_position_invalidate_cache_error(mock_invalidate_cache):
    """Test that get_contract_position propagates invalidate_cache errors."""

    # Setup mock to raise an exception
    mock_invalidate_cache.side_effect = Exception("Cache invalidation error")

    # Verify the exception propagates out of get_contract_position
    with pytest.raises(Exception, match="Cache invalidation error"):
        get_contract_position("123456")


# ==================== suppress_messages Tests ====================

@patch('app.ibkr.orders.api_post')
def test_suppress_messages_success(mock_api_post):
    """Test that suppress_messages successfully calls the API."""

    # Setup mock
    mock_api_post.return_value = {"success": True}

    # Call the function
    suppress_messages(["1", "2"])

    # Verify the API was called with the correct endpoint and data
    mock_api_post.assert_called_once_with("iserver/questions/suppress", {"messageIds": ["1", "2"]})


@patch('app.ibkr.orders.api_post')
def test_suppress_messages_error(mock_api_post):
    """Test that suppress_messages handles errors gracefully."""
    # Setup mock to raise an exception
    mock_api_post.side_effect = Exception("API error")

    # Call the function (should not raise an exception)
    suppress_messages(["1", "2"])

    # Verify the API was called
    mock_api_post.assert_called_once()


# ==================== place_order Tests ====================

def test_place_order_new_buy_position(mock_get_contract_position, mock_api_post_orders):
    """Test that place_order creates a new buy position when no position exists"""

    # Mock contract position to return 0 (no existing position) and configure API response
    mock_get_contract_position.return_value = 0  # No existing position
    mock_api_post_orders.return_value = {"id": "123456"}

    # Call place_order with a contract ID and buy side indicator
    result = place_order("123456", "B")

    # Verify the function returns the API response, checks position, places a BUY order with correct quantity
    assert result == {"id": "123456"}
    mock_get_contract_position.assert_called_once_with("123456")
    mock_api_post_orders.assert_called_once()
    order_details = mock_api_post_orders.call_args[0][1]
    assert order_details["orders"][0]["side"] == "BUY"
    assert order_details["orders"][0]["quantity"] == QUANTITY_TO_TRADE


def test_place_order_new_sell_position(mock_get_contract_position, mock_api_post_orders):
    """Test that place_order creates a new sell position when no position exists"""

    # Mock contract position to return 0 (no existing position) and configure API response
    mock_get_contract_position.return_value = 0
    mock_api_post_orders.return_value = {"id": "123456"}

    # Call place_order with a contract ID and sell side indicator
    result = place_order("123456", "S")

    # Verify the function returns the API response, checks position, places a SELL order with correct quantity
    assert result == {"id": "123456"}
    mock_get_contract_position.assert_called_once_with("123456")
    mock_api_post_orders.assert_called_once()
    order_details = mock_api_post_orders.call_args[0][1]
    assert order_details["orders"][0]["side"] == "SELL"
    assert order_details["orders"][0]["quantity"] == QUANTITY_TO_TRADE


def test_place_order_existing_same_position(mock_get_contract_position):
    """Test that place_order takes no action when the desired position already exists"""

    # Test case 1: Mock existing long position (1) and attempt to buy more
    mock_get_contract_position.return_value = 1

    # Call place_order with buy side when already in a long position
    result = place_order("123456", "B")

    # Verify function returns success message without placing an order since position already exists
    assert result == {"success": True, "message": "No action needed: already in desired position"}
    mock_get_contract_position.assert_called_once_with("123456")

    # Test case 2: Mock existing short position (-1) and attempt to sell more
    mock_get_contract_position.reset_mock()
    mock_get_contract_position.return_value = -1

    # Call place_order with sell side when already in a short position
    result = place_order("123456", "S")

    # Verify function returns success message without placing an order since position already exists
    assert result == {"success": True, "message": "No action needed: already in desired position"}
    mock_get_contract_position.assert_called_once_with("123456")


def test_place_order_invalid_side(mock_get_contract_position):
    """Test that place_order raises ValueError when an invalid side is provided"""

    # Mock no existing position so validation is reached
    mock_get_contract_position.return_value = 0

    # Verify that an invalid side raises ValueError with a clear message
    with pytest.raises(ValueError, match="Invalid side 'X': expected 'B' or 'S'"):
        place_order("123456", "X")


def test_place_order_reverse_position(
    mock_get_contract_position, mock_api_post_orders
):
    """Test that place_order reverses an existing position with standard quantity"""

    # Mock existing short position and API response
    mock_get_contract_position.return_value = -1  # existing short
    mock_api_post_orders.return_value = {"id": "123456"}

    # Call place_order with buy side to reverse an existing short position
    result = place_order("123456", "B")

    # Verify function returns API response and places a BUY order with standard quantity
    assert result == {"id": "123456"}
    mock_get_contract_position.assert_called_once_with("123456")
    mock_api_post_orders.assert_called_once()
    order_details = mock_api_post_orders.call_args[0][1]
    assert order_details["orders"][0]["side"] == "BUY"
    assert order_details["orders"][0]["quantity"] == QUANTITY_TO_TRADE


def test_place_order_with_message_suppression(
    mock_get_contract_position, mock_api_post_orders, mock_suppress_messages
):
    """Test that place_order handles message suppression before completing an order"""

    # Reset mock, set no existing position, and configure API to first return message IDs then success
    mock_get_contract_position.reset_mock()
    mock_get_contract_position.return_value = 0
    # First call returns message IDs, second call returns success
    mock_api_post_orders.side_effect = [
        [{"messageIds": ["1", "2"]}],
        {"id": "123456"},
    ]

    # Call place_order which should handle message suppression before completing the order
    result = place_order("123456", "B")

    # Verify function returns final API response, checks position, calls API twice, and suppresses messages
    assert result == {"id": "123456"}
    mock_get_contract_position.assert_called_once_with("123456")
    assert mock_api_post_orders.call_count == 2
    mock_suppress_messages.assert_called_once_with(["1", "2"])


def test_place_order_exceeded_suppression_retries(
    mock_logger_orders, mock_get_contract_position, mock_api_post_orders, mock_suppress_messages
):
    """Test that place_order returns an error after exceeding the maximum suppression retries"""

    # Mock no existing position and API always returning message IDs, never clearing them
    mock_get_contract_position.return_value = 0
    mock_api_post_orders.return_value = [{"messageIds": ["1", "2"]}]

    # Call place_order which should give up after MAX_SUPPRESS_RETRIES attempts
    result = place_order("123456", "B")

    # Verify function returns error after exhausting retries and suppressed messages each time
    assert result == {"success": False, "error": "Exceeded maximum suppression retries"}
    assert mock_api_post_orders.call_count == MAX_SUPPRESS_RETRIES
    assert mock_suppress_messages.call_count == MAX_SUPPRESS_RETRIES
    mock_logger_orders.error.assert_called_once()


def test_place_order_insufficient_funds_error(
    mock_logger_orders, mock_api_post_orders, mock_get_contract_position
):
    """Test that place_order handles and logs insufficient funds errors"""

    # Mock no existing position and API returning an insufficient funds error
    mock_get_contract_position.return_value = 0
    mock_api_post_orders.return_value = {"error": "available funds are insufficient"}

    # Call place_order which should handle the insufficient funds error
    result = place_order("123456", "B")

    # Verify function returns error response with appropriate message, logs the error, and doesn't place order
    assert result["success"] is False
    assert result["error"] == "Insufficient funds"
    mock_get_contract_position.assert_called_once_with("123456")
    mock_api_post_orders.assert_called_once()
    mock_logger_orders.error.assert_called_once()


def test_place_order_derivative_rules_error(
    mock_logger_orders, mock_api_post_orders, mock_get_contract_position
):
    """Test that place_order handles and logs derivative rules compliance errors"""

    # Mock no existing position and API returning a derivative rules compliance error
    mock_get_contract_position.return_value = 0
    mock_api_post_orders.return_value = {"error": "does not comply with our order handling rules for derivatives"}

    # Call place_order which should handle the derivative rules error
    result = place_order("123456", "B")

    # Verify function returns error with appropriate message, logs the error, and doesn't complete the order
    assert result["success"] is False
    assert result["error"] == "Non-compliance with derivative rules"
    mock_get_contract_position.assert_called_once_with("123456")
    mock_api_post_orders.assert_called_once()
    mock_logger_orders.error.assert_called_once()


def test_place_order_unhandled_error(
    mock_logger_orders, mock_api_post_orders, mock_get_contract_position
):
    """Test that place_order handles and logs unrecognized error types"""

    # Mock no existing position and API returning an unrecognized error type
    mock_get_contract_position.return_value = 0
    mock_api_post_orders.return_value = {"error": "some other error"}

    # Call place_order which should handle unknown error types with a generic response
    result = place_order("123456", "B")

    # Verify function returns generic error message, logs the error, and doesn't complete the order
    assert result["success"] is False
    assert result["error"] == "Unhandled error"
    mock_get_contract_position.assert_called_once_with("123456")
    mock_api_post_orders.assert_called_once()
    mock_logger_orders.error.assert_called_once()


def test_place_order_unexpected_exception(
    mock_logger_orders, mock_api_post_orders, mock_get_contract_position
):
    """Test that place_order catches and logs unexpected exceptions"""

    # Mock no existing position and API raising an unexpected exception
    mock_get_contract_position.return_value = 0
    mock_api_post_orders.side_effect = Exception("Test error")

    # Call place_order which should catch and handle any unexpected exceptions
    result = place_order("123456", "B")

    # Verify function returns generic error message, logs the exception, and gracefully handles the failure
    assert result["success"] is False
    assert result["error"] == "An unexpected error occurred"
    mock_get_contract_position.assert_called_once_with("123456")
    mock_api_post_orders.assert_called_once()
    mock_logger_orders.exception.assert_called_once()
