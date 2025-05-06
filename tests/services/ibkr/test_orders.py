from app.services.ibkr.orders import place_order
from config import QUANTITY_TO_TRADE


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


def test_place_order_reverse_position_aggressive(
        monkeypatch, mock_get_contract_position, mock_api_post_orders
):
    """Test that place_order reverses an existing position with double quantity in aggressive mode"""

    # Enable aggressive trading mode and mock existing long position with API response
    monkeypatch.setattr("app.services.ibkr.orders.AGGRESSIVE_TRADING", True)
    mock_get_contract_position.return_value = 1  # existing long
    mock_api_post_orders.return_value = {"id": "123456"}

    # Call place_order with sell side to reverse an existing long position in aggressive mode
    result = place_order("123456", "S")

    # Verify function returns API response and places a SELL order with double quantity to close and reverse
    assert result == {"id": "123456"}
    mock_get_contract_position.assert_called_once_with("123456")
    mock_api_post_orders.assert_called_once()
    order_details = mock_api_post_orders.call_args[0][1]
    assert order_details["orders"][0]["side"] == "SELL"
    assert order_details["orders"][0]["quantity"] == QUANTITY_TO_TRADE * 2


def test_place_order_reverse_position_not_aggressive(
        mock_get_contract_position, mock_api_post_orders
):
    """Test that place_order reverses an existing position with standard quantity in non-aggressive mode"""

    # Mock existing short position and API response (with default non-aggressive mode)
    mock_get_contract_position.return_value = -1  # existing short
    mock_api_post_orders.return_value = {"id": "123456"}

    # Call place_order with buy side to reverse an existing short position in non-aggressive mode
    result = place_order("123456", "B")

    # Verify function returns API response and places a BUY order with standard quantity (not doubled)
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
