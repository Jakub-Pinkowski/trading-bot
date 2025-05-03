from unittest.mock import patch

from app.services.ibkr.orders import place_order
from config import QUANTITY_TO_TRADE


@patch('app.services.ibkr.orders.get_contract_position')
@patch('app.services.ibkr.orders.api_post')
def test_place_order_new_buy_position(mock_api_post, mock_get_contract_position):
    # Setup
    mock_get_contract_position.return_value = 0  # No existing position
    mock_api_post.return_value = {"id": "123456"}  # Successful order response

    # Execute
    result = place_order("123456", "B")

    # Assert
    assert result == {"id": "123456"}
    mock_get_contract_position.assert_called_once_with("123456")
    mock_api_post.assert_called_once()
    # Verify the order details
    order_details = mock_api_post.call_args[0][1]
    assert order_details["orders"][0]["side"] == "BUY"
    assert order_details["orders"][0]["quantity"] == QUANTITY_TO_TRADE


@patch('app.services.ibkr.orders.get_contract_position')
@patch('app.services.ibkr.orders.api_post')
def test_place_order_new_sell_position(mock_api_post, mock_get_contract_position):
    # Setup
    mock_get_contract_position.return_value = 0  # No existing position
    mock_api_post.return_value = {"id": "123456"}  # Successful order response

    # Execute
    result = place_order("123456", "S")

    # Assert
    assert result == {"id": "123456"}
    mock_get_contract_position.assert_called_once_with("123456")
    mock_api_post.assert_called_once()
    # Verify the order details
    order_details = mock_api_post.call_args[0][1]
    assert order_details["orders"][0]["side"] == "SELL"
    assert order_details["orders"][0]["quantity"] == QUANTITY_TO_TRADE


@patch('app.services.ibkr.orders.get_contract_position')
def test_place_order_existing_same_position(mock_get_contract_position):
    # Setup - existing long position, trying to buy more
    mock_get_contract_position.return_value = 1

    # Execute
    result = place_order("123456", "B")

    # Assert
    assert result == {"success": True, "message": "No action needed: already in desired position"}
    mock_get_contract_position.assert_called_once_with("123456")

    # Reset mock for second test
    mock_get_contract_position.reset_mock()

    # Setup - existing short position, trying to sell more
    mock_get_contract_position.return_value = -1

    # Execute
    result = place_order("123456", "S")

    # Assert
    assert result == {"success": True, "message": "No action needed: already in desired position"}
    mock_get_contract_position.assert_called_once_with("123456")


@patch('app.services.ibkr.orders.AGGRESSIVE_TRADING', True)
@patch('app.services.ibkr.orders.get_contract_position')
@patch('app.services.ibkr.orders.api_post')
def test_place_order_reverse_position_aggressive(mock_api_post, mock_get_contract_position):
    # Setup - existing long position, trying to sell
    mock_get_contract_position.return_value = 1
    mock_api_post.return_value = {"id": "123456"}

    # Execute
    result = place_order("123456", "S")

    # Assert
    assert result == {"id": "123456"}
    mock_get_contract_position.assert_called_once_with("123456")
    mock_api_post.assert_called_once()
    # Verify the order details - should be double quantity for aggressive trading
    order_details = mock_api_post.call_args[0][1]
    assert order_details["orders"][0]["side"] == "SELL"
    assert order_details["orders"][0]["quantity"] == QUANTITY_TO_TRADE * 2


@patch('app.services.ibkr.orders.AGGRESSIVE_TRADING', False)
@patch('app.services.ibkr.orders.get_contract_position')
@patch('app.services.ibkr.orders.api_post')
def test_place_order_reverse_position_not_aggressive(mock_api_post, mock_get_contract_position):
    # Setup - existing short position, trying to buy
    mock_get_contract_position.return_value = -1
    mock_api_post.return_value = {"id": "123456"}

    # Execute
    result = place_order("123456", "B")

    # Assert
    assert result == {"id": "123456"}
    mock_get_contract_position.assert_called_once_with("123456")
    mock_api_post.assert_called_once()
    # Verify the order details - should be normal quantity for non-aggressive trading
    order_details = mock_api_post.call_args[0][1]
    assert order_details["orders"][0]["side"] == "BUY"
    assert order_details["orders"][0]["quantity"] == QUANTITY_TO_TRADE


@patch('app.services.ibkr.orders.get_contract_position')
@patch('app.services.ibkr.orders.api_post')
@patch('app.services.ibkr.orders.suppress_messages')
def test_place_order_with_message_suppression(mock_suppress_messages, mock_api_post, mock_get_contract_position):
    # Setup
    mock_get_contract_position.return_value = 0
    # First call returns message IDs, second call returns success
    mock_api_post.side_effect = [
        [{"messageIds": ["1", "2"]}],
        {"id": "123456"}
    ]

    # Execute
    result = place_order("123456", "B")

    # Assert
    assert result == {"id": "123456"}
    mock_get_contract_position.assert_called_once_with("123456")
    assert mock_api_post.call_count == 2
    mock_suppress_messages.assert_called_once_with(["1", "2"])


@patch('app.services.ibkr.orders.get_contract_position')
@patch('app.services.ibkr.orders.api_post')
@patch('app.services.ibkr.orders.logger')
def test_place_order_insufficient_funds_error(mock_logger, mock_api_post, mock_get_contract_position):
    # Setup
    mock_get_contract_position.return_value = 0
    mock_api_post.return_value = {"error": "available funds are insufficient"}

    # Execute
    result = place_order("123456", "B")

    # Assert
    assert result["success"] == False
    assert result["error"] == "Insufficient funds"
    mock_get_contract_position.assert_called_once_with("123456")
    mock_api_post.assert_called_once()
    mock_logger.error.assert_called_once()


@patch('app.services.ibkr.orders.get_contract_position')
@patch('app.services.ibkr.orders.api_post')
@patch('app.services.ibkr.orders.logger')
def test_place_order_derivative_rules_error(mock_logger, mock_api_post, mock_get_contract_position):
    # Setup
    mock_get_contract_position.return_value = 0
    mock_api_post.return_value = {"error": "does not comply with our order handling rules for derivatives"}

    # Execute
    result = place_order("123456", "B")

    # Assert
    assert result["success"] == False
    assert result["error"] == "Non-compliance with derivative rules"
    mock_get_contract_position.assert_called_once_with("123456")
    mock_api_post.assert_called_once()
    mock_logger.error.assert_called_once()


@patch('app.services.ibkr.orders.get_contract_position')
@patch('app.services.ibkr.orders.api_post')
@patch('app.services.ibkr.orders.logger')
def test_place_order_unhandled_error(mock_logger, mock_api_post, mock_get_contract_position):
    # Setup
    mock_get_contract_position.return_value = 0
    mock_api_post.return_value = {"error": "some other error"}

    # Execute
    result = place_order("123456", "B")

    # Assert
    assert result["success"] == False
    assert result["error"] == "Unhandled error"
    mock_get_contract_position.assert_called_once_with("123456")
    mock_api_post.assert_called_once()
    mock_logger.error.assert_called_once()


@patch('app.services.ibkr.orders.get_contract_position')
@patch('app.services.ibkr.orders.api_post')
@patch('app.services.ibkr.orders.logger')
def test_place_order_unexpected_exception(mock_logger, mock_api_post, mock_get_contract_position):
    # Setup
    mock_get_contract_position.return_value = 0
    mock_api_post.side_effect = Exception("Test error")

    # Execute
    result = place_order("123456", "B")

    # Assert
    assert result["success"] == False
    assert result["error"] == "An unexpected error occurred"
    mock_get_contract_position.assert_called_once_with("123456")
    mock_api_post.assert_called_once()
    mock_logger.exception.assert_called_once()
