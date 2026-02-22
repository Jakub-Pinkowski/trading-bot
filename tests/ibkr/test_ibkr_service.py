import pytest

from app.ibkr.ibkr_service import process_trading_data


def test_process_trading_data_normal(mock_logger_ibkr_service, mock_place_order, mock_get_contract_id):
    """Test that process_trading_data correctly processes normal trading data and places an order"""

    # Configure mocks to return expected values and create normal trading data with "dummy" set to "NO"
    mock_get_contract_id.return_value = "123456"
    mock_place_order.return_value = {"id": "order123"}
    trading_data = {
        "dummy": "NO",
        "symbol": "ZC",
        "side": "B",
        "price": "4500.00"
    }

    # Call process_trading_data with the prepared trading data
    result = process_trading_data(trading_data)

    # Verify logger recorded the trading data, contract ID was retrieved, order was placed, and success was logged
    mock_logger_ibkr_service.info.assert_any_call(f"Trading data received: {trading_data}")
    mock_get_contract_id.assert_called_once_with("ZC")
    mock_place_order.assert_called_once_with("123456", "B")
    mock_logger_ibkr_service.info.assert_any_call(f"Order placed: {{'id': 'order123'}}")
    assert result == {'status': 'order_placed', 'order': {"id": "order123"}}


def test_process_trading_data_dummy(mock_logger_ibkr_service, mock_place_order, mock_get_contract_id):
    """Test that process_trading_data skips the API call and order when in dummy mode"""

    # Create trading data with "dummy" set to "YES" to test dummy mode
    trading_data = {
        "dummy": "YES",
        "symbol": "ZC",
        "side": "B",
        "price": "4500.00"
    }

    # Call process_trading_data with dummy trading data
    result = process_trading_data(trading_data)

    # Verify logger recorded the trading data, no API call was made, and no order was placed
    mock_logger_ibkr_service.info.assert_called_once_with(f"Trading data received: {trading_data}")
    mock_get_contract_id.assert_not_called()
    mock_place_order.assert_not_called()
    assert result == {'status': 'dummy_skip'}


def test_process_trading_data_sell(mock_logger_ibkr_service, mock_place_order, mock_get_contract_id):
    """Test that process_trading_data correctly processes sell orders"""

    # Configure mocks and create trading data with "side" set to "S" (sell) to test sell orders
    mock_get_contract_id.return_value = "123456"
    mock_place_order.return_value = {"id": "order123"}
    trading_data = {
        "dummy": "NO",
        "symbol": "ZC",
        "side": "S",
        "price": "4500.00"
    }

    # Call process_trading_data with sell order trading data
    result = process_trading_data(trading_data)

    # Verify logger recorded the trading data, contract ID was retrieved, sell order was placed, and success was logged
    mock_logger_ibkr_service.info.assert_any_call(f"Trading data received: {trading_data}")
    mock_get_contract_id.assert_called_once_with("ZC")
    mock_place_order.assert_called_once_with("123456", "S")
    mock_logger_ibkr_service.info.assert_any_call(f"Order placed: {{'id': 'order123'}}")
    assert result == {'status': 'order_placed', 'order': {"id": "order123"}}


def test_process_trading_data_missing_dummy_field(mock_logger_ibkr_service, mock_place_order, mock_get_contract_id):
    """Test that process_trading_data places an order when the dummy field is absent"""

    # Configure mock and create trading data with missing "dummy" field to test default behavior
    mock_get_contract_id.return_value = "123456"
    mock_place_order.return_value = {"id": "order123"}
    trading_data = {
        "symbol": "ZC",
        "side": "B",
        "price": "4500.00"
    }

    # Call process_trading_data with incomplete trading data
    result = process_trading_data(trading_data)

    # Verify logger recorded the trading data, contract ID was retrieved, and order was placed despite missing dummy field
    mock_logger_ibkr_service.info.assert_any_call(f"Trading data received: {trading_data}")
    mock_get_contract_id.assert_called_once_with("ZC")
    mock_place_order.assert_called_once_with("123456", "B")
    assert result == {'status': 'order_placed', 'order': {"id": "order123"}}


def test_process_trading_data_missing_symbol(mock_logger_ibkr_service, mock_place_order, mock_get_contract_id):
    """Test that process_trading_data raises ValueError when symbol is missing"""

    # Create trading data without a symbol field
    trading_data = {
        "dummy": "NO",
        "side": "B",
        "price": "4500.00"
    }

    # Verify that a missing symbol raises ValueError before any API call is made
    with pytest.raises(ValueError, match="Missing required field: symbol"):
        process_trading_data(trading_data)

    mock_get_contract_id.assert_not_called()
    mock_place_order.assert_not_called()


def test_process_trading_data_missing_side(mock_logger_ibkr_service, mock_place_order, mock_get_contract_id):
    """Test that process_trading_data raises ValueError when side is missing"""

    # Create trading data without a side field
    trading_data = {
        "dummy": "NO",
        "symbol": "ZC",
        "price": "4500.00"
    }

    # Verify that a missing side raises ValueError before any API call is made
    with pytest.raises(ValueError, match="Missing required field: side"):
        process_trading_data(trading_data)

    mock_get_contract_id.assert_not_called()
    mock_place_order.assert_not_called()


def test_process_trading_data_order_failed(mock_logger_ibkr_service, mock_place_order, mock_get_contract_id):
    """Test that process_trading_data returns order_failed status when place_order reports failure"""

    # Configure place_order to return a failure response (e.g. insufficient funds)
    mock_get_contract_id.return_value = "123456"
    mock_place_order.return_value = {"success": False, "error": "Insufficient funds", "details": {}}
    trading_data = {
        "dummy": "NO",
        "symbol": "ZC",
        "side": "B",
        "price": "4500.00"
    }

    # Call process_trading_data and verify it reports failure rather than success
    result = process_trading_data(trading_data)

    # Verify the error is logged and the status reflects the failure
    assert result == {'status': 'order_failed',
                      'order': {"success": False, "error": "Insufficient funds", "details": {}}}
    mock_logger_ibkr_service.error.assert_called_once()
    mock_logger_ibkr_service.info.assert_called_once_with(f"Trading data received: {trading_data}")


def test_process_trading_data_error_handling(mock_logger_ibkr_service, mock_place_order, mock_get_contract_id):
    """Test that process_trading_data propagates exceptions from get_contract_id"""

    # Configure mock to raise an exception when get_contract_id is called and create normal trading data
    mock_get_contract_id.side_effect = ValueError("Test error")
    trading_data = {
        "dummy": "NO",
        "symbol": "ZC",
        "side": "B",
        "price": "4500.00"
    }

    # Verify that the ValueError from get_contract_id propagates through process_trading_data
    with pytest.raises(ValueError, match="Test error"):
        process_trading_data(trading_data)

    # Verify logger recorded the trading data, get_contract_id was called but failed, and place_order was never called
    mock_logger_ibkr_service.info.assert_called_once_with(f"Trading data received: {trading_data}")
    mock_get_contract_id.assert_called_once_with("ZC")
    mock_place_order.assert_not_called()
