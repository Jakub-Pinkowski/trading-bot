import pytest

from app.services.ibkr_service import process_trading_data


def test_process_trading_data_normal(mock_logger_ibkr_service, mock_place_order, mock_get_contract_id):
    """Test that process_trading_data correctly processes normal trading data and places an order"""

    # Configure mocks to return expected values and create normal trading data with "dummy" set to "NO"
    mock_get_contract_id.return_value = "123456"
    mock_place_order.return_value = {"id": "order123"}
    trading_data = {
        "dummy": "NO",
        "symbol": "ES",
        "side": "B",
        "price": "4500.00"
    }

    # Call process_trading_data with the prepared trading data
    process_trading_data(trading_data)

    # Verify logger recorded the trading data, contract ID was retrieved, order was placed, and success was logged
    mock_logger_ibkr_service.info.assert_any_call(f"Trading data received: {trading_data}")
    mock_get_contract_id.assert_called_once_with("ES")
    mock_place_order.assert_called_once_with("123456", "B")
    mock_logger_ibkr_service.info.assert_any_call(f"Order placed: {{'id': 'order123'}}")


def test_process_trading_data_dummy(mock_logger_ibkr_service, mock_place_order, mock_get_contract_id):
    """Test that process_trading_data doesn't place orders when in dummy mode"""

    # Configure mock and create trading data with "dummy" set to "YES" to test dummy mode
    mock_get_contract_id.return_value = "123456"
    trading_data = {
        "dummy": "YES",
        "symbol": "ES",
        "side": "B",
        "price": "4500.00"
    }

    # Call process_trading_data with dummy trading data
    process_trading_data(trading_data)

    # Verify logger recorded the trading data, contract ID was retrieved, but no order was placed (dummy mode)
    mock_logger_ibkr_service.info.assert_called_once_with(f"Trading data received: {trading_data}")
    mock_get_contract_id.assert_called_once_with("ES")
    mock_place_order.assert_not_called()


def test_process_trading_data_sell(mock_logger_ibkr_service, mock_place_order, mock_get_contract_id):
    """Test that process_trading_data correctly processes sell orders"""

    # Configure mocks and create trading data with "side" set to "S" (sell) to test sell orders
    mock_get_contract_id.return_value = "123456"
    mock_place_order.return_value = {"id": "order123"}
    trading_data = {
        "dummy": "NO",
        "symbol": "ES",
        "side": "S",
        "price": "4500.00"
    }

    # Call process_trading_data with sell order trading data
    process_trading_data(trading_data)

    # Verify logger recorded the trading data, contract ID was retrieved, sell order was placed, and success was logged
    mock_logger_ibkr_service.info.assert_any_call(f"Trading data received: {trading_data}")
    mock_get_contract_id.assert_called_once_with("ES")
    mock_place_order.assert_called_once_with("123456", "S")
    mock_logger_ibkr_service.info.assert_any_call(f"Order placed: {{'id': 'order123'}}")


def test_process_trading_data_missing_fields(mock_logger_ibkr_service, mock_place_order, mock_get_contract_id):
    """Test that process_trading_data handles trading data with missing fields"""

    # Configure mock and create trading data with missing "dummy" field to test default behavior
    mock_get_contract_id.return_value = "123456"
    trading_data = {
        # Missing dummy field
        "symbol": "ES",
        "side": "B",
        "price": "4500.00"
    }

    # Call process_trading_data with incomplete trading data
    process_trading_data(trading_data)

    # Verify logger recorded the trading data, contract ID was retrieved, and order was placed despite missing field
    mock_logger_ibkr_service.info.assert_any_call(f"Trading data received: {trading_data}")
    mock_get_contract_id.assert_called_once_with("ES")
    mock_place_order.assert_called_once_with("123456", "B")


def test_process_trading_data_error_handling(mock_logger_ibkr_service, mock_place_order, mock_get_contract_id):
    """Test that process_trading_data propagates exceptions from get_contract_id"""

    # Configure mock to raise an exception when get_contract_id is called and create normal trading data
    mock_get_contract_id.side_effect = ValueError("Test error")
    trading_data = {
        "dummy": "NO",
        "symbol": "ES",
        "side": "B",
        "price": "4500.00"
    }

    # Verify that the ValueError from get_contract_id propagates through process_trading_data
    # The function doesn't have explicit error handling, so we expect the exception to propagate
    with pytest.raises(ValueError):
        process_trading_data(trading_data)

    # Verify logger recorded the trading data, get_contract_id was called but failed, and place_order was never called
    mock_logger_ibkr_service.info.assert_called_once_with(f"Trading data received: {trading_data}")
    mock_get_contract_id.assert_called_once_with("ES")
    mock_place_order.assert_not_called()
