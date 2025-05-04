import pytest

from app.services.ibkr_service import process_trading_data


def test_process_trading_data_normal(mock_logger_ibkr_service, mock_place_order, mock_get_contract_id):
    # Setup
    mock_get_contract_id.return_value = "123456"
    mock_place_order.return_value = {"id": "order123"}
    trading_data = {
        "dummy": "NO",
        "symbol": "ES",
        "side": "B",
        "price": "4500.00"
    }

    # Execute
    process_trading_data(trading_data)

    # Assert
    mock_logger_ibkr_service.info.assert_any_call(f"Trading data received: {trading_data}")
    mock_get_contract_id.assert_called_once_with("ES")
    mock_place_order.assert_called_once_with("123456", "B")
    mock_logger_ibkr_service.info.assert_any_call(f"Order placed: {{'id': 'order123'}}")


def test_process_trading_data_dummy(mock_logger_ibkr_service, mock_place_order, mock_get_contract_id):
    # Setup
    mock_get_contract_id.return_value = "123456"
    trading_data = {
        "dummy": "YES",
        "symbol": "ES",
        "side": "B",
        "price": "4500.00"
    }

    # Execute
    process_trading_data(trading_data)

    # Assert
    mock_logger_ibkr_service.info.assert_called_once_with(f"Trading data received: {trading_data}")
    mock_get_contract_id.assert_called_once_with("ES")
    mock_place_order.assert_not_called()


def test_process_trading_data_sell(mock_logger_ibkr_service, mock_place_order, mock_get_contract_id):
    # Setup
    mock_get_contract_id.return_value = "123456"
    mock_place_order.return_value = {"id": "order123"}
    trading_data = {
        "dummy": "NO",
        "symbol": "ES",
        "side": "S",
        "price": "4500.00"
    }

    # Execute
    process_trading_data(trading_data)

    # Assert
    mock_logger_ibkr_service.info.assert_any_call(f"Trading data received: {trading_data}")
    mock_get_contract_id.assert_called_once_with("ES")
    mock_place_order.assert_called_once_with("123456", "S")
    mock_logger_ibkr_service.info.assert_any_call(f"Order placed: {{'id': 'order123'}}")


def test_process_trading_data_missing_fields(mock_logger_ibkr_service, mock_place_order, mock_get_contract_id):
    # Setup
    mock_get_contract_id.return_value = "123456"
    trading_data = {
        # Missing dummy field
        "symbol": "ES",
        "side": "B",
        "price": "4500.00"
    }

    # Execute
    process_trading_data(trading_data)

    # Assert
    mock_logger_ibkr_service.info.assert_any_call(f"Trading data received: {trading_data}")
    mock_get_contract_id.assert_called_once_with("ES")
    mock_place_order.assert_called_once_with("123456", "B")


def test_process_trading_data_error_handling(mock_logger_ibkr_service, mock_place_order, mock_get_contract_id):
    # Setup
    mock_get_contract_id.side_effect = ValueError("Test error")
    trading_data = {
        "dummy": "NO",
        "symbol": "ES",
        "side": "B",
        "price": "4500.00"
    }

    # Execute and Assert
    # The function doesn't have explicit error handling, so we expect the exception to propagate
    with pytest.raises(ValueError):
        process_trading_data(trading_data)

    mock_logger_ibkr_service.info.assert_called_once_with(f"Trading data received: {trading_data}")
    mock_get_contract_id.assert_called_once_with("ES")
    mock_place_order.assert_not_called()
