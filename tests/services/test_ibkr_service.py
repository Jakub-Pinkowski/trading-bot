from unittest.mock import patch

import pytest

from app.services.ibkr_service import process_trading_data


@patch('app.services.ibkr_service.get_contract_id')
@patch('app.services.ibkr_service.place_order')
@patch('app.services.ibkr_service.logger')
def test_process_trading_data_normal(mock_logger, mock_place_order, mock_get_contract_id):
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
    mock_logger.info.assert_any_call(f"Trading data received: {trading_data}")
    mock_get_contract_id.assert_called_once_with("ES")
    mock_place_order.assert_called_once_with("123456", "B")
    mock_logger.info.assert_any_call(f"Order placed: {{'id': 'order123'}}")


@patch('app.services.ibkr_service.get_contract_id')
@patch('app.services.ibkr_service.place_order')
@patch('app.services.ibkr_service.logger')
def test_process_trading_data_dummy(mock_logger, mock_place_order, mock_get_contract_id):
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
    mock_logger.info.assert_called_once_with(f"Trading data received: {trading_data}")
    mock_get_contract_id.assert_called_once_with("ES")
    mock_place_order.assert_not_called()


@patch('app.services.ibkr_service.get_contract_id')
@patch('app.services.ibkr_service.place_order')
@patch('app.services.ibkr_service.logger')
def test_process_trading_data_sell(mock_logger, mock_place_order, mock_get_contract_id):
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
    mock_logger.info.assert_any_call(f"Trading data received: {trading_data}")
    mock_get_contract_id.assert_called_once_with("ES")
    mock_place_order.assert_called_once_with("123456", "S")
    mock_logger.info.assert_any_call(f"Order placed: {{'id': 'order123'}}")


@patch('app.services.ibkr_service.get_contract_id')
@patch('app.services.ibkr_service.place_order')
@patch('app.services.ibkr_service.logger')
def test_process_trading_data_missing_fields(mock_logger, mock_place_order, mock_get_contract_id):
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
    mock_logger.info.assert_any_call(f"Trading data received: {trading_data}")
    mock_get_contract_id.assert_called_once_with("ES")
    mock_place_order.assert_called_once_with("123456", "B")


@patch('app.services.ibkr_service.get_contract_id')
@patch('app.services.ibkr_service.place_order')
@patch('app.services.ibkr_service.logger')
def test_process_trading_data_error_handling(mock_logger, mock_place_order, mock_get_contract_id):
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

    mock_logger.info.assert_called_once_with(f"Trading data received: {trading_data}")
    mock_get_contract_id.assert_called_once_with("ES")
    mock_place_order.assert_not_called()
