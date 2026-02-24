"""
Tests for IBKR Trading Module.

Tests cover:
- Normal order placement for buy and sell sides
- Dummy mode skip behavior
- Missing required field validation
- Order failure status handling
- Exception propagation from dependencies
"""
import pytest

from app.ibkr.trading import process_trading_data


# ==================== Test Classes ====================

class TestProcessTradingData:
    """Test trading data processing and order dispatch."""

    # --- Successful Order Placement ---

    def test_normal_buy_places_order(
        self, mock_logger_trading, mock_place_order, mock_get_contract_id
    ):
        """Test buy order is placed and success is logged."""
        mock_get_contract_id.return_value = "123456"
        mock_place_order.return_value = {"id": "order123"}
        trading_data = {"dummy": "NO", "symbol": "ZC", "side": "B", "price": "4500.00"}

        result = process_trading_data(trading_data)

        assert result == {"status": "order_placed", "order": {"id": "order123"}}
        mock_get_contract_id.assert_called_once_with("ZC")
        mock_place_order.assert_called_once_with("123456", "B")
        mock_logger_trading.info.assert_any_call(f"Trading data received: {trading_data}")
        mock_logger_trading.info.assert_any_call("Order placed: {'id': 'order123'}")

    def test_normal_sell_places_order(
        self, mock_logger_trading, mock_place_order, mock_get_contract_id
    ):
        """Test sell order is placed and success is logged."""
        mock_get_contract_id.return_value = "123456"
        mock_place_order.return_value = {"id": "order123"}
        trading_data = {"dummy": "NO", "symbol": "ZC", "side": "S", "price": "4500.00"}

        result = process_trading_data(trading_data)

        assert result == {"status": "order_placed", "order": {"id": "order123"}}
        mock_place_order.assert_called_once_with("123456", "S")

    def test_missing_dummy_field_places_order(
        self, mock_logger_trading, mock_place_order, mock_get_contract_id
    ):
        """Test order is placed when dummy field is absent (defaults to live mode)."""
        mock_get_contract_id.return_value = "123456"
        mock_place_order.return_value = {"id": "order123"}
        trading_data = {"symbol": "ZC", "side": "B", "price": "4500.00"}

        result = process_trading_data(trading_data)

        assert result == {"status": "order_placed", "order": {"id": "order123"}}
        mock_get_contract_id.assert_called_once_with("ZC")
        mock_place_order.assert_called_once_with("123456", "B")

    # --- Dummy Mode ---

    def test_dummy_mode_skips_order(
        self, mock_logger_trading, mock_place_order, mock_get_contract_id
    ):
        """Test dummy=YES skips API call and order placement."""
        trading_data = {"dummy": "YES", "symbol": "ZC", "side": "B", "price": "4500.00"}

        result = process_trading_data(trading_data)

        assert result == {"status": "dummy_skip"}
        mock_get_contract_id.assert_not_called()
        mock_place_order.assert_not_called()
        mock_logger_trading.info.assert_called_once_with(
            f"Trading data received: {trading_data}"
        )

    # --- Validation Errors ---

    def test_missing_symbol_raises_value_error(
        self, mock_logger_trading, mock_place_order, mock_get_contract_id
    ):
        """Test ValueError raised before any API call when symbol is missing."""
        trading_data = {"dummy": "NO", "side": "B", "price": "4500.00"}

        with pytest.raises(ValueError, match="Missing required field: symbol"):
            process_trading_data(trading_data)

        mock_get_contract_id.assert_not_called()
        mock_place_order.assert_not_called()

    def test_missing_side_raises_value_error(
        self, mock_logger_trading, mock_place_order, mock_get_contract_id
    ):
        """Test ValueError raised before any API call when side is missing."""
        trading_data = {"dummy": "NO", "symbol": "ZC", "price": "4500.00"}

        with pytest.raises(ValueError, match="Missing required field: side"):
            process_trading_data(trading_data)

        mock_get_contract_id.assert_not_called()
        mock_place_order.assert_not_called()

    # --- Order Failure Handling ---

    def test_order_failure_returns_failed_status(
        self, mock_logger_trading, mock_place_order, mock_get_contract_id
    ):
        """Test order_failed status returned and error logged when place_order reports failure."""
        mock_get_contract_id.return_value = "123456"
        mock_place_order.return_value = {"success": False, "error": "Insufficient funds", "details": {}}
        trading_data = {"dummy": "NO", "symbol": "ZC", "side": "B", "price": "4500.00"}

        result = process_trading_data(trading_data)

        assert result == {
            "status": "order_failed",
            "order": {"success": False, "error": "Insufficient funds", "details": {}},
        }
        mock_logger_trading.error.assert_called_once()

    # --- Exception Propagation ---

    def test_contract_id_exception_propagates(
        self, mock_logger_trading, mock_place_order, mock_get_contract_id
    ):
        """Test ValueError from get_contract_id propagates out of process_trading_data."""
        mock_get_contract_id.side_effect = ValueError("Test error")
        trading_data = {"dummy": "NO", "symbol": "ZC", "side": "B", "price": "4500.00"}

        with pytest.raises(ValueError, match="Test error"):
            process_trading_data(trading_data)

        mock_get_contract_id.assert_called_once_with("ZC")
        mock_place_order.assert_not_called()

    def test_place_order_exception_propagates(
        self, mock_logger_trading, mock_place_order, mock_get_contract_id
    ):
        """Test exception from place_order propagates out of process_trading_data."""
        mock_get_contract_id.return_value = "123456"
        mock_place_order.side_effect = ValueError("Invalid side 'X': expected 'B' or 'S'")
        trading_data = {"dummy": "NO", "symbol": "ZC", "side": "X", "price": "4500.00"}

        with pytest.raises(ValueError, match="Invalid side"):
            process_trading_data(trading_data)

        mock_get_contract_id.assert_called_once_with("ZC")
        mock_place_order.assert_called_once_with("123456", "X")
