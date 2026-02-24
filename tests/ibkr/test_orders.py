"""
Tests for IBKR Orders Module.

Tests cover:
- Cache invalidation via the IBKR portfolio API
- Contract position retrieval with automatic cache refresh
- Order message suppression
- Order placement: new positions, already-in-position skips, reversals,
  message suppression flow, known error responses, and unexpected exceptions
"""
import pytest

from app.ibkr.orders import (
    _get_contract_position,
    _invalidate_cache,
    place_order,
    _suppress_messages,
    MAX_SUPPRESS_RETRIES,
    QUANTITY_TO_TRADE,
)
from config import ACCOUNT_ID


# ==================== Test Classes ====================

class TestInvalidateCache:
    """Test portfolio cache invalidation."""

    def test_calls_correct_api_endpoint(self, mock_api_post_orders):
        """Test cache invalidation calls the portfolio invalidate endpoint."""
        _invalidate_cache()

        mock_api_post_orders.assert_called_once_with(
            f"portfolio/{ACCOUNT_ID}/positions/invalidate", {}
        )

    def test_api_error_propagates(self, mock_api_post_orders):
        """Test API exception propagates so callers are not left with stale data."""
        mock_api_post_orders.side_effect = Exception("API error")

        with pytest.raises(Exception, match="API error"):
            _invalidate_cache()

        mock_api_post_orders.assert_called_once()


class TestGetContractPosition:
    """Test contract position retrieval with cache invalidation."""

    def test_returns_position_when_found(self, mock_api_get_orders, mock_invalidate_cache):
        """Test correct position value returned when contract exists in portfolio."""
        mock_api_get_orders.return_value = [
            {"conid": "123456", "position": 5},
            {"conid": "789012", "position": -3},
        ]

        result = _get_contract_position("123456")

        assert result == 5
        mock_invalidate_cache.assert_called_once()
        mock_api_get_orders.assert_called_once_with(f"portfolio/{ACCOUNT_ID}/positions")

    def test_returns_zero_when_not_found(self, mock_api_get_orders, mock_invalidate_cache):
        """Test zero returned when the contract is not in the portfolio."""
        mock_api_get_orders.return_value = [{"conid": "789012", "position": -3}]

        result = _get_contract_position("123456")

        assert result == 0
        mock_invalidate_cache.assert_called_once()

    def test_returns_zero_on_api_error(self, mock_api_get_orders, mock_invalidate_cache):
        """Test zero returned gracefully when API raises an exception."""
        mock_api_get_orders.side_effect = Exception("API error")

        result = _get_contract_position("123456")

        assert result == 0
        mock_invalidate_cache.assert_called_once()

    def test_propagates_invalidate_cache_error(self, mock_invalidate_cache):
        """Test cache invalidation errors propagate to callers."""
        mock_invalidate_cache.side_effect = Exception("Cache invalidation error")

        with pytest.raises(Exception, match="Cache invalidation error"):
            _get_contract_position("123456")


class TestSuppressMessages:
    """Test IBKR interactive message suppression."""

    def test_calls_correct_api_endpoint(self, mock_api_post_orders):
        """Test suppress_messages calls the suppress endpoint with the provided IDs."""
        mock_api_post_orders.return_value = {"success": True}

        _suppress_messages(["1", "2"])

        mock_api_post_orders.assert_called_once_with(
            "iserver/questions/suppress", {"messageIds": ["1", "2"]}
        )

    def test_api_error_is_swallowed(self, mock_api_post_orders):
        """Test API errors during suppression do not propagate."""
        mock_api_post_orders.side_effect = Exception("API error")

        # Should not raise
        _suppress_messages(["1", "2"])

        mock_api_post_orders.assert_called_once()


class TestPlaceOrder:
    """Test order placement including position checks, reversals, and error handling."""

    # --- New Positions ---

    def test_new_buy_position(self, mock_get_contract_position, mock_api_post_orders):
        """Test BUY order placed with correct quantity when no position exists."""
        mock_get_contract_position.return_value = 0
        mock_api_post_orders.return_value = {"id": "123456"}

        result = place_order("123456", "B")

        assert result == {"id": "123456"}
        mock_get_contract_position.assert_called_once_with("123456")
        order_details = mock_api_post_orders.call_args[0][1]
        assert order_details["orders"][0]["side"] == "BUY"
        assert order_details["orders"][0]["quantity"] == QUANTITY_TO_TRADE

    def test_new_sell_position(self, mock_get_contract_position, mock_api_post_orders):
        """Test SELL order placed with correct quantity when no position exists."""
        mock_get_contract_position.return_value = 0
        mock_api_post_orders.return_value = {"id": "123456"}

        result = place_order("123456", "S")

        assert result == {"id": "123456"}
        order_details = mock_api_post_orders.call_args[0][1]
        assert order_details["orders"][0]["side"] == "SELL"
        assert order_details["orders"][0]["quantity"] == QUANTITY_TO_TRADE

    # --- Already In Desired Position ---

    def test_already_long_skips_buy(self, mock_get_contract_position):
        """Test no order placed when already long and a buy is requested."""
        mock_get_contract_position.return_value = 1

        result = place_order("123456", "B")

        assert result == {"success": True, "message": "No action needed: already in desired position"}
        mock_get_contract_position.assert_called_once_with("123456")

    def test_already_short_skips_sell(self, mock_get_contract_position):
        """Test no order placed when already short and a sell is requested."""
        mock_get_contract_position.return_value = -1

        result = place_order("123456", "S")

        assert result == {"success": True, "message": "No action needed: already in desired position"}
        mock_get_contract_position.assert_called_once_with("123456")

    def test_invalid_side_raises_value_error(self, mock_get_contract_position):
        """Test ValueError raised for unrecognized side indicator."""
        mock_get_contract_position.return_value = 0

        with pytest.raises(ValueError, match="Invalid side 'X': expected 'B' or 'S'"):
            place_order("123456", "X")

    # --- Position Reversals ---

    def test_reverses_short_to_long(self, mock_get_contract_position, mock_api_post_orders):
        """Test BUY order placed with standard quantity to reverse an existing short."""
        mock_get_contract_position.return_value = -1
        mock_api_post_orders.return_value = {"id": "123456"}

        result = place_order("123456", "B")

        assert result == {"id": "123456"}
        order_details = mock_api_post_orders.call_args[0][1]
        assert order_details["orders"][0]["side"] == "BUY"
        assert order_details["orders"][0]["quantity"] == QUANTITY_TO_TRADE

    def test_reverses_long_to_short(self, mock_get_contract_position, mock_api_post_orders):
        """Test SELL order placed with standard quantity to reverse an existing long."""
        mock_get_contract_position.return_value = 1
        mock_api_post_orders.return_value = {"id": "123456"}

        result = place_order("123456", "S")

        assert result == {"id": "123456"}
        order_details = mock_api_post_orders.call_args[0][1]
        assert order_details["orders"][0]["side"] == "SELL"
        assert order_details["orders"][0]["quantity"] == QUANTITY_TO_TRADE

    # --- Message Suppression Flow ---

    def test_suppresses_messages_then_retries(
        self, mock_get_contract_position, mock_api_post_orders, mock_suppress_messages
    ):
        """Test message suppression handled and order retried successfully."""
        mock_get_contract_position.return_value = 0
        # First call requires suppression, second call succeeds
        mock_api_post_orders.side_effect = [
            [{"messageIds": ["1", "2"]}],
            {"id": "123456"},
        ]

        result = place_order("123456", "B")

        assert result == {"id": "123456"}
        assert mock_api_post_orders.call_count == 2
        mock_suppress_messages.assert_called_once_with(["1", "2"])

    def test_returns_error_after_max_suppression_retries(
        self,
        mock_logger_orders,
        mock_get_contract_position,
        mock_api_post_orders,
        mock_suppress_messages,
    ):
        """Test error returned after exceeding the maximum suppression retry limit."""
        mock_get_contract_position.return_value = 0
        # Always returns message IDs, suppression never clears
        mock_api_post_orders.return_value = [{"messageIds": ["1", "2"]}]

        result = place_order("123456", "B")

        assert result == {"success": False, "error": "Exceeded maximum suppression retries"}
        assert mock_api_post_orders.call_count == MAX_SUPPRESS_RETRIES
        assert mock_suppress_messages.call_count == MAX_SUPPRESS_RETRIES
        mock_logger_orders.error.assert_called_once()

    # --- Known Error Responses ---

    def test_insufficient_funds_returns_error(
        self, mock_logger_orders, mock_api_post_orders, mock_get_contract_position
    ):
        """Test insufficient funds error returns a structured failure response."""
        mock_get_contract_position.return_value = 0
        mock_api_post_orders.return_value = {"error": "available funds are insufficient"}

        result = place_order("123456", "B")

        assert result["success"] is False
        assert result["error"] == "Insufficient funds"
        mock_get_contract_position.assert_called_once_with("123456")
        mock_logger_orders.error.assert_called_once()

    def test_insufficient_funds_alternate_spelling_returns_error(
        self, mock_logger_orders, mock_api_post_orders, mock_get_contract_position
    ):
        """Test IBKR API typo variant 'in sufficient' (with space) is also handled."""
        mock_get_contract_position.return_value = 0
        mock_api_post_orders.return_value = {"error": "available funds are in sufficient"}

        result = place_order("123456", "B")

        assert result["success"] is False
        assert result["error"] == "Insufficient funds"
        mock_logger_orders.error.assert_called_once()

    def test_derivative_rules_error_returns_error(
        self, mock_logger_orders, mock_api_post_orders, mock_get_contract_position
    ):
        """Test derivative rules non-compliance returns a structured failure response."""
        mock_get_contract_position.return_value = 0
        mock_api_post_orders.return_value = {
            "error": "does not comply with our order handling rules for derivatives"
        }

        result = place_order("123456", "B")

        assert result["success"] is False
        assert result["error"] == "Non-compliance with derivative rules"
        mock_get_contract_position.assert_called_once_with("123456")
        mock_logger_orders.error.assert_called_once()

    def test_unrecognized_error_returns_generic_message(
        self, mock_logger_orders, mock_api_post_orders, mock_get_contract_position
    ):
        """Test unrecognized error type returns a generic failure message."""
        mock_get_contract_position.return_value = 0
        mock_api_post_orders.return_value = {"error": "some other error"}

        result = place_order("123456", "B")

        assert result["success"] is False
        assert result["error"] == "Unhandled error"
        mock_logger_orders.error.assert_called_once()

    # --- Unexpected Exceptions ---

    def test_unexpected_exception_caught_and_logged(
        self, mock_logger_orders, mock_api_post_orders, mock_get_contract_position
    ):
        """Test unexpected exception is caught, logged, and returns a failure response."""
        mock_get_contract_position.return_value = 0
        mock_api_post_orders.side_effect = Exception("Test error")

        result = place_order("123456", "B")

        assert result["success"] is False
        assert result["error"] == "An unexpected error occurred"
        mock_logger_orders.exception.assert_called_once()
