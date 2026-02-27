"""
Tests for IBKR Rollover Module.

Tests cover:
- Missing required field validation
- Delegation and return value pass-through
- Exception propagation from dependencies
- Outside warning window early return
- No open position warning
- Long and short rollover with REOPEN_ON_ROLLOVER=True
- Long and short close-only with REOPEN_ON_ROLLOVER=False
- Close order failure handling
"""
from datetime import datetime, timedelta
from unittest.mock import MagicMock, PropertyMock

import pytest

from app.ibkr.rollover import _check_and_rollover_position, process_rollover_data


# ==================== Helpers ====================

def _make_contracts():
    """Return a (current, new) contract pair for use in tests."""
    return (
        {'conid': 'old123', 'expirationDate': 20260313},
        {'conid': 'new456', 'expirationDate': 20260514},
    )


# ==================== Test Classes ====================

class TestProcessRolloverData:
    """Test rollover data processing and delegation."""

    def test_missing_symbol_raises_value_error(
        self, mock_logger_rollover, mock_check_and_rollover_position
    ):
        """Test ValueError raised before delegation when symbol is missing."""
        with pytest.raises(ValueError, match='Missing required field: symbol'):
            process_rollover_data({})

        mock_check_and_rollover_position.assert_not_called()

    def test_delegates_to_check_and_rollover_position(
        self, mock_logger_rollover, mock_check_and_rollover_position
    ):
        """Test symbol from payload is passed to _check_and_rollover_position."""
        mock_check_and_rollover_position.return_value = {'status': 'no_rollover_needed'}

        result = process_rollover_data({'symbol': 'ZC1!'})

        mock_check_and_rollover_position.assert_called_once_with('ZC1!')
        assert result == {'status': 'no_rollover_needed'}

    def test_returns_result_unchanged(
        self, mock_logger_rollover, mock_check_and_rollover_position
    ):
        """Test result from _check_and_rollover_position is returned as-is."""
        expected = {'status': 'rolled', 'old_conid': 'old123', 'new_conid': 'new456', 'side': 'B'}
        mock_check_and_rollover_position.return_value = expected

        result = process_rollover_data({'symbol': 'ZC1!'})

        assert result is expected

    def test_value_error_from_check_propagates(
        self, mock_logger_rollover, mock_check_and_rollover_position
    ):
        """Test ValueError from _check_and_rollover_position propagates out."""
        mock_check_and_rollover_position.side_effect = ValueError('No switch dates')

        with pytest.raises(ValueError, match='No switch dates'):
            process_rollover_data({'symbol': 'ZC1!'})


class TestCheckAndRolloverPosition:
    """Test position check and rollover execution logic."""

    # --- Outside Warning Window ---

    def test_outside_window_returns_no_rollover_needed(
        self, mock_logger_rollover, mock_contract_resolver_rollover,
        mock_place_order_rollover, mock_get_contract_position_rollover
    ):
        """Test no action taken when outside the warning window."""
        mock_contract_resolver_rollover.next_switch_date = datetime.today() + timedelta(days=10)

        result = _check_and_rollover_position('ZC1!')

        assert result == {'status': 'no_rollover_needed'}
        mock_get_contract_position_rollover.assert_not_called()
        mock_place_order_rollover.assert_not_called()

    # --- No Open Position ---

    def test_within_window_no_position_returns_warning(
        self, mock_logger_rollover, mock_contract_resolver_rollover,
        mock_place_order_rollover, mock_get_contract_position_rollover
    ):
        """Test warning returned and no order placed when position is 0."""
        mock_contract_resolver_rollover.next_switch_date = datetime.today() + timedelta(hours=12)
        mock_contract_resolver_rollover.ibkr_symbol = 'ZC'
        mock_contract_resolver_rollover.get_rollover_pair.return_value = _make_contracts()
        mock_get_contract_position_rollover.return_value = 0

        result = _check_and_rollover_position('ZC1!')

        assert result['status'] == 'warning'
        assert 'days_until_switch' in result
        mock_place_order_rollover.assert_not_called()
        mock_logger_rollover.warning.assert_called_once()

    # --- Rollover: REOPEN_ON_ROLLOVER=True ---

    def test_long_position_rolls_to_new_contract(
        self, monkeypatch, mock_logger_rollover, mock_contract_resolver_rollover,
        mock_place_order_rollover, mock_get_contract_position_rollover
    ):
        """Test long position closed with SELL and reopened with BUY on new contract."""
        monkeypatch.setattr('app.ibkr.rollover.REOPEN_ON_ROLLOVER', True)
        mock_contract_resolver_rollover.next_switch_date = datetime.today() + timedelta(hours=12)
        mock_contract_resolver_rollover.ibkr_symbol = 'ZC'
        mock_contract_resolver_rollover.get_rollover_pair.return_value = _make_contracts()
        mock_get_contract_position_rollover.return_value = 1
        mock_place_order_rollover.return_value = {'orderId': 'abc'}

        result = _check_and_rollover_position('ZC1!')

        assert result == {'status': 'rolled', 'old_conid': 'old123', 'new_conid': 'new456', 'side': 'B'}
        assert mock_place_order_rollover.call_count == 2
        mock_place_order_rollover.assert_any_call('old123', 'S')
        mock_place_order_rollover.assert_any_call('new456', 'B')

    def test_short_position_rolls_to_new_contract(
        self, monkeypatch, mock_logger_rollover, mock_contract_resolver_rollover,
        mock_place_order_rollover, mock_get_contract_position_rollover
    ):
        """Test short position closed with BUY and reopened with SELL on new contract."""
        monkeypatch.setattr('app.ibkr.rollover.REOPEN_ON_ROLLOVER', True)
        mock_contract_resolver_rollover.next_switch_date = datetime.today() + timedelta(hours=12)
        mock_contract_resolver_rollover.ibkr_symbol = 'ZC'
        mock_contract_resolver_rollover.get_rollover_pair.return_value = _make_contracts()
        mock_get_contract_position_rollover.return_value = -1
        mock_place_order_rollover.return_value = {'orderId': 'abc'}

        result = _check_and_rollover_position('ZC1!')

        assert result == {'status': 'rolled', 'old_conid': 'old123', 'new_conid': 'new456', 'side': 'S'}
        mock_place_order_rollover.assert_any_call('old123', 'B')
        mock_place_order_rollover.assert_any_call('new456', 'S')

    # --- Close Only: REOPEN_ON_ROLLOVER=False ---

    def test_long_position_closes_only_when_reopen_disabled(
        self, monkeypatch, mock_logger_rollover, mock_contract_resolver_rollover,
        mock_place_order_rollover, mock_get_contract_position_rollover
    ):
        """Test long position closed only when REOPEN_ON_ROLLOVER is False."""
        monkeypatch.setattr('app.ibkr.rollover.REOPEN_ON_ROLLOVER', False)
        mock_contract_resolver_rollover.next_switch_date = datetime.today() + timedelta(hours=12)
        mock_contract_resolver_rollover.ibkr_symbol = 'ZC'
        mock_contract_resolver_rollover.get_rollover_pair.return_value = _make_contracts()
        mock_get_contract_position_rollover.return_value = 1
        mock_place_order_rollover.return_value = {'orderId': 'abc'}

        result = _check_and_rollover_position('ZC1!')

        assert result == {'status': 'closed', 'old_conid': 'old123', 'new_conid': 'new456'}
        mock_place_order_rollover.assert_called_once_with('old123', 'S')

    def test_short_position_closes_only_when_reopen_disabled(
        self, monkeypatch, mock_logger_rollover, mock_contract_resolver_rollover,
        mock_place_order_rollover, mock_get_contract_position_rollover
    ):
        """Test short position closed only when REOPEN_ON_ROLLOVER is False."""
        monkeypatch.setattr('app.ibkr.rollover.REOPEN_ON_ROLLOVER', False)
        mock_contract_resolver_rollover.next_switch_date = datetime.today() + timedelta(hours=12)
        mock_contract_resolver_rollover.ibkr_symbol = 'ZC'
        mock_contract_resolver_rollover.get_rollover_pair.return_value = _make_contracts()
        mock_get_contract_position_rollover.return_value = -1
        mock_place_order_rollover.return_value = {'orderId': 'abc'}

        result = _check_and_rollover_position('ZC1!')

        assert result == {'status': 'closed', 'old_conid': 'old123', 'new_conid': 'new456'}
        mock_place_order_rollover.assert_called_once_with('old123', 'B')

    # --- Close Order Failure ---

    def test_close_order_failure_returns_error(
        self, mock_logger_rollover, mock_contract_resolver_rollover,
        mock_place_order_rollover, mock_get_contract_position_rollover
    ):
        """Test error status returned and reopen skipped when close order fails."""
        mock_contract_resolver_rollover.next_switch_date = datetime.today() + timedelta(hours=12)
        mock_contract_resolver_rollover.ibkr_symbol = 'ZC'
        mock_contract_resolver_rollover.get_rollover_pair.return_value = _make_contracts()
        mock_get_contract_position_rollover.return_value = 1
        mock_place_order_rollover.return_value = {'success': False, 'error': 'Insufficient funds', 'details': {}}

        result = _check_and_rollover_position('ZC1!')

        assert result == {
            'status': 'error',
            'order': {'success': False, 'error': 'Insufficient funds', 'details': {}},
        }
        mock_place_order_rollover.assert_called_once_with('old123', 'S')
        mock_logger_rollover.error.assert_called_once()

    def test_reopen_order_failure_returns_reopen_failed(
        self, monkeypatch, mock_logger_rollover, mock_contract_resolver_rollover,
        mock_place_order_rollover, mock_get_contract_position_rollover
    ):
        """Test reopen_failed status returned when close succeeds but reopen fails."""
        monkeypatch.setattr('app.ibkr.rollover.REOPEN_ON_ROLLOVER', True)
        mock_contract_resolver_rollover.next_switch_date = datetime.today() + timedelta(hours=12)
        mock_contract_resolver_rollover.ibkr_symbol = 'ZC'
        mock_contract_resolver_rollover.get_rollover_pair.return_value = _make_contracts()
        mock_get_contract_position_rollover.return_value = 1
        mock_place_order_rollover.side_effect = [
            {'orderId': 'close_ok'},
            {'success': False, 'error': 'Insufficient funds', 'details': {}},
        ]

        result = _check_and_rollover_position('ZC1!')

        assert result == {
            'status': 'reopen_failed',
            'old_conid': 'old123',
            'new_conid': 'new456',
            'order': {'success': False, 'error': 'Insufficient funds', 'details': {}},
        }
        assert mock_place_order_rollover.call_count == 2
        mock_logger_rollover.critical.assert_called_once()
        mock_logger_rollover.error.assert_called_once()

    # --- Position Check Failure ---

    def test_position_check_failure_returns_error(
        self, mock_logger_rollover, mock_contract_resolver_rollover,
        mock_place_order_rollover, mock_get_contract_position_rollover
    ):
        """Test error returned and no order placed when position check fails (API error)."""
        mock_contract_resolver_rollover.next_switch_date = datetime.today() + timedelta(hours=12)
        mock_contract_resolver_rollover.ibkr_symbol = 'ZC'
        mock_contract_resolver_rollover.get_rollover_pair.return_value = _make_contracts()
        mock_get_contract_position_rollover.return_value = None

        result = _check_and_rollover_position('ZC1!')

        assert result['status'] == 'error'
        mock_place_order_rollover.assert_not_called()
        mock_logger_rollover.error.assert_called_once()

    # --- Exception Propagation ---

    def test_next_switch_date_error_propagates(
        self, monkeypatch, mock_logger_rollover,
        mock_place_order_rollover, mock_get_contract_position_rollover
    ):
        """Test ValueError from next_switch_date propagates out."""
        mock_instance = MagicMock()
        mock_class = MagicMock(return_value=mock_instance)
        monkeypatch.setattr('app.ibkr.rollover.ContractResolver', mock_class)
        type(mock_instance).next_switch_date = PropertyMock(
            side_effect=ValueError('No switch dates found')
        )

        with pytest.raises(ValueError, match='No switch dates found'):
            _check_and_rollover_position('ZC1!')

        mock_place_order_rollover.assert_not_called()
