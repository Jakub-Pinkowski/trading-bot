"""
Tests for TrailingStopManager.

Tests cover:
- Initialization and configuration
- Trailing stop calculation logic
- Stop trigger detection (long and short)
- Stop update logic (tightening only)
- Look-ahead bias prevention
- Integration with PositionManager
- Edge cases (gaps, exact stop hits, volatile bars)
- Callback functionality
"""
import pandas as pd
import pytest

from app.backtesting.strategies.base.position_manager import PositionManager
from app.backtesting.strategies.base.trailing_stop_manager import (
    TrailingStopManager,
    _should_trigger_stop,
    _update_trailing_stop
)


# ==================== Fixtures ====================

@pytest.fixture
def trailing_stop_manager():
    """Standard trailing stop manager with 5% trailing stop."""
    return TrailingStopManager(trailing_percentage=5.0)


@pytest.fixture
def position_manager_with_trailing():
    """Position manager with trailing stop enabled."""
    return PositionManager(
        slippage_ticks=0,
        symbol='ZS',
        trailing=5.0
    )


@pytest.fixture
def sample_timestamp():
    """Sample timestamp for testing."""
    return pd.Timestamp('2025-01-01 10:00:00')


# ==================== Test Classes ====================

class TestTrailingStopManagerInitialization:
    """Test TrailingStopManager initialization."""

    @pytest.mark.parametrize("trailing_percentage,callback", [
        (3.0, None),
        (5.0, lambda pos, price: None),
        (1.0, None),
        (10.0, None),
    ])
    def test_initialization_with_various_configs(self, trailing_percentage, callback):
        """Test trailing stop manager initializes correctly with various configurations."""
        tsm = TrailingStopManager(
            trailing_percentage=trailing_percentage,
            on_stop_triggered=callback
        )

        assert tsm.trailing_percentage == trailing_percentage
        assert tsm.on_stop_triggered == callback


class TestTrailingStopCalculation:
    """Test trailing stop calculation logic."""

    # --- Long Position Calculations ---

    def test_calculate_stop_long_position(self, trailing_stop_manager):
        """Test stop calculation for long position uses bar high."""
        position = 1  # Long
        price_high = 110.0
        price_low = 105.0

        new_stop = trailing_stop_manager.calculate_new_trailing_stop(
            position, price_high, price_low
        )

        # Long: high * (1 - 5%) = 110 * 0.95 = 104.50
        expected = 110.0 * (1 - 5.0 / 100)
        assert new_stop == pytest.approx(expected, abs=0.01)

    def test_calculate_stop_long_uses_high_not_low(self, trailing_stop_manager):
        """Test long stop calculation uses high price, not low."""
        position = 1
        price_high = 120.0
        price_low = 100.0  # Should be ignored for long

        new_stop = trailing_stop_manager.calculate_new_trailing_stop(
            position, price_high, price_low
        )

        # Should use high (120), not low (100)
        expected = 120.0 * 0.95
        assert new_stop == pytest.approx(expected, abs=0.01)
        assert new_stop > 100.0  # Stop above low

    # --- Short Position Calculations ---

    def test_calculate_stop_short_position(self, trailing_stop_manager):
        """Test stop calculation for short position uses bar low."""
        position = -1  # Short
        price_high = 105.0
        price_low = 100.0

        new_stop = trailing_stop_manager.calculate_new_trailing_stop(
            position, price_high, price_low
        )

        # Short: low * (1 + 5%) = 100 * 1.05 = 105.00
        expected = 100.0 * (1 + 5.0 / 100)
        assert new_stop == pytest.approx(expected, abs=0.01)

    def test_calculate_stop_short_uses_low_not_high(self, trailing_stop_manager):
        """Test short stop calculation uses low price, not high."""
        position = -1
        price_high = 120.0  # Should be ignored for short
        price_low = 100.0

        new_stop = trailing_stop_manager.calculate_new_trailing_stop(
            position, price_high, price_low
        )

        # Should use low (100), not high (120)
        expected = 100.0 * 1.05
        assert new_stop == pytest.approx(expected, abs=0.01)
        assert new_stop < 120.0  # Stop below high

    # --- Different Trailing Percentages ---

    def test_calculate_stop_with_different_percentages(self):
        """Test stop calculation with various trailing percentages."""
        test_cases = [
            (1.0, 100.0, 99.0),  # 1% trailing
            (2.0, 100.0, 98.0),  # 2% trailing
            (5.0, 100.0, 95.0),  # 5% trailing
            (10.0, 100.0, 90.0),  # 10% trailing
        ]

        for pct, price, expected_stop in test_cases:
            tsm = TrailingStopManager(trailing_percentage=pct)
            new_stop = tsm.calculate_new_trailing_stop(1, price, price)
            assert new_stop == pytest.approx(expected_stop, abs=0.01)

    # --- Edge Cases ---

    def test_calculate_stop_with_zero_position(self, trailing_stop_manager):
        """Test calculation returns None for invalid position."""
        new_stop = trailing_stop_manager.calculate_new_trailing_stop(
            0, 100.0, 100.0  # Invalid position
        )
        assert new_stop is None

    def test_calculate_stop_rounds_to_two_decimals(self, trailing_stop_manager):
        """Test stop calculation rounds to 2 decimal places."""
        position = 1
        price_high = 123.456
        price_low = 120.0

        new_stop = trailing_stop_manager.calculate_new_trailing_stop(
            position, price_high, price_low
        )

        # Should be rounded to 2 decimals
        assert isinstance(new_stop, float)
        assert len(str(new_stop).split('.')[-1]) <= 2


class TestStopTriggerDetection:
    """Test stop trigger detection logic."""

    # --- Long Position Triggers ---

    def test_long_stop_triggered_when_low_hits_stop(self):
        """Test long stop triggers when bar low hits stop level."""
        position = 1
        trailing_stop = 95.0
        price_high = 105.0
        price_low = 95.0  # Exactly at stop

        triggered = _should_trigger_stop(position, trailing_stop, price_high, price_low)
        assert triggered is True

    def test_long_stop_triggered_when_low_below_stop(self):
        """Test long stop triggers when bar low goes below stop."""
        position = 1
        trailing_stop = 95.0
        price_high = 100.0
        price_low = 90.0  # Below stop (gap down)

        triggered = _should_trigger_stop(position, trailing_stop, price_high, price_low)
        assert triggered is True

    def test_long_stop_not_triggered_when_low_above_stop(self):
        """Test long stop doesn't trigger when bar low stays above stop."""
        position = 1
        trailing_stop = 95.0
        price_high = 105.0
        price_low = 97.0  # Above stop

        triggered = _should_trigger_stop(position, trailing_stop, price_high, price_low)
        assert triggered is False

    # --- Short Position Triggers ---

    def test_short_stop_triggered_when_high_hits_stop(self):
        """Test short stop triggers when bar high hits stop level."""
        position = -1
        trailing_stop = 105.0
        price_high = 105.0  # Exactly at stop
        price_low = 95.0

        triggered = _should_trigger_stop(position, trailing_stop, price_high, price_low)
        assert triggered is True

    def test_short_stop_triggered_when_high_above_stop(self):
        """Test short stop triggers when bar high goes above stop."""
        position = -1
        trailing_stop = 105.0
        price_high = 110.0  # Above stop (gap up)
        price_low = 100.0

        triggered = _should_trigger_stop(position, trailing_stop, price_high, price_low)
        assert triggered is True

    def test_short_stop_not_triggered_when_high_below_stop(self):
        """Test short stop doesn't trigger when bar high stays below stop."""
        position = -1
        trailing_stop = 105.0
        price_high = 103.0  # Below stop
        price_low = 95.0

        triggered = _should_trigger_stop(position, trailing_stop, price_high, price_low)
        assert triggered is False


class TestStopUpdateLogic:
    """Test stop update logic (tightening only)."""

    # --- Long Position Updates ---

    def test_long_stop_tightens_upward(self, position_manager_with_trailing):
        """Test long stop tightens (moves up) when price moves favorably."""
        # Open long position
        position_manager_with_trailing.position = 1
        position_manager_with_trailing.trailing_stop = 95.0

        # Update with higher stop
        new_stop = 97.0
        _update_trailing_stop(position_manager_with_trailing, 1, 95.0, new_stop)

        # Stop should be updated (tightened)
        assert position_manager_with_trailing.trailing_stop == 97.0

    def test_long_stop_does_not_loosen_downward(self, position_manager_with_trailing):
        """Test long stop doesn't loosen (move down) - only tightens."""
        position_manager_with_trailing.position = 1
        position_manager_with_trailing.trailing_stop = 95.0

        # Attempt to update with lower stop
        new_stop = 93.0
        _update_trailing_stop(position_manager_with_trailing, 1, 95.0, new_stop)

        # Stop should NOT be updated (can't loosen)
        assert position_manager_with_trailing.trailing_stop == 95.0

    # --- Short Position Updates ---

    def test_short_stop_tightens_downward(self, position_manager_with_trailing):
        """Test short stop tightens (moves down) when price moves favorably."""
        position_manager_with_trailing.position = -1
        position_manager_with_trailing.trailing_stop = 105.0

        # Update with lower stop
        new_stop = 103.0
        _update_trailing_stop(position_manager_with_trailing, -1, 105.0, new_stop)

        # Stop should be updated (tightened)
        assert position_manager_with_trailing.trailing_stop == 103.0

    def test_short_stop_does_not_loosen_upward(self, position_manager_with_trailing):
        """Test short stop doesn't loosen (move up) - only tightens."""
        position_manager_with_trailing.position = -1
        position_manager_with_trailing.trailing_stop = 105.0

        # Attempt to update with higher stop
        new_stop = 107.0
        _update_trailing_stop(position_manager_with_trailing, -1, 105.0, new_stop)

        # Stop should NOT be updated (can't loosen)
        assert position_manager_with_trailing.trailing_stop == 105.0


class TestIntegrationWithPositionManager:
    """Test trailing stop manager integration with position manager."""

    def test_handle_trailing_stop_closes_long_when_triggered(
        self, trailing_stop_manager, position_manager_with_trailing, sample_timestamp
    ):
        """Test handling trailing stop closes long position when triggered."""
        # Open long position with trailing stop
        position_manager_with_trailing.open_position(1, sample_timestamp, 100.0)
        position_manager_with_trailing.trailing_stop = 95.0

        # Bar that hits stop
        price_high = 100.0
        price_low = 95.0  # Hits stop

        # Handle trailing stop
        closed = trailing_stop_manager.handle_trailing_stop(
            position_manager_with_trailing, sample_timestamp, price_high, price_low
        )

        # Position should be closed
        assert closed is True
        assert position_manager_with_trailing.has_open_position() is False

        # Trade should be recorded
        trades = position_manager_with_trailing.get_trades()
        assert len(trades) == 1
        assert trades[0]['exit_price'] == 95.0  # Closed at stop level

    def test_handle_trailing_stop_closes_short_when_triggered(
        self, trailing_stop_manager, position_manager_with_trailing, sample_timestamp
    ):
        """Test handling trailing stop closes short position when triggered."""
        # Open short position with trailing stop
        position_manager_with_trailing.open_position(-1, sample_timestamp, 100.0)
        position_manager_with_trailing.trailing_stop = 105.0

        # Bar that hits stop
        price_high = 105.0  # Hits stop
        price_low = 95.0

        # Handle trailing stop
        closed = trailing_stop_manager.handle_trailing_stop(
            position_manager_with_trailing, sample_timestamp, price_high, price_low
        )

        # Position should be closed
        assert closed is True
        assert position_manager_with_trailing.has_open_position() is False

        # Trade closed at stop level
        trades = position_manager_with_trailing.get_trades()
        assert len(trades) == 1
        assert trades[0]['exit_price'] == 105.0

    def test_handle_trailing_stop_updates_long_when_not_triggered(
        self, trailing_stop_manager, position_manager_with_trailing, sample_timestamp
    ):
        """Test stop updates when not triggered (long position)."""
        # Open long position
        position_manager_with_trailing.open_position(1, sample_timestamp, 100.0)
        initial_stop = position_manager_with_trailing.trailing_stop

        # Bar with favorable movement (high above entry)
        price_high = 110.0
        price_low = 105.0  # Above stop

        # Handle trailing stop
        closed = trailing_stop_manager.handle_trailing_stop(
            position_manager_with_trailing, sample_timestamp, price_high, price_low
        )

        # Position should NOT be closed
        assert closed is False
        assert position_manager_with_trailing.has_open_position() is True

        # Stop should be tightened (moved up)
        new_stop = position_manager_with_trailing.trailing_stop
        assert new_stop > initial_stop

        # New stop should be 5% below high (110 * 0.95 = 104.5)
        expected_stop = 110.0 * 0.95
        assert new_stop == pytest.approx(expected_stop, abs=0.01)

    def test_handle_trailing_stop_updates_short_when_not_triggered(
        self, trailing_stop_manager, position_manager_with_trailing, sample_timestamp
    ):
        """Test stop updates when not triggered (short position)."""
        # Open short position
        position_manager_with_trailing.open_position(-1, sample_timestamp, 100.0)
        initial_stop = position_manager_with_trailing.trailing_stop

        # Bar with favorable movement (low below entry)
        price_high = 95.0  # Below stop
        price_low = 90.0

        # Handle trailing stop
        closed = trailing_stop_manager.handle_trailing_stop(
            position_manager_with_trailing, sample_timestamp, price_high, price_low
        )

        # Position should NOT be closed
        assert closed is False
        assert position_manager_with_trailing.has_open_position() is True

        # Stop should be tightened (moved down)
        new_stop = position_manager_with_trailing.trailing_stop
        assert new_stop < initial_stop

        # New stop should be 5% above low (90 * 1.05 = 94.5)
        expected_stop = 90.0 * 1.05
        assert new_stop == pytest.approx(expected_stop, abs=0.01)

    def test_handle_trailing_stop_without_open_position(
        self, trailing_stop_manager, position_manager_with_trailing, sample_timestamp
    ):
        """Test handling trailing stop when no position is open."""
        # No position open
        assert position_manager_with_trailing.has_open_position() is False

        # Try to handle trailing stop
        closed = trailing_stop_manager.handle_trailing_stop(
            position_manager_with_trailing, sample_timestamp, 110.0, 90.0
        )

        # Should return False (nothing happened)
        assert closed is False


class TestLookAheadBiasPrevention:
    """Test that look-ahead bias is properly prevented."""

    def test_stop_not_updated_after_trigger_long(
        self, trailing_stop_manager, position_manager_with_trailing, sample_timestamp
    ):
        """Test stop doesn't update after being triggered (long) - prevents look-ahead bias."""
        # Open long at 100 with stop at 95
        position_manager_with_trailing.open_position(1, sample_timestamp, 100.0)
        position_manager_with_trailing.trailing_stop = 95.0

        # Volatile bar: Low hits stop (95), High would calculate better stop (104.5)
        # This is the KEY scenario - we should close at 95, NOT update to 104.5
        price_high = 110.0  # Would calculate stop of 110 * 0.95 = 104.5
        price_low = 95.0  # Triggers current stop

        # Handle stop
        closed = trailing_stop_manager.handle_trailing_stop(
            position_manager_with_trailing, sample_timestamp, price_high, price_low
        )

        # Position should be closed
        assert closed is True

        # Trade should show exit at stop level (95.0), NOT at better calculated level
        trades = position_manager_with_trailing.get_trades()
        assert len(trades) == 1
        assert trades[0]['exit_price'] == 95.0  # Stopped out

        # The stop should NOT have been updated to 104.5 before closing
        # (We can't verify this directly since position is closed, but the exit price proves it)

    def test_stop_not_updated_after_trigger_short(
        self, trailing_stop_manager, position_manager_with_trailing, sample_timestamp
    ):
        """Test stop doesn't update after being triggered (short) - prevents look-ahead bias."""
        # Open short at 100 with stop at 105
        position_manager_with_trailing.open_position(-1, sample_timestamp, 100.0)
        position_manager_with_trailing.trailing_stop = 105.0

        # Volatile bar: High hits stop (105), Low would calculate better stop (94.5)
        price_high = 105.0  # Triggers current stop
        price_low = 90.0  # Would calculate stop of 90 * 1.05 = 94.5

        # Handle stop
        closed = trailing_stop_manager.handle_trailing_stop(
            position_manager_with_trailing, sample_timestamp, price_high, price_low
        )

        # Position should be closed at stop level
        assert closed is True

        trades = position_manager_with_trailing.get_trades()
        assert trades[0]['exit_price'] == 105.0  # Stopped out at original stop


class TestEdgeCases:
    """Test edge cases and unusual scenarios."""

    def test_gap_down_through_stop_long(
        self, trailing_stop_manager, position_manager_with_trailing, sample_timestamp
    ):
        """Test gap down through stop (long) - close at stop level, not gap price."""
        # Long position with stop at 95
        position_manager_with_trailing.open_position(1, sample_timestamp, 100.0)
        position_manager_with_trailing.trailing_stop = 95.0

        # Gap down: Bar opens below stop
        price_high = 92.0
        price_low = 88.0  # Well below stop

        closed = trailing_stop_manager.handle_trailing_stop(
            position_manager_with_trailing, sample_timestamp, price_high, price_low
        )

        assert closed is True

        # Should close at stop level (95), not at low (88)
        trades = position_manager_with_trailing.get_trades()
        assert trades[0]['exit_price'] == 95.0

    def test_gap_up_through_stop_short(
        self, trailing_stop_manager, position_manager_with_trailing, sample_timestamp
    ):
        """Test gap up through stop (short) - close at stop level, not gap price."""
        # Short position with stop at 105
        position_manager_with_trailing.open_position(-1, sample_timestamp, 100.0)
        position_manager_with_trailing.trailing_stop = 105.0

        # Gap up: Bar opens above stop
        price_high = 112.0  # Well above stop
        price_low = 108.0

        closed = trailing_stop_manager.handle_trailing_stop(
            position_manager_with_trailing, sample_timestamp, price_high, price_low
        )

        assert closed is True

        # Should close at stop level (105), not at high (112)
        trades = position_manager_with_trailing.get_trades()
        assert trades[0]['exit_price'] == 105.0

    def test_exact_stop_hit_long(
        self, trailing_stop_manager, position_manager_with_trailing, sample_timestamp
    ):
        """Test bar low exactly equals stop level (long) - should trigger."""
        position_manager_with_trailing.open_position(1, sample_timestamp, 100.0)
        position_manager_with_trailing.trailing_stop = 95.0

        # Bar low exactly at stop
        price_high = 102.0
        price_low = 95.0  # Exactly at stop

        closed = trailing_stop_manager.handle_trailing_stop(
            position_manager_with_trailing, sample_timestamp, price_high, price_low
        )

        # Should trigger (use <= comparison)
        assert closed is True

    def test_exact_stop_hit_short(
        self, trailing_stop_manager, position_manager_with_trailing, sample_timestamp
    ):
        """Test bar high exactly equals stop level (short) - should trigger."""
        position_manager_with_trailing.open_position(-1, sample_timestamp, 100.0)
        position_manager_with_trailing.trailing_stop = 105.0

        # Bar high exactly at stop
        price_high = 105.0  # Exactly at stop
        price_low = 98.0

        closed = trailing_stop_manager.handle_trailing_stop(
            position_manager_with_trailing, sample_timestamp, price_high, price_low
        )

        # Should trigger (use >= comparison)
        assert closed is True

    def test_very_tight_trailing_percentage(
        self, position_manager_with_trailing, sample_timestamp
    ):
        """Test with very tight trailing stop (0.5%)."""
        tsm = TrailingStopManager(trailing_percentage=0.5)

        position_manager_with_trailing.open_position(1, sample_timestamp, 100.0)
        position_manager_with_trailing.trailing_stop = 99.5  # 0.5% below entry

        # Small unfavorable movement triggers stop
        price_high = 100.5
        price_low = 99.4  # Just below stop

        closed = tsm.handle_trailing_stop(
            position_manager_with_trailing, sample_timestamp, price_high, price_low
        )

        assert closed is True

    def test_very_loose_trailing_percentage(
        self, position_manager_with_trailing, sample_timestamp
    ):
        """Test with very loose trailing stop (20%)."""
        tsm = TrailingStopManager(trailing_percentage=20.0)

        position_manager_with_trailing.open_position(1, sample_timestamp, 100.0)
        position_manager_with_trailing.trailing_stop = 80.0  # 20% below entry

        # Large unfavorable movement still doesn't trigger
        price_high = 105.0
        price_low = 85.0  # Above stop

        closed = tsm.handle_trailing_stop(
            position_manager_with_trailing, sample_timestamp, price_high, price_low
        )

        assert closed is False
        assert position_manager_with_trailing.has_open_position() is True


class TestCallbackFunctionality:
    """Test callback function when stop is triggered."""

    def test_callback_called_on_long_stop_trigger(
        self, position_manager_with_trailing, sample_timestamp
    ):
        """Test callback is called when long stop triggers."""
        # Track callback invocations
        callback_data = {'called': False, 'position': None, 'price': None}

        def on_stop(position, price):
            callback_data['called'] = True
            callback_data['position'] = position
            callback_data['price'] = price

        tsm = TrailingStopManager(trailing_percentage=5.0, on_stop_triggered=on_stop)

        # Open long with stop
        position_manager_with_trailing.open_position(1, sample_timestamp, 100.0)
        position_manager_with_trailing.trailing_stop = 95.0

        # Trigger stop
        tsm.handle_trailing_stop(position_manager_with_trailing, sample_timestamp, 100.0, 95.0)

        # Callback should be invoked
        assert callback_data['called'] is True
        assert callback_data['position'] == 1
        assert callback_data['price'] == 95.0

    def test_callback_called_on_short_stop_trigger(
        self, position_manager_with_trailing, sample_timestamp
    ):
        """Test callback is called when short stop triggers."""
        callback_data = {'called': False, 'position': None, 'price': None}

        def on_stop(position, price):
            callback_data['called'] = True
            callback_data['position'] = position
            callback_data['price'] = price

        tsm = TrailingStopManager(trailing_percentage=5.0, on_stop_triggered=on_stop)

        # Open short with stop
        position_manager_with_trailing.open_position(-1, sample_timestamp, 100.0)
        position_manager_with_trailing.trailing_stop = 105.0

        # Trigger stop
        tsm.handle_trailing_stop(position_manager_with_trailing, sample_timestamp, 105.0, 95.0)

        # Callback should be invoked
        assert callback_data['called'] is True
        assert callback_data['position'] == -1
        assert callback_data['price'] == 105.0

    def test_callback_not_called_when_stop_not_triggered(
        self, position_manager_with_trailing, sample_timestamp
    ):
        """Test callback is NOT called when stop doesn't trigger."""
        callback_data = {'called': False}

        def on_stop(position, price):
            callback_data['called'] = True

        tsm = TrailingStopManager(trailing_percentage=5.0, on_stop_triggered=on_stop)

        # Open position
        position_manager_with_trailing.open_position(1, sample_timestamp, 100.0)
        position_manager_with_trailing.trailing_stop = 95.0

        # Don't trigger stop (prices above stop)
        tsm.handle_trailing_stop(position_manager_with_trailing, sample_timestamp, 110.0, 98.0)

        # Callback should NOT be invoked
        assert callback_data['called'] is False

    def test_no_error_when_callback_not_provided(
        self, trailing_stop_manager, position_manager_with_trailing, sample_timestamp
    ):
        """Test no error when callback not provided and stop triggers."""
        # Manager without callback
        assert trailing_stop_manager.on_stop_triggered is None

        # Open position and trigger stop
        position_manager_with_trailing.open_position(1, sample_timestamp, 100.0)
        position_manager_with_trailing.trailing_stop = 95.0

        # Should not raise error
        closed = trailing_stop_manager.handle_trailing_stop(
            position_manager_with_trailing, sample_timestamp, 100.0, 95.0
        )

        assert closed is True


class TestRealisticScenarios:
    """Test realistic trading scenarios."""

    def test_long_trade_with_profitable_trailing_exit(
        self, trailing_stop_manager, position_manager_with_trailing, sample_timestamp
    ):
        """Test complete long trade with profitable trailing stop exit."""
        # Open long at 100
        position_manager_with_trailing.open_position(1, sample_timestamp, 100.0)
        initial_stop = position_manager_with_trailing.trailing_stop  # ~95

        # Bar 1: Price moves up, stop tightens
        trailing_stop_manager.handle_trailing_stop(
            position_manager_with_trailing, sample_timestamp, 110.0, 105.0
        )
        stop_after_bar1 = position_manager_with_trailing.trailing_stop
        assert stop_after_bar1 > initial_stop  # Stop moved up
        assert stop_after_bar1 == pytest.approx(110.0 * 0.95, abs=0.01)

        # Bar 2: Price moves up more, stop tightens again
        trailing_stop_manager.handle_trailing_stop(
            position_manager_with_trailing, sample_timestamp, 115.0, 110.0
        )
        stop_after_bar2 = position_manager_with_trailing.trailing_stop
        assert stop_after_bar2 > stop_after_bar1  # Stop moved up again
        assert stop_after_bar2 == pytest.approx(115.0 * 0.95, abs=0.01)

        # Bar 3: Price drops, trailing stop triggered
        closed = trailing_stop_manager.handle_trailing_stop(
            position_manager_with_trailing, sample_timestamp, 110.0, stop_after_bar2
        )
        assert closed is True

        # Trade should be profitable
        trades = position_manager_with_trailing.get_trades()
        assert len(trades) == 1
        assert trades[0]['exit_price'] > 100.0  # Profitable exit

    def test_short_trade_with_profitable_trailing_exit(
        self, trailing_stop_manager, position_manager_with_trailing, sample_timestamp
    ):
        """Test complete short trade with profitable trailing stop exit."""
        # Open short at 100
        position_manager_with_trailing.open_position(-1, sample_timestamp, 100.0)
        initial_stop = position_manager_with_trailing.trailing_stop  # ~105

        # Bar 1: Price moves down, stop tightens
        trailing_stop_manager.handle_trailing_stop(
            position_manager_with_trailing, sample_timestamp, 95.0, 90.0
        )
        stop_after_bar1 = position_manager_with_trailing.trailing_stop
        assert stop_after_bar1 < initial_stop  # Stop moved down
        assert stop_after_bar1 == pytest.approx(90.0 * 1.05, abs=0.01)

        # Bar 2: Price moves down more, stop tightens again
        trailing_stop_manager.handle_trailing_stop(
            position_manager_with_trailing, sample_timestamp, 88.0, 85.0
        )
        stop_after_bar2 = position_manager_with_trailing.trailing_stop
        assert stop_after_bar2 < stop_after_bar1  # Stop moved down again

        # Bar 3: Price rises, trailing stop triggered
        closed = trailing_stop_manager.handle_trailing_stop(
            position_manager_with_trailing, sample_timestamp, stop_after_bar2, 85.0
        )
        assert closed is True

        # Trade should be profitable
        trades = position_manager_with_trailing.get_trades()
        assert len(trades) == 1
        assert trades[0]['exit_price'] < 100.0  # Profitable exit
