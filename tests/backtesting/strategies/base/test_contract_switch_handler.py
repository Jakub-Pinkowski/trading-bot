"""
Tests for ContractSwitchHandler.

Tests cover:
- Initialization and configuration
- Contract switch detection and timing
- Position closing at contract expiration
- Position reopening with rollover enabled
- Signal skipping during switch
- Multiple switch dates handling
- Integration with PositionManager
- Edge cases (no switches, multiple switches, gaps)
"""
import pandas as pd
import pytest

from app.backtesting.strategies.base.contract_switch_handler import ContractSwitchHandler
from app.backtesting.strategies.base.position_manager import PositionManager


# ==================== Fixtures ====================

@pytest.fixture
def switch_handler_no_rollover():
    """Contract switch handler without rollover (closes positions only)."""
    return ContractSwitchHandler(switch_dates=[], rollover=False)


@pytest.fixture
def switch_handler_with_rollover():
    """Contract switch handler with rollover enabled (reopens positions)."""
    return ContractSwitchHandler(switch_dates=[], rollover=True)


@pytest.fixture
def position_manager():
    """Position manager for testing."""
    return PositionManager(
        slippage_ticks=0,
        symbol='ZS',
        trailing=None
    )


@pytest.fixture
def sample_timestamps():
    """Sample timestamps for testing switch scenarios."""
    return [
        pd.Timestamp('2025-01-01 10:00:00'),
        pd.Timestamp('2025-01-01 11:00:00'),
        pd.Timestamp('2025-01-01 12:00:00'),
        pd.Timestamp('2025-01-01 13:00:00'),
        pd.Timestamp('2025-01-01 14:00:00'),
    ]


@pytest.fixture
def sample_row():
    """Sample OHLCV row for testing."""
    return pd.Series({
        'open': 100.0,
        'high': 102.0,
        'low': 98.0,
        'close': 101.0,
        'volume': 1000
    })


# ==================== Test Classes ====================

class TestContractSwitchHandlerInitialization:
    """Test ContractSwitchHandler initialization."""

    @pytest.mark.parametrize("switch_dates,rollover,expected_dates,expected_rollover", [
        (
                [pd.Timestamp('2025-03-01'), pd.Timestamp('2025-06-01')], False,
                [pd.Timestamp('2025-03-01'), pd.Timestamp('2025-06-01')], False
        ),
        ([pd.Timestamp('2025-03-01')], True, [pd.Timestamp('2025-03-01')], True),
        ([], False, [], False),
        (None, False, None, False),
    ])
    def test_initialization_with_various_configs(
        self, switch_dates, rollover, expected_dates, expected_rollover
    ):
        """Test handler initializes correctly with various configurations."""
        handler = ContractSwitchHandler(switch_dates=switch_dates, rollover=rollover)

        assert handler.switch_dates == expected_dates
        assert handler.rollover == expected_rollover
        assert handler.next_switch_idx == 0
        assert handler.next_switch is None  # Not set until set_switch_dates() or reset()
        assert handler.must_reopen is None
        assert handler.skip_signal_this_bar is False


class TestSetSwitchDates:
    """Test setting and updating switch dates."""

    def test_set_switch_dates_initializes_first_switch(self, switch_handler_no_rollover):
        """Test setting switch dates initializes first switch correctly."""
        switch_dates = [pd.Timestamp('2025-03-01'), pd.Timestamp('2025-06-01')]
        switch_handler_no_rollover.set_switch_dates(switch_dates)

        assert switch_handler_no_rollover.switch_dates == switch_dates
        assert switch_handler_no_rollover.next_switch == switch_dates[0]
        assert switch_handler_no_rollover.next_switch_idx == 0

    def test_set_switch_dates_resets_index(self):
        """Test setting new switch dates resets the index."""
        handler = ContractSwitchHandler(
            switch_dates=[pd.Timestamp('2025-01-01')],
            rollover=False
        )

        # Simulate progressing through switch
        handler.next_switch_idx = 1

        # Set new dates
        new_dates = [pd.Timestamp('2025-06-01')]
        handler.set_switch_dates(new_dates)

        # Index should be reset
        assert handler.next_switch_idx == 0
        assert handler.next_switch == new_dates[0]

    def test_set_empty_switch_dates(self, switch_handler_no_rollover):
        """Test setting empty switch dates."""
        switch_handler_no_rollover.set_switch_dates([])

        assert switch_handler_no_rollover.switch_dates == []
        assert switch_handler_no_rollover.next_switch is None


class TestReset:
    """Test reset functionality."""

    def test_reset_restores_initial_state(self):
        """Test reset restores handler to initial state."""
        switch_dates = [pd.Timestamp('2025-03-01'), pd.Timestamp('2025-06-01')]
        handler = ContractSwitchHandler(switch_dates=switch_dates, rollover=True)

        # Modify state
        handler.next_switch_idx = 1
        handler.next_switch = switch_dates[1]
        handler.must_reopen = 1
        handler.skip_signal_this_bar = True

        # Reset
        handler.reset()

        # State should be restored
        assert handler.next_switch_idx == 0
        assert handler.next_switch == switch_dates[0]
        assert handler.must_reopen is None
        assert handler.skip_signal_this_bar is False

    def test_reset_with_empty_switch_dates(self):
        """Test reset with empty switch dates."""
        handler = ContractSwitchHandler(switch_dates=[], rollover=False)
        handler.must_reopen = 1

        handler.reset()

        assert handler.next_switch_idx == 0
        assert handler.next_switch is None
        assert handler.must_reopen is None


class TestContractSwitchDetection:
    """Test contract switch detection and timing."""

    def test_switch_detected_when_time_equals_switch_date(
        self, switch_handler_no_rollover, position_manager, sample_timestamps, sample_row
    ):
        """Test switch is detected when current time equals switch date."""
        switch_date = sample_timestamps[2]
        switch_handler_no_rollover.set_switch_dates([switch_date])

        # Open position before switch
        position_manager.open_position(1, sample_timestamps[0], 100.0)

        # Process bars before switch
        current_time = sample_timestamps[1]
        prev_time = sample_timestamps[0]
        should_skip = switch_handler_no_rollover.handle_contract_switch(
            current_time, position_manager, current_time, 100.0, prev_time, sample_row
        )

        # Switch not triggered yet
        assert should_skip is False
        assert position_manager.has_open_position() is True

        # Process bar at switch date
        current_time = sample_timestamps[2]
        prev_time = sample_timestamps[1]
        should_skip = switch_handler_no_rollover.handle_contract_switch(
            current_time, position_manager, current_time, 100.0, prev_time, sample_row
        )

        # Position should be closed
        assert position_manager.has_open_position() is False
        assert len(position_manager.get_trades()) == 1

    def test_switch_detected_when_time_after_switch_date(
        self, switch_handler_no_rollover, position_manager, sample_timestamps, sample_row
    ):
        """Test switch is detected when current time is after switch date."""
        switch_date = sample_timestamps[1]
        switch_handler_no_rollover.set_switch_dates([switch_date])

        # Open position
        position_manager.open_position(1, sample_timestamps[0], 100.0)

        # Jump to time after switch date
        current_time = sample_timestamps[3]  # After switch
        prev_time = sample_timestamps[0]
        switch_handler_no_rollover.handle_contract_switch(
            current_time, position_manager, current_time, 100.0, prev_time, sample_row
        )

        # Position should be closed
        assert position_manager.has_open_position() is False

    def test_no_switch_when_time_before_switch_date(
        self, switch_handler_no_rollover, position_manager, sample_timestamps, sample_row
    ):
        """Test no switch occurs when current time is before switch date."""
        switch_date = sample_timestamps[3]
        switch_handler_no_rollover.set_switch_dates([switch_date])

        # Open position
        position_manager.open_position(1, sample_timestamps[0], 100.0)

        # Process bar before switch
        current_time = sample_timestamps[1]
        prev_time = sample_timestamps[0]
        switch_handler_no_rollover.handle_contract_switch(
            current_time, position_manager, current_time, 100.0, prev_time, sample_row
        )

        # Position should still be open
        assert position_manager.has_open_position() is True
        assert len(position_manager.get_trades()) == 0


class TestPositionClosingAtSwitch:
    """Test position closing behavior at contract switch."""

    def test_long_position_closed_at_switch(
        self, switch_handler_no_rollover, position_manager, sample_timestamps, sample_row
    ):
        """Test long position is closed at contract switch."""
        switch_date = sample_timestamps[2]
        switch_handler_no_rollover.set_switch_dates([switch_date])

        # Open long position
        position_manager.open_position(1, sample_timestamps[0], 100.0)
        entry_price = position_manager.entry_price

        # Trigger switch
        switch_handler_no_rollover.handle_contract_switch(
            sample_timestamps[2], position_manager, sample_timestamps[2],
            105.0, sample_timestamps[1], sample_row
        )

        # Position should be closed
        assert position_manager.has_open_position() is False

        # Trade should be recorded with switch flag
        trades = position_manager.get_trades()
        assert len(trades) == 1
        assert trades[0]['side'] == 'long'
        assert trades[0]['entry_price'] == entry_price
        assert trades[0]['switch'] is True

    def test_short_position_closed_at_switch(
        self, switch_handler_no_rollover, position_manager, sample_timestamps, sample_row
    ):
        """Test short position is closed at contract switch."""
        switch_date = sample_timestamps[2]
        switch_handler_no_rollover.set_switch_dates([switch_date])

        # Open short position
        position_manager.open_position(-1, sample_timestamps[0], 100.0)

        # Trigger switch
        switch_handler_no_rollover.handle_contract_switch(
            sample_timestamps[2], position_manager, sample_timestamps[2],
            95.0, sample_timestamps[1], sample_row
        )

        # Position should be closed
        assert position_manager.has_open_position() is False

        # Trade should have switch flag
        trades = position_manager.get_trades()
        assert trades[0]['side'] == 'short'
        assert trades[0]['switch'] is True

    def test_position_closed_at_previous_bar(
        self, switch_handler_no_rollover, position_manager, sample_timestamps
    ):
        """Test position is closed at previous bar's data (conservative approach)."""
        switch_date = sample_timestamps[2]
        switch_handler_no_rollover.set_switch_dates([switch_date])

        # Open position
        position_manager.open_position(1, sample_timestamps[0], 100.0)

        # Previous bar data
        prev_row = pd.Series({
            'open': 105.0,
            'high': 107.0,
            'low': 104.0,
            'close': 106.0,
            'volume': 1000
        })

        # Trigger switch
        switch_handler_no_rollover.handle_contract_switch(
            sample_timestamps[2], position_manager, sample_timestamps[2],
            110.0, sample_timestamps[1], prev_row
        )

        # Exit should use previous bar's open price
        trades = position_manager.get_trades()
        assert trades[0]['exit_time'] == sample_timestamps[1]
        assert trades[0]['exit_price'] == 105.0  # prev_row['open']

    def test_no_closure_if_no_open_position(
        self, switch_handler_no_rollover, position_manager, sample_timestamps, sample_row
    ):
        """Test no closure occurs if no position is open at switch."""
        switch_date = sample_timestamps[2]
        switch_handler_no_rollover.set_switch_dates([switch_date])

        # No position open
        assert position_manager.has_open_position() is False

        # Trigger switch
        switch_handler_no_rollover.handle_contract_switch(
            sample_timestamps[2], position_manager, sample_timestamps[2],
            100.0, sample_timestamps[1], sample_row
        )

        # No trades should be created
        assert len(position_manager.get_trades()) == 0


class TestRolloverBehavior:
    """Test rollover behavior (reopening positions on new contract)."""

    def test_long_position_reopened_with_rollover(
        self, switch_handler_with_rollover, position_manager, sample_timestamps, sample_row
    ):
        """Test long position is reopened after switch with rollover enabled."""
        switch_date = sample_timestamps[2]
        switch_handler_with_rollover.set_switch_dates([switch_date])

        # Open long position
        position_manager.open_position(1, sample_timestamps[0], 100.0)

        # Trigger switch (closes position and immediately reopens it)
        switch_handler_with_rollover.handle_contract_switch(
            sample_timestamps[2], position_manager, sample_timestamps[2],
            105.0, sample_timestamps[1], sample_row
        )

        # Position should be reopened immediately on the same bar
        assert position_manager.has_open_position() is True
        assert position_manager.position == 1  # Long
        assert position_manager.entry_price == 105.0  # Reopened at current bar's open
        assert switch_handler_with_rollover.must_reopen is None  # Cleared after reopening

        # Should have one trade from the switch close
        trades = position_manager.get_trades()
        assert len(trades) == 1
        assert trades[0]['switch'] is True

    def test_short_position_reopened_with_rollover(
        self, switch_handler_with_rollover, position_manager, sample_timestamps, sample_row
    ):
        """Test short position is reopened after switch with rollover enabled."""
        switch_date = sample_timestamps[2]
        switch_handler_with_rollover.set_switch_dates([switch_date])

        # Open short position
        position_manager.open_position(-1, sample_timestamps[0], 100.0)

        # Trigger switch (closes and reopens immediately)
        switch_handler_with_rollover.handle_contract_switch(
            sample_timestamps[2], position_manager, sample_timestamps[2],
            95.0, sample_timestamps[1], sample_row
        )

        # Position should be reopened immediately
        assert position_manager.has_open_position() is True
        assert position_manager.position == -1  # Short
        assert position_manager.entry_price == 95.0
        assert switch_handler_with_rollover.must_reopen is None  # Cleared

    def test_position_not_reopened_without_rollover(
        self, switch_handler_no_rollover, position_manager, sample_timestamps, sample_row
    ):
        """Test position is NOT reopened when rollover is disabled."""
        switch_date = sample_timestamps[2]
        switch_handler_no_rollover.set_switch_dates([switch_date])

        # Open position
        position_manager.open_position(1, sample_timestamps[0], 100.0)

        # Trigger switch
        switch_handler_no_rollover.handle_contract_switch(
            sample_timestamps[2], position_manager, sample_timestamps[2],
            105.0, sample_timestamps[1], sample_row
        )

        # Should NOT be marked for reopening
        assert switch_handler_no_rollover.must_reopen is None

        # Next bar: Position should remain closed
        switch_handler_no_rollover.handle_contract_switch(
            sample_timestamps[3], position_manager, sample_timestamps[3],
            106.0, sample_timestamps[2], sample_row
        )

        assert position_manager.has_open_position() is False

    def test_must_reopen_cleared_after_reopening(
        self, switch_handler_with_rollover, position_manager, sample_timestamps, sample_row
    ):
        """Test must_reopen flag is cleared after position is reopened."""
        switch_date = sample_timestamps[2]
        switch_handler_with_rollover.set_switch_dates([switch_date])

        # Open and trigger switch
        position_manager.open_position(1, sample_timestamps[0], 100.0)
        switch_handler_with_rollover.handle_contract_switch(
            sample_timestamps[2], position_manager, sample_timestamps[2],
            105.0, sample_timestamps[1], sample_row
        )

        # Reopen on next bar
        switch_handler_with_rollover.handle_contract_switch(
            sample_timestamps[3], position_manager, sample_timestamps[3],
            106.0, sample_timestamps[2], sample_row
        )

        # must_reopen should be cleared
        assert switch_handler_with_rollover.must_reopen is None


class TestSignalSkipping:
    """Test signal skipping during contract switch."""

    def test_signal_skipped_on_switch_bar_with_rollover(
        self, switch_handler_with_rollover, position_manager, sample_timestamps, sample_row
    ):
        """Test signal is skipped on the switch bar when rollover is enabled."""
        switch_date = sample_timestamps[2]
        switch_handler_with_rollover.set_switch_dates([switch_date])

        # Open position
        position_manager.open_position(1, sample_timestamps[0], 100.0)

        # Trigger switch
        should_skip = switch_handler_with_rollover.handle_contract_switch(
            sample_timestamps[2], position_manager, sample_timestamps[2],
            105.0, sample_timestamps[1], sample_row
        )

        # Signal should be skipped on switch bar
        assert should_skip is True

    def test_signal_not_skipped_after_switch_bar(
        self, switch_handler_with_rollover, position_manager, sample_timestamps, sample_row
    ):
        """Test signal is not skipped after switch bar."""
        switch_date = sample_timestamps[2]
        switch_handler_with_rollover.set_switch_dates([switch_date])

        # Open and trigger switch
        position_manager.open_position(1, sample_timestamps[0], 100.0)
        switch_handler_with_rollover.handle_contract_switch(
            sample_timestamps[2], position_manager, sample_timestamps[2],
            105.0, sample_timestamps[1], sample_row
        )

        # Next bar: Signal should NOT be skipped
        should_skip = switch_handler_with_rollover.handle_contract_switch(
            sample_timestamps[3], position_manager, sample_timestamps[3],
            106.0, sample_timestamps[2], sample_row
        )

        assert should_skip is False

    def test_signal_not_skipped_without_rollover(
        self, switch_handler_no_rollover, position_manager, sample_timestamps, sample_row
    ):
        """Test signal is not skipped when rollover is disabled."""
        switch_date = sample_timestamps[2]
        switch_handler_no_rollover.set_switch_dates([switch_date])

        # Open position
        position_manager.open_position(1, sample_timestamps[0], 100.0)

        # Trigger switch
        should_skip = switch_handler_no_rollover.handle_contract_switch(
            sample_timestamps[2], position_manager, sample_timestamps[2],
            105.0, sample_timestamps[1], sample_row
        )

        # Signal should NOT be skipped (no rollover)
        assert should_skip is False

    def test_skip_flag_cleared_after_one_bar(
        self, switch_handler_with_rollover, position_manager, sample_timestamps, sample_row
    ):
        """Test skip flag is only active for one bar."""
        switch_date = sample_timestamps[2]
        switch_handler_with_rollover.set_switch_dates([switch_date])

        # Open and switch
        position_manager.open_position(1, sample_timestamps[0], 100.0)
        should_skip1 = switch_handler_with_rollover.handle_contract_switch(
            sample_timestamps[2], position_manager, sample_timestamps[2],
            105.0, sample_timestamps[1], sample_row
        )
        assert should_skip1 is True

        # Next bar
        should_skip2 = switch_handler_with_rollover.handle_contract_switch(
            sample_timestamps[3], position_manager, sample_timestamps[3],
            106.0, sample_timestamps[2], sample_row
        )
        assert should_skip2 is False

        # Bar after that
        should_skip3 = switch_handler_with_rollover.handle_contract_switch(
            sample_timestamps[4], position_manager, sample_timestamps[4],
            107.0, sample_timestamps[3], sample_row
        )
        assert should_skip3 is False


class TestMultipleSwitchDates:
    """Test handling multiple contract switches."""

    def test_multiple_switches_processed_sequentially(
        self, switch_handler_no_rollover, position_manager, sample_row
    ):
        """Test multiple switch dates are processed in sequence."""
        timestamps = [pd.Timestamp(f'2025-0{i + 1}-01') for i in range(5)]
        switch_dates = [timestamps[1], timestamps[3]]  # Two switches
        switch_handler_no_rollover.set_switch_dates(switch_dates)

        # Open position before first switch
        position_manager.open_position(1, timestamps[0], 100.0)

        # First switch
        switch_handler_no_rollover.handle_contract_switch(
            timestamps[1], position_manager, timestamps[1],
            105.0, timestamps[0], sample_row
        )
        assert position_manager.has_open_position() is False
        assert len(position_manager.get_trades()) == 1

        # Open new position
        position_manager.open_position(1, timestamps[2], 105.0)

        # Second switch
        switch_handler_no_rollover.handle_contract_switch(
            timestamps[3], position_manager, timestamps[3],
            110.0, timestamps[2], sample_row
        )
        assert position_manager.has_open_position() is False
        assert len(position_manager.get_trades()) == 2

    def test_switch_index_advances_correctly(
        self, switch_handler_no_rollover, position_manager, sample_row
    ):
        """Test switch index advances correctly through multiple switches."""
        timestamps = [pd.Timestamp(f'2025-0{i + 1}-01') for i in range(4)]
        switch_dates = [timestamps[1], timestamps[2], timestamps[3]]
        switch_handler_no_rollover.set_switch_dates(switch_dates)

        # Initial state
        assert switch_handler_no_rollover.next_switch_idx == 0
        assert switch_handler_no_rollover.next_switch == switch_dates[0]

        # Process first switch
        position_manager.open_position(1, timestamps[0], 100.0)
        switch_handler_no_rollover.handle_contract_switch(
            timestamps[1], position_manager, timestamps[1],
            100.0, timestamps[0], sample_row
        )
        assert switch_handler_no_rollover.next_switch_idx == 1
        assert switch_handler_no_rollover.next_switch == switch_dates[1]

        # Process second switch
        position_manager.open_position(1, timestamps[1], 100.0)
        switch_handler_no_rollover.handle_contract_switch(
            timestamps[2], position_manager, timestamps[2],
            100.0, timestamps[1], sample_row
        )
        assert switch_handler_no_rollover.next_switch_idx == 2
        assert switch_handler_no_rollover.next_switch == switch_dates[2]

        # Process third switch
        position_manager.open_position(1, timestamps[2], 100.0)
        switch_handler_no_rollover.handle_contract_switch(
            timestamps[3], position_manager, timestamps[3],
            100.0, timestamps[2], sample_row
        )
        assert switch_handler_no_rollover.next_switch_idx == 3
        assert switch_handler_no_rollover.next_switch is None  # No more switches

    def test_all_switches_in_single_call(
        self, switch_handler_no_rollover, position_manager, sample_row
    ):
        """Test multiple switches can be processed in single call (gap scenario)."""
        timestamps = [pd.Timestamp(f'2025-0{i + 1}-01') for i in range(5)]
        switch_dates = [timestamps[1], timestamps[2]]
        switch_handler_no_rollover.set_switch_dates(switch_dates)

        # Open position
        position_manager.open_position(1, timestamps[0], 100.0)

        # Jump to time after both switches
        switch_handler_no_rollover.handle_contract_switch(
            timestamps[4], position_manager, timestamps[4],
            100.0, timestamps[0], sample_row
        )

        # Should have processed both switches (closed position)
        assert position_manager.has_open_position() is False
        assert switch_handler_no_rollover.next_switch is None
        assert switch_handler_no_rollover.next_switch_idx == 2


class TestEdgeCases:
    """Test edge cases and unusual scenarios."""

    def test_no_switches_defined(
        self, switch_handler_no_rollover, position_manager, sample_timestamps, sample_row
    ):
        """Test handler works correctly with no switch dates defined."""
        switch_handler_no_rollover.set_switch_dates([])

        # Open position
        position_manager.open_position(1, sample_timestamps[0], 100.0)

        # Process bars (no switches should occur)
        for i in range(1, len(sample_timestamps)):
            should_skip = switch_handler_no_rollover.handle_contract_switch(
                sample_timestamps[i], position_manager, sample_timestamps[i],
                100.0, sample_timestamps[i - 1], sample_row
            )
            assert should_skip is False

        # Position should remain open
        assert position_manager.has_open_position() is True

    def test_switch_without_previous_bar_data(
        self, switch_handler_no_rollover, position_manager, sample_timestamps
    ):
        """Test switch handling when prev_row is None."""
        switch_date = sample_timestamps[1]
        switch_handler_no_rollover.set_switch_dates([switch_date])

        # Open position
        position_manager.open_position(1, sample_timestamps[0], 100.0)

        # Trigger switch with None prev_row
        switch_handler_no_rollover.handle_contract_switch(
            sample_timestamps[1], position_manager, sample_timestamps[1],
            100.0, None, None  # prev_row is None
        )

        # Position should remain open (can't close without prev_row)
        assert position_manager.has_open_position() is True

    def test_switch_at_very_first_bar(
        self, switch_handler_no_rollover, position_manager, sample_timestamps, sample_row
    ):
        """Test switch occurring at the very first bar."""
        switch_date = sample_timestamps[0]
        switch_handler_no_rollover.set_switch_dates([switch_date])

        # No previous bar data (first bar)
        switch_handler_no_rollover.handle_contract_switch(
            sample_timestamps[0], position_manager, sample_timestamps[0],
            100.0, None, None
        )

        # Should handle gracefully (no position to close)
        assert switch_handler_no_rollover.next_switch_idx == 1

    def test_rollover_with_alternating_positions(
        self, switch_handler_with_rollover, position_manager, sample_row
    ):
        """Test rollover correctly handles alternating long/short positions."""
        timestamps = [pd.Timestamp(f'2025-0{i + 1}-01') for i in range(4)]
        switch_dates = [timestamps[1], timestamps[3]]
        switch_handler_with_rollover.set_switch_dates(switch_dates)

        # Long position -> switch (closes and reopens as long immediately)
        position_manager.open_position(1, timestamps[0], 100.0)
        switch_handler_with_rollover.handle_contract_switch(
            timestamps[1], position_manager, timestamps[1],
            105.0, timestamps[0], sample_row
        )
        # Position should be reopened as long
        assert position_manager.has_open_position() is True
        assert position_manager.position == 1

        # Manually close and flip to short
        position_manager.close_position(timestamps[2], 106.0, switch=False)
        position_manager.open_position(-1, timestamps[2], 106.0)

        # Short position -> switch (closes and reopens as short)
        switch_handler_with_rollover.handle_contract_switch(
            timestamps[3], position_manager, timestamps[3],
            100.0, timestamps[2], sample_row
        )
        # Position should be reopened as short
        assert position_manager.has_open_position() is True
        assert position_manager.position == -1


class TestRealisticScenarios:
    """Test realistic trading scenarios."""

    def test_quarterly_rollover_scenario(
        self, switch_handler_with_rollover, position_manager, sample_row
    ):
        """Test realistic quarterly contract rollover scenario."""
        # Quarterly switch dates (Mar, Jun, Sep, Dec)
        switch_dates = [
            pd.Timestamp('2025-03-15'),
            pd.Timestamp('2025-06-15'),
            pd.Timestamp('2025-09-15'),
            pd.Timestamp('2025-12-15')
        ]
        switch_handler_with_rollover.set_switch_dates(switch_dates)

        # Trading period covering two rollovers
        timestamps = [
            pd.Timestamp('2025-03-01'),
            pd.Timestamp('2025-03-15'),  # First rollover
            pd.Timestamp('2025-06-15'),  # Second rollover
            pd.Timestamp('2025-06-16'),
        ]

        # Open position before first rollover
        position_manager.open_position(1, timestamps[0], 1000.0)

        # First rollover (closes and immediately reopens)
        switch_handler_with_rollover.handle_contract_switch(
            timestamps[1], position_manager, timestamps[1],
            1050.0, timestamps[0], sample_row
        )
        assert position_manager.has_open_position() is True
        assert position_manager.entry_price == 1050.0
        assert len(position_manager.get_trades()) == 1
        assert position_manager.get_trades()[0]['switch'] is True

        # Second rollover (closes and immediately reopens)
        switch_handler_with_rollover.handle_contract_switch(
            timestamps[2], position_manager, timestamps[2],
            1100.0, timestamps[1], sample_row
        )
        assert position_manager.has_open_position() is True
        assert position_manager.entry_price == 1100.0
        assert len(position_manager.get_trades()) == 2

    def test_no_rollover_closes_permanently(
        self, switch_handler_no_rollover, position_manager, sample_row
    ):
        """Test position stays closed after switch without rollover."""
        switch_date = pd.Timestamp('2025-03-15')
        switch_handler_no_rollover.set_switch_dates([switch_date])

        timestamps = [
            pd.Timestamp('2025-03-01'),
            pd.Timestamp('2025-03-15'),
            pd.Timestamp('2025-03-16'),
            pd.Timestamp('2025-03-17'),
        ]

        # Open and hold through switch
        position_manager.open_position(1, timestamps[0], 100.0)

        # Switch closes position
        switch_handler_no_rollover.handle_contract_switch(
            timestamps[1], position_manager, timestamps[1],
            105.0, timestamps[0], sample_row
        )
        assert position_manager.has_open_position() is False

        # Process subsequent bars - position should remain closed
        for i in range(2, len(timestamps)):
            switch_handler_no_rollover.handle_contract_switch(
                timestamps[i], position_manager, timestamps[i],
                110.0, timestamps[i - 1], sample_row
            )
            assert position_manager.has_open_position() is False

        # Only one trade (the switch close)
        assert len(position_manager.get_trades()) == 1
