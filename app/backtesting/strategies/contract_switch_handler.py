"""
Contract Switch Handler for Backtesting Strategies

This module manages contract rollover logic for futures backtesting.
Handles closing positions at contract expiration and optionally
reopening positions on the new contract.
"""


class ContractSwitchHandler:
    """Manages contract rollover logic for futures backtesting."""

    # ==================== Initialization ====================

    def __init__(self, switch_dates=None, rollover=False):
        """
        Initialize the contract switch handler.

        Args:
            switch_dates: List of contract rollover dates
            rollover: Whether to reopen positions on new contracts
        """
        self.switch_dates = switch_dates
        self.rollover = rollover
        self.next_switch_idx = 0
        self.next_switch = None
        self.must_reopen = None
        self.skip_signal_this_bar = False

    # ==================== Public Methods ====================

    def set_switch_dates(self, switch_dates):
        """Set or update the contract switch dates"""
        self.switch_dates = switch_dates
        self.next_switch_idx = 0
        self.next_switch = switch_dates[0] if switch_dates else None

    def reset(self):
        """Reset handler state"""
        self.next_switch_idx = 0
        self.next_switch = self.switch_dates[0] if self.switch_dates else None
        self.must_reopen = None
        self.skip_signal_this_bar = False

    def should_switch(self, current_time):
        """
        Check if we should switch contracts at the current time.

        Args:
            current_time: Current datetime

        Returns:
            bool: True if switch should occur
        """
        return self.next_switch is not None and current_time >= self.next_switch

    def handle_contract_switch(self, current_time, position_manager, idx, price_open, prev_time=None, prev_row=None):
        """
        Handle contract switch logic including closing positions at switch and reopening.

        Args:
            current_time: Current datetime
            position_manager: PositionManager instance
            idx: Current bar index
            price_open: Opening price of current bar
            prev_time: Previous bar datetime (needed for closing at switch)
            prev_row: Previous bar data (needed for closing at switch)

        Returns:
            bool: True if signal should be skipped this bar
        """
        # Check and process all switch dates that have been reached
        while self.next_switch and current_time >= self.next_switch:
            # Close position at the previous bar's data when switching contracts
            if position_manager.has_open_position() and prev_row is not None:
                prev_position = position_manager.close_position_at_switch(prev_time, prev_row)

                # Mark for reopening if rollover is enabled
                if self.rollover:
                    self.must_reopen = prev_position
                    self.skip_signal_this_bar = True

            # Move to next switch date
            self.next_switch_idx += 1
            self.next_switch = self.switch_dates[self.next_switch_idx] if self.next_switch_idx < len(
                self.switch_dates
            ) else None

        # Reopen on the following contract if rollover is enabled
        if self.must_reopen is not None and not position_manager.has_open_position():
            position_manager.open_position(self.must_reopen, idx, price_open)
            self.must_reopen = None

        # Return whether signal should be skipped
        should_skip = self.skip_signal_this_bar
        if self.skip_signal_this_bar:
            self.skip_signal_this_bar = False  # skip *this* bar only
        return should_skip
