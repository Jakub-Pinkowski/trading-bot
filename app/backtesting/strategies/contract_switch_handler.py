"""
Contract Switch Handler for Backtesting Strategies

This module manages contract rollover logic for futures backtesting.
Handles closing positions at contract expiration and optionally
reopening positions on the new contract.
"""


# ==================== Helper Functions ====================

def prepare_reopen(position_manager):
    """
    Prepare for contract switch by closing position.

    Args:
        position_manager: PositionManager instance

    Returns:
        Previous position direction (for potential reopening)
    """
    # This function is called before the actual switch handling
    # It returns the position direction for potential reopening
    return position_manager.position


def execute_queued_signal(queued_signal, position_manager, idx, price_open):
    """
    Execute a queued signal from the previous bar.

    Args:
        queued_signal: Signal to execute (1 for long, -1 for short, None for no signal)
        position_manager: PositionManager instance
        idx: Current bar index
        price_open: Opening price of current bar

    Returns:
        bool: True if signal was executed
    """
    if queued_signal is not None:
        flip = None
        if queued_signal == 1 and position_manager.position != 1:
            flip = 1
        elif queued_signal == -1 and position_manager.position != -1:
            flip = -1

        if flip is not None:
            # Close if currently in position
            if position_manager.has_open_position():
                position_manager.close_position(idx, price_open, switch=False)
            # Open a new position at this (current) bar
            position_manager.open_position(flip, idx, price_open)
            return True

    return False


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
        return self.next_switch and current_time >= self.next_switch

    def handle_contract_switch(self, current_time, position_manager, idx, price_open):
        """
        Handle contract switch logic.

        Args:
            current_time: Current datetime
            position_manager: PositionManager instance
            idx: Current bar index
            price_open: Opening price of current bar

        Returns:
            bool: True if signal should be skipped this bar
        """
        # Check and process all switch dates that have been reached
        while self.should_switch(current_time):
            # On rollover date close at the price of *last bar before switch*
            if position_manager.has_open_position():
                prev_position = prepare_reopen(position_manager)

                # If rollover is enabled, mark for reopening
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
            if self.rollover:
                direction = self.must_reopen
                position_manager.open_position(direction, idx, price_open)
            self.must_reopen = None

        # Return whether signal should be skipped
        should_skip = self.skip_signal_this_bar
        if self.skip_signal_this_bar:
            self.skip_signal_this_bar = False  # skip *this* bar only
        return should_skip
