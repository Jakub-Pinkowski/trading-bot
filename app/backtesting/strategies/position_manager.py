"""
Position Manager for Backtesting Strategies

This module manages position state and lifecycle during backtesting, including:
- Opening and closing positions
- Position state tracking
- Slippage calculations for entry and exit prices
"""


class PositionManager:
    """Manages position state and lifecycle during backtesting."""

    # ==================== Initialization ====================

    def __init__(self, slippage=0, symbol=None, trailing=None):
        """
        Initialize the position manager.

        Args:
            slippage: Slippage percentage (e.g., 0.05 = 0.05%)
            symbol: The futures symbol (e.g., 'ZC', 'GC')
            trailing: Trailing stop percentage (if used)
        """
        self.slippage = slippage
        self.symbol = symbol
        self.trailing = trailing

        # Initialize position state
        self.position = None
        self.entry_time = None
        self.entry_price = None
        self.trailing_stop = None
        self.trades = []

    # ==================== State Management ====================

    def reset(self):
        """Reset all state variables"""
        self.position = None
        self.entry_time = None
        self.entry_price = None
        self.trailing_stop = None
        self.trades = []

    def reset_position(self):
        """Reset position variables"""
        self.entry_time = None
        self.entry_price = None
        self.position = None
        self.trailing_stop = None

    def has_open_position(self):
        """Check if there is an open position"""
        return self.position is not None

    def get_position_details(self):
        """Get current position details"""
        return {
            'position': self.position,
            'entry_time': self.entry_time,
            'entry_price': self.entry_price,
            'trailing_stop': self.trailing_stop
        }

    def get_trades(self):
        """Get list of completed trades"""
        return self.trades

    # ==================== Slippage Calculations ====================

    def apply_slippage_to_entry_price(self, direction, price):
        """Apply slippage to entry price based on a position direction"""
        if direction == 1:  # Long position
            # For long positions, pay more on entry (higher price)
            adjusted_price = price * (1 + self.slippage / 100)
        else:  # Short position
            # For short positions, receive less on entry (lower price)
            adjusted_price = price * (1 - self.slippage / 100)

        return round(adjusted_price, 2)

    def apply_slippage_to_exit_price(self, direction, price):
        """Apply slippage to exit price based on a position direction"""
        if direction == 1:  # Long position
            # For long positions, receive less on exit (lower price)
            adjusted_price = price * (1 - self.slippage / 100)
        else:  # Short position
            # For short positions, pay more on exit (higher price)
            adjusted_price = price * (1 + self.slippage / 100)

        return round(adjusted_price, 2)

    # ==================== Position Operations ====================

    def open_position(self, direction, idx, price_open):
        """
        Open a new position.

        Args:
            direction: 1 for long, -1 for short
            idx: Entry time index
            price_open: Opening price
        """
        self.position = direction
        self.entry_time = idx

        # Apply slippage to entry price
        self.entry_price = self.apply_slippage_to_entry_price(direction, price_open)

        # Set initial trailing stop if trailing is enabled
        if self.trailing is not None:
            # Calculate initial trailing stop based on entry price
            if direction == 1:  # Long position
                self.trailing_stop = round(self.entry_price * (1 - self.trailing / 100), 2)
            else:  # Short position
                self.trailing_stop = round(self.entry_price * (1 + self.trailing / 100), 2)

    def close_position(self, exit_time, exit_price, switch=False):
        """
        Close the current position and record the trade.

        Args:
            exit_time: Exit time index
            exit_price: Exit price
            switch: Whether this close is due to contract switch
        """
        # Apply slippage to exit price
        adjusted_exit_price = self.apply_slippage_to_exit_price(self.position, exit_price)

        trade = {
            'entry_time': self.entry_time,
            'entry_price': self.entry_price,
            'exit_time': exit_time,
            'exit_price': adjusted_exit_price,
            'side': 'long' if self.position == 1 else 'short',
        }
        if switch:
            trade['switch'] = True
        self.trades.append(trade)
        self.reset_position()

    def close_position_at_switch(self, prev_time, prev_row):
        """
        Close position at contract switch.

        Args:
            prev_time: Time of the last bar before switch
            prev_row: Data row of the last bar before switch

        Returns:
            Previous position direction (for potential reopening)
        """
        # Exit at OPEN of last bar before switch (conservative assumption)
        # This assumes we exit early rather than waiting until close
        exit_price = prev_row['open']
        prev_position = self.position

        self.close_position(prev_time, exit_price, switch=True)

        return prev_position
