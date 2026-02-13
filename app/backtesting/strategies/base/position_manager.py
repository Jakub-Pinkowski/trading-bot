"""
Position Manager for Backtesting Strategies

This module manages position state and lifecycle during backtesting, including:
- Opening and closing positions
- Position state tracking
- Tick-based slippage calculations for entry and exit prices
"""

from futures_config import get_tick_size


class PositionManager:
    """
    Manage position state and lifecycle during backtesting.

    Tracks open positions, handles slippage application, manages trade records,
    and integrates with trailing stop functionality. Maintains position direction
    (long/short), entry details, and completed trade history. Applies realistic
    slippage to entry and exit prices based on position direction.
    """

    # ==================== Initialization ====================

    def __init__(self, slippage_ticks, symbol, trailing):
        """
        Initialize the position manager.

        Args:
            slippage_ticks: Number of ticks of slippage (e.g., 2 = 2 ticks)
            symbol: Futures symbol for contract specifications (e.g., 'ZC', 'GC')
            trailing: Trailing stop percentage if enabled (None = disabled)
        """
        self.slippage_ticks = slippage_ticks
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

    def _reset_position(self):
        """Reset position variables (internal use only)"""
        self.entry_time = None
        self.entry_price = None
        self.position = None
        self.trailing_stop = None

    def has_open_position(self):
        """Check if there is an open position"""
        return self.position is not None

    def get_trades(self):
        """Get list of completed trades"""
        return self.trades

    # ==================== Slippage Calculations ====================

    def apply_slippage_to_entry_price(self, direction, price):
        """
        Apply tick-based slippage to entry price.

        Long positions pay more on entry (higher price), short positions receive
        less on entry (lower price). Simulates realistic market impact using
        actual tick sizes.

        Args:
            direction: Position direction (1 = long, -1 = short)
            price: Base entry price before slippage

        Returns:
            Adjusted entry price with slippage applied
        """
        tick_size = get_tick_size(self.symbol)
        slippage_amount = self.slippage_ticks * tick_size

        if direction == 1:  # Long position
            adjusted_price = price + slippage_amount
        else:  # Short position
            adjusted_price = price - slippage_amount

        return round(adjusted_price, 2)

    def apply_slippage_to_exit_price(self, direction, price):
        """
        Apply tick-based slippage to exit price.

        Long positions receive less on exit (lower price), short positions pay
        more on exit (higher price). Simulates realistic market impact using
        actual tick sizes.

        Args:
            direction: Position direction being closed (1 = long, -1 = short)
            price: Base exit price before slippage

        Returns:
            Adjusted exit price with slippage applied
        """
        tick_size = get_tick_size(self.symbol)
        slippage_amount = self.slippage_ticks * tick_size

        if direction == 1:  # Long position
            adjusted_price = price - slippage_amount
        else:  # Short position
            adjusted_price = price + slippage_amount

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

    def close_position(self, exit_time, exit_price, switch):
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
        self._reset_position()

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
