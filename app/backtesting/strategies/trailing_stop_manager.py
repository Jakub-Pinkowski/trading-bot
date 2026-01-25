"""
Trailing Stop Manager for Backtesting Strategies

This module manages trailing stop functionality with proper look-ahead bias prevention.

CRITICAL: Order of Operations
============================
The trailing stop logic must follow a specific sequence to prevent look-ahead bias:

Step 1: Check if stop was triggered using worst-case intra-bar price
Step 2: If triggered, close position and return early (no stop update)
Step 3: If NOT triggered, update stop level using best-case intra-bar price

This ensures we never benefit from BOTH stopping out AND favorable price
movement within the same bar, which would create unrealistic backtest results.

Rationale:
- In real trading, once a stop is triggered, the position is closed immediately
- We cannot also update the stop level after the position is already closed
- By checking trigger FIRST, we prevent "free" stop updates after exit

Price Selection Logic:
- LONG positions:
  * Trigger check uses price_low (worst case - if we hit stop, we hit it)
  * Stop update uses price_high (best case - tighten stop based on upward movement)
- SHORT positions:
  * Trigger check uses price_high (worst case - if we hit stop, we hit it)
  * Stop update uses price_low (best case - tighten stop based on downward movement)

Edge Cases Handled:
- Gap through stop: Position closed at stop level (not at gap price)
- Bar touches stop exactly: Position closed (no stop update occurs)
- Volatile bar: Stop only updated if NOT triggered during the bar

Example Scenarios:

Scenario 1: Stop triggered at bar low, bar high would calculate better stop
- Entry: $100, Current stop: $95, Trailing: 5%
- Bar: Low=$95 (at stop), High=$110 (would calculate new stop of $104.50)
- Result: Close at $95, NO stop update (prevents look-ahead bias)

Scenario 2: Stop NOT triggered, normal update
- Entry: $100, Current stop: $95, Trailing: 5%
- Bar: Low=$98 (above stop), High=$105 (favorable movement)
- Result: Stop updated from $95 to $99.75 ($105 * 0.95)

Scenario 3: Gap down through stop
- Current stop: $95
- Bar: Open=$90, Low=$88, High=$92
- Result: Close at $95 (not at $88 - assume stop filled at stop level)
"""


# ==================== Helper Functions ====================

def should_trigger_stop(position, trailing_stop, price_high, price_low):
    """
    Check if trailing stop should be triggered.

    Args:
        position: 1 for long, -1 for short
        trailing_stop: Current trailing stop level
        price_high: Highest price during the bar
        price_low: Lowest price during the bar

    Returns:
        bool: True if stop should trigger, False otherwise
    """
    if position == 1 and price_low <= trailing_stop:
        # Long stop triggered
        return True
    elif position == -1 and price_high >= trailing_stop:
        # Short stop triggered
        return True
    return False


def update_trailing_stop(position_manager, position, current_stop, new_stop):
    """
    Update trailing stop if it represents a tightening of the stop.

    Args:
        position_manager: PositionManager instance
        position: 1 for long, -1 for short
        current_stop: Current trailing stop level
        new_stop: New calculated stop level
    """
    # Only tighten stop (never loosen for trailing stops)
    if position == 1 and new_stop > current_stop:
        # Long: Only move stop UP
        position_manager.trailing_stop = new_stop
    elif position == -1 and new_stop < current_stop:
        # Short: Only move stop DOWN
        position_manager.trailing_stop = new_stop


class TrailingStopManager:
    """
    Manages trailing stop functionality with proper look-ahead bias prevention.
    
    See comprehensive documentation in module docstring about order of operations.
    """

    # ==================== Initialization ====================

    def __init__(self, trailing_percentage, on_stop_triggered=None):
        """
        Initialize the trailing stop manager.

        Args:
            trailing_percentage: Trailing stop percentage (e.g., 5 for 5%)
            on_stop_triggered: Optional callback function called when trailing stop is triggered.
                              Signature: on_stop_triggered(position, stop_price)
        """
        self.trailing_percentage = trailing_percentage
        self.on_stop_triggered = on_stop_triggered

    # ==================== Public Methods ====================

    def handle_trailing_stop(self, position_manager, idx, price_high, price_low):
        """
        Manage trailing stop trigger and update logic.

        CRITICAL: This method implements the two-step process that prevents look-ahead bias.

        Args:
            position_manager: PositionManager instance containing position state
            idx: Current bar index in the dataframe
            price_high: Highest price during the bar
            price_low: Lowest price during the bar

        Returns:
            bool: True if position was closed due to stop trigger, False otherwise
        """
        # STEP 1: Check if trailing stop was triggered (conservative assumption)
        # For longs: Use low price (if we hit stop, assume it happened before the high)
        # For shorts: Use high price (if we hit stop, assume it happened before the low)
        if position_manager.has_open_position() and position_manager.trailing_stop is not None:
            position = position_manager.position
            trailing_stop = position_manager.trailing_stop

            if should_trigger_stop(position, trailing_stop, price_high, price_low):
                # Stop triggered - close at stop level (not at low/high price)
                position_manager.close_position(idx, trailing_stop, switch=False)
                
                # Call callback if provided
                if self.on_stop_triggered is not None:
                    self.on_stop_triggered(position, trailing_stop)
                
                return True  # Exit early - no stop update after position closed (prevents look-ahead bias)

        # STEP 2: Update trailing stop based on favorable price movement
        # Only reached if stop was NOT triggered in STEP 1
        if position_manager.has_open_position() and position_manager.trailing_stop is not None:
            position = position_manager.position
            trailing_stop = position_manager.trailing_stop

            # Calculate new stop level based on position direction
            new_stop = self.calculate_new_trailing_stop(position, price_high, price_low)

            if new_stop is not None:
                update_trailing_stop(position_manager, position, trailing_stop, new_stop)

        return False

    # ==================== Helper Methods ====================

    def calculate_new_trailing_stop(self, position, price_high, price_low):
        """
        Calculate new trailing stop level based on position direction and favorable price movement.

        This is extracted as a helper method for clarity and testability.

        Args:
            position: 1 for long, -1 for short
            price_high: Highest price during the bar (used for long positions)
            price_low: Lowest price during the bar (used for short positions)

        Returns:
            New stop level (float) or None if position type is invalid

        Logic:
        - LONG positions: Calculate stop below the HIGH (benefit from upward movement)
          Formula: high * (1 - trailing_percentage)
          Example: $110 high with 5% trailing = $110 * 0.95 = $104.50 stop

        - SHORT positions: Calculate stop above the LOW (benefit from downward movement)
          Formula: low * (1 + trailing_percentage)
          Example: $90 low with 5% trailing = $90 * 1.05 = $94.50 stop
        """
        if position == 1:  # Long position
            # Use bar high to calculate stop (most favorable price for long)
            return round(price_high * (1 - self.trailing_percentage / 100), 2)
        elif position == -1:  # Short position
            # Use bar low to calculate stop (most favorable price for short)
            return round(price_low * (1 + self.trailing_percentage / 100), 2)
        return None

    def initialize_trailing_stop(self, position_manager, entry_price, direction):
        """
        Initialize trailing stop when opening a position.

        Args:
            position_manager: PositionManager instance
            entry_price: Entry price of the position
            direction: 1 for long, -1 for short

        Returns:
            Initial trailing stop level
        """
        if direction == 1:  # Long position
            trailing_stop = round(entry_price * (1 - self.trailing_percentage / 100), 2)
        else:  # Short position
            trailing_stop = round(entry_price * (1 + self.trailing_percentage / 100), 2)

        position_manager.trailing_stop = trailing_stop
        return trailing_stop
