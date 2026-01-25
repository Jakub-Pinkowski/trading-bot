import pandas as pd

from app.utils.backtesting_utils.indicators_utils import hash_series

# Strategy Execution Constants
# INDICATOR_WARMUP_PERIOD: Number of initial candles to skip before generating signals
# Rationale:
#   - Technical indicators (MA, EMA, RSI, etc.) need historical data to stabilize
#   - Example: A 100-period moving average needs 100 bars before it's valid
#   - Prevents generating signals based on incomplete/unstable indicator values
#   - 100 is chosen as a safe default that covers most common indicator periods:
#     * RSI (typically 14 periods)
#     * MACD (26 slow period)
#     * Bollinger Bands (typically 20 periods)
#     * Longer EMAs (up to 100 periods)
# Usage: Strategy execution starts from index INDICATOR_WARMUP_PERIOD in the DataFrame
INDICATOR_WARMUP_PERIOD = 100


class BaseStrategy:
    def __init__(self, rollover=False, trailing=None, slippage=0, slippage_type='percentage', symbol=None):
        """
        Initialize the base strategy.

        Args:
            rollover: Whether to handle contract rollovers
            trailing: Trailing stop percentage (if used)
            slippage: Slippage value (interpretation depends on slippage_type)
            slippage_type: Either 'percentage' or 'ticks'
                - 'percentage': slippage is a percentage (e.g., 0.05 = 0.05%)
                - 'ticks': slippage is number of ticks (e.g., 2 = 2 ticks)
            symbol: The futures symbol (e.g., 'ZC', 'GC') - required for tick-based slippage
        """
        self.switch_dates = None
        self.rollover = rollover
        self.trailing = trailing
        self.slippage = slippage
        self.slippage_type = slippage_type
        self.symbol = symbol

        # Get tick size for tick-based slippage
        if slippage_type == 'ticks' and symbol:
            from app.backtesting.tick_sizes import get_tick_size, get_decimal_places
            self.tick_size = get_tick_size(symbol)
            self.decimal_places = get_decimal_places(self.tick_size)
        else:
            self.tick_size = None
            self.decimal_places = 2

        # Initialize attributes that are reset in _reset()
        self.position = None
        self.entry_time = None
        self.entry_price = None
        self.trailing_stop = None
        self.next_switch_idx = 0
        self.next_switch = None
        self.must_reopen = None
        self.prev_row = None
        self.prev_time = None
        self.skip_signal_this_bar = False
        self.queued_signal = None
        self.trades = []

        self._reset()

    # ==================== Public API ====================

    def run(self, df, switch_dates):
        """
        Run the strategy.
        """
        df = self.add_indicators(df)
        df = self.generate_signals(df)
        trades = self._extract_trades(df, switch_dates)

        return trades

    def add_indicators(self, df):
        """Add indicators to the dataframe. To be implemented by subclasses."""
        raise NotImplementedError('Subclasses must implement add_indicators method')

    def generate_signals(self, df):
        """
        Generate signals based on indicators. To be implemented by subclasses.
        Signals:
            1: Long entry
           -1: Short entry
            0: No action
        """
        raise NotImplementedError('Subclasses must implement generate_signals method')

    # ==================== Helper Methods for Subclasses ====================

    def _precompute_hashes(self, df):
        """
        Pre-compute hashes for commonly used price series.

        This method calculates hashes once for price series that are typically
        passed to multiple indicator functions. This eliminates redundant hash
        computations and significantly improves performance when using multiple
        indicators.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dictionary with pre-computed hashes for common price series:
            {
                'close': hash of df['close'],
                'high': hash of df['high'],
                'low': hash of df['low'],
                'open': hash of df['open'],
                'volume': hash of df['volume']
            }

        Example:
            def add_indicators(self, df):
                # Pre-compute all hashes once
                hashes = self._precompute_hashes(df)

                # Pass pre-computed hashes to indicators (no redundant hashing!)
                df['rsi'] = calculate_rsi(df['close'], period=14,
                                         prices_hash=hashes['close'])
                df['rsi_long'] = calculate_rsi(df['close'], period=21,
                                              prices_hash=hashes['close'])
                df['ema'] = calculate_ema(df['close'], period=9,
                                         prices_hash=hashes['close'])
                return df
        """
        hashes = {}

        # Pre-compute hashes for OHLCV columns that exist in the DataFrame
        for col in ['close', 'high', 'low', 'open', 'volume']:
            if col in df.columns:
                hashes[col] = hash_series(df[col])

        return hashes

    # --- Signal Detection Helpers ---

    def _detect_crossover(self, series1, series2, direction='above'):
        """
        Detect when series1 crosses series2.

        Args:
            series1 (pd.Series): First series (e.g., fast EMA, MACD line)
            series2 (pd.Series): Second series (e.g., slow EMA, signal line)
            direction (str): 'above' for bullish crossover, 'below' for bearish crossover

        Returns:
            pd.Series: Boolean series indicating crossover points
        """
        prev_series1 = series1.shift(1)
        prev_series2 = series2.shift(1)

        if direction == 'above':
            # Series1 crosses above series2 (bullish)
            return (prev_series1 <= prev_series2) & (series1 > series2)
        else:  # direction == 'below'
            # Series1 crosses below series2 (bearish)
            return (prev_series1 >= prev_series2) & (series1 < series2)

    def _detect_threshold_cross(self, series, threshold, direction='below'):
        """
        Detect when a series crosses a threshold value.

        Args:
            series (pd.Series): The series to check (e.g., RSI, price)
            threshold (float): The threshold value to cross
            direction (str): 'below' for crossing downward, 'above' for crossing upward

        Returns:
            pd.Series: Boolean series indicating threshold cross-points
        """
        prev_series = series.shift(1)

        if direction == 'below':
            # Series crosses below a threshold (bearish)
            return (prev_series > threshold) & (series <= threshold)
        else:  # direction == 'above'
            # Series crosses above a threshold (bullish)
            return (prev_series < threshold) & (series >= threshold)

    # ==================== Private Methods ====================

    # --- State Management ---

    def _reset(self):
        """Reset all state variables"""
        self.position = None
        self.entry_time = None
        self.entry_price = None
        self.trailing_stop = None
        self.next_switch_idx = 0
        self.next_switch = None
        self.must_reopen = None
        self.prev_row = None
        self.prev_time = None
        self.skip_signal_this_bar = False
        self.queued_signal = None
        self.trades = []

    def _reset_position(self):
        """Reset position variables"""
        self.entry_time = None
        self.entry_price = None
        self.position = None
        self.trailing_stop = None

    # --- Trade Extraction & Execution ---

    def _extract_trades(self, df, switch_dates):
        """Extract trades based on signals.

        SIGNAL EXECUTION TIMING:
        ----------------------
        Signals are generated based on indicator values calculated from each bar's close.
        The signal is queued and executed at the OPEN of the NEXT bar, creating a 1-bar delay.

        Example Timeline (15-minute bars):
            10:15:00 - Bar N closes at 1074.25
                     → RSI = 29.8 (crosses below 30)
                     → Signal = 1 (BUY) - Generated and queued

            10:30:00 - Bar N+1 opens at 1074.25
                     → Queued signal executed
                     → Position opened at 1074.25 + slippage
        This is standard backtesting practice because:
        1. You can only see the signal AFTER the bar closes
        2. Execution happens at the next available opportunity (next bar's open)
        3. Slippage is applied to account for:
           - Execution delay between signal and fill
           - Bid-ask spread
           - Market movement during order placement

        This approach is:
        - Industry standard (used by TradingView, Backtrader, etc.)
        - Realistic and conservative
        - Not actual look-ahead bias (proper time sequencing)

        Args:
            df: DataFrame with OHLCV data and signals
            switch_dates: List of contract rollover dates

        Returns:
            List of trade dictionaries with entry/exit details
        """
        self.switch_dates = switch_dates
        self._reset()
        self.next_switch = switch_dates[self.next_switch_idx] if switch_dates else None

        # Counter to skip the first candles for indicator warm-up
        candle_count = 0

        for idx, row in df.iterrows():
            current_time = pd.to_datetime(idx)
            signal = row['signal']
            price_open = row['open']
            price_high = row['high']
            price_low = row['low']

            # Increment candle counter
            candle_count += 1

            # Skip signal processing for the first candles to allow indicators to warm up
            if candle_count <= INDICATOR_WARMUP_PERIOD:
                self.prev_time = current_time
                self.prev_row = row
                continue

            # Handle trailing stop logic if enabled
            self._handle_trailing_stop(idx, price_high, price_low)

            # Handle contract switches. Close an old position and potentially open a new one
            self._handle_contract_switch(current_time, idx, price_open)

            # Skip signal for this bar if we are in a rollover position, and we are about to switch
            if self.skip_signal_this_bar:
                self.skip_signal_this_bar = False  # skip *this* bar only
                self.prev_time = current_time
                self.prev_row = row
                continue

            # Execute queued signal from the previous bar
            self._execute_queued_signal(idx, price_open)

            # Set/overwrite queued_signal for next bar execution
            if signal != 0:
                self.queued_signal = signal

            self.prev_time = current_time
            self.prev_row = row

        return self.trades

    # --- Slippage Calculations ---

    def _apply_slippage_to_entry_price(self, direction, price):
        """Apply slippage to entry price based on a position direction"""
        if direction == 1:  # Long position
            # For long positions, pay more on entry (higher price)
            adjusted_price = price * (1 + self.slippage / 100)
        else:  # Short position
            # For short positions, receive less on entry (lower price)
            adjusted_price = price * (1 - self.slippage / 100)

        return round(adjusted_price, 2)

    def _apply_slippage_to_exit_price(self, direction, price):
        """Apply slippage to exit price based on a position direction"""
        if direction == 1:  # Long position
            # For long positions, receive less on exit (lower price)
            adjusted_price = price * (1 - self.slippage / 100)
        else:  # Short position
            # For short positions, pay more on exit (higher price)
            adjusted_price = price * (1 + self.slippage / 100)

        return round(adjusted_price, 2)

    # --- Position Management ---

    def _open_new_position(self, direction, idx, price_open):
        self.position = direction
        self.entry_time = idx

        # Apply slippage to entry price
        self.entry_price = self._apply_slippage_to_entry_price(direction, price_open)

        # Set initial trailing stop if trailing is enabled
        if self.trailing is not None:
            if direction == 1:  # Long position
                self.trailing_stop = round(price_open * (1 - self.trailing / 100), 2)
            else:  # Short position
                self.trailing_stop = round(price_open * (1 + self.trailing / 100), 2)

    def _close_position(self, exit_time, exit_price, switch=False):
        # Apply slippage to exit price
        adjusted_exit_price = self._apply_slippage_to_exit_price(self.position, exit_price)

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

    def _close_position_at_switch(self):
        # Exit at OPEN of last bar before switch (conservative assumption)
        # This assumes we exit early rather than waiting until close
        exit_price = self.prev_row['open']
        prev_position = self.position

        self._close_position(self.prev_time, exit_price, switch=True)

        if self.rollover:
            self.must_reopen = prev_position  # Use previous position value
            self.skip_signal_this_bar = True
        else:
            self.must_reopen = None

    # --- Event Handlers ---

    def _handle_trailing_stop(self, idx, price_high, price_low):
        """
        Manage trailing stop trigger and update logic.

        CRITICAL: Order of operations prevents look-ahead bias.

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

        Args:
            idx: Current bar index in the dataframe
            price_high: Highest price during the bar
            price_low: Lowest price during the bar

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
        # STEP 1: Check if trailing stop was triggered (conservative assumption)
        # For longs: Use low price (if we hit stop, assume it happened before the high)
        # For shorts: Use high price (if we hit stop, assume it happened before the low)
        if self.position is not None and self.trailing_stop is not None:
            if self.position == 1 and price_low <= self.trailing_stop:
                # Long stop triggered - close at stop level (not at low price)
                self._close_position(idx, self.trailing_stop, switch=False)
                return  # Exit early - no stop update after position closed (prevents look-ahead bias)
            elif self.position == -1 and price_high >= self.trailing_stop:
                # Short stop triggered - close at stop level (not at high price)
                self._close_position(idx, self.trailing_stop, switch=False)
                return  # Exit early - no stop update after position closed (prevents look-ahead bias)

        # STEP 2: Update trailing stop based on favorable price movement
        # Only reached if stop was NOT triggered in STEP 1
        if self.position is not None and self.trailing_stop is not None:
            # Calculate new stop level based on position direction
            new_stop = self._calculate_new_trailing_stop(self.position, price_high, price_low)

            if new_stop is not None:
                # Only tighten stop (never loosen for trailing stops)
                if self.position == 1 and new_stop > self.trailing_stop:
                    # Long: Only move stop UP
                    self.trailing_stop = new_stop
                elif self.position == -1 and new_stop < self.trailing_stop:
                    # Short: Only move stop DOWN
                    self.trailing_stop = new_stop

    def _calculate_new_trailing_stop(self, position, price_high, price_low):
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
            return round(price_high * (1 - self.trailing / 100), 2)
        elif position == -1:  # Short position
            # Use bar low to calculate stop (most favorable price for short)
            return round(price_low * (1 + self.trailing / 100), 2)
        return None

    def _handle_contract_switch(self, current_time, idx, price_open):
        while self.next_switch and current_time >= self.next_switch:
            # On rollover date close at the price of *last bar before switch* (prev_row)
            if self.position is not None and self.entry_time is not None and self.prev_row is not None:
                self._close_position_at_switch()
            self.next_switch_idx += 1
            self.next_switch = self.switch_dates[self.next_switch_idx] if self.next_switch_idx < len(
                self.switch_dates
            ) else None

        # Reopen on the following contract if rollover is enabled
        if self.must_reopen is not None and self.position is None:
            if self.rollover:
                direction = self.must_reopen
                self.position = direction
                self.entry_time = idx

                # Apply slippage to entry price
                self.entry_price = self._apply_slippage_to_entry_price(direction, price_open)
            self.must_reopen = None

    def _execute_queued_signal(self, idx, price_open):
        """Execute queued signal from the previous bar"""

        if self.queued_signal is not None:
            flip = None
            if self.queued_signal == 1 and self.position != 1:
                flip = 1
            elif self.queued_signal == -1 and self.position != -1:
                flip = -1

            if flip is not None:
                # Close if currently in position
                if self.position is not None and self.entry_time is not None:
                    self._close_position(idx, price_open, switch=False)
                # Open a new position at this (current) bar
                self._open_new_position(flip, idx, price_open)

            # Reset after using
            self.queued_signal = None
