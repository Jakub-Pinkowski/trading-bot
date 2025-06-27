import pandas as pd


class BaseStrategy:
    def __init__(self, rollover=False, trailing=None, slippage=0):
        self.switch_dates = None
        self.rollover = rollover
        self.trailing = trailing
        self.slippage = slippage

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

    def run(self, df, switch_dates):
        """Run the strategy"""
        df = df.copy()
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

    # --- Private methods ---

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

    def _extract_trades(self, df, switch_dates):
        """Extract trades based on signals"""
        self.switch_dates = switch_dates
        self._reset()
        self.next_switch = switch_dates[self.next_switch_idx] if switch_dates else None

        # Counter to skip the first 100 candles for indicator warm-up
        candle_count = 0

        for idx, row in df.iterrows():
            current_time = pd.to_datetime(idx)
            signal = row['signal']
            price_open = row['open']
            price_high = row['high']
            price_low = row['low']

            # Increment candle counter
            candle_count += 1

            # Skip signal processing for the first 100 candles to allow indicators to warm up
            if candle_count <= 100:
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

    def _close_position_at_switch(self, current_time):
        exit_price = self.prev_row['open']
        prev_position = self.position

        self._close_position(self.prev_time, exit_price, switch=True)

        if self.rollover:
            self.must_reopen = prev_position  # Use previous position value
            self.skip_signal_this_bar = True
        else:
            self.must_reopen = None

    def _handle_trailing_stop(self, idx, price_high, price_low):
        """Manage trailing stop trigger and update logic."""

        # First, check if a trailing stop has been triggered
        if self.position is not None and self.trailing_stop is not None:
            if self.position == 1 and price_low <= self.trailing_stop:
                self._close_position(idx, self.trailing_stop, switch=False)
                return  # Exit early if the position is closed
            elif self.position == -1 and price_high >= self.trailing_stop:
                self._close_position(idx, self.trailing_stop, switch=False)
                return  # Exit early if the position is closed

        # Only update trailing stop if position wasn't closed
        if self.position is not None and self.trailing_stop is not None:
            if self.position == 1:  # Long position
                new_stop = round(price_high * (1 - self.trailing / 100), 2)
                if new_stop > self.trailing_stop:
                    self.trailing_stop = new_stop
            elif self.position == -1:  # Short position
                new_stop = round(price_low * (1 + self.trailing / 100), 2)
                if new_stop < self.trailing_stop:
                    self.trailing_stop = new_stop

    def _handle_contract_switch(self, current_time, idx, price_open):
        while self.next_switch and current_time >= self.next_switch:
            # On rollover date close at the price of *last bar before switch* (prev_row)
            if self.position is not None and self.entry_time is not None and self.prev_row is not None:
                self._close_position_at_switch(current_time)
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
