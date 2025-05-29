import pandas as pd

from app.backtesting.indicators import calculate_rsi

# Define parameters
RSI_PERIOD = 14
LOWER = 30
UPPER = 70


class RSIStrategy:
    def __init__(self, rsi_period=RSI_PERIOD, lower=LOWER, upper=UPPER, rollover=False):
        self.rsi_period = rsi_period
        self.lower = lower
        self.upper = upper
        self.switch_dates = None
        self.rollover = rollover

        # Initialize attributes that are reset in _reset()
        self.position = None
        self.entry_time = None
        self.entry_price = None
        self.next_switch_idx = 0
        self.next_switch = None
        self.must_reopen = None
        self.prev_row = None
        self.skip_signal_this_bar = False
        self.queued_signal = None
        self.trades = []

        self._reset()

    def run(self, df, switch_dates):
        """Run the RSI strategy"""
        df = df.copy()
        df = self.add_rsi_indicator(df)
        df = self.generate_signals(df)
        trades = self.extract_trades(df, switch_dates)
        return trades

    def add_rsi_indicator(self, df):
        df['rsi'] = calculate_rsi(df["close"], period=self.rsi_period)
        return df

    def generate_signals(self, df):
        """
        Signals:
            1: Long entry
           -1: Short entry
            0: No action
        """
        df['signal'] = 0
        prev_rsi = df['rsi'].shift(1)

        # Buy signal: RSI crosses below a lower threshold
        df.loc[(prev_rsi > self.lower) & (df['rsi'] <= self.lower), 'signal'] = 1

        # Sell signal: RSI crosses above an upper threshold
        df.loc[(prev_rsi < self.upper) & (df['rsi'] >= self.upper), 'signal'] = -1

        return df

    def extract_trades(self, df, switch_dates):
        """Extract trades based on signals"""
        self.switch_dates = switch_dates
        self._reset()
        self.next_switch = switch_dates[self.next_switch_idx] if switch_dates else None

        for idx, row in df.iterrows():
            current_time = pd.to_datetime(idx)
            signal = row['signal']
            price_open = row['open']

            # Handle contract switches. Close an old position and potentially open a new one
            self._handle_contract_switch(current_time, idx, price_open)

            if self.skip_signal_this_bar:
                self.skip_signal_this_bar = False  # skip *this* bar only
                self.prev_row = row
                continue

            # Execute queued signal from the previous bar
            self._execute_queued_signal(idx, price_open)

            # Set/overwrite queued_signal for next bar execution
            if signal != 0:
                self.queued_signal = signal

            self.prev_row = row

        return self.trades

    # --- Private methods ---

    def _handle_contract_switch(self, current_time, idx, price_open):
        while self.next_switch and current_time >= self.next_switch:
            # On rollover date close at the price of *last bar before switch* (prev_row)
            if self.position is not None and self.entry_time is not None and self.prev_row is not None:
                self._close_position_at_switch(current_time)
            self.next_switch_idx += 1
            self.next_switch = self.switch_dates[self.next_switch_idx] if self.next_switch_idx < len(self.switch_dates) else None

        if self.must_reopen is not None and self.position is None:
            if self.rollover:
                self.position = self.must_reopen
                self.entry_time = idx
                self.entry_price = price_open
            self.must_reopen = None

    def _close_position_at_switch(self, current_time):
        exit_price = self.prev_row['open']
        prev_position = self.position

        self._close_position(current_time, exit_price, switch=True)

        if self.rollover:
            self.must_reopen = prev_position  # Use previous position value
            self.skip_signal_this_bar = True
        else:
            self.must_reopen = None

    def _reset(self):
        """Reset all state variables"""
        self.position = None
        self.entry_time = None
        self.entry_price = None
        self.next_switch_idx = 0
        self.next_switch = None
        self.must_reopen = None
        self.prev_row = None
        self.skip_signal_this_bar = False
        self.queued_signal = None
        self.trades = []

    def _reset_position(self):
        """Reset position variables"""
        self.entry_time = None
        self.entry_price = None
        self.position = None

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

    def _close_position(self, exit_time, exit_price, switch=False):
        trade = {
            "entry_time": self.entry_time,
            "entry_price": self.entry_price,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "side": "long" if self.position == 1 else "short",
        }
        if switch:
            trade["switch"] = True
        self.trades.append(trade)
        self._reset_position()

    def _open_new_position(self, direction, idx, price_open):
        self.position = direction
        self.entry_time = idx
        self.entry_price = price_open
