import pandas as pd

from app.backtesting.indicators import calculate_rsi
from app.utils.backtesting_utils.backtesting_utils import format_trades

# Define parameters
RSI_PERIOD = 14
LOWER = 30
UPPER = 70


class RSIStrategy:
    def __init__(self, rsi_period=RSI_PERIOD, lower=LOWER, upper=UPPER):
        self.rsi_period = rsi_period
        self.lower = lower
        self.upper = upper
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
        self.switch_dates = None
        self.rollover = False

    def run(self, df, switch_dates, rollover):
        """Run the RSI strategy"""
        df = self.add_rsi_indicator(df)
        df = self.generate_signals(df)
        trades = self.extract_trades(df, switch_dates, rollover)
        summary = self.compute_summary(trades)
        print(summary)
        return trades, summary

    def add_rsi_indicator(self, df):
        df = df.copy()
        df['rsi'] = calculate_rsi(df["close"], period=self.rsi_period)
        return df

    def generate_signals(self, df):
        """
        Signals:
            1: Long entry
           -1: Short entry
            0: No action
        """
        df = df.copy()
        df['signal'] = 0
        prev_rsi = df['rsi'].shift(1)

        # Buy signal: RSI crosses below a lower threshold
        df.loc[(prev_rsi > self.lower) & (df['rsi'] <= self.lower), 'signal'] = 1

        # Sell signal: RSI crosses above an upper threshold
        df.loc[(prev_rsi < self.upper) & (df['rsi'] >= self.upper), 'signal'] = -1

        return df

    def extract_trades(self, df, switch_dates, rollover):
        """Extract trades based on signals"""
        self.trades = []
        self.position = None
        self.entry_time = None
        self.entry_price = None
        self.next_switch_idx = 0
        self.next_switch = switch_dates[self.next_switch_idx] if switch_dates else None
        self.must_reopen = None
        self.prev_row = None
        self.skip_signal_this_bar = False
        self.queued_signal = None
        self.switch_dates = switch_dates
        self.rollover = rollover

        for idx, row in df.iterrows():
            current_time = pd.to_datetime(idx)
            signal = row['signal']
            price_open = row['open']

            # Handle contract switches
            self._handle_contract_switch(current_time)

            # Open a new position on the next iteration (only if rollover enabled)
            self._handle_reopen(idx, price_open)

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

        return format_trades(self.trades)

    @staticmethod
    def compute_summary(trades):
        """Compute summary of trades"""
        total_pnl = sum(trade['pnl'] for trade in trades)
        summary = {
            "num_trades": len(trades),
            "total_pnl": total_pnl
        }
        return summary

    # --- Private methods ---

    def _handle_contract_switch(self, current_time):
        """Handle contract switches"""
        while self.next_switch and current_time >= self.next_switch:
            # On rollover: close at the price of *last bar before switch* (prev_row)
            if self.position is not None and self.entry_time is not None and self.prev_row is not None:
                self._close_position_at_switch(current_time)
            self.next_switch_idx += 1
            self.next_switch = self.switch_dates[self.next_switch_idx] if self.next_switch_idx < len(self.switch_dates) else None

    def _close_position_at_switch(self, current_time):
        """Close position at contract switch"""
        exit_price = self.prev_row['open']

        pnl = (exit_price - self.entry_price) * self.position
        self.trades.append({
            "entry_time": self.entry_time,
            "entry_price": self.entry_price,
            "exit_time": current_time,
            "exit_price": exit_price,
            "side": "long" if self.position == 1 else "short",
            "pnl": pnl,
            "switch": True,
        })
        if self.rollover:
            self.must_reopen = self.position  # Mark to reopen with the same direction
            self.skip_signal_this_bar = True  # Skip signal for this bar, only one trade per bar allowed
        else:
            self.must_reopen = None  # Do NOT reopen if ROLLOVER is False
        self._reset_position()

    def _reset_position(self):
        """Reset position variables"""
        self.entry_time = None
        self.entry_price = None
        self.position = None

    def _handle_reopen(self, idx, price_open):
        """Handle reopening position after rollover"""
        if self.must_reopen is not None and self.position is None:
            if self.rollover:
                self.position = self.must_reopen
                self.entry_time = idx
                self.entry_price = price_open
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
                    self._close_current_position(idx, price_open)
                # Open a new position at this (current) bar
                self._open_new_position(flip, idx, price_open)

            # Reset after using
            self.queued_signal = None

    def _close_current_position(self, idx, price_open):
        """Close current position"""
        exit_price = price_open
        side = self.position
        pnl = (exit_price - self.entry_price) * side
        self.trades.append({
            "entry_time": self.entry_time,
            "entry_price": self.entry_price,
            "exit_time": idx,
            "exit_price": exit_price,
            "side": "long" if side == 1 else "short",
            "pnl": pnl,
        })

    def _open_new_position(self, direction, idx, price_open):
        """Open a new position"""
        self.position = direction
        self.entry_time = idx
        self.entry_price = price_open
